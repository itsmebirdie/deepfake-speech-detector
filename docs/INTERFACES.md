# Interfaces (the contract between the three members)

Everything below is the agreed shape of data crossing a member boundary. If you must change one, open a PR
that edits this file first and tag the other two members. Conventions: waveforms are 16 kHz mono float32;
`B` = batch, `T` = samples (`audio.num_samples`); label **0 = bona fide, 1 = synthetic**; `vocoder_id` **0 = bona fide, 1..6 = vocoders**.

## 1. Dataset -> everyone (owner A)
Item: `{waveform [T], label [], vocoder_id [], utt_id str}`. Batch: `{waveform [B,T], label [B], vocoder_id [B], utt_id list[str]}`.

## 2. Signal layer (owner B)
`MultiResSTFT(wav [B,T]) -> list[R] of complex [B, F_r, N_r]`, `F_r = n_fft//2+1`. Always fp32.

## 3. Phase branch -> fusion (owner B -> C)  **hand-off**
`PhaseBranch(wav [B,T]) -> {'H_d': [B,T_d,C_d], 'cons_per_sample': [B], 'delta_phi': list (optional)}`
`cons_per_sample[i] = mean over resolutions, bins, frames of w * (1 - cos(delta_phi))`.
`T_d` is a config constant (`model.phase_branch.encoder.t_out`) so C can size attention without running B's code.
Until B7 lands, C develops against a `DummyPhaseBranch` that returns random tensors of exactly these shapes.

## 4. Baseline wrapper -> fusion (owner C, internal but documented)
`BaselineRawNet2.forward_features(wav [B,T]) -> H_b [B,T_b,C_b]`. Discovered in PR-08 (checked against the baseline code at commit 74b8527):
- **C_b = 1024**: the GRU hidden size. `H_b` is the GRU output at every time step, before the baseline's pooling.
- **T_b = 43** for the native input of 96,000 samples (4 s at 24 kHz). In general `T_b = floor(...floor(floor((T - 1024) / 3) / 3).../ 3)` with seven divisions by 3
  (Sinc conv with a 1025-tap kernel and no padding, then one max-pool of 3 after the front-end and one in each of the six residual blocks).
  `BaselineRawNet2.feature_length(T)` computes it; it is 28 for T = 64,000 (4 s at 16 kHz) and 1 for the minimum input of 3,211 samples.
  **PR-19: set `model.phase_branch.encoder.t_out` from this** (the dummy branch uses 64 until it is set).
- **Pooling:** the baseline's heads read only the LAST time step, `H[:, -1, :]`. PhaseConNet pools the same way by default (`model.heads.pooling: last`; `mean` is available), so with the phase branch off it reproduces the baseline.
- **Sample rate:** the released weights and the Sinc front-end assume **24 kHz / 96,000 samples**. `configs/default.yaml` still says 16 kHz / 64,000 (DATASET_NOTES section 7, decision pending); `build_model` warns while they differ. Real weights must be run at 24 kHz.
- **Conventions:** `logits_bin` of the wrapper is the baseline's own log-softmax output with the two columns swapped to the repo order `[bona fide, synthetic]`. Log-probabilities are logits up to a per-row constant; cross-entropy and the score `logits_bin[:,1] - logits_bin[:,0]` do not depend on that constant. The baseline's binary target is `y == 0` (1 = bona fide), the opposite of this repo; `BaselineRawNet2.repo_label_to_baseline` converts.

## 5. Model -> trainer / evaluator (owner C -> A)
`model(wav [B,T]) -> {'logits_bin': [B,2], 'logits_voc': [B,n_voc]?, 'cons_per_sample': [B]?}`
The evaluator uses ONLY `logits_bin`: `score = logits_bin[:,1] - logits_bin[:,0]` (higher = more synthetic). This is the single
place the score convention is defined; A's metrics and baseline-parity checks depend on it.

## 6. Results (owner A)
`experiments/results/<run_name>/{config.yaml, metrics.json, scores.csv}`; `metrics.json = {eer, eer_threshold, auc, accuracy, per_vocoder: {...}}`.
Written only through `evaluation/results.py::write_results` (read back with `read_results`); `scripts/evaluate.py` produces it for any model.
- `config.yaml`: the full merged config of the run.
- `metrics.json`: `eer`, `eer_threshold`, `auc`, `accuracy` (at score 0), `accuracy_at_eer`, `per_vocoder: {<name>: {eer, auc, caught, n_fake}}`, `bona_fide_correct`, plus `run_name`, `split`, `protocol`, `held_out`, `checkpoint`, `git_hash`, `git_dirty`, `created`.
- `scores.csv`: `utt_id,label,vocoder_id,score`, one row per clip, higher score = more synthetic.
- A folder that already has `metrics.json` is never overwritten unless asked (`--overwrite`).

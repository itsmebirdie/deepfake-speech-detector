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
`BaselineRawNet2.forward_features(wav [B,T]) -> H_b [B,T_b,C_b]`. `T_b`, `C_b` are discovered in step C1 and written here:
- T_b = TBD (C1)   - C_b = TBD (C1)

## 5. Model -> trainer / evaluator (owner C -> A)
`model(wav [B,T]) -> {'logits_bin': [B,2], 'logits_voc': [B,n_voc]?, 'cons_per_sample': [B]?}`
The evaluator uses ONLY `logits_bin`: `score = logits_bin[:,1] - logits_bin[:,0]` (higher = more synthetic). This is the single
place the score convention is defined; A's metrics and baseline-parity checks depend on it.

## 6. Results (owner A)
`experiments/results/<run_name>/{config.yaml, metrics.json, scores.csv}`; `metrics.json = {eer, eer_threshold, auc, accuracy, per_vocoder: {...}}`.

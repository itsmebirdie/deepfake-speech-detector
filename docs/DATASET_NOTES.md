# Dataset & baseline notes (filled in by Member A, steps A2 / A4)

Status: PR-02 facts are filled in below. Items marked _later_ belong to PR-06, PR-09 and PR-13.

## Summary

| Item | Value |
|---|---|
| Baseline repo commit (read for these notes) | `74b85277fe95583b23a2766bf45d15463c772ee5` (2025-06-12). The submodule from PR-01 (`make baseline`) is not added yet; pin it to this commit. |
| Archive | `LibriSeVoc.zip`, 42 GB, unzipped to `data/raw/LibriSeVoc/` |
| Folders | `gt` (bona fide) + 6 vocoders: `diffwave`, `melgan`, `parallel_wave_gan`, `wavegrad`, `wavenet`, `wavernn` |
| Files | 13,201 per folder, 92,407 in total |
| Audio format | WAV, **24 kHz**, mono, in every folder |
| Clip length | 3.99 s to 33.3 s, median about 8.2 s, mean about 9.5 s |
| Total audio | about 243 hours (about 35 h per folder) |
| Official splits | 7,920 / 2,640 / 2,641 utterances (train / dev / test) |
| Baseline input length | **4 s = 96,000 samples at 24 kHz** (not 64,600) |
| Baseline binary label | **1 = bona fide, 0 = synthetic** (opposite of this repo) |
| Baseline vocoder head | 7 classes (`fc2_multi_gru`, `out_features=7`), 0 = bona fide |

## 1. Downloads

- **Dataset:** the Drive link in `scripts/download_data.sh`. `gdown` hit Google Drive's download quota ("Too many users have viewed or downloaded this file recently"), so the file was downloaded through a browser. Zip integrity was checked: every one of the 92,407 extracted files matches the size recorded in the zip.
- **Weights:** the Drive link in `scripts/download_weights.sh`. The original file name is `librifake_pretrained_lambda0.5_epoch_25.pth` (71.3 MB), saved as `weights/baseline_pretrained.pth`. "lambda0.5" matches the baseline's loss weight `LAMDA = 0.5`, and "epoch_25" is the training epoch.
- `gdown --fuzzy` fails with current gdown versions (the option was removed), so both scripts now pass the bare Drive file ID.

## 2. Archive layout

```
data/raw/LibriSeVoc/
  gt/                  13,201 bona fide clips   e.g. 27_123349_000006_000000.wav
  diffwave/            13,201 synthetic clips   e.g. 27_123349_000006_000000_gen.wav
  melgan/              13,201
  parallel_wave_gan/   13,201
  wavegrad/            13,201
  wavenet/             13,201
  wavernn/             13,201
```

- Bona fide names follow LibriTTS: `<speaker>_<chapter>_<paragraph>_<sentence>.wav`.
- Each synthetic clip is a re-synthesis of the bona fide clip with the same name plus `_gen`. Every bona fide utterance has exactly one fake per vocoder.

## 3. Vocoders and `vocoder_id`

Folder names, as written in `configs/data/librisevoc.yaml`, in the order used for `vocoder_id` 1..6:

| vocoder_id | Folder | Name in the paper |
|---|---|---|
| 0 | `gt` | bona fide |
| 1 | `wavegrad` | WaveGrad |
| 2 | `diffwave` | DiffWave |
| 3 | `parallel_wave_gan` | Parallel WaveGAN |
| 4 | `wavernn` | WaveRNN |
| 5 | `wavenet` | WaveNet |
| 6 | `melgan` | MelGAN |

Why this order: the released vocoder head was trained with labels assigned in `os.listdir()` order on the authors' machine (`main.py`). The only record of that order is the print statement in `eval.py`: `gt, wavegrad, diffwave, parallel wave gan, wavernn, wavenet, melgan`. Using the same order lets the pretrained vocoder head be reused without remapping. **To verify in PR-06 / PR-08:** run the released weights on a few clips per folder and confirm that the predicted vocoder class matches this table.

## 4. Official splits

| Split | Utterances | Clips (×7: 1 real + 6 fake) |
|---|---|---|
| train | 7,920 | 55,440 |
| dev | 2,640 | 18,480 |
| test | 2,641 | 18,487 |
| total | 13,201 | 92,407 |

Checked on the extracted data: every name in each split file exists in `gt/`, all 6 fakes exist for every name, and no name appears in more than one split.

**Caveat (affects PR-06):** the baseline's `main.py` does **not** read `train.txt` / `dev.txt` / `test.txt`. It lists each folder with `os.listdir()` and slices `[0:7920]`, `[7920:10560]`, `[10560:13201]`. `os.listdir()` order depends on the operating system and filesystem, and each folder is listed separately. So (a) the released weights may have been trained on utterances that our `test.txt` treats as test, and (b) a real clip and its fakes can land in different splits. This could make the released model look better on our test split than it really is. PR-06 should compare our reproduced numbers with the paper and keep this in mind; LOVO already retrains the baseline (PR-13 / PR-26).

## 5. Audio statistics (all 92,407 files, read from WAV headers)

| Folder | Sample rate | Channels | Min | Median | Mean | Max | Total |
|---|---|---|---|---|---|---|---|
| gt | 24,000 | 1 | 5.00 s | 8.27 s | 9.52 s | 33.25 s | 34.9 h |
| diffwave | 24,000 | 1 | 5.00 s | 8.28 s | 9.53 s | 33.26 s | 34.9 h |
| melgan | 24,000 | 1 | 3.99 s | 8.17 s | 9.43 s | 33.07 s | 34.6 h |
| parallel_wave_gan | 24,000 | 1 | 3.99 s | 8.17 s | 9.43 s | 33.07 s | 34.6 h |
| wavegrad | 24,000 | 1 | 5.01 s | 8.28 s | 9.53 s | 33.26 s | 35.0 h |
| wavenet | 24,000 | 1 | 4.73 s | 8.22 s | 9.48 s | 33.22 s | 34.8 h |
| wavernn | 24,000 | 1 | 4.61 s | 8.05 s | 9.31 s | 33.04 s | 34.1 h |

- Under 0.05% of clips are shorter than 4 s (only a few MelGAN / Parallel WaveGAN clips at 3.99 s). Almost every clip is therefore cropped, not padded, at a 4 s input.
- Fakes are slightly shorter or longer than their source clip (vocoder framing), so a real clip and its fake are not sample-aligned. Keep this in mind for any paired real/fake comparison (PR-10).

## 6. Baseline facts (read from the code at the commit above)

- **Sample rate:** 24 kHz. `main.py`: `SAMPLE_RATE = 24000`; `eval.py` resamples any input to 24 kHz.
- **Input length:** 96,000 samples (4 s). `main.py` `__getitem__` sets `self.cut = SAMPLE_RATE * 4` and calls `pad(X, self.cut)`. `eval.py` uses `max_len = 96000` and splits long files into 4 s segments.
  - The value 64,600 appears only as an unused default (`pad(x, max_len=64600)`) and as `nb_samp` in `model_config_RawNet.yaml`, which `model.py` does not use (it reads the length from the input tensor).
- **Short and long clips:** clips longer than 4 s are cropped to the **first** 4 s (no random crop); shorter clips are repeated (tiled) until they reach 4 s.
- **Labels:** the multi-class target is 0 = gt, 1..6 = vocoders (in `os.listdir()` order). The binary target is `y == 0`, so **1 = bona fide, 0 = synthetic**. This is the opposite of this repo's convention (0 = bona fide, 1 = synthetic); the baseline wrapper (PR-08) must convert.
- **Heads:** `fc2_binary_gru` has 2 outputs and `fc2_multi_gru` has 7 outputs.
- **Loss:** `LAMDA * CE(binary) + (1 - LAMDA) * CE(multi)` with `LAMDA = 0.5`.
- **EER:** the baseline repo has **no EER implementation**. `main.py` reports only accuracy. So there is no baseline function to match. EER now lives in `src/phasecon/evaluation/metrics.py` (PR-07, ROC-based); `tests/test_metrics.py` checks it gives exactly the same EER as the ASVspoof challenge formula and a brute-force threshold sweep.
- **Logits to score:** both heads end in `LogSoftmax`. Binary output index 0 = synthetic, index 1 = bona fide. Our score (higher = more synthetic, `INTERFACES.md` section 5) is `score = logp[:, 0] - logp[:, 1]`, i.e. in repo order `logits_bin = [logp[:, 1], logp[:, 0]]`. `scripts/eval_baseline.py` uses exactly this.

## 7. Decisions needed from the team

1. **Sample rate.** The data and the pretrained baseline are 24 kHz, but `docs/INTERFACES.md` and `configs/default.yaml` say 16 kHz. Option (a): use 24 kHz and `num_samples: 96000`, which matches the released weights and needs no resampling. Option (b): resample to 16 kHz, which changes the input the pretrained SincConv front-end was trained on. Recommended: (a). **Decided: (a)** in the INTERFACES.md PR that moved `configs/default.yaml` to `sample_rate: 24000`, `num_samples: 96000` (PR-02 had kept 16 kHz until that PR).
2. **`vocoder_id` order:** confirm the order in section 3.
3. **Split caveat:** agree how PR-06 reports the baseline given section 4.

## 8. Released baseline reproduced (PR-06)

**Result:** on the official test split the released weights give **EER 11.86%**, not the paper's **0.13%**. The gap comes almost entirely from two vocoders: the model calls **WaveRNN** fakes real 98.7% of the time and catches only 28.7% of **WaveNet** fakes. On the other four vocoders it matches the paper's level (per-vocoder EER 0.11% to 0.51%).

Run: `python scripts/eval_baseline.py --baseline-dir ../baseline-repo --split test` (mode `first4s`: first 4 s at 24 kHz, as in the authors' `main.py`), CPU, 18,487 clips, 21 min. Weights: `weights/baseline_pretrained.pth` (released `librifake_pretrained_lambda0.5_epoch_25.pth`). Baseline repo commit as in section "Summary".

| | Value |
|---|---|
| EER (pooled, test) | **11.86%** (paper, Table 3 "Ours": 0.13%) |
| AUC | 0.9438 |
| Accuracy (argmax of the binary head) | 75.37% |
| Bona fide clips called bona fide | about 99.8% (2,637 of 2,641, derived from the accuracy above) |
| Vocoder-ID accuracy (7 classes) | 74.06% |

| Vocoder | EER vs bona fide | Fakes caught | Vocoder named right |
|---|---|---|---|
| wavegrad | 0.11% | 100.00% | 100.00% |
| diffwave | 0.19% | 99.70% | 99.58% |
| parallel_wave_gan | 0.17% | 99.43% | 97.88% |
| wavernn | **36.54%** | **1.29%** | 0.00% |
| wavenet | **8.14%** | **28.66%** | 23.93% |
| melgan | 0.51% | 98.64% | 97.12% |

**What we ruled out** (quick runs on 20 utterances, 140 clips each):

- *Input method:* the authors' `eval.py` way (whole clip in 4 s pieces, probabilities averaged; `--mode eval_py`) gives the same failure: EER 14.2%, WaveRNN caught 0%.
- *Test-set leakage:* the dev split gives the same failure (EER 10.4%, WaveRNN caught 0%). Leakage would make scores better, not worse.
- *Our label or vocoder order:* the four working vocoders are predicted as exactly the `vocoder_id` in section 3, and bona fide clips are recognised, so neither mapping is flipped. WaveRNN clips are predicted as class 0 (bona fide); WaveNet mostly as class 0.

**Most likely explanation (not proven):** the released weights were not trained on the WaveRNN / WaveNet audio in this archive. The file is named `librifake_...`, suggesting an earlier version of the dataset (the earlier preprint calls it "LibriVoc"), whose WaveRNN / WaveNet clips may have been generated differently. Confirming this needs the authors.

**Consequences:**

1. The released weights are **not** a 0.13%-EER baseline on this archive. Any comparison must use a baseline we evaluate ourselves on the same data, never the paper's number.
2. PR-08 (baseline wrapper) can still use these weights for logit parity, but PR-24 (baseline through our trainer) is the fair baseline row: fine-tuning on our train split should learn WaveRNN / WaveNet.
3. WaveRNN and WaveNet (the two autoregressive vocoders) are where the baseline is weakest, which is worth tracking for PhaseCon-Net.
4. Open action: ask the authors (GitHub issue) whether the released weights match LibriSeVoc.

Sources: [Sun et al., CVPRW 2023, arXiv 2304.13085](https://arxiv.org/pdf/2304.13085) (Table 3) · [earlier preprint, arXiv 2302.09198](https://arxiv.org/pdf/2302.09198)

## 9. Balanced subset (PR-09)

**Definition:** training and tuning may use a fixed, 50/50 balanced subset; final numbers are always reported on the **full official test split**.

| Split | per_vocoder | Utterances | Clips | Real | Fake (each vocoder) | Manifest |
|---|---|---|---|---|---|---|
| train | 1000 | 6,000 of 7,920 | 12,000 | 6,000 | 6,000 (1,000) | `data/subsets/train.txt` |
| dev | 200 | 1,200 of 2,640 | 2,400 | 1,200 | 1,200 (200) | `data/subsets/dev.txt` |
| test | none | all 2,641 | 18,487 | 2,641 | 15,846 (2,641) | none: full official split |

**How it is built** (`build_balanced_subset`, seed = `seed` in `configs/default.yaml` = 1234): shuffle the split's utterances with the seed, take the first 6 x per_vocoder, and give each one its real clip plus exactly one fake, cycling through the vocoders in `data.vocoders` order. So every vocoder has the same count, real = fake, and every fake sits next to the real recording it was made from (same speaker and words). A larger per_vocoder keeps every utterance of a smaller one.

**Use it:** set `data.subset.enabled: true` (or `--override data.subset.enabled=true`). Rebuild with `python scripts/make_subset.py`; the same config always gives byte-identical manifests, and `tests/test_dataset.py::test_committed_manifests_match_config` checks the committed files match.

**Why these counts:** 12,000 training clips is about 22% of the full 55,440, so an epoch is about 4.6x faster. They are a starting point; if the slowest GPU in the team needs less, lower `per_vocoder_train` and re-run the script (the maximum is 1,320 for train and 440 for dev).

## 10. Leave-one-vocoder-out protocol (PR-13)

**Question it answers:** can a detector catch a vocoder it has **never heard** during training? This matters because real attackers will use new vocoders.

**Protocol (fixed; changing it after PR-26 invalidates the baseline LOVO runs):** one fold per vocoder, six folds in total. For the fold that holds out vocoder V:

| Split | Utterances | What it contains | Clips |
|---|---|---|---|
| train | official train (7,920) | bona fide + the 5 vocoders other than V | 47,520 (7,920 real, 39,600 fake) |
| dev | official dev (2,640) | bona fide + the 5 vocoders other than V | 15,840 (2,640 real, 13,200 fake) |
| test | official test (2,641) | bona fide + **only V's fakes** | 5,282 (2,641 real, 2,641 fake) |

- Train, dev and test never share an utterance (the official splits guarantee it; `build_lovo_splits` asserts it).
- `vocoder_id` keeps its global meaning (1 to 6 in `data.vocoders` order), so V's id never appears in train/dev.
- No randomness: each fold is an exact filter of the official split files. Files: `data/splits/lovo/<V>/{train,dev,test}.txt`, built by `python scripts/make_lovo_splits.py` and checked by `test_committed_lovo_folds_match_protocol`.
- **Reporting:** for each fold, the EER on its test split (V vs real); then the mean over the six folds. Each model is trained from scratch once per fold with the same settings and seed. The released baseline weights saw every vocoder, so the baseline is **retrained** per fold (PR-26).
- **Use it:** `--override data.lovo.held_out=<V>`. With `data.subset.enabled: true` too, train/dev are balanced with the PR-09 rule over the fold's 5 vocoders (1,000 fakes per vocoder + 5,000 real for train; 200 + 1,000 for dev), deterministic by seed. Test is never shrunk.

## 11. Baseline through our pipeline (PR-24)

**Result:** the pipeline reproduces the released baseline (parity within tolerance), and a baseline trained through our trainer on the balanced subset reaches **EER 0.30%** on the full test split, against 11.86% for the released weights. The released weights' failure on WaveRNN and WaveNet (section 8) is a property of those weights, not of the architecture or the data.

Run: `.\scripts\run_pr24.ps1` (Windows) or `bash scripts/run_pr24.sh`; model `configs/model/baseline_only.yaml`, 24 kHz / 96,000 samples, CUDA with AMP, seed 1234, one run per row. Training uses the balanced subset of section 9 (12,000 train clips, 2,400 dev clips). Evaluation is `scripts/evaluate.py` on the **full official test split** (18,487 clips). Settings that mirror the authors' recipe: Adam lr 1e-4 on all weights, `beta=1.0` (ours is `L_bin + beta*L_voc`; their `0.5*CE_bin + 0.5*CE_voc` is the same up to a factor 2), `weight_decay=2e-4` (cancels that factor under Adam's L2 decay), `mu=0`. Cosine schedule with 2 warm-up epochs, gradient clipping 5.0, early stopping on dev EER with patience 8 (the fine-tuning run stopped after epoch 18).

**Check 1, parity** (released weights through `scripts/evaluate.py`, compared with the PR-06 numbers of section 8; `scripts/compare_results.py --parity`):

| | EER | AUC | Accuracy | wavegrad | diffwave | parallel_wave_gan | wavernn | wavenet | melgan |
|---|---|---|---|---|---|---|---|---|---|
| PR-06 (direct) | 11.86% | 0.9438 | 75.37% | 0.11% | 0.19% | 0.17% | 36.54% | 8.14% | 0.51% |
| Our pipeline | 11.85% | 0.9438 | 75.37% | 0.08% | 0.19% | 0.19% | 36.58% | 8.14% | 0.53% |
| Allowed difference | 0.05 pp | 0.001 | 0.1 pp | 0.1 pp | 0.1 pp | 0.1 pp | 0.1 pp | 0.1 pp | 0.1 pp |

All nine metrics pass; the largest difference is 0.037 pp (WaveRNN). The tolerance allows only floating-point noise (the PR-06 run was on the CPU, this one on the GPU). The wrapper itself is bit-identical to the authors' code with the released weights (`tests/test_baseline_wrapper.py`, both weight variants).

**Check 2, trained runs** (`scripts/compare_results.py --no-regress`: pooled EER may be at most 0.5 pp worse than the released weights; better is expected):

| Run | EER | AUC | Accuracy | wavegrad | diffwave | parallel_wave_gan | wavernn | wavenet | melgan |
|---|---|---|---|---|---|---|---|---|---|
| Released weights (PR-06) | 11.86% | 0.9438 | 75.37% | 0.11% | 0.19% | 0.17% | 36.54% | 8.14% | 0.51% |
| Frozen encoder, heads trained | 9.39% | 0.9659 | 85.97% | 0.19% | 0.15% | 0.23% | 27.11% | 5.45% | 0.19% |
| **Fine-tuned (whole model)** | **0.30%** | 0.99997 | 99.68% | 0.00% | 0.08% | 0.23% | 0.38% | 0.42% | 0.11% |

Both pass. Fine-tuned fakes caught: wavegrad 100.00%, diffwave 99.96%, parallel_wave_gan 99.77%, wavernn 99.20%, wavenet 99.09%, melgan 99.89%.

**What it means**

1. **The fine-tuned row is the baseline for every later comparison** (G5). It is trained and evaluated exactly like PhaseCon-Net will be.
2. **WaveRNN 36.54% to 0.38% and WaveNet 8.14% to 0.42%** confirm the explanation in section 8: the released weights had not seen these clips. Training only the heads helps WaveRNN less (27.11%), so the released *encoder* lacks the information; the heads cannot recover it.
3. **In-distribution has almost no headroom** (0.30% EER, paper 0.13%). Differences of a few hundredths of a point between models are within what one seed can produce. The in-distribution run cannot show what phase adds; the leave-one-vocoder-out folds (section 10) can.
4. **Do not credit a WaveRNN gain to phase** unless it beats 0.38% here (and its LOVO counterpart), as `ARCHITECTURE.md` section 3 requires. The phase study found WaveRNN separable by magnitude alone, so a trained baseline learning it is expected.
5. **Single seed.** Each row is one run. Report more seeds before claiming differences of this size.

Re-running: the scripts refuse to overwrite finished results and checkpoints. To repeat the whole check, set a new prefix first (PowerShell `$env:PREFIX = "baseline_pipeline_2"`, bash `PREFIX=baseline_pipeline_2`).

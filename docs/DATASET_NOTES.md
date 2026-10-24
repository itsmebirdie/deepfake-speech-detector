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

1. **Sample rate.** The data and the pretrained baseline are 24 kHz, but `docs/INTERFACES.md` and `configs/default.yaml` say 16 kHz. Option (a): use 24 kHz and `num_samples: 96000`, which matches the released weights and needs no resampling. Option (b): resample to 16 kHz, which changes the input the pretrained SincConv front-end was trained on. Recommended: (a). Per the interface rule, this needs its own `INTERFACES.md` PR before any config change, so PR-02 keeps `sample_rate: 16000` and only corrects `num_samples` from 64,600 to 64,000 (4 s at 16 kHz). Moving to 24 kHz / 96,000 happens in that PR.
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

## Later items

- Subset definition (per-vocoder counts, seed, manifest files): _later_ (PR-09)
- LOVO protocol (exact definition): _later_ (PR-13)

# Issues and PR Workflow

This is the single source of truth for **what to build, in what order, and who does it**. It has no dates: order is defined by PR numbers and gates.
It complements `docs/WORK_PLAN.md` (the why) and `docs/INTERFACES.md` (the shapes).

## The five rules
1. **One global order.** PRs merge in numeric order: PR-01, PR-02, ... PR-36. Branches are developed in parallel, but they land one at a time, lowest number first. If PR-N is ready while a lower-numbered PR is still open, wait for it (or ask in the channel).
2. **Start rule.** You may open your branch for PR-N once everything in its **Blocked by** list is merged. Before that, read, plan and write against stubs; do not open the PR.
3. **Gates are full stops for merging.** At a gate (G0 to G7) nothing merges until all three members comment `ack` on the gate issue. You may keep developing on your branch while you wait.
4. **Tests define done.** A PR is ready when every test listed in it is un-skipped and passing, and the **Done when** line is true.
5. **Interfaces change first.** If a shape or key must change, edit `docs/INTERFACES.md` in its own PR, tagged to the other two, before changing any code.

## How to work a PR (every time)
1. Pull `main`. Create the branch named in the PR section: `a/prNN-slug`, `b/prNN-slug`, or `c/prNN-slug`.
2. Work through the **Tasks** checklist in order.
3. Un-skip the tests listed. Run `ruff check src tests scripts` and `pytest -q` locally.
4. Open the PR from the template. Title: `[PR-NN] <title>`. First line of the description: `Closes #<issue>`.
5. Ask the **Reviewer(s)** named in the PR section. They review against **Done when**, not taste.
6. Squash-merge once approved and CI is green. Post a one-line "merged" note in the team channel naming what it unblocks.

## If you are blocked
- Work against the documented stub (for example `DummyPhaseBranch`, random `H_d`) and say so in the PR.
- Never edit another member's files to unblock yourself. Comment on their PR or issue.
- If a blocker turns out to be wrong (a shape, a convention), raise it on the issue; fix it through rule 5.

## Master order

| PR | Sprint | Owner | Title | Blocked by | Reviewer(s) |
|---|---|---|---|---|---|
| [PR-01](#pr-01) | S0 | A | Bootstrap repository, CI and baseline submodule | none | B, C |
| **G0** | | all | **GATE: Foundations ready** | PR-01 merged | all |
| [PR-02](#pr-02) | S1 | A | Acquire LibriSeVoc and released weights; document them | PR-01 | B, C |
| [PR-03](#pr-03) | S1 | B | Multi-resolution STFT | PR-01 | C |
| [PR-04](#pr-04) | S1 | A | Dataset, transforms, splits and dataloader | PR-02 | C |
| [PR-05](#pr-05) | S1 | B | Fast Griffin-Lim reference and instantaneous-frequency utilities | PR-03 | A |
| [PR-06](#pr-06) | S1 | A | Reproduce the released baseline | PR-02 | C |
| [PR-07](#pr-07) | S1 | A | Metrics and model-agnostic evaluator | PR-04, PR-06 | C |
| [PR-08](#pr-08) | S1 | C | Baseline wrapper with feature access | PR-02, PR-06 | A |
| **G1** | | all | **GATE: Data and baseline verified** | PR-08 merged | all |
| [PR-09](#pr-09) | S2 | A | Balanced subset manifests | PR-04 | C |
| [PR-10](#pr-10) | S2 | B | Phase feasibility study (GO / NO-GO) | PR-04, PR-05, PR-09 | A, C |
| **G2** | | all | **GATE: Phase go / no-go decision** | PR-10 merged | all |
| [PR-11](#pr-11) | S2 | C | Classification heads and gradient reversal | PR-08 | B |
| [PR-12](#pr-12) | S2 | C | Fusion blocks (gated cross-attention and concat) | PR-08 | B |
| [PR-13](#pr-13) | S2 | A | Leave-one-vocoder-out splits | PR-04 | C |
| [PR-14](#pr-14) | S2 | B | ULCP: unrolled learnable consistency projection | PR-10 | C |
| [PR-15](#pr-15) | S2 | C | Classification losses and combined loss (Eq. 6) | PR-01 | B |
| [PR-16](#pr-16) | S2 | B | Phase-discrepancy tensor | PR-05, PR-10 | C |
| [PR-17](#pr-17) | S2 | A | Evaluation CLI and results registry | PR-04, PR-07 | C |
| [PR-18](#pr-18) | S2 | C | PhaseConNet assembly with DummyPhaseBranch and build_model | PR-11, PR-12 | B |
| [PR-19](#pr-19) | S2 | B | Discrepancy encoder | PR-16, PR-08 | C |
| [PR-20](#pr-20) | S3 | C | Trainer, optimizer groups, checkpointing, early stopping | PR-04, PR-07, PR-15, PR-18 | A |
| [PR-21](#pr-21) | S3 | B | PhaseBranch packaging (B to C hand-off) | PR-14, PR-16, PR-19, PR-08 | C |
| **G3** | | all | **GATE: Interfaces delivered** | PR-21 merged | all |
| [PR-22](#pr-22) | S3 | B | Consistency loss | PR-21 | C |
| [PR-23](#pr-23) | S3 | C | Integrate the real PhaseBranch and run the overfit sanity check | PR-18, PR-20, PR-21, PR-22 | A, B |
| **G4** | | all | **GATE: End-to-end model trains** | PR-23 merged | all |
| [PR-24](#pr-24) | S4 | C | Pipeline validation: baseline through our trainer | PR-06, PR-09, PR-20 | A |
| [PR-25](#pr-25) | S4 | C | First PhaseCon-Net training run (in-distribution) | PR-09, PR-23, PR-24 | B |
| [PR-26](#pr-26) | S4 | A | Baseline LOVO retraining | PR-13, PR-20, PR-24 | C |
| **G5** | | all | **GATE: First results trustworthy** | PR-26 merged | all |
| [PR-27](#pr-27) | S5 | B | Wire phase-branch ablations | PR-23 | C |
| [PR-28](#pr-28) | S5 | C | Wire remaining ablations | PR-23 | B |
| [PR-29](#pr-29) | S5 | A | Ablation runner | PR-17, PR-27, PR-28 | C |
| [PR-30](#pr-30) | S5 | A | Experiment campaign: LOVO for PhaseCon-Net, ablations, seeds | PR-25, PR-26, PR-29 | B, C |
| **G6** | | all | **GATE: Experiments complete and frozen** | PR-30 merged | all |
| [PR-31](#pr-31) | S6 | A | Aggregation: tables and figures | PR-30 | C |
| [PR-32](#pr-32) | S6 | C | Analysis: what helps and where it fails | PR-30, PR-31 | A, B |
| [PR-33](#pr-33) | S6 | B | Performance and robustness | PR-30 | C |
| [PR-34](#pr-34) | S6 | A | Reproducibility pass | PR-31 | B |
| [PR-35](#pr-35) | S6 | C | Documentation and model card | PR-32 | A, B |
| [PR-36](#pr-36) | S6 | C | Invention disclosure draft | PR-32 | A, B |
| **G7** | | all | **GATE: Delivery** | PR-36 merged | all |

## Who works on what, per sprint (parallel lanes)

Each column is one person. Read a column top to bottom; PRs in the same row range can be developed at the same time but still merge in numeric order.

| Sprint | A: Liza | B: Nisha | C: Maanas |
|---|---|---|---|
| Sprint 0: Foundations | PR-01 | wait; read `INTERFACES.md`, `ARCHITECTURE.md` | wait; read the baseline repo (read-only) |
| Sprint 1: Data, baseline and signal primitives | PR-02, 04, 06, 07 | PR-03, 05 | read baseline code until PR-06 merges, then PR-08 |
| Sprint 2: Feasibility gate and building blocks | PR-09, 13, 17 | PR-10 (gate G2), then 14, 16, 19 | PR-11, 12, 15, 18 |
| Sprint 3: Assembly | no PR: review duty; pre-flight GPU, disk, and a full-size loader dry run | PR-21, 22 | PR-20, 23 |
| Sprint 4: First results | PR-26 | no PR: phase-residual visualisations from the first checkpoint, to help interpret PR-25 | PR-24, 25 |
| Sprint 5: Ablations and full experiments | PR-29, then 30 | PR-27 | PR-28 |
| Sprint 6: Analysis and wrap-up | PR-31, 34 | PR-33 | PR-32, 35, 36 |

## Gates

### G0: Foundations ready (after PR-01)

- [ ] Fresh clone passes `make install && pytest -q` on all three machines
- [ ] `main` is protected (PR + 1 review + green CI)
- [ ] Labels and milestones exist (ran `scripts/create_github_issues.sh`)
- [ ] Baseline submodule is pinned; commit hash is written in `docs/DATASET_NOTES.md`
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G1: Data and baseline verified (after PR-08)

- [ ] Loader iterates a full epoch on the official split with no errors
- [ ] Our EER equals the baseline repo's EER on identical scores
- [ ] Baseline test EER / ACC / AUC reproduced and recorded next to the paper's numbers
- [ ] `T_b`, `C_b` filled in `docs/INTERFACES.md` section 4
- [ ] `docs/DATASET_NOTES.md` has no `_TBD_` for vocoder names, split sizes, input length, label convention
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G2: Phase go / no-go decision (after PR-10)

- [ ] Written verdict exists in `docs/ARCHITECTURE.md` (known issue 2): GO, WEAK, or NO-GO
- [ ] Per-vocoder, per-resolution separability numbers are in the PR
- [ ] All three members have commented `ack` on the gate issue
- [ ] If NO-GO or WEAK: hold a short planning call and **edit this file** (PR-14, 16, 19, 21, 22 may change) before anyone continues
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G3: Interfaces delivered (after PR-21)

- [ ] `PhaseBranch` output matches `docs/INTERFACES.md` section 3 on real data (keys and shapes)
- [ ] Maanas confirms `H_d` works with the fusion block on real tensors
- [ ] Heads, fusion and wrapper (PR-08, 11, 12) are merged
- [ ] No interface change happened without a prior edit to `INTERFACES.md`
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G4: End-to-end model trains (after PR-23)

- [ ] Full PhaseCon-Net does forward + backward on a real batch
- [ ] `overfit_batch` drives loss near zero on the full model
- [ ] AMP run does not produce NaNs (STFT/ULCP stay fp32)
- [ ] `DummyPhaseBranch` is no longer used by any config or script
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G5: First results trustworthy (after PR-26)

- [ ] Baseline trained through *our* pipeline matches PR-06 numbers within the documented tolerance
- [ ] PhaseCon-Net in-distribution numbers recorded with seeds and configs
- [ ] Baseline LOVO (all six folds) recorded
- [ ] Hyperparameters are frozen (chosen on dev only)
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G6: Experiments complete and frozen (after PR-30)

- [ ] All six PhaseCon-Net LOVO folds finished
- [ ] All ablations (a1 to a7, K and R sweeps) finished
- [ ] Headline comparison has at least 3 seeds
- [ ] Git tag `results-v1` created; no further training runs after this tag
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

### G7: Delivery (after PR-36)

- [ ] One command regenerates every table and figure
- [ ] A teammate reproduced one result row from the README alone
- [ ] Analysis, model card and disclosure draft merged
- [ ] Disclosure draft claims only what ablations support
- [ ] Liza `ack`  - [ ] Nisha `ack`  - [ ] Maanas `ack`

## PR details

---

## Sprint 0: Foundations

### PR-01: Bootstrap repository, CI and baseline submodule

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr01-bootstrap-repository-ci-and`
- **Blocked by:** none
- **Reviewer(s):** B, C
- **Goal:** Everyone starts from the same working repo, with CI and rules in place.
- **Files:** whole skeleton, `.github/`, Makefile, `third_party/ (submodule)`, `docs/DATASET_NOTES.md`

**Tasks**

- [ ] Create the GitHub repo and push the skeleton
- [ ] Protect `main`: PR required, 1 approval, CI green
- [ ] Fill `.github/CODEOWNERS` with the real GitHub handles
- [ ] Run `bash scripts/create_github_issues.sh` to create labels, milestones and all issues from this file
- [ ] Run `make baseline`; commit the submodule pointer; write the pinned commit hash in `docs/DATASET_NOTES.md`
- [ ] Every member runs `make install && pytest -q` on their machine and pastes the result in the PR

**Tests to un-skip:** all tests are skipped, none fail

**Done when:** A fresh clone passes `make install && pytest -q` on all three machines.

**Watch out**

- Do not install heavy dependencies into the base image; CI uses CPU torch only.

> **GATE G0 follows this PR.** Stop and complete the gate checklist before merging PR-02.

---

## Sprint 1: Data, baseline and signal primitives

### PR-02: Acquire LibriSeVoc and released weights; document them

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr02-acquire-librisevoc-and-released`
- **Blocked by:** PR-01
- **Reviewer(s):** B, C
- **Goal:** Replace every guess about the dataset and baseline with a documented fact.
- **Files:** `docs/DATASET_NOTES.md`, `configs/data/librisevoc.yaml`, `data/splits/*.txt`, `scripts/download_*.sh`

**Tasks**

- [ ] Run `make data` and `make weights`; unzip into `data/raw/LibriSeVoc`
- [ ] Copy the official `train.txt`, `dev.txt`, `test.txt` from the baseline repo to `data/splits/` (commit them)
- [ ] Write the folder layout, the **six vocoder names**, split sizes, duration statistics in `DATASET_NOTES.md`
- [ ] Fill `data.vocoders` in `configs/data/librisevoc.yaml`
- [ ] Record the baseline's input length, label convention and sample rate; fix `audio.num_samples` in `configs/default.yaml` if the assumed 64600 is wrong
- [ ] Resolve every `VERIFY` comment in `configs/`

**Tests to un-skip:** none

**Done when:** `DATASET_NOTES.md` has no `_TBD_` for vocoder names, split sizes, input length, label convention.

**Watch out**

- Do not commit audio or weights (they are gitignored).
- If the Drive links fail, tell the team the same day; PR-04, 06, 08 are blocked on this.

### PR-03: Multi-resolution STFT

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr03-multi-resolution-stft`
- **Blocked by:** PR-01
- **Reviewer(s):** C
- **Goal:** A tested, differentiable STFT / iSTFT / consistency projection that everything in the phase stream builds on.
- **Files:** `src/phasecon/signal/stft.py`, `tests/test_stft.py`

**Tasks**

- [ ] Implement `MultiResSTFT.forward`, `inverse`, `project` (S after S^-1) for each configured `(n_fft, hop, win)`
- [ ] Force float32 inside (disable autocast); output complex64
- [ ] Un-skip and pass every test in `tests/test_stft.py`

**Tests to un-skip:** test_stft.py (all 5)

**Done when:** Round trip reconstructs the waveform, `project` is idempotent, gradients reach the input, fp32 under autocast.

**Watch out**

- Window normalisation errors show up as a failed round trip: check this before moving on.
- Needs only PyTorch; you do not need the dataset for this PR.

### PR-04: Dataset, transforms, splits and dataloader

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr04-dataset-transforms-splits-and`
- **Blocked by:** PR-02
- **Reviewer(s):** C
- **Goal:** A loader that returns exactly the batch in `INTERFACES.md` section 1.
- **Files:** `data/transforms.py`, `data/splits.py (read_split_file)`, `data/librisevoc.py`, `data/collate.py`, `tests/test_dataset.py`

**Tasks**

- [ ] Implement `pad_or_crop` exactly as the baseline treats short and long clips
- [ ] Implement `read_split_file` using the vocoder names from PR-02; enforce label 0 real / 1 fake, vocoder_id 0..6
- [ ] Implement `LibriSeVocDataset`, `collate_fn`, `build_dataloader`
- [ ] Un-skip: item shapes, pad/crop, no utt_id overlap between splits, collate shapes
- [ ] Iterate one full epoch over the real train split and paste throughput in the PR

**Tests to un-skip:** test_dataset.py: parsing, shapes, pad/crop, overlap, collate

**Done when:** Loader iterates a full epoch; tests pass.

**Watch out**

- Do **not** invent a different label convention for convenience. Convert at the baseline wrapper (PR-08) instead.

### PR-05: Fast Griffin-Lim reference and instantaneous-frequency utilities

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr05-fast-griffin-lim-reference`
- **Blocked by:** PR-03
- **Reviewer(s):** A
- **Goal:** Classical reference implementations used for the feasibility study, for unit-testing ULCP, and for ablation (ii).
- **Files:** `signal/griffin_lim.py`, `signal/instantaneous_frequency.py`, `tests/test_griffin_lim.py`

**Tasks**

- [ ] Implement `fast_griffin_lim` with momentum and optional initial phase
- [ ] Implement `wrap_to_pi` and `instantaneous_frequency` (principal-value difference, hop and bin corrected)
- [ ] Un-skip and pass `tests/test_griffin_lim.py`

**Tests to un-skip:** test_griffin_lim.py (all 4)

**Done when:** Magnitude preserved, spectral convergence improves with iterations, IF of a pure tone is correct.

**Watch out**

- Phase is only defined up to 2pi; every difference of phases goes through `wrap_to_pi` or cos/sin.

### PR-06: Reproduce the released baseline

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr06-reproduce-the-released-baseline`
- **Blocked by:** PR-02
- **Reviewer(s):** C
- **Goal:** Know the baseline's true numbers before comparing anything against it.
- **Files:** `scripts/eval_baseline.py`, `docs/DATASET_NOTES.md`

**Tasks**

- [ ] Run the authors' `eval.py` with the released weights on the official test split
- [ ] Wrap it in `scripts/eval_baseline.py` and save scores per utterance
- [ ] Record EER / accuracy / AUC in `DATASET_NOTES.md` beside the base paper's reported values
- [ ] Write down where the baseline computes EER and how its logits map to a score (PR-07 depends on this)
- [ ] If our numbers differ from the paper's, explain why in writing

**Tests to un-skip:** none (script-level)

**Done when:** Reproduced numbers are within a documented tolerance of the paper, or the gap is explained.

**Watch out**

- Do not change the baseline repo's code. If a fix is unavoidable, record it as a patch file.

### PR-07: Metrics and model-agnostic evaluator

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr07-metrics-and-model-agnostic`
- **Blocked by:** PR-04, PR-06
- **Reviewer(s):** C
- **Goal:** A single trusted place for EER, AUC and accuracy that every later comparison uses.
- **Files:** `evaluation/metrics.py`, `evaluation/evaluate.py`, `tests/test_metrics.py`

**Tasks**

- [ ] Implement `compute_eer`, `compute_auc`, `compute_accuracy`, `summarize`
- [ ] Score convention: `score = logits_bin[:,1] - logits_bin[:,0]`, higher means more synthetic
- [ ] Implement `run_inference` and `evaluate` (use only `logits_bin`; per-vocoder breakdown; optional scores CSV)
- [ ] Add the parity test: identical EER to the baseline repo's function on identical scores
- [ ] Un-skip all tests in `tests/test_metrics.py`

**Tests to un-skip:** test_metrics.py (all 5)

**Done when:** Parity test passes against the baseline's own EER.

**Watch out**

- If EER differs from the baseline's by even a little, stop and find out why: every table depends on it.

### PR-08: Baseline wrapper with feature access

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr08-baseline-wrapper-with-feature`
- **Blocked by:** PR-02, PR-06
- **Reviewer(s):** A
- **Goal:** Expose the released RawNet2's pre-pooling feature sequence while keeping its original outputs bit-identical.
- **Files:** `models/baseline/rawnet2_encoder.py`, `docs/INTERFACES.md (section 4)`, `tests/test_baseline_wrapper.py`

**Tasks**

- [ ] Add `third_party/...` to `sys.path` only inside this module; do not copy baseline code into `src/`
- [ ] Load the released weights; implement `forward_features` and `forward`
- [ ] Map label conventions between this repo (0 real / 1 fake) and the baseline
- [ ] Implement the `freeze_encoder` flag
- [ ] Write the discovered `T_b` and `C_b` into `INTERFACES.md` section 4
- [ ] Un-skip: logit parity with the baseline repo, feature shape, freeze flag, label mapping

**Tests to un-skip:** test_baseline_wrapper.py (all 4)

**Done when:** Logits equal the baseline repo's on identical clips, and `T_b`, `C_b` are documented.

**Watch out**

- This PR sets `T_d` for Nisha's encoder (PR-19). Announce the shapes in the team channel when merged.

> **GATE G1 follows this PR.** Stop and complete the gate checklist before merging PR-09.

---

## Sprint 2: Feasibility gate and building blocks

### PR-09: Balanced subset manifests

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr09-balanced-subset-manifests`
- **Blocked by:** PR-04
- **Reviewer(s):** C
- **Goal:** A fixed, reproducible subset of LibriSeVoc so everyone trains on identical clips.
- **Files:** `data/subset.py`, `scripts/make_subset.py`, `data/subsets/*.txt`, `docs/DATASET_NOTES.md`

**Tasks**

- [ ] Implement `build_balanced_subset` (deterministic by seed; equal real/fake totals; fixed per-vocoder counts)
- [ ] Keep each real clip together with its resynthesis where possible
- [ ] Implement `save_manifest` and `scripts/make_subset.py`; commit the small manifest files
- [ ] Define the subset (counts, seed, files) in `DATASET_NOTES.md`; the final report needs this
- [ ] Un-skip determinism and balance tests

**Tests to un-skip:** test_dataset.py::test_balanced_subset_is_deterministic

**Done when:** Same seed gives identical manifests; `data.subset.enabled=true` works in the loader.

**Watch out**

- Choose counts that fit the slowest GPU in the team.

### PR-10: Phase feasibility study (GO / NO-GO)

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr10-phase-feasibility-study-go`
- **Blocked by:** PR-04, PR-05, PR-09
- **Reviewer(s):** A, C
- **Goal:** Find out, before building the learnable parts, whether the phase residual separates real from synthetic.
- **Files:** `scripts/visualize_phase.py`, `notebooks/phase_feasibility.ipynb`, `docs/ARCHITECTURE.md`

**Tasks**

- [ ] Using **fixed** Griffin-Lim only, compute confidence-weighted `1 - cos(dphi)` per clip for real and each of the six vocoders
- [ ] Try zero initialisation and one other deterministic initialisation
- [ ] Report a per-vocoder, per-resolution separability number (e.g. AUC of the clip-level mean)
- [ ] Make residual maps with `scripts/visualize_phase.py` for a few real/fake pairs of the same utterance
- [ ] Write the verdict (GO / WEAK / NO-GO) in `ARCHITECTURE.md` under known issue 2

**Tests to un-skip:** none

**Done when:** A written verdict exists and all three members have acknowledged it on the G2 gate issue.

**Watch out**

- NO-GO fallbacks (pick one at the gate): (a) initialise from a learned phase predictor, (b) mel-conditioned phase estimate, (c) feed raw phase / IF through the same fusion (ablation i becomes the main model).
- Do not start PR-14, 16, 19, 21 or 22 until the gate is acknowledged.

> **GATE G2 follows this PR.** Stop and complete the gate checklist before merging PR-11.

### PR-11: Classification heads and gradient reversal

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr11-classification-heads-and-gradient`
- **Blocked by:** PR-08
- **Reviewer(s):** B
- **Goal:** Binary and vocoder-ID heads, initialised from the baseline where shapes allow.
- **Files:** `models/heads.py`

**Tasks**

- [ ] Implement `BinaryHead` (logits `[B,2]`) and `VocoderHead` (logits `[B, n_voc]`)
- [ ] Implement `GradReverse` and the `gradient_reversal` switch
- [ ] Initialise from the baseline's pretrained head weights where compatible
- [ ] Add tests: head output shapes; gradient reversal flips the sign of gradients

**Tests to un-skip:** new tests in tests/test_phasecon_net.py or a new tests/test_heads.py

**Done when:** Heads run on pooled `[B, C_b]` features and the gradient-reversal test passes.

### PR-12: Fusion blocks (gated cross-attention and concat)

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr12-fusion-blocks-gated-cross`
- **Blocked by:** PR-08
- **Reviewer(s):** B
- **Goal:** Eq. 5 fusion, developed against random `H_d` of the documented shape. Do not wait for Nisha.
- **Files:** `models/fusion.py`, `tests/test_fusion.py`

**Tasks**

- [ ] Implement `GatedCrossAttentionFusion`: query from the baseline stream, gate on a time-aligned `H_d`
- [ ] Handle `T_b != T_d`
- [ ] Implement `ConcatFusion` (ablation iii)
- [ ] Un-skip all tests in `tests/test_fusion.py`

**Tests to un-skip:** test_fusion.py (all 5)

**Done when:** Output shape equals `H_b`; a zero gate gives identity; gradients reach both streams.

### PR-13: Leave-one-vocoder-out splits

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr13-leave-one-vocoder-out`
- **Blocked by:** PR-04
- **Reviewer(s):** C
- **Goal:** Define and generate the LOVO protocol once, exactly, so the baseline and PhaseCon-Net are compared fairly.
- **Files:** `data/splits.py (build_lovo_splits, write_split)`, `scripts/make_lovo_splits.py`, `data/splits/lovo/*`, `docs/DATASET_NOTES.md`

**Tasks**

- [ ] Write the protocol in `DATASET_NOTES.md`: train/dev = real + all vocoders except the held-out one; test = real test clips + held-out fakes
- [ ] Implement `build_lovo_splits`, `write_split`, `make_lovo_splits.py`; generate six folds
- [ ] Un-skip `test_lovo_split_excludes_held_out`; add an assertion on utt_id overlap

**Tests to un-skip:** test_dataset.py::test_lovo_split_excludes_held_out

**Done when:** Six fold manifests committed; held-out vocoder never appears in train or dev.

**Watch out**

- Changing this protocol after PR-26 invalidates the baseline LOVO runs. Get it right now.

### PR-14: ULCP: unrolled learnable consistency projection

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr14-ulcp-unrolled-learnable-consistency`
- **Blocked by:** PR-10
- **Reviewer(s):** C
- **Goal:** Eqs. 1 to 3 as a module, with a non-learnable mode that equals fixed Griffin-Lim.
- **Files:** `models/ulcp.py`, `tests/test_ulcp.py`

**Tasks**

- [ ] Implement `ULCPStep`: projection, learnable momentum, small conv correction, magnitude re-imposition
- [ ] Implement `ULCP` with deterministic init and K steps from config
- [ ] Add `learnable=False` mode (fixed momentum, no correction) that matches PR-05's reference
- [ ] Un-skip all tests in `tests/test_ulcp.py`; profile memory for K = 3, 5, 8

**Tests to un-skip:** test_ulcp.py (all 6)

**Done when:** |Y_K| equals A for every K, K=0 returns the init, non-learnable mode matches fixed fast Griffin-Lim.

**Watch out**

- Start only after the G2 gate says GO or WEAK (or you were told a fallback design).

### PR-15: Classification losses and combined loss (Eq. 6)

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr15-classification-losses-and-combined`
- **Blocked by:** PR-01
- **Reviewer(s):** B
- **Goal:** L = L_bin + beta * L_voc + mu * L_cons, tolerant of missing terms.
- **Files:** `losses/classification.py`, `losses/combined.py`, `tests/test_losses.py`

**Tasks**

- [ ] Implement `binary_loss` (same class weighting as the baseline if it uses one) and `vocoder_loss`
- [ ] Implement `total_loss`: skip terms whose keys are absent; return float parts for logging
- [ ] Import `consistency_loss` from `losses/consistency.py`; test the cons term with a monkeypatch until PR-22 lands
- [ ] Un-skip all tests in `tests/test_losses.py`

**Tests to un-skip:** test_losses.py (all 3)

**Done when:** Composition test and missing-term test pass.

### PR-16: Phase-discrepancy tensor

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr16-phase-discrepancy-tensor`
- **Blocked by:** PR-05, PR-10
- **Reviewer(s):** C
- **Goal:** Eq. 4: confidence-weighted cos/sin of the phase residual plus an instantaneous-frequency channel.
- **Files:** `models/discrepancy.py`, `tests/test_discrepancy.py`

**Tasks**

- [ ] Implement `confidence_weight`, `phase_discrepancy` (use `X * conj(Y_K)`), `build_discrepancy_tensor`
- [ ] Support `mode='raw_phase'` for ablation (i)
- [ ] Un-skip all tests in `tests/test_discrepancy.py`

**Tests to un-skip:** test_discrepancy.py (all 6)

**Done when:** Identical phase gives zero residual; adding 2pi leaves D unchanged; low-energy bins are suppressed.

**Watch out**

- The report's Eq. 4 prints `X * Y`; the conjugate is intentional (see `ARCHITECTURE.md`).

### PR-17: Evaluation CLI and results registry

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr17-evaluation-cli-and-results`
- **Blocked by:** PR-04, PR-07
- **Reviewer(s):** C
- **Goal:** One standard on-disk format for every run so aggregation later is trivial.
- **Files:** `evaluation/results.py (new)`, `scripts/evaluate.py`, `docs/INTERFACES.md (section 6)`

**Tasks**

- [ ] Write `write_results(run_dir, cfg, metrics, scores)` producing `config.yaml`, `metrics.json`, `scores.csv`, git hash
- [ ] Implement `scripts/evaluate.py` for in-distribution and LOVO (`configs/eval/*.yaml`)
- [ ] Test it end-to-end using the released baseline

**Tests to un-skip:** tests/test_metrics.py (extend with a results round trip)

**Done when:** `python scripts/evaluate.py ...` on the baseline produces a complete results folder.

### PR-18: PhaseConNet assembly with DummyPhaseBranch and build_model

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr18-phaseconnet-assembly-with-dummyphasebranch`
- **Blocked by:** PR-11, PR-12
- **Reviewer(s):** B
- **Goal:** The full model, with the phase stream faked, so training code can be built before Nisha's branch exists.
- **Files:** `models/phasecon_net.py`, `models/__init__.py (build_model)`, `tests/test_phasecon_net.py`

**Tasks**

- [ ] Implement `PhaseConNet.forward` per `INTERFACES.md` section 5
- [ ] `phase_branch.enabled=false` must reproduce the baseline logits exactly
- [ ] Use `DummyPhaseBranch` when the real branch is unavailable
- [ ] Un-skip `forward_keys_and_shapes` and `phase_branch_disabled_equals_baseline`

**Tests to un-skip:** test_phasecon_net.py: two tests

**Done when:** Both tests pass; disabled phase branch equals the baseline.

### PR-19: Discrepancy encoder

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr19-discrepancy-encoder`
- **Blocked by:** PR-16, PR-08
- **Reviewer(s):** C
- **Goal:** Turn per-resolution discrepancy tensors into the sequence `H_d [B, T_d, C_d]` that fusion consumes.
- **Files:** `models/discrepancy_encoder.py`, `tests/test_discrepancy_encoder.py`

**Tasks**

- [ ] Per-resolution 2-D CNN, pooling over frequency, time-axis resampling to a common `T_d`, projection to `out_dim`
- [ ] Choose `T_d` using `T_b` from PR-08; set `model.phase_branch.encoder.t_out` in the config
- [ ] Un-skip both tests in `tests/test_discrepancy_encoder.py`

**Tests to un-skip:** test_discrepancy_encoder.py (both)

**Done when:** Output is `[B, T_d, out_dim]` for any resolution count; gradients reach every branch.

---

## Sprint 3: Assembly

### PR-20: Trainer, optimizer groups, checkpointing, early stopping

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr20-trainer-optimizer-groups-checkpointing`
- **Blocked by:** PR-04, PR-07, PR-15, PR-18
- **Reviewer(s):** A
- **Goal:** A training loop that is trusted before the real phase branch arrives.
- **Files:** `training/*.py`, `scripts/train.py`, `models/phasecon_net.py (param_groups)`, `tests/test_trainer.py`

**Tasks**

- [ ] Optimizer groups: small LR for the pretrained encoder, larger for new modules
- [ ] Cosine schedule with warmup; AMP; gradient clipping; resume
- [ ] Checkpoints store config snapshot and git hash; early stopping on dev EER (PR-07 metrics)
- [ ] `overfit_batch` debug mode; run it on the Dummy-branch model
- [ ] `scripts/train.py --config ... --override ...` works; un-skip `tests/test_trainer.py`

**Tests to un-skip:** test_trainer.py (all 3)

**Done when:** An epoch runs on dummy data and a single batch can be overfit.

### PR-21: PhaseBranch packaging (B to C hand-off)

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr21-phasebranch-packaging-b-to`
- **Blocked by:** PR-14, PR-16, PR-19, PR-08
- **Reviewer(s):** C
- **Goal:** One module that turns a waveform into `{H_d, cons_per_sample}` exactly as documented.
- **Files:** `models/phase_branch.py`, `tests/test_phase_branch.py`

**Tasks**

- [ ] Compose MultiResSTFT, ULCP, discrepancy tensor, discrepancy encoder per `INTERFACES.md` section 3
- [ ] Compute `cons_per_sample` = mean over resolutions, bins, frames of `w * (1 - cos dphi)`
- [ ] Ensure sample i does not depend on other samples in the batch
- [ ] Hand a real batch's output to Maanas and confirm the fusion block accepts it
- [ ] Un-skip `test_interface_keys` and `test_batch_independence`

**Tests to un-skip:** test_phase_branch.py (two of three)

**Done when:** Output keys and shapes match the interface on real data.

**Watch out**

- Do not change output keys or shapes without editing `INTERFACES.md` first.

> **GATE G3 follows this PR.** Stop and complete the gate checklist before merging PR-22.

### PR-22: Consistency loss

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr22-consistency-loss`
- **Blocked by:** PR-21
- **Reviewer(s):** C
- **Goal:** L_cons: pushes the projection to be accurate on bona fide audio only.
- **Files:** `losses/consistency.py`, `tests/test_phase_branch.py`

**Tasks**

- [ ] Implement `consistency_loss(cons_per_sample, label)`: mean over label == 0 only
- [ ] All-fake batch returns zero with the graph intact (no NaN)
- [ ] Un-skip `test_consistency_loss_masks_bona_fide`; remove the monkeypatch workaround from PR-15's tests

**Tests to un-skip:** test_phase_branch.py::test_consistency_loss_masks_bona_fide

**Done when:** Masks correctly, handles the all-fake batch, and `total_loss` consumes it.

### PR-23: Integrate the real PhaseBranch and run the overfit sanity check

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr23-integrate-the-real-phasebranch`
- **Blocked by:** PR-18, PR-20, PR-21, PR-22
- **Reviewer(s):** A, B
- **Goal:** Replace the dummy with the real stream and prove the full model can learn.
- **Files:** `models/phasecon_net.py`, `tests/test_phasecon_net.py`

**Tasks**

- [ ] Swap `DummyPhaseBranch` for `PhaseBranch` behind the config switch; remove default uses of the dummy
- [ ] Un-skip `backward_updates_new_modules` and `overfit_single_batch`
- [ ] Check AMP behaviour: ULCP/STFT stay fp32; no NaNs
- [ ] Record GPU memory and step time at K=5, R=3 in the PR

**Tests to un-skip:** test_phasecon_net.py: backward_updates_new_modules, overfit_single_batch

**Done when:** Full model overfits one real batch to near-zero loss without NaN.

**Watch out**

- If memory is too high, tell Nisha (PR-33 covers optimisation) and lower K or R for now.

> **GATE G4 follows this PR.** Stop and complete the gate checklist before merging PR-24.

---

## Sprint 4: First results

### PR-24: Pipeline validation: baseline through our trainer

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr24-pipeline-validation-baseline-through`
- **Blocked by:** PR-06, PR-09, PR-20
- **Reviewer(s):** A
- **Goal:** Prove our trainer and evaluator reproduce the baseline so later gains cannot be a pipeline artefact.
- **Files:** `configs/model/baseline_only.yaml`, `experiments/results/baseline_pipeline_*`

**Tasks**

- [ ] Train baseline-only (frozen encoder, then fine-tuned) with `scripts/train.py` on the subset
- [ ] Evaluate with `scripts/evaluate.py` and compare against PR-06 numbers
- [ ] Write the comparison table and tolerance in the PR

**Tests to un-skip:** none

**Done when:** Baseline-via-our-pipeline matches PR-06 within the documented tolerance.

### PR-25: First PhaseCon-Net training run (in-distribution)

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr25-first-phasecon-net-training`
- **Blocked by:** PR-09, PR-23, PR-24
- **Reviewer(s):** B
- **Goal:** The first real result.
- **Files:** `configs/train/*`, `experiments/results/phasecon_*`

**Tasks**

- [ ] Train PhaseCon-Net on the balanced subset
- [ ] Tune only `lr_new`, `mu`, `beta`, fusion width on **dev**; never on test
- [ ] Evaluate once on the test split and record EER / ACC / AUC and per-vocoder numbers
- [ ] Record seeds, config and checkpoint path

**Tests to un-skip:** none

**Done when:** `metrics.json` exists; the run is reproducible from its saved config.

**Watch out**

- Report the result whether it beats the baseline or not.

### PR-26: Baseline LOVO retraining

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr26-baseline-lovo-retraining`
- **Blocked by:** PR-13, PR-20, PR-24
- **Reviewer(s):** C
- **Goal:** Retrain the baseline architecture once per held-out vocoder (released weights saw all vocoders).
- **Files:** `evaluation/lovo.py`, `experiments/results/baseline_lovo_*`

**Tasks**

- [ ] Implement `run_lovo` for the baseline architecture
- [ ] Run all six folds with fixed seeds and the same hyperparameters as PR-24
- [ ] Save per-fold and pooled metrics through the results registry (PR-17)

**Tests to un-skip:** none

**Done when:** Six fold result folders exist with per-vocoder metrics.

> **GATE G5 follows this PR.** Stop and complete the gate checklist before merging PR-27.

---

## Sprint 5: Ablations and full experiments

### PR-27: Wire phase-branch ablations

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr27-wire-phase-branch-ablations`
- **Blocked by:** PR-23
- **Reviewer(s):** C
- **Goal:** Ablations (i), (ii), (vi) selectable purely by config.
- **Files:** `configs/ablation/a1*, a2*, a6_*`, `models/phase_branch.py`, `tests/test_phasecon_net.py`

**Tasks**

- [ ] Verify `mode: raw_phase` (i), `ulcp.learnable: false` (ii), K sweep and R sweep (vi) all build and run forward/backward
- [ ] Report time and memory per setting
- [ ] Un-skip `test_each_ablation_config_builds` for these configs

**Tests to un-skip:** test_phasecon_net.py::test_each_ablation_config_builds (B's configs)

**Done when:** Every phase-side ablation config runs forward and backward.

### PR-28: Wire remaining ablations

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr28-wire-remaining-ablations`
- **Blocked by:** PR-23
- **Reviewer(s):** B
- **Goal:** Ablations (iii), (iv), (v), (vii) selectable purely by config.
- **Files:** `configs/ablation/a3*, a4*, a5*, a7*`, `models/fusion.py`, `models/heads.py`, `training/*`

**Tasks**

- [ ] Verify concat fusion (iii), `mu=0` (iv), gradient reversal (v), frozen encoder (vii)
- [ ] Un-skip `test_each_ablation_config_builds` for the rest

**Tests to un-skip:** test_phasecon_net.py::test_each_ablation_config_builds (C's configs)

**Done when:** Every ablation config in `configs/ablation/` runs forward and backward.

### PR-29: Ablation runner

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr29-ablation-runner`
- **Blocked by:** PR-17, PR-27, PR-28
- **Reviewer(s):** C
- **Goal:** One command runs every ablation with fixed seeds and standard result folders.
- **Files:** `scripts/run_ablations.py`

**Tasks**

- [ ] Iterate `configs/ablation/*.yaml` and the K in {1,3,5,8} and R in {1,2,3} sweeps
- [ ] Support `--seeds`, `--subset`, `--dry-run` (1 epoch, tiny subset)
- [ ] Do a dry run of everything and paste the summary in the PR

**Tests to un-skip:** none

**Done when:** A dry run produces a `metrics.json` for every ablation.

### PR-30: Experiment campaign: LOVO for PhaseCon-Net, ablations, seeds

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr30-experiment-campaign-lovo-for`
- **Blocked by:** PR-25, PR-26, PR-29
- **Reviewer(s):** B, C
- **Goal:** Run everything that will appear in the final report, once, then freeze.
- **Files:** `experiments/results/*`, `docs/DATASET_NOTES.md`

**Tasks**

- [ ] PhaseCon-Net LOVO, six folds
- [ ] All ablations via `run_ablations.py`
- [ ] Headline comparison (baseline vs PhaseCon-Net) with at least 3 seeds
- [ ] Tag `results-v1` when finished; no more runs after that tag

**Tests to un-skip:** none

**Done when:** All runs complete, tagged, and listed in a run index in the PR.

**Watch out**

- Nisha and Maanas help monitor and re-run failed jobs, but only Liza merges result folders.

> **GATE G6 follows this PR.** Stop and complete the gate checklist before merging PR-31.

---

## Sprint 6: Analysis and wrap-up

### PR-31: Aggregation: tables and figures

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr31-aggregation-tables-and-figures`
- **Blocked by:** PR-30
- **Reviewer(s):** C
- **Goal:** One command turns `experiments/results/` into every table and plot.
- **Files:** `evaluation/aggregate.py`, `docs/figures/`, `docs/tables/`

**Tasks**

- [ ] In-distribution table; LOVO per-vocoder table; ablation table
- [ ] DET curves and per-vocoder bar plots
- [ ] Include seed mean and standard deviation for the headline comparison

**Tests to un-skip:** none

**Done when:** `make report` (or one documented command) regenerates all outputs.

### PR-32: Analysis: what helps and where it fails

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr32-analysis-what-helps-and`
- **Blocked by:** PR-30, PR-31
- **Reviewer(s):** A, B
- **Goal:** An honest written explanation of the ablation results.
- **Files:** `docs/ANALYSIS.md`

**Tasks**

- [ ] Which component gives the gain, with effect sizes compared to seed variance
- [ ] Per-vocoder failures, especially vocoder families different from the GAN ones
- [ ] Link to the limitations in the report (single corpus, compression sensitivity, Fourier-based vocoders)

**Tests to un-skip:** none

**Done when:** Every claim cites a table or figure from PR-31.

### PR-33: Performance and robustness

- **Owner:** Nisha (Member B: Phase Stream)
- **Branch:** `b/pr33-performance-and-robustness`
- **Blocked by:** PR-30
- **Reviewer(s):** C
- **Goal:** Cost and sensitivity numbers for the final report.
- **Files:** `docs/ARCHITECTURE.md`, `notebooks/`

**Tasks**

- [ ] Profile time and memory per batch at K=5, R=3; note mixed-precision behaviour
- [ ] If feasible, test sensitivity to resampling or compression on a small held-out set
- [ ] Write findings into `ARCHITECTURE.md`

**Tests to un-skip:** none

**Done when:** Profile numbers and robustness notes are written down.

### PR-34: Reproducibility pass

- **Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
- **Branch:** `a/pr34-reproducibility-pass`
- **Blocked by:** PR-31
- **Reviewer(s):** B
- **Goal:** Anyone can regenerate a result from a clean clone.
- **Files:** `README.md`, Makefile

**Tasks**

- [ ] README section 'Reproducing results' with exact commands
- [ ] Confirm seeds, config snapshots and git hashes exist in every result folder
- [ ] A teammate who did not run the experiment reproduces one row

**Tests to un-skip:** none

**Done when:** One result row reproduced from the README alone, by someone else.

### PR-35: Documentation and model card

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr35-documentation-and-model-card`
- **Blocked by:** PR-32
- **Reviewer(s):** A, B
- **Goal:** Current docs matching the final code.
- **Files:** `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/INTERFACES.md`

**Tasks**

- [ ] Update all docs to match what was actually built
- [ ] Model card: inputs, outputs, training data, intended use, limits

**Tests to un-skip:** none

**Done when:** No document contradicts the code.

### PR-36: Invention disclosure draft

- **Owner:** Maanas Nair (Member C: Model, Training & IPR)
- **Branch:** `c/pr36-invention-disclosure-draft`
- **Blocked by:** PR-32
- **Reviewer(s):** A, B
- **Goal:** A disclosure for the IPR cell that claims only what the evidence supports.
- **Files:** `ipr/invention_disclosure_template.md`

**Tasks**

- [ ] Fill each section from `experiments/results/` and `docs/ANALYSIS.md`
- [ ] State clearly that a formal prior-art search has not been done
- [ ] List every public disclosure so far (reports, talks, public repos); patentability needs non-prior disclosure

**Tests to un-skip:** none

**Done when:** Draft handed to the IPR contact; all three members signed off.

**Watch out**

- Check with the IPR cell **before** making the repo public.

> **GATE G7 follows this PR.** Stop and complete the gate checklist before merging anything else.

---

## GitHub setup (done once, in PR-01)

`scripts/create_github_issues.sh` (needs the GitHub CLI `gh`, logged in) creates:
- labels: `owner:A`, `owner:B`, `owner:C`, `gate`, and one `sprint:N` per sprint
- one milestone per sprint
- one issue per PR (`[PR-NN][A] title`) and one per gate, created in numeric order

GitHub numbers issues and pull requests from one shared counter, and gate issues sit between PR issues, so the issue for PR-02 is not `#2`. The `[PR-NN]` prefix in each title is the reliable key; use the real GitHub number in `Closes #<number>`.

Run it once, from one machine. It refuses to run if the repo already has issues. Windows PowerShell: `scripts/create_github_issues.ps1`; macOS, Linux or Git Bash: `scripts/create_github_issues.sh`. Each issue has the goal, blockers, task checklist and a link back to its section here.

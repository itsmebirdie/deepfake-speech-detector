#!/usr/bin/env bash
# Generated from docs/ISSUES.md. Run ONCE (it refuses to run if the repo already has issues). Requires: gh auth login.
# Windows PowerShell users: use scripts/create_github_issues.ps1 instead.
set -e
REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
if [ -n "$(gh issue list -R "$REPO" --state all --limit 1 --json number -q ".[0].number")" ]; then echo "This repo already has issues; refusing to run again."; exit 1; fi
echo "Creating labels, milestones and issues in $REPO"
gh label create "owner:A" --force -R "$REPO" >/dev/null
gh label create "owner:B" --force -R "$REPO" >/dev/null
gh label create "owner:C" --force -R "$REPO" >/dev/null
gh label create "gate" --force -R "$REPO" >/dev/null
gh label create "sprint:0" --force -R "$REPO" >/dev/null
gh label create "sprint:1" --force -R "$REPO" >/dev/null
gh label create "sprint:2" --force -R "$REPO" >/dev/null
gh label create "sprint:3" --force -R "$REPO" >/dev/null
gh label create "sprint:4" --force -R "$REPO" >/dev/null
gh label create "sprint:5" --force -R "$REPO" >/dev/null
gh label create "sprint:6" --force -R "$REPO" >/dev/null
gh api "repos/$REPO/milestones" -f title="Sprint 0: Foundations" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 1: Data, baseline and signal primitives" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 2: Feasibility gate and building blocks" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 3: Assembly" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 4: First results" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 5: Ablations and full experiments" >/dev/null 2>&1 || true
gh api "repos/$REPO/milestones" -f title="Sprint 6: Analysis and wrap-up" >/dev/null 2>&1 || true
gh issue create -R "$REPO" --title "[PR-01][A] Bootstrap repository, CI and baseline submodule" --label "owner:A,sprint:0" --milestone "Sprint 0: Foundations" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** none
**Reviewer(s):** B, C

**Goal:** Everyone starts from the same working repo, with CI and rules in place.

**Tasks**
- [ ] Create the GitHub repo and push the skeleton
- [ ] Protect `main`: PR required, 1 approval, CI green
- [ ] Fill `.github/CODEOWNERS` with the real GitHub handles
- [ ] Run `bash scripts/create_github_issues.sh` to create labels, milestones and all issues from this file
- [ ] Run `make baseline`; commit the submodule pointer; write the pinned commit hash in `docs/DATASET_NOTES.md`
- [ ] Every member runs `make install && pytest -q` on their machine and pastes the result in the PR

**Done when:** A fresh clone passes `make install && pytest -q` on all three machines.

Full details: `docs/ISSUES.md#pr-01`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G0] Foundations ready" --label "gate,sprint:0" --milestone "Sprint 0: Foundations" --body "$(cat <<'ISSUE_EOF'
Opens after PR-01 is merged. All three members comment `ack` when every box is true.

- [ ] Fresh clone passes `make install && pytest -q` on all three machines
- [ ] `main` is protected (PR + 1 review + green CI)
- [ ] Labels and milestones exist (ran `scripts/create_github_issues.sh`)
- [ ] Baseline submodule is pinned; commit hash is written in `docs/DATASET_NOTES.md`
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-02][A] Acquire LibriSeVoc and released weights; document them" --label "owner:A,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-01
**Reviewer(s):** B, C

**Goal:** Replace every guess about the dataset and baseline with a documented fact.

**Tasks**
- [ ] Run `make data` and `make weights`; unzip into `data/raw/LibriSeVoc`
- [ ] Copy the official `train.txt`, `dev.txt`, `test.txt` from the baseline repo to `data/splits/` (commit them)
- [ ] Write the folder layout, the **six vocoder names**, split sizes, duration statistics in `DATASET_NOTES.md`
- [ ] Fill `data.vocoders` in `configs/data/librisevoc.yaml`
- [ ] Record the baseline's input length, label convention and sample rate; fix `audio.num_samples` in `configs/default.yaml` if the assumed 64600 is wrong
- [ ] Resolve every `VERIFY` comment in `configs/`

**Done when:** `DATASET_NOTES.md` has no `_TBD_` for vocoder names, split sizes, input length, label convention.

Full details: `docs/ISSUES.md#pr-02`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-03][B] Multi-resolution STFT" --label "owner:B,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-01
**Reviewer(s):** C

**Goal:** A tested, differentiable STFT / iSTFT / consistency projection that everything in the phase stream builds on.

**Tasks**
- [ ] Implement `MultiResSTFT.forward`, `inverse`, `project` (S after S^-1) for each configured `(n_fft, hop, win)`
- [ ] Force float32 inside (disable autocast); output complex64
- [ ] Un-skip and pass every test in `tests/test_stft.py`

**Done when:** Round trip reconstructs the waveform, `project` is idempotent, gradients reach the input, fp32 under autocast.

Full details: `docs/ISSUES.md#pr-03`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-04][A] Dataset, transforms, splits and dataloader" --label "owner:A,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-02
**Reviewer(s):** C

**Goal:** A loader that returns exactly the batch in `INTERFACES.md` section 1.

**Tasks**
- [ ] Implement `pad_or_crop` exactly as the baseline treats short and long clips
- [ ] Implement `read_split_file` using the vocoder names from PR-02; enforce label 0 real / 1 fake, vocoder_id 0..6
- [ ] Implement `LibriSeVocDataset`, `collate_fn`, `build_dataloader`
- [ ] Un-skip: item shapes, pad/crop, no utt_id overlap between splits, collate shapes
- [ ] Iterate one full epoch over the real train split and paste throughput in the PR

**Done when:** Loader iterates a full epoch; tests pass.

Full details: `docs/ISSUES.md#pr-04`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-05][B] Fast Griffin-Lim reference and instantaneous-frequency utilities" --label "owner:B,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-03
**Reviewer(s):** A

**Goal:** Classical reference implementations used for the feasibility study, for unit-testing ULCP, and for ablation (ii).

**Tasks**
- [ ] Implement `fast_griffin_lim` with momentum and optional initial phase
- [ ] Implement `wrap_to_pi` and `instantaneous_frequency` (principal-value difference, hop and bin corrected)
- [ ] Un-skip and pass `tests/test_griffin_lim.py`

**Done when:** Magnitude preserved, spectral convergence improves with iterations, IF of a pure tone is correct.

Full details: `docs/ISSUES.md#pr-05`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-06][A] Reproduce the released baseline" --label "owner:A,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-02
**Reviewer(s):** C

**Goal:** Know the baseline's true numbers before comparing anything against it.

**Tasks**
- [ ] Run the authors' `eval.py` with the released weights on the official test split
- [ ] Wrap it in `scripts/eval_baseline.py` and save scores per utterance
- [ ] Record EER / accuracy / AUC in `DATASET_NOTES.md` beside the base paper's reported values
- [ ] Write down where the baseline computes EER and how its logits map to a score (PR-07 depends on this)
- [ ] If our numbers differ from the paper's, explain why in writing

**Done when:** Reproduced numbers are within a documented tolerance of the paper, or the gap is explained.

Full details: `docs/ISSUES.md#pr-06`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-07][A] Metrics and model-agnostic evaluator" --label "owner:A,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-04, PR-06
**Reviewer(s):** C

**Goal:** A single trusted place for EER, AUC and accuracy that every later comparison uses.

**Tasks**
- [ ] Implement `compute_eer`, `compute_auc`, `compute_accuracy`, `summarize`
- [ ] Score convention: `score = logits_bin[:,1] - logits_bin[:,0]`, higher means more synthetic
- [ ] Implement `run_inference` and `evaluate` (use only `logits_bin`; per-vocoder breakdown; optional scores CSV)
- [ ] Add the parity test: identical EER to the baseline repo's function on identical scores
- [ ] Un-skip all tests in `tests/test_metrics.py`

**Done when:** Parity test passes against the baseline's own EER.

Full details: `docs/ISSUES.md#pr-07`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-08][C] Baseline wrapper with feature access" --label "owner:C,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-02, PR-06
**Reviewer(s):** A

**Goal:** Expose the released RawNet2's pre-pooling feature sequence while keeping its original outputs bit-identical.

**Tasks**
- [ ] Add `third_party/...` to `sys.path` only inside this module; do not copy baseline code into `src/`
- [ ] Load the released weights; implement `forward_features` and `forward`
- [ ] Map label conventions between this repo (0 real / 1 fake) and the baseline
- [ ] Implement the `freeze_encoder` flag
- [ ] Write the discovered `T_b` and `C_b` into `INTERFACES.md` section 4
- [ ] Un-skip: logit parity with the baseline repo, feature shape, freeze flag, label mapping

**Done when:** Logits equal the baseline repo's on identical clips, and `T_b`, `C_b` are documented.

Full details: `docs/ISSUES.md#pr-08`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G1] Data and baseline verified" --label "gate,sprint:1" --milestone "Sprint 1: Data, baseline and signal primitives" --body "$(cat <<'ISSUE_EOF'
Opens after PR-08 is merged. All three members comment `ack` when every box is true.

- [ ] Loader iterates a full epoch on the official split with no errors
- [ ] Our EER equals the baseline repo's EER on identical scores
- [ ] Baseline test EER / ACC / AUC reproduced and recorded next to the paper's numbers
- [ ] `T_b`, `C_b` filled in `docs/INTERFACES.md` section 4
- [ ] `docs/DATASET_NOTES.md` has no `_TBD_` for vocoder names, split sizes, input length, label convention
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-09][A] Balanced subset manifests" --label "owner:A,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-04
**Reviewer(s):** C

**Goal:** A fixed, reproducible subset of LibriSeVoc so everyone trains on identical clips.

**Tasks**
- [ ] Implement `build_balanced_subset` (deterministic by seed; equal real/fake totals; fixed per-vocoder counts)
- [ ] Keep each real clip together with its resynthesis where possible
- [ ] Implement `save_manifest` and `scripts/make_subset.py`; commit the small manifest files
- [ ] Define the subset (counts, seed, files) in `DATASET_NOTES.md`; the final report needs this
- [ ] Un-skip determinism and balance tests

**Done when:** Same seed gives identical manifests; `data.subset.enabled=true` works in the loader.

Full details: `docs/ISSUES.md#pr-09`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-10][B] Phase feasibility study (GO / NO-GO)" --label "owner:B,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-04, PR-05, PR-09
**Reviewer(s):** A, C

**Goal:** Find out, before building the learnable parts, whether the phase residual separates real from synthetic.

**Tasks**
- [ ] Using **fixed** Griffin-Lim only, compute confidence-weighted `1 - cos(dphi)` per clip for real and each of the six vocoders
- [ ] Try zero initialisation and one other deterministic initialisation
- [ ] Report a per-vocoder, per-resolution separability number (e.g. AUC of the clip-level mean)
- [ ] Make residual maps with `scripts/visualize_phase.py` for a few real/fake pairs of the same utterance
- [ ] Write the verdict (GO / WEAK / NO-GO) in `ARCHITECTURE.md` under known issue 2

**Done when:** A written verdict exists and all three members have acknowledged it on the G2 gate issue.

Full details: `docs/ISSUES.md#pr-10`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G2] Phase go / no-go decision" --label "gate,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
Opens after PR-10 is merged. All three members comment `ack` when every box is true.

- [ ] Written verdict exists in `docs/ARCHITECTURE.md` (known issue 2): GO, WEAK, or NO-GO
- [ ] Per-vocoder, per-resolution separability numbers are in the PR
- [ ] All three members have commented `ack` on the gate issue
- [ ] If NO-GO or WEAK: hold a short planning call and **edit this file** (PR-14, 16, 19, 21, 22 may change) before anyone continues
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-11][C] Classification heads and gradient reversal" --label "owner:C,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-08
**Reviewer(s):** B

**Goal:** Binary and vocoder-ID heads, initialised from the baseline where shapes allow.

**Tasks**
- [ ] Implement `BinaryHead` (logits `[B,2]`) and `VocoderHead` (logits `[B, n_voc]`)
- [ ] Implement `GradReverse` and the `gradient_reversal` switch
- [ ] Initialise from the baseline's pretrained head weights where compatible
- [ ] Add tests: head output shapes; gradient reversal flips the sign of gradients

**Done when:** Heads run on pooled `[B, C_b]` features and the gradient-reversal test passes.

Full details: `docs/ISSUES.md#pr-11`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-12][C] Fusion blocks (gated cross-attention and concat)" --label "owner:C,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-08
**Reviewer(s):** B

**Goal:** Eq. 5 fusion, developed against random `H_d` of the documented shape. Do not wait for Nisha.

**Tasks**
- [ ] Implement `GatedCrossAttentionFusion`: query from the baseline stream, gate on a time-aligned `H_d`
- [ ] Handle `T_b != T_d`
- [ ] Implement `ConcatFusion` (ablation iii)
- [ ] Un-skip all tests in `tests/test_fusion.py`

**Done when:** Output shape equals `H_b`; a zero gate gives identity; gradients reach both streams.

Full details: `docs/ISSUES.md#pr-12`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-13][A] Leave-one-vocoder-out splits" --label "owner:A,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-04
**Reviewer(s):** C

**Goal:** Define and generate the LOVO protocol once, exactly, so the baseline and PhaseCon-Net are compared fairly.

**Tasks**
- [ ] Write the protocol in `DATASET_NOTES.md`: train/dev = real + all vocoders except the held-out one; test = real test clips + held-out fakes
- [ ] Implement `build_lovo_splits`, `write_split`, `make_lovo_splits.py`; generate six folds
- [ ] Un-skip `test_lovo_split_excludes_held_out`; add an assertion on utt_id overlap

**Done when:** Six fold manifests committed; held-out vocoder never appears in train or dev.

Full details: `docs/ISSUES.md#pr-13`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-14][B] ULCP: unrolled learnable consistency projection" --label "owner:B,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-10
**Reviewer(s):** C

**Goal:** Eqs. 1 to 3 as a module, with a non-learnable mode that equals fixed Griffin-Lim.

**Tasks**
- [ ] Implement `ULCPStep`: projection, learnable momentum, small conv correction, magnitude re-imposition
- [ ] Implement `ULCP` with deterministic init and K steps from config
- [ ] Add `learnable=False` mode (fixed momentum, no correction) that matches PR-05's reference
- [ ] Un-skip all tests in `tests/test_ulcp.py`; profile memory for K = 3, 5, 8

**Done when:** |Y_K| equals A for every K, K=0 returns the init, non-learnable mode matches fixed fast Griffin-Lim.

Full details: `docs/ISSUES.md#pr-14`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-15][C] Classification losses and combined loss (Eq. 6)" --label "owner:C,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-01
**Reviewer(s):** B

**Goal:** L = L_bin + beta * L_voc + mu * L_cons, tolerant of missing terms.

**Tasks**
- [ ] Implement `binary_loss` (same class weighting as the baseline if it uses one) and `vocoder_loss`
- [ ] Implement `total_loss`: skip terms whose keys are absent; return float parts for logging
- [ ] Import `consistency_loss` from `losses/consistency.py`; test the cons term with a monkeypatch until PR-22 lands
- [ ] Un-skip all tests in `tests/test_losses.py`

**Done when:** Composition test and missing-term test pass.

Full details: `docs/ISSUES.md#pr-15`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-16][B] Phase-discrepancy tensor" --label "owner:B,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-05, PR-10
**Reviewer(s):** C

**Goal:** Eq. 4: confidence-weighted cos/sin of the phase residual plus an instantaneous-frequency channel.

**Tasks**
- [ ] Implement `confidence_weight`, `phase_discrepancy` (use `X * conj(Y_K)`), `build_discrepancy_tensor`
- [ ] Support `mode='raw_phase'` for ablation (i)
- [ ] Un-skip all tests in `tests/test_discrepancy.py`

**Done when:** Identical phase gives zero residual; adding 2pi leaves D unchanged; low-energy bins are suppressed.

Full details: `docs/ISSUES.md#pr-16`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-17][A] Evaluation CLI and results registry" --label "owner:A,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-04, PR-07
**Reviewer(s):** C

**Goal:** One standard on-disk format for every run so aggregation later is trivial.

**Tasks**
- [ ] Write `write_results(run_dir, cfg, metrics, scores)` producing `config.yaml`, `metrics.json`, `scores.csv`, git hash
- [ ] Implement `scripts/evaluate.py` for in-distribution and LOVO (`configs/eval/*.yaml`)
- [ ] Test it end-to-end using the released baseline

**Done when:** `python scripts/evaluate.py ...` on the baseline produces a complete results folder.

Full details: `docs/ISSUES.md#pr-17`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-18][C] PhaseConNet assembly with DummyPhaseBranch and build_model" --label "owner:C,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-11, PR-12
**Reviewer(s):** B

**Goal:** The full model, with the phase stream faked, so training code can be built before Nisha's branch exists.

**Tasks**
- [ ] Implement `PhaseConNet.forward` per `INTERFACES.md` section 5
- [ ] `phase_branch.enabled=false` must reproduce the baseline logits exactly
- [ ] Use `DummyPhaseBranch` when the real branch is unavailable
- [ ] Un-skip `forward_keys_and_shapes` and `phase_branch_disabled_equals_baseline`

**Done when:** Both tests pass; disabled phase branch equals the baseline.

Full details: `docs/ISSUES.md#pr-18`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-19][B] Discrepancy encoder" --label "owner:B,sprint:2" --milestone "Sprint 2: Feasibility gate and building blocks" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-16, PR-08
**Reviewer(s):** C

**Goal:** Turn per-resolution discrepancy tensors into the sequence `H_d [B, T_d, C_d]` that fusion consumes.

**Tasks**
- [ ] Per-resolution 2-D CNN, pooling over frequency, time-axis resampling to a common `T_d`, projection to `out_dim`
- [ ] Choose `T_d` using `T_b` from PR-08; set `model.phase_branch.encoder.t_out` in the config
- [ ] Un-skip both tests in `tests/test_discrepancy_encoder.py`

**Done when:** Output is `[B, T_d, out_dim]` for any resolution count; gradients reach every branch.

Full details: `docs/ISSUES.md#pr-19`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-20][C] Trainer, optimizer groups, checkpointing, early stopping" --label "owner:C,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-04, PR-07, PR-15, PR-18
**Reviewer(s):** A

**Goal:** A training loop that is trusted before the real phase branch arrives.

**Tasks**
- [ ] Optimizer groups: small LR for the pretrained encoder, larger for new modules
- [ ] Cosine schedule with warmup; AMP; gradient clipping; resume
- [ ] Checkpoints store config snapshot and git hash; early stopping on dev EER (PR-07 metrics)
- [ ] `overfit_batch` debug mode; run it on the Dummy-branch model
- [ ] `scripts/train.py --config ... --override ...` works; un-skip `tests/test_trainer.py`

**Done when:** An epoch runs on dummy data and a single batch can be overfit.

Full details: `docs/ISSUES.md#pr-20`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-21][B] PhaseBranch packaging (B to C hand-off)" --label "owner:B,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-14, PR-16, PR-19, PR-08
**Reviewer(s):** C

**Goal:** One module that turns a waveform into `{H_d, cons_per_sample}` exactly as documented.

**Tasks**
- [ ] Compose MultiResSTFT, ULCP, discrepancy tensor, discrepancy encoder per `INTERFACES.md` section 3
- [ ] Compute `cons_per_sample` = mean over resolutions, bins, frames of `w * (1 - cos dphi)`
- [ ] Ensure sample i does not depend on other samples in the batch
- [ ] Hand a real batch's output to Maanas and confirm the fusion block accepts it
- [ ] Un-skip `test_interface_keys` and `test_batch_independence`

**Done when:** Output keys and shapes match the interface on real data.

Full details: `docs/ISSUES.md#pr-21`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G3] Interfaces delivered" --label "gate,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
Opens after PR-21 is merged. All three members comment `ack` when every box is true.

- [ ] `PhaseBranch` output matches `docs/INTERFACES.md` section 3 on real data (keys and shapes)
- [ ] Maanas confirms `H_d` works with the fusion block on real tensors
- [ ] Heads, fusion and wrapper (PR-08, 11, 12) are merged
- [ ] No interface change happened without a prior edit to `INTERFACES.md`
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-22][B] Consistency loss" --label "owner:B,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-21
**Reviewer(s):** C

**Goal:** L_cons: pushes the projection to be accurate on bona fide audio only.

**Tasks**
- [ ] Implement `consistency_loss(cons_per_sample, label)`: mean over label == 0 only
- [ ] All-fake batch returns zero with the graph intact (no NaN)
- [ ] Un-skip `test_consistency_loss_masks_bona_fide`; remove the monkeypatch workaround from PR-15's tests

**Done when:** Masks correctly, handles the all-fake batch, and `total_loss` consumes it.

Full details: `docs/ISSUES.md#pr-22`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-23][C] Integrate the real PhaseBranch and run the overfit sanity check" --label "owner:C,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-18, PR-20, PR-21, PR-22
**Reviewer(s):** A, B

**Goal:** Replace the dummy with the real stream and prove the full model can learn.

**Tasks**
- [ ] Swap `DummyPhaseBranch` for `PhaseBranch` behind the config switch; remove default uses of the dummy
- [ ] Un-skip `backward_updates_new_modules` and `overfit_single_batch`
- [ ] Check AMP behaviour: ULCP/STFT stay fp32; no NaNs
- [ ] Record GPU memory and step time at K=5, R=3 in the PR

**Done when:** Full model overfits one real batch to near-zero loss without NaN.

Full details: `docs/ISSUES.md#pr-23`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G4] End-to-end model trains" --label "gate,sprint:3" --milestone "Sprint 3: Assembly" --body "$(cat <<'ISSUE_EOF'
Opens after PR-23 is merged. All three members comment `ack` when every box is true.

- [ ] Full PhaseCon-Net does forward + backward on a real batch
- [ ] `overfit_batch` drives loss near zero on the full model
- [ ] AMP run does not produce NaNs (STFT/ULCP stay fp32)
- [ ] `DummyPhaseBranch` is no longer used by any config or script
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-24][C] Pipeline validation: baseline through our trainer" --label "owner:C,sprint:4" --milestone "Sprint 4: First results" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-06, PR-09, PR-20
**Reviewer(s):** A

**Goal:** Prove our trainer and evaluator reproduce the baseline so later gains cannot be a pipeline artefact.

**Tasks**
- [ ] Train baseline-only (frozen encoder, then fine-tuned) with `scripts/train.py` on the subset
- [ ] Evaluate with `scripts/evaluate.py` and compare against PR-06 numbers
- [ ] Write the comparison table and tolerance in the PR

**Done when:** Baseline-via-our-pipeline matches PR-06 within the documented tolerance.

Full details: `docs/ISSUES.md#pr-24`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-25][C] First PhaseCon-Net training run (in-distribution)" --label "owner:C,sprint:4" --milestone "Sprint 4: First results" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-09, PR-23, PR-24
**Reviewer(s):** B

**Goal:** The first real result.

**Tasks**
- [ ] Train PhaseCon-Net on the balanced subset
- [ ] Tune only `lr_new`, `mu`, `beta`, fusion width on **dev**; never on test
- [ ] Evaluate once on the test split and record EER / ACC / AUC and per-vocoder numbers
- [ ] Record seeds, config and checkpoint path

**Done when:** `metrics.json` exists; the run is reproducible from its saved config.

Full details: `docs/ISSUES.md#pr-25`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-26][A] Baseline LOVO retraining" --label "owner:A,sprint:4" --milestone "Sprint 4: First results" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-13, PR-20, PR-24
**Reviewer(s):** C

**Goal:** Retrain the baseline architecture once per held-out vocoder (released weights saw all vocoders).

**Tasks**
- [ ] Implement `run_lovo` for the baseline architecture
- [ ] Run all six folds with fixed seeds and the same hyperparameters as PR-24
- [ ] Save per-fold and pooled metrics through the results registry (PR-17)

**Done when:** Six fold result folders exist with per-vocoder metrics.

Full details: `docs/ISSUES.md#pr-26`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G5] First results trustworthy" --label "gate,sprint:4" --milestone "Sprint 4: First results" --body "$(cat <<'ISSUE_EOF'
Opens after PR-26 is merged. All three members comment `ack` when every box is true.

- [ ] Baseline trained through *our* pipeline matches PR-06 numbers within the documented tolerance
- [ ] PhaseCon-Net in-distribution numbers recorded with seeds and configs
- [ ] Baseline LOVO (all six folds) recorded
- [ ] Hyperparameters are frozen (chosen on dev only)
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-27][B] Wire phase-branch ablations" --label "owner:B,sprint:5" --milestone "Sprint 5: Ablations and full experiments" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-23
**Reviewer(s):** C

**Goal:** Ablations (i), (ii), (vi) selectable purely by config.

**Tasks**
- [ ] Verify `mode: raw_phase` (i), `ulcp.learnable: false` (ii), K sweep and R sweep (vi) all build and run forward/backward
- [ ] Report time and memory per setting
- [ ] Un-skip `test_each_ablation_config_builds` for these configs

**Done when:** Every phase-side ablation config runs forward and backward.

Full details: `docs/ISSUES.md#pr-27`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-28][C] Wire remaining ablations" --label "owner:C,sprint:5" --milestone "Sprint 5: Ablations and full experiments" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-23
**Reviewer(s):** B

**Goal:** Ablations (iii), (iv), (v), (vii) selectable purely by config.

**Tasks**
- [ ] Verify concat fusion (iii), `mu=0` (iv), gradient reversal (v), frozen encoder (vii)
- [ ] Un-skip `test_each_ablation_config_builds` for the rest

**Done when:** Every ablation config in `configs/ablation/` runs forward and backward.

Full details: `docs/ISSUES.md#pr-28`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-29][A] Ablation runner" --label "owner:A,sprint:5" --milestone "Sprint 5: Ablations and full experiments" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-17, PR-27, PR-28
**Reviewer(s):** C

**Goal:** One command runs every ablation with fixed seeds and standard result folders.

**Tasks**
- [ ] Iterate `configs/ablation/*.yaml` and the K in {1,3,5,8} and R in {1,2,3} sweeps
- [ ] Support `--seeds`, `--subset`, `--dry-run` (1 epoch, tiny subset)
- [ ] Do a dry run of everything and paste the summary in the PR

**Done when:** A dry run produces a `metrics.json` for every ablation.

Full details: `docs/ISSUES.md#pr-29`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-30][A] Experiment campaign: LOVO for PhaseCon-Net, ablations, seeds" --label "owner:A,sprint:5" --milestone "Sprint 5: Ablations and full experiments" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-25, PR-26, PR-29
**Reviewer(s):** B, C

**Goal:** Run everything that will appear in the final report, once, then freeze.

**Tasks**
- [ ] PhaseCon-Net LOVO, six folds
- [ ] All ablations via `run_ablations.py`
- [ ] Headline comparison (baseline vs PhaseCon-Net) with at least 3 seeds
- [ ] Tag `results-v1` when finished; no more runs after that tag

**Done when:** All runs complete, tagged, and listed in a run index in the PR.

Full details: `docs/ISSUES.md#pr-30`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G6] Experiments complete and frozen" --label "gate,sprint:5" --milestone "Sprint 5: Ablations and full experiments" --body "$(cat <<'ISSUE_EOF'
Opens after PR-30 is merged. All three members comment `ack` when every box is true.

- [ ] All six PhaseCon-Net LOVO folds finished
- [ ] All ablations (a1 to a7, K and R sweeps) finished
- [ ] Headline comparison has at least 3 seeds
- [ ] Git tag `results-v1` created; no further training runs after this tag
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-31][A] Aggregation: tables and figures" --label "owner:A,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-30
**Reviewer(s):** C

**Goal:** One command turns `experiments/results/` into every table and plot.

**Tasks**
- [ ] In-distribution table; LOVO per-vocoder table; ablation table
- [ ] DET curves and per-vocoder bar plots
- [ ] Include seed mean and standard deviation for the headline comparison

**Done when:** `make report` (or one documented command) regenerates all outputs.

Full details: `docs/ISSUES.md#pr-31`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-32][C] Analysis: what helps and where it fails" --label "owner:C,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-30, PR-31
**Reviewer(s):** A, B

**Goal:** An honest written explanation of the ablation results.

**Tasks**
- [ ] Which component gives the gain, with effect sizes compared to seed variance
- [ ] Per-vocoder failures, especially vocoder families different from the GAN ones
- [ ] Link to the limitations in the report (single corpus, compression sensitivity, Fourier-based vocoders)

**Done when:** Every claim cites a table or figure from PR-31.

Full details: `docs/ISSUES.md#pr-32`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-33][B] Performance and robustness" --label "owner:B,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Nisha (Member B: Phase Stream)
**Blocked by:** PR-30
**Reviewer(s):** C

**Goal:** Cost and sensitivity numbers for the final report.

**Tasks**
- [ ] Profile time and memory per batch at K=5, R=3; note mixed-precision behaviour
- [ ] If feasible, test sensitivity to resampling or compression on a small held-out set
- [ ] Write findings into `ARCHITECTURE.md`

**Done when:** Profile numbers and robustness notes are written down.

Full details: `docs/ISSUES.md#pr-33`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-34][A] Reproducibility pass" --label "owner:A,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Liza Borah (Member A: Data, Baseline & Evaluation)
**Blocked by:** PR-31
**Reviewer(s):** B

**Goal:** Anyone can regenerate a result from a clean clone.

**Tasks**
- [ ] README section 'Reproducing results' with exact commands
- [ ] Confirm seeds, config snapshots and git hashes exist in every result folder
- [ ] A teammate who did not run the experiment reproduces one row

**Done when:** One result row reproduced from the README alone, by someone else.

Full details: `docs/ISSUES.md#pr-34`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-35][C] Documentation and model card" --label "owner:C,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-32
**Reviewer(s):** A, B

**Goal:** Current docs matching the final code.

**Tasks**
- [ ] Update all docs to match what was actually built
- [ ] Model card: inputs, outputs, training data, intended use, limits

**Done when:** No document contradicts the code.

Full details: `docs/ISSUES.md#pr-35`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[PR-36][C] Invention disclosure draft" --label "owner:C,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
**Owner:** Maanas Nair (Member C: Model, Training & IPR)
**Blocked by:** PR-32
**Reviewer(s):** A, B

**Goal:** A disclosure for the IPR cell that claims only what the evidence supports.

**Tasks**
- [ ] Fill each section from `experiments/results/` and `docs/ANALYSIS.md`
- [ ] State clearly that a formal prior-art search has not been done
- [ ] List every public disclosure so far (reports, talks, public repos); patentability needs non-prior disclosure

**Done when:** Draft handed to the IPR contact; all three members signed off.

Full details: `docs/ISSUES.md#pr-36`. Do not start until all blockers are merged.
ISSUE_EOF
)"
gh issue create -R "$REPO" --title "[GATE G7] Delivery" --label "gate,sprint:6" --milestone "Sprint 6: Analysis and wrap-up" --body "$(cat <<'ISSUE_EOF'
Opens after PR-36 is merged. All three members comment `ack` when every box is true.

- [ ] One command regenerates every table and figure
- [ ] A teammate reproduced one result row from the README alone
- [ ] Analysis, model card and disclosure draft merged
- [ ] Disclosure draft claims only what ablations support
ISSUE_EOF
)"

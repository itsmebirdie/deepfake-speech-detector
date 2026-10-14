# Work Plan: PhaseCon-Net

Three members, three vertical slices, one set of contracts (`INTERFACES.md`). There are **no dates** here. Order is enforced by
**steps** (per person) and **gates** (shared checkpoints where everyone's work meets). A step is finished only when its
"Done when" line is true and its PR is merged.

| | Member A: Data, Baseline & Evaluation | Member B: Phase Stream | Member C: Model, Training & IPR |
|---|---|---|---|
| Suggested owner | Liza Borah | Nisha | Maanas Nair |
| Owns (code) | `data/`, `evaluation/`, `scripts/{make_*,eval_baseline,evaluate,run_ablations}`, `docs/DATASET_NOTES.md` | `signal/`, `models/{ulcp,discrepancy,discrepancy_encoder,phase_branch}.py`, `losses/consistency.py`, `scripts/visualize_phase.py` | `models/{baseline,fusion,heads,phasecon_net}`, `losses/{classification,combined}.py`, `training/`, `scripts/train.py`, `ipr/` |
| Core question | "Are our numbers trustworthy and comparable?" | "Does the phase residual carry signal, and can we compute it efficiently?" | "Does the fused model beat the baseline, and why?" |

Names are suggestions; swap freely, the split is by skill area (data/metrics, DSP/phase, deep-learning integration).

---

## Order of development at a glance

```
A1 ──► G0 ──► A2 ► A3 ► A4 ► A5 ──► G1 ─────────────────────────────┐
       │                                                              │
       ├─► B1 ► B2 ► B3 ──► G2 (go/no-go) ► B4 ► B5 ► B6 ► B7 ► B8 ──┤──► G3
       │                                                              │
       └─► C1* ► C2 ► C3 ► (C4 with DummyPhaseBranch) ► C5 ► C6 ──────┘
                                                                      │
 G3 ► G4 (end-to-end overfit) ► A6 ► C7 ► A7 ► G5 ► B9, C8, A8 ► G6 ► A9 ► A10, B10, C9–C11 ► G7
 (*C1 needs the baseline repo + weights from A2; until then C reads the baseline code only)
```

### Gates (everyone stops and syncs here)
| Gate | Meaning | Requires |
|---|---|---|
| **G0** | Skeleton merged, env reproducible, CI green | A1 |
| **G1** | Data loads, metrics verified, **baseline reproduced** | A2–A5 |
| **G2** | **Phase go/no-go**: does Δφ from *fixed* Griffin-Lim separate real from fake? | B1–B3 (needs A3 loader or a handful of hand-loaded clips) |
| **G3** | Interfaces delivered: `PhaseBranch` (B7) and baseline wrapper + heads (C1–C2) | B7, C1, C2 |
| **G4** | Full model runs forward+backward and **overfits one batch** | C4, C5, C6, B7, B8 |
| **G5** | In-distribution results for baseline-via-our-pipeline and PhaseCon-Net | A6, C7, A7 (baseline LOVO) |
| **G6** | All ablations and LOVO folds finished; results frozen | B9, C8, A8 |
| **G7** | Final tables, figures, report text, disclosure draft | A9, A10, B10, C9–C11 |

---

## Member A: Data, Baseline & Evaluation

Build order (each step unblocks the next; later steps listed with who they unblock).

**A1. Bootstrap the repo.** Create the Git repo from this skeleton, protect `main`, require CI + one review, run `make install`,
set up pre-commit. Run `make baseline` (adds the authors' repo as a submodule at a pinned commit).
*Done when:* fresh clone → `make install && pytest -q` passes (all skips) on all three machines. **→ G0, unblocks B and C.**

**A2. Acquire dataset and weights; document them.** `make data`, `make weights`; copy the official `train/dev/test.txt` to `data/splits/`.
Inspect the archive and fill `docs/DATASET_NOTES.md`: folder layout, the **six vocoder names** (write into `configs/data/librisevoc.yaml`),
split sizes, sample rate/duration statistics, baseline input length, baseline label convention. Resolve the `VERIFY` markers in the configs.
*Done when:* `DATASET_NOTES.md` has no `_TBD_` for these fields. **→ unblocks A3, C1.**

**A3. Dataset and loader.** Implement `transforms.py`, `splits.py:read_split_file`, `librisevoc.py`, `collate.py`, `build_dataloader`.
Enforce the repo label convention (0 real / 1 fake, vocoder_id 0..6). Handle short/long clips exactly like the baseline.
*Done when:* `tests/test_dataset.py` (item shapes, pad/crop, no split overlap, collate) passes and a loader iterates a full epoch. **→ unblocks B3, C6.**

**A4. Reproduce the released baseline.** Run the authors' `eval.py` on the official test split via `scripts/eval_baseline.py`; record EER/ACC/AUC in
`DATASET_NOTES.md` next to the numbers in the base paper. Note exactly where the baseline computes EER and how logits map to scores.
*Done when:* our reproduced numbers are within a small, documented tolerance of the paper's, or the discrepancy is explained in writing. **→ unblocks A5 parity test, C1 parity test.**

**A5. Metrics and model-agnostic evaluation.** Implement `metrics.py` (EER, AUC, accuracy at EER threshold and at 0.5), `evaluate.py`
(uses only `logits_bin`; score = `logits_bin[:,1] - logits_bin[:,0]`), per-vocoder breakdown, scores CSV dump.
*Done when:* `test_metrics.py` passes, including **identical EER to the baseline repo's function on identical scores**. **→ G1; unblocks C6 (early stopping).**

**A6. Balanced subset manifests.** Implement `subset.py` and `make_subset.py`. Deterministic by seed; equal real/fake totals; per-vocoder counts fixed;
keep real/re-synthesis pairs together. Commit the small manifest `.txt` files. Define the subset in `DATASET_NOTES.md` (the final report needs it).
*Done when:* determinism and balance tests pass; B and C can switch `data.subset.enabled=true` and get identical clips. **→ unblocks C7.**

**A7. Leave-one-vocoder-out splits and baseline retraining.** Implement `build_lovo_splits`, `write_split`, `make_lovo_splits.py`, and `lovo.py` for the **baseline architecture**
(retrained per fold, since the released weights saw every vocoder). Write the exact protocol in `DATASET_NOTES.md`.
*Done when:* six fold manifests exist, tests confirm the held-out vocoder never appears in train/dev, and baseline LOVO metrics are saved under `experiments/results/`. **→ G5 (together with C7).**

**A8. Ablation runner.** Implement `run_ablations.py`: iterates `configs/ablation/*.yaml` (and K ∈ {1,3,5,8}, R ∈ {1,2,3} sweeps), calls the trainer/evaluator with fixed seeds, writes the standard results layout.
*Done when:* a dry run (1 epoch, tiny subset) of every ablation config produces a `metrics.json`. **→ unblocks the real ablation runs (G6).**

**A9. Run and aggregate experiments.** Execute PhaseCon-Net LOVO folds and the ablation suite using C's trainer; implement `aggregate.py` to produce tables (in-distribution, LOVO per-vocoder, ablations) and plots (DET curves, per-vocoder bars).
*Done when:* one command regenerates every table and figure from `experiments/results/`. **→ G6 → G7.**

**A10. Reproducibility pass.** Seeds, config snapshots, commit hashes in every result folder; README "Reproducing results" section; confirm a clean clone can regenerate the in-distribution table.
*Done when:* a teammate who did not run the experiment reproduces one result row from the README alone.

---

## Member B: Phase Stream

Build bottom-up: signal primitives → feasibility check → learnable projection → discrepancy → encoder → packaged branch.

**B1. Multi-resolution STFT.** Implement `MultiResSTFT` (`forward`, `inverse`, `project = S∘S⁻¹`), configurable `(n_fft, hop, win)` triples, fp32 under autocast, correct window normalisation for perfect reconstruction.
*Done when:* `test_stft.py` passes (round-trip, idempotent projection, shapes, gradients, fp32). **→ unblocks B2, B4.**
(*Needs only PyTorch; can start at G0 without data.*)

**B2. Griffin-Lim reference and IF utilities.** Implement `fast_griffin_lim`, `instantaneous_frequency`, `wrap_to_pi`.
*Done when:* `test_griffin_lim.py` passes (magnitude preserved, spectral convergence improves with iterations, IF of a pure tone correct). **→ unblocks B3, B4, ablation ii.**

**B3. Feasibility check (go/no-go for the whole idea).** Using fixed Griffin-Lim only (no learning), compute the confidence-weighted Δφ statistics for real vs. synthetic clips from each of the six vocoders.
Use `scripts/visualize_phase.py` and a notebook; try zero-init and one other deterministic init; report a simple separability number (e.g. AUC of the mean weighted `1 - cos Δφ` per clip) per vocoder and per resolution.
Write findings in `docs/ARCHITECTURE.md` (known issue 2).
*Done when:* a written verdict exists. **→ G2.**
- **Go:** residual shows separation for several vocoders → continue to B4.
- **Weak/mixed:** record which vocoders/resolutions work; proceed but let the learnable ULCP and consistency loss carry more weight; tell A and C so ablation expectations are realistic.
- **No-go:** do not build B4 as specified. Agreed fallbacks (decide at G2): (a) initialise from a learned phase predictor, (b) compare against a mel-conditioned phase estimate, (c) feed raw-phase / IF features through the same fusion (ablation i becomes the main model). The report must say which was taken.

**B4. ULCP.** Implement `ULCPStep` and `ULCP` (Eqs. 1–3): deterministic init, learnable momentum αₖ, small conv correction g_θ, `learnable=False` mode equals fixed fast Griffin-Lim.
*Done when:* `test_ulcp.py` passes (K=0 → init, |Y_K| = A, equals fixed FGL when non-learnable, deterministic, gradcheck, memory scales ~linearly in K). **→ unblocks B5.**

**B5. Discrepancy tensor.** Implement `confidence_weight`, `phase_discrepancy` (with conjugate, see `ARCHITECTURE.md`), `build_discrepancy_tensor` (cos/sin/IF channels, `raw_phase` mode).
*Done when:* `test_discrepancy.py` passes (zero residual for identical phase, 2π-wrap invariance, weight range, low-energy suppression). **→ unblocks B6.**

**B6. Discrepancy encoder.** Implement `DiscrepancyEncoder`: per-resolution 2-D CNN → frequency pooling → time-axis resampling to common `T_d` → projection to `out_dim`.
*Done when:* `test_discrepancy_encoder.py` passes. **→ unblocks B7.**

**B7. Package `PhaseBranch` (hand-off to C).** Compose B1+B4+B5+B6 into one module with the **exact** output dict in `INTERFACES.md` §3, including `cons_per_sample`.
Verify with C that `H_d` shape works with C's fusion, using real tensors.
*Done when:* `test_phase_branch.py` passes and C has swapped `DummyPhaseBranch` for it without edits to either side. **→ G3.**

**B8. Consistency loss.** Implement `losses/consistency.py` (bona fide only, safe on all-fake batches).
*Done when:* `test_phase_branch.py::test_consistency_loss_masks_bona_fide` passes and C's `total_loss` consumes it. **→ G4.**

**B9. Phase-branch ablations.** Wire config flags and verify each builds and runs: (i) raw phase vs residual, (ii) fixed Griffin-Lim vs learnable ULCP, (vi) sweeps over K and R. Report compute/memory per setting.
*Done when:* `configs/ablation/{a1,a2,a6_*}.yaml` instantiate and pass a forward/backward in `test_phasecon_net.py::test_each_ablation_config_builds`. **→ G6.**

**B10. Performance and robustness.** Profile the branch, chunk long inputs if needed, check mixed-precision behaviour, evaluate sensitivity to compression/resampling if time allows (to ground the report's limitations section).
*Done when:* profile numbers (time, memory per batch at K=5, R=3) are written into `ARCHITECTURE.md`.

---

## Member C: Model, Training & IPR

Build the model against stubs first, then replace the stubs with B's real module at G3.

**C1. Baseline wrapper.** *(Read baseline code from G0; load weights after A2.)* Implement `BaselineRawNet2`: load released weights, expose `forward_features → H_b [B,T_b,C_b]`, preserve original heads' outputs, map label conventions, support `freeze_encoder`.
Write the discovered `T_b`, `C_b` into `INTERFACES.md` §4.
*Done when:* `test_baseline_wrapper.py::test_logit_parity_with_baseline_repo` passes (identical logits on identical clips) and the shapes are documented. **→ unblocks C2, and B6's choice of `T_d`.**

**C2. Heads.** Implement `BinaryHead`, `VocoderHead`, `GradReverse`; initialise heads from the baseline's pretrained heads where shapes allow.
*Done when:* heads run on `[B,C_b]` pooled features; gradient-reversal flips the sign of gradients in a unit test. **→ G3 (with B7).**

**C3. Fusion blocks.** Implement `GatedCrossAttentionFusion` (Eq. 5) and `ConcatFusion` (ablation iii). Develop against random `H_d` of the documented shape; do not wait for B.
*Done when:* `test_fusion.py` passes (shape equals `H_b`, zero-gate identity, gate in (0,1), `T_b ≠ T_d` works, gradients reach both streams). **→ unblocks C4.**

**C4. Assemble `PhaseConNet` and `build_model`.** Use `DummyPhaseBranch` until B7; `phase_branch.enabled=false` must reproduce the baseline exactly.
*Done when:* `test_phasecon_net.py::{forward_keys_and_shapes, phase_branch_disabled_equals_baseline}` pass. **→ unblocks C6.**

**C5. Losses.** Implement `binary_loss`, `vocoder_loss`, `total_loss` (Eq. 6; skip absent terms; log each part).
*Done when:* `test_losses.py` passes. **→ unblocks C6; consumes B8.**

**C6. Trainer.** Implement optimizer groups (small LR for the pretrained encoder, larger for new modules), cosine schedule with warmup, AMP (STFT/ULCP forced fp32), gradient clipping, checkpointing with config snapshot, early stopping on dev EER (A's metrics), `overfit_batch` debug mode, `scripts/train.py`.
*Done when:* `test_trainer.py` passes and `overfit_batch` drives the loss near zero on the full model. **→ G4.**

**C7. Pipeline validation, then first real training.**
1. Train **baseline-only** through our trainer (frozen, then fine-tuned) and confirm the numbers match A4. This proves the pipeline is not the source of any later gain.
2. Train PhaseCon-Net on the balanced subset (A6); tune only `lr_new`, `mu`, `beta`, fusion size on **dev**, never test.
3. Record hyperparameters and seeds.
*Done when:* both runs have `metrics.json` and the baseline-via-our-pipeline row matches A4. **→ G5.**

**C8. Remaining ablations.** Configs and code paths for (iv) `mu=0`, (v) gradient reversal, (vii) frozen vs fine-tuned encoder; check every ablation config builds. Coordinate with A8.
*Done when:* all `configs/ablation/*.yaml` pass `test_each_ablation_config_builds`. **→ G6.**

**C9. Analysis of what helps.** Work with A to interpret ablations honestly: which component gives the gain, effect sizes vs. seed variance (run ≥3 seeds for the headline comparison), where PhaseCon-Net fails (per-vocoder).
*Done when:* a short written analysis exists in `docs/` with seed-variance numbers.

**C10. Documentation.** Keep `ARCHITECTURE.md` and model docstrings current; add a model-card style summary (inputs, outputs, training data, limits).

**C11. Invention disclosure.** Fill `ipr/invention_disclosure_template.md` from the evidence only; keep the report's caveat that patentability needs a formal prior-art search and non-prior disclosure (check before any public repo/talk).
*Done when:* draft handed to the IPR cell contact. **→ G7.**

---

## Cross-member dependency matrix

| Step | Needs from others | Provides to others |
|---|---|---|
| A1 | none | env, CI, submodule → B1, C1 |
| A2 | A1 | weights, vocoder names, baseline facts → A3, C1, B3 |
| A3 | A2 | loader → B3, C6 |
| A4 | A2 | baseline numbers → A5, C1, C7 |
| A5 | A3, A4 | metrics/evaluator → C6, C7 |
| B1–B2 | A1 | STFT/GL primitives → B3, B4 |
| B3 | B2, A3 (or a few hand-loaded clips) | **go/no-go** → everyone |
| B4–B6 | B3 verdict | pieces → B7 |
| B7 | B4–B6, C1 (`T_b` informs `T_d`) | `PhaseBranch` → C4 |
| B8 | B7 | `consistency_loss` → C5 |
| C1 | A2, A4 | `H_b`, wrapper → C2–C4, B6/B7 |
| C3–C4 | C1 (shapes) | model → C6 |
| C6 | A3, A5, C5, B8 | trainer → C7, A7, A8 |
| A6 | A3 | manifests → C7 |
| A7 | A3, A5, C6 (trainer) | baseline LOVO → G5 |

## Working agreements
- **Branches/PRs:** `main` is protected. Branch names `a/<step>-<slug>`, `b/...`, `c/...`. Every PR ends with one line: "Closes step B4".
  Review by the owner of the *consuming* side when a PR touches an interface (e.g. B's `PhaseBranch` PR is reviewed by C).
- **Interfaces change only via `docs/INTERFACES.md`** first, in a PR tagging the other two.
- **Tests are the definition of done.** A step's tests (already specced, currently skipped in `tests/`) must be un-skipped and passing.
- **No test-set tuning.** Hyperparameters and early stopping use dev only; the test split is touched for final reporting runs.
- **Same splits, same metrics, same seeds** for every row in every table. Headline comparisons use ≥3 seeds.
- **Large files** (audio, weights, checkpoints) are never committed; manifests and configs are.
- **Honest reporting.** If a component does not help, the ablation table says so; the disclosure and report claim only what the evidence supports.
- **Blocked?** Work against the documented stub (e.g. `DummyPhaseBranch`, random `H_d`) and say so in the PR rather than waiting.
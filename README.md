# PhaseCon-Net

Neural-vocoder phase-discrepancy deepfake audio detector. Extends the vocoder-artifact detector of
Sun et al. (CVPRW 2023) with a learned phase-consistency residual stream and discrepancy-gated
cross-attention fusion. Single dataset: **LibriSeVoc**. Baseline: the authors' released RawNet2 +
vocoder-ID model.

Status: **boilerplate only.** Every module is a stub with a documented interface.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make install
make baseline      # adds third_party/Synthetic-Voice-Detection-Vocoder-Artifacts as a submodule
make data          # LibriSeVoc -> data/raw/
make weights       # released pretrained baseline -> weights/
pytest -q          # most tests are skipped until implemented
```

## Layout
| Path | Purpose |
|---|---|
| `configs/` | YAML configs (`_base_` inheritance). `ablation/` has one file per planned ablation |
| `src/phasecon/data/` | LibriSeVoc dataset, split parsing, balanced subsets, LOVO splits |
| `src/phasecon/signal/` | Multi-resolution STFT, fast Griffin-Lim reference, instantaneous frequency |
| `src/phasecon/models/` | ULCP, discrepancy tensor, discrepancy encoder, fusion, heads, PhaseCon-Net |
| `src/phasecon/models/baseline/` | Wrapper around the released RawNet2 encoder |
| `src/phasecon/losses/` | Binary, vocoder-ID, consistency, combined loss (Eq. 6) |
| `src/phasecon/training/` | Trainer, optimizer groups, checkpointing, callbacks |
| `src/phasecon/evaluation/` | EER / accuracy / AUC, evaluation loop, LOVO driver, result aggregation |
| `scripts/` | CLI entry points and download helpers |
| `tests/` | Unit tests (specified, skipped until implemented) |
| `docs/` | `WORK_PLAN.md`, `INTERFACES.md`, `ARCHITECTURE.md`, `DATASET_NOTES.md` |
| `ipr/` | Invention-disclosure template for the university IPR cell |

## Read first
1. `docs/WORK_PLAN.md`: who builds what, in what order
2. `docs/INTERFACES.md`: tensor shapes and module contracts all three members code against

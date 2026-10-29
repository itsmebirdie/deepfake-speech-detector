### Gate check (PR-23), batch source: real LibriSeVoc train batch (8 clips, labels [0, 1, 1, 1, 1, 1, 1, 1]), NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.15.0+cu134

| Check | Result | Detail |
|---|---|---|
| G3 PhaseBranch matches INTERFACES.md s.3 on a real batch | PASS | H_d [8, 43, 128], cons [8], H_b [8, 43, 1024]; real branch: True |
| G3 fusion accepts real H_d / H_b | PASS | shape ok: True, finite: True, grads reach both: True |
| G4 forward + backward on a real batch | PASS | loss 1.9584, grads finite: True, params without grad: 25 (10 dead in the plain baseline, 15 last ULCP step, 0 unexpected) |
| G4 AMP run without NaN | PASS | 30 steps fp16, non-finite skipped: 0, phase stream saw {'in': 'torch.float32', 'autocast': False, 'out': ['torch.float32']} |
| G4 overfit_batch drives loss to ~0 | PASS | 100 steps @ lr 0.0003: loss 1.963 -> 0.0921; classification part 0.0001; L_cons 0.919; acc 1.00; NaN skips 0 |
| G4 DummyPhaseBranch not in use | PASS | real branch: True |

**Cost at K=5, R=3** (batch 8 x 96000 samples, AMP True): GPU peak 6.40 GB, step time 0.95 s (median of 5, min 0.73 s).

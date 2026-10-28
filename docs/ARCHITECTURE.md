# Architecture notes

```
wav [B,T] --+--> BaselineRawNet2.forward_features --> H_b [B,T_b,C_b] ----------+
            |                                                                   v
            +--> PhaseBranch: MultiResSTFT -> ULCP -> discrepancy -> encoder -> H_d --> GatedCrossAttentionFusion -> H
                                     |                                                                    |
                                     +--> cons_per_sample (-> L_cons, bona fide only)        pool -> BinaryHead (logits_bin)
                                                                                                       -> VocoderHead (logits_voc)
```
Loss: `L = L_bin + beta*L_voc + mu*L_cons` (Eq. 6). Equation references follow the preliminary report.

## Known issues to resolve during implementation
1. **Eq. 4 conjugate.** The report prints `angle(X * Y^(K))`; a phase difference needs `X * conj(Y^(K))`. Implemented with the conjugate.
2. **Phase-retrieval ambiguity.** Griffin-Lim from a deterministic init converges to *a* consistent phase for the given magnitude,
   not necessarily the true one (global sign / local ambiguities). Whether the residual separates real from fake after K steps is an
   empirical question. Step B3 is a go/no-go check using fixed Griffin-Lim *before* investing in the learnable ULCP.
   Fallbacks are listed in WORK_PLAN.md.

   **PR-10 verdict (28 Oct 2026): WEAK.** Fixed fast Griffin-Lim, 100 dev utterances x 7 versions (700 clips), 24 kHz,
   first 4 s, 3 resolutions x 2 inits (`zeros`, `linear_group_delay`) x K in {5, 32}; clip score = mean of
   `w * (1 - cos dphi)` with `w = A / (A + tau)`, `tau = 1e-3` (`scripts/phase_feasibility.py`, `signal/residual.py`).
   Decision rule fixed before the run: GO = separability `max(AUC, 1-AUC)` >= 0.80 for >= 4 of 6 vocoders in some
   setting; WEAK = >= 0.65 for >= 2; else NO-GO.

   | Vocoder | Separability, registered score (range over the 12 settings) | AUC at n_fft=512, zeros, K=5 (fake vs real mean) |
   |---|---|---|
   | wavegrad | 0.89 to 0.98 | 0.973 (0.940 vs 0.835) |
   | diffwave | 0.52 to 0.58 | 0.425 |
   | parallel_wave_gan | 0.53 to 0.57 | 0.567 |
   | wavernn | **1.000** in every setting | 0.000 (0.591 vs 0.835: fakes LOWER) |
   | wavenet | 0.65 to 0.71 | 0.683 |
   | melgan | 0.52 to 0.61 | 0.567 |
   | pooled | AUC 0.49 to 0.55 | 0.536 (directions differ by vocoder, so pooling cancels) |

   Two vocoders pass 0.80, so the registered rule gives **WEAK**, not GO. Resolution, init and K change little.

   **Follow-up: what drives the two strong vocoders** (`scripts/phase_decomposition.py`, same 700 clips, n_fft=512,
   zeros, K=5). The score mixes a magnitude pattern (`w`) with the phase difference, so each part was tested alone (AUC):

   | Vocoder | Loudness (RMS) | Mean weight `w` (magnitude pattern) | Timing only: `sum(w(1-cos dphi)) / sum(w)` |
   |---|---|---|---|
   | wavegrad | 0.479 | **0.972** | 0.687 |
   | diffwave | 0.403 | 0.435 | 0.406 |
   | parallel_wave_gan | 0.453 | 0.592 | **0.247** |
   | wavernn | 0.510 | **0.000** | 0.423 |
   | wavenet | 0.535 | 0.699 | 0.361 |
   | melgan | 0.434 | 0.583 | 0.368 |

   - **WaveRNN's perfect separation is a magnitude effect, not a phase effect:** the clips are as loud as real ones,
     but their energy sits in fewer time-frequency bins (mean `w` separates perfectly), while the phase-only part is
     close to chance (0.42). WaveGrad is the opposite magnitude pattern (energy in more bins, a noise floor) plus some
     real phase signal (0.69).
   - **The phase-only signal is modest for every vocoder:** separability 0.58 to 0.75 (strongest for Parallel WaveGAN).
     This is what the learnable ULCP and the consistency loss have to amplify.
   - Cross-check with PR-06: a simple magnitude cue separates WaveRNN perfectly, yet the released baseline catches
     1.3% of WaveRNN fakes. That supports DATASET_NOTES section 8 (the released weights did not see these WaveRNN
     clips) and means a baseline fine-tuned on our data (PR-24) will probably learn WaveRNN by itself. **Any later
     PhaseCon-Net gain on WaveRNN must be checked against the fine-tuned baseline before it is credited to phase.**

   **Decision (WEAK path, WORK_PLAN step B3):** continue with PR-14, 16, 19, 21, 22 as designed; let the learnable
   ULCP (PR-14) and `L_cons` (PR-22) carry more weight; report per-vocoder numbers everywhere; keep ablation (i)
   (raw phase / IF instead of the residual) as the closest fallback. Needs the G2 acknowledgement of all three members.
3. **Vocoder families.** The report's survey emphasises GAN vocoders; LibriSeVoc also contains non-GAN vocoders (verify names in A2).
   LOVO results may differ by family; report per-vocoder numbers.
4. **Label / score conventions** differ between repos; only INTERFACES.md section 5 defines ours.

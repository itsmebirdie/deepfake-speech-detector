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
3. **Vocoder families.** The report's survey emphasises GAN vocoders; LibriSeVoc also contains non-GAN vocoders (verify names in A2).
   LOVO results may differ by family; report per-vocoder numbers.
4. **Label / score conventions** differ between repos; only INTERFACES.md section 5 defines ours.

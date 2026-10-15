# Invention disclosure (draft; owner: Member C, step C11)
Title: Detection of synthetic speech via learned magnitude-implied phase discrepancy
Inventors: Liza Borah, Nisha, Maanas Nair
## 1. Technical field / problem
## 2. Closest prior art (Sun et al. 2023; handcrafted phase features, Wu et al. 2015; others found in prior-art search)
## 3. Candidate inventive features (from report section 4.7)
1. Residual between observed STFT phase and phase from a learnable unrolled consistency projection of the magnitude
2. Multi-resolution, confidence-weighted cos/sin discrepancy tensors with an IF-discrepancy channel
3. Discrepancy-gated cross-attention fusion with a pretrained raw-waveform detector
4. Joint training with bona-fide consistency loss + vocoder-ID multitask loss
## 4. Experimental evidence (fill from results/; only claim what ablations support)
## 5. Public disclosures to date (reports, talks, repos: patentability requires non-prior disclosure)
## 6. Open items: formal prior-art search NOT yet done

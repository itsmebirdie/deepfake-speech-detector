| Run | EER % | AUC | Acc % | wavegrad EER % | diffwave EER % | parallel_wave_gan EER % | wavernn EER % | wavenet EER % | melgan EER % |
|---|---|---|---|---|---|---|---|---|---|
| reference (pr06) | 11.86 | 0.9438 | 75.37 | 0.11 | 0.19 | 0.17 | 36.54 | 8.14 | 0.51 |
| baseline_pipeline_finetune | 0.30 | 1.0000 | 99.68 | 0.00 | 0.08 | 0.23 | 0.38 | 0.42 | 0.11 |
| phasecon_seed1234 | 0.53 | 0.9998 | 99.61 | 0.11 | 0.23 | 0.27 | 0.83 | 0.68 | 0.15 |

Change vs the reference, in percentage points for EER and accuracy (negative EER = better):

| Run | EER % | AUC | Acc % | wavegrad EER % | diffwave EER % | parallel_wave_gan EER % | wavernn EER % | wavenet EER % | melgan EER % |
|---|---|---|---|---|---|---|---|---|---|
| baseline_pipeline_finetune | -11.56 | +0.0562 | +24.31 | -0.11 | -0.11 | +0.06 | -36.16 | -7.72 | -0.40 |
| phasecon_seed1234 | -11.33 | +0.0560 | +24.24 | +0.00 | +0.04 | +0.10 | -35.71 | -7.46 | -0.36 |

# sbc/wrong-guide-delta

SVI with a point-estimate (delta) guide: it must FAIL calibration (all its draws are one number).

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (replications 120, ranks 19), 0.5 s.

**Passed**: detected: the deliberately wrong method failed calibration

| quantity | chi-square p | ECDF p  | verdict                                                 | rank histogram                            |
| -------- | ------------ | ------- | ------------------------------------------------------- | ----------------------------------------- |
| m        | 7.2e-219     | 0.00025 | FAILS (U-shaped: the posterior is too narrow or biased) | 65 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 55 |

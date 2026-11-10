# sbc/wrong-guide-delta

SVI with a point-estimate (delta) guide: it must FAIL calibration (all its draws are one number).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 40, ranks 19), 0.2 s.

**Passed**: detected: the deliberately wrong method failed calibration

| quantity | chi-square p | ECDF p  | verdict                                                 | rank histogram                            |
| -------- | ------------ | ------- | ------------------------------------------------------- | ----------------------------------------- |
| m        | 1.2e-66      | 0.00025 | FAILS (U-shaped: the posterior is too narrow or biased) | 23 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17 |

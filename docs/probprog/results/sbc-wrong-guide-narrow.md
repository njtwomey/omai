# sbc/wrong-guide-narrow

The exact posterior with its spread scaled by 0.4, a too-confident guide: it must FAIL calibration.

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (replications 300, ranks 19), 0.1 s.

**Passed**: detected: the deliberately wrong method failed calibration

| quantity | chi-square p | ECDF p  | verdict                                                 | rank histogram                                   |
| -------- | ------------ | ------- | ------------------------------------------------------- | ------------------------------------------------ |
| m        | 2.3e-108     | 0.00025 | FAILS (U-shaped: the posterior is too narrow or biased) | 78 19 14 10 10 4 7 7 1 8 6 5 4 9 10 13 3 4 15 73 |

# kernels/sbc-chees

Simulation-based calibration of the ChEES-HMC: the chains of one ensemble are independent draws, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 26.1 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                     |
| -------- | ------------ | ------ | ------- | -------------------------------------------------- |
| m        | 0.12         | 0.53   | passes  | 13 17 8 12 9 3 10 11 7 8 10 8 8 18 14 8 4 10 12 10 |

# kernels/sbc-slice

Simulation-based calibration of the coordinate-wise slice sampling, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 1.2 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                     |
| -------- | ------------ | ------ | ------- | -------------------------------------------------- |
| m        | 0.60         | 0.65   | passes  | 9 7 9 6 17 10 15 8 6 12 15 12 10 8 9 10 10 10 10 7 |

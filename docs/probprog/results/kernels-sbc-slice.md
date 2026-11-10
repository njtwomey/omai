# kernels/sbc-slice

Simulation-based calibration of the coordinate-wise slice sampling, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 0.7 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                           |
| -------- | ------------ | ------ | ------- | ---------------------------------------- |
| m        | 0.81         | 0.73   | passes  | 3 4 5 6 5 2 4 8 5 5 2 6 5 5 10 5 5 7 3 5 |

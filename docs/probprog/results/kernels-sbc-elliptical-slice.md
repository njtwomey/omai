# kernels/sbc-elliptical-slice

Simulation-based calibration of the elliptical slice sampling, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 0.4 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                           |
| -------- | ------------ | ------ | ------- | ---------------------------------------- |
| m        | 0.58         | 0.73   | passes  | 6 5 3 6 4 4 6 5 5 6 3 8 4 5 7 1 4 4 3 11 |

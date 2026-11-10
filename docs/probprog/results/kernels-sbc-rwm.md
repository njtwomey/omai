# kernels/sbc-rwm

Simulation-based calibration of the random-walk Metropolis with an adapted covariance, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 1.2 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                          |
| -------- | ------------ | ------ | ------- | --------------------------------------- |
| m        | 0.93         | 0.97   | passes  | 4 4 5 5 6 7 2 7 6 5 5 4 8 2 7 7 4 3 4 5 |

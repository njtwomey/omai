# kernels/sbc-rwm

Simulation-based calibration of the random-walk Metropolis with an adapted covariance, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 2.2 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                    |
| -------- | ------------ | ------ | ------- | ------------------------------------------------- |
| m        | 0.84         | 0.87   | passes  | 9 12 3 9 12 13 8 7 14 9 11 11 9 12 13 7 9 12 11 9 |

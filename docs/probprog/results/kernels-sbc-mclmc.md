# kernels/sbc-mclmc

Simulation-based calibration of the microcanonical Langevin Monte Carlo (unadjusted: a small step-size bias), through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 2.6 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                            |
| -------- | ------------ | ------ | ------- | ----------------------------------------- |
| a        | 0.023        | 0.73   | passes  | 5 0 10 5 1 3 8 3 8 5 4 10 8 4 1 5 9 3 4 4 |
| b        | 0.99         | 0.97   | passes  | 6 7 4 5 3 3 5 4 7 6 6 6 4 5 7 5 6 3 5 3   |

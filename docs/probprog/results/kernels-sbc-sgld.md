# kernels/sbc-sgld

Simulation-based calibration of the stochastic-gradient Langevin dynamics with the exact gradient (a step-size bias), through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 2.8 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                          |
| -------- | ------------ | ------ | ------- | --------------------------------------- |
| a        | 0.90         | 0.73   | passes  | 8 6 3 5 8 6 4 2 5 4 4 4 6 7 6 4 4 2 7 5 |
| b        | 0.96         | 0.89   | passes  | 4 3 6 2 7 4 6 6 4 4 6 5 6 5 9 3 5 5 4 6 |

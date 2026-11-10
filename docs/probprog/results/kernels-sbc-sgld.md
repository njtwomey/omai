# kernels/sbc-sgld

Simulation-based calibration of the stochastic-gradient Langevin dynamics with the exact gradient (a step-size bias), through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 5.5 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                       |
| -------- | ------------ | ------ | ------- | ---------------------------------------------------- |
| a        | 0.78         | 0.42   | passes  | 8 10 7 16 12 13 10 15 10 8 8 10 9 10 11 11 6 11 10 5 |
| b        | 0.97         | 0.87   | passes  | 9 10 16 11 7 7 9 11 12 10 10 8 12 12 13 8 9 8 9 9    |

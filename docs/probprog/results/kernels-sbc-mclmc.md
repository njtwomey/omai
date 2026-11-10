# kernels/sbc-mclmc

Simulation-based calibration of the microcanonical Langevin Monte Carlo (unadjusted: a small step-size bias), through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 4.9 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                       |
| -------- | ------------ | ------ | ------- | ---------------------------------------------------- |
| a        | 0.46         | 0.65   | passes  | 13 7 10 7 7 10 15 13 13 10 6 10 16 9 11 12 4 10 11 6 |
| b        | 0.50         | 0.098  | passes  | 12 12 6 9 6 9 8 7 10 12 9 4 15 11 8 9 12 14 11 16    |

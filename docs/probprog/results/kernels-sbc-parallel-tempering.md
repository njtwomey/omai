# kernels/sbc-parallel-tempering

Simulation-based calibration of the parallel tempering over HMC, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 15.4 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                      |
| -------- | ------------ | ------ | ------- | --------------------------------------------------- |
| m        | 0.84         | 0.94   | passes  | 13 12 11 7 13 10 8 9 7 7 7 10 15 11 11 12 7 13 6 11 |

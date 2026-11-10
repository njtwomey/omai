# kernels/sbc-hmc

Simulation-based calibration of the HMC with a fixed trajectory and window adaptation, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 4.6 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                     |
| -------- | ------------ | ------ | ------- | -------------------------------------------------- |
| m        | 0.48         | 0.32   | passes  | 13 7 12 12 10 11 10 12 10 5 7 8 8 9 5 9 14 15 7 16 |

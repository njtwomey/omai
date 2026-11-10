# kernels/sbc-elliptical-slice

Simulation-based calibration of the elliptical slice sampling, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 0.6 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                    |
| -------- | ------------ | ------ | ------- | ------------------------------------------------- |
| m        | 0.94         | 0.94   | passes  | 8 8 10 8 14 12 12 6 13 9 9 11 9 9 13 11 6 13 11 8 |

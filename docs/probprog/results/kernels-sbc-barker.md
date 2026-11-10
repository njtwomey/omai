# kernels/sbc-barker

Simulation-based calibration of the Barker's proposal with window adaptation, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 2.4 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                     |
| -------- | ------------ | ------ | ------- | -------------------------------------------------- |
| m        | 0.72         | 0.76   | passes  | 7 14 11 11 14 8 13 8 9 4 12 11 15 11 9 10 9 6 8 10 |

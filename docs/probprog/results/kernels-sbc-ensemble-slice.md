# kernels/sbc-ensemble-slice

Simulation-based calibration of the ensemble slice sampling: the walkers of one iteration are independent draws, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 17.5 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                    |
| -------- | ------------ | ------ | ------- | ------------------------------------------------- |
| m        | 0.21         | 0.32   | passes  | 7 6 7 8 13 13 17 6 10 7 7 15 11 9 12 17 10 9 10 6 |

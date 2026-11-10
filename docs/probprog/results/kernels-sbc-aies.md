# kernels/sbc-aies

Simulation-based calibration of the the affine-invariant ensemble sampler: the walkers of one iteration are independent draws, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 3.9 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                       |
| -------- | ------------ | ------ | ------- | ---------------------------------------------------- |
| m        | 0.54         | 0.94   | passes  | 8 8 11 14 6 11 13 3 13 12 7 12 13 10 11 9 5 10 14 10 |

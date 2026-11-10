# kernels/sbc-aies

Simulation-based calibration of the the affine-invariant ensemble sampler: the walkers of one iteration are independent draws, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (replications 100, ranks 19), 2.2 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                          |
| -------- | ------------ | ------ | ------- | --------------------------------------- |
| m        | 0.20         | 0.57   | passes  | 8 9 2 2 2 8 3 7 6 4 2 4 5 9 5 4 7 1 6 6 |

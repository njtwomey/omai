# kernels/sbc-ensemble-slice

Simulation-based calibration of the ensemble slice sampling: the walkers of one iteration are independent draws, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 100, ranks 19), 14.1 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| m | 0.35 | 0.29 | passes | 2 6 5 3 3 6 1 11 6 5 7 4 8 5 4 8 3 3 5 5 |

# kernels/sbc-parallel-tempering

Simulation-based calibration of the parallel tempering over HMC, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (replications 100, ranks 19), 8.7 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| m | 0.37 | 0.12 | passes | 10 6 6 6 8 2 6 4 2 8 6 4 6 4 1 4 6 3 2 6 |

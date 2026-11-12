# kernels/sbc-hybrid

Simulation-based calibration of the the user plan: the exact conjugate conditional of b with NUTS on a, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 100, ranks 19), 3.8 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| a | 0.14 | 0.29 | passes | 4 4 5 8 3 3 11 3 3 4 9 7 5 10 2 4 3 3 4 5 |
| b | 0.25 | 0.57 | passes | 5 2 3 3 12 6 3 4 6 7 7 3 2 4 7 5 6 6 2 7 |

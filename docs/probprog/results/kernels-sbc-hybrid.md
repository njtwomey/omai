# kernels/sbc-hybrid

Simulation-based calibration of the the user plan: the exact conjugate conditional of b with NUTS on a, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 6.6 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                      |
| -------- | ------------ | ------ | ------- | --------------------------------------------------- |
| a        | 0.44         | 0.76   | passes  | 6 15 11 13 9 7 8 8 11 10 5 14 11 14 6 7 9 8 13 15   |
| b        | 0.92         | 0.32   | passes  | 10 7 10 12 10 12 6 11 9 8 8 12 6 8 9 10 12 13 15 12 |

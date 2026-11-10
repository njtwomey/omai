# kernels/sbc-mala

Simulation-based calibration of the MALA with window adaptation, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 2 (replications 200, ranks 19), 2.2 s.

**Passed**: calibrated over 200 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                                    |
| -------- | ------------ | ------ | ------- | ------------------------------------------------- |
| m        | 0.89         | 0.76   | passes  | 9 9 12 6 12 11 12 12 9 8 11 8 9 16 12 12 8 10 5 9 |

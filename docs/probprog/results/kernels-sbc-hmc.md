# kernels/sbc-hmc

Simulation-based calibration of the HMC with a fixed trajectory and window adaptation, through its method (knobs, warm-up, thinning and posterior).

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 100, ranks 19), 2.9 s.

**Passed**: calibrated over 100 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| m | 0.95 | 0.57 | passes | 5 7 4 4 6 4 5 3 5 5 4 7 2 6 3 3 8 6 7 6 |

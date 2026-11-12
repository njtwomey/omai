# sbc/wrong-guide-narrow

The exact posterior with its spread scaled by 0.4, a too-confident guide: it must FAIL calibration.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 100, ranks 19), 0.0 s.

**Passed**: detected: the deliberately wrong method failed calibration

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| m | 8.7e-51 | 0.00025 | FAILS (U-shaped: the posterior is too narrow or biased) | 33 3 4 4 3 0 2 1 0 3 1 2 1 3 1 5 0 1 6 27 |

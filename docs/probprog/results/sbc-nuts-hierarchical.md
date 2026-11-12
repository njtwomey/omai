# sbc/nuts-hierarchical

NUTS on a four-group hierarchical model, non-centred by `reparam`: calibration of mu, tau and the first group effect.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 40, ranks 19, warmup 200, draws 380), 31.6 s.

**Passed**: calibrated over 40 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram |
| --- | --- | --- | --- | --- |
| mu | 0.52 | 0.46 | passes | 0 1 3 2 0 1 3 6 1 3 3 2 1 2 2 3 1 2 3 1 |
| tau | 0.66 | 0.46 | passes | 4 1 3 3 2 0 4 3 3 1 1 1 4 3 2 2 1 1 0 1 |
| theta[0] | 0.84 | 0.94 | passes | 3 0 1 4 4 3 2 1 3 1 2 1 2 1 4 2 2 1 2 1 |

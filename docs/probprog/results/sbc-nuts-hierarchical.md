# sbc/nuts-hierarchical

NUTS on a four-group hierarchical model, non-centred by `reparam`: calibration of mu, tau and the first group effect.

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (replications 120, ranks 19, warmup 200, draws 380), 90.1 s.

**Passed**: calibrated over 120 replications

| quantity | chi-square p | ECDF p | verdict | rank histogram                            |
| -------- | ------------ | ------ | ------- | ----------------------------------------- |
| mu       | 0.27         | 0.66   | passes  | 4 3 5 8 4 6 9 11 4 9 9 7 1 4 4 7 4 9 8 4  |
| tau      | 0.59         | 0.19   | passes  | 8 7 8 7 8 5 5 8 6 4 4 11 5 5 10 5 2 3 5 4 |
| theta[0] | 0.46         | 0.041  | passes  | 6 4 3 10 8 9 8 6 10 8 8 5 6 4 7 6 3 2 4 3 |

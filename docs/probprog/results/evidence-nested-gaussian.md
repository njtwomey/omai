# evidence/nested-gaussian

Nested sampling against the analytic evidence of a Gaussian likelihood under a Gaussian prior in 1, 2, 5 and 10 dimensions: the error against the stated error over seeds, and how often the stated error covers the truth.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (seeds 12, dimensions 4), 31.6 s.

**Passed**: the stated error covers the analytic evidence at its nominal rate in every dimension, with no bias beyond it

Prior N(0, 3^2 I), likelihood exp(-|x - y|^2 / (2 0.5^2)); 12 seeds per dimension, slice sampling with 5 sweeps per replacement. Error is the estimate minus the analytic log Z; coverage counts |error| < 2 stated error.

| dim | live points | mean error | sd of error | mean stated error | coverage | likelihood evaluations | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 150 | 0.035 | 0.087 | 0.094 | 12/12 | 24938 | ok |
| 2 | 150 | 0.018 | 0.131 | 0.134 | 12/12 | 66391 | ok |
| 5 | 150 | -0.059 | 0.089 | 0.212 | 12/12 | 294147 | ok |
| 10 | 200 | -0.061 | 0.224 | 0.259 | 12/12 | 1329865 | ok |

The stated error is the information-based `sqrt(H / n)` of Skilling; its calibration is what this table checks. The cost grows with the information `H` (about the dimension times the log of the prior-to-posterior width ratio).

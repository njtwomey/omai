# evidence/harmonic-mean

The harmonic mean estimator is refused by default; shown with its infinite variance acknowledged, on exact posterior draws of a diffuse-prior sharp-likelihood model, it fails where bridge sampling on the same draws does not.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (replications 40, draws 400, widths 3), 0.3 s.

**Passed**: refused by default; its error is more than three times bridge sampling's at every likelihood width

Prior N(0, 10^2), likelihood exp(-theta^2 / (2 s^2)) at three widths s; 40 replications of 400 exact posterior draws each. Bias and rms error of the log evidence against the exact one.

| likelihood sd s | log Z | harmonic bias | harmonic rms | bridge bias | bridge rms | rms ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | -2.997 | 1.888 | 1.963 | -0.000 | 0.006 | 313.9 |
| 0.10 | -4.605 | 3.565 | 3.576 | 0.001 | 0.003 | 1241.9 |
| 0.02 | -6.215 | 5.170 | 5.176 | -0.000 | 0.004 | 1329.5 |

The harmonic mean is biased upwards (a few draws near the likelihood's peak dominate its average) and its error grows as the likelihood narrows against the prior; the stated standard error of the estimator does not show it.

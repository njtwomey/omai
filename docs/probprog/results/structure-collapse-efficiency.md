# structure/collapse-efficiency

NUTS on eight schools with the group effects integrated out by collapse, against the full model.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (chains 4, warmup 500, draws 1000), 39.3 s.

**Passed**: collapsed NUTS gets 42 times the ESS of tau per gradient evaluation of the full centred model, with no divergences

Eight schools (Rubin's data), four chains of 500 warm-up and 1000 draws, target acceptance 0.9. Collapse plan for eight-schools-centred: integrate out `theta`, leaving 2 of 10 unconstrained coordinates (`mu`, `tau`) for a sampler.

| run | ESS mu | ESS tau | gradient evaluations | ESS tau per gradient | ESS mu per gradient | divergences |
| --- | --- | --- | --- | --- | --- | --- |
| full (centred) | 626 | 147 | 105269 | 1.39e-3 | 5.95e-3 | 19 |
| collapsed (theta integrated out) | 1941 | 1306 | 22390 | 5.83e-2 | 8.67e-2 | 0 |
| non-centred, no collapse | 5302 | 2906 | 57112 | 5.09e-2 | 9.28e-2 | 0 |

Gradient evaluations are the leapfrog steps of the sampling draws (warm-up is not counted, as the sampler does not report it). The collapsed program is the marginal likelihood of `y` given `mu` and `tau`; the draws of `theta` come back from `recover` and are not part of the cost.

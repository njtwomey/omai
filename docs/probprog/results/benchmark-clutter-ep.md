# benchmark/clutter-ep

Minka's clutter problem: EP with numeric sites, the Laplace fit and the mean-field guide against the exact posterior.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (scalePercent 100), 5.5 s.

**Passed**: EP converges under every rule; the integrating rules land on the dominant mode

**The clutter problem** (n = 12): a grid of 12001 values of theta over [-60, 60] (spacing 0.0100), the product of the prior and the n mixture densities, the clutter indicators summed out in closed form; the evidence is the trapezoid integral.

Exact posterior of theta: mean 6.395, sd 2.317, dominant mode near 7.8.

| method | mean | sd | mean error (exact sd) | sd ratio | run |
| --- | --- | --- | --- | --- | --- |
| EP, quadrature rule | 7.788 | 0.671 | 0.601 | 0.290 | converged |
| EP, gauss-hermite rule | 7.788 | 0.671 | 0.601 | 0.290 | converged |
| EP, laplace rule | 2.823 | 0.461 | -1.542 | 0.199 | converged |
| nuts | 7.483 | 1.345 | 0.470 | 0.581 | reported |
| laplace | 2.821 | 0.461 | -1.542 | 0.199 | reported |
| svi | 2.851 | 0.504 | -1.529 | 0.217 | reported |

EP with numeric sites integrates each clutter factor against its cavity by the rule given and projects the tilted distribution to a Gaussian; with two modes it settles on one (the integrating rules on the dominant one, the Laplace rule on whichever its sites lead to), as the Laplace fit and the mean-field guide (single normals at one mode) do; the exact mean sits between the modes.

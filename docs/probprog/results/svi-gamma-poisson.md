# svi/gamma-poisson

Gamma-Poisson rate: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 3, rows 3, scalePercent 100), 0.4 s.

**Passed**: 2 method(s) agree with the exact truth; 1 reported

**Gamma-Poisson rate** (gamma-poisson): conjugate, exact, counts, one-dimensional, positive

The rate of Poisson counts under a Gamma(2, 0.5) prior: conjugate, a Gamma posterior on the positive reals.

Truth (exact): the Gamma(a + sum k, b + n) posterior and its closed-form evidence (probprog/validation/exact.ts). Log evidence -30.5610.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: elliptical-slice (the prior of rate is gamma, not normal or multivariateNormal); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'rate' is continuous); em (no params to estimate); particle-filter (the model has no single scan); gaussian ('rate' is gamma, not Gaussian); enumeration (continuous latent nodes (rate)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| svi-conjugate-natural | must agree | completed (1 steps) | - | batch CAVI: the Gamma posterior in closed form |
| pathfinder | must agree | converged (6 steps) | weights ESS 1986 | a normal on the log rate, corrected by Pareto-smoothed importance weights towards the Gamma posterior |
| svi-stl | reported | completed (1500 steps) | - | a normal on the log rate: the right location and spread, the skew of the Gamma is not representable |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svi-conjugate-natural | rate | 3.23 | 0.459 | 0.000197 | 0.0124 | 1.01 | 0.00776 | 0.0117 | 0.00214 | 4000 |
| pathfinder | rate | 3.23 | 0.447 | 0.0148 | 0.671 | 0.98 | 0.0176 | 0.0238 | 0.00254 | 1990 |
| svi-stl | rate | 3.19 | 0.422 | -0.0888 | -6.07 | 0.926 | 0.0511 | 0.102 | 0.0115 | 4000 |

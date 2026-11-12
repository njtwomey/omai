# benchmark/gamma-poisson

Gamma-Poisson rate: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (plans 7, rows 7, scalePercent 100), 3.6 s.

**Passed**: 5 method(s) agree with the exact truth; 2 reported

**Gamma-Poisson rate** (gamma-poisson): conjugate, exact, counts, one-dimensional, positive

The rate of Poisson counts under a Gamma(2, 0.5) prior: conjugate, a Gamma posterior on the positive reals.

Truth (exact): the Gamma(a + sum k, b + n) posterior and its closed-form evidence (probprog/validation/exact.ts). Log evidence -30.5610.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: elliptical-slice (the prior of rate is gamma, not normal or multivariateNormal); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'rate' is continuous); em (no params to estimate); particle-filter (the model has no single scan); gaussian ('rate' is gamma, not Gaussian); enumeration (continuous latent nodes (rate)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| conjugate | must agree | converged (1 steps) | - | the exact posterior of this model: it is the truth |
| gibbs | must agree | completed (2300 steps) | - | draws each site from its conditional: exact conditionals, so consistent |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| importance | must agree | completed (10 steps) | weights ESS 5076 | weighted draws from the prior: unbiased, with an effective sample size that shrinks as the posterior concentrates |
| tempered-smc | must agree | converged (2 steps) | weights ESS 655 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| laplace | reported | converged (7 steps) | - | a normal at the mode of log rate: close, with a small skew missed |
| svi | reported | completed (1500 steps) | - | a mean-field normal on log rate: close, with a small skew missed |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conjugate | rate | 3.23 | 0.45 | 0.0115 | 0.734 | 0.987 | 0.0149 | 0.0205 | 0.00219 | 4000 |
| gibbs | rate | 3.22 | 0.452 | -0.00467 | -0.203 | 0.992 | 0.011 | 0.0161 | 0.00298 | 1860 |
| nuts | rate | 3.22 | 0.43 | -0.0228 | -0.489 | 0.942 | 0.0377 | 0.0531 | 0.014 | 410 |
| importance | rate | 3.23 | 0.459 | -0.0000143 | -0.00101 | 1.01 | 0.0109 | 0.011 | 0.00138 | 5080 |
| tempered-smc | rate | 3.22 | 0.442 | -0.00316 | -0.103 | 0.969 | 0.0179 | 0.0298 | 0.00528 | 1000 |
| laplace | rate | 3.27 | 0.462 | 0.0957 | 5.97 | 1.01 | 0.0388 | 0.0959 | 0.00563 | 4000 |
| svi | rate | 3.33 | 0.445 | 0.222 | 14.4 | 0.974 | 0.095 | 0.222 | 0.0318 | 4000 |

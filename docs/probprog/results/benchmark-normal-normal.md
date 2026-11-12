# benchmark/normal-normal

Normal-normal mean: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (plans 8, rows 8, scalePercent 100), 2.7 s.

**Passed**: 7 method(s) agree with the exact truth; 1 reported

**Normal-normal mean** (normal-normal): conjugate, exact, gaussian, linear-gaussian, one-dimensional

The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.

Truth (exact): the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts). Log evidence -23.9485.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, gaussian, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); particle-filter (the model has no single scan); enumeration (continuous latent nodes (mu)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| conjugate | must agree | converged (1 steps) | - | the exact posterior of this model: it is the truth |
| gaussian | must agree | converged (1 steps) | - | the exact posterior of this model: it is the truth |
| gibbs | must agree | completed (2300 steps) | - | draws each site from its conditional: exact conditionals, so consistent |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| importance | must agree | completed (10 steps) | weights ESS 4671 | weighted draws from the prior: unbiased, with an effective sample size that shrinks as the posterior concentrates |
| tempered-smc | must agree | converged (2 steps) | weights ESS 595 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| laplace | must agree | converged (4 steps) | - | exact: the posterior is normal and the Laplace fit is a normal |
| svi | reported | completed (1500 steps) | - | a mean-field normal is the right family here; what is left is the noise of the stochastic optimiser |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conjugate | mu | 1.4 | 0.42 | 0.0108 | 0.688 | 0.994 | 0.0133 | 0.0168 | 0.00236 | 4000 |
| gaussian | mu | 1.41 | 0.418 | 0.0151 | 0.966 | 0.987 | 0.0214 | 0.0243 | 0.00436 | 4000 |
| gibbs | mu | 1.42 | 0.428 | 0.0502 | 2.15 | 1.01 | 0.0322 | 0.055 | 0.00502 | 1880 |
| nuts | mu | 1.36 | 0.427 | -0.0975 | -1.92 | 1.01 | 0.0512 | 0.0984 | 0.0155 | 395 |
| importance | mu | 1.4 | 0.42 | 0.00141 | 0.0973 | 0.993 | 0.00733 | 0.0108 | 0.00164 | 4670 |
| tempered-smc | mu | 1.41 | 0.425 | 0.0352 | 1.11 | 1 | 0.0272 | 0.0382 | 0.00444 | 1000 |
| laplace | mu | 1.4 | 0.423 | 0.00678 | 0.428 | 1 | 0.00918 | 0.0153 | 0.00286 | 4000 |
| svi | mu | 1.49 | 0.394 | 0.202 | 13.7 | 0.932 | 0.0916 | 0.203 | 0.0312 | 4000 |

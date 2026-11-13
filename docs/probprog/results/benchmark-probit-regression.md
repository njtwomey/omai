# benchmark/probit-regression

Bayesian probit regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 5, rows 15, scalePercent 100), 34.9 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Bayesian probit regression** (probit-regression): glm, classification, probit, quadrature, three-dimensional, slice

Probit regression with an intercept and two covariates and a N(0, 4) prior: the slice model of the plan, the test of expectation propagation, with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 2.5e-11). Log evidence -28.8592.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for w).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| tempered-smc | must agree | converged (5 steps) | weights ESS 618 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| importance | reported | completed (10 steps) | weights ESS 67 | weights from the prior: the posterior is concentrated, so the effective sample size is small |
| laplace | reported | converged (13 steps) | - | near Gaussian at this sample size: the mean is close, the spread slightly off |
| svi | reported | completed (1500 steps) | - | a mean-field normal ignores the correlation of the intercept and the slopes: spread a little too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | w[0] | 0.218 | 0.208 | 0.037 | 1.17 | 0.959 | 0.0505 | 0.0689 | 0.0197 | 924 |
| nuts | w[1] | 0.948 | 0.273 | 0.0313 | 0.926 | 1.04 | 0.0222 | 0.0404 | 0.013 | 942 |
| nuts | w[2] | -0.449 | 0.185 | -0.029 | -0.952 | 0.946 | 0.0422 | 0.0688 | 0.0137 | 967 |
| tempered-smc | w[0] | 0.207 | 0.213 | -0.0155 | -0.5 | 0.979 | 0.0215 | 0.0355 | 0.0121 | 1000 |
| tempered-smc | w[1] | 0.946 | 0.268 | 0.0212 | 0.66 | 1.02 | 0.0198 | 0.0329 | 0.00766 | 1000 |
| tempered-smc | w[2] | -0.442 | 0.2 | 0.0037 | 0.114 | 1.02 | 0.0188 | 0.0307 | 0.00898 | 1000 |
| importance | w[0] | 0.202 | 0.216 | -0.0354 | -0.292 | 0.997 | 0.0969 | 0.0692 | 0.0972 | 67.3 |
| importance | w[1] | 0.967 | 0.284 | 0.104 | 0.792 | 1.08 | 0.129 | 0.131 | 0.32 | 67.3 |
| importance | w[2] | -0.442 | 0.199 | 0.00769 | 0.0621 | 1.02 | 0.0766 | 0.0913 | 0.0915 | 67.3 |
| laplace | w[0] | 0.202 | 0.216 | -0.0385 | -2.45 | 0.993 | 0.0189 | 0.0407 | 0.00283 | 4000 |
| laplace | w[1] | 0.886 | 0.26 | -0.204 | -13.1 | 0.987 | 0.0693 | 0.204 | 0.0274 | 4000 |
| laplace | w[2] | -0.416 | 0.192 | 0.138 | 8.91 | 0.983 | 0.0504 | 0.138 | 0.0136 | 4000 |
| svi | w[0] | 0.19 | 0.207 | -0.0908 | -6.02 | 0.954 | 0.0523 | 0.0957 | 0.00993 | 4000 |
| svi | w[1] | 0.954 | 0.247 | 0.0512 | 3.45 | 0.938 | 0.0521 | 0.0996 | 0.0143 | 4000 |
| svi | w[2] | -0.486 | 0.203 | -0.22 | -13.4 | 1.04 | 0.101 | 0.221 | 0.0253 | 4000 |

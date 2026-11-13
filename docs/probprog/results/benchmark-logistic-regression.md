# benchmark/logistic-regression

Bayesian logistic regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 5, rows 15, scalePercent 100), 19.4 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Bayesian logistic regression** (logistic-regression): glm, classification, logistic, quadrature, three-dimensional

Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 6.0e-9). Log evidence -32.2069.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for w).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| tempered-smc | must agree | converged (4 steps) | weights ESS 541 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| importance | reported | completed (10 steps) | weights ESS 223 | weights from the prior: the posterior is concentrated, so the effective sample size is small |
| laplace | reported | converged (13 steps) | - | near Gaussian at this sample size: the mean is close, the spread slightly off |
| svi | reported | completed (1500 steps) | - | a mean-field normal ignores the correlation of the intercept and the slopes: spread a little too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | w[0] | 0.274 | 0.353 | 0.125 | 3.6 | 1.02 | 0.077 | 0.13 | 0.0176 | 865 |
| nuts | w[1] | 1.03 | 0.388 | -0.0071 | -0.193 | 1.04 | 0.0305 | 0.0576 | 0.0126 | 807 |
| nuts | w[2] | -0.641 | 0.308 | -0.00364 | -0.0979 | 0.976 | 0.0247 | 0.0387 | 0.00832 | 688 |
| tempered-smc | w[0] | 0.227 | 0.341 | -0.0124 | -0.395 | 0.988 | 0.0295 | 0.0468 | 0.0183 | 1000 |
| tempered-smc | w[1] | 1.05 | 0.364 | 0.0377 | 1.22 | 0.977 | 0.0263 | 0.0593 | 0.00887 | 1000 |
| tempered-smc | w[2] | -0.617 | 0.308 | 0.0716 | 2.32 | 0.975 | 0.0523 | 0.0878 | 0.0151 | 1000 |
| importance | w[0] | 0.252 | 0.348 | 0.0606 | 0.897 | 1.01 | 0.065 | 0.0799 | 0.0475 | 223 |
| importance | w[1] | 1.04 | 0.365 | 0.0227 | 0.345 | 0.982 | 0.0487 | 0.0509 | 0.0546 | 223 |
| importance | w[2] | -0.651 | 0.334 | -0.0354 | -0.501 | 1.06 | 0.049 | 0.0731 | 0.0797 | 223 |
| laplace | w[0] | 0.224 | 0.334 | -0.019 | -1.25 | 0.967 | 0.0201 | 0.0527 | 0.00771 | 4000 |
| laplace | w[1] | 0.938 | 0.354 | -0.259 | -17.2 | 0.952 | 0.103 | 0.259 | 0.0408 | 4000 |
| laplace | w[2] | -0.582 | 0.302 | 0.185 | 12.3 | 0.956 | 0.0703 | 0.185 | 0.0267 | 4000 |
| svi | w[0] | 0.265 | 0.4 | 0.0989 | 5.4 | 1.16 | 0.0638 | 0.133 | 0.0203 | 4000 |
| svi | w[1] | 1.01 | 0.331 | -0.0597 | -4.24 | 0.89 | 0.0483 | 0.0981 | 0.0241 | 4000 |
| svi | w[2] | -0.693 | 0.301 | -0.168 | -11.1 | 0.954 | 0.0903 | 0.18 | 0.0238 | 4000 |

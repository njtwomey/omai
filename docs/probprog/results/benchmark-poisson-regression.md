# benchmark/poisson-regression

Poisson regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 5, rows 10, scalePercent 100), 12.0 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Poisson regression** (poisson-regression): glm, counts, poisson, log-link, quadrature, two-dimensional

Counts with a log-linear mean in one covariate and N(0, 4) priors on the intercept and the slope: a two-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 161 points per axis over 2 coordinate(s) (32482 density evaluations, evidence error 4.4e-12). Log evidence -179.4826.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for a, b); gaussian ('k' is poisson, not Gaussian); enumeration (continuous latent nodes (a, b)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method       | expectation | run                    | diagnostics     | why                                                                                             |
| ------------ | ----------- | ---------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (1001 steps) | 0 divergences   | a consistent sampler: agrees within Monte Carlo error                                           |
| tempered-smc | must agree  | converged (7 steps)    | weights ESS 919 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS                   |
| importance   | reported    | completed (10 steps)   | weights ESS 22  | weights from the prior: the posterior is concentrated, so the effective sample size is small    |
| laplace      | reported    | converged (15 steps)   | -               | near Gaussian: close to the grid                                                                |
| svi          | reported    | completed (1500 steps) | -               | a mean-field normal: close in the means, spread a little too small where a and b are correlated |

| method       | quantity | mean  | sd     | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ----- | ------ | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | a        | 0.535 | 0.13   | 0.0705          | 1.05   | 1.03     | 0.0471 | 0.0772  | 0.00906 | 236  |
| nuts         | b        | 0.716 | 0.0586 | -0.0632         | -0.941 | 1.04     | 0.0348 | 0.0757  | 0.0132  | 239  |
| tempered-smc | a        | 0.527 | 0.125  | 0.0107          | 0.34   | 0.992    | 0.0236 | 0.0295  | 0.0085  | 1000 |
| tempered-smc | b        | 0.717 | 0.0556 | -0.0338         | -1.08  | 0.986    | 0.0267 | 0.0387  | 0.00586 | 1000 |
| importance   | a        | 0.55  | 0.1    | 0.197           | 1.16   | 0.798    | 0.207  | 0.258   | 0.97    | 22   |
| importance   | b        | 0.709 | 0.0477 | -0.173          | -0.961 | 0.845    | 0.193  | 0.205   | 0.612   | 22   |
| laplace      | a        | 0.532 | 0.124  | 0.0472          | 3.02   | 0.989    | 0.0288 | 0.0484  | 0.003   | 4000 |
| laplace      | b        | 0.718 | 0.0557 | -0.0189         | -1.21  | 0.987    | 0.0212 | 0.0249  | 0.00413 | 4000 |
| svi          | a        | 0.536 | 0.0425 | 0.0816          | 15.3   | 0.338    | 0.266  | 0.531   | 1.61    | 4000 |
| svi          | b        | 0.723 | 0.0194 | 0.0685          | 12.6   | 0.345    | 0.265  | 0.526   | 1.58    | 4000 |

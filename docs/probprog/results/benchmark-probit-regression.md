# benchmark/probit-regression

Bayesian probit regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 5, rows 15, scalePercent 100), 36.3 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Bayesian probit regression** (probit-regression): glm, classification, probit, quadrature, three-dimensional, slice

Probit regression with an intercept and two covariates and a N(0, 4) prior: the slice model of the plan, the test of expectation propagation, with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 2.5e-11). Log evidence -28.8592.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method       | expectation | run                    | diagnostics     | why                                                                                                    |
| ------------ | ----------- | ---------------------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| nuts         | must agree  | converged (1001 steps) | 0 divergences   | a consistent sampler: agrees within Monte Carlo error                                                  |
| tempered-smc | must agree  | converged (5 steps)    | weights ESS 577 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS                          |
| importance   | reported    | completed (10 steps)   | weights ESS 67  | weights from the prior: the posterior is concentrated, so the effective sample size is small           |
| laplace      | reported    | converged (13 steps)   | -               | near Gaussian at this sample size: the mean is close, the spread slightly off                          |
| svi          | reported    | completed (1500 steps) | -               | a mean-field normal ignores the correlation of the intercept and the slopes: spread a little too small |

| method       | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | w[0]     | 0.22   | 0.214 | 0.0443          | 1.34   | 0.986    | 0.0483 | 0.0575  | 0.0152  | 886  |
| nuts         | w[1]     | 0.942  | 0.271 | 0.00691         | 0.206  | 1.03     | 0.0139 | 0.0206  | 0.00741 | 936  |
| nuts         | w[2]     | -0.446 | 0.184 | -0.015          | -0.484 | 0.941    | 0.0455 | 0.0654  | 0.0222  | 916  |
| tempered-smc | w[0]     | 0.208  | 0.218 | -0.00828        | -0.261 | 1        | 0.0271 | 0.0324  | 0.00879 | 1000 |
| tempered-smc | w[1]     | 0.951  | 0.256 | 0.0431          | 1.4    | 0.971    | 0.0384 | 0.0586  | 0.00878 | 1000 |
| tempered-smc | w[2]     | -0.44  | 0.188 | 0.014           | 0.462  | 0.961    | 0.0286 | 0.0495  | 0.00808 | 1000 |
| importance   | w[0]     | 0.202  | 0.216 | -0.0354         | -0.292 | 0.997    | 0.0969 | 0.0692  | 0.0972  | 67.3 |
| importance   | w[1]     | 0.967  | 0.284 | 0.104           | 0.792  | 1.08     | 0.129  | 0.131   | 0.32    | 67.3 |
| importance   | w[2]     | -0.442 | 0.199 | 0.00769         | 0.0621 | 1.02     | 0.0766 | 0.0913  | 0.0915  | 67.3 |
| laplace      | w[0]     | 0.202  | 0.216 | -0.0385         | -2.45  | 0.993    | 0.0189 | 0.0407  | 0.00283 | 4000 |
| laplace      | w[1]     | 0.886  | 0.26  | -0.204          | -13.1  | 0.987    | 0.0693 | 0.204   | 0.0274  | 4000 |
| laplace      | w[2]     | -0.416 | 0.192 | 0.138           | 8.91   | 0.983    | 0.0504 | 0.138   | 0.0136  | 4000 |
| svi          | w[0]     | 0.19   | 0.207 | -0.0908         | -6.02  | 0.954    | 0.0523 | 0.0957  | 0.00993 | 4000 |
| svi          | w[1]     | 0.954  | 0.247 | 0.0512          | 3.45   | 0.938    | 0.0521 | 0.0996  | 0.0143  | 4000 |
| svi          | w[2]     | -0.486 | 0.203 | -0.22           | -13.4  | 1.04     | 0.101  | 0.221   | 0.0253  | 4000 |

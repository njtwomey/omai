# kernels/agreement-logistic-regression

Bayesian logistic regression: the kernel methods against the quadrature truth.

Run on 2026-11-10 at commit bcd4c412; seed 0, budget scale 1 (plans 13, rows 39, scalePercent 100), 30.9 s.

**Passed**: 9 kernel method(s) agree with the quadrature truth; 4 reported

**Bayesian logistic regression** (logistic-regression): glm, classification, logistic, quadrature, three-dimensional

Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 6.0e-9). Log evidence -32.2069.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for w).

| method             | expectation | run                    | diagnostics                                                            | why                                                                         |
| ------------------ | ----------- | ---------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| rwm                | must agree  | converged (6001 steps) | 0 divergences                                                          | a consistent sampler, slower mixing: agrees within Monte Carlo error        |
| mala               | must agree  | converged (1101 steps) | 0 divergences                                                          | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| barker             | must agree  | converged (1301 steps) | 0 divergences; R-hat high: w[0]                                        | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| hmc                | must agree  | converged (1201 steps) | 0 divergences; R-hat high: w[0], w[2]; ESS low: w[0]                   | a fixed trajectory with a tuned step size: exact up to Monte Carlo error    |
| slice              | must agree  | converged (901 steps)  | 0 divergences                                                          | coordinate-wise slice sampling: exact up to Monte Carlo error               |
| aies               | must agree  | converged (701 steps)  | 0 divergences; R-hat high: w[0], w[1], w[2]; ESS low: w[0], w[1], w[2] | an affine-invariant ensemble: exact up to Monte Carlo error                 |
| ensemble-slice     | must agree  | converged (401 steps)  | 0 divergences; R-hat high: w[0], w[1], w[2]; ESS low: w[0], w[1], w[2] | an ensemble of slice steps: exact up to Monte Carlo error                   |
| chees              | must agree  | converged (601 steps)  | 0 divergences                                                          | jittered HMC with an adapted trajectory: exact up to Monte Carlo error      |
| parallel-tempering | must agree  | converged (901 steps)  | 0 divergences                                                          | the cold replica of a tempered ladder: exact up to Monte Carlo error        |
| mclmc              | reported    | converged (1601 steps) | 0 divergences; R-hat high: w[0]                                        | unadjusted: a small bias that shrinks with the step size                    |
| sgld               | reported    | converged (3201 steps) | 0 divergences; R-hat high: w[0], w[1], w[2]; ESS low: w[0]             | unadjusted Langevin with a constant step: a step-size bias in the spread    |
| sghmc              | reported    | converged (1301 steps) | 0 divergences; R-hat high: w[0], w[2]; ESS low: w[2]                   | friction-damped Hamiltonian dynamics with a constant step: a step-size bias |
| elliptical-slice   | reported    | converged (901 steps)  | 0 divergences; R-hat high: w[2]                                        | a Gaussian-prior sampler, where it applies                                  |

| method             | quantity | mean   | sd    | mean error (sd) | mean z  | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------------ | -------- | ------ | ----- | --------------- | ------- | -------- | ------ | ------- | ------- | ---- |
| rwm                | w[0]     | 0.219  | 0.342 | -0.0336         | -0.882  | 0.991    | 0.0421 | 0.0694  | 0.0179  | 677  |
| rwm                | w[1]     | 1.06   | 0.386 | 0.0719          | 1.74    | 1.04     | 0.0433 | 0.0733  | 0.0103  | 633  |
| rwm                | w[2]     | -0.655 | 0.322 | -0.0483         | -1.22   | 1.02     | 0.034  | 0.0492  | 0.00956 | 670  |
| mala               | w[0]     | 0.236  | 0.336 | 0.0129          | 0.344   | 0.972    | 0.0415 | 0.0568  | 0.0218  | 671  |
| mala               | w[1]     | 1.02   | 0.367 | -0.0468         | -1.15   | 0.985    | 0.0483 | 0.06    | 0.019   | 587  |
| mala               | w[2]     | -0.612 | 0.301 | 0.0893          | 2.28    | 0.955    | 0.0531 | 0.0971  | 0.0371  | 597  |
| barker             | w[0]     | 0.241  | 0.346 | 0.0275          | 0.587   | 1        | 0.029  | 0.0474  | 0.0184  | 455  |
| barker             | w[1]     | 1.06   | 0.36  | 0.074           | 1.73    | 0.967    | 0.0558 | 0.0916  | 0.0308  | 513  |
| barker             | w[2]     | -0.634 | 0.323 | 0.0191          | 0.384   | 1.02     | 0.0517 | 0.0581  | 0.0296  | 424  |
| hmc                | w[0]     | 0.232  | 0.32  | 0.00247         | 0.0198  | 0.926    | 0.0478 | 0.0922  | 0.0225  | 55.2 |
| hmc                | w[1]     | 1.03   | 0.351 | -0.0103         | -0.208  | 0.943    | 0.0304 | 0.0606  | 0.0104  | 365  |
| hmc                | w[2]     | -0.652 | 0.318 | -0.0381         | -0.625  | 1.01     | 0.035  | 0.0447  | 0.0123  | 272  |
| slice              | w[0]     | 0.239  | 0.338 | 0.0226          | 0.821   | 0.979    | 0.0254 | 0.0492  | 0.0104  | 1260 |
| slice              | w[1]     | 1.05   | 0.368 | 0.0325          | 1.2     | 0.988    | 0.0304 | 0.0452  | 0.00509 | 1340 |
| slice              | w[2]     | -0.651 | 0.307 | -0.0358         | -1.39   | 0.974    | 0.024  | 0.0511  | 0.00969 | 1420 |
| aies               | w[0]     | 0.229  | 0.343 | -0.00577        | -0.0591 | 0.993    | 0.039  | 0.0595  | 0.0106  | 104  |
| aies               | w[1]     | 1.09   | 0.404 | 0.145           | 1.18    | 1.09     | 0.0593 | 0.145   | 0.0117  | 79.3 |
| aies               | w[2]     | -0.671 | 0.323 | -0.0975         | -1.14   | 1.02     | 0.0476 | 0.0993  | 0.0081  | 144  |
| ensemble-slice     | w[0]     | 0.241  | 0.348 | 0.0289          | 0.851   | 1.01     | 0.0279 | 0.0411  | 0.00438 | 881  |
| ensemble-slice     | w[1]     | 1.06   | 0.381 | 0.0674          | 2.06    | 1.02     | 0.0339 | 0.0701  | 0.0075  | 978  |
| ensemble-slice     | w[2]     | -0.657 | 0.323 | -0.0531         | -1.55   | 1.02     | 0.0275 | 0.0554  | 0.00472 | 896  |
| chees              | w[0]     | 0.223  | 0.342 | -0.023          | -1.56   | 0.991    | 0.02   | 0.0388  | 0.00416 | 4540 |
| chees              | w[1]     | 1.04   | 0.375 | 0.024           | 1.51    | 1.01     | 0.0197 | 0.0291  | 0.00257 | 3980 |
| chees              | w[2]     | -0.648 | 0.325 | -0.0244         | -1.75   | 1.03     | 0.0173 | 0.0274  | 0.00191 | 5450 |
| parallel-tempering | w[0]     | 0.232  | 0.344 | 0.00224         | 0.0976  | 0.995    | 0.0237 | 0.035   | 0.00662 | 1870 |
| parallel-tempering | w[1]     | 1.04   | 0.376 | 0.0128          | 0.479   | 1.01     | 0.0209 | 0.0357  | 0.00652 | 1430 |
| parallel-tempering | w[2]     | -0.648 | 0.331 | -0.026          | -1.28   | 1.05     | 0.0217 | 0.0431  | 0.00728 | 2690 |
| mclmc              | w[0]     | 0.206  | 0.362 | -0.0738         | -1.48   | 1.05     | 0.0385 | 0.0752  | 0.00562 | 441  |
| mclmc              | w[1]     | 1.04   | 0.378 | 0.0208          | 0.465   | 1.02     | 0.0225 | 0.0317  | 0.00368 | 516  |
| mclmc              | w[2]     | -0.642 | 0.306 | -0.00597        | -0.145  | 0.97     | 0.0282 | 0.0534  | 0.00596 | 551  |
| sgld               | w[0]     | 0.254  | 0.368 | 0.0657          | 0.845   | 1.07     | 0.031  | 0.0667  | 0.00517 | 188  |
| sgld               | w[1]     | 1.06   | 0.39  | 0.0546          | 0.755   | 1.05     | 0.0232 | 0.0611  | 0.00379 | 210  |
| sgld               | w[2]     | -0.606 | 0.322 | 0.107           | 1.75    | 1.02     | 0.0419 | 0.108   | 0.00869 | 277  |
| sghmc              | w[0]     | 0.249  | 0.364 | 0.0521          | 0.785   | 1.05     | 0.0426 | 0.0661  | 0.00757 | 252  |
| sghmc              | w[1]     | 1.06   | 0.413 | 0.0682          | 0.981   | 1.11     | 0.0326 | 0.0886  | 0.0107  | 255  |
| sghmc              | w[2]     | -0.61  | 0.355 | 0.0953          | 1.2     | 1.12     | 0.066  | 0.116   | 0.0202  | 198  |
| elliptical-slice   | w[0]     | 0.234  | 0.338 | 0.00732         | 0.168   | 0.977    | 0.0331 | 0.0524  | 0.015   | 500  |
| elliptical-slice   | w[1]     | 1.03   | 0.4   | -0.0162         | -0.252  | 1.08     | 0.0232 | 0.0611  | 0.0064  | 281  |
| elliptical-slice   | w[2]     | -0.647 | 0.318 | -0.0232         | -0.435  | 1.01     | 0.0234 | 0.0418  | 0.00457 | 359  |

# svi/logistic-regression

Bayesian logistic regression: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 12, scalePercent 100), 14.3 s.

**Passed**: 0 method(s) agree with the quadrature truth; 4 reported

**Bayesian logistic regression** (logistic-regression): glm, classification, logistic, quadrature, three-dimensional

Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 6.0e-9). Log evidence -32.2069.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for w).

| method         | expectation | run                    | diagnostics      | why                                                                                                     |
| -------------- | ----------- | ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| pathfinder     | reported    | converged (9 steps)    | weights ESS 1702 | four L-BFGS paths, importance draws from the mixture of their best normals with Pareto-smoothed weights |
| svi-structured | reported    | completed (1000 steps) | -                | a full-rank guide started at the Laplace approximation: the coefficients are correlated                 |
| svi-natural    | reported    | completed (400 steps)  | -                | natural-gradient steps for a full-rank normal                                                           |
| svi-stl        | reported    | completed (1500 steps) | -                | a mean-field normal with sticking the landing: close in the means, narrow                               |

| method         | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| -------------- | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| pathfinder     | w[0]     | 0.218  | 0.352 | -0.038          | -1.54  | 1.02     | 0.0261 | 0.0527  | 0.0118  | 1700 |
| pathfinder     | w[1]     | 1.03   | 0.366 | -0.011          | -0.461 | 0.983    | 0.0149 | 0.0266  | 0.00366 | 1700 |
| pathfinder     | w[2]     | -0.633 | 0.314 | 0.0217          | 0.901  | 0.994    | 0.0128 | 0.026   | 0.00208 | 1700 |
| svi-structured | w[0]     | 0.225  | 0.327 | -0.0178         | -1.19  | 0.945    | 0.0286 | 0.0712  | 0.00975 | 4000 |
| svi-structured | w[1]     | 1.07   | 0.364 | 0.106           | 6.89   | 0.977    | 0.0654 | 0.126   | 0.0124  | 4000 |
| svi-structured | w[2]     | -0.676 | 0.301 | -0.116          | -7.67  | 0.953    | 0.0658 | 0.131   | 0.0137  | 4000 |
| svi-natural    | w[0]     | 0.241  | 0.348 | 0.0295          | 1.85   | 1.01     | 0.0254 | 0.0371  | 0.00281 | 4000 |
| svi-natural    | w[1]     | 1.04   | 0.356 | 0.00489         | 0.323  | 0.957    | 0.0285 | 0.0578  | 0.00616 | 4000 |
| svi-natural    | w[2]     | -0.634 | 0.28  | 0.0197          | 1.4    | 0.888    | 0.0366 | 0.1     | 0.0187  | 4000 |
| svi-stl        | w[0]     | 0.287  | 0.348 | 0.161           | 10.1   | 1.01     | 0.0662 | 0.162   | 0.0158  | 4000 |
| svi-stl        | w[1]     | 1.06   | 0.37  | 0.0773          | 4.91   | 0.995    | 0.0628 | 0.103   | 0.0114  | 4000 |
| svi-stl        | w[2]     | -0.618 | 0.277 | 0.07            | 5.04   | 0.878    | 0.0527 | 0.112   | 0.0248  | 4000 |

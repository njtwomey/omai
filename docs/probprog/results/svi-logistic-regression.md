# svi/logistic-regression

Bayesian logistic regression: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 547817e0; seed 0, budget scale 1 (plans 4, rows 12, scalePercent 100), 13.3 s.

**Passed**: 0 method(s) agree with the quadrature truth; 4 reported

**Bayesian logistic regression** (logistic-regression): glm, classification, logistic, quadrature, three-dimensional

Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 6.0e-9). Log evidence -32.2069.

Methods whose own `check` says they apply: nuts, svi, ep, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: vmp (the operator dot-constant has no variational message rule); bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)); svi-conjugate-natural (no usable conjugate structure for w).

| method         | expectation | run                    | diagnostics      | why                                                                                                     |
| -------------- | ----------- | ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| pathfinder     | reported    | converged (12 steps)   | weights ESS 1737 | four L-BFGS paths, importance draws from the mixture of their best normals with Pareto-smoothed weights |
| svi-structured | reported    | completed (1000 steps) | -                | a full-rank guide started at the Laplace approximation: the coefficients are correlated                 |
| svi-natural    | reported    | completed (400 steps)  | -                | natural-gradient steps for a full-rank normal                                                           |
| svi-stl        | reported    | completed (1500 steps) | -                | a mean-field normal with sticking the landing: close in the means, narrow                               |

| method         | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| -------------- | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| pathfinder     | w[0]     | 0.223  | 0.337 | -0.024          | -1.03  | 0.974    | 0.0257 | 0.0469  | 0.00842 | 1740 |
| pathfinder     | w[1]     | 1.03   | 0.378 | -0.0107         | -0.437 | 1.02     | 0.0124 | 0.0191  | 0.00473 | 1740 |
| pathfinder     | w[2]     | -0.635 | 0.292 | 0.017           | 0.767  | 0.924    | 0.0274 | 0.0668  | 0.00978 | 1740 |
| svi-structured | w[0]     | 0.264  | 0.342 | 0.0958          | 6.12   | 0.99     | 0.0524 | 0.102   | 0.00932 | 4000 |
| svi-structured | w[1]     | 1.03   | 0.366 | -0.0167         | -1.07  | 0.984    | 0.0215 | 0.0522  | 0.00694 | 4000 |
| svi-structured | w[2]     | -0.66  | 0.286 | -0.0626         | -4.37  | 0.906    | 0.0556 | 0.112   | 0.0183  | 4000 |
| svi-natural    | w[0]     | 0.22   | 0.333 | -0.0331         | -2.17  | 0.965    | 0.0268 | 0.0616  | 0.00774 | 4000 |
| svi-natural    | w[1]     | 1.02   | 0.374 | -0.0524         | -3.3   | 1        | 0.0196 | 0.0522  | 0.00511 | 4000 |
| svi-natural    | w[2]     | -0.633 | 0.32  | 0.0208          | 1.3    | 1.01     | 0.0174 | 0.038   | 0.00445 | 4000 |
| svi-stl        | w[0]     | 0.28   | 0.282 | 0.141           | 10.9   | 0.816    | 0.104  | 0.206   | 0.0685  | 4000 |
| svi-stl        | w[1]     | 1.03   | 0.386 | -0.0105         | -0.641 | 1.04     | 0.0243 | 0.0486  | 0.00747 | 4000 |
| svi-stl        | w[2]     | -0.697 | 0.266 | -0.179          | -13.5  | 0.841    | 0.121  | 0.219   | 0.0632  | 4000 |

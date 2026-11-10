# benchmark/banana

Haario's banana: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 3, rows 6, scalePercent 100), 2.0 s.

**Passed**: 0 method(s) agree with the analytic truth, 2 differ as expected; 1 reported

**Haario's banana** (banana): geometry, curved, density, analytic, two-dimensional, correlated

A curved two-dimensional ridge: x0 ~ N(0, 4), x1 | x0 ~ N(0.5 (x0^2 - 4), 1). Marginals are known; the shape is far from Gaussian.

Truth (analytic): x0 is N(0, 4) exactly; x1 has mean 0 and variance 2 b² a⁴ + 1 = 9, its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply to the compiled program (a target with no observed node is refused at the spec level): nuts, svi, laplace, importance, tempered-smc. Not applicable: gibbs (no discrete or conjugate conditionals (use NUTS)); em (no params to estimate); particle-filter (no single markov loop); abc-smc (no observed sites); conjugate (no conjugate structure for x0, x1); gaussian (the location of x1 is not affine in its parents); enumeration (continuous latent sites (x0, x1)); chain (the program needs exactly one markov loop); abc-rejection (no observed sites).

| method  | expectation        | run                    | diagnostics                                          | why                                                                                                                                                     |
| ------- | ------------------ | ---------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts    | reported           | converged (2301 steps) | 168 divergences; R-hat high: x0, x1; ESS low: x0, x1 | follows the ridge but with divergences in the arms and an effective sample size of a few percent of the draws: the spread of x1 is off by about a fifth |
| laplace | expected to differ | converged (10 steps)   | -                                                    | a normal at the mode (0, -2): a straight ellipse for a curved ridge, wrong in the spread of x1 and in its skew                                          |
| svi     | expected to differ | completed (1500 steps) | -                                                    | a mean-field normal: no dependence between x0 and x1, so it cannot represent the ridge                                                                  |

| method  | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts    | x0       | 0.83   | 2.62  | 0.415           | 0.92   | 1.31     | 0.166  | 0.415   | 0.102   | 8.41 |
| nuts    | x1       | 1.83   | 4.99  | 0.61            | 0.931  | 1.66     | 0.156  | 0.621   | 0.0981  | 6.44 |
| laplace | x0       | 0.0326 | 1.99  | 0.0163          | 1.04   | 0.997    | 0.0161 | 0.0207  | 0.00253 | 4000 |
| laplace | x1       | -1.99  | 0.996 | -0.664          | -126   | 0.332    | 0.389  | 0.658   | 1.23    | 4000 |
| svi     | x0       | 0.0828 | 0.953 | 0.0414          | 5.5    | 0.476    | 0.182  | 0.419   | 0.762   | 4000 |
| svi     | x1       | -1.71  | 1.07  | -0.569          | -100   | 0.358    | 0.31   | 0.564   | 1.06    | 4000 |

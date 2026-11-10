# benchmark/gaussian-mixture-ordered

Gaussian mixture, ordered means: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 3, rows 6, scalePercent 100), 6.9 s.

**Passed**: 1 method(s) agree with the quadrature truth, 1 differ as expected; 1 reported

**Gaussian mixture, ordered means** (gaussian-mixture-ordered): mixture, discrete-latent, enumeration, ordered, quadrature, two-dimensional

Two normal components with known weights and unit noise, unknown means constrained mu_0 < mu_1 and a discrete label per row: the mixture without label switching.

Truth (quadrature): grid quadrature of the two means around one mode, the component labels summed out exactly (121 points per axis, evidence error 3.2e-12); that mode is the ordered posterior and the unconstrained one is its even mixture with the relabelling. Log evidence -87.8202.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, abc-rejection. Not applicable: em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for z); gaussian ('z' is categorical, not Gaussian); enumeration (continuous latent nodes (mu)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method  | expectation        | run                    | diagnostics   | why                                                                                                                                                   |
| ------- | ------------------ | ---------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts    | must agree         | converged (2001 steps) | 0 divergences | the labels are summed out and the ordering removes the second mode: agrees with the truth                                                             |
| gibbs   | expected to differ | completed (1800 steps) | -             | its draws do not respect the ordered constraint (mu[0] ends above mu[1]): the Gibbs method samples the means from their conditionals on the real line |
| laplace | reported           | converged (14 steps)   | -             | a normal at the mode of the ordered density: close, the labels summed out                                                                             |

| method  | quantity | mean  | sd    | mean error (sd) | mean z  | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ----- | --------------- | ------- | -------- | ------ | ------- | ------- | ---- |
| nuts    | mu[0]    | -1.88 | 0.29  | -0.0255         | -0.456  | 0.963    | 0.0464 | 0.0669  | 0.0126  | 296  |
| nuts    | mu[1]    | 2.32  | 0.227 | 0.0267          | 0.902   | 1.01     | 0.0268 | 0.0351  | 0.00928 | 1170 |
| gibbs   | mu[0]    | 2.32  | 0.227 | 13.9            | 565     | 0.757    | 1      | 13.9    | 4.62    | 940  |
| gibbs   | mu[1]    | -1.87 | 0.306 | -18.7           | -354    | 1.37     | 1      | 18.7    | 4.62    | 670  |
| laplace | mu[0]    | -1.87 | 0.304 | -0.000366       | -0.0229 | 1.01     | 0.0104 | 0.0168  | 0.00221 | 4000 |
| laplace | mu[1]    | 2.34  | 0.229 | 0.0935          | 5.79    | 1.02     | 0.0355 | 0.094   | 0.00668 | 4000 |

Notes:
- svi did not run: plate row: site y has batch shape [40, 2], which does not fit plates of shape [40]

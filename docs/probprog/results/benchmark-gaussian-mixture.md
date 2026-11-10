# benchmark/gaussian-mixture

Gaussian mixture, label switching: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 8, scalePercent 100), 7.6 s.

**Passed**: 1 method(s) agree with the quadrature truth; 3 reported

**Gaussian mixture, label switching** (gaussian-mixture): mixture, discrete-latent, multimodal, label-switching, quadrature, two-dimensional

The two-component mixture with exchangeable means: a bimodal posterior (the relabelling), a multimodal target for the raw means and a unimodal one for the sorted means.

Truth (quadrature): grid quadrature of the two means around one mode, the component labels summed out exactly (121 points per axis, evidence error 3.2e-12); that mode is the ordered posterior and the unconstrained one is its even mixture with the relabelling. Log evidence -87.8202.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, gibbs, laplace, importance, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of z is categorical, not normal or multivariateNormal); vmp (the operator select has no variational message rule); bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for z); gaussian ('z' is categorical, not Gaussian); enumeration (continuous latent nodes (mu)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for z).

| method  | expectation | run                    | diagnostics                                                    | why                                                                                                                                          |
| ------- | ----------- | ---------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts    | must agree  | converged (2001 steps) | 0 divergences; R-hat high: mu[0], mu[1]; ESS low: mu[0], mu[1] | the sorted means are invariant to the relabelling: every chain agrees on them, whichever mode it started in                                  |
| nuts    | reported    | converged (2001 steps) | 0 divergences; R-hat high: mu[0], mu[1]; ESS low: mu[0], mu[1] | the raw means: the truth is bimodal and chains are split between the modes (or all in one), so the draws are not the truth and R-hat is poor |
| gibbs   | reported    | completed (1800 steps) | -                                                              | sorted means: agrees when the sweep does not stick                                                                                           |
| laplace | reported    | converged (11 steps)   | -                                                              | a normal at one mode: the sorted means are right, the raw means are one mode only                                                            |

| method  | quantity | mean  | sd    | mean error (sd) | mean z   | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ----- | --------------- | -------- | -------- | ------ | ------- | ------- | ---- |
| nuts    | mu(1)    | -1.86 | 0.298 | 0.0288          | 0.917    | 0.992    | 0.0368 | 0.0444  | 0.0155  | 1000 |
| nuts    | mu(2)    | 2.33  | 0.229 | 0.0319          | 0.989    | 1.02     | 0.022  | 0.0397  | 0.0111  | 1000 |
| nuts    | mu[0]    | 0.214 | 2.1   | -0.00424        | -0.0073  | 0.996    | 0.0268 | 0.00674 | 0.0103  | 2.94 |
| nuts    | mu[1]    | 0.221 | 2.12  | -0.000869       | -0.00149 | 1        | 0.0158 | 0.0053  | 0.00919 | 2.96 |
| gibbs   | mu(1)    | -1.87 | 0.303 | 0.00831         | 0.32     | 1.01     | 0.0174 | 0.0233  | 0.0036  | 1500 |
| gibbs   | mu(2)    | 2.31  | 0.223 | -0.0246         | -0.958   | 0.994    | 0.0248 | 0.0372  | 0.00661 | 1500 |
| laplace | mu(1)    | -1.85 | 0.299 | 0.0649          | 4.13     | 0.993    | 0.0261 | 0.065   | 0.00507 | 4000 |
| laplace | mu(2)    | 2.32  | 0.227 | 0.00663         | 0.414    | 1.01     | 0.0114 | 0.0181  | 0.00254 | 4000 |

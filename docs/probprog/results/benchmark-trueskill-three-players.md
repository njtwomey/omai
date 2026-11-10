# benchmark/trueskill-three-players

TrueSkill, three players: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 5, rows 15, scalePercent 100), 25.3 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**TrueSkill, three players** (trueskill-three-players): ranking, trueskill, probit, factor-graph, ep, quadrature, three-dimensional

Three players with N(0, 1) skills and 24 pairwise games won with probability Phi((s_a - s_b) / (sqrt 2 beta)): the TrueSkill performance model with a grid truth, the target of expectation propagation.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 1.6e-11). Log evidence -8.9325.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'skill' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for skill); gaussian ('win' is bernoulli, not Gaussian); enumeration (continuous latent nodes (skill)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for skill).

| method       | expectation | run                    | diagnostics     | why                                                                                            |
| ------------ | ----------- | ---------------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (1001 steps) | 0 divergences   | a consistent sampler: agrees within Monte Carlo error                                          |
| tempered-smc | must agree  | converged (3 steps)    | weights ESS 535 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS                  |
| importance   | reported    | completed (10 steps)   | weights ESS 840 | weights from the prior: a concentrated posterior leaves few effective draws                    |
| laplace      | reported    | converged (14 steps)   | -               | near Gaussian: close to the grid, a little narrow                                              |
| svi          | reported    | completed (1500 steps) | -               | a mean-field normal ignores the negative correlation between the skills of two players who met |

| method       | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | skill[0] | 1.57   | 0.715 | -0.0175         | -0.383 | 0.971    | 0.0352 | 0.0543  | 0.0151  | 453  |
| nuts         | skill[1] | -0.438 | 0.67  | 0.00225         | 0.049  | 0.994    | 0.0219 | 0.0376  | 0.0102  | 467  |
| nuts         | skill[2] | -1.15  | 0.688 | -0.00618        | -0.132 | 0.997    | 0.0201 | 0.0245  | 0.00781 | 453  |
| tempered-smc | skill[0] | 1.59   | 0.778 | 0.0142          | 0.425  | 1.06     | 0.0249 | 0.0452  | 0.00999 | 1000 |
| tempered-smc | skill[1] | -0.459 | 0.689 | -0.0297         | -0.918 | 1.02     | 0.0298 | 0.0342  | 0.0136  | 1000 |
| tempered-smc | skill[2] | -1.12  | 0.689 | 0.0303          | 0.96   | 0.998    | 0.0241 | 0.0396  | 0.011   | 1000 |
| importance   | skill[0] | 1.56   | 0.69  | -0.0275         | -0.85  | 0.938    | 0.0356 | 0.0682  | 0.0161  | 840  |
| importance   | skill[1] | -0.438 | 0.7   | 0.00223         | 0.0621 | 1.04     | 0.028  | 0.0343  | 0.0118  | 840  |
| importance   | skill[2] | -1.14  | 0.693 | 0.00364         | 0.105  | 1        | 0.0163 | 0.0182  | 0.00658 | 840  |
| laplace      | skill[0] | 1.47   | 0.726 | -0.153          | -9.84  | 0.987    | 0.0666 | 0.153   | 0.0147  | 4000 |
| laplace      | skill[1] | -0.402 | 0.666 | 0.0546          | 3.49   | 0.987    | 0.0407 | 0.0549  | 0.00488 | 4000 |
| laplace      | skill[2] | -1.08  | 0.678 | 0.0927          | 5.96   | 0.983    | 0.0468 | 0.094   | 0.00702 | 4000 |
| svi          | skill[0] | 1.57   | 0.657 | -0.0149         | -1.06  | 0.893    | 0.0368 | 0.0986  | 0.0169  | 4000 |
| svi          | skill[1] | -0.407 | 0.522 | 0.0468          | 3.83   | 0.774    | 0.0816 | 0.186   | 0.0769  | 4000 |
| svi          | skill[2] | -1.1   | 0.537 | 0.0632          | 5.14   | 0.778    | 0.0845 | 0.192   | 0.074   | 4000 |

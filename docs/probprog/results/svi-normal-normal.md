# svi/normal-normal

Normal-normal mean: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 6, rows 6, scalePercent 100), 1.6 s.

**Passed**: 6 method(s) agree with the exact truth; 0 reported

**Normal-normal mean** (normal-normal): conjugate, exact, gaussian, linear-gaussian, one-dimensional

The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.

Truth (exact): the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts). Log evidence -23.9485.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, gaussian, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); particle-filter (the model has no single scan); enumeration (continuous latent nodes (mu)); chain (the model has no single scan).

| method                | expectation | run                    | diagnostics      | why                                                                                                                                                               |
| --------------------- | ----------- | ---------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| svi-stl               | must agree  | completed (1500 steps) | -                | a mean-field normal is the right family; sticking the landing quiets the gradient near the optimum                                                                |
| svi-trace-graph       | must agree  | completed (1500 steps) | -                | no discrete or unreparameterised site: the estimator reduces to the plain ELBO (more particles and a smaller step than sticking the landing needs, for its noise) |
| svi-natural           | must agree  | completed (300 steps)  | -                | natural-gradient steps in the mean parameters: the Gaussian guide reaches the conjugate posterior in tens of steps                                                |
| svi-conjugate-natural | must agree  | completed (1 steps)    | -                | batch CAVI: one step of rate one is the exact posterior                                                                                                           |
| svi-structured        | must agree  | completed (1500 steps) | -                | a one-dimensional structured guide is a normal                                                                                                                    |
| pathfinder            | must agree  | converged (3 steps)    | weights ESS 2000 | the posterior is normal and the first Newton step finds it; the importance weights are uniform                                                                    |

| method                | quantity | mean | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| --------------------- | -------- | ---- | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| svi-stl               | mu       | 1.4  | 0.429 | 0.0101          | 0.628  | 1.01     | 0.0106 | 0.0183  | 0.00218 | 4000 |
| svi-trace-graph       | mu       | 1.41 | 0.442 | 0.0225          | 1.36   | 1.04     | 0.0183 | 0.0439  | 0.00297 | 4000 |
| svi-natural           | mu       | 1.39 | 0.425 | -0.0149         | -0.939 | 1        | 0.0185 | 0.0213  | 0.00157 | 4000 |
| svi-conjugate-natural | mu       | 1.4  | 0.422 | 0.00805         | 0.511  | 0.997    | 0.015  | 0.0178  | 0.00268 | 4000 |
| svi-structured        | mu       | 1.4  | 0.428 | 0.00851         | 0.532  | 1.01     | 0.0124 | 0.0164  | 0.00157 | 4000 |
| pathfinder            | mu       | 1.39 | 0.421 | -0.0272         | -1.22  | 0.994    | 0.0204 | 0.0322  | 0.00348 | 2000 |

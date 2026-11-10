# svi/normal-normal

Normal-normal mean: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 547817e0; seed 0, budget scale 1 (plans 6, rows 6, scalePercent 100), 1.5 s.

**Passed**: 6 method(s) agree with the exact truth; 0 reported

**Normal-normal mean** (normal-normal): conjugate, exact, gaussian, linear-gaussian, one-dimensional

The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.

Truth (exact): the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts). Log evidence -23.9485.

Methods whose own `check` says they apply: nuts, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, gaussian, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); enumeration (continuous latent nodes (mu)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

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
| svi-stl               | mu       | 1.4  | 0.419 | 0.00934         | 0.597  | 0.99     | 0.0109 | 0.0178  | 0.00165 | 4000 |
| svi-trace-graph       | mu       | 1.42 | 0.382 | 0.0542          | 3.8    | 0.903    | 0.0455 | 0.0872  | 0.0126  | 4000 |
| svi-natural           | mu       | 1.39 | 0.398 | -0.0343         | -2.31  | 0.94     | 0.0305 | 0.0539  | 0.00529 | 4000 |
| svi-conjugate-natural | mu       | 1.4  | 0.422 | 0.0113          | 0.717  | 0.997    | 0.0121 | 0.018   | 0.00205 | 4000 |
| svi-structured        | mu       | 1.41 | 0.433 | 0.0304          | 1.88   | 1.02     | 0.021  | 0.0377  | 0.00353 | 4000 |
| pathfinder            | mu       | 1.41 | 0.419 | 0.0286          | 1.29   | 0.991    | 0.024  | 0.0301  | 0.00458 | 2000 |

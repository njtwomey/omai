# svi/gamma-poisson

Gamma-Poisson rate: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 547817e0; seed 0, budget scale 1 (plans 3, rows 3, scalePercent 100), 0.4 s.

**Passed**: 2 method(s) agree with the exact truth; 1 reported

**Gamma-Poisson rate** (gamma-poisson): conjugate, exact, counts, one-dimensional, positive

The rate of Poisson counts under a Gamma(2, 0.5) prior: conjugate, a Gamma posterior on the positive reals.

Truth (exact): the Gamma(a + sum k, b + n) posterior and its closed-form evidence (probprog/validation/exact.ts). Log evidence -30.5610.

Methods whose own `check` says they apply: nuts, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'rate' is continuous); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); gaussian ('rate' is gamma, not Gaussian); enumeration (continuous latent nodes (rate)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method                | expectation | run                    | diagnostics      | why                                                                                                   |
| --------------------- | ----------- | ---------------------- | ---------------- | ----------------------------------------------------------------------------------------------------- |
| svi-conjugate-natural | must agree  | completed (1 steps)    | -                | batch CAVI: the Gamma posterior in closed form                                                        |
| pathfinder            | must agree  | converged (6 steps)    | weights ESS 1983 | a normal on the log rate, corrected by Pareto-smoothed importance weights towards the Gamma posterior |
| svi-stl               | reported    | completed (1500 steps) | -                | a normal on the log rate: the right location and spread, the skew of the Gamma is not representable   |

| method                | quantity | mean | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| --------------------- | -------- | ---- | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| svi-conjugate-natural | rate     | 3.22 | 0.453 | -0.0202         | -1.29  | 0.993    | 0.0137 | 0.0233  | 0.00261 | 4000 |
| pathfinder            | rate     | 3.22 | 0.453 | -0.0116         | -0.519 | 0.993    | 0.016  | 0.0191  | 0.00507 | 1980 |
| svi-stl               | rate     | 3.26 | 0.491 | 0.0848          | 4.98   | 1.08     | 0.0359 | 0.0853  | 0.00664 | 4000 |

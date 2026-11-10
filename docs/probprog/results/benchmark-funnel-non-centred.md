# benchmark/funnel-non-centred

Neal's funnel, non-centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 2.4 s.

**Passed**: 1 method(s) agree with the analytic truth; 1 reported

**Neal's funnel, non-centred** (funnel-non-centred): funnel, geometry, non-centred, reparametrisation, density, analytic

Neal's funnel with x_i = e^(v/2) z_i and z_i standard normal: the same law as the centred funnel, with no funnel in the density.

Truth (analytic): v is N(0, 9) exactly; x_i | v ~ N(0, e^v) is a scale mixture with mean 0 and sd exp(2.25), its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, laplace, importance, tempered-smc, gaussian, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep (x: operator 'exp' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); vmp (x: operator 'exp' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); bp (belief propagation sums over finite discrete variables, and 'v', 'z' are continuous); gibbs (no discrete or conjugate conditionals); em (observed nodes have no likelihood); particle-filter (the model has no single scan); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); enumeration (observed nodes have no likelihood); chain (the model has no single scan); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method | expectation | run                    | diagnostics                     | why                                                                      |
| ------ | ----------- | ---------------------- | ------------------------------- | ------------------------------------------------------------------------ |
| nuts   | must agree  | converged (1001 steps) | 0 divergences; R-hat high: z[1] | no funnel in the density: no divergences, agrees with the truth          |
| svi    | reported    | completed (1500 steps) | -                               | v and z are independent normals, the mean-field family is exact for them |

| method | quantity | mean    | sd   | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------ | -------- | ------- | ---- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts   | v        | -0.0667 | 2.86 | -0.0222         | -1.04  | 0.953    | 0.0318 | 0.0482  | 0.0112  | 2000 |
| nuts   | x[0]     | 0.0487  | 10.9 | 0.00513         | 0.14   | 1.15     | 0.0231 | 0.0774  | 0.00676 | 982  |
| svi    | v        | -0.221  | 3.19 | -0.0736         | -4.38  | 1.06     | 0.0398 | 0.0823  | 0.00678 | 4000 |

# benchmark/funnel-non-centred

Neal's funnel, non-centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 2.6 s.

**Passed**: 1 method(s) agree with the analytic truth; 1 reported

**Neal's funnel, non-centred** (funnel-non-centred): funnel, geometry, non-centred, reparametrisation, density, analytic

Neal's funnel with x_i = e^(v/2) z_i and z_i standard normal: the same law as the centred funnel, with no funnel in the density.

Truth (analytic): v is N(0, 9) exactly; x_i | v ~ N(0, e^v) is a scale mixture with mean 0 and sd exp(2.25), its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply to the compiled program (a target with no observed node is refused at the spec level): nuts, svi, laplace, importance, tempered-smc, gaussian. Not applicable: gibbs (no discrete or conjugate conditionals (use NUTS)); em (no params to estimate); particle-filter (no single markov loop); abc-smc (no observed sites); conjugate (no conjugate structure for v, z); enumeration (continuous latent sites (v, z)); chain (the program needs exactly one markov loop); abc-rejection (no observed sites).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (1001 steps) | 0 divergences; R-hat high: z[1] | no funnel in the density: no divergences, agrees with the truth |
| svi | reported | completed (1500 steps) | - | v and z are independent normals, the mean-field family is exact for them |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | v | -0.0667 | 2.86 | -0.0222 | -1.04 | 0.953 | 0.0318 | 0.0482 | 0.0112 | 2000 |
| nuts | x[0] | 0.0487 | 10.9 | 0.00513 | 0.14 | 1.15 | 0.0231 | 0.0774 | 0.00676 | 982 |
| svi | v | -0.221 | 3.19 | -0.0736 | -4.38 | 1.06 | 0.0398 | 0.0823 | 0.00678 | 4000 |

# benchmark/gaussian-mixture-ordered

Gaussian mixture, ordered means: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (plans 4, rows 8, scalePercent 100), 8.6 s.

**Passed**: 2 method(s) agree with the quadrature truth; 2 reported

**Gaussian mixture, ordered means** (gaussian-mixture-ordered): mixture, discrete-latent, enumeration, ordered, quadrature, two-dimensional

Two normal components with known weights and unit noise, unknown means constrained mu_0 < mu_1 and a discrete label per row: the mixture without label switching.

Truth (quadrature): grid quadrature of the two means around one mode, the component labels summed out exactly (121 points per axis, evidence error 3.2e-12); that mode is the ordered posterior and the unconstrained one is its even mixture with the relabelling. Log evidence -87.8202.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, ep-multistart, gibbs, laplace, importance, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of z is categorical, not normal or multivariateNormal); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); vmp (the operator greater-than has no variational message rule (greater-than: a hard step is zero off the half-line, so KL(q || p) is infinite for every Gaussian q (the variational rule needs a step with noise: the probit link, or expectation propagation))); bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for z); gaussian ('z' is categorical, not Gaussian); enumeration (continuous latent nodes (mu)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for z).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (2001 steps) | 0 divergences | the labels are summed out and the ordering removes the second mode: agrees with the truth |
| gibbs | must agree | completed (1800 steps) | - | the means have no closed-form conditional under the ordering, so they are updated by slice steps on the ordered scale (the labels are exact): agrees with the truth |
| laplace | reported | converged (14 steps) | - | a normal at the mode of the ordered density: close, the labels summed out |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the unconstrained coordinates (the first mean and the log gap): close in the means, but mu[0] too narrow and mu[1], their sum, too wide |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | mu[0] | -1.88 | 0.29 | -0.0255 | -0.456 | 0.963 | 0.0464 | 0.0669 | 0.0126 | 296 |
| nuts | mu[1] | 2.32 | 0.227 | 0.0267 | 0.902 | 1.01 | 0.0268 | 0.0351 | 0.00928 | 1170 |
| gibbs | mu[0] | -1.87 | 0.301 | 0.0189 | 0.322 | 1 | 0.0244 | 0.0326 | 0.00484 | 290 |
| gibbs | mu[1] | 2.31 | 0.23 | -0.0156 | -0.504 | 1.02 | 0.0151 | 0.0269 | 0.00533 | 1100 |
| laplace | mu[0] | -1.87 | 0.304 | -0.000366 | -0.0229 | 1.01 | 0.0104 | 0.0168 | 0.00221 | 4000 |
| laplace | mu[1] | 2.34 | 0.229 | 0.0935 | 5.79 | 1.02 | 0.0355 | 0.094 | 0.00668 | 4000 |
| svi | mu[0] | -1.86 | 0.162 | 0.0517 | 6.06 | 0.539 | 0.157 | 0.368 | 0.489 | 4000 |
| svi | mu[1] | 2.24 | 0.258 | -0.364 | -20 | 1.15 | 0.152 | 0.365 | 0.0707 | 4000 |

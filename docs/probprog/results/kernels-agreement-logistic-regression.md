# kernels/agreement-logistic-regression

Bayesian logistic regression: the kernel methods against the quadrature truth.

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 1 (plans 13, rows 39, scalePercent 100), 37.4 s.

**Passed**: 9 kernel method(s) agree with the quadrature truth; 4 reported

**Bayesian logistic regression** (logistic-regression): glm, classification, logistic, quadrature, three-dimensional

Logistic regression with an intercept and two covariates and a N(0, 4) prior on the weights: a smooth three-dimensional posterior with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (286563 density evaluations, evidence error 6.0e-9). Log evidence -32.2069.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: bp (belief propagation sums over finite discrete variables, and 'w' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for w); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (w)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method             | expectation | run                    | diagnostics                                                            | why                                                                         |
| ------------------ | ----------- | ---------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| rwm                | must agree  | converged (1901 steps) | 0 divergences; R-hat high: w[0], w[1]; ESS low: w[0]                   | a consistent sampler, slower mixing: agrees within Monte Carlo error        |
| mala               | must agree  | converged (1101 steps) | 0 divergences                                                          | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| barker             | must agree  | converged (1301 steps) | 0 divergences; R-hat high: w[1]                                        | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| hmc                | must agree  | converged (1201 steps) | 0 divergences                                                          | a fixed trajectory with a tuned step size: exact up to Monte Carlo error    |
| slice              | must agree  | converged (901 steps)  | 0 divergences                                                          | coordinate-wise slice sampling: exact up to Monte Carlo error               |
| aies               | must agree  | converged (701 steps)  | 0 divergences; R-hat high: w[0], w[1], w[2]; ESS low: w[0], w[1], w[2] | an affine-invariant ensemble: exact up to Monte Carlo error                 |
| ensemble-slice     | must agree  | converged (401 steps)  | 0 divergences; R-hat high: w[0], w[1], w[2]; ESS low: w[0], w[1], w[2] | an ensemble of slice steps: exact up to Monte Carlo error                   |
| chees              | must agree  | converged (601 steps)  | 0 divergences                                                          | jittered HMC with an adapted trajectory: exact up to Monte Carlo error      |
| parallel-tempering | must agree  | converged (901 steps)  | 0 divergences                                                          | the cold replica of a tempered ladder: exact up to Monte Carlo error        |
| mclmc              | reported    | converged (1601 steps) | 0 divergences                                                          | unadjusted: a small bias that shrinks with the step size                    |
| sgld               | reported    | converged (3201 steps) | 0 divergences; R-hat high: w[0]                                        | unadjusted Langevin with a constant step: a step-size bias in the spread    |
| sghmc              | reported    | converged (1301 steps) | 0 divergences; R-hat high: w[1], w[2]                                  | friction-damped Hamiltonian dynamics with a constant step: a step-size bias |
| elliptical-slice   | reported    | converged (901 steps)  | 0 divergences                                                          | a Gaussian-prior sampler, where it applies                                  |

| method             | quantity | mean   | sd    | mean error (sd) | mean z  | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------------ | -------- | ------ | ----- | --------------- | ------- | -------- | ------ | ------- | ------- | ---- |
| rwm                | w[0]     | 0.272  | 0.311 | 0.117           | 1.7     | 0.899    | 0.0848 | 0.141   | 0.0535  | 170  |
| rwm                | w[1]     | 1.05   | 0.364 | 0.0294          | 0.47    | 0.977    | 0.0502 | 0.0618  | 0.056   | 243  |
| rwm                | w[2]     | -0.65  | 0.32  | -0.031          | -0.498  | 1.01     | 0.0625 | 0.0645  | 0.0595  | 265  |
| mala               | w[0]     | 0.263  | 0.349 | 0.0921          | 2.24    | 1.01     | 0.0643 | 0.096   | 0.0274  | 604  |
| mala               | w[1]     | 1.03   | 0.359 | -0.0263         | -0.705  | 0.965    | 0.0343 | 0.0549  | 0.0139  | 667  |
| mala               | w[2]     | -0.622 | 0.321 | 0.0567          | 1.54    | 1.02     | 0.0384 | 0.059   | 0.0172  | 757  |
| barker             | w[0]     | 0.219  | 0.318 | -0.0342         | -0.635  | 0.921    | 0.0979 | 0.129   | 0.0913  | 292  |
| barker             | w[1]     | 1.01   | 0.372 | -0.057          | -0.901  | 0.999    | 0.0813 | 0.095   | 0.052   | 249  |
| barker             | w[2]     | -0.653 | 0.303 | -0.0397         | -0.753  | 0.961    | 0.041  | 0.0607  | 0.0401  | 332  |
| hmc                | w[0]     | 0.227  | 0.348 | -0.0107         | -0.74   | 1.01     | 0.0229 | 0.0344  | 0.0119  | 4820 |
| hmc                | w[1]     | 1.02   | 0.373 | -0.0293         | -1.15   | 1        | 0.0209 | 0.0348  | 0.00708 | 1560 |
| hmc                | w[2]     | -0.633 | 0.314 | 0.0234          | 1.42    | 0.993    | 0.0233 | 0.03    | 0.00509 | 3640 |
| slice              | w[0]     | 0.237  | 0.353 | 0.0163          | 0.597   | 1.02     | 0.0217 | 0.0284  | 0.00924 | 1400 |
| slice              | w[1]     | 1.04   | 0.375 | 0.014           | 0.508   | 1.01     | 0.0183 | 0.024   | 0.00347 | 1330 |
| slice              | w[2]     | -0.633 | 0.311 | 0.0223          | 0.896   | 0.985    | 0.0243 | 0.0341  | 0.00711 | 1560 |
| aies               | w[0]     | 0.252  | 0.342 | 0.0617          | 0.678   | 0.991    | 0.0348 | 0.072   | 0.0104  | 118  |
| aies               | w[1]     | 1.03   | 0.372 | -0.0136         | -0.193  | 1        | 0.02   | 0.0245  | 0.00402 | 201  |
| aies               | w[2]     | -0.642 | 0.321 | -0.00786        | -0.0988 | 1.02     | 0.017  | 0.0304  | 0.00578 | 163  |
| ensemble-slice     | w[0]     | 0.221  | 0.345 | -0.0296         | -0.84   | 0.997    | 0.0185 | 0.0376  | 0.00327 | 801  |
| ensemble-slice     | w[1]     | 1.05   | 0.361 | 0.042           | 1.26    | 0.97     | 0.0365 | 0.0577  | 0.00778 | 848  |
| ensemble-slice     | w[2]     | -0.63  | 0.316 | 0.0309          | 0.888   | 1        | 0.0268 | 0.0488  | 0.00448 | 828  |
| chees              | w[0]     | 0.225  | 0.352 | -0.0165         | -1.1    | 1.02     | 0.0141 | 0.0238  | 0.00213 | 4610 |
| chees              | w[1]     | 1.04   | 0.376 | 0.00819         | 0.499   | 1.01     | 0.0129 | 0.0195  | 0.00418 | 3790 |
| chees              | w[2]     | -0.641 | 0.312 | -0.00298        | -0.222  | 0.988    | 0.014  | 0.0305  | 0.00398 | 5400 |
| parallel-tempering | w[0]     | 0.221  | 0.353 | -0.0295         | -1.75   | 1.02     | 0.023  | 0.0363  | 0.0104  | 3700 |
| parallel-tempering | w[1]     | 1.04   | 0.368 | 0.00787         | 0.406   | 0.989    | 0.0194 | 0.0336  | 0.00489 | 2600 |
| parallel-tempering | w[2]     | -0.641 | 0.295 | -0.00303        | -0.183  | 0.933    | 0.0281 | 0.0733  | 0.0205  | 3180 |
| mclmc              | w[0]     | 0.244  | 0.342 | 0.0371          | 1.09    | 0.991    | 0.0204 | 0.0423  | 0.00548 | 846  |
| mclmc              | w[1]     | 1.02   | 0.368 | -0.0395         | -1.16   | 0.988    | 0.0233 | 0.0481  | 0.00462 | 840  |
| mclmc              | w[2]     | -0.661 | 0.315 | -0.0657         | -1.8    | 0.999    | 0.0339 | 0.0666  | 0.00844 | 753  |
| sgld               | w[0]     | 0.238  | 0.338 | 0.0195          | 0.346   | 0.977    | 0.0272 | 0.0486  | 0.0046  | 299  |
| sgld               | w[1]     | 1.01   | 0.379 | -0.0658         | -0.924  | 1.02     | 0.0294 | 0.0657  | 0.00465 | 205  |
| sgld               | w[2]     | -0.607 | 0.332 | 0.104           | 1.94    | 1.05     | 0.042  | 0.105   | 0.00763 | 384  |
| sghmc              | w[0]     | 0.239  | 0.369 | 0.0232          | 0.324   | 1.07     | 0.0235 | 0.0352  | 0.00697 | 223  |
| sghmc              | w[1]     | 1.07   | 0.39  | 0.0815          | 1.26    | 1.05     | 0.0441 | 0.0855  | 0.00908 | 263  |
| sghmc              | w[2]     | -0.671 | 0.338 | -0.0988         | -1.61   | 1.07     | 0.0493 | 0.102   | 0.0108  | 303  |
| elliptical-slice   | w[0]     | 0.236  | 0.328 | 0.0132          | 0.247   | 0.95     | 0.0281 | 0.0682  | 0.0115  | 313  |
| elliptical-slice   | w[1]     | 1.01   | 0.349 | -0.0663         | -1.32   | 0.936    | 0.0452 | 0.0797  | 0.011   | 350  |
| elliptical-slice   | w[2]     | -0.637 | 0.332 | 0.0103          | 0.169   | 1.05     | 0.0177 | 0.0343  | 0.0053  | 299  |

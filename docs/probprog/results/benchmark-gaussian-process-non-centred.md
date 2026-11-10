# benchmark/gaussian-process-non-centred

Gaussian process regression, non-centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 52, scalePercent 100), 15.8 s.

**Passed**: 1 method(s) agree with the quadrature truth; 3 reported

**Gaussian process regression, non-centred** (gaussian-process-non-centred): gaussian-process, kernel, regression, non-centred, custom, quadrature

The Gaussian process regression written as f = L z with a whitened z: the geometry the sampler sees no longer depends on the length scale, and NUTS agrees with the quadrature truth.

Truth (quadrature): quadrature over the length scale (321 points, evidence error 1.8e-15) of the marginal likelihood N(y; 0, K + sigma² I) with the function integrated out; each function value is the grid mixture of its exact Gaussian conditionals (mean K (K + sigma² I)^-1 y). Log evidence -11.8396.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of ell is logNormal, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for ell, z); gaussian ('ell' is logNormal, not Gaussian); enumeration (continuous latent nodes (ell, z)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for ell, z).

| method       | expectation | run                    | diagnostics                                                               | why                                                                                                                 |
| ------------ | ----------- | ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (401 steps)  | 0 divergences; R-hat high: ell, z[2], z[3], ...; ESS low: ell, z[5], z[6] | whitened, so no funnel: agrees within Monte Carlo error                                                             |
| tempered-smc | reported    | converged (7 steps)    | weights ESS 888                                                           | tempered particles in thirteen dimensions: consistent, with a noisy tail                                            |
| laplace      | reported    | converged (64 steps)   | -                                                                         | a Gaussian at the mode of the whitened density: close for the function, the length scale is skewed on the log scale |
| svi          | reported    | completed (1500 steps) | -                                                                         | a mean-field normal on the whitened coordinates: the posterior of z is nearly independent, so the spreads are close |

| method       | quantity | mean   | sd    | mean error (sd) | mean z  | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ------ | ----- | --------------- | ------- | -------- | ------ | ------- | ------- | ---- |
| nuts         | ell      | 0.974  | 0.253 | -0.00763        | -0.0524 | 0.984    | 0.0285 | 0.0607  | 0.0327  | 45.6 |
| nuts         | f[0]     | -0.623 | 0.202 | -0.0131         | -0.245  | 0.99     | 0.0294 | 0.0336  | 0.0282  | 344  |
| nuts         | f[1]     | -0.656 | 0.172 | -0.0179         | -0.319  | 0.979    | 0.0293 | 0.0404  | 0.0133  | 303  |
| nuts         | f[2]     | -0.72  | 0.187 | -0.00453        | -0.0876 | 0.973    | 0.0395 | 0.0661  | 0.0232  | 354  |
| nuts         | f[3]     | -1.04  | 0.213 | 0.0122          | 0.238   | 0.966    | 0.0403 | 0.0611  | 0.0242  | 358  |
| nuts         | f[4]     | -1.23  | 0.24  | 0.0165          | 0.346   | 0.98     | 0.0501 | 0.044   | 0.0219  | 424  |
| nuts         | f[5]     | 0.482  | 0.27  | 0.017           | 0.283   | 1.06     | 0.0462 | 0.0849  | 0.0212  | 313  |
| nuts         | f[6]     | 1.14   | 0.2   | -0.049          | -0.922  | 1.05     | 0.069  | 0.0934  | 0.0324  | 392  |
| nuts         | f[7]     | 1.54   | 0.214 | -0.0678         | -1.06   | 1.02     | 0.0625 | 0.0983  | 0.0219  | 253  |
| nuts         | f[8]     | 1.51   | 0.196 | -0.031          | -0.631  | 0.97     | 0.0386 | 0.055   | 0.0323  | 390  |
| nuts         | f[9]     | 0.962  | 0.241 | -0.00191        | -0.0406 | 0.999    | 0.0316 | 0.042   | 0.0231  | 451  |
| nuts         | f[10]    | 0.176  | 0.191 | 0.0184          | 0.397   | 0.948    | 0.0286 | 0.0504  | 0.0296  | 419  |
| nuts         | f[11]    | 0.162  | 0.194 | 0.0165          | 0.36    | 0.95     | 0.0317 | 0.049   | 0.0266  | 428  |
| tempered-smc | ell      | 0.972  | 0.244 | -0.0146         | -0.486  | 0.95     | 0.0618 | 0.0997  | 0.0347  | 1000 |
| tempered-smc | f[0]     | -0.617 | 0.195 | 0.0132          | 0.437   | 0.956    | 0.0308 | 0.0461  | 0.0143  | 1000 |
| tempered-smc | f[1]     | -0.656 | 0.168 | -0.0193         | -0.64   | 0.956    | 0.034  | 0.0503  | 0.0223  | 1000 |
| tempered-smc | f[2]     | -0.731 | 0.185 | -0.065          | -2.13   | 0.964    | 0.0492 | 0.0684  | 0.0153  | 1000 |
| tempered-smc | f[3]     | -1.06  | 0.196 | -0.0729         | -2.6    | 0.888    | 0.0656 | 0.118   | 0.0318  | 1000 |
| tempered-smc | f[4]     | -1.22  | 0.237 | 0.0612          | 2.01    | 0.966    | 0.0544 | 0.0639  | 0.0185  | 1000 |
| tempered-smc | f[5]     | 0.479  | 0.259 | 0.00291         | 0.0902  | 1.02     | 0.0368 | 0.0384  | 0.0126  | 1000 |
| tempered-smc | f[6]     | 1.13   | 0.185 | -0.0736         | -2.38   | 0.978    | 0.064  | 0.0814  | 0.0159  | 1000 |
| tempered-smc | f[7]     | 1.55   | 0.198 | -0.0236         | -0.786  | 0.948    | 0.0474 | 0.0619  | 0.0179  | 1000 |
| tempered-smc | f[8]     | 1.53   | 0.192 | 0.0907          | 3.01    | 0.952    | 0.0531 | 0.102   | 0.0132  | 1000 |
| tempered-smc | f[9]     | 0.978  | 0.23  | 0.0654          | 2.17    | 0.952    | 0.0555 | 0.0842  | 0.0189  | 1000 |
| tempered-smc | f[10]    | 0.174  | 0.199 | 0.00948         | 0.303   | 0.989    | 0.0318 | 0.0367  | 0.0191  | 1000 |
| tempered-smc | f[11]    | 0.162  | 0.202 | 0.0166          | 0.529   | 0.991    | 0.0259 | 0.0349  | 0.0149  | 1000 |
| laplace      | ell      | 0.773  | 0.296 | -0.789          | -43.3   | 1.15     | 0.355  | 0.807   | 0.336   | 4000 |
| laplace      | f[0]     | -0.633 | 0.212 | -0.0646         | -3.94   | 1.04     | 0.0403 | 0.0691  | 0.00712 | 4000 |
| laplace      | f[1]     | -0.658 | 0.181 | -0.0284         | -1.74   | 1.03     | 0.0231 | 0.0438  | 0.00341 | 4000 |
| laplace      | f[2]     | -0.667 | 0.239 | 0.268           | 13.6    | 1.24     | 0.106  | 0.279   | 0.038   | 4000 |
| laplace      | f[3]     | -1     | 0.301 | 0.17            | 7.91    | 1.36     | 0.0945 | 0.289   | 0.0512  | 4000 |
| laplace      | f[4]     | -1.26  | 0.375 | -0.0893         | -3.69   | 1.53     | 0.0731 | 0.316   | 0.0471  | 4000 |
| laplace      | f[5]     | 0.149  | 0.423 | -1.3            | -49.2   | 1.66     | 0.37   | 1.3     | 0.396   | 4000 |
| laplace      | f[6]     | 1      | 0.432 | -0.755          | -20.9   | 2.28     | 0.177  | 0.8     | 0.126   | 4000 |
| laplace      | f[7]     | 1.58   | 0.442 | 0.107           | 3.22    | 2.11     | 0.149  | 0.721   | 0.139   | 4000 |
| laplace      | f[8]     | 1.55   | 0.334 | 0.174           | 6.67    | 1.66     | 0.123  | 0.445   | 0.0779  | 4000 |
| laplace      | f[9]     | 0.959  | 0.35  | -0.0124         | -0.54   | 1.45     | 0.0933 | 0.337   | 0.0789  | 4000 |
| laplace      | f[10]    | 0.151  | 0.382 | -0.103          | -3.41   | 1.9      | 0.111  | 0.583   | 0.124   | 4000 |
| laplace      | f[11]    | 0.141  | 0.386 | -0.0845         | -2.83   | 1.89     | 0.106  | 0.575   | 0.121   | 4000 |
| svi          | ell      | 0.693  | 0.064 | -1.1            | -280    | 0.249    | 0.688  | 1.12    | 3.25    | 4000 |
| svi          | f[0]     | -0.647 | 0.164 | -0.135          | -10.6   | 0.804    | 0.0974 | 0.192   | 0.0603  | 4000 |
| svi          | f[1]     | -0.68  | 0.167 | -0.156          | -10.4   | 0.951    | 0.0704 | 0.157   | 0.0175  | 4000 |
| svi          | f[2]     | -0.739 | 0.203 | -0.105          | -6.31   | 1.06     | 0.0517 | 0.106   | 0.0114  | 4000 |
| svi          | f[3]     | -1.09  | 0.329 | -0.212          | -8.99   | 1.49     | 0.146  | 0.434   | 0.124   | 4000 |
| svi          | f[4]     | -1.3   | 0.362 | -0.263          | -11.3   | 1.48     | 0.155  | 0.445   | 0.122   | 4000 |
| svi          | f[5]     | 0.408  | 0.206 | -0.275          | -21.5   | 0.811    | 0.152  | 0.286   | 0.116   | 4000 |
| svi          | f[6]     | 1.17   | 0.22  | 0.151           | 8.24    | 1.16     | 0.0865 | 0.191   | 0.0302  | 4000 |
| svi          | f[7]     | 1.6    | 0.242 | 0.208           | 11.4    | 1.16     | 0.109  | 0.235   | 0.0386  | 4000 |
| svi          | f[8]     | 1.43   | 0.29  | -0.418          | -18.4   | 1.44     | 0.182  | 0.488   | 0.128   | 4000 |
| svi          | f[9]     | 0.694  | 0.366 | -1.11           | -46.4   | 1.52     | 0.377  | 1.12    | 0.378   | 4000 |
| svi          | f[10]    | 0.0355 | 0.321 | -0.679          | -26.9   | 1.6      | 0.267  | 0.759   | 0.233   | 4000 |
| svi          | f[11]    | 0.0298 | 0.321 | -0.63           | -25.4   | 1.57     | 0.251  | 0.711   | 0.215   | 4000 |

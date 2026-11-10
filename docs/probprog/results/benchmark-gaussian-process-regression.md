# benchmark/gaussian-process-regression

Gaussian process regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 52, scalePercent 100), 55.6 s.

**Passed**: 0 method(s) agree with the quadrature truth, 1 differ as expected; 3 reported

**Gaussian process regression** (gaussian-process-regression): gaussian-process, kernel, regression, multivariate-normal, linear-gaussian, custom, quadrature

A latent function with a squared-exponential kernel of unknown length scale observed with known noise: the function posterior is Gaussian given the length scale, so the truth is a one-dimensional quadrature mixture.

Truth (quadrature): quadrature over the length scale (321 points, evidence error 1.8e-15) of the marginal likelihood N(y; 0, K + sigma² I) with the function integrated out; each function value is the grid mixture of its exact Gaussian conditionals (mean K (K + sigma² I)^-1 y). Log evidence -11.8396.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, gibbs, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of ell is logNormal, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for ell, f); gaussian ('ell' is logNormal, not Gaussian); enumeration (continuous latent nodes (ell, f)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for ell, f).

| method       | expectation        | run                    | diagnostics                                                                      | why                                                                                                                                                               |
| ------------ | ------------------ | ---------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts         | expected to differ | converged (401 steps)  | 119 divergences; R-hat high: ell, f[0], f[1], ...; ESS low: ell, f[0], f[1], ... | the centred form has a funnel between the length scale and the function: divergences, and the length scale is explored poorly (see the non-centred variant)       |
| tempered-smc | reported           | converged (7 steps)    | weights ESS 873                                                                  | tempered particles in thirteen dimensions: consistent, with a noisy tail                                                                                          |
| laplace      | reported           | completed (400 steps)  | -                                                                                | the joint mode of the function and the length scale of a hierarchy is degenerate (the density grows without bound as the function follows the data): no posterior |
| svi          | reported           | completed (1500 steps) | -                                                                                | a mean-field normal ignores the correlation between neighbouring function values: spreads too small                                                               |

| method       | quantity | mean    | sd       | mean error (sd) | mean z           | sd ratio | KS     | W1 (sd) | KL     | ESS  |
| ------------ | -------- | ------- | -------- | --------------- | ---------------- | -------- | ------ | ------- | ------ | ---- |
| nuts         | ell      | 0.824   | 0.155    | -0.591          | -2.11            | 0.603    | 0.456  | 0.591   | 1.35   | 4.65 |
| nuts         | f[0]     | -0.599  | 0.151    | 0.103           | 1.93             | 0.739    | 0.221  | 0.335   | 0.338  | 193  |
| nuts         | f[1]     | -0.675  | 0.143    | -0.124          | -1.59            | 0.812    | 0.242  | 0.322   | 0.337  | 108  |
| nuts         | f[2]     | -0.796  | 0.212    | -0.403          | -1.28            | 1.1      | 0.364  | 0.447   | 0.414  | 12.3 |
| nuts         | f[3]     | -1.07   | 0.195    | -0.127          | -0.591           | 0.882    | 0.227  | 0.237   | 0.25   | 16.8 |
| nuts         | f[4]     | -1.23   | 0.224    | 0.014           | 0.157            | 0.915    | 0.221  | 0.243   | 0.365  | 105  |
| nuts         | f[5]     | 0.43    | 0.198    | -0.189          | -3.34            | 0.778    | 0.261  | 0.315   | 0.36   | 188  |
| nuts         | f[6]     | 1.19    | 0.145    | 0.251           | 5.31             | 0.762    | 0.293  | 0.336   | 0.354  | 260  |
| nuts         | f[7]     | 1.65    | 0.146    | 0.421           | 9.01             | 0.698    | 0.356  | 0.448   | 0.443  | 223  |
| nuts         | f[8]     | 1.58    | 0.176    | 0.343           | 2.98             | 0.871    | 0.335  | 0.401   | 0.318  | 57.2 |
| nuts         | f[9]     | 0.983   | 0.198    | 0.0888          | 0.923            | 0.822    | 0.208  | 0.271   | 0.281  | 72.8 |
| nuts         | f[10]    | 0.0697  | 0.182    | -0.509          | -1.87            | 0.903    | 0.345  | 0.558   | 0.372  | 11   |
| nuts         | f[11]    | 0.0536  | 0.185    | -0.514          | -1.85            | 0.907    | 0.339  | 0.562   | 0.361  | 10.7 |
| tempered-smc | ell      | 0.876   | 0.324    | -0.388          | -9.73            | 1.26     | 0.408  | 0.646   | 1.27   | 1000 |
| tempered-smc | f[0]     | -0.641  | 0.213    | -0.104          | -3.15            | 1.04     | 0.0987 | 0.123   | 0.107  | 1000 |
| tempered-smc | f[1]     | -0.652  | 0.173    | 0.00635         | 0.204            | 0.985    | 0.076  | 0.0699  | 0.0914 | 1000 |
| tempered-smc | f[2]     | -0.704  | 0.171    | 0.0739          | 2.62             | 0.891    | 0.131  | 0.131   | 0.0934 | 1000 |
| tempered-smc | f[3]     | -1.07   | 0.217    | -0.16           | -5.15            | 0.984    | 0.118  | 0.2     | 0.144  | 1000 |
| tempered-smc | f[4]     | -1.17   | 0.263    | 0.289           | 8.51             | 1.07     | 0.138  | 0.298   | 0.0994 | 1000 |
| tempered-smc | f[5]     | 0.493   | 0.299    | 0.0596          | 1.6              | 1.17     | 0.126  | 0.195   | 0.11   | 1000 |
| tempered-smc | f[6]     | 1.13    | 0.187    | -0.0742         | -2.38            | 0.984    | 0.119  | 0.114   | 0.0862 | 1000 |
| tempered-smc | f[7]     | 1.6     | 0.222    | 0.18            | 5.36             | 1.06     | 0.151  | 0.259   | 0.183  | 1000 |
| tempered-smc | f[8]     | 1.58    | 0.24     | 0.316           | 8.41             | 1.19     | 0.177  | 0.358   | 0.205  | 1000 |
| tempered-smc | f[9]     | 0.903   | 0.247    | -0.242          | -7.5             | 1.02     | 0.128  | 0.244   | 0.179  | 1000 |
| tempered-smc | f[10]    | 0.241   | 0.262    | 0.344           | 8.36             | 1.3      | 0.172  | 0.383   | 0.114  | 1000 |
| tempered-smc | f[11]    | 0.245   | 0.26     | 0.421           | 10.5             | 1.27     | 0.199  | 0.444   | 0.134  | 1000 |
| laplace      | ell      | 2       | 1.64e-13 | 3.97            | 393000000000000  | 6.38e-13 | 1      | 3.97    | 5.54   | 4000 |
| laplace      | f[0]     | -0.646  | 3.02e-14 | -0.13           | -55400000000000  | 1.48e-13 | 0.552  | 0.804   | 5.54   | 4000 |
| laplace      | f[1]     | -0.717  | 0        | -0.363          | -Infinity        | 0        | 0.642  | 0.85    | 5.54   | 4000 |
| laplace      | f[2]     | -0.85   | 1.58e-14 | -0.681          | -526000000000000 | 8.19e-14 | 0.753  | 0.969   | 5.54   | 4000 |
| laplace      | f[3]     | -1.06   | 2.62e-14 | -0.104          | -55700000000000  | 1.19e-13 | 0.536  | 0.797   | 5.54   | 4000 |
| laplace      | f[4]     | -0.993  | 1.29e-14 | 1               | 1200000000000000 | 5.26e-14 | 0.842  | 1.16    | 5.54   | 4000 |
| laplace      | f[5]     | 0.745   | 4.27e-14 | 1.05            | 395000000000000  | 1.68e-13 | 0.853  | 1.19    | 5.54   | 4000 |
| laplace      | f[6]     | 1.05    | 4.64e-14 | -0.506          | -131000000000000 | 2.45e-13 | 0.692  | 0.891   | 5.54   | 4000 |
| laplace      | f[7]     | 1.25    | 4.77e-14 | -1.48           | -411000000000000 | 2.28e-13 | 0.937  | 1.53    | 5.54   | 4000 |
| laplace      | f[8]     | 1.28    | 6.22e-14 | -1.17           | -240000000000000 | 3.08e-13 | 0.881  | 1.29    | 5.54   | 4000 |
| laplace      | f[9]     | 1.06    | 9.1e-14  | 0.393           | 65900000000000   | 3.77e-13 | 0.643  | 0.845   | 5.54   | 4000 |
| laplace      | f[10]    | 0.322   | 1.5e-15  | 0.747           | 6340000000000000 | 7.45e-15 | 0.772  | 1.01    | 5.54   | 4000 |
| laplace      | f[11]    | 0.297   | 4.5e-15  | 0.679           | 1950000000000000 | 2.2e-14  | 0.751  | 0.975   | 5.54   | 4000 |
| svi          | ell      | 0.533   | 0.0279   | -1.72           | -1010            | 0.108    | 0.935  | 1.73    | 5.27   | 4000 |
| svi          | f[0]     | -0.833  | 0.0808   | -1.04           | -167             | 0.396    | 0.608  | 1.07    | 2.63   | 4000 |
| svi          | f[1]     | -0.509  | 0.0549   | 0.817           | 165              | 0.313    | 0.569  | 0.896   | 2.56   | 4000 |
| svi          | f[2]     | 0.173   | 0.104    | 4.64            | 543              | 0.54     | 0.997  | 4.64    | 5.54   | 4000 |
| svi          | f[3]     | -0.0129 | 0.208    | 4.64            | 312              | 0.941    | 0.982  | 4.64    | 5.49   | 4000 |
| svi          | f[4]     | -0.597  | 0.128    | 2.61            | 316              | 0.524    | 0.922  | 2.61    | 5.04   | 4000 |
| svi          | f[5]     | -0.218  | 0.168    | -2.74           | -263             | 0.659    | 0.903  | 2.74    | 4.61   | 4000 |
| svi          | f[6]     | 0.811   | 0.119    | -1.76           | -178             | 0.625    | 0.737  | 1.76    | 3.05   | 4000 |
| svi          | f[7]     | 0.987   | 0.0729   | -2.73           | -495             | 0.349    | 0.97   | 2.73    | 5.46   | 4000 |
| svi          | f[8]     | 0.504   | 0.124    | -5.01           | -516             | 0.614    | 0.998  | 5.01    | 5.54   | 4000 |
| svi          | f[9]     | 0.652   | 0.204    | -1.28           | -95.9            | 0.846    | 0.53   | 1.28    | 1.18   | 4000 |
| svi          | f[10]    | -0.863  | 0.041    | -5.15           | -1600            | 0.204    | 1      | 5.15    | 5.54   | 4000 |
| svi          | f[11]    | -0.899  | 0.0424   | -5.17           | -1580            | 0.207    | 1      | 5.17    | 5.54   | 4000 |

# benchmark/gaussian-process-regression

Gaussian process regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 4, rows 52, scalePercent 100), 69.8 s.

**Passed**: 0 method(s) agree with the quadrature truth, 1 differ as expected; 3 reported

**Gaussian process regression** (gaussian-process-regression): gaussian-process, kernel, regression, multivariate-normal, linear-gaussian, custom, quadrature

A latent function with a squared-exponential kernel of unknown length scale observed with known noise: the function posterior is Gaussian given the length scale, so the truth is a one-dimensional quadrature mixture.

Truth (quadrature): quadrature over the length scale (321 points, evidence error 1.8e-15) of the marginal likelihood N(y; 0, K + sigma² I) with the function integrated out; each function value is the grid mixture of its exact Gaussian conditionals (mean K (K + sigma² I)^-1 y). Log evidence -11.8396.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: ep (expectation propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for ell, f); gaussian ('ell' is logNormal, not Gaussian); enumeration (continuous latent nodes (ell, f)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | expected to differ | converged (401 steps) | 57 divergences; R-hat high: ell, f[1], f[2], ...; ESS low: ell, f[0], f[1], ... | the centred form has a funnel between the length scale and the function: divergences, and the length scale is explored poorly (see the non-centred variant) |
| tempered-smc | reported | converged (7 steps) | weights ESS 881 | tempered particles in thirteen dimensions: consistent, with a noisy tail |
| laplace | reported | completed (400 steps) | - | the joint mode of the function and the length scale of a hierarchy is degenerate (the density grows without bound as the function follows the data): no posterior |
| svi | reported | completed (1500 steps) | - | a mean-field normal ignores the correlation between neighbouring function values: spreads too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | ell | 0.84 | 0.146 | -0.53 | -4.57 | 0.566 | 0.352 | 0.532 | 0.948 | 23.9 |
| nuts | f[0] | -0.645 | 0.226 | -0.121 | -1.24 | 1.11 | 0.113 | 0.163 | 0.0793 | 130 |
| nuts | f[1] | -0.678 | 0.19 | -0.141 | -1.04 | 1.08 | 0.116 | 0.186 | 0.059 | 64.1 |
| nuts | f[2] | -0.741 | 0.208 | -0.113 | -0.677 | 1.08 | 0.0903 | 0.146 | 0.0527 | 41.7 |
| nuts | f[3] | -1.05 | 0.26 | -0.0519 | -0.43 | 1.18 | 0.105 | 0.212 | 0.0842 | 95.2 |
| nuts | f[4] | -1.22 | 0.243 | 0.073 | 0.838 | 0.993 | 0.0521 | 0.0948 | 0.0569 | 130 |
| nuts | f[5] | 0.374 | 0.244 | -0.41 | -3.5 | 0.961 | 0.199 | 0.415 | 0.127 | 67.3 |
| nuts | f[6] | 1.11 | 0.173 | -0.201 | -1.89 | 0.911 | 0.137 | 0.211 | 0.0937 | 72.8 |
| nuts | f[7] | 1.58 | 0.208 | 0.11 | 1 | 0.995 | 0.0944 | 0.12 | 0.046 | 83.2 |
| nuts | f[8] | 1.54 | 0.2 | 0.133 | 1.61 | 0.991 | 0.0905 | 0.142 | 0.0761 | 144 |
| nuts | f[9] | 0.948 | 0.232 | -0.0567 | -0.773 | 0.962 | 0.0462 | 0.0721 | 0.034 | 172 |
| nuts | f[10] | 0.215 | 0.212 | 0.214 | 1.62 | 1.05 | 0.129 | 0.221 | 0.0889 | 63.5 |
| nuts | f[11] | 0.205 | 0.213 | 0.226 | 1.75 | 1.04 | 0.138 | 0.232 | 0.0743 | 65.3 |
| tempered-smc | ell | 0.812 | 0.117 | -0.637 | -44.4 | 0.453 | 0.432 | 0.649 | 1.13 | 1000 |
| tempered-smc | f[0] | -0.597 | 0.175 | 0.111 | 4.08 | 0.858 | 0.103 | 0.176 | 0.0496 | 1000 |
| tempered-smc | f[1] | -0.683 | 0.149 | -0.172 | -6.41 | 0.849 | 0.106 | 0.192 | 0.0721 | 1000 |
| tempered-smc | f[2] | -0.803 | 0.166 | -0.438 | -16.1 | 0.863 | 0.221 | 0.439 | 0.152 | 1000 |
| tempered-smc | f[3] | -0.977 | 0.208 | 0.281 | 9.44 | 0.942 | 0.15 | 0.282 | 0.0746 | 1000 |
| tempered-smc | f[4] | -1.09 | 0.24 | 0.602 | 19.4 | 0.981 | 0.24 | 0.602 | 0.189 | 1000 |
| tempered-smc | f[5] | 0.392 | 0.216 | -0.337 | -12.5 | 0.85 | 0.2 | 0.342 | 0.115 | 1000 |
| tempered-smc | f[6] | 1.14 | 0.174 | -0.0506 | -1.75 | 0.917 | 0.0614 | 0.0877 | 0.031 | 1000 |
| tempered-smc | f[7] | 1.62 | 0.196 | 0.291 | 9.83 | 0.936 | 0.148 | 0.291 | 0.0762 | 1000 |
| tempered-smc | f[8] | 1.53 | 0.183 | 0.0747 | 2.61 | 0.906 | 0.0677 | 0.119 | 0.0264 | 1000 |
| tempered-smc | f[9] | 0.805 | 0.213 | -0.649 | -23.2 | 0.884 | 0.335 | 0.652 | 0.298 | 1000 |
| tempered-smc | f[10] | 0.0382 | 0.238 | -0.666 | -17.8 | 1.18 | 0.261 | 0.666 | 0.188 | 1000 |
| tempered-smc | f[11] | 0.0288 | 0.247 | -0.635 | -16.6 | 1.21 | 0.25 | 0.635 | 0.165 | 1000 |
| laplace | ell | 2 | 2.93e-14 | 3.97 | 2200000000000000 | 1.14e-13 | 1 | 3.97 | 5.54 | 4000 |
| laplace | f[0] | -0.648 | 2.39e-14 | -0.138 | -74300000000000 | 1.17e-13 | 0.556 | 0.805 | 5.54 | 4000 |
| laplace | f[1] | -0.716 | 2.42e-14 | -0.358 | -164000000000000 | 1.38e-13 | 0.64 | 0.848 | 5.54 | 4000 |
| laplace | f[2] | -0.844 | 7.8e-14 | -0.654 | -102000000000000 | 4.06e-13 | 0.744 | 0.955 | 5.54 | 4000 |
| laplace | f[3] | -1.05 | 3.86e-14 | -0.0559 | -20200000000000 | 1.75e-13 | 0.516 | 0.794 | 5.54 | 4000 |
| laplace | f[4] | -0.984 | 6.63e-14 | 1.04 | 243000000000000 | 2.7e-13 | 0.851 | 1.19 | 5.54 | 4000 |
| laplace | f[5] | 0.737 | 3.65e-14 | 1.02 | 448000000000000 | 1.44e-13 | 0.846 | 1.17 | 5.54 | 4000 |
| laplace | f[6] | 1.04 | 5.15e-14 | -0.549 | -128000000000000 | 2.72e-13 | 0.707 | 0.908 | 5.54 | 4000 |
| laplace | f[7] | 1.24 | 4.75e-14 | -1.52 | -423000000000000 | 2.27e-13 | 0.941 | 1.56 | 5.54 | 4000 |
| laplace | f[8] | 1.27 | 6.35e-14 | -1.2 | -242000000000000 | 3.15e-13 | 0.888 | 1.31 | 5.54 | 4000 |
| laplace | f[9] | 1.05 | 7.55e-14 | 0.375 | 75800000000000 | 3.13e-13 | 0.636 | 0.84 | 5.54 | 4000 |
| laplace | f[10] | 0.322 | 4.22e-15 | 0.743 | 2240000000000000 | 2.1e-14 | 0.771 | 1.01 | 5.54 | 4000 |
| laplace | f[11] | 0.297 | 1.24e-14 | 0.676 | 703000000000000 | 6.08e-14 | 0.75 | 0.973 | 5.54 | 4000 |
| svi | ell | 0.345 | 0.032 | -2.45 | -1250 | 0.125 | 0.991 | 2.45 | 5.54 | 4000 |
| svi | f[0] | -0.608 | 0.146 | 0.0561 | 4.97 | 0.714 | 0.0989 | 0.232 | 0.119 | 4000 |
| svi | f[1] | -0.698 | 0.109 | -0.255 | -25.9 | 0.622 | 0.201 | 0.366 | 0.347 | 4000 |
| svi | f[2] | -0.671 | 0.211 | 0.245 | 14.2 | 1.09 | 0.106 | 0.247 | 0.0377 | 4000 |
| svi | f[3] | -0.903 | 0.263 | 0.617 | 32.9 | 1.19 | 0.237 | 0.619 | 0.161 | 4000 |
| svi | f[4] | -1.5 | 0.311 | -1.09 | -54.3 | 1.27 | 0.389 | 1.09 | 0.41 | 4000 |
| svi | f[5] | 0.285 | 0.243 | -0.759 | -50.2 | 0.957 | 0.324 | 0.76 | 0.313 | 4000 |
| svi | f[6] | 1.26 | 0.215 | 0.621 | 34.6 | 1.13 | 0.237 | 0.622 | 0.161 | 4000 |
| svi | f[7] | 1.86 | 0.218 | 1.43 | 87.2 | 1.04 | 0.522 | 1.43 | 0.947 | 4000 |
| svi | f[8] | 1.42 | 0.207 | -0.452 | -27.9 | 1.03 | 0.182 | 0.453 | 0.097 | 4000 |
| svi | f[9] | 0.796 | 0.264 | -0.687 | -39.8 | 1.09 | 0.275 | 0.688 | 0.207 | 4000 |
| svi | f[10] | 0.169 | 0.0457 | -0.0159 | -4.42 | 0.227 | 0.311 | 0.616 | 2.64 | 4000 |
| svi | f[11] | 0.144 | 0.0484 | -0.0711 | -19 | 0.237 | 0.328 | 0.611 | 2.51 | 4000 |

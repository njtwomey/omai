# benchmark/ar1-state-space

AR(1) state space: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 5, rows 60, scalePercent 100), 10.4 s.

**Passed**: 3 method(s) agree with the exact truth; 2 reported

**AR(1) state space** (ar1-state-space): state-space, markov, linear-gaussian, time-series, exact, program

A stationary AR(1) latent state observed with Gaussian noise, as a markov program: the linear-Gaussian chain whose exact smoothing posterior is the truth.

Truth (exact): dense Gaussian conditioning of the joint of the states and the observations: posterior S (S + R)^-1 y, evidence N(y; 0, S + R). Log evidence -19.2199.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, particle-filter, abc-smc, gaussian, abc-rejection. Not applicable: gibbs (no discrete or conjugate conditionals (use NUTS)); em (no params to estimate); conjugate (no conjugate structure for x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11); enumeration (continuous latent sites (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)); chain (continuous latent sites).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| gaussian | must agree | converged (1 steps) | - | the Kalman smoother: exact |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| laplace | must agree | converged (23 steps) | - | the posterior is Gaussian, so a normal at the mode is exact |
| particle-filter | reported | completed (12 steps) | - | a bootstrap filter: unbiased evidence, filtering marginals right, early smoothing marginals thinned by the genealogy |
| svi | reported | completed (1500 steps) | - | a mean-field normal ignores the correlation of neighbouring states: too narrow |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian | x0 | 0.537 | 0.429 | -0.0258 | -1.62 | 1.01 | 0.0167 | 0.0277 | 0.00234 | 4000 |
| gaussian | x1 | 0.613 | 0.385 | -0.0123 | -0.792 | 0.985 | 0.0172 | 0.0196 | 0.00207 | 4000 |
| gaussian | x2 | 0.828 | 0.386 | -0.00737 | -0.465 | 1 | 0.00925 | 0.016 | 0.00233 | 4000 |
| gaussian | x3 | 1.05 | 0.385 | -0.0145 | -0.916 | 1 | 0.0119 | 0.0221 | 0.00347 | 4000 |
| gaussian | x4 | 1 | 0.389 | -0.00465 | -0.29 | 1.01 | 0.0136 | 0.0194 | 0.00193 | 4000 |
| gaussian | x5 | 1.44 | 0.387 | -0.0051 | -0.32 | 1.01 | 0.00783 | 0.0121 | 0.00176 | 4000 |
| gaussian | x6 | 2.18 | 0.381 | -0.00383 | -0.244 | 0.994 | 0.00923 | 0.017 | 0.00196 | 4000 |
| gaussian | x7 | 2.35 | 0.378 | 0.017 | 1.09 | 0.985 | 0.0124 | 0.0204 | 0.00129 | 4000 |
| gaussian | x8 | 1.73 | 0.38 | -0.0109 | -0.697 | 0.991 | 0.00874 | 0.0154 | 0.00105 | 4000 |
| gaussian | x9 | 0.87 | 0.382 | 0.00182 | 0.116 | 0.993 | 0.0109 | 0.0149 | 0.00365 | 4000 |
| gaussian | x10 | 0.383 | 0.387 | 0.00968 | 0.618 | 0.99 | 0.00989 | 0.0168 | 0.00152 | 4000 |
| gaussian | x11 | 0.57 | 0.421 | -0.0077 | -0.492 | 0.989 | 0.00894 | 0.0145 | 0.00156 | 4000 |
| nuts | x0 | 0.543 | 0.419 | -0.0125 | -0.495 | 0.984 | 0.0221 | 0.0312 | 0.00594 | 1510 |
| nuts | x1 | 0.616 | 0.394 | -0.00476 | -0.163 | 1.01 | 0.0166 | 0.0349 | 0.00777 | 1180 |
| nuts | x2 | 0.837 | 0.388 | 0.0164 | 0.52 | 1.01 | 0.0211 | 0.0222 | 0.00705 | 1020 |
| nuts | x3 | 1.06 | 0.374 | -0.000596 | -0.0214 | 0.975 | 0.0187 | 0.0231 | 0.00435 | 1220 |
| nuts | x4 | 1.01 | 0.371 | 0.0255 | 0.888 | 0.967 | 0.0234 | 0.0416 | 0.00458 | 1130 |
| nuts | x5 | 1.45 | 0.387 | 0.0301 | 1.04 | 1.01 | 0.0243 | 0.0346 | 0.00644 | 1220 |
| nuts | x6 | 2.18 | 0.387 | -0.00681 | -0.228 | 1.01 | 0.0203 | 0.0232 | 0.00998 | 1140 |
| nuts | x7 | 2.34 | 0.384 | -0.00537 | -0.172 | 1 | 0.0142 | 0.0196 | 0.0059 | 1030 |
| nuts | x8 | 1.76 | 0.399 | 0.0512 | 1.64 | 1.04 | 0.04 | 0.075 | 0.0166 | 1100 |
| nuts | x9 | 0.874 | 0.385 | 0.0111 | 0.39 | 1 | 0.0255 | 0.0343 | 0.00987 | 1230 |
| nuts | x10 | 0.39 | 0.384 | 0.0283 | 0.959 | 0.984 | 0.0243 | 0.0346 | 0.00603 | 1110 |
| nuts | x11 | 0.573 | 0.407 | -0.00135 | -0.0522 | 0.955 | 0.027 | 0.048 | 0.0145 | 1360 |
| laplace | x0 | 0.538 | 0.425 | -0.0242 | -1.53 | 0.998 | 0.0182 | 0.0262 | 0.00293 | 4000 |
| laplace | x1 | 0.615 | 0.386 | -0.00908 | -0.581 | 0.989 | 0.00838 | 0.0145 | 0.00108 | 4000 |
| laplace | x2 | 0.828 | 0.392 | -0.00758 | -0.47 | 1.02 | 0.0181 | 0.0264 | 0.00413 | 4000 |
| laplace | x3 | 1.05 | 0.383 | -0.01 | -0.634 | 0.998 | 0.0089 | 0.014 | 0.00189 | 4000 |
| laplace | x4 | 1 | 0.389 | 0.00411 | 0.256 | 1.01 | 0.00949 | 0.0143 | 0.00214 | 4000 |
| laplace | x5 | 1.44 | 0.378 | -0.00755 | -0.484 | 0.986 | 0.014 | 0.0206 | 0.00209 | 4000 |
| laplace | x6 | 2.18 | 0.387 | -0.00324 | -0.203 | 1.01 | 0.00974 | 0.0165 | 0.00233 | 4000 |
| laplace | x7 | 2.34 | 0.379 | -0.00999 | -0.639 | 0.989 | 0.0112 | 0.0158 | 0.00137 | 4000 |
| laplace | x8 | 1.73 | 0.391 | -0.0209 | -1.3 | 1.02 | 0.0156 | 0.0308 | 0.00417 | 4000 |
| laplace | x9 | 0.868 | 0.382 | -0.00282 | -0.18 | 0.993 | 0.0159 | 0.0265 | 0.00285 | 4000 |
| laplace | x10 | 0.379 | 0.392 | -0.000313 | -0.0197 | 1.01 | 0.0131 | 0.0168 | 0.00232 | 4000 |
| laplace | x11 | 0.578 | 0.435 | 0.0108 | 0.67 | 1.02 | 0.0136 | 0.0241 | 0.00322 | 4000 |
| particle-filter | x0 | 0.568 | 0.401 | 0.0462 | 1.79 | 0.94 | 0.0982 | 0.153 | 0.441 | 1330 |
| particle-filter | x1 | 0.594 | 0.332 | -0.0615 | -2.64 | 0.85 | 0.114 | 0.19 | 0.514 | 1330 |
| particle-filter | x2 | 0.667 | 0.43 | -0.426 | -13.9 | 1.12 | 0.19 | 0.426 | 0.309 | 1330 |
| particle-filter | x3 | 1.08 | 0.44 | 0.0732 | 2.33 | 1.15 | 0.146 | 0.222 | 0.375 | 1330 |
| particle-filter | x4 | 0.971 | 0.38 | -0.0811 | -2.99 | 0.991 | 0.0945 | 0.134 | 0.404 | 1330 |
| particle-filter | x5 | 1.47 | 0.377 | 0.0897 | 3.32 | 0.984 | 0.0829 | 0.158 | 0.419 | 1330 |
| particle-filter | x6 | 2.23 | 0.383 | 0.119 | 4.34 | 0.999 | 0.118 | 0.207 | 0.88 | 1330 |
| particle-filter | x7 | 2.36 | 0.353 | 0.0434 | 1.72 | 0.92 | 0.0545 | 0.0945 | 0.0333 | 1330 |
| particle-filter | x8 | 1.72 | 0.379 | -0.0511 | -1.89 | 0.987 | 0.0725 | 0.0605 | 0.0247 | 1330 |
| particle-filter | x9 | 0.875 | 0.384 | 0.0139 | 0.509 | 0.998 | 0.0397 | 0.0506 | 0.0341 | 1330 |
| particle-filter | x10 | 0.36 | 0.389 | -0.05 | -1.83 | 0.995 | 0.0422 | 0.056 | 0.00493 | 1330 |
| particle-filter | x11 | 0.563 | 0.422 | -0.0246 | -0.906 | 0.991 | 0.0308 | 0.0352 | 0.00939 | 1330 |
| svi | x0 | 0.542 | 0.353 | -0.014 | -1.07 | 0.828 | 0.0528 | 0.143 | 0.0347 | 4000 |
| svi | x1 | 0.662 | 0.372 | 0.113 | 7.47 | 0.953 | 0.0569 | 0.114 | 0.0119 | 4000 |
| svi | x2 | 0.889 | 0.351 | 0.152 | 10.5 | 0.912 | 0.0745 | 0.154 | 0.0241 | 4000 |
| svi | x3 | 1.01 | 0.327 | -0.106 | -7.85 | 0.853 | 0.0813 | 0.143 | 0.032 | 4000 |
| svi | x4 | 0.967 | 0.386 | -0.0893 | -5.6 | 1.01 | 0.0421 | 0.0896 | 0.00587 | 4000 |
| svi | x5 | 1.44 | 0.31 | -0.00725 | -0.567 | 0.809 | 0.0624 | 0.154 | 0.046 | 4000 |
| svi | x6 | 2.26 | 0.3 | 0.215 | 17.3 | 0.783 | 0.128 | 0.248 | 0.102 | 4000 |
| svi | x7 | 2.28 | 0.3 | -0.157 | -12.7 | 0.783 | 0.112 | 0.217 | 0.0787 | 4000 |
| svi | x8 | 1.73 | 0.336 | -0.0187 | -1.35 | 0.877 | 0.0463 | 0.0997 | 0.0194 | 4000 |
| svi | x9 | 0.868 | 0.289 | -0.00322 | -0.271 | 0.75 | 0.0737 | 0.195 | 0.0863 | 4000 |
| svi | x10 | 0.409 | 0.27 | 0.076 | 6.97 | 0.69 | 0.116 | 0.258 | 0.152 | 4000 |
| svi | x11 | 0.558 | 0.383 | -0.0359 | -2.53 | 0.898 | 0.0383 | 0.0897 | 0.0137 | 4000 |

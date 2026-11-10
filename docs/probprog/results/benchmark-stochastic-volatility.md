# benchmark/stochastic-volatility

Stochastic volatility: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 631180e6; seed 0, budget scale 1 (plans 5, rows 150, scalePercent 100), 57.8 s.

**Passed**: 1 method(s) agree with the quadrature truth; 4 reported

**Stochastic volatility** (stochastic-volatility): state-space, scan, markov, time-series, non-gaussian, volatility, quadrature

A log-variance following an AR(1) scales Gaussian returns, the parameters known: a non-linear, non-Gaussian state-space model written as a scan, with a point-mass smoother as the truth.

Truth (quadrature): point-mass filter and smoother on a grid of 800 log-variance values over mu +- 6 stationary sd (spacing 0.0120, transition sd 0.25), the parameters known; the evidence is the product of the filter's normalising constants. Log evidence -49.1248.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, particle-filter, abc-smc, abc-rejection. Not applicable: ep (y[0]: the scale of 'y[0]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[1]: the scale of 'y[1]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[2]: the scale of 'y[2]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[3]: the scale of 'y[3]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[4]: the scale of 'y[4]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[5]: the scale of 'y[5]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[6]: the scale of 'y[6]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[7]: the scale of 'y[7]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[8]: the scale of 'y[8]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[9]: the scale of 'y[9]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[10]: the scale of 'y[10]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[11]: the scale of 'y[11]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[12]: the scale of 'y[12]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[13]: the scale of 'y[13]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[14]: the scale of 'y[14]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[15]: the scale of 'y[15]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[16]: the scale of 'y[16]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[17]: the scale of 'y[17]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[18]: the scale of 'y[18]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[19]: the scale of 'y[19]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[20]: the scale of 'y[20]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[21]: the scale of 'y[21]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[22]: the scale of 'y[22]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[23]: the scale of 'y[23]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[24]: the scale of 'y[24]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[25]: the scale of 'y[25]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[26]: the scale of 'y[26]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[27]: the scale of 'y[27]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[28]: the scale of 'y[28]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[29]: the scale of 'y[29]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)); vmp (y[0]: the scale of 'y[0]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[1]: the scale of 'y[1]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[2]: the scale of 'y[2]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[3]: the scale of 'y[3]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[4]: the scale of 'y[4]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[5]: the scale of 'y[5]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[6]: the scale of 'y[6]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[7]: the scale of 'y[7]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[8]: the scale of 'y[8]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[9]: the scale of 'y[9]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[10]: the scale of 'y[10]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[11]: the scale of 'y[11]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[12]: the scale of 'y[12]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[13]: the scale of 'y[13]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[14]: the scale of 'y[14]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[15]: the scale of 'y[15]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[16]: the scale of 'y[16]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[17]: the scale of 'y[17]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[18]: the scale of 'y[18]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[19]: the scale of 'y[19]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[20]: the scale of 'y[20]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[21]: the scale of 'y[21]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[22]: the scale of 'y[22]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[23]: the scale of 'y[23]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[24]: the scale of 'y[24]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[25]: the scale of 'y[25]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[26]: the scale of 'y[26]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[27]: the scale of 'y[27]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[28]: the scale of 'y[28]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only); y[29]: the scale of 'y[29]' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)); bp (belief propagation sums over finite discrete variables, and 'h' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); conjugate (no usable conjugate structure for h); gaussian (the scan 'h': the scale of emission 'y' depends on a state or a latent); enumeration (continuous latent nodes (h)); chain (the states of 'h' are not finite and discrete).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (601 steps) | 0 divergences; R-hat high: h[5], h[6], h[7], ...; ESS low: h[6], h[7], h[8], ... | a consistent sampler through the unrolled recurrence: agrees within Monte Carlo error |
| tempered-smc | reported | converged (4 steps) | weights ESS 272 | tempered particles over thirty correlated states: consistent, with a noisy tail |
| particle-filter | reported | completed (30 steps) | - | a bootstrap filter: unbiased evidence and good filtering marginals; the smoothing marginals are the genealogies of the final particles, thin toward the start |
| laplace | reported | converged (87 steps) | - | a Gaussian at the mode of the joint density: the log-variance posterior is close to Gaussian, the tails of the return likelihood are not |
| svi | reported | completed (800 steps) | - | a mean-field normal ignores the correlation between neighbouring states: spreads too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | h[0] | -0.000359 | 0.557 | 0.00493 | 0.0764 | 1.03 | 0.0257 | 0.041 | 0.0143 | 253 |
| nuts | h[1] | 0.0619 | 0.542 | 0.00628 | 0.0933 | 1.04 | 0.025 | 0.0393 | 0.0105 | 240 |
| nuts | h[2] | 0.139 | 0.508 | -0.0186 | -0.282 | 1.03 | 0.0257 | 0.0378 | 0.0205 | 243 |
| nuts | h[3] | 0.267 | 0.47 | -0.0163 | -0.246 | 1.02 | 0.0306 | 0.0416 | 0.0137 | 238 |
| nuts | h[4] | 0.399 | 0.45 | -0.0041 | -0.0588 | 1.04 | 0.0326 | 0.0467 | 0.0177 | 223 |
| nuts | h[5] | 0.551 | 0.404 | -0.0107 | -0.149 | 1.02 | 0.0221 | 0.0295 | 0.0118 | 204 |
| nuts | h[6] | 0.648 | 0.388 | -0.0301 | -0.399 | 1.03 | 0.0355 | 0.0496 | 0.0151 | 188 |
| nuts | h[7] | 0.751 | 0.368 | -0.00711 | -0.0998 | 1.04 | 0.0303 | 0.049 | 0.0176 | 212 |
| nuts | h[8] | 0.693 | 0.373 | -0.00839 | -0.122 | 0.995 | 0.0454 | 0.0625 | 0.0116 | 210 |
| nuts | h[9] | 0.656 | 0.387 | -0.00323 | -0.0464 | 1.01 | 0.0354 | 0.0569 | 0.0117 | 210 |
| nuts | h[10] | 0.654 | 0.385 | 0.00546 | 0.0844 | 1.03 | 0.0271 | 0.0481 | 0.0105 | 253 |
| nuts | h[11] | 0.573 | 0.391 | 0.0126 | 0.206 | 1 | 0.0293 | 0.0341 | 0.0202 | 270 |
| nuts | h[12] | 0.513 | 0.385 | 0.061 | 1.05 | 0.967 | 0.0446 | 0.0762 | 0.019 | 279 |
| nuts | h[13] | 0.426 | 0.38 | 0.0458 | 0.805 | 0.934 | 0.0492 | 0.0751 | 0.0213 | 269 |
| nuts | h[14] | 0.36 | 0.375 | 0.0087 | 0.152 | 0.931 | 0.0321 | 0.0545 | 0.0262 | 264 |
| nuts | h[15] | 0.303 | 0.373 | -0.00142 | -0.0252 | 0.937 | 0.0375 | 0.0566 | 0.0271 | 278 |
| nuts | h[16] | 0.257 | 0.361 | -0.0207 | -0.371 | 0.934 | 0.0377 | 0.0517 | 0.0212 | 280 |
| nuts | h[17] | 0.11 | 0.379 | -0.0467 | -0.712 | 0.932 | 0.0365 | 0.0655 | 0.0189 | 202 |
| nuts | h[18] | -0.00749 | 0.378 | -0.051 | -0.749 | 0.903 | 0.045 | 0.0781 | 0.0222 | 176 |
| nuts | h[19] | -0.0767 | 0.388 | -0.0179 | -0.249 | 0.93 | 0.0356 | 0.0698 | 0.0137 | 168 |
| nuts | h[20] | -0.108 | 0.386 | 0.0308 | 0.418 | 0.955 | 0.0444 | 0.0611 | 0.0122 | 168 |
| nuts | h[21] | -0.244 | 0.396 | 0.0496 | 0.745 | 0.928 | 0.0608 | 0.0987 | 0.0229 | 194 |
| nuts | h[22] | -0.376 | 0.428 | 0.0568 | 0.792 | 0.958 | 0.0701 | 0.0753 | 0.0255 | 179 |
| nuts | h[23] | -0.469 | 0.427 | 0.0766 | 1.05 | 0.936 | 0.0631 | 0.103 | 0.0249 | 165 |
| nuts | h[24] | -0.524 | 0.455 | 0.119 | 1.54 | 0.994 | 0.0644 | 0.123 | 0.0163 | 163 |
| nuts | h[25] | -0.641 | 0.498 | 0.141 | 2.05 | 1.02 | 0.0744 | 0.148 | 0.0321 | 216 |
| nuts | h[26] | -0.736 | 0.547 | 0.165 | 2.41 | 1.05 | 0.0741 | 0.168 | 0.0301 | 237 |
| nuts | h[27] | -0.833 | 0.584 | 0.136 | 1.82 | 1.06 | 0.0798 | 0.141 | 0.039 | 202 |
| nuts | h[28] | -0.874 | 0.595 | 0.157 | 2.35 | 1.04 | 0.0838 | 0.161 | 0.0404 | 241 |
| nuts | h[29] | -0.895 | 0.608 | 0.161 | 2.39 | 1.02 | 0.0742 | 0.162 | 0.0157 | 228 |
| tempered-smc | h[0] | 0.125 | 0.484 | 0.237 | 5.93 | 0.893 | 0.139 | 0.243 | 0.0612 | 500 |
| tempered-smc | h[1] | 0.171 | 0.485 | 0.217 | 5.19 | 0.935 | 0.134 | 0.221 | 0.0546 | 500 |
| tempered-smc | h[2] | 0.185 | 0.477 | 0.0745 | 1.72 | 0.967 | 0.0618 | 0.0761 | 0.0126 | 500 |
| tempered-smc | h[3] | 0.342 | 0.438 | 0.146 | 3.43 | 0.953 | 0.0898 | 0.153 | 0.0312 | 500 |
| tempered-smc | h[4] | 0.441 | 0.399 | 0.0925 | 2.24 | 0.924 | 0.0825 | 0.123 | 0.0295 | 500 |
| tempered-smc | h[5] | 0.523 | 0.35 | -0.0826 | -2.08 | 0.887 | 0.0706 | 0.118 | 0.03 | 500 |
| tempered-smc | h[6] | 0.572 | 0.339 | -0.234 | -5.79 | 0.902 | 0.116 | 0.238 | 0.0573 | 500 |
| tempered-smc | h[7] | 0.732 | 0.335 | -0.0607 | -1.43 | 0.947 | 0.0501 | 0.0744 | 0.024 | 500 |
| tempered-smc | h[8] | 0.609 | 0.366 | -0.234 | -5.35 | 0.977 | 0.118 | 0.238 | 0.0458 | 500 |
| tempered-smc | h[9] | 0.597 | 0.349 | -0.158 | -3.88 | 0.91 | 0.0883 | 0.159 | 0.0385 | 500 |
| tempered-smc | h[10] | 0.622 | 0.366 | -0.0792 | -1.81 | 0.977 | 0.0446 | 0.0854 | 0.0209 | 500 |
| tempered-smc | h[11] | 0.518 | 0.374 | -0.13 | -3.02 | 0.963 | 0.0606 | 0.136 | 0.0279 | 500 |
| tempered-smc | h[12] | 0.433 | 0.372 | -0.138 | -3.31 | 0.933 | 0.0723 | 0.138 | 0.0254 | 500 |
| tempered-smc | h[13] | 0.351 | 0.342 | -0.139 | -3.7 | 0.84 | 0.106 | 0.184 | 0.059 | 500 |
| tempered-smc | h[14] | 0.378 | 0.365 | 0.0539 | 1.33 | 0.906 | 0.0621 | 0.0942 | 0.0217 | 500 |
| tempered-smc | h[15] | 0.342 | 0.399 | 0.0963 | 2.15 | 1 | 0.0589 | 0.1 | 0.048 | 500 |
| tempered-smc | h[16] | 0.257 | 0.396 | -0.0222 | -0.484 | 1.03 | 0.0481 | 0.0623 | 0.0221 | 500 |
| tempered-smc | h[17] | 0.105 | 0.415 | -0.0589 | -1.29 | 1.02 | 0.033 | 0.0623 | 0.0158 | 500 |
| tempered-smc | h[18] | 0.0292 | 0.425 | 0.0368 | 0.81 | 1.01 | 0.0314 | 0.0532 | 0.0151 | 500 |
| tempered-smc | h[19] | -0.0298 | 0.414 | 0.0945 | 2.13 | 0.991 | 0.0482 | 0.0964 | 0.0262 | 500 |
| tempered-smc | h[20] | -0.11 | 0.411 | 0.0253 | 0.555 | 1.02 | 0.0285 | 0.0449 | 0.0118 | 500 |
| tempered-smc | h[21] | -0.248 | 0.397 | 0.0421 | 1.01 | 0.931 | 0.0493 | 0.0831 | 0.0155 | 500 |
| tempered-smc | h[22] | -0.407 | 0.426 | -0.0113 | -0.265 | 0.954 | 0.0379 | 0.0523 | 0.0224 | 500 |
| tempered-smc | h[23] | -0.495 | 0.427 | 0.0176 | 0.421 | 0.937 | 0.04 | 0.0559 | 0.02 | 500 |
| tempered-smc | h[24] | -0.616 | 0.432 | -0.0828 | -1.96 | 0.943 | 0.0664 | 0.0933 | 0.0201 | 500 |
| tempered-smc | h[25] | -0.783 | 0.48 | -0.148 | -3.38 | 0.979 | 0.0748 | 0.149 | 0.0255 | 500 |
| tempered-smc | h[26] | -0.891 | 0.526 | -0.133 | -2.94 | 1.01 | 0.0664 | 0.133 | 0.0268 | 500 |
| tempered-smc | h[27] | -0.928 | 0.555 | -0.0355 | -0.784 | 1.01 | 0.0393 | 0.0625 | 0.0263 | 500 |
| tempered-smc | h[28] | -0.954 | 0.563 | 0.0181 | 0.412 | 0.981 | 0.0323 | 0.0354 | 0.0177 | 500 |
| tempered-smc | h[29] | -1.04 | 0.599 | -0.0888 | -1.98 | 1 | 0.0765 | 0.0915 | 0.0292 | 500 |
| particle-filter | h[0] | -0.015 | 0.561 | -0.0221 | -0.956 | 1.04 | 0.055 | 0.0735 | 0.164 | 2000 |
| particle-filter | h[1] | 0.017 | 0.483 | -0.0801 | -3.85 | 0.931 | 0.0545 | 0.0977 | 0.142 | 2000 |
| particle-filter | h[2] | 0.121 | 0.457 | -0.0552 | -2.66 | 0.928 | 0.0704 | 0.0869 | 0.126 | 2000 |
| particle-filter | h[3] | 0.266 | 0.471 | -0.0195 | -0.85 | 1.03 | 0.0745 | 0.0928 | 0.144 | 2000 |
| particle-filter | h[4] | 0.394 | 0.473 | -0.0154 | -0.63 | 1.1 | 0.118 | 0.166 | 0.139 | 2000 |
| particle-filter | h[5] | 0.566 | 0.386 | 0.0272 | 1.24 | 0.978 | 0.0903 | 0.0941 | 0.32 | 2000 |
| particle-filter | h[6] | 0.682 | 0.373 | 0.0594 | 2.68 | 0.993 | 0.079 | 0.084 | 0.0659 | 2000 |
| particle-filter | h[7] | 0.763 | 0.342 | 0.0281 | 1.3 | 0.966 | 0.0548 | 0.0616 | 0.0614 | 2000 |
| particle-filter | h[8] | 0.713 | 0.364 | 0.0452 | 2.08 | 0.97 | 0.0497 | 0.0605 | 0.0238 | 2000 |
| particle-filter | h[9] | 0.681 | 0.377 | 0.0619 | 2.82 | 0.983 | 0.0393 | 0.0665 | 0.0203 | 2000 |
| particle-filter | h[10] | 0.668 | 0.361 | 0.042 | 1.95 | 0.963 | 0.04 | 0.0563 | 0.0154 | 2000 |
| particle-filter | h[11] | 0.581 | 0.379 | 0.0337 | 1.55 | 0.975 | 0.0415 | 0.05 | 0.0323 | 2000 |
| particle-filter | h[12] | 0.523 | 0.375 | 0.0871 | 4.14 | 0.94 | 0.0806 | 0.1 | 0.0445 | 2000 |
| particle-filter | h[13] | 0.419 | 0.381 | 0.0286 | 1.37 | 0.937 | 0.0469 | 0.0615 | 0.03 | 2000 |
| particle-filter | h[14] | 0.371 | 0.38 | 0.0362 | 1.71 | 0.944 | 0.0334 | 0.0562 | 0.0226 | 2000 |
| particle-filter | h[15] | 0.306 | 0.38 | 0.00792 | 0.371 | 0.954 | 0.0299 | 0.0523 | 0.0196 | 2000 |
| particle-filter | h[16] | 0.267 | 0.372 | 0.00536 | 0.249 | 0.963 | 0.0294 | 0.0461 | 0.0154 | 2000 |
| particle-filter | h[17] | 0.135 | 0.393 | 0.0153 | 0.709 | 0.966 | 0.0423 | 0.0592 | 0.0217 | 2000 |
| particle-filter | h[18] | 0.00429 | 0.411 | -0.0229 | -1.04 | 0.983 | 0.0408 | 0.0483 | 0.0171 | 2000 |
| particle-filter | h[19] | -0.085 | 0.399 | -0.0379 | -1.77 | 0.955 | 0.0386 | 0.063 | 0.0166 | 2000 |
| particle-filter | h[20] | -0.144 | 0.397 | -0.0583 | -2.65 | 0.983 | 0.0485 | 0.0678 | 0.0256 | 2000 |
| particle-filter | h[21] | -0.296 | 0.412 | -0.0715 | -3.31 | 0.966 | 0.0588 | 0.0815 | 0.0195 | 2000 |
| particle-filter | h[22] | -0.424 | 0.437 | -0.0499 | -2.28 | 0.979 | 0.0574 | 0.0724 | 0.0284 | 2000 |
| particle-filter | h[23] | -0.516 | 0.459 | -0.0269 | -1.19 | 1.01 | 0.0308 | 0.0435 | 0.00726 | 2000 |
| particle-filter | h[24] | -0.577 | 0.458 | 0.00404 | 0.181 | 1 | 0.0174 | 0.0251 | 0.0073 | 2000 |
| particle-filter | h[25] | -0.718 | 0.487 | -0.0152 | -0.683 | 0.993 | 0.0281 | 0.04 | 0.00703 | 2000 |
| particle-filter | h[26] | -0.838 | 0.529 | -0.031 | -1.37 | 1.02 | 0.0257 | 0.0376 | 0.00761 | 2000 |
| particle-filter | h[27] | -0.922 | 0.567 | -0.0257 | -1.11 | 1.03 | 0.0308 | 0.0355 | 0.00999 | 2000 |
| particle-filter | h[28] | -0.987 | 0.592 | -0.0392 | -1.7 | 1.03 | 0.0268 | 0.0459 | 0.00831 | 2000 |
| particle-filter | h[29] | -1 | 0.606 | -0.0192 | -0.847 | 1.02 | 0.0327 | 0.0361 | 0.0119 | 2000 |
| laplace | h[0] | -0.0615 | 0.55 | -0.108 | -6.72 | 1.02 | 0.047 | 0.108 | 0.00846 | 4000 |
| laplace | h[1] | 0.00303 | 0.529 | -0.107 | -6.64 | 1.02 | 0.0489 | 0.107 | 0.00755 | 4000 |
| laplace | h[2] | 0.0973 | 0.495 | -0.104 | -6.56 | 1 | 0.0446 | 0.104 | 0.00694 | 4000 |
| laplace | h[3] | 0.223 | 0.456 | -0.112 | -7.13 | 0.993 | 0.045 | 0.112 | 0.00842 | 4000 |
| laplace | h[4] | 0.349 | 0.428 | -0.121 | -7.71 | 0.992 | 0.0471 | 0.121 | 0.00854 | 4000 |
| laplace | h[5] | 0.504 | 0.391 | -0.13 | -8.28 | 0.99 | 0.0481 | 0.13 | 0.00989 | 4000 |
| laplace | h[6] | 0.608 | 0.376 | -0.138 | -8.72 | 1 | 0.0523 | 0.139 | 0.011 | 4000 |
| laplace | h[7] | 0.699 | 0.354 | -0.152 | -9.63 | 1 | 0.0522 | 0.152 | 0.0142 | 4000 |
| laplace | h[8] | 0.639 | 0.364 | -0.152 | -9.89 | 0.971 | 0.0581 | 0.152 | 0.0144 | 4000 |
| laplace | h[9] | 0.594 | 0.374 | -0.165 | -10.7 | 0.975 | 0.0685 | 0.165 | 0.0165 | 4000 |
| laplace | h[10] | 0.594 | 0.371 | -0.156 | -9.97 | 0.99 | 0.06 | 0.156 | 0.0165 | 4000 |
| laplace | h[11] | 0.51 | 0.379 | -0.15 | -9.76 | 0.975 | 0.0596 | 0.151 | 0.0127 | 4000 |
| laplace | h[12] | 0.423 | 0.389 | -0.164 | -10.6 | 0.977 | 0.0618 | 0.164 | 0.0146 | 4000 |
| laplace | h[13] | 0.341 | 0.401 | -0.163 | -10.5 | 0.987 | 0.0694 | 0.163 | 0.0154 | 4000 |
| laplace | h[14] | 0.293 | 0.397 | -0.159 | -10.2 | 0.986 | 0.0604 | 0.159 | 0.0167 | 4000 |
| laplace | h[15] | 0.233 | 0.403 | -0.178 | -11.1 | 1.01 | 0.066 | 0.178 | 0.0167 | 4000 |
| laplace | h[16] | 0.194 | 0.389 | -0.185 | -11.6 | 1.01 | 0.0639 | 0.186 | 0.0194 | 4000 |
| laplace | h[17] | 0.058 | 0.414 | -0.175 | -10.9 | 1.02 | 0.065 | 0.175 | 0.0165 | 4000 |
| laplace | h[18] | -0.0531 | 0.421 | -0.16 | -10.1 | 1.01 | 0.0614 | 0.16 | 0.0165 | 4000 |
| laplace | h[19] | -0.137 | 0.421 | -0.163 | -10.3 | 1.01 | 0.0641 | 0.163 | 0.0167 | 4000 |
| laplace | h[20] | -0.189 | 0.404 | -0.17 | -10.8 | 1 | 0.0635 | 0.17 | 0.0165 | 4000 |
| laplace | h[21] | -0.344 | 0.427 | -0.183 | -11.6 | 1 | 0.0714 | 0.183 | 0.0186 | 4000 |
| laplace | h[22] | -0.477 | 0.446 | -0.169 | -10.7 | 0.999 | 0.0651 | 0.169 | 0.0177 | 4000 |
| laplace | h[23] | -0.577 | 0.455 | -0.162 | -10.3 | 0.998 | 0.0609 | 0.162 | 0.0157 | 4000 |
| laplace | h[24] | -0.656 | 0.461 | -0.169 | -10.6 | 1.01 | 0.0577 | 0.169 | 0.0166 | 4000 |
| laplace | h[25] | -0.785 | 0.492 | -0.152 | -9.58 | 1 | 0.0593 | 0.152 | 0.014 | 4000 |
| laplace | h[26] | -0.893 | 0.521 | -0.137 | -8.65 | 1 | 0.0544 | 0.137 | 0.00953 | 4000 |
| laplace | h[27] | -0.978 | 0.553 | -0.127 | -7.95 | 1.01 | 0.0509 | 0.127 | 0.00907 | 4000 |
| laplace | h[28] | -1.03 | 0.579 | -0.109 | -6.82 | 1.01 | 0.0476 | 0.11 | 0.00717 | 4000 |
| laplace | h[29] | -1.05 | 0.603 | -0.0987 | -6.17 | 1.01 | 0.0422 | 0.0992 | 0.00684 | 4000 |
| svi | h[0] | -0.0801 | 0.222 | -0.142 | -22 | 0.409 | 0.252 | 0.482 | 1.11 | 4000 |
| svi | h[1] | 0.0698 | 0.188 | 0.0215 | 3.75 | 0.363 | 0.24 | 0.508 | 1.43 | 4000 |
| svi | h[2] | 0.0444 | 0.188 | -0.211 | -35.1 | 0.381 | 0.285 | 0.519 | 1.39 | 4000 |
| svi | h[3] | 0.246 | 0.19 | -0.0626 | -9.55 | 0.415 | 0.221 | 0.469 | 1.08 | 4000 |
| svi | h[4] | 0.353 | 0.154 | -0.112 | -19.8 | 0.357 | 0.263 | 0.518 | 1.42 | 4000 |
| svi | h[5] | 0.496 | 0.163 | -0.149 | -22.8 | 0.414 | 0.247 | 0.478 | 1.08 | 4000 |
| svi | h[6] | 0.578 | 0.175 | -0.218 | -29.5 | 0.466 | 0.243 | 0.453 | 0.876 | 4000 |
| svi | h[7] | 0.735 | 0.202 | -0.0528 | -5.85 | 0.57 | 0.148 | 0.342 | 0.374 | 4000 |
| svi | h[8] | 0.632 | 0.171 | -0.17 | -23.7 | 0.455 | 0.238 | 0.448 | 0.918 | 4000 |
| svi | h[9] | 0.642 | 0.192 | -0.0382 | -4.82 | 0.501 | 0.175 | 0.397 | 0.634 | 4000 |
| svi | h[10] | 0.597 | 0.168 | -0.147 | -20.6 | 0.45 | 0.232 | 0.451 | 0.895 | 4000 |
| svi | h[11] | 0.515 | 0.169 | -0.137 | -19.8 | 0.436 | 0.233 | 0.461 | 0.88 | 4000 |
| svi | h[12] | 0.425 | 0.15 | -0.159 | -26.7 | 0.378 | 0.274 | 0.511 | 1.28 | 4000 |
| svi | h[13] | 0.331 | 0.194 | -0.188 | -24.9 | 0.476 | 0.232 | 0.44 | 0.778 | 4000 |
| svi | h[14] | 0.386 | 0.185 | 0.0737 | 10.1 | 0.46 | 0.213 | 0.436 | 0.814 | 4000 |
| svi | h[15] | 0.215 | 0.193 | -0.221 | -28.9 | 0.484 | 0.24 | 0.443 | 0.812 | 4000 |
| svi | h[16] | 0.231 | 0.181 | -0.0892 | -12 | 0.469 | 0.201 | 0.425 | 0.77 | 4000 |
| svi | h[17] | 0.146 | 0.176 | 0.0406 | 5.95 | 0.432 | 0.212 | 0.451 | 1.08 | 4000 |
| svi | h[18] | -0.0462 | 0.177 | -0.144 | -21.5 | 0.423 | 0.248 | 0.469 | 1.09 | 4000 |
| svi | h[19] | -0.132 | 0.175 | -0.15 | -22.6 | 0.419 | 0.246 | 0.477 | 1.06 | 4000 |
| svi | h[20] | -0.126 | 0.175 | -0.0153 | -2.24 | 0.432 | 0.198 | 0.452 | 1.01 | 4000 |
| svi | h[21] | -0.327 | 0.166 | -0.143 | -23.2 | 0.389 | 0.259 | 0.496 | 1.3 | 4000 |
| svi | h[22] | -0.437 | 0.208 | -0.0778 | -10.6 | 0.466 | 0.198 | 0.427 | 0.911 | 4000 |
| svi | h[23] | -0.545 | 0.2 | -0.0921 | -13.3 | 0.44 | 0.216 | 0.45 | 0.945 | 4000 |
| svi | h[24] | -0.58 | 0.195 | -0.0027 | -0.402 | 0.426 | 0.204 | 0.458 | 0.979 | 4000 |
| svi | h[25] | -0.748 | 0.165 | -0.0773 | -14.5 | 0.337 | 0.262 | 0.53 | 1.58 | 4000 |
| svi | h[26] | -0.802 | 0.179 | 0.0382 | 7.05 | 0.343 | 0.259 | 0.526 | 1.55 | 4000 |
| svi | h[27] | -0.891 | 0.186 | 0.0308 | 5.74 | 0.339 | 0.258 | 0.529 | 1.57 | 4000 |
| svi | h[28] | -0.995 | 0.181 | -0.0531 | -10.6 | 0.316 | 0.272 | 0.546 | 1.8 | 4000 |
| svi | h[29] | -1.03 | 0.245 | -0.0634 | -9.75 | 0.412 | 0.22 | 0.471 | 1.14 | 4000 |

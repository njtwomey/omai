# benchmark/gaussian-process-non-centred

Gaussian process regression, non-centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 4, rows 52, scalePercent 100), 17.2 s.

**Passed**: 1 method(s) agree with the quadrature truth; 3 reported

**Gaussian process regression, non-centred** (gaussian-process-non-centred): gaussian-process, kernel, regression, non-centred, custom, quadrature

The Gaussian process regression written as f = L z with a whitened z: the geometry the sampler sees no longer depends on the length scale, and NUTS agrees with the quadrature truth.

Truth (quadrature): quadrature over the length scale (321 points, evidence error 1.8e-15) of the marginal likelihood N(y; 0, K + sigma² I) with the function integrated out; each function value is the grid mixture of its exact Gaussian conditionals (mean K (K + sigma² I)^-1 y). Log evidence -11.8396.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: ep (expectation propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'f' is opaque to message passing (a custom call, or a family with no exponential-family form)); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for ell, z); gaussian ('ell' is logNormal, not Gaussian); enumeration (continuous latent nodes (ell, z)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (401 steps) | 0 divergences; R-hat high: ell, z[2], z[3], ...; ESS low: ell, z[5], z[6] | whitened, so no funnel: agrees within Monte Carlo error |
| tempered-smc | reported | converged (7 steps) | weights ESS 924 | tempered particles in thirteen dimensions: consistent, with a noisy tail |
| laplace | reported | converged (77 steps) | - | a Gaussian at the mode of the whitened density: close for the function, the length scale is skewed on the log scale |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the whitened coordinates: the posterior of z is nearly independent, so the spreads are close |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | ell | 0.944 | 0.245 | -0.125 | -1.41 | 0.953 | 0.0718 | 0.145 | 0.0338 | 115 |
| nuts | f[0] | -0.614 | 0.199 | 0.0299 | 0.54 | 0.976 | 0.045 | 0.058 | 0.0331 | 311 |
| nuts | f[1] | -0.649 | 0.175 | 0.0239 | 0.464 | 0.997 | 0.0371 | 0.0491 | 0.0177 | 374 |
| nuts | f[2] | -0.715 | 0.197 | 0.0171 | 0.396 | 1.03 | 0.0517 | 0.0623 | 0.0325 | 565 |
| nuts | f[3] | -1.04 | 0.237 | 0.00679 | 0.134 | 1.07 | 0.0409 | 0.0681 | 0.0225 | 444 |
| nuts | f[4] | -1.24 | 0.26 | 0.00929 | 0.185 | 1.06 | 0.0376 | 0.067 | 0.0374 | 447 |
| nuts | f[5] | 0.485 | 0.251 | 0.0291 | 0.53 | 0.986 | 0.0332 | 0.0419 | 0.0188 | 324 |
| nuts | f[6] | 1.15 | 0.198 | 0.0212 | 0.375 | 1.04 | 0.0289 | 0.0494 | 0.0226 | 341 |
| nuts | f[7] | 1.56 | 0.203 | 0.00613 | 0.0899 | 0.97 | 0.0254 | 0.0355 | 0.0189 | 203 |
| nuts | f[8] | 1.51 | 0.196 | -0.0315 | -0.594 | 0.974 | 0.0415 | 0.0513 | 0.0239 | 337 |
| nuts | f[9] | 0.951 | 0.248 | -0.0473 | -0.999 | 1.03 | 0.0411 | 0.0569 | 0.0189 | 472 |
| nuts | f[10] | 0.175 | 0.214 | 0.0139 | 0.282 | 1.07 | 0.0374 | 0.0717 | 0.0336 | 469 |
| nuts | f[11] | 0.161 | 0.219 | 0.0113 | 0.233 | 1.07 | 0.0354 | 0.0729 | 0.0419 | 489 |
| tempered-smc | ell | 0.971 | 0.203 | -0.0171 | -0.684 | 0.79 | 0.0931 | 0.209 | 0.102 | 1000 |
| tempered-smc | f[0] | -0.63 | 0.19 | -0.0506 | -1.72 | 0.931 | 0.0427 | 0.0774 | 0.0206 | 1000 |
| tempered-smc | f[1] | -0.652 | 0.17 | 0.00323 | 0.106 | 0.965 | 0.0188 | 0.0335 | 0.0127 | 1000 |
| tempered-smc | f[2] | -0.709 | 0.184 | 0.0496 | 1.64 | 0.955 | 0.0378 | 0.0573 | 0.00893 | 1000 |
| tempered-smc | f[3] | -1.04 | 0.196 | 0.00615 | 0.219 | 0.888 | 0.0383 | 0.095 | 0.0187 | 1000 |
| tempered-smc | f[4] | -1.26 | 0.234 | -0.0898 | -2.97 | 0.957 | 0.0533 | 0.0959 | 0.0187 | 1000 |
| tempered-smc | f[5] | 0.469 | 0.242 | -0.0353 | -1.17 | 0.953 | 0.0287 | 0.0547 | 0.0116 | 1000 |
| tempered-smc | f[6] | 1.13 | 0.188 | -0.0847 | -2.7 | 0.994 | 0.0397 | 0.0852 | 0.012 | 1000 |
| tempered-smc | f[7] | 1.55 | 0.204 | -0.0507 | -1.64 | 0.975 | 0.0351 | 0.059 | 0.00739 | 1000 |
| tempered-smc | f[8] | 1.52 | 0.202 | 0.0445 | 1.4 | 1 | 0.0369 | 0.0458 | 0.0105 | 1000 |
| tempered-smc | f[9] | 0.958 | 0.23 | -0.0163 | -0.54 | 0.954 | 0.0431 | 0.0605 | 0.0109 | 1000 |
| tempered-smc | f[10] | 0.174 | 0.192 | 0.012 | 0.396 | 0.956 | 0.0242 | 0.0371 | 0.0121 | 1000 |
| tempered-smc | f[11] | 0.163 | 0.194 | 0.0211 | 0.701 | 0.95 | 0.0318 | 0.045 | 0.015 | 1000 |
| laplace | ell | 0.781 | 0.302 | -0.756 | -40.7 | 1.17 | 0.341 | 0.774 | 0.314 | 4000 |
| laplace | f[0] | -0.632 | 0.216 | -0.0591 | -3.53 | 1.06 | 0.0365 | 0.0728 | 0.00673 | 4000 |
| laplace | f[1] | -0.655 | 0.182 | -0.0152 | -0.929 | 1.04 | 0.0199 | 0.0348 | 0.00425 | 4000 |
| laplace | f[2] | -0.666 | 0.242 | 0.276 | 13.8 | 1.26 | 0.108 | 0.287 | 0.0401 | 4000 |
| laplace | f[3] | -0.999 | 0.293 | 0.181 | 8.61 | 1.33 | 0.102 | 0.289 | 0.0515 | 4000 |
| laplace | f[4] | -1.25 | 0.35 | -0.0679 | -3.01 | 1.43 | 0.0696 | 0.273 | 0.0447 | 4000 |
| laplace | f[5] | 0.138 | 0.426 | -1.34 | -50.5 | 1.68 | 0.382 | 1.34 | 0.412 | 4000 |
| laplace | f[6] | 0.982 | 0.441 | -0.862 | -23.4 | 2.32 | 0.198 | 0.892 | 0.149 | 4000 |
| laplace | f[7] | 1.56 | 0.451 | 0.015 | 0.439 | 2.16 | 0.143 | 0.722 | 0.145 | 4000 |
| laplace | f[8] | 1.54 | 0.34 | 0.145 | 5.42 | 1.69 | 0.11 | 0.435 | 0.0705 | 4000 |
| laplace | f[9] | 0.961 | 0.352 | -0.00373 | -0.162 | 1.46 | 0.0838 | 0.334 | 0.0704 | 4000 |
| laplace | f[10] | 0.149 | 0.384 | -0.114 | -3.78 | 1.91 | 0.113 | 0.594 | 0.129 | 4000 |
| laplace | f[11] | 0.139 | 0.387 | -0.0977 | -3.26 | 1.89 | 0.109 | 0.583 | 0.127 | 4000 |
| svi | ell | 0.716 | 0.0844 | -1.01 | -195 | 0.328 | 0.609 | 1.03 | 2.46 | 4000 |
| svi | f[0] | -0.602 | 0.15 | 0.0884 | 7.6 | 0.735 | 0.0988 | 0.221 | 0.107 | 4000 |
| svi | f[1] | -0.646 | 0.159 | 0.0359 | 2.51 | 0.904 | 0.036 | 0.0772 | 0.0111 | 4000 |
| svi | f[2] | -0.732 | 0.213 | -0.0682 | -3.9 | 1.11 | 0.0507 | 0.104 | 0.0157 | 4000 |
| svi | f[3] | -1.13 | 0.331 | -0.41 | -17.3 | 1.5 | 0.191 | 0.514 | 0.145 | 4000 |
| svi | f[4] | -1.35 | 0.399 | -0.446 | -17.3 | 1.63 | 0.213 | 0.624 | 0.187 | 4000 |
| svi | f[5] | 0.395 | 0.241 | -0.327 | -21.8 | 0.948 | 0.138 | 0.327 | 0.0707 | 4000 |
| svi | f[6] | 1.18 | 0.25 | 0.173 | 8.33 | 1.32 | 0.118 | 0.303 | 0.0663 | 4000 |
| svi | f[7] | 1.65 | 0.254 | 0.465 | 24.2 | 1.21 | 0.193 | 0.478 | 0.109 | 4000 |
| svi | f[8] | 1.55 | 0.278 | 0.186 | 8.53 | 1.38 | 0.135 | 0.352 | 0.0894 | 4000 |
| svi | f[9] | 0.892 | 0.356 | -0.289 | -12.4 | 1.48 | 0.156 | 0.46 | 0.121 | 4000 |
| svi | f[10] | 0.234 | 0.31 | 0.307 | 12.6 | 1.54 | 0.16 | 0.485 | 0.133 | 4000 |
| svi | f[11] | 0.224 | 0.31 | 0.322 | 13.4 | 1.51 | 0.162 | 0.476 | 0.128 | 4000 |

# benchmark/lda-lite

LDA on a tiny corpus: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 4, rows 59, scalePercent 100), 42.1 s.

**Passed**: 2 method(s) agree with the exact truth; 2 reported

**LDA on a tiny corpus** (lda-lite): topic-model, lda, dirichlet, discrete-latent, plates, exact, enumeration

Latent Dirichlet allocation with three documents, two topics and four words: Dirichlet and categorical plates whose exact posterior comes from enumerating the topic assignments.

Truth (exact): enumeration of the 2^15 topic assignments with the mixtures and topics integrated out (Dirichlet-multinomial marginals, written out here): the evidence is their sum, each assignment marginal a weighted count, and each mixture or topic probability a mixture of Betas over the counts. Log evidence -22.6532.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, abc-rejection. Not applicable: ep (the factor graph does not compile: familyOf: dirichlet needs a size); vmp (the factor graph does not compile: familyOf: dirichlet needs a size); bp (belief propagation sums over finite discrete variables, and 'theta', 'phi' are continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for theta, z); gaussian ('theta' is dirichlet, not Gaussian); enumeration (continuous latent nodes (theta, phi)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (2001 steps) | 0 divergences; R-hat high: theta[1], theta[2], phi[4], ...; ESS low: theta[0], theta[1], phi[4] | NUTS on the simplex-constrained mixtures and topics with the assignments summed out: consistent, within the main mode |
| gibbs | must agree | completed (2300 steps) | - | exact Dirichlet and categorical conditionals in a sweep: consistent, correlated between sweeps (the Kolmogorov distance of a two-valued assignment is not small for a correct sampler, so it is not checked) |
| laplace | reported | converged (31 steps) | - | a Gaussian at one mode of the unconstrained density, with the assignments summed out: the simplex coordinates are far from normal at this sample size |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the unconstrained simplex coordinates: biased means and spreads too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | theta[1] | 0.362 | 0.257 | -0.0422 | -0.697 | 1.02 | 0.0393 | 0.0493 | 0.014 | 285 |
| nuts | theta[3] | 0.415 | 0.32 | -0.0582 | -0.749 | 1.01 | 0.0389 | 0.0651 | 0.0133 | 169 |
| nuts | theta[5] | 0.755 | 0.234 | 0.0536 | 0.963 | 0.868 | 0.0438 | 0.0986 | 0.0202 | 244 |
| nuts | phi[0] | 0.394 | 0.154 | -0.0458 | -1.13 | 0.946 | 0.0312 | 0.0606 | 0.0168 | 549 |
| nuts | phi[1] | 0.366 | 0.157 | 0.023 | 0.515 | 1.01 | 0.0255 | 0.0371 | 0.0108 | 507 |
| nuts | phi[2] | 0.166 | 0.147 | 0.0765 | 1.09 | 0.982 | 0.0664 | 0.0813 | 0.0324 | 197 |
| nuts | phi[3] | 0.0743 | 0.11 | -0.0619 | -1.32 | 0.898 | 0.0585 | 0.0666 | 0.0517 | 367 |
| nuts | phi[4] | 0.05 | 0.072 | -0.036 | -0.832 | 0.913 | 0.0739 | 0.0409 | 0.0626 | 446 |
| nuts | phi[5] | 0.0787 | 0.0982 | -0.066 | -0.859 | 1.03 | 0.0769 | 0.0905 | 0.0677 | 179 |
| nuts | phi[6] | 0.341 | 0.173 | -0.0334 | -0.491 | 1 | 0.0353 | 0.0406 | 0.0165 | 217 |
| nuts | phi[7] | 0.531 | 0.183 | 0.0806 | 1.3 | 0.988 | 0.0389 | 0.0809 | 0.0221 | 255 |
| gibbs | z[0] | 0.758 | 0.428 | -0.00669 | -0.135 | 1 | 0.758 | 0.00469 | 4.56 | 411 |
| gibbs | z[1] | 0.117 | 0.321 | 0.00149 | 0.0293 | 1 | 0.883 | 0.00156 | 4.59 | 389 |
| gibbs | z[2] | 0.558 | 0.497 | -0.0441 | -0.72 | 1.01 | 0.558 | 0.0436 | 4.54 | 269 |
| gibbs | z[3] | 0.112 | 0.315 | -0.0157 | -0.346 | 0.981 | 0.883 | 0.0187 | 4.59 | 470 |
| gibbs | z[4] | 0.151 | 0.358 | -0.0267 | -0.704 | 0.975 | 0.84 | 0.0259 | 4.58 | 663 |
| gibbs | z[5] | 0.546 | 0.498 | -0.0733 | -0.932 | 1.01 | 0.546 | 0.073 | 4.54 | 165 |
| gibbs | z[6] | 0.562 | 0.496 | -0.0419 | -0.536 | 1.01 | 0.562 | 0.0416 | 4.54 | 166 |
| gibbs | z[7] | 0.557 | 0.497 | -0.051 | -0.684 | 1.01 | 0.557 | 0.0507 | 4.54 | 182 |
| gibbs | z[8] | 0.155 | 0.362 | 0.0138 | 0.245 | 1.01 | 0.85 | 0.014 | 4.58 | 323 |
| gibbs | z[9] | 0.191 | 0.393 | -0.033 | -0.682 | 0.975 | 0.796 | 0.036 | 4.57 | 407 |
| gibbs | z[10] | 0.861 | 0.345 | -0.0473 | -0.507 | 1.05 | 0.861 | 0.0487 | 4.58 | 127 |
| gibbs | z[11] | 0.864 | 0.343 | -0.0397 | -0.431 | 1.04 | 0.864 | 0.0411 | 4.58 | 129 |
| gibbs | z[12] | 0.865 | 0.341 | -0.0351 | -0.391 | 1.04 | 0.865 | 0.0365 | 4.58 | 134 |
| gibbs | z[13] | 0.869 | 0.337 | -0.0229 | -0.255 | 1.03 | 0.869 | 0.0244 | 4.58 | 130 |
| gibbs | z[14] | 0.403 | 0.491 | -0.0639 | -1.31 | 0.989 | 0.565 | 0.0646 | 4.55 | 412 |
| gibbs | theta[1] | 0.366 | 0.25 | -0.0259 | -0.525 | 0.995 | 0.0219 | 0.0301 | 0.00333 | 406 |
| gibbs | theta[3] | 0.413 | 0.314 | -0.0637 | -0.797 | 0.992 | 0.0348 | 0.0637 | 0.00469 | 154 |
| gibbs | theta[5] | 0.72 | 0.278 | -0.0776 | -0.981 | 1.03 | 0.0449 | 0.0776 | 0.0115 | 169 |
| gibbs | phi[0] | 0.397 | 0.158 | -0.0264 | -0.726 | 0.974 | 0.0224 | 0.0361 | 0.00396 | 714 |
| gibbs | phi[1] | 0.358 | 0.151 | -0.0268 | -0.916 | 0.968 | 0.0262 | 0.0394 | 0.00579 | 1100 |
| gibbs | phi[2] | 0.158 | 0.147 | 0.0268 | 0.404 | 0.984 | 0.0316 | 0.0352 | 0.00977 | 221 |
| gibbs | phi[3] | 0.0864 | 0.128 | 0.0366 | 0.543 | 1.05 | 0.0628 | 0.0384 | 0.0693 | 243 |
| gibbs | phi[4] | 0.0537 | 0.0824 | 0.012 | 0.294 | 1.04 | 0.0733 | 0.0229 | 0.0501 | 657 |
| gibbs | phi[5] | 0.0828 | 0.0938 | -0.0233 | -0.774 | 0.982 | 0.0445 | 0.0265 | 0.0271 | 1060 |
| gibbs | phi[6] | 0.35 | 0.178 | 0.0188 | 0.283 | 1.03 | 0.0179 | 0.0347 | 0.00358 | 242 |
| gibbs | phi[7] | 0.514 | 0.189 | -0.0106 | -0.145 | 1.02 | 0.0169 | 0.0231 | 0.00675 | 194 |
| laplace | theta[1] | 0.355 | 0.214 | -0.0713 | -5.28 | 0.853 | 0.0601 | 0.133 | 0.105 | 4000 |
| laplace | theta[3] | 0.298 | 0.28 | -0.427 | -30.6 | 0.884 | 0.213 | 0.429 | 0.154 | 4000 |
| laplace | theta[5] | 0.773 | 0.218 | 0.121 | 9.47 | 0.809 | 0.0778 | 0.148 | 0.19 | 4000 |
| laplace | phi[0] | 0.387 | 0.135 | -0.0911 | -6.95 | 0.829 | 0.0595 | 0.138 | 0.0424 | 4000 |
| laplace | phi[1] | 0.315 | 0.13 | -0.302 | -22.9 | 0.834 | 0.13 | 0.303 | 0.0736 | 4000 |
| laplace | phi[2] | 0.235 | 0.139 | 0.54 | 36.7 | 0.93 | 0.303 | 0.54 | 0.907 | 4000 |
| laplace | phi[3] | 0.0632 | 0.0822 | -0.153 | -14.4 | 0.673 | 0.156 | 0.197 | 0.276 | 4000 |
| laplace | phi[4] | 0.101 | 0.154 | 0.61 | 19.7 | 1.96 | 0.213 | 0.61 | 0.394 | 4000 |
| laplace | phi[5] | 0.145 | 0.149 | 0.626 | 25.4 | 1.56 | 0.224 | 0.626 | 0.718 | 4000 |
| laplace | phi[6] | 0.248 | 0.157 | -0.574 | -39.8 | 0.912 | 0.265 | 0.574 | 0.185 | 4000 |
| laplace | phi[7] | 0.507 | 0.184 | -0.0486 | -3.1 | 0.994 | 0.0275 | 0.0523 | 0.00888 | 4000 |
| svi | theta[1] | 0.348 | 0.207 | -0.0967 | -7.42 | 0.825 | 0.0684 | 0.17 | 0.135 | 4000 |
| svi | theta[3] | 0.552 | 0.248 | 0.373 | 30 | 0.784 | 0.223 | 0.388 | 0.701 | 4000 |
| svi | theta[5] | 0.844 | 0.166 | 0.385 | 39.5 | 0.616 | 0.185 | 0.393 | 0.339 | 4000 |
| svi | phi[0] | 0.451 | 0.142 | 0.301 | 21.8 | 0.874 | 0.154 | 0.315 | 0.0983 | 4000 |
| svi | phi[1] | 0.386 | 0.157 | 0.155 | 9.7 | 1.01 | 0.0843 | 0.165 | 0.0212 | 4000 |
| svi | phi[2] | 0.099 | 0.114 | -0.371 | -30.8 | 0.762 | 0.196 | 0.372 | 0.128 | 4000 |
| svi | phi[3] | 0.0641 | 0.0951 | -0.146 | -11.8 | 0.778 | 0.0632 | 0.145 | 0.0612 | 4000 |
| svi | phi[4] | 0.0426 | 0.0885 | -0.129 | -7.25 | 1.12 | 0.123 | 0.214 | 0.164 | 4000 |
| svi | phi[5] | 0.0796 | 0.0891 | -0.0563 | -3.82 | 0.933 | 0.195 | 0.179 | 0.738 | 4000 |
| svi | phi[6] | 0.386 | 0.136 | 0.232 | 18.5 | 0.791 | 0.152 | 0.278 | 0.177 | 4000 |
| svi | phi[7] | 0.491 | 0.144 | -0.132 | -10.7 | 0.778 | 0.102 | 0.207 | 0.148 | 4000 |

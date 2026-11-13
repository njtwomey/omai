# kernels/agreement-eight-schools-non-centred

Eight schools, non-centred: the kernel methods against the reference truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 12, rows 120, scalePercent 100), 35.3 s.

**Passed**: 4 kernel method(s) agree with the quadrature truth; 8 reported

**Eight schools, non-centred** (eight-schools-non-centred): hierarchical, non-centred, reparametrisation, quadrature, eight-schools

Rubin's eight schools with theta_j = mu + tau z_j: the same posterior as the centred hierarchy with the funnel removed.

Truth (quadrature): grid quadrature of the posterior of (mu, tau) with the school effects integrated out in closed form (121 points per axis, evidence error 1.5e-5); each theta_j is the grid mixture of its conditional normals. Log evidence -31.3113.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of tau is halfCauchy, not normal or multivariateNormal); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'mu', 'tau', 'z' are continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for mu, tau, z); gaussian ('tau' is halfCauchy, not Gaussian); enumeration (continuous latent nodes (mu, tau, z)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for mu, tau, z).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| rwm | reported | converged (6001 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | a consistent sampler, slower mixing: agrees within Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| mala | reported | converged (1101 steps) | 0 divergences; R-hat high: tau, z[6]; ESS low: tau, z[0], z[3], ... | Metropolis-adjusted, so exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| barker | reported | converged (1301 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | Metropolis-adjusted, so exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| hmc | must agree | converged (1201 steps) | 0 divergences; R-hat high: z[3], z[7] | a fixed trajectory with a tuned step size: exact up to Monte Carlo error |
| slice | must agree | converged (901 steps) | 0 divergences | coordinate-wise slice sampling: exact up to Monte Carlo error |
| aies | reported | converged (701 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | an affine-invariant ensemble: exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| ensemble-slice | reported | converged (401 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | an ensemble of slice steps: exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| chees | must agree | converged (601 steps) | 0 divergences; R-hat high: mu; ESS low: mu, tau | jittered HMC with an adapted trajectory: exact up to Monte Carlo error |
| parallel-tempering | must agree | converged (901 steps) | 0 divergences | the cold replica of a tempered ladder: exact up to Monte Carlo error |
| mclmc | reported | converged (1601 steps) | 0 divergences; ESS low: tau | unadjusted: a small bias that shrinks with the step size |
| sgld | reported | converged (3201 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | unadjusted Langevin with a constant step: a step-size bias in the spread |
| sghmc | reported | converged (1301 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | friction-damped Hamiltonian dynamics with a constant step: a step-size bias |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rwm | mu | 3.94 | 2.03 | -0.137 | -3.06 | 0.611 | 0.242 | 0.418 | 0.359 | 186 |
| rwm | tau | 2.71 | 2.96 | -0.276 | -0.569 | 0.919 | 0.201 | 0.304 | 0.104 | 3.58 |
| rwm | theta[0] | 5.21 | 4.22 | -0.18 | -0.98 | 0.754 | 0.265 | 0.369 | 0.401 | 16.9 |
| rwm | theta[1] | 4.36 | 3.59 | -0.124 | -3.03 | 0.768 | 0.174 | 0.276 | 0.164 | 350 |
| rwm | theta[2] | 3.51 | 3.95 | -0.0791 | -1.73 | 0.752 | 0.209 | 0.332 | 0.301 | 271 |
| rwm | theta[3] | 3.25 | 3.5 | -0.316 | -6.11 | 0.733 | 0.275 | 0.337 | 0.281 | 201 |
| rwm | theta[4] | 3.05 | 3.46 | -0.121 | -2.42 | 0.744 | 0.222 | 0.307 | 0.328 | 223 |
| rwm | theta[5] | 3.81 | 3.48 | -0.0488 | -1.49 | 0.722 | 0.206 | 0.304 | 0.235 | 486 |
| rwm | theta[6] | 5.05 | 3.72 | -0.246 | -5.23 | 0.732 | 0.239 | 0.377 | 0.272 | 243 |
| rwm | theta[7] | 5.36 | 4.19 | 0.095 | 2.13 | 0.792 | 0.186 | 0.245 | 0.225 | 314 |
| mala | mu | 4.21 | 3.06 | -0.0553 | -0.974 | 0.923 | 0.048 | 0.0679 | 0.0194 | 265 |
| mala | tau | 3.9 | 3.15 | 0.0929 | 1.04 | 0.977 | 0.0665 | 0.106 | 0.0123 | 119 |
| mala | theta[0] | 6.36 | 5.48 | 0.0263 | 0.392 | 0.98 | 0.0342 | 0.0606 | 0.019 | 213 |
| mala | theta[1] | 4.75 | 4.99 | -0.0409 | -0.692 | 1.07 | 0.0411 | 0.07 | 0.00916 | 327 |
| mala | theta[2] | 3.54 | 4.83 | -0.0742 | -1.31 | 0.918 | 0.0548 | 0.0864 | 0.0273 | 264 |
| mala | theta[3] | 4.79 | 4.67 | 0.00627 | 0.0953 | 0.978 | 0.0322 | 0.0466 | 0.012 | 221 |
| mala | theta[4] | 3.05 | 4.63 | -0.121 | -1.94 | 0.995 | 0.0617 | 0.125 | 0.0187 | 255 |
| mala | theta[5] | 4.32 | 4.68 | 0.0577 | 1.01 | 0.97 | 0.0369 | 0.067 | 0.0186 | 286 |
| mala | theta[6] | 6.32 | 5.13 | 0.00485 | 0.0808 | 1.01 | 0.0165 | 0.0249 | 0.00877 | 283 |
| mala | theta[7] | 4.36 | 5.41 | -0.0927 | -1.44 | 1.02 | 0.0543 | 0.0957 | 0.0161 | 253 |
| barker | mu | 4.79 | 3.28 | 0.117 | 1.25 | 0.99 | 0.0896 | 0.129 | 0.0356 | 111 |
| barker | tau | 3.75 | 3.24 | 0.0484 | 0.3 | 1.01 | 0.0378 | 0.0543 | 0.0195 | 39 |
| barker | theta[0] | 7.16 | 6.28 | 0.169 | 1.77 | 1.12 | 0.0803 | 0.171 | 0.0583 | 138 |
| barker | theta[1] | 5.52 | 4.52 | 0.124 | 1.86 | 0.968 | 0.0794 | 0.13 | 0.0181 | 213 |
| barker | theta[2] | 4.31 | 4.96 | 0.0735 | 0.894 | 0.942 | 0.0699 | 0.0934 | 0.0288 | 131 |
| barker | theta[3] | 5.89 | 4.65 | 0.238 | 3.5 | 0.972 | 0.11 | 0.239 | 0.06 | 205 |
| barker | theta[4] | 3.83 | 4.6 | 0.0459 | 0.662 | 0.988 | 0.0415 | 0.0655 | 0.0267 | 204 |
| barker | theta[5] | 4.34 | 5 | 0.0618 | 0.783 | 1.04 | 0.0693 | 0.1 | 0.035 | 172 |
| barker | theta[6] | 6.5 | 4.7 | 0.0408 | 0.659 | 0.925 | 0.076 | 0.0992 | 0.0372 | 223 |
| barker | theta[7] | 5.41 | 5.14 | 0.106 | 1.32 | 0.972 | 0.0952 | 0.12 | 0.0411 | 148 |
| hmc | mu | 4.43 | 3.4 | 0.0102 | 0.712 | 1.02 | 0.0155 | 0.0271 | 0.00868 | 5130 |
| hmc | tau | 3.71 | 3.45 | 0.0348 | 1.1 | 1.07 | 0.0104 | 0.0175 | 0.00175 | 1160 |
| hmc | theta[0] | 6.34 | 5.64 | 0.0237 | 1.17 | 1.01 | 0.0187 | 0.0369 | 0.00619 | 2450 |
| hmc | theta[1] | 5.12 | 4.81 | 0.0391 | 1.83 | 1.03 | 0.0267 | 0.0425 | 0.00699 | 2320 |
| hmc | theta[2] | 3.99 | 5.78 | 0.0112 | 0.552 | 1.1 | 0.0254 | 0.0673 | 0.00799 | 2920 |
| hmc | theta[3] | 5 | 4.86 | 0.0505 | 3.29 | 1.02 | 0.0334 | 0.054 | 0.0111 | 4390 |
| hmc | theta[4] | 3.58 | 4.53 | -0.00724 | -0.439 | 0.973 | 0.0233 | 0.0387 | 0.00589 | 3480 |
| hmc | theta[5] | 4.32 | 4.87 | 0.0585 | 3.29 | 1.01 | 0.0284 | 0.0604 | 0.0073 | 3210 |
| hmc | theta[6] | 6.38 | 5 | 0.0165 | 0.901 | 0.984 | 0.0222 | 0.0344 | 0.00781 | 2880 |
| hmc | theta[7] | 4.94 | 5.51 | 0.0157 | 0.836 | 1.04 | 0.0153 | 0.028 | 0.00601 | 3060 |
| slice | mu | 4.46 | 3.29 | 0.0186 | 0.624 | 0.991 | 0.0294 | 0.0311 | 0.00804 | 1100 |
| slice | tau | 3.48 | 3.21 | -0.0355 | -1.05 | 0.998 | 0.0415 | 0.065 | 0.0106 | 862 |
| slice | theta[0] | 6.17 | 5.58 | -0.00755 | -0.246 | 0.997 | 0.0133 | 0.0166 | 0.00349 | 1060 |
| slice | theta[1] | 4.83 | 4.76 | -0.0231 | -0.904 | 1.02 | 0.0122 | 0.0314 | 0.00775 | 1590 |
| slice | theta[2] | 3.95 | 5.4 | 0.00434 | 0.159 | 1.03 | 0.0171 | 0.0339 | 0.00408 | 1410 |
| slice | theta[3] | 4.74 | 4.9 | -0.00267 | -0.102 | 1.02 | 0.0205 | 0.0314 | 0.0128 | 1520 |
| slice | theta[4] | 3.72 | 4.69 | 0.0218 | 0.798 | 1.01 | 0.0248 | 0.0313 | 0.00731 | 1360 |
| slice | theta[5] | 4.34 | 4.62 | 0.062 | 2.45 | 0.958 | 0.0364 | 0.0664 | 0.00852 | 1430 |
| slice | theta[6] | 6.18 | 4.95 | -0.0233 | -0.913 | 0.975 | 0.0277 | 0.0381 | 0.00931 | 1460 |
| slice | theta[7] | 4.88 | 5.58 | 0.00548 | 0.193 | 1.05 | 0.0234 | 0.0362 | 0.00697 | 1380 |
| aies | mu | 2.94 | 3.41 | -0.44 | -2.4 | 1.03 | 0.19 | 0.441 | 0.115 | 31.3 |
| aies | tau | 3.68 | 3.47 | 0.0242 | 0.122 | 1.08 | 0.0263 | 0.036 | 0.00974 | 29.4 |
| aies | theta[0] | 4.94 | 5.09 | -0.227 | -1.51 | 0.91 | 0.119 | 0.227 | 0.0404 | 36.9 |
| aies | theta[1] | 3.44 | 4.91 | -0.321 | -1.78 | 1.05 | 0.119 | 0.321 | 0.0582 | 34 |
| aies | theta[2] | 1.95 | 5.43 | -0.375 | -2.44 | 1.03 | 0.146 | 0.376 | 0.0866 | 45.1 |
| aies | theta[3] | 3.6 | 4.66 | -0.241 | -1.76 | 0.976 | 0.124 | 0.241 | 0.0401 | 50.5 |
| aies | theta[4] | 2.39 | 4.75 | -0.262 | -1.66 | 1.02 | 0.127 | 0.263 | 0.0525 | 41.5 |
| aies | theta[5] | 3.15 | 4.58 | -0.184 | -1.26 | 0.949 | 0.113 | 0.187 | 0.0383 | 42 |
| aies | theta[6] | 5.59 | 5.64 | -0.14 | -0.906 | 1.11 | 0.0846 | 0.174 | 0.034 | 51.8 |
| aies | theta[7] | 3.67 | 5.59 | -0.225 | -1.32 | 1.06 | 0.124 | 0.241 | 0.0452 | 38.9 |
| ensemble-slice | mu | 4.57 | 3.56 | 0.0536 | 0.55 | 1.07 | 0.0296 | 0.0578 | 0.00796 | 122 |
| ensemble-slice | tau | 3.85 | 3.52 | 0.0782 | 0.663 | 1.09 | 0.0381 | 0.0528 | 0.0116 | 86 |
| ensemble-slice | theta[0] | 6.79 | 6.02 | 0.103 | 1.29 | 1.08 | 0.0435 | 0.105 | 0.00888 | 181 |
| ensemble-slice | theta[1] | 5.41 | 5 | 0.101 | 1.27 | 1.07 | 0.0492 | 0.106 | 0.00983 | 180 |
| ensemble-slice | theta[2] | 3.83 | 5.36 | -0.0184 | -0.242 | 1.02 | 0.0244 | 0.0404 | 0.00252 | 179 |
| ensemble-slice | theta[3] | 5.27 | 5.05 | 0.106 | 1.25 | 1.06 | 0.0414 | 0.107 | 0.00921 | 154 |
| ensemble-slice | theta[4] | 3.99 | 4.58 | 0.0797 | 0.99 | 0.984 | 0.0358 | 0.082 | 0.00796 | 149 |
| ensemble-slice | theta[5] | 4.12 | 5.09 | 0.0168 | 0.202 | 1.05 | 0.0196 | 0.0443 | 0.0048 | 159 |
| ensemble-slice | theta[6] | 6.73 | 4.91 | 0.0859 | 1.32 | 0.968 | 0.0448 | 0.0928 | 0.00978 | 220 |
| ensemble-slice | theta[7] | 5.79 | 5.61 | 0.177 | 2.07 | 1.06 | 0.0867 | 0.179 | 0.0209 | 153 |
| chees | mu | 4.32 | 3.18 | -0.0232 | -0.264 | 0.958 | 0.0268 | 0.0433 | 0.00804 | 119 |
| chees | tau | 3.63 | 3.46 | 0.00993 | 0.286 | 1.08 | 0.021 | 0.0328 | 0.00336 | 958 |
| chees | theta[0] | 6.2 | 5.54 | -0.00162 | -0.0393 | 0.991 | 0.0127 | 0.0187 | 0.00485 | 579 |
| chees | theta[1] | 4.83 | 4.57 | -0.0234 | -0.559 | 0.977 | 0.0186 | 0.0297 | 0.00396 | 547 |
| chees | theta[2] | 4 | 5.02 | 0.0134 | 0.342 | 0.953 | 0.0165 | 0.0354 | 0.00391 | 592 |
| chees | theta[3] | 4.7 | 4.7 | -0.0111 | -0.247 | 0.983 | 0.0143 | 0.0244 | 0.00319 | 480 |
| chees | theta[4] | 3.49 | 4.56 | -0.0261 | -0.543 | 0.979 | 0.0206 | 0.0353 | 0.00357 | 415 |
| chees | theta[5] | 4.06 | 4.66 | 0.0041 | 0.0966 | 0.965 | 0.018 | 0.0344 | 0.00345 | 517 |
| chees | theta[6] | 6.32 | 5.06 | 0.00515 | 0.125 | 0.997 | 0.0103 | 0.0214 | 0.00342 | 588 |
| chees | theta[7] | 4.87 | 5.71 | 0.00307 | 0.0595 | 1.08 | 0.01 | 0.0312 | 0.00178 | 435 |
| parallel-tempering | mu | 4.39 | 3.34 | -0.00333 | -0.108 | 1.01 | 0.0161 | 0.0223 | 0.0069 | 1070 |
| parallel-tempering | tau | 3.61 | 3.33 | 0.00433 | 0.11 | 1.03 | 0.0284 | 0.0329 | 0.0083 | 693 |
| parallel-tempering | theta[0] | 6.08 | 5.69 | -0.0244 | -0.802 | 1.02 | 0.0287 | 0.0374 | 0.00636 | 1120 |
| parallel-tempering | theta[1] | 4.76 | 4.47 | -0.0383 | -1.58 | 0.956 | 0.0316 | 0.049 | 0.00887 | 1560 |
| parallel-tempering | theta[2] | 3.95 | 4.96 | 0.00518 | 0.197 | 0.943 | 0.019 | 0.0423 | 0.00541 | 1290 |
| parallel-tempering | theta[3] | 4.62 | 4.92 | -0.0287 | -0.947 | 1.03 | 0.0242 | 0.0509 | 0.00705 | 1150 |
| parallel-tempering | theta[4] | 3.6 | 4.66 | -0.00263 | -0.0938 | 1 | 0.0176 | 0.0252 | 0.00745 | 1280 |
| parallel-tempering | theta[5] | 3.89 | 4.69 | -0.0311 | -1.24 | 0.972 | 0.0283 | 0.0396 | 0.00694 | 1500 |
| parallel-tempering | theta[6] | 6.36 | 5.21 | 0.0131 | 0.503 | 1.03 | 0.0179 | 0.0297 | 0.00617 | 1540 |
| parallel-tempering | theta[7] | 4.71 | 5.45 | -0.0275 | -0.932 | 1.03 | 0.0306 | 0.0532 | 0.0107 | 1220 |
| mclmc | mu | 4.32 | 3.3 | -0.0226 | -0.529 | 0.995 | 0.0155 | 0.0292 | 0.0049 | 541 |
| mclmc | tau | 3.35 | 3.04 | -0.0767 | -1.4 | 0.944 | 0.0429 | 0.105 | 0.0126 | 296 |
| mclmc | theta[0] | 6.11 | 5.58 | -0.019 | -0.455 | 0.998 | 0.0202 | 0.0324 | 0.00404 | 574 |
| mclmc | theta[1] | 4.98 | 4.65 | 0.0079 | 0.213 | 0.994 | 0.0142 | 0.0211 | 0.00381 | 721 |
| mclmc | theta[2] | 3.8 | 5.03 | -0.0232 | -0.576 | 0.956 | 0.0215 | 0.0409 | 0.00824 | 562 |
| mclmc | theta[3] | 4.72 | 4.47 | -0.00854 | -0.244 | 0.935 | 0.0264 | 0.0495 | 0.00938 | 713 |
| mclmc | theta[4] | 3.7 | 4.47 | 0.0187 | 0.486 | 0.96 | 0.0299 | 0.0452 | 0.00743 | 622 |
| mclmc | theta[5] | 4.12 | 4.52 | 0.0152 | 0.412 | 0.937 | 0.0156 | 0.0433 | 0.00568 | 642 |
| mclmc | theta[6] | 5.97 | 4.7 | -0.0651 | -1.92 | 0.926 | 0.0377 | 0.074 | 0.00825 | 745 |
| mclmc | theta[7] | 4.85 | 5.04 | -0.00139 | -0.0363 | 0.953 | 0.018 | 0.0359 | 0.00671 | 619 |
| sgld | mu | 3.71 | 3.1 | -0.208 | -0.683 | 0.934 | 0.176 | 0.265 | 0.194 | 9.39 |
| sgld | tau | 3.21 | 2.63 | -0.119 | -0.794 | 0.817 | 0.0581 | 0.15 | 0.0362 | 29.8 |
| sgld | theta[0] | 5.11 | 4.92 | -0.197 | -0.782 | 0.879 | 0.142 | 0.22 | 0.0626 | 12.2 |
| sgld | theta[1] | 4.49 | 4.32 | -0.0969 | -0.496 | 0.925 | 0.104 | 0.161 | 0.0687 | 22.4 |
| sgld | theta[2] | 3.69 | 4.57 | -0.0452 | -0.144 | 0.869 | 0.134 | 0.195 | 0.119 | 7.7 |
| sgld | theta[3] | 4.93 | 4.26 | 0.0367 | 0.172 | 0.89 | 0.0658 | 0.12 | 0.121 | 17.4 |
| sgld | theta[4] | 2.85 | 3.52 | -0.164 | -0.621 | 0.755 | 0.143 | 0.279 | 0.125 | 8.23 |
| sgld | theta[5] | 3.43 | 4.43 | -0.127 | -0.61 | 0.917 | 0.103 | 0.138 | 0.0647 | 19.6 |
| sgld | theta[6] | 5.37 | 4.62 | -0.182 | -0.633 | 0.91 | 0.101 | 0.2 | 0.0541 | 9.98 |
| sgld | theta[7] | 4.68 | 5.1 | -0.033 | -0.148 | 0.963 | 0.098 | 0.12 | 0.049 | 18.7 |
| sghmc | mu | 3.58 | 2.76 | -0.246 | -0.571 | 0.832 | 0.133 | 0.286 | 0.558 | 3.73 |
| sghmc | tau | 3.58 | 2.85 | -0.00466 | -0.0349 | 0.886 | 0.0634 | 0.0993 | 0.0513 | 44 |
| sghmc | theta[0] | 4.69 | 4.33 | -0.272 | -1.59 | 0.775 | 0.163 | 0.277 | 0.11 | 20.5 |
| sghmc | theta[1] | 4.39 | 4.59 | -0.118 | -0.455 | 0.983 | 0.0614 | 0.126 | 0.0159 | 14.4 |
| sghmc | theta[2] | 2.17 | 5.71 | -0.334 | -0.694 | 1.08 | 0.16 | 0.335 | 0.0743 | 5.06 |
| sghmc | theta[3] | 4.48 | 4.32 | -0.0577 | -0.179 | 0.904 | 0.052 | 0.0979 | 0.0445 | 7.88 |
| sghmc | theta[4] | 2.8 | 4.46 | -0.175 | -0.697 | 0.958 | 0.1 | 0.179 | 0.0566 | 14.5 |
| sghmc | theta[5] | 3.09 | 4.51 | -0.197 | -0.628 | 0.934 | 0.115 | 0.204 | 0.0538 | 8.89 |
| sghmc | theta[6] | 4.95 | 4.88 | -0.265 | -0.755 | 0.961 | 0.132 | 0.267 | 0.0613 | 7.5 |
| sghmc | theta[7] | 4.06 | 4.47 | -0.151 | -0.791 | 0.845 | 0.107 | 0.172 | 0.0479 | 19.7 |

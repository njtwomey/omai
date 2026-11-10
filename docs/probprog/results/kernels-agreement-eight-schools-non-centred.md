# kernels/agreement-eight-schools-non-centred

Eight schools, non-centred: the kernel methods against the reference truth.

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 1 (plans 12, rows 120, scalePercent 100), 48.0 s.

**Passed**: 6 kernel method(s) agree with the quadrature truth; 6 reported

**Eight schools, non-centred** (eight-schools-non-centred): hierarchical, non-centred, reparametrisation, quadrature, eight-schools

Rubin's eight schools with theta_j = mu + tau z_j: the same posterior as the centred hierarchy with the funnel removed.

Truth (quadrature): grid quadrature of the posterior of (mu, tau) with the school effects integrated out in closed form (121 points per axis, evidence error 1.5e-5); each theta_j is the grid mixture of its conditional normals. Log evidence -31.3113.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: elliptical-slice (the prior of tau is halfCauchy, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for mu, tau, z); gaussian ('tau' is halfCauchy, not Gaussian); enumeration (continuous latent nodes (mu, tau, z)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method             | expectation | run                    | diagnostics                                                                | why                                                                                                                        |
| ------------------ | ----------- | ---------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| rwm                | reported    | converged (1901 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | a consistent sampler, slower mixing: agrees within Monte Carlo error; on this hierarchy it mixes too slowly at this budget |
| mala               | must agree  | converged (1101 steps) | 0 divergences; R-hat high: mu, tau; ESS low: tau                           | Metropolis-adjusted, so exact up to Monte Carlo error                                                                      |
| barker             | must agree  | converged (1301 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | Metropolis-adjusted, so exact up to Monte Carlo error                                                                      |
| hmc                | must agree  | converged (1201 steps) | 0 divergences; R-hat high: z[2], z[4]                                      | a fixed trajectory with a tuned step size: exact up to Monte Carlo error                                                   |
| slice              | must agree  | converged (901 steps)  | 0 divergences                                                              | coordinate-wise slice sampling: exact up to Monte Carlo error                                                              |
| aies               | reported    | converged (701 steps)  | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | an affine-invariant ensemble: exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget          |
| ensemble-slice     | reported    | converged (401 steps)  | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | an ensemble of slice steps: exact up to Monte Carlo error; on this hierarchy it mixes too slowly at this budget            |
| chees              | must agree  | converged (601 steps)  | 1 divergences; R-hat high: mu, tau; ESS low: mu, tau                       | jittered HMC with an adapted trajectory: exact up to Monte Carlo error                                                     |
| parallel-tempering | must agree  | converged (901 steps)  | 0 divergences                                                              | the cold replica of a tempered ladder: exact up to Monte Carlo error                                                       |
| mclmc              | reported    | converged (1601 steps) | 0 divergences; R-hat high: tau; ESS low: tau                               | unadjusted: a small bias that shrinks with the step size                                                                   |
| sgld               | reported    | converged (3201 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | unadjusted Langevin with a constant step: a step-size bias in the spread                                                   |
| sghmc              | reported    | converged (1301 steps) | 0 divergences; R-hat high: mu, tau, z[0], ...; ESS low: mu, tau, z[0], ... | friction-damped Hamiltonian dynamics with a constant step: a step-size bias                                                |

| method             | quantity | mean | sd   | mean error (sd) | mean z  | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------------ | -------- | ---- | ---- | --------------- | ------- | -------- | ------ | ------- | ------- | ---- |
| rwm                | mu       | 6.86 | 1.7  | 0.741           | 2.63    | 0.511    | 0.429  | 0.819   | 1.37    | 3.29 |
| rwm                | tau      | 4.26 | 2.27 | 0.205           | 2.95    | 0.704    | 0.268  | 0.355   | 0.271   | 103  |
| rwm                | theta[0] | 8.76 | 4.12 | 0.456           | 1.58    | 0.737    | 0.321  | 0.506   | 0.37    | 6.5  |
| rwm                | theta[1] | 7.97 | 4.84 | 0.649           | 1.16    | 1.04     | 0.324  | 0.669   | 0.311   | 3.41 |
| rwm                | theta[2] | 6.24 | 4.44 | 0.44            | 6.17    | 0.844    | 0.25   | 0.459   | 0.191   | 140  |
| rwm                | theta[3] | 9.5  | 3.75 | 0.991           | 2.32    | 0.785    | 0.442  | 0.998   | 1.04    | 3.38 |
| rwm                | theta[4] | 6.52 | 5.31 | 0.624           | 1.12    | 1.14     | 0.347  | 0.64    | 0.33    | 4.2  |
| rwm                | theta[5] | 3.9  | 3.64 | -0.0298         | -0.411  | 0.754    | 0.135  | 0.264   | 0.191   | 108  |
| rwm                | theta[6] | 9.12 | 3.96 | 0.556           | 2.41    | 0.781    | 0.367  | 0.568   | 0.539   | 11.5 |
| rwm                | theta[7] | 8.59 | 4.76 | 0.706           | 2.1     | 0.899    | 0.338  | 0.721   | 0.372   | 7.16 |
| mala               | mu       | 4.57 | 3.46 | 0.0522          | 0.892   | 1.04     | 0.0358 | 0.0639  | 0.013   | 317  |
| mala               | tau      | 3.72 | 2.75 | 0.0391          | 0.448   | 0.855    | 0.082  | 0.132   | 0.047   | 96   |
| mala               | theta[0] | 6.66 | 5.67 | 0.0804          | 1.47    | 1.01     | 0.0564 | 0.0954  | 0.0239  | 342  |
| mala               | theta[1] | 5.05 | 4.81 | 0.023           | 0.451   | 1.03     | 0.0327 | 0.0534  | 0.0114  | 407  |
| mala               | theta[2] | 3.93 | 5.55 | 0.00109         | 0.018   | 1.05     | 0.0225 | 0.0565  | 0.0113  | 304  |
| mala               | theta[3] | 4.61 | 4.92 | -0.0311         | -0.502  | 1.03     | 0.0566 | 0.067   | 0.0247  | 276  |
| mala               | theta[4] | 3.83 | 4.61 | 0.0462          | 1.12    | 0.99     | 0.0364 | 0.053   | 0.00977 | 578  |
| mala               | theta[5] | 3.72 | 4.87 | -0.0664         | -1.25   | 1.01     | 0.0506 | 0.0865  | 0.0142  | 363  |
| mala               | theta[6] | 6.52 | 5.34 | 0.0449          | 0.876   | 1.05     | 0.0405 | 0.0574  | 0.0137  | 422  |
| mala               | theta[7] | 4.9  | 5.22 | 0.0088          | 0.186   | 0.986    | 0.0257 | 0.0333  | 0.00842 | 435  |
| barker             | mu       | 4.8  | 3.02 | 0.123           | 1.73    | 0.91     | 0.0947 | 0.141   | 0.0465  | 165  |
| barker             | tau      | 3.28 | 3.11 | -0.1            | -0.747  | 0.966    | 0.0869 | 0.127   | 0.0257  | 51.8 |
| barker             | theta[0] | 6.08 | 5.1  | -0.0238         | -0.332  | 0.912    | 0.0587 | 0.073   | 0.0473  | 162  |
| barker             | theta[1] | 5.42 | 4.42 | 0.102           | 1.41    | 0.946    | 0.112  | 0.126   | 0.0465  | 171  |
| barker             | theta[2] | 4.16 | 4.58 | 0.0436          | 0.747   | 0.871    | 0.0787 | 0.159   | 0.0735  | 223  |
| barker             | theta[3] | 5.02 | 5.24 | 0.0547          | 0.667   | 1.1      | 0.0635 | 0.0997  | 0.0533  | 179  |
| barker             | theta[4] | 3.62 | 4.23 | 0.000695        | 0.0119  | 0.908    | 0.0469 | 0.0718  | 0.0347  | 243  |
| barker             | theta[5] | 4.57 | 4.56 | 0.108           | 1.18    | 0.944    | 0.0938 | 0.154   | 0.0655  | 106  |
| barker             | theta[6] | 6.42 | 4.99 | 0.0246          | 0.346   | 0.982    | 0.0425 | 0.0804  | 0.0242  | 190  |
| barker             | theta[7] | 5.12 | 5.52 | 0.05            | 0.627   | 1.04     | 0.0938 | 0.107   | 0.0407  | 171  |
| hmc                | mu       | 4.33 | 3.43 | -0.0196         | -1.36   | 1.03     | 0.0173 | 0.0314  | 0.0069  | 5130 |
| hmc                | tau      | 3.64 | 3.35 | 0.0142          | 0.483   | 1.04     | 0.0255 | 0.042   | 0.0118  | 1250 |
| hmc                | theta[0] | 6.13 | 5.51 | -0.0151         | -0.735  | 0.986    | 0.0283 | 0.04    | 0.00674 | 2290 |
| hmc                | theta[1] | 4.92 | 4.55 | -0.00527        | -0.285  | 0.972    | 0.0256 | 0.0305  | 0.00816 | 2760 |
| hmc                | theta[2] | 3.95 | 5.92 | 0.00464         | 0.232   | 1.12     | 0.0253 | 0.0705  | 0.00765 | 3150 |
| hmc                | theta[3] | 4.77 | 4.92 | 0.00337         | 0.182   | 1.03     | 0.0169 | 0.0259  | 0.0046  | 3100 |
| hmc                | theta[4] | 3.69 | 4.67 | 0.0151          | 0.839   | 1        | 0.0182 | 0.0257  | 0.00645 | 3110 |
| hmc                | theta[5] | 3.94 | 5.02 | -0.0215         | -1.02   | 1.04     | 0.0231 | 0.0446  | 0.00791 | 2450 |
| hmc                | theta[6] | 6.41 | 4.87 | 0.0218          | 1.09    | 0.959    | 0.0258 | 0.0423  | 0.00619 | 2290 |
| hmc                | theta[7] | 4.86 | 5.23 | 0.00056         | 0.0328  | 0.988    | 0.0159 | 0.0189  | 0.00347 | 3350 |
| slice              | mu       | 4.6  | 3.18 | 0.061           | 1.81    | 0.959    | 0.0461 | 0.0743  | 0.00913 | 808  |
| slice              | tau      | 3.64 | 3.21 | 0.0122          | 0.334   | 0.998    | 0.0213 | 0.0286  | 0.0053  | 750  |
| slice              | theta[0] | 6.47 | 5.56 | 0.046           | 1.73    | 0.993    | 0.0379 | 0.0549  | 0.00631 | 1400 |
| slice              | theta[1] | 5.16 | 4.51 | 0.0463          | 1.77    | 0.965    | 0.049  | 0.0662  | 0.0151  | 1370 |
| slice              | theta[2] | 4.06 | 5.33 | 0.025           | 0.796   | 1.01     | 0.0425 | 0.0511  | 0.00862 | 1040 |
| slice              | theta[3] | 5.12 | 4.79 | 0.0769          | 2.86    | 1        | 0.0453 | 0.0831  | 0.00954 | 1390 |
| slice              | theta[4] | 3.84 | 4.55 | 0.0475          | 1.68    | 0.977    | 0.0271 | 0.0572  | 0.00565 | 1190 |
| slice              | theta[5] | 4.29 | 4.78 | 0.0513          | 1.83    | 0.989    | 0.0369 | 0.0569  | 0.00696 | 1250 |
| slice              | theta[6] | 6.38 | 4.94 | 0.0159          | 0.576   | 0.972    | 0.0221 | 0.0354  | 0.00687 | 1250 |
| slice              | theta[7] | 5.06 | 5.04 | 0.0383          | 1.46    | 0.953    | 0.0295 | 0.0531  | 0.00653 | 1320 |
| aies               | mu       | 3.6  | 3.08 | -0.24           | -1.75   | 0.929    | 0.113  | 0.244   | 0.0448  | 46.3 |
| aies               | tau      | 3.72 | 3.2  | 0.0392          | 0.219   | 0.994    | 0.0518 | 0.0781  | 0.0324  | 30.7 |
| aies               | theta[0] | 6.35 | 6.01 | 0.0242          | 0.154   | 1.07     | 0.0222 | 0.0491  | 0.00955 | 47   |
| aies               | theta[1] | 4.76 | 4.4  | -0.0396         | -0.297  | 0.942    | 0.0288 | 0.0517  | 0.00625 | 50   |
| aies               | theta[2] | 3.38 | 5.42 | -0.103          | -0.668  | 1.03     | 0.0845 | 0.137   | 0.025   | 44.5 |
| aies               | theta[3] | 4.82 | 4.74 | 0.0131          | 0.0906  | 0.991    | 0.0176 | 0.0362  | 0.00713 | 47.2 |
| aies               | theta[4] | 3.04 | 4.41 | -0.123          | -0.887  | 0.946    | 0.0657 | 0.134   | 0.0239  | 46.5 |
| aies               | theta[5] | 3.53 | 4.13 | -0.107          | -0.815  | 0.856    | 0.0595 | 0.146   | 0.027   | 42.9 |
| aies               | theta[6] | 6.22 | 5.48 | -0.0147         | -0.0975 | 1.08     | 0.055  | 0.0775  | 0.0193  | 51.1 |
| aies               | theta[7] | 5.09 | 5.4  | 0.0451          | 0.268   | 1.02     | 0.0512 | 0.0721  | 0.0116  | 36.9 |
| ensemble-slice     | mu       | 3.96 | 3.12 | -0.132          | -1.52   | 0.941    | 0.0664 | 0.133   | 0.0179  | 117  |
| ensemble-slice     | tau      | 3.48 | 3.45 | -0.0372         | -0.413  | 1.07     | 0.0367 | 0.0818  | 0.00631 | 142  |
| ensemble-slice     | theta[0] | 5.91 | 5.43 | -0.0545         | -0.692  | 0.971    | 0.0645 | 0.092   | 0.0201  | 151  |
| ensemble-slice     | theta[1] | 4.36 | 4.26 | -0.124          | -2      | 0.911    | 0.076  | 0.128   | 0.0236  | 218  |
| ensemble-slice     | theta[2] | 2.79 | 5.14 | -0.216          | -2.5    | 0.977    | 0.107  | 0.217   | 0.0434  | 127  |
| ensemble-slice     | theta[3] | 4.48 | 4.9  | -0.0589         | -0.839  | 1.02     | 0.0446 | 0.0681  | 0.00809 | 213  |
| ensemble-slice     | theta[4] | 3.1  | 4.56 | -0.111          | -1.29   | 0.978    | 0.0694 | 0.112   | 0.0238  | 128  |
| ensemble-slice     | theta[5] | 3.57 | 4.33 | -0.0976         | -1.17   | 0.896    | 0.0742 | 0.117   | 0.0284  | 116  |
| ensemble-slice     | theta[6] | 6.15 | 4.86 | -0.0284         | -0.361  | 0.957    | 0.0336 | 0.059   | 0.0094  | 148  |
| ensemble-slice     | theta[7] | 4.13 | 5.03 | -0.136          | -2.26   | 0.951    | 0.0668 | 0.137   | 0.0152  | 248  |
| chees              | mu       | 4.27 | 3.3  | -0.0394         | -0.506  | 0.996    | 0.0315 | 0.0407  | 0.00399 | 164  |
| chees              | tau      | 3.59 | 3.25 | -0.00182        | -0.0519 | 1.01     | 0.015  | 0.0314  | 0.00421 | 830  |
| chees              | theta[0] | 5.99 | 5.49 | -0.0405         | -0.814  | 0.982    | 0.0209 | 0.0414  | 0.00283 | 389  |
| chees              | theta[1] | 4.69 | 4.72 | -0.054          | -1.16   | 1.01     | 0.0312 | 0.0567  | 0.00316 | 469  |
| chees              | theta[2] | 3.83 | 5.2  | -0.0191         | -0.416  | 0.988    | 0.0215 | 0.0332  | 0.00332 | 466  |
| chees              | theta[3] | 4.49 | 4.95 | -0.0568         | -1.24   | 1.03     | 0.0309 | 0.0606  | 0.00395 | 508  |
| chees              | theta[4] | 3.37 | 4.63 | -0.0521         | -1.05   | 0.994    | 0.0375 | 0.0549  | 0.00389 | 404  |
| chees              | theta[5] | 3.96 | 4.65 | -0.0166         | -0.315  | 0.964    | 0.0184 | 0.0398  | 0.00179 | 334  |
| chees              | theta[6] | 6.19 | 5.27 | -0.0207         | -0.385  | 1.04     | 0.0252 | 0.0455  | 0.00374 | 371  |
| chees              | theta[7] | 4.72 | 5.1  | -0.0261         | -0.561  | 0.964    | 0.0195 | 0.0348  | 0.00242 | 430  |
| parallel-tempering | mu       | 4.26 | 3.45 | -0.0399         | -1.23   | 1.04     | 0.0257 | 0.0444  | 0.00599 | 1030 |
| parallel-tempering | tau      | 3.64 | 3.36 | 0.0119          | 0.319   | 1.04     | 0.0186 | 0.0336  | 0.0121  | 784  |
| parallel-tempering | theta[0] | 6.01 | 5.19 | -0.0355         | -1.33   | 0.928    | 0.0201 | 0.0429  | 0.00646 | 1220 |
| parallel-tempering | theta[1] | 5.02 | 4.59 | 0.0169          | 0.652   | 0.981    | 0.0395 | 0.0498  | 0.014   | 1430 |
| parallel-tempering | theta[2] | 3.84 | 5.22 | -0.0174         | -0.636  | 0.992    | 0.0296 | 0.0264  | 0.00614 | 1310 |
| parallel-tempering | theta[3] | 4.54 | 4.77 | -0.0463         | -1.82   | 0.998    | 0.0237 | 0.0479  | 0.00694 | 1540 |
| parallel-tempering | theta[4] | 3.56 | 4.63 | -0.012          | -0.461  | 0.993    | 0.0219 | 0.0302  | 0.00691 | 1460 |
| parallel-tempering | theta[5] | 4.03 | 4.6  | -0.00278        | -0.112  | 0.952    | 0.0185 | 0.0312  | 0.00843 | 1470 |
| parallel-tempering | theta[6] | 6.22 | 5.01 | -0.0157         | -0.612  | 0.986    | 0.0132 | 0.028   | 0.00916 | 1480 |
| parallel-tempering | theta[7] | 4.65 | 5.22 | -0.0382         | -1.37   | 0.987    | 0.0291 | 0.0551  | 0.00699 | 1260 |
| mclmc              | mu       | 4.49 | 3.37 | 0.028           | 0.585   | 1.02     | 0.0235 | 0.0301  | 0.00479 | 451  |
| mclmc              | tau      | 3.24 | 3    | -0.11           | -1.88   | 0.931    | 0.0571 | 0.137   | 0.0145  | 252  |
| mclmc              | theta[0] | 6.04 | 5.19 | -0.0314         | -0.783  | 0.927    | 0.0152 | 0.0437  | 0.00353 | 536  |
| mclmc              | theta[1] | 4.94 | 4.42 | -0.000735       | -0.0195 | 0.946    | 0.0161 | 0.0392  | 0.00322 | 628  |
| mclmc              | theta[2] | 4.2  | 5.07 | 0.0521          | 1.28    | 0.963    | 0.0258 | 0.0596  | 0.00417 | 558  |
| mclmc              | theta[3] | 4.74 | 4.52 | -0.00369        | -0.0926 | 0.946    | 0.0136 | 0.0341  | 0.00348 | 563  |
| mclmc              | theta[4] | 3.64 | 4.44 | 0.0053          | 0.13    | 0.953    | 0.0175 | 0.0354  | 0.00437 | 547  |
| mclmc              | theta[5] | 4.3  | 4.67 | 0.053           | 1.35    | 0.967    | 0.0294 | 0.0538  | 0.00758 | 603  |
| mclmc              | theta[6] | 5.97 | 4.93 | -0.065          | -1.62   | 0.971    | 0.0344 | 0.067   | 0.00605 | 588  |
| mclmc              | theta[7] | 4.94 | 5.34 | 0.016           | 0.378   | 1.01     | 0.0148 | 0.0291  | 0.00429 | 573  |
| sgld               | mu       | 5.19 | 2.68 | 0.238           | 0.905   | 0.809    | 0.208  | 0.347   | 0.262   | 9.47 |
| sgld               | tau      | 3.59 | 3    | -0.00257        | -0.0125 | 0.931    | 0.0321 | 0.0695  | 0.0272  | 20.6 |
| sgld               | theta[0] | 6.56 | 4.95 | 0.0623          | 0.389   | 0.885    | 0.0907 | 0.123   | 0.0557  | 30.5 |
| sgld               | theta[1] | 5.97 | 4.77 | 0.22            | 1.22    | 1.02     | 0.181  | 0.22    | 0.121   | 32.1 |
| sgld               | theta[2] | 4.95 | 4.62 | 0.194           | 0.623   | 0.877    | 0.16   | 0.218   | 0.0782  | 7.98 |
| sgld               | theta[3] | 5.15 | 3.49 | 0.0821          | 0.524   | 0.729    | 0.134  | 0.265   | 0.129   | 21.7 |
| sgld               | theta[4] | 3.31 | 5.03 | -0.0647         | -0.266  | 1.08     | 0.102  | 0.188   | 0.0991  | 19.7 |
| sgld               | theta[5] | 4.5  | 4.29 | 0.0944          | 0.52    | 0.889    | 0.121  | 0.189   | 0.0802  | 23.9 |
| sgld               | theta[6] | 6.83 | 4.79 | 0.105           | 0.661   | 0.943    | 0.105  | 0.131   | 0.0516  | 35   |
| sgld               | theta[7] | 5.8  | 4.8  | 0.179           | 1.03    | 0.906    | 0.193  | 0.203   | 0.111   | 26.9 |
| sghmc              | mu       | 3.4  | 3.3  | -0.301          | -0.663  | 0.996    | 0.188  | 0.33    | 0.133   | 4.82 |
| sghmc              | tau      | 4.41 | 3.57 | 0.251           | 1.49    | 1.11     | 0.127  | 0.224   | 0.11    | 43   |
| sghmc              | theta[0] | 7.34 | 6.72 | 0.201           | 0.816   | 1.2      | 0.096  | 0.233   | 0.0591  | 23.7 |
| sghmc              | theta[1] | 4.03 | 5.9  | -0.194          | -0.631  | 1.26     | 0.123  | 0.268   | 0.0625  | 16.9 |
| sghmc              | theta[2] | 2.63 | 5.81 | -0.246          | -0.715  | 1.1      | 0.177  | 0.271   | 0.0908  | 10.3 |
| sghmc              | theta[3] | 3.09 | 5.38 | -0.35           | -1.08   | 1.12     | 0.191  | 0.354   | 0.0819  | 12.2 |
| sghmc              | theta[4] | 2.26 | 5.06 | -0.291          | -1.45   | 1.09     | 0.186  | 0.295   | 0.0916  | 29.2 |
| sghmc              | theta[5] | 2.76 | 5.58 | -0.266          | -0.648  | 1.16     | 0.179  | 0.282   | 0.0827  | 7.91 |
| sghmc              | theta[6] | 5.33 | 5.29 | -0.19           | -0.533  | 1.04     | 0.117  | 0.19    | 0.0432  | 8.54 |
| sghmc              | theta[7] | 4.64 | 5.38 | -0.0405         | -0.131  | 1.02     | 0.0813 | 0.116   | 0.0386  | 10.8 |

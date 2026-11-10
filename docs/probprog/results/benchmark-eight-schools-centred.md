# benchmark/eight-schools-centred

Eight schools, centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 40, scalePercent 100), 22.1 s.

**Passed**: 0 method(s) agree with the quadrature truth, 2 differ as expected; 2 reported

**Eight schools, centred** (eight-schools-centred): hierarchical, funnel, divergences, centred, quadrature, eight-schools

Rubin's eight schools with theta_j ~ N(mu, tau²): the centred hierarchy, whose funnel makes a sampler diverge when tau is small.

Truth (quadrature): grid quadrature of the posterior of (mu, tau) with the school effects integrated out in closed form (121 points per axis, evidence error 1.5e-5); each theta_j is the grid mixture of its conditional normals. Log evidence -31.3113.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, gibbs, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of tau is halfCauchy, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for mu, tau, theta); gaussian ('tau' is halfCauchy, not Gaussian); enumeration (continuous latent nodes (mu, tau, theta)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for mu, tau, theta).

| method       | expectation        | run                    | diagnostics                                                                 | why                                                                                                         |
| ------------ | ------------------ | ---------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| nuts         | reported           | converged (1001 steps) | 8 divergences; R-hat high: mu, theta[6]; ESS low: tau                       | divergent transitions near tau = 0 and a tau that is too large: the funnel (the non-centred model has none) |
| tempered-smc | reported           | converged (2 steps)    | weights ESS 1051                                                            | particles with MCMC moves in the funnel: noisier tau than the non-centred model                             |
| laplace      | expected to differ | completed (75 steps)   | degenerate mode (on the boundary of a scale, or an ill-conditioned Hessian) | the mode of the centred density is at tau -> 0 (the density is unbounded there): the fit is not a posterior |
| svi          | expected to differ | completed (1500 steps) | -                                                                           | a mean-field normal on log tau: underestimates the spread of tau and of every effect                        |

| method       | quantity | mean     | sd   | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | -------- | ---- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | mu       | 5.14     | 3.36 | 0.224           | 3.15   | 1.01     | 0.0796 | 0.227   | 0.0348  | 203  |
| nuts         | tau      | 4.21     | 3.58 | 0.189           | 1.41   | 1.11     | 0.117  | 0.165   | 0.271   | 68.1 |
| nuts         | theta[0] | 7.34     | 5.95 | 0.202           | 3.26   | 1.06     | 0.098  | 0.204   | 0.0316  | 294  |
| nuts         | theta[1] | 5.76     | 4.71 | 0.175           | 2.99   | 1.01     | 0.0894 | 0.183   | 0.0278  | 297  |
| nuts         | theta[2] | 4.57     | 5.86 | 0.122           | 2.13   | 1.11     | 0.079  | 0.152   | 0.0224  | 382  |
| nuts         | theta[3] | 5.66     | 4.98 | 0.19            | 3.75   | 1.04     | 0.0867 | 0.193   | 0.0245  | 424  |
| nuts         | theta[4] | 4.18     | 4.85 | 0.121           | 2.05   | 1.04     | 0.0677 | 0.122   | 0.0227  | 312  |
| nuts         | theta[5] | 4.39     | 5.51 | 0.0723          | 1.14   | 1.14     | 0.0615 | 0.143   | 0.0237  | 322  |
| nuts         | theta[6] | 7.42     | 5.27 | 0.221           | 3.35   | 1.04     | 0.106  | 0.223   | 0.0355  | 246  |
| nuts         | theta[7] | 5.52     | 5.85 | 0.126           | 2.06   | 1.11     | 0.0976 | 0.156   | 0.0315  | 326  |
| tempered-smc | mu       | 4.45     | 3.22 | 0.0168          | 0.669  | 0.971    | 0.0514 | 0.049   | 0.0166  | 1500 |
| tempered-smc | tau      | 3.69     | 3.18 | 0.0279          | 1.1    | 0.986    | 0.0566 | 0.0497  | 0.0188  | 1500 |
| tempered-smc | theta[0] | 6.44     | 5.34 | 0.0404          | 1.64   | 0.954    | 0.0467 | 0.0571  | 0.0145  | 1500 |
| tempered-smc | theta[1] | 4.82     | 4.65 | -0.0257         | -1     | 0.994    | 0.0267 | 0.0383  | 0.0136  | 1500 |
| tempered-smc | theta[2] | 3.83     | 5.5  | -0.0178         | -0.659 | 1.05     | 0.0409 | 0.0525  | 0.0193  | 1500 |
| tempered-smc | theta[3] | 4.85     | 4.89 | 0.0199          | 0.751  | 1.02     | 0.026  | 0.0456  | 0.0147  | 1500 |
| tempered-smc | theta[4] | 3.8      | 4.66 | 0.0399          | 1.54   | 1        | 0.0508 | 0.0512  | 0.0116  | 1500 |
| tempered-smc | theta[5] | 3.96     | 4.67 | -0.0172         | -0.689 | 0.968    | 0.036  | 0.0532  | 0.0184  | 1500 |
| tempered-smc | theta[6] | 6.59     | 5.13 | 0.0578          | 2.21   | 1.01     | 0.0354 | 0.0589  | 0.00897 | 1500 |
| tempered-smc | theta[7] | 5.11     | 5.44 | 0.0478          | 1.8    | 1.03     | 0.0441 | 0.0618  | 0.0183  | 1500 |
| laplace      | mu       | 1.36     | 3.13 | -0.915          | -61.4  | 0.942    | 0.372  | 0.915   | 0.468   | 4000 |
| laplace      | tau      | Infinity | NaN  | Infinity        | NaN    | NaN      | 0.51   | NaN     | 5.16    | 4000 |
| laplace      | theta[0] | 1.36     | 3.13 | -0.867          | -98.1  | 0.559    | 0.449  | 0.869   | 1.14    | 4000 |
| laplace      | theta[1] | 1.36     | 3.13 | -0.765          | -72.4  | 0.669    | 0.384  | 0.774   | 0.77    | 4000 |
| laplace      | theta[2] | 1.36     | 3.13 | -0.487          | -51.9  | 0.594    | 0.318  | 0.565   | 0.547   | 4000 |
| laplace      | theta[3] | 1.36     | 3.13 | -0.71           | -68.7  | 0.654    | 0.37   | 0.725   | 0.735   | 4000 |
| laplace      | theta[4] | 1.36     | 3.13 | -0.484          | -45.6  | 0.671    | 0.29   | 0.537   | 0.401   | 4000 |
| laplace      | theta[5] | 1.36     | 3.13 | -0.555          | -54.2  | 0.647    | 0.322  | 0.6     | 0.502   | 4000 |
| laplace      | theta[6] | 1.36     | 3.13 | -0.972          | -99.9  | 0.615    | 0.468  | 0.972   | 1.2     | 4000 |
| laplace      | theta[7] | 1.36     | 3.13 | -0.66           | -70.7  | 0.591    | 0.372  | 0.69    | 0.768   | 4000 |
| svi          | mu       | 3.82     | 2.09 | -0.174          | -17.5  | 0.629    | 0.174  | 0.329   | 0.302   | 4000 |
| svi          | tau      | 6.48     | 1.7  | 0.894           | 107    | 0.529    | 0.614  | 0.963   | 2.9     | 4000 |
| svi          | theta[0] | 6.7      | 5.25 | 0.0879          | 5.92   | 0.939    | 0.0967 | 0.169   | 0.0371  | 4000 |
| svi          | theta[1] | 4.7      | 3.94 | -0.0512         | -3.85  | 0.842    | 0.0325 | 0.107   | 0.0208  | 4000 |
| svi          | theta[2] | 3.08     | 5.05 | -0.162          | -10.7  | 0.96     | 0.105  | 0.19    | 0.0294  | 4000 |
| svi          | theta[3] | 4.2      | 5.32 | -0.116          | -6.61  | 1.11     | 0.076  | 0.154   | 0.0266  | 4000 |
| svi          | theta[4] | 2.54     | 4.24 | -0.231          | -16    | 0.911    | 0.131  | 0.26    | 0.0484  | 4000 |
| svi          | theta[5] | 3.11     | 4.27 | -0.193          | -13.8  | 0.885    | 0.109  | 0.228   | 0.0388  | 4000 |
| svi          | theta[6] | 7.28     | 4.51 | 0.195           | 13.9   | 0.887    | 0.137  | 0.248   | 0.0543  | 4000 |
| svi          | theta[7] | 4.22     | 4.84 | -0.119          | -8.23  | 0.915    | 0.0609 | 0.135   | 0.0165  | 4000 |

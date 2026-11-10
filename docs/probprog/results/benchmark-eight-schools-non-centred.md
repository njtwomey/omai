# benchmark/eight-schools-non-centred

Eight schools, non-centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 4, rows 24, scalePercent 100), 5.1 s.

**Passed**: 1 method(s) agree with the quadrature truth; 3 reported

**Eight schools, non-centred** (eight-schools-non-centred): hierarchical, non-centred, reparametrisation, quadrature, eight-schools

Rubin's eight schools with theta_j = mu + tau z_j: the same posterior as the centred hierarchy with the funnel removed.

Truth (quadrature): grid quadrature of the posterior of (mu, tau) with the school effects integrated out in closed form (121 points per axis, evidence error 1.5e-5); each theta_j is the grid mixture of its conditional normals. Log evidence -31.3113.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of tau is halfCauchy, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'tau' is opaque to message passing (a custom call, or a family with no exponential-family form)); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for mu, tau, z); gaussian ('tau' is halfCauchy, not Gaussian); enumeration (continuous latent nodes (mu, tau, z)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for mu, tau, z).

| method       | expectation | run                    | diagnostics      | why                                                                                                                           |
| ------------ | ----------- | ---------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (1001 steps) | 0 divergences    | no funnel: no divergences, and agreement with the truth within Monte Carlo error (tau is heavy tailed, so a looser tolerance) |
| tempered-smc | reported    | converged (2 steps)    | weights ESS 1085 | particles with MCMC moves: close, with more noise than NUTS in the tail of tau                                                |
| laplace      | reported    | converged (34 steps)   | -                | a normal at the mode on the log tau scale: too narrow for the heavy-tailed tau                                                |
| svi          | reported    | completed (1500 steps) | -                | a mean-field normal on log tau: underestimates the spread of tau and of the effects                                           |

| method       | quantity | mean | sd   | mean error (sd) | mean z     | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ---- | ---- | --------------- | ---------- | -------- | ------ | ------- | ------- | ---- |
| nuts         | mu       | 4.33 | 3.3  | -0.0187         | -0.634     | 0.994    | 0.0293 | 0.0299  | 0.00537 | 1130 |
| nuts         | tau      | 3.42 | 3.17 | -0.0562         | -1.57      | 0.984    | 0.0546 | 0.0858  | 0.0219  | 758  |
| nuts         | theta[0] | 6.06 | 5.46 | -0.0269         | -0.866     | 0.976    | 0.0226 | 0.0351  | 0.00907 | 990  |
| nuts         | theta[1] | 4.95 | 4.58 | 0.00286         | 0.0962     | 0.98     | 0.0225 | 0.0392  | 0.00628 | 1080 |
| nuts         | theta[2] | 3.86 | 5.14 | -0.0125         | -0.409     | 0.976    | 0.0201 | 0.0327  | 0.00692 | 1030 |
| nuts         | theta[3] | 4.69 | 4.68 | -0.0143         | -0.538     | 0.978    | 0.0204 | 0.0285  | 0.0072  | 1350 |
| nuts         | theta[4] | 3.71 | 4.83 | 0.0195          | 0.7        | 1.04     | 0.0287 | 0.0449  | 0.0117  | 1390 |
| nuts         | theta[5] | 4.06 | 4.74 | 0.00405         | 0.144      | 0.981    | 0.0188 | 0.0434  | 0.00954 | 1230 |
| nuts         | theta[6] | 6.05 | 5.02 | -0.0479         | -1.68      | 0.988    | 0.0319 | 0.0491  | 0.00634 | 1200 |
| nuts         | theta[7] | 4.99 | 5.4  | 0.026           | 0.842      | 1.02     | 0.0192 | 0.0384  | 0.00817 | 1090 |
| tempered-smc | mu       | 4.38 | 3.43 | -0.00426        | -0.16      | 1.03     | 0.0283 | 0.0392  | 0.00866 | 1500 |
| tempered-smc | tau      | 3.52 | 3.21 | -0.023          | -0.891     | 0.998    | 0.0248 | 0.0599  | 0.0107  | 1500 |
| tempered-smc | theta[0] | 6.21 | 5.69 | -5.16e-7        | -0.0000196 | 1.02     | 0.0195 | 0.0271  | 0.00723 | 1500 |
| tempered-smc | theta[1] | 4.86 | 4.58 | -0.0179         | -0.708     | 0.979    | 0.0205 | 0.0302  | 0.00923 | 1500 |
| tempered-smc | theta[2] | 3.8  | 5.34 | -0.0239         | -0.911     | 1.02     | 0.0224 | 0.0372  | 0.00418 | 1500 |
| tempered-smc | theta[3] | 4.99 | 4.9  | 0.049           | 1.85       | 1.03     | 0.0388 | 0.0537  | 0.00675 | 1500 |
| tempered-smc | theta[4] | 3.69 | 4.77 | 0.0158          | 0.598      | 1.02     | 0.0205 | 0.0306  | 0.00512 | 1500 |
| tempered-smc | theta[5] | 4.06 | 4.83 | 0.00387         | 0.15       | 1        | 0.0234 | 0.0374  | 0.00932 | 1500 |
| tempered-smc | theta[6] | 6.09 | 5.02 | -0.0416         | -1.63      | 0.989    | 0.0243 | 0.0459  | 0.00762 | 1500 |
| tempered-smc | theta[7] | 4.82 | 5.35 | -0.0062         | -0.237     | 1.01     | 0.0228 | 0.0382  | 0.00573 | 1500 |
| laplace      | mu       | 1.42 | 4.82 | -0.896          | -39        | 1.45     | 0.326  | 0.906   | 0.286   | 4000 |
| laplace      | tau      | 44.1 | 49.4 | 12.6            | 51.8       | 15.3     | 0.826  | 12.5    | 3.09    | 4000 |
| svi          | mu       | 4.21 | 3.21 | -0.0562         | -3.67      | 0.968    | 0.0307 | 0.062   | 0.00682 | 4000 |
| svi          | tau      | 2.92 | 2.59 | -0.212          | -16.6      | 0.804    | 0.139  | 0.287   | 0.232   | 4000 |

Notes:
- laplace: no draws of theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7] (its posterior does not hold those sites)
- svi: no draws of theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7] (its posterior does not hold those sites)

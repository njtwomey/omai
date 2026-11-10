# benchmark/radon

Hierarchical radon regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 4, rows 30, scalePercent 100), 19.8 s.

**Passed**: 1 method(s) agree with the quadrature truth, 1 differ as expected; 2 reported

**Hierarchical radon regression** (radon): hierarchical, varying-intercepts, linear-gaussian, regression, collapse, quadrature

Log radon in houses nested in counties: a varying intercept per county and a common floor effect, with unknown scales. A Gaussian block that could be collapsed, with an exact truth.

Truth (quadrature): grid quadrature of (log tau, log sigma) (101 x 101 points over the box that holds the mass), the location parameters integrated out in closed form: the posterior of (mu, beta, e) given the scales is normal, the marginal likelihood N(y; 0, sigma² I + A D A'). Log evidence -61.7871.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for mu, beta, tau, sigma, alpha); gaussian ('tau' is halfNormal, not Gaussian); enumeration (continuous latent nodes (mu, beta, tau, sigma, alpha)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method       | expectation        | run                    | diagnostics                     | why                                                                                                                                              |
| ------------ | ------------------ | ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| nuts         | must agree         | converged (1001 steps) | 4 divergences; R-hat high: beta | a four-scale hierarchy with six groups: agrees within Monte Carlo error (tau is poorly identified, so its effective sample size is the smallest) |
| tempered-smc | reported           | converged (8 steps)    | weights ESS 914                 | particles with MCMC moves in ten dimensions: close, noisier than NUTS                                                                            |
| laplace      | expected to differ | failed (72 steps)      | -                               | the mode of the centred density is at tau -> 0 (the density is unbounded there): the fit is not a posterior                                      |
| svi          | reported           | completed (1500 steps) | -                               | a mean-field normal: underestimates the spread of tau and of the county intercepts                                                               |

| method       | quantity | mean   | sd     | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | -------- | ------ | ------ | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | mu       | 2.1    | 0.259  | 0.0375          | 0.852  | 0.986    | 0.0377 | 0.0495  | 0.00834 | 501  |
| nuts         | beta     | -0.816 | 0.229  | -0.0165         | -0.29  | 1.02     | 0.0288 | 0.0525  | 0.0153  | 322  |
| nuts         | tau      | 0.471  | 0.232  | -0.0132         | -0.265 | 0.917    | 0.0252 | 0.0653  | 0.0164  | 340  |
| nuts         | sigma    | 0.713  | 0.0822 | 0.0277          | 0.728  | 1.01     | 0.0206 | 0.0386  | 0.0112  | 698  |
| nuts         | alpha[0] | 2.17   | 0.235  | 0.0746          | 1.97   | 1.01     | 0.0578 | 0.0763  | 0.016   | 714  |
| nuts         | alpha[1] | 1.86   | 0.256  | 0.0271          | 0.547  | 0.981    | 0.0226 | 0.0328  | 0.00685 | 392  |
| nuts         | alpha[2] | 2.18   | 0.232  | 0.0734          | 1.7    | 1.02     | 0.048  | 0.0761  | 0.0124  | 560  |
| nuts         | alpha[3] | 1.94   | 0.259  | 0.0468          | 0.976  | 0.988    | 0.0363 | 0.0759  | 0.0258  | 426  |
| nuts         | alpha[4] | 1.81   | 0.29   | 0.0678          | 1.23   | 1        | 0.0399 | 0.0678  | 0.0151  | 331  |
| nuts         | alpha[5] | 2.68   | 0.264  | 0.068           | 1.57   | 0.957    | 0.0417 | 0.0743  | 0.0178  | 488  |
| tempered-smc | mu       | 1.97   | 0.263  | -0.462          | -17.9  | 0.998    | 0.186  | 0.463   | 0.116   | 1500 |
| tempered-smc | beta     | -0.817 | 0.224  | -0.0212         | -0.823 | 0.998    | 0.0237 | 0.0252  | 0.00964 | 1500 |
| tempered-smc | tau      | 0.521  | 0.247  | 0.182           | 7.22   | 0.978    | 0.0803 | 0.183   | 0.0529  | 1500 |
| tempered-smc | sigma    | 0.73   | 0.0787 | 0.229           | 9.22   | 0.964    | 0.112  | 0.228   | 0.0538  | 1500 |
| tempered-smc | alpha[0] | 2.14   | 0.228  | -0.0885         | -3.51  | 0.977    | 0.0523 | 0.0938  | 0.0138  | 1500 |
| tempered-smc | alpha[1] | 1.88   | 0.257  | 0.114           | 4.48   | 0.984    | 0.0703 | 0.117   | 0.0126  | 1500 |
| tempered-smc | alpha[2] | 2.14   | 0.217  | -0.1            | -4.06  | 0.957    | 0.0531 | 0.102   | 0.0133  | 1500 |
| tempered-smc | alpha[3] | 1.88   | 0.25   | -0.163          | -6.63  | 0.954    | 0.0881 | 0.167   | 0.0254  | 1500 |
| tempered-smc | alpha[4] | 1.83   | 0.293  | 0.142           | 5.41   | 1.01     | 0.0619 | 0.143   | 0.0163  | 1500 |
| tempered-smc | alpha[5] | 2.67   | 0.259  | 0.0253          | 1.04   | 0.94     | 0.0287 | 0.0491  | 0.015   | 1500 |
| svi          | mu       | 2.07   | 0.174  | -0.0808         | -7.73  | 0.661    | 0.108  | 0.255   | 0.151   | 4000 |
| svi          | beta     | -0.854 | 0.154  | -0.188          | -17.3  | 0.686    | 0.154  | 0.294   | 0.192   | 4000 |
| svi          | tau      | 0.529  | 0.16   | 0.214           | 21.4   | 0.632    | 0.236  | 0.355   | 0.591   | 4000 |
| svi          | sigma    | 0.697  | 0.0696 | -0.173          | -12.9  | 0.852    | 0.0774 | 0.175   | 0.0401  | 4000 |
| svi          | alpha[0] | 2.12   | 0.207  | -0.141          | -10    | 0.888    | 0.0742 | 0.155   | 0.0268  | 4000 |
| svi          | alpha[1] | 1.85   | 0.199  | -0.0322         | -2.68  | 0.762    | 0.0808 | 0.196   | 0.0788  | 4000 |
| svi          | alpha[2] | 2.16   | 0.215  | -0.0401         | -2.68  | 0.946    | 0.0252 | 0.054   | 0.00652 | 4000 |
| svi          | alpha[3] | 1.89   | 0.227  | -0.134          | -9.83  | 0.864    | 0.0835 | 0.17    | 0.0355  | 4000 |
| svi          | alpha[4] | 1.71   | 0.188  | -0.258          | -25.1  | 0.65     | 0.199  | 0.361   | 0.273   | 4000 |
| svi          | alpha[5] | 2.73   | 0.22   | 0.244           | 19.4   | 0.798    | 0.131  | 0.261   | 0.101   | 4000 |

Notes:
- laplace: no draws of mu, beta, tau, sigma, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5] (its posterior does not hold those sites)

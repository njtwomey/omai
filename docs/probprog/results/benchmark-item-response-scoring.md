# benchmark/item-response-scoring

Item response theory (2PL scoring): the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 5, rows 60, scalePercent 100), 10.2 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Item response theory (2PL scoring)** (item-response-scoring): irt, psychometrics, logistic, binary-data, plates, quadrature

Respondents with latent abilities answer items of known difficulty and discrimination through a logistic link: independent one-dimensional posteriors with a quadrature truth.

Truth (quadrature): an independent quadrature per respondent (4001 points over [-8, 8]) of N(theta; 0, 1) times the 2PL response likelihood; the abilities are independent given the items, so the joint is the product. Log evidence -60.5640.

Methods whose own `check` says they apply: nuts, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: bp (belief propagation sums over finite discrete variables, and 'theta' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for theta); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (theta)); chain (the model has no single scan).

| method       | expectation | run                    | diagnostics     | why                                                                                                                                                  |
| ------------ | ----------- | ---------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (1001 steps) | 0 divergences   | twelve independent near-normal posteriors: agrees within Monte Carlo error                                                                           |
| tempered-smc | reported    | converged (5 steps)    | weights ESS 687 | tempered particles over twelve abilities: consistent, with Monte Carlo error larger than NUTS                                                        |
| ep           | must agree  | converged (17 steps)   | -               | expectation propagation projects each logistic site onto a Gaussian: close to the quadrature on these near-normal posteriors                         |
| laplace      | reported    | converged (14 steps)   | -               | the posteriors are near normal: the mode and curvature are close                                                                                     |
| svi          | reported    | completed (1500 steps) | -               | independent abilities, so the mean-field family is right in form; a mean-field normal fitted by stochastic gradients: close, with optimisation noise |

| method       | quantity  | mean     | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------ | --------- | -------- | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts         | theta[0]  | -0.156   | 0.541 | 0.00164         | 0.0723 | 0.974    | 0.0188 | 0.043   | 0.00995 | 1840 |
| nuts         | theta[1]  | -1.6     | 0.74  | -0.0388         | -1.59  | 1.08     | 0.0354 | 0.0729  | 0.0095  | 1950 |
| nuts         | theta[2]  | 0.739    | 0.533 | 0.0371          | 1.68   | 0.991    | 0.0354 | 0.0442  | 0.00793 | 2010 |
| nuts         | theta[3]  | 0.000594 | 0.545 | 0.00971         | 0.533  | 0.997    | 0.0154 | 0.0242  | 0.00857 | 3000 |
| nuts         | theta[4]  | -1.23    | 0.653 | -0.0122         | -0.663 | 1.01     | 0.0183 | 0.0338  | 0.0144  | 3000 |
| nuts         | theta[5]  | -0.0298  | 0.548 | 0.00892         | 0.42   | 0.999    | 0.0204 | 0.0265  | 0.0124  | 2210 |
| nuts         | theta[6]  | -1.56    | 0.68  | 0.0137          | 0.663  | 0.99     | 0.029  | 0.0473  | 0.0119  | 2300 |
| nuts         | theta[7]  | -0.11    | 0.526 | 0.0282          | 1.35   | 0.951    | 0.032  | 0.045   | 0.0119  | 2070 |
| nuts         | theta[8]  | 0.111    | 0.536 | -0.00584        | -0.265 | 0.989    | 0.0139 | 0.0227  | 0.0058  | 2010 |
| nuts         | theta[9]  | -0.467   | 0.575 | 0.0151          | 0.725  | 0.997    | 0.0259 | 0.0444  | 0.0103  | 2270 |
| nuts         | theta[10] | -0.301   | 0.56  | -0.0345         | -1.75  | 0.996    | 0.0301 | 0.0534  | 0.00764 | 2540 |
| nuts         | theta[11] | 0.873    | 0.529 | 0.0152          | 0.708  | 0.973    | 0.0205 | 0.0298  | 0.0133  | 2060 |
| tempered-smc | theta[0]  | -0.167   | 0.557 | -0.0197         | -0.62  | 1        | 0.0205 | 0.0362  | 0.00873 | 1000 |
| tempered-smc | theta[1]  | -1.55    | 0.659 | 0.0404          | 1.33   | 0.96     | 0.031  | 0.0516  | 0.0111  | 1000 |
| tempered-smc | theta[2]  | 0.708    | 0.509 | -0.0209         | -0.697 | 0.947    | 0.0348 | 0.0741  | 0.0156  | 1000 |
| tempered-smc | theta[3]  | 0.0372   | 0.524 | 0.0766          | 2.53   | 0.958    | 0.0544 | 0.0884  | 0.0142  | 1000 |
| tempered-smc | theta[4]  | -1.2     | 0.626 | 0.0242          | 0.792  | 0.967    | 0.0237 | 0.039   | 0.00695 | 1000 |
| tempered-smc | theta[5]  | 0.0189   | 0.53  | 0.0977          | 3.2    | 0.966    | 0.0511 | 0.0982  | 0.0146  | 1000 |
| tempered-smc | theta[6]  | -1.59    | 0.694 | -0.0209         | -0.655 | 1.01     | 0.025  | 0.0313  | 0.00688 | 1000 |
| tempered-smc | theta[7]  | -0.19    | 0.549 | -0.117          | -3.72  | 0.992    | 0.0538 | 0.117   | 0.0192  | 1000 |
| tempered-smc | theta[8]  | 0.2      | 0.524 | 0.159           | 5.21   | 0.967    | 0.0938 | 0.162   | 0.026   | 1000 |
| tempered-smc | theta[9]  | -0.459   | 0.532 | 0.0299          | 1.03   | 0.923    | 0.0495 | 0.0704  | 0.0125  | 1000 |
| tempered-smc | theta[10] | -0.279   | 0.583 | 0.00361         | 0.11   | 1.04     | 0.0271 | 0.0415  | 0.0123  | 1000 |
| tempered-smc | theta[11] | 0.821    | 0.502 | -0.0803         | -2.75  | 0.924    | 0.0512 | 0.1     | 0.0134  | 1000 |
| ep           | theta[0]  | -0.161   | 0.548 | -0.00888        | -0.569 | 0.987    | 0.0174 | 0.0263  | 0.00341 | 4000 |
| ep           | theta[1]  | -1.58    | 0.678 | -0.00584        | -0.374 | 0.988    | 0.0325 | 0.0545  | 0.00806 | 4000 |
| ep           | theta[2]  | 0.705    | 0.541 | -0.0256         | -1.61  | 1        | 0.0197 | 0.0266  | 0.00145 | 4000 |
| ep           | theta[3]  | -0.0166  | 0.549 | -0.0218         | -1.37  | 1        | 0.0234 | 0.033   | 0.00231 | 4000 |
| ep           | theta[4]  | -1.21    | 0.644 | 0.0128          | 0.812  | 0.995    | 0.0141 | 0.0305  | 0.00297 | 4000 |
| ep           | theta[5]  | -0.0249  | 0.554 | 0.0179          | 1.12   | 1.01     | 0.0125 | 0.0299  | 0.00396 | 4000 |
| ep           | theta[6]  | -1.58    | 0.682 | -0.00436        | -0.278 | 0.994    | 0.0177 | 0.0317  | 0.00382 | 4000 |
| ep           | theta[7]  | -0.115   | 0.556 | 0.019           | 1.19   | 1.01     | 0.0106 | 0.0209  | 0.00197 | 4000 |
| ep           | theta[8]  | 0.109    | 0.547 | -0.009          | -0.564 | 1.01     | 0.0112 | 0.0199  | 0.00117 | 4000 |
| ep           | theta[9]  | -0.471   | 0.583 | 0.00795         | 0.498  | 1.01     | 0.0117 | 0.0253  | 0.00209 | 4000 |
| ep           | theta[10] | -0.283   | 0.563 | -0.00358        | -0.226 | 1        | 0.0167 | 0.0225  | 0.00198 | 4000 |
| ep           | theta[11] | 0.863    | 0.54  | -0.00312        | -0.199 | 0.994    | 0.0101 | 0.0157  | 0.00203 | 4000 |
| laplace      | theta[0]  | -0.115   | 0.547 | 0.0743          | 4.76   | 0.986    | 0.0249 | 0.0743  | 0.00579 | 4000 |
| laplace      | theta[1]  | -1.49    | 0.68  | 0.115           | 7.36   | 0.992    | 0.0364 | 0.115   | 0.00959 | 4000 |
| laplace      | theta[2]  | 0.712    | 0.522 | -0.0128         | -0.834 | 0.971    | 0.0126 | 0.0223  | 0.00231 | 4000 |
| laplace      | theta[3]  | 0.0228   | 0.531 | 0.0503          | 3.28   | 0.97     | 0.0242 | 0.0514  | 0.00369 | 4000 |
| laplace      | theta[4]  | -1.13    | 0.629 | 0.129           | 8.42   | 0.973    | 0.0421 | 0.129   | 0.0131  | 4000 |
| laplace      | theta[5]  | -0.0136  | 0.532 | 0.0385          | 2.51   | 0.971    | 0.0189 | 0.0405  | 0.00348 | 4000 |
| laplace      | theta[6]  | -1.48    | 0.684 | 0.131           | 8.28   | 0.997    | 0.0377 | 0.13    | 0.0158  | 4000 |
| laplace      | theta[7]  | -0.0987  | 0.547 | 0.0489          | 3.13   | 0.988    | 0.0222 | 0.0494  | 0.00421 | 4000 |
| laplace      | theta[8]  | 0.12     | 0.528 | 0.0121          | 0.785  | 0.975    | 0.0117 | 0.0252  | 0.00326 | 4000 |
| laplace      | theta[9]  | -0.453   | 0.557 | 0.0402          | 2.64   | 0.965    | 0.0233 | 0.0411  | 0.00379 | 4000 |
| laplace      | theta[10] | -0.235   | 0.549 | 0.0825          | 5.35   | 0.975    | 0.0302 | 0.0826  | 0.00782 | 4000 |
| laplace      | theta[11] | 0.852    | 0.527 | -0.0233         | -1.52  | 0.969    | 0.021  | 0.033   | 0.00536 | 4000 |
| svi          | theta[0]  | -0.212   | 0.564 | -0.101          | -6.27  | 1.02     | 0.0498 | 0.104   | 0.00752 | 4000 |
| svi          | theta[1]  | -1.63    | 0.507 | -0.0846         | -7.24  | 0.739    | 0.109  | 0.226   | 0.107   | 4000 |
| svi          | theta[2]  | 0.766    | 0.533 | 0.0881          | 5.63   | 0.991    | 0.0497 | 0.0935  | 0.00747 | 4000 |
| svi          | theta[3]  | -0.174   | 0.557 | -0.309          | -19.2  | 1.02     | 0.142  | 0.309   | 0.0498  | 4000 |
| svi          | theta[4]  | -1.45    | 0.624 | -0.354          | -23.2  | 0.965    | 0.163  | 0.355   | 0.0712  | 4000 |
| svi          | theta[5]  | -0.0984  | 0.572 | -0.116          | -7.04  | 1.04     | 0.0577 | 0.122   | 0.0103  | 4000 |
| svi          | theta[6]  | -1.52    | 0.803 | 0.0804          | 4.35   | 1.17     | 0.0603 | 0.143   | 0.025   | 4000 |
| svi          | theta[7]  | -0.133   | 0.533 | -0.014          | -0.919 | 0.963    | 0.0182 | 0.0353  | 0.0027  | 4000 |
| svi          | theta[8]  | 0.202    | 0.554 | 0.162           | 10     | 1.02     | 0.0736 | 0.163   | 0.0146  | 4000 |
| svi          | theta[9]  | -0.403   | 0.575 | 0.127           | 8.04   | 0.997    | 0.0468 | 0.127   | 0.0116  | 4000 |
| svi          | theta[10] | -0.439   | 0.494 | -0.281          | -20.2  | 0.878    | 0.138  | 0.285   | 0.0688  | 4000 |
| svi          | theta[11] | 0.796    | 0.538 | -0.127          | -8.08  | 0.991    | 0.0487 | 0.127   | 0.00938 | 4000 |

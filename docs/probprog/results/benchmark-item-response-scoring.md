# benchmark/item-response-scoring

Item response theory (2PL scoring): the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit bcd4c412; seed 0, budget scale 1 (plans 5, rows 60, scalePercent 100), 12.4 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Item response theory (2PL scoring)** (item-response-scoring): irt, psychometrics, logistic, binary-data, plates, quadrature

Respondents with latent abilities answer items of known difficulty and discrimination through a logistic link: independent one-dimensional posteriors with a quadrature truth.

Truth (quadrature): an independent quadrature per respondent (4001 points over [-8, 8]) of N(theta; 0, 1) times the 2PL response likelihood; the abilities are independent given the items, so the joint is the product. Log evidence -60.5640.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'theta' is continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for theta); gaussian ('y' is bernoulli, not Gaussian); enumeration (continuous latent nodes (theta)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for theta).

| method       | expectation | run                    | diagnostics     | why                                                                                                                                                  |
| ------------ | ----------- | ---------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| nuts         | must agree  | converged (2001 steps) | 0 divergences   | twelve independent near-normal posteriors: agrees within Monte Carlo error                                                                           |
| tempered-smc | reported    | converged (5 steps)    | weights ESS 833 | tempered particles over twelve abilities: consistent, with Monte Carlo error larger than NUTS                                                        |
| ep           | must agree  | converged (17 steps)   | -               | expectation propagation projects each logistic site onto a Gaussian: close to the quadrature on these near-normal posteriors                         |
| laplace      | reported    | converged (13 steps)   | -               | the posteriors are near normal: the mode and curvature are close                                                                                     |
| svi          | reported    | completed (1500 steps) | -               | independent abilities, so the mean-field family is right in form; a mean-field normal fitted by stochastic gradients: close, with optimisation noise |

| method       | quantity  | mean    | sd    | mean error (sd) | mean z  | sd ratio | KS      | W1 (sd) | KL      | ESS  |
| ------------ | --------- | ------- | ----- | --------------- | ------- | -------- | ------- | ------- | ------- | ---- |
| nuts         | theta[0]  | -0.157  | 0.577 | -0.000415       | -0.0301 | 1.04     | 0.0177  | 0.0317  | 0.00436 | 5670 |
| nuts         | theta[1]  | -1.56   | 0.692 | 0.0134          | 1.15    | 1.01     | 0.011   | 0.0222  | 0.00182 | 7420 |
| nuts         | theta[2]  | 0.718   | 0.534 | -0.00232        | -0.17   | 0.993    | 0.0163  | 0.022   | 0.00302 | 5290 |
| nuts         | theta[3]  | -0.0074 | 0.573 | -0.00491        | -0.409  | 1.05     | 0.0221  | 0.0381  | 0.00451 | 7580 |
| nuts         | theta[4]  | -1.21   | 0.612 | 0.0163          | 1.42    | 0.945    | 0.0253  | 0.0489  | 0.00657 | 6800 |
| nuts         | theta[5]  | -0.0323 | 0.55  | 0.0044          | 0.373   | 1        | 0.0154  | 0.0205  | 0.00324 | 7250 |
| nuts         | theta[6]  | -1.58   | 0.671 | -0.0163         | -1.37   | 0.977    | 0.0155  | 0.0261  | 0.00374 | 6790 |
| nuts         | theta[7]  | -0.131  | 0.572 | -0.00988        | -0.75   | 1.03     | 0.0165  | 0.0292  | 0.0046  | 6160 |
| nuts         | theta[8]  | 0.113   | 0.538 | -0.001          | -0.0851 | 0.992    | 0.0106  | 0.0147  | 0.00179 | 7070 |
| nuts         | theta[9]  | -0.483  | 0.595 | -0.0125         | -0.995  | 1.03     | 0.0145  | 0.0287  | 0.00257 | 6760 |
| nuts         | theta[10] | -0.279  | 0.578 | 0.00473         | 0.38    | 1.03     | 0.0124  | 0.021   | 0.00255 | 6790 |
| nuts         | theta[11] | 0.859   | 0.535 | -0.0104         | -0.803  | 0.985    | 0.0157  | 0.0237  | 0.00342 | 5750 |
| tempered-smc | theta[0]  | -0.217  | 0.554 | -0.109          | -3.45   | 0.998    | 0.0512  | 0.11    | 0.0127  | 1000 |
| tempered-smc | theta[1]  | -1.45   | 0.691 | 0.174           | 5.48    | 1.01     | 0.0833  | 0.175   | 0.0211  | 1000 |
| tempered-smc | theta[2]  | 0.739   | 0.532 | 0.037           | 1.18    | 0.989    | 0.0348  | 0.0488  | 0.0144  | 1000 |
| tempered-smc | theta[3]  | -0.0683 | 0.537 | -0.116          | -3.74   | 0.982    | 0.0548  | 0.117   | 0.0199  | 1000 |
| tempered-smc | theta[4]  | -1.2    | 0.665 | 0.0213          | 0.656   | 1.03     | 0.0312  | 0.0398  | 0.00891 | 1000 |
| tempered-smc | theta[5]  | -0.0502 | 0.55  | -0.0282         | -0.89   | 1        | 0.0209  | 0.0318  | 0.00582 | 1000 |
| tempered-smc | theta[6]  | -1.35   | 0.614 | 0.327           | 11.6    | 0.894    | 0.145   | 0.327   | 0.0729  | 1000 |
| tempered-smc | theta[7]  | -0.204  | 0.551 | -0.142          | -4.51   | 0.996    | 0.066   | 0.142   | 0.0154  | 1000 |
| tempered-smc | theta[8]  | 0.0972  | 0.542 | -0.0307         | -0.972  | 0.999    | 0.018   | 0.0319  | 0.00987 | 1000 |
| tempered-smc | theta[9]  | -0.382  | 0.562 | 0.162           | 5.26    | 0.974    | 0.0911  | 0.164   | 0.0247  | 1000 |
| tempered-smc | theta[10] | -0.226  | 0.565 | 0.0983          | 3.09    | 1        | 0.0511  | 0.1     | 0.0128  | 1000 |
| tempered-smc | theta[11] | 0.837   | 0.528 | -0.0517         | -1.68   | 0.972    | 0.0403  | 0.0565  | 0.0134  | 1000 |
| ep           | theta[0]  | -0.155  | 0.545 | 0.00167         | 0.108   | 0.982    | 0.0172  | 0.0245  | 0.00373 | 4000 |
| ep           | theta[1]  | -1.56   | 0.692 | 0.0196          | 1.23    | 1.01     | 0.00983 | 0.0248  | 0.00178 | 4000 |
| ep           | theta[2]  | 0.736   | 0.523 | 0.0315          | 2.05    | 0.972    | 0.0214  | 0.0427  | 0.00398 | 4000 |
| ep           | theta[3]  | -0.0042 | 0.547 | 0.00094         | 0.0594  | 1        | 0.0142  | 0.0175  | 0.00309 | 4000 |
| ep           | theta[4]  | -1.21   | 0.638 | 0.0092          | 0.59    | 0.986    | 0.016   | 0.033   | 0.00463 | 4000 |
| ep           | theta[5]  | -0.0237 | 0.548 | 0.0202          | 1.28    | 0.999    | 0.0139  | 0.0226  | 0.00248 | 4000 |
| ep           | theta[6]  | -1.58   | 0.675 | -0.0105         | -0.674  | 0.984    | 0.0227  | 0.049   | 0.00468 | 4000 |
| ep           | theta[7]  | -0.133  | 0.565 | -0.0135         | -0.836  | 1.02     | 0.0174  | 0.031   | 0.00299 | 4000 |
| ep           | theta[8]  | 0.115   | 0.546 | 0.00234         | 0.147   | 1.01     | 0.0114  | 0.0214  | 0.00346 | 4000 |
| ep           | theta[9]  | -0.478  | 0.594 | -0.00344        | -0.211  | 1.03     | 0.0175  | 0.0363  | 0.00508 | 4000 |
| ep           | theta[10] | -0.275  | 0.558 | 0.011           | 0.699   | 0.992    | 0.011   | 0.0188  | 0.00226 | 4000 |
| ep           | theta[11] | 0.862   | 0.547 | -0.0046         | -0.289  | 1.01     | 0.0144  | 0.022   | 0.00239 | 4000 |
| laplace      | theta[0]  | -0.119  | 0.539 | 0.0669          | 4.35    | 0.972    | 0.027   | 0.0671  | 0.00516 | 4000 |
| laplace      | theta[1]  | -1.51   | 0.664 | 0.0974          | 6.36    | 0.968    | 0.0345  | 0.0973  | 0.0087  | 4000 |
| laplace      | theta[2]  | 0.704   | 0.52  | -0.0275         | -1.8    | 0.967    | 0.0162  | 0.0327  | 0.0031  | 4000 |
| laplace      | theta[3]  | 0.0335  | 0.541 | 0.0699          | 4.47    | 0.99     | 0.0285  | 0.07    | 0.00499 | 4000 |
| laplace      | theta[4]  | -1.16   | 0.638 | 0.0876          | 5.62    | 0.986    | 0.0308  | 0.0875  | 0.00654 | 4000 |
| laplace      | theta[5]  | -0.0059 | 0.54  | 0.0525          | 3.37    | 0.985    | 0.021   | 0.053   | 0.00391 | 4000 |
| laplace      | theta[6]  | -1.49   | 0.674 | 0.115           | 7.42    | 0.983    | 0.043   | 0.115   | 0.0103  | 4000 |
| laplace      | theta[7]  | -0.0928 | 0.536 | 0.0595          | 3.88    | 0.97     | 0.0235  | 0.0595  | 0.00589 | 4000 |
| laplace      | theta[8]  | 0.14    | 0.536 | 0.0485          | 3.11    | 0.989    | 0.0229  | 0.0488  | 0.00304 | 4000 |
| laplace      | theta[9]  | -0.423  | 0.569 | 0.0916          | 5.87    | 0.987    | 0.0316  | 0.0915  | 0.00646 | 4000 |
| laplace      | theta[10] | -0.25   | 0.539 | 0.0562          | 3.71    | 0.958    | 0.0288  | 0.0575  | 0.0058  | 4000 |
| laplace      | theta[11] | 0.854   | 0.52  | -0.0194         | -1.28   | 0.958    | 0.0173  | 0.0307  | 0.0054  | 4000 |
| svi          | theta[0]  | -0.115  | 0.574 | 0.075           | 4.59    | 1.03     | 0.0383  | 0.0753  | 0.00773 | 4000 |
| svi          | theta[1]  | -1.4    | 0.653 | 0.248           | 16.5    | 0.952    | 0.09    | 0.248   | 0.0379  | 4000 |
| svi          | theta[2]  | 0.736   | 0.518 | 0.0325          | 2.13    | 0.962    | 0.0233  | 0.045   | 0.00415 | 4000 |
| svi          | theta[3]  | 0.0761  | 0.584 | 0.148           | 8.75    | 1.07     | 0.0671  | 0.148   | 0.0142  | 4000 |
| svi          | theta[4]  | -1.19   | 0.602 | 0.0445          | 3.03    | 0.93     | 0.0293  | 0.0574  | 0.00837 | 4000 |
| svi          | theta[5]  | -0.0286 | 0.557 | 0.0111          | 0.69    | 1.02     | 0.0109  | 0.0172  | 0.00191 | 4000 |
| svi          | theta[6]  | -1.67   | 0.569 | -0.138          | -10.5   | 0.83     | 0.103   | 0.197   | 0.0542  | 4000 |
| svi          | theta[7]  | -0.0118 | 0.583 | 0.206           | 12.4    | 1.05     | 0.0822  | 0.206   | 0.0249  | 4000 |
| svi          | theta[8]  | 0.058   | 0.645 | -0.103          | -5.48   | 1.19     | 0.0739  | 0.179   | 0.0288  | 4000 |
| svi          | theta[9]  | -0.447  | 0.551 | 0.0499          | 3.31    | 0.955    | 0.0251  | 0.0528  | 0.00323 | 4000 |
| svi          | theta[10] | -0.267  | 0.526 | 0.026           | 1.76    | 0.935    | 0.0238  | 0.0533  | 0.00539 | 4000 |
| svi          | theta[11] | 0.956   | 0.604 | 0.168           | 9.57    | 1.11     | 0.0912  | 0.182   | 0.0244  | 4000 |

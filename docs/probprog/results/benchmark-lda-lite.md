# benchmark/lda-lite

LDA on a tiny corpus: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit bcd4c412; seed 0, budget scale 1 (plans 4, rows 59, scalePercent 100), 51.7 s.

**Passed**: 2 method(s) agree with the exact truth; 2 reported

**LDA on a tiny corpus** (lda-lite): topic-model, lda, dirichlet, discrete-latent, plates, exact, enumeration

Latent Dirichlet allocation with three documents, two topics and four words: Dirichlet and categorical plates whose exact posterior comes from enumerating the topic assignments.

Truth (exact): enumeration of the 2^15 topic assignments with the mixtures and topics integrated out (Dirichlet-multinomial marginals, written out here): the evidence is their sum, each assignment marginal a weighted count, and each mixture or topic probability a mixture of Betas over the counts. Log evidence -22.6532.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, gibbs, laplace, importance, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of theta is dirichlet, not normal or multivariateNormal); ep (the factor graph does not compile: familyOf: dirichlet needs a size); vmp (the factor graph does not compile: familyOf: dirichlet needs a size); bp (belief propagation sums over finite discrete variables, and 'theta', 'phi' are continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for theta, z); gaussian ('theta' is dirichlet, not Gaussian); enumeration (continuous latent nodes (theta, phi)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for theta, z).

| method  | expectation | run                    | diagnostics                                                | why                                                                                                                                                                                                          |
| ------- | ----------- | ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| nuts    | must agree  | converged (2001 steps) | 0 divergences; R-hat high: phi[3]; ESS low: phi[3], phi[4] | NUTS on the simplex-constrained mixtures and topics with the assignments summed out: consistent, within the main mode                                                                                        |
| gibbs   | must agree  | completed (8300 steps) | -                                                          | exact Dirichlet and categorical conditionals in a sweep: consistent, correlated between sweeps (the Kolmogorov distance of a two-valued assignment is not small for a correct sampler, so it is not checked) |
| laplace | reported    | converged (34 steps)   | -                                                          | a Gaussian at one mode of the unconstrained density, with the assignments summed out: the simplex coordinates are far from normal at this sample size                                                        |
| svi     | reported    | completed (1500 steps) | -                                                          | a mean-field normal on the unconstrained simplex coordinates: biased means and spreads too small                                                                                                             |

| method  | quantity | mean   | sd     | mean error (sd) | mean z  | sd ratio | KS      | W1 (sd) | KL      | ESS  |
| ------- | -------- | ------ | ------ | --------------- | ------- | -------- | ------- | ------- | ------- | ---- |
| nuts    | theta[1] | 0.363  | 0.243  | -0.0391         | -0.809  | 0.969    | 0.0297  | 0.0421  | 0.00724 | 402  |
| nuts    | theta[3] | 0.441  | 0.315  | 0.0225          | 0.4     | 0.996    | 0.0235  | 0.0284  | 0.00896 | 314  |
| nuts    | theta[5] | 0.74   | 0.258  | 0.0000524       | 0.00104 | 0.955    | 0.0386  | 0.0419  | 0.0129  | 358  |
| nuts    | phi[0]   | 0.405  | 0.162  | 0.0187          | 0.368   | 0.996    | 0.0295  | 0.0419  | 0.0121  | 386  |
| nuts    | phi[1]   | 0.37   | 0.157  | 0.0476          | 1.42    | 1.01     | 0.0337  | 0.0488  | 0.00733 | 897  |
| nuts    | phi[2]   | 0.144  | 0.14   | -0.0666         | -1.14   | 0.936    | 0.0359  | 0.0667  | 0.0239  | 257  |
| nuts    | phi[3]   | 0.0814 | 0.12   | -0.00411        | -0.0649 | 0.982    | 0.0619  | 0.0156  | 0.048   | 241  |
| nuts    | phi[4]   | 0.0499 | 0.0767 | -0.0365         | -0.471  | 0.973    | 0.104   | 0.0435  | 0.0658  | 157  |
| nuts    | phi[5]   | 0.0807 | 0.0915 | -0.0458         | -0.849  | 0.958    | 0.062   | 0.0483  | 0.0495  | 315  |
| nuts    | phi[6]   | 0.356  | 0.171  | 0.0531          | 1.03    | 0.991    | 0.038   | 0.0561  | 0.00702 | 368  |
| nuts    | phi[7]   | 0.514  | 0.177  | -0.0102         | -0.185  | 0.96     | 0.0222  | 0.0403  | 0.00811 | 304  |
| gibbs   | z[0]     | 0.782  | 0.413  | 0.0493          | 2.34    | 0.968    | 0.782   | 0.0513  | 5.81    | 2110 |
| gibbs   | z[1]     | 0.109  | 0.312  | -0.0234         | -1.18   | 0.971    | 0.883   | 0.0265  | 5.83    | 2410 |
| gibbs   | z[2]     | 0.57   | 0.495  | -0.0208         | -0.789  | 1        | 0.57    | 0.0203  | 5.79    | 1450 |
| gibbs   | z[3]     | 0.114  | 0.318  | -0.00825        | -0.416  | 0.99     | 0.883   | 0.0113  | 5.83    | 2490 |
| gibbs   | z[4]     | 0.16   | 0.367  | 0.000587        | 0.0337  | 1        | 0.84    | 0.00136 | 5.82    | 3300 |
| gibbs   | z[5]     | 0.553  | 0.497  | -0.0604         | -1.79   | 1.01     | 0.553   | 0.0601  | 5.79    | 894  |
| gibbs   | z[6]     | 0.555  | 0.497  | -0.0563         | -1.67   | 1.01     | 0.555   | 0.056   | 5.79    | 888  |
| gibbs   | z[7]     | 0.553  | 0.497  | -0.0591         | -1.68   | 1.01     | 0.554   | 0.0588  | 5.79    | 818  |
| gibbs   | z[8]     | 0.131  | 0.337  | -0.0544         | -2.65   | 0.944    | 0.85    | 0.0543  | 5.83    | 2120 |
| gibbs   | z[9]     | 0.189  | 0.392  | -0.0364         | -1.75   | 0.972    | 0.796   | 0.0394  | 5.81    | 2200 |
| gibbs   | z[10]    | 0.92   | 0.271  | 0.131           | 4.67    | 0.825    | 0.92    | 0.13    | 5.85    | 864  |
| gibbs   | z[11]    | 0.914  | 0.28   | 0.113           | 3.8     | 0.853    | 0.914   | 0.112   | 5.85    | 825  |
| gibbs   | z[12]    | 0.92   | 0.272  | 0.13            | 4.53    | 0.828    | 0.92    | 0.128   | 5.85    | 834  |
| gibbs   | z[13]    | 0.914  | 0.28   | 0.113           | 3.81    | 0.854    | 0.914   | 0.111   | 5.84    | 834  |
| gibbs   | z[14]    | 0.449  | 0.497  | 0.0282          | 1.31    | 1        | 0.565   | 0.0275  | 5.79    | 2160 |
| gibbs   | theta[1] | 0.374  | 0.251  | 0.00689         | 0.271   | 0.999    | 0.00802 | 0.0106  | 0.00083 | 1540 |
| gibbs   | theta[3] | 0.413  | 0.315  | -0.0642         | -1.86   | 0.996    | 0.034   | 0.0642  | 0.00461 | 833  |
| gibbs   | theta[5] | 0.772  | 0.234  | 0.119           | 4.32    | 0.868    | 0.0444  | 0.119   | 0.0144  | 1000 |
| gibbs   | phi[0]   | 0.404  | 0.162  | 0.0117          | 0.686   | 0.995    | 0.0115  | 0.0169  | 0.00163 | 3410 |
| gibbs   | phi[1]   | 0.366  | 0.157  | 0.0267          | 1.7     | 1.01     | 0.0177  | 0.0286  | 0.00154 | 4160 |
| gibbs   | phi[2]   | 0.161  | 0.149  | 0.0419          | 1.44    | 0.998    | 0.0301  | 0.0436  | 0.0128  | 1190 |
| gibbs   | phi[3]   | 0.0696 | 0.106  | -0.101          | -5.06   | 0.863    | 0.0758  | 0.101   | 0.0533  | 1880 |
| gibbs   | phi[4]   | 0.0494 | 0.0737 | -0.0429         | -2.89   | 0.935    | 0.0763  | 0.0439  | 0.0531  | 3960 |
| gibbs   | phi[5]   | 0.0842 | 0.0951 | -0.00857        | -0.556  | 0.995    | 0.0446  | 0.0114  | 0.0246  | 4180 |
| gibbs   | phi[6]   | 0.336  | 0.169  | -0.0591         | -2.31   | 0.98     | 0.0247  | 0.0595  | 0.00312 | 1470 |
| gibbs   | phi[7]   | 0.53   | 0.178  | 0.0779          | 2.99    | 0.964    | 0.0318  | 0.0782  | 0.00585 | 1370 |
| laplace | theta[1] | 0.356  | 0.218  | -0.0656         | -4.79   | 0.867    | 0.0611  | 0.132   | 0.136   | 4000 |
| laplace | theta[3] | 0.289  | 0.274  | -0.457          | -33.4   | 0.864    | 0.233   | 0.458   | 0.16    | 4000 |
| laplace | theta[5] | 0.772  | 0.215  | 0.117           | 9.32    | 0.796    | 0.076   | 0.144   | 0.248   | 4000 |
| laplace | phi[0]   | 0.394  | 0.138  | -0.0487         | -3.63   | 0.849    | 0.0492  | 0.118   | 0.0343  | 4000 |
| laplace | phi[1]   | 0.309  | 0.129  | -0.341          | -26     | 0.83     | 0.145   | 0.342   | 0.0949  | 4000 |
| laplace | phi[2]   | 0.234  | 0.14   | 0.537           | 36.2    | 0.939    | 0.299   | 0.537   | 0.859   | 4000 |
| laplace | phi[3]   | 0.0628 | 0.0813 | -0.156          | -14.9   | 0.665    | 0.155   | 0.196   | 0.264   | 4000 |
| laplace | phi[4]   | 0.103  | 0.158  | 0.637           | 20.1    | 2.01     | 0.219   | 0.638   | 0.421   | 4000 |
| laplace | phi[5]   | 0.15   | 0.154  | 0.682           | 26.8    | 1.61     | 0.231   | 0.682   | 0.746   | 4000 |
| laplace | phi[6]   | 0.243  | 0.154  | -0.599          | -42.3   | 0.897    | 0.278   | 0.599   | 0.206   | 4000 |
| laplace | phi[7]   | 0.504  | 0.185  | -0.0659         | -4.15   | 1        | 0.034   | 0.0681  | 0.0138  | 4000 |
| svi     | theta[1] | 0.358  | 0.203  | -0.0587         | -4.6    | 0.807    | 0.0799  | 0.166   | 0.172   | 4000 |
| svi     | theta[3] | 0.509  | 0.245  | 0.237           | 19.4    | 0.774    | 0.198   | 0.28    | 0.535   | 4000 |
| svi     | theta[5] | 0.862  | 0.141  | 0.45            | 54.6    | 0.522    | 0.223   | 0.459   | 0.526   | 4000 |
| svi     | phi[0]   | 0.45   | 0.135  | 0.297           | 22.6    | 0.831    | 0.164   | 0.319   | 0.126   | 4000 |
| svi     | phi[1]   | 0.394  | 0.157  | 0.202           | 12.6    | 1.01     | 0.12    | 0.215   | 0.0374  | 4000 |
| svi     | phi[2]   | 0.0916 | 0.113  | -0.42           | -35.3   | 0.754    | 0.223   | 0.421   | 0.169   | 4000 |
| svi     | phi[3]   | 0.0649 | 0.0939 | -0.139          | -11.5   | 0.769    | 0.0496  | 0.141   | 0.0698  | 4000 |
| svi     | phi[4]   | 0.0495 | 0.106  | -0.0416         | -1.95   | 1.35     | 0.108   | 0.237   | 0.131   | 4000 |
| svi     | phi[5]   | 0.112  | 0.128  | 0.286           | 13.5    | 1.34     | 0.191   | 0.286   | 0.609   | 4000 |
| svi     | phi[6]   | 0.379  | 0.136  | 0.187           | 14.9    | 0.792    | 0.136   | 0.244   | 0.126   | 4000 |
| svi     | phi[7]   | 0.46   | 0.143  | -0.304          | -24.8   | 0.774    | 0.165   | 0.326   | 0.215   | 4000 |

# benchmark/negative-binomial-regression

Negative binomial regression: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 5, rows 15, scalePercent 100), 45.5 s.

**Passed**: 2 method(s) agree with the quadrature truth; 3 reported

**Negative binomial regression** (negative-binomial-regression): glm, counts, negative-binomial, overdispersion, log-link, quadrature, three-dimensional

Overdispersed counts with a log-linear mean and a HalfNormal(1) overdispersion: three coordinates, one of them a skewed positive scale, with a grid truth.

Truth (quadrature): grid quadrature of the posterior density, 61 points per axis over 3 coordinate(s) (316354 density evaluations, evidence error 9.6e-5). Log evidence -273.2025.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); conjugate (no usable conjugate structure for a, b, alpha); gaussian ('alpha' is halfNormal, not Gaussian); enumeration (continuous latent nodes (a, b, alpha)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| tempered-smc | must agree | converged (6 steps) | weights ESS 607 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| importance | reported | completed (10 steps) | weights ESS 22 | weights from the prior: the posterior is concentrated, so the effective sample size is small |
| laplace | reported | converged (20 steps) | - | a normal on log alpha: misses the skew of the overdispersion |
| svi | reported | completed (1500 steps) | - | a mean-field normal on log alpha: misses the skew, and the correlation with the intercept |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | a | 0.271 | 0.196 | 0.0233 | 0.539 | 0.927 | 0.0448 | 0.0807 | 0.0239 | 461 |
| nuts | b | 0.812 | 0.101 | -0.00829 | -0.185 | 0.943 | 0.0376 | 0.0792 | 0.0227 | 444 |
| nuts | alpha | 0.622 | 0.127 | 0.086 | 1.74 | 1.04 | 0.0426 | 0.0868 | 0.0128 | 443 |
| tempered-smc | a | 0.267 | 0.214 | 0.00724 | 0.226 | 1.01 | 0.02 | 0.0284 | 0.00623 | 1000 |
| tempered-smc | b | 0.815 | 0.108 | 0.0143 | 0.445 | 1.01 | 0.0204 | 0.038 | 0.0125 | 1000 |
| tempered-smc | alpha | 0.609 | 0.121 | -0.0175 | -0.562 | 0.988 | 0.0368 | 0.0669 | 0.0119 | 1000 |
| importance | a | 0.32 | 0.23 | 0.255 | 1.11 | 1.09 | 0.188 | 0.268 | 1.21 | 22.3 |
| importance | b | 0.788 | 0.126 | -0.234 | -0.937 | 1.18 | 0.197 | 0.288 | 1.12 | 22.3 |
| importance | alpha | 0.639 | 0.11 | 0.226 | 1.18 | 0.903 | 0.229 | 0.245 | 0.769 | 22.3 |
| laplace | a | 0.272 | 0.212 | 0.0289 | 1.83 | 1 | 0.0193 | 0.0325 | 0.00378 | 4000 |
| laplace | b | 0.807 | 0.108 | -0.06 | -3.76 | 1.01 | 0.0321 | 0.0618 | 0.00793 | 4000 |
| laplace | alpha | 0.597 | 0.12 | -0.113 | -7.28 | 0.978 | 0.0547 | 0.117 | 0.00841 | 4000 |
| svi | a | 0.281 | 0.0997 | 0.0714 | 9.57 | 0.472 | 0.2 | 0.437 | 0.934 | 4000 |
| svi | b | 0.829 | 0.0449 | 0.145 | 21.8 | 0.421 | 0.26 | 0.509 | 1.19 | 4000 |
| svi | alpha | 0.606 | 0.129 | -0.0427 | -2.57 | 1.05 | 0.0333 | 0.0618 | 0.00602 | 4000 |

# Validation results

Written by `make probprog-validate` on 2026-11-10 at commit 4ce8b576 (seed 0, budget scale 3). Each file is one campaign; a diff of these tables is a regression.

| campaign                                              | verdict                                                                                            | seconds |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------- |
| [sbc/nuts-conjugate](sbc-nuts-conjugate.md)           | passed: calibrated over 300 replications                                                           | 8.0     |
| [sbc/svi-normal-guide](sbc-svi-normal-guide.md)       | passed: calibrated over 300 replications                                                           | 1.7     |
| [sbc/wrong-guide-delta](sbc-wrong-guide-delta.md)     | passed: detected: the deliberately wrong method failed calibration                                 | 0.5     |
| [sbc/wrong-guide-narrow](sbc-wrong-guide-narrow.md)   | passed: detected: the deliberately wrong method failed calibration                                 | 0.1     |
| [sbc/nuts-hierarchical](sbc-nuts-hierarchical.md)     | passed: calibrated over 120 replications                                                           | 90.1    |
| [agreement/logistic-grid](agreement-logistic-grid.md) | passed: NUTS agrees with the grid (log evidence -9.6273); Laplace and SVI are reported as they are | 5.9     |
| [funnel/non-centred](funnel-non-centred.md)           | passed: the centred model diverges, the non-centred one does not                                   | 10.6    |
| [geweke/kernels](geweke-kernels.md)                   | passed: both kernels pass                                                                          | 1.6     |

The `benchmark/` campaigns (the cross-engine agreement of each benchmark model with its truth, `probprog/models/campaigns.ts`) were run with `--only benchmark/ --scale 1` at seed 0 (about four minutes in all, one core at low priority); each file names the commit it ran at.

| campaign                                                                            | verdict                                                                                                                          | seconds |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------- |
| [benchmark/beta-bernoulli](benchmark-beta-bernoulli.md)                             | passed: 5 method(s) agree with the exact truth; 2 reported                                                                       | 3.4     |
| [benchmark/normal-normal](benchmark-normal-normal.md)                               | passed: 7 method(s) agree with the exact truth; 1 reported                                                                       | 1.9     |
| [benchmark/gamma-poisson](benchmark-gamma-poisson.md)                               | passed: 5 method(s) agree with the exact truth; 2 reported                                                                       | 2.6     |
| [benchmark/logistic-regression](benchmark-logistic-regression.md)                   | passed: 2 method(s) agree with the quadrature truth; 3 reported                                                                  | 16.7    |
| [benchmark/probit-regression](benchmark-probit-regression.md)                       | passed: 2 method(s) agree with the quadrature truth; 3 reported                                                                  | 36.3    |
| [benchmark/poisson-regression](benchmark-poisson-regression.md)                     | passed: 2 method(s) agree with the quadrature truth; 3 reported                                                                  | 12.0    |
| [benchmark/negative-binomial-regression](benchmark-negative-binomial-regression.md) | passed: 2 method(s) agree with the quadrature truth; 3 reported                                                                  | 45.5    |
| [benchmark/eight-schools-centred](benchmark-eight-schools-centred.md)               | passed: 0 method(s) agree with the quadrature truth, 2 differ as expected; 2 reported                                            | 25.2    |
| [benchmark/eight-schools-non-centred](benchmark-eight-schools-non-centred.md)       | passed: 1 method(s) agree with the quadrature truth; 3 reported                                                                  | 5.3     |
| [benchmark/gaussian-mixture-ordered](benchmark-gaussian-mixture-ordered.md)         | passed: 1 method(s) agree with the quadrature truth, 1 differ as expected; 1 reported                                            | 6.9     |
| [benchmark/gaussian-mixture](benchmark-gaussian-mixture.md)                         | passed: 1 method(s) agree with the quadrature truth; 3 reported                                                                  | 8.5     |
| [benchmark/funnel-centred](benchmark-funnel-centred.md)                             | passed: 0 method(s) agree with the analytic truth, 2 differ as expected; 0 reported                                              | 9.1     |
| [benchmark/funnel-non-centred](benchmark-funnel-non-centred.md)                     | passed: 1 method(s) agree with the analytic truth; 1 reported                                                                    | 2.6     |
| [benchmark/banana](benchmark-banana.md)                                             | passed: 0 method(s) agree with the analytic truth, 2 differ as expected; 1 reported                                              | 2.0     |
| [benchmark/bimodal](benchmark-bimodal.md)                                           | passed: 1 method(s) agree with the analytic truth, 1 differ as expected; 1 reported                                              | 3.0     |
| [benchmark/radon](benchmark-radon.md)                                               | passed: 1 method(s) agree with the quadrature truth, 1 differ as expected; 2 reported                                            | 19.8    |
| [benchmark/trueskill-three-players](benchmark-trueskill-three-players.md)           | passed: 2 method(s) agree with the quadrature truth; 3 reported                                                                  | 25.5    |
| [benchmark/ar1-state-space](benchmark-ar1-state-space.md)                           | passed: 3 method(s) agree with the exact truth; 2 reported                                                                       | 10.4    |
| [evidence/conjugate](evidence-conjugate.md)                                         | passed: 15 asserted estimates within their stated error (plus slack) of the closed-form log evidence                             | 20.0    |
| [evidence/harmonic-mean](evidence-harmonic-mean.md)                                 | passed: refused by default; its error is more than three times bridge sampling's at every likelihood width                       | 0.3     |
| [evidence/nested-gaussian](evidence-nested-gaussian.md)                             | passed: the stated error covers the analytic evidence at its nominal rate in every dimension, with no bias beyond it             | 29.3    |
| [evidence/particle-filter](evidence-particle-filter.md)                             | passed: every proposal unbiased on the natural scale; the guided proposal is less variable than the bootstrap                    | 1.6     |
| [evidence/pmcmc-ar1](evidence-pmcmc-ar1.md)                                         | passed: every chain reproduces the exact posterior mean (within four Monte Carlo standard errors) and spread (within 30 percent) | 38.3    |
| [structure/benchmarks](structure-benchmarks.md)                                     | passed: 18 models: every finding as expected                                                                                     | 0.0     |
| [structure/collapse-efficiency](structure-collapse-efficiency.md)                   | passed: collapsed NUTS gets 42 times the ESS of tau per gradient evaluation of the full centred model, with no divergences       | 34.5    |

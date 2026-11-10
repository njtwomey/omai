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

| campaign                                                                            | verdict                                                                               | seconds |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------- |
| [benchmark/beta-bernoulli](benchmark-beta-bernoulli.md)                             | passed: 5 method(s) agree with the exact truth; 2 reported                            | 3.4     |
| [benchmark/normal-normal](benchmark-normal-normal.md)                               | passed: 7 method(s) agree with the exact truth; 1 reported                            | 1.9     |
| [benchmark/gamma-poisson](benchmark-gamma-poisson.md)                               | passed: 5 method(s) agree with the exact truth; 2 reported                            | 2.6     |
| [benchmark/logistic-regression](benchmark-logistic-regression.md)                   | passed: 2 method(s) agree with the quadrature truth; 3 reported                       | 16.7    |
| [benchmark/probit-regression](benchmark-probit-regression.md)                       | passed: 2 method(s) agree with the quadrature truth; 3 reported                       | 36.3    |
| [benchmark/poisson-regression](benchmark-poisson-regression.md)                     | passed: 2 method(s) agree with the quadrature truth; 3 reported                       | 12.0    |
| [benchmark/negative-binomial-regression](benchmark-negative-binomial-regression.md) | passed: 2 method(s) agree with the quadrature truth; 3 reported                       | 45.5    |
| [benchmark/eight-schools-centred](benchmark-eight-schools-centred.md)               | passed: 0 method(s) agree with the quadrature truth, 2 differ as expected; 2 reported | 25.2    |
| [benchmark/eight-schools-non-centred](benchmark-eight-schools-non-centred.md)       | passed: 1 method(s) agree with the quadrature truth; 3 reported                       | 5.3     |
| [benchmark/gaussian-mixture-ordered](benchmark-gaussian-mixture-ordered.md)         | passed: 1 method(s) agree with the quadrature truth, 1 differ as expected; 1 reported | 6.9     |
| [benchmark/gaussian-mixture](benchmark-gaussian-mixture.md)                         | passed: 1 method(s) agree with the quadrature truth; 3 reported                       | 8.5     |
| [benchmark/funnel-centred](benchmark-funnel-centred.md)                             | passed: 0 method(s) agree with the analytic truth, 2 differ as expected; 0 reported   | 9.1     |
| [benchmark/funnel-non-centred](benchmark-funnel-non-centred.md)                     | passed: 1 method(s) agree with the analytic truth; 1 reported                         | 2.6     |
| [benchmark/banana](benchmark-banana.md)                                             | passed: 0 method(s) agree with the analytic truth, 2 differ as expected; 1 reported   | 2.0     |
| [benchmark/bimodal](benchmark-bimodal.md)                                           | passed: 1 method(s) agree with the analytic truth, 1 differ as expected; 1 reported   | 3.0     |
| [benchmark/radon](benchmark-radon.md)                                               | passed: 1 method(s) agree with the quadrature truth, 1 differ as expected; 2 reported | 19.8    |
| [benchmark/trueskill-three-players](benchmark-trueskill-three-players.md)           | passed: 2 method(s) agree with the quadrature truth; 3 reported                       | 25.5    |
| [benchmark/ar1-state-space](benchmark-ar1-state-space.md)                           | passed: 3 method(s) agree with the exact truth; 2 reported                            | 10.4    |

The `kernels/` campaigns (the kernel methods of `probprog/inference`, `probprog/models/kernel-campaigns.ts`) were run with `--only kernels/geweke,kernels/agreement,kernels/bimodal --scale 1` and `--only kernels/sbc --scale 2` at seed 0; each file names the commit it ran at.

| campaign                                                                                      | verdict                                                                                                                                                                            | seconds |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| [kernels/geweke](kernels-geweke.md)                                                           | passed: as expected: random walk: passes; MALA: passes; Barker: passes; HMC: passes; slice: passes; elliptical slice: passes; ULA without the Metropolis step (must fail): differs | 1.6     |
| [kernels/sbc-rwm](kernels-sbc-rwm.md)                                                         | passed: calibrated over 200 replications                                                                                                                                           | 2.2     |
| [kernels/sbc-mala](kernels-sbc-mala.md)                                                       | passed: calibrated over 200 replications                                                                                                                                           | 2.2     |
| [kernels/sbc-barker](kernels-sbc-barker.md)                                                   | passed: calibrated over 200 replications                                                                                                                                           | 2.4     |
| [kernels/sbc-hmc](kernels-sbc-hmc.md)                                                         | passed: calibrated over 200 replications                                                                                                                                           | 4.6     |
| [kernels/sbc-slice](kernels-sbc-slice.md)                                                     | passed: calibrated over 200 replications                                                                                                                                           | 1.2     |
| [kernels/sbc-elliptical-slice](kernels-sbc-elliptical-slice.md)                               | passed: calibrated over 200 replications                                                                                                                                           | 0.6     |
| [kernels/sbc-aies](kernels-sbc-aies.md)                                                       | passed: calibrated over 200 replications                                                                                                                                           | 3.9     |
| [kernels/sbc-ensemble-slice](kernels-sbc-ensemble-slice.md)                                   | passed: calibrated over 200 replications                                                                                                                                           | 17.5    |
| [kernels/sbc-parallel-tempering](kernels-sbc-parallel-tempering.md)                           | passed: calibrated over 200 replications                                                                                                                                           | 15.4    |
| [kernels/sbc-chees](kernels-sbc-chees.md)                                                     | passed: calibrated over 200 replications                                                                                                                                           | 26.1    |
| [kernels/sbc-mclmc](kernels-sbc-mclmc.md)                                                     | passed: calibrated over 200 replications                                                                                                                                           | 4.9     |
| [kernels/sbc-sgld](kernels-sbc-sgld.md)                                                       | passed: calibrated over 200 replications                                                                                                                                           | 5.5     |
| [kernels/sbc-hybrid](kernels-sbc-hybrid.md)                                                   | passed: calibrated over 200 replications                                                                                                                                           | 6.6     |
| [kernels/agreement-normal-normal](kernels-agreement-normal-normal.md)                         | passed: 9 kernel method(s) agree with the exact truth; 3 reported                                                                                                                  | 8.5     |
| [kernels/agreement-logistic-regression](kernels-agreement-logistic-regression.md)             | passed: 9 kernel method(s) agree with the quadrature truth; 4 reported                                                                                                             | 37.4    |
| [kernels/agreement-eight-schools-non-centred](kernels-agreement-eight-schools-non-centred.md) | passed: 6 kernel method(s) agree with the quadrature truth; 6 reported                                                                                                             | 48.0    |
| [kernels/bimodal](kernels-bimodal.md)                                                         | passed: parallel tempering recovers the mode weights in every chain; every NUTS chain stays in one mode                                                                            | 5.4     |

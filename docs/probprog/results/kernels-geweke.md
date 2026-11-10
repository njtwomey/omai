# kernels/geweke

Geweke joint-distribution tests of the kernels of the sampling library on a two-parameter Gaussian model: every adjusted kernel must pass, the unadjusted Langevin step at a large step size must fail.

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 1 (iterations 8000), 1.6 s.

**Passed**: as expected: random walk: passes; MALA: passes; Barker: passes; HMC: passes; slice: passes; elliptical slice: passes; ULA without the Metropolis step (must fail): differs

| kernel                                      | statistic | marginal-conditional | successive-conditional | z     |
| ------------------------------------------- | --------- | -------------------- | ---------------------- | ----- |
| random walk                                 | a         | 0.006                | 0.022                  | 0.36  |
| random walk                                 | b         | 0.019                | -0.042                 | -1.37 |
| random walk                                 | y         | -0.003               | -0.013                 | -0.12 |
| random walk                                 | a^2       | 1.028                | 0.945                  | -1.72 |
| random walk                                 | y^2       | 2.974                | 2.995                  | 0.18  |
| random walk                                 | a b       | 0.013                | 0.005                  | -0.22 |
| MALA                                        | a         | -0.013               | 0.008                  | 0.71  |
| MALA                                        | b         | 0.013                | -0.030                 | -1.39 |
| MALA                                        | y         | 0.006                | -0.026                 | -0.56 |
| MALA                                        | a^2       | 1.001                | 1.030                  | 0.76  |
| MALA                                        | y^2       | 3.030                | 3.192                  | 1.44  |
| MALA                                        | a b       | 0.004                | 0.015                  | 0.40  |
| Barker                                      | a         | -0.028               | 0.023                  | 1.52  |
| Barker                                      | b         | 0.016                | 0.014                  | -0.08 |
| Barker                                      | y         | 0.010                | 0.036                  | 0.46  |
| Barker                                      | a^2       | 1.005                | 1.035                  | 0.81  |
| Barker                                      | y^2       | 2.977                | 3.026                  | 0.48  |
| Barker                                      | a b       | -0.006               | -0.019                 | -0.48 |
| HMC                                         | a         | 0.002                | 0.016                  | 0.70  |
| HMC                                         | b         | -0.007               | 0.017                  | 1.12  |
| HMC                                         | y         | -0.026               | 0.053                  | 1.75  |
| HMC                                         | a^2       | 1.007                | 0.987                  | -0.73 |
| HMC                                         | y^2       | 3.049                | 3.050                  | 0.00  |
| HMC                                         | a b       | -0.010               | 0.021                  | 1.50  |
| slice                                       | a         | -0.008               | -0.022                 | -0.58 |
| slice                                       | b         | 0.010                | -0.031                 | -1.62 |
| slice                                       | y         | 0.015                | -0.072                 | -1.84 |
| slice                                       | a^2       | 0.970                | 1.018                  | 1.59  |
| slice                                       | y^2       | 2.944                | 3.072                  | 1.35  |
| slice                                       | a b       | 0.008                | 0.011                  | 0.12  |
| elliptical slice                            | a         | -0.004               | 0.027                  | 0.96  |
| elliptical slice                            | b         | 0.016                | 0.026                  | 0.31  |
| elliptical slice                            | y         | 0.014                | 0.051                  | 0.63  |
| elliptical slice                            | a^2       | 1.050                | 1.007                  | -1.11 |
| elliptical slice                            | y^2       | 3.135                | 2.987                  | -1.34 |
| elliptical slice                            | a b       | 0.002                | 0.004                  | 0.06  |
| ULA without the Metropolis step (must fail) | a         | 0.004                | 0.001                  | -0.11 |
| ULA without the Metropolis step (must fail) | b         | -0.003               | 0.001                  | 0.11  |
| ULA without the Metropolis step (must fail) | y         | 0.031                | -0.004                 | -0.74 |
| ULA without the Metropolis step (must fail) | a^2       | 0.969                | 4.007                  | 44.06 |
| ULA without the Metropolis step (must fail) | y^2       | 3.081                | 12.105                 | 44.85 |
| ULA without the Metropolis step (must fail) | a b       | 0.007                | 1.520                  | 30.04 |

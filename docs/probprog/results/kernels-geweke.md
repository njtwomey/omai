# kernels/geweke

Geweke joint-distribution tests of the kernels of the sampling library on a two-parameter Gaussian model: every adjusted kernel must pass, the unadjusted Langevin step at a large step size must fail.

Run on 2026-11-12 at commit 2519305a; seed 0, budget scale 1 (iterations 8000), 1.7 s.

**Passed**: as expected: random walk: passes; MALA: passes; Barker: passes; HMC: passes; slice: passes; elliptical slice: passes; ULA without the Metropolis step (must fail): differs

| kernel | statistic | marginal-conditional | successive-conditional | z |
| --- | --- | --- | --- | --- |
| random walk | a | -0.014 | 0.012 | 0.56 |
| random walk | b | 0.029 | 0.062 | 0.75 |
| random walk | y | 0.004 | 0.089 | 1.08 |
| random walk | a^2 | 0.992 | 0.984 | -0.18 |
| random walk | y^2 | 3.067 | 2.935 | -1.10 |
| random walk | a b | 0.021 | -0.003 | -0.75 |
| MALA | a | -0.021 | -0.005 | 0.47 |
| MALA | b | -0.012 | -0.006 | 0.17 |
| MALA | y | -0.026 | -0.010 | 0.28 |
| MALA | a^2 | 1.008 | 0.987 | -0.54 |
| MALA | y^2 | 2.886 | 3.059 | 1.67 |
| MALA | a b | -0.021 | 0.033 | 1.93 |
| Barker | a | -0.014 | -0.017 | -0.08 |
| Barker | b | 0.026 | 0.026 | -0.02 |
| Barker | y | 0.043 | -0.004 | -0.77 |
| Barker | a^2 | 1.024 | 1.008 | -0.43 |
| Barker | y^2 | 2.977 | 3.043 | 0.60 |
| Barker | a b | -0.009 | 0.007 | 0.56 |
| HMC | a | 0.026 | -0.011 | -1.80 |
| HMC | b | 0.015 | -0.017 | -1.57 |
| HMC | y | 0.019 | -0.047 | -1.50 |
| HMC | a^2 | 0.981 | 0.980 | -0.01 |
| HMC | y^2 | 3.049 | 2.992 | -0.63 |
| HMC | a b | 0.012 | 0.012 | 0.02 |
| slice | a | 0.004 | 0.012 | 0.33 |
| slice | b | -0.007 | -0.004 | 0.10 |
| slice | y | 0.006 | 0.016 | 0.19 |
| slice | a^2 | 0.998 | 0.990 | -0.26 |
| slice | y^2 | 3.010 | 3.054 | 0.43 |
| slice | a b | 0.000 | -0.001 | -0.04 |
| elliptical slice | a | -0.013 | -0.030 | -0.55 |
| elliptical slice | b | -0.025 | -0.075 | -1.62 |
| elliptical slice | y | -0.020 | -0.125 | -1.70 |
| elliptical slice | a^2 | 1.036 | 1.005 | -0.78 |
| elliptical slice | y^2 | 3.022 | 2.982 | -0.37 |
| elliptical slice | a b | 0.012 | -0.018 | -1.04 |
| ULA without the Metropolis step (must fail) | a | -0.018 | -0.023 | -0.16 |
| ULA without the Metropolis step (must fail) | b | -0.018 | 0.007 | 0.91 |
| ULA without the Metropolis step (must fail) | y | -0.031 | -0.028 | 0.05 |
| ULA without the Metropolis step (must fail) | a^2 | 0.990 | 3.949 | 41.38 |
| ULA without the Metropolis step (must fail) | y^2 | 3.015 | 11.754 | 44.11 |
| ULA without the Metropolis step (must fail) | a b | 0.007 | 1.429 | 27.60 |

# benchmark/beta-bernoulli

Beta-Bernoulli coin: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 7, rows 7, scalePercent 100), 3.4 s.

**Passed**: 5 method(s) agree with the exact truth; 2 reported

**Beta-Bernoulli coin** (beta-bernoulli): conjugate, exact, discrete-data, one-dimensional, unit-interval

A coin with a Beta(2, 2) prior on its bias and twenty flips: the conjugate model with a closed-form Beta posterior and evidence.

Truth (exact): the Beta(a + k, b + n - k) posterior and the beta function evidence (probprog/validation/exact.ts). Log evidence -14.0191.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, abc-rejection. Not applicable: em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); gaussian ('p' is beta, not Gaussian); enumeration (continuous latent nodes (p)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| conjugate | must agree | converged (1 steps) | - | the exact posterior of this model: it is the truth |
| gibbs | must agree | completed (2300 steps) | - | draws each site from its conditional: exact conditionals, so consistent |
| nuts | must agree | converged (1001 steps) | 0 divergences | a consistent sampler: agrees within Monte Carlo error |
| importance | must agree | completed (10 steps) | weights ESS 9389 | weighted draws from the prior: unbiased, with an effective sample size that shrinks as the posterior concentrates |
| tempered-smc | must agree | converged (2 steps) | weights ESS 992 | tempered particles with MCMC moves: consistent, with a noisier tail than NUTS |
| laplace | reported | converged (6 steps) | - | a normal at the mode on the logit scale: misses the skew of a Beta posterior |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the logit scale: close in the mean, symmetric where the posterior is skewed |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conjugate | p | 0.376 | 0.0954 | 0.0136 | 0.871 | 0.986 | 0.0126 | 0.0195 | 0.00294 | 4000 |
| gibbs | p | 0.374 | 0.0983 | -0.0138 | -0.6 | 1.01 | 0.0203 | 0.0266 | 0.00384 | 1960 |
| nuts | p | 0.372 | 0.098 | -0.029 | -0.577 | 1.01 | 0.037 | 0.0485 | 0.0105 | 406 |
| importance | p | 0.376 | 0.0967 | 0.0115 | 1.12 | 0.999 | 0.0142 | 0.0143 | 0.000812 | 9390 |
| tempered-smc | p | 0.377 | 0.0951 | 0.0213 | 0.687 | 0.982 | 0.0337 | 0.0499 | 0.0104 | 1000 |
| laplace | p | 0.377 | 0.0959 | 0.024 | 1.53 | 0.991 | 0.0153 | 0.0246 | 0.0015 | 4000 |
| svi | p | 0.363 | 0.0838 | -0.124 | -9.04 | 0.865 | 0.0747 | 0.154 | 0.0306 | 4000 |

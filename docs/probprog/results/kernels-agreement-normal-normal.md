# kernels/agreement-normal-normal

Normal-normal mean: the kernel methods against the exact truth.

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 1 (plans 12, rows 12, scalePercent 100), 8.5 s.

**Passed**: 9 kernel method(s) agree with the exact truth; 3 reported

**Normal-normal mean** (normal-normal): conjugate, exact, gaussian, linear-gaussian, one-dimensional

The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.

Truth (exact): the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts). Log evidence -23.9485.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, gaussian, abc-rejection. Not applicable: bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); enumeration (continuous latent nodes (mu)); chain (a spec has no recurrences yet (the chain is a markov loop of a program)).

| method             | expectation | run                    | diagnostics                                | why                                                                         |
| ------------------ | ----------- | ---------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| rwm                | must agree  | converged (1901 steps) | 0 divergences; R-hat high: mu              | a consistent sampler, slower mixing: agrees within Monte Carlo error        |
| mala               | must agree  | converged (1101 steps) | 0 divergences                              | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| barker             | must agree  | converged (1301 steps) | 0 divergences                              | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| hmc                | must agree  | converged (1201 steps) | 0 divergences                              | a fixed trajectory with a tuned step size: exact up to Monte Carlo error    |
| slice              | must agree  | converged (901 steps)  | 0 divergences                              | coordinate-wise slice sampling: exact up to Monte Carlo error               |
| aies               | must agree  | converged (701 steps)  | 0 divergences; R-hat high: mu; ESS low: mu | an affine-invariant ensemble: exact up to Monte Carlo error                 |
| ensemble-slice     | must agree  | converged (401 steps)  | 0 divergences                              | an ensemble of slice steps: exact up to Monte Carlo error                   |
| chees              | must agree  | converged (601 steps)  | 0 divergences                              | jittered HMC with an adapted trajectory: exact up to Monte Carlo error      |
| parallel-tempering | must agree  | converged (901 steps)  | 0 divergences                              | the cold replica of a tempered ladder: exact up to Monte Carlo error        |
| sgld               | reported    | converged (3201 steps) | 0 divergences; R-hat high: mu; ESS low: mu | unadjusted Langevin with a constant step: a step-size bias in the spread    |
| sghmc              | reported    | converged (1301 steps) | 0 divergences; R-hat high: mu              | friction-damped Hamiltonian dynamics with a constant step: a step-size bias |
| elliptical-slice   | reported    | converged (901 steps)  | 0 divergences                              | a Gaussian-prior sampler, where it applies                                  |

| method             | quantity | mean | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------------------ | -------- | ---- | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| rwm                | mu       | 1.42 | 0.439 | 0.0584          | 1.44   | 1.04     | 0.039  | 0.0614  | 0.0265  | 659  |
| mala               | mu       | 1.42 | 0.448 | 0.0591          | 1.4    | 1.06     | 0.0721 | 0.0921  | 0.0283  | 628  |
| barker             | mu       | 1.41 | 0.423 | 0.0276          | 0.721  | 0.999    | 0.0435 | 0.0414  | 0.0149  | 681  |
| hmc                | mu       | 1.41 | 0.424 | 0.0345          | 0.886  | 1        | 0.0269 | 0.0433  | 0.0104  | 663  |
| slice              | mu       | 1.42 | 0.428 | 0.049           | 1.84   | 1.01     | 0.0278 | 0.0501  | 0.00548 | 1440 |
| aies               | mu       | 1.4  | 0.407 | 0.000704        | 0.0132 | 0.962    | 0.0151 | 0.0303  | 0.0026  | 325  |
| ensemble-slice     | mu       | 1.4  | 0.417 | 0.00115         | 0.0827 | 0.985    | 0.0148 | 0.0236  | 0.00311 | 5040 |
| chees              | mu       | 1.4  | 0.425 | 0.00908         | 0.863  | 1        | 0.0146 | 0.0179  | 0.0019  | 9090 |
| parallel-tempering | mu       | 1.41 | 0.423 | 0.0204          | 0.491  | 1        | 0.0255 | 0.0492  | 0.0106  | 578  |
| sgld               | mu       | 1.39 | 0.419 | -0.0138         | -0.157 | 0.99     | 0.0205 | 0.0248  | 0.00243 | 127  |
| sghmc              | mu       | 1.44 | 0.446 | 0.102           | 1.53   | 1.05     | 0.0473 | 0.103   | 0.0103  | 247  |
| elliptical-slice   | mu       | 1.41 | 0.421 | 0.0123          | 0.358  | 0.994    | 0.0209 | 0.0199  | 0.00317 | 834  |

Notes:
- mclmc did not run: mclmc: the model needs at least two unconstrained coordinates

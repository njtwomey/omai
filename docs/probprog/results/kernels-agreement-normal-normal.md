# kernels/agreement-normal-normal

Normal-normal mean: the kernel methods against the exact truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 12, rows 12, scalePercent 100), 8.9 s.

**Passed**: 9 kernel method(s) agree with the exact truth; 3 reported

**Normal-normal mean** (normal-normal): conjugate, exact, gaussian, linear-gaussian, one-dimensional

The mean of normal observations with known noise sd 1.5 under a N(0, 4) prior: conjugate, linear-Gaussian, a normal posterior.

Truth (exact): the normal posterior of a normal mean with known variance and its closed-form evidence (probprog/validation/exact.ts). Log evidence -23.9485.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, gibbs, laplace, importance, tempered-smc, abc-smc, conjugate, gaussian, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, svi-conjugate-natural, pathfinder. Not applicable: bp (belief propagation sums over finite discrete variables, and 'mu' is continuous); em (no params to estimate); particle-filter (the model has no single scan); enumeration (continuous latent nodes (mu)); chain (the model has no single scan).

| method             | expectation | run                    | diagnostics                                | why                                                                         |
| ------------------ | ----------- | ---------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| rwm                | must agree  | converged (1901 steps) | 0 divergences; R-hat high: mu              | a consistent sampler, slower mixing: agrees within Monte Carlo error        |
| mala               | must agree  | converged (1101 steps) | 0 divergences                              | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| barker             | must agree  | converged (1301 steps) | 0 divergences                              | Metropolis-adjusted, so exact up to Monte Carlo error                       |
| hmc                | must agree  | converged (1201 steps) | 0 divergences; R-hat high: mu              | a fixed trajectory with a tuned step size: exact up to Monte Carlo error    |
| slice              | must agree  | converged (901 steps)  | 0 divergences                              | coordinate-wise slice sampling: exact up to Monte Carlo error               |
| aies               | must agree  | converged (701 steps)  | 0 divergences; R-hat high: mu; ESS low: mu | an affine-invariant ensemble: exact up to Monte Carlo error                 |
| ensemble-slice     | must agree  | converged (401 steps)  | 0 divergences                              | an ensemble of slice steps: exact up to Monte Carlo error                   |
| chees              | must agree  | converged (601 steps)  | 0 divergences                              | jittered HMC with an adapted trajectory: exact up to Monte Carlo error      |
| parallel-tempering | must agree  | converged (901 steps)  | 0 divergences                              | the cold replica of a tempered ladder: exact up to Monte Carlo error        |
| sgld               | reported    | converged (3201 steps) | 0 divergences; R-hat high: mu; ESS low: mu | unadjusted Langevin with a constant step: a step-size bias in the spread    |
| sghmc              | reported    | converged (1301 steps) | 0 divergences; R-hat high: mu              | friction-damped Hamiltonian dynamics with a constant step: a step-size bias |
| elliptical-slice   | reported    | converged (901 steps)  | 0 divergences                              | a Gaussian-prior sampler, where it applies                                  |

| method             | quantity | mean | sd    | mean error (sd) | mean z  | sd ratio | KS      | W1 (sd) | KL      | ESS  |
| ------------------ | -------- | ---- | ----- | --------------- | ------- | -------- | ------- | ------- | ------- | ---- |
| rwm                | mu       | 1.43 | 0.434 | 0.0643          | 1.3     | 1.03     | 0.057   | 0.0743  | 0.0222  | 428  |
| mala               | mu       | 1.42 | 0.441 | 0.0411          | 1.29    | 1.04     | 0.0488  | 0.0555  | 0.0195  | 1070 |
| barker             | mu       | 1.37 | 0.442 | -0.0764         | -1.79   | 1.05     | 0.0545  | 0.0903  | 0.0231  | 603  |
| hmc                | mu       | 1.42 | 0.458 | 0.0363          | 1.39    | 1.08     | 0.1     | 0.195   | 0.125   | 1720 |
| slice              | mu       | 1.41 | 0.418 | 0.0188          | 0.774   | 0.988    | 0.0138  | 0.0221  | 0.00564 | 1650 |
| aies               | mu       | 1.38 | 0.414 | -0.0424         | -0.657  | 0.978    | 0.0197  | 0.0445  | 0.00285 | 229  |
| ensemble-slice     | mu       | 1.4  | 0.428 | 0.000112        | 0.00761 | 1.01     | 0.00745 | 0.0138  | 0.0009  | 4690 |
| chees              | mu       | 1.4  | 0.423 | 0.00501         | 0.437   | 1        | 0.01    | 0.016   | 0.00166 | 7620 |
| parallel-tempering | mu       | 1.43 | 0.422 | 0.0675          | 1.76    | 0.996    | 0.0422  | 0.07    | 0.00838 | 673  |
| sgld               | mu       | 1.37 | 0.426 | -0.0679         | -0.663  | 1.01     | 0.0272  | 0.0682  | 0.0055  | 96.8 |
| sghmc              | mu       | 1.38 | 0.452 | -0.0424         | -0.562  | 1.07     | 0.0277  | 0.0522  | 0.00815 | 200  |
| elliptical-slice   | mu       | 1.41 | 0.443 | 0.0273          | 0.799   | 1.05     | 0.0209  | 0.0423  | 0.00539 | 939  |

Notes:
- mclmc did not run: mclmc: the model needs at least two unconstrained coordinates

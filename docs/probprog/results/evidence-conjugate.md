# evidence/conjugate

Every evidence estimator on the three conjugate benchmark models against their closed-form log evidence: each must be within four of its own standard errors plus a slack for the bias the error does not state.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (models 3, estimators 6, scalePercent 100), 25.7 s.

**Passed**: 15 asserted estimates within their stated error (plus slack) of the closed-form log evidence

The truth is the closed-form evidence of `probprog/validation/exact.ts` (one run per cell; the error column is the estimate minus the truth, the standard error the estimator's own, z their ratio).

| model | estimator | estimate | truth | error | se | z | run | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beta-bernoulli | bridge-sampling | -14.0153 | -14.0191 | 0.0038 | 0.0051 | 0.75 | 1800 steps (completed) | ok |
| beta-bernoulli | stepping-stone | -14.0183 | -14.0191 | 0.0008 | 0.0269 | 0.03 | 6 steps (converged) | ok |
| beta-bernoulli | thermodynamic-integration | -14.0796 | -14.0191 | -0.0605 | 0.0304 | -1.99 | 6 steps (converged) | ok |
| beta-bernoulli | importance-evidence | -14.0194 | -14.0191 | -0.0003 | 0.0030 | -0.11 | 44 steps (completed) | ok |
| beta-bernoulli | nested | -13.9932 | -14.0191 | 0.0259 | 0.0626 | 0.41 | 610 steps (converged) | ok |
| beta-bernoulli | tempered-smc | -14.0512 | -14.0191 | -0.0321 | n/a | n/a | 2 steps (converged) | ok |
| normal-normal | bridge-sampling | -23.9486 | -23.9485 | -0.0001 | 0.0012 | -0.05 | 1800 steps (completed) | ok |
| normal-normal | stepping-stone | -23.9739 | -23.9485 | -0.0254 | 0.0352 | -0.72 | 10 steps (converged) | ok |
| normal-normal | thermodynamic-integration | -23.9996 | -23.9485 | -0.0511 | 0.0402 | -1.27 | 10 steps (converged) | ok |
| normal-normal | importance-evidence | -23.9493 | -23.9485 | -0.0008 | 0.0032 | -0.24 | 44 steps (completed) | ok |
| normal-normal | nested | -24.0714 | -23.9485 | -0.1229 | 0.0968 | -1.27 | 743 steps (converged) | ok |
| normal-normal | tempered-smc | -23.9219 | -23.9485 | 0.0266 | n/a | n/a | 2 steps (converged) | ok |
| gamma-poisson | bridge-sampling | -30.5568 | -30.5610 | 0.0042 | 0.0018 | 2.36 | 1800 steps (completed) | ok |
| gamma-poisson | stepping-stone | -30.5893 | -30.5610 | -0.0282 | 0.0345 | -0.82 | 10 steps (converged) | ok |
| gamma-poisson | thermodynamic-integration | -30.5862 | -30.5610 | -0.0252 | 0.0397 | -0.63 | 9 steps (converged) | ok |
| gamma-poisson | importance-evidence | -30.5615 | -30.5610 | -0.0005 | 0.0034 | -0.16 | 44 steps (completed) | ok |
| gamma-poisson | nested | -30.5785 | -30.5610 | -0.0175 | 0.0914 | -0.19 | 710 steps (converged) | ok |
| gamma-poisson | tempered-smc | -30.6538 | -30.5610 | -0.0928 | n/a | n/a | 2 steps (converged) | ok |

- `bridge-sampling`: slice-sampling posterior draws, fitted Gaussian; slack 0.03
- `stepping-stone`: adaptive tempering ladder at target ESS 0.95; slack 0.06
- `thermodynamic-integration`: the same ladder, trapezoid rule: a discretisation bias remains; slack 0.1
- `importance-evidence`: pilot-fitted Gaussian proposal with a defensive prior share; slack 0.03
- `nested`: static nested sampling, slice sweeps; slack 0.03
- `tempered-smc`: the existing tempered SMC at its default ladder of two or three rungs (no error estimate): reported, not asserted; slack 0.12

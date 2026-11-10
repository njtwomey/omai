# benchmark/stick-breaking-mixture

Stick-breaking mixture (truncated Dirichlet process): the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 3, rows 13, scalePercent 100), 7.5 s.

**Passed**: 1 method(s) agree with the quadrature truth; 2 reported

**Stick-breaking mixture (truncated Dirichlet process)** (stick-breaking-mixture): mixture, nonparametric, dirichlet-process, stick-breaking, discrete-latent, custom, quadrature

The weights of a Dirichlet-process mixture truncated at four known normal components: three Beta stick breaks and a label per row, whose posterior is a three-dimensional quadrature with the labels summed out.

Truth (quadrature): a midpoint rule with 60 nodes per axis on the unit cube for the three breaks, the component labels summed out exactly (the log evidence agrees with a rule of twice the nodes to 1e-4); the weights pi[k] are the nodes' weights under the posterior. Log evidence -68.6031.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, gibbs, laplace, importance, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of v is beta, not normal or multivariateNormal); ep (expectation propagation needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for v, z); gaussian ('v' is beta, not Gaussian); enumeration (continuous latent nodes (v)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for v, z).

| method  | expectation | run                    | diagnostics                     | why                                                                                                       |
| ------- | ----------- | ---------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| nuts    | must agree  | converged (2001 steps) | 0 divergences; R-hat high: v[2] | NUTS on the logit-scale breaks with the labels summed out: consistent                                     |
| laplace | reported    | converged (11 steps)   | -                               | a Gaussian at the mode on the logit scale: the breaks are bounded and skewed, so the tails are mis-shaped |
| svi     | reported    | completed (1500 steps) | -                               | a mean-field normal on the logit breaks ignores their coupling through the weights: spreads too small     |

| method  | quantity | mean  | sd     | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ------ | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts    | v[0]     | 0.318 | 0.0968 | -0.0443         | -1.15  | 0.98     | 0.0388 | 0.0503  | 0.0174  | 644  |
| nuts    | v[1]     | 0.468 | 0.16   | 0.0277          | 0.704  | 0.971    | 0.0246 | 0.0341  | 0.0103  | 608  |
| nuts    | v[2]     | 0.675 | 0.214  | 0.0152          | 0.28   | 1.04     | 0.0331 | 0.0405  | 0.0122  | 367  |
| nuts    | pi[0]    | 0.318 | 0.0968 | -0.0443         | -1.15  | 0.98     | 0.0649 | 0.0615  | 0.411   | 644  |
| nuts    | pi[1]    | 0.324 | 0.129  | 0.0401          | 0.995  | 0.979    | 0.0367 | 0.0498  | 0.0134  | 591  |
| nuts    | pi[2]    | 0.25  | 0.118  | 0.00442         | 0.1    | 0.991    | 0.0231 | 0.0277  | 0.00726 | 507  |
| nuts    | pi[3]    | 0.108 | 0.0678 | -0.0213         | -0.443 | 1.02     | 0.0306 | 0.0383  | 0.00978 | 450  |
| laplace | v[0]     | 0.327 | 0.0958 | 0.0532          | 3.47   | 0.969    | 0.0293 | 0.0544  | 0.00991 | 4000 |
| laplace | v[1]     | 0.48  | 0.147  | 0.101           | 7.16   | 0.893    | 0.0521 | 0.105   | 0.0374  | 4000 |
| laplace | v[2]     | 0.65  | 0.181  | -0.102          | -7.38  | 0.878    | 0.0787 | 0.152   | 0.072   | 4000 |
| svi     | v[0]     | 0.333 | 0.101  | 0.107           | 6.6    | 1.02     | 0.0336 | 0.107   | 0.00895 | 4000 |
| svi     | v[1]     | 0.438 | 0.136  | -0.154          | -11.8  | 0.823    | 0.105  | 0.221   | 0.0753  | 4000 |
| svi     | v[2]     | 0.687 | 0.2    | 0.074           | 4.82   | 0.972    | 0.048  | 0.0822  | 0.0141  | 4000 |

Notes:
- laplace: no draws of pi[0], pi[1], pi[2], pi[3] (its posterior does not hold those sites)
- svi: no draws of pi[0], pi[1], pi[2], pi[3] (its posterior does not hold those sites)

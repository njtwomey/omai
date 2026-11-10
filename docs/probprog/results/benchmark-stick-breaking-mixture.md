# benchmark/stick-breaking-mixture

Stick-breaking mixture (truncated Dirichlet process): the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 3, rows 13, scalePercent 100), 8.6 s.

**Passed**: 1 method(s) agree with the quadrature truth; 2 reported

**Stick-breaking mixture (truncated Dirichlet process)** (stick-breaking-mixture): mixture, nonparametric, dirichlet-process, stick-breaking, discrete-latent, custom, quadrature

The weights of a Dirichlet-process mixture truncated at four known normal components: three Beta stick breaks and a label per row, whose posterior is a three-dimensional quadrature with the labels summed out.

Truth (quadrature): a midpoint rule with 60 nodes per axis on the unit cube for the three breaks, the component labels summed out exactly (the log evidence agrees with a rule of twice the nodes to 1e-4); the weights pi[k] are the nodes' weights under the posterior. Log evidence -68.6031.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, abc-rejection. Not applicable: ep (expectation propagation needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); vmp (variational message passing needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); bp (belief propagation needs an exponential-family factor graph: 'pi' is opaque to message passing (a custom call, or a family with no exponential-family form)); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for v, z); gaussian ('v' is beta, not Gaussian); enumeration (continuous latent nodes (v)); chain (the model has no single scan).

| method  | expectation | run                    | diagnostics   | why                                                                                                       |
| ------- | ----------- | ---------------------- | ------------- | --------------------------------------------------------------------------------------------------------- |
| nuts    | must agree  | converged (2001 steps) | 0 divergences | NUTS on the logit-scale breaks with the labels summed out: consistent                                     |
| laplace | reported    | converged (12 steps)   | -             | a Gaussian at the mode on the logit scale: the breaks are bounded and skewed, so the tails are mis-shaped |
| svi     | reported    | completed (1500 steps) | -             | a mean-field normal on the logit breaks ignores their coupling through the weights: spreads too small     |

| method  | quantity | mean  | sd     | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ------ | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts    | v[0]     | 0.33  | 0.102  | 0.0804          | 1.79   | 1.04     | 0.0448 | 0.0819  | 0.0116  | 536  |
| nuts    | v[1]     | 0.447 | 0.171  | -0.101          | -2.08  | 1.04     | 0.0513 | 0.102   | 0.0128  | 456  |
| nuts    | v[2]     | 0.682 | 0.208  | 0.0518          | 1.33   | 1.01     | 0.0431 | 0.0614  | 0.00943 | 678  |
| nuts    | pi[0]    | 0.33  | 0.102  | 0.0804          | 1.79   | 1.04     | 0.0702 | 0.086   | 0.387   | 536  |
| nuts    | pi[1]    | 0.305 | 0.135  | -0.108          | -2.19  | 1.03     | 0.0614 | 0.109   | 0.016   | 431  |
| nuts    | pi[2]    | 0.258 | 0.124  | 0.0772          | 1.88   | 1.04     | 0.0495 | 0.0811  | 0.0128  | 642  |
| nuts    | pi[3]    | 0.107 | 0.0657 | -0.0433         | -1.31  | 0.988    | 0.0475 | 0.0459  | 0.0193  | 894  |
| laplace | v[0]     | 0.328 | 0.0965 | 0.0625          | 4.05   | 0.977    | 0.0333 | 0.0632  | 0.00996 | 4000 |
| laplace | v[1]     | 0.478 | 0.146  | 0.0859          | 6.15   | 0.883    | 0.0528 | 0.103   | 0.0338  | 4000 |
| laplace | v[2]     | 0.656 | 0.178  | -0.0755         | -5.53  | 0.864    | 0.0833 | 0.135   | 0.0768  | 4000 |
| svi     | v[0]     | 0.358 | 0.0984 | 0.364           | 23.1   | 0.996    | 0.146  | 0.364   | 0.0781  | 4000 |
| svi     | v[1]     | 0.495 | 0.169  | 0.188           | 11.6   | 1.02     | 0.0784 | 0.188   | 0.0311  | 4000 |
| svi     | v[2]     | 0.657 | 0.213  | -0.0703         | -4.31  | 1.03     | 0.0414 | 0.0765  | 0.0109  | 4000 |

Notes:
- laplace: no draws of pi[0], pi[1], pi[2], pi[3] (its posterior does not hold those sites)
- svi: no draws of pi[0], pi[1], pi[2], pi[3] (its posterior does not hold those sites)

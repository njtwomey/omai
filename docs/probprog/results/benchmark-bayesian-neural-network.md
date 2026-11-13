# benchmark/bayesian-neural-network

Bayesian neural network: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 4, rows 20, scalePercent 100), 80.2 s.

**Passed**: 0 method(s) agree with the reference truth; 4 reported

**Bayesian neural network** (bayesian-neural-network): neural-network, regression, non-identifiable, reference, uncertainty

A one-hidden-layer tanh network with Gaussian priors on a noisy sine with a gap in the inputs: no closed form, a stored long-NUTS reference for the output, and the predictive uncertainty that should grow away from the data.

Truth (reference): the mean and standard deviation of the network output at five test inputs from a long NUTS run (four chains, 1000 warm-up and 2500 draws each, key(7), repeated by `bnnReferenceRun`; no divergences, R-hat at most 1.0014 and ESS above 3000 for the five function values); stored, not draws; the weights have no single-mode marginals (hidden-unit permutation and sign symmetries), so the function values are compared.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, elliptical-slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, vmp, laplace, importance, tempered-smc, abc-smc, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep (the graph has the units of a neural-network layer (an `apply` of an affine map of weights that rows share, multiplied by a weight that rows share): the weights are not identified (sign and permutation symmetries), so a Gaussian per weight averages the modes or keeps one and drops the other units, and the run does not converge to a posterior); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (the graph has the units of a neural-network layer (an `apply` of an affine map of weights that rows share, multiplied by a weight that rows share): the weights are not identified (sign and permutation symmetries), so a Gaussian per weight averages the modes or keeps one and drops the other units, and the run does not converge to a posterior); bp (belief propagation sums over finite discrete variables, and 'w1', 'b1', 'w2', 'b2' are continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w1, b1, w2, b2); gaussian (the mean of 'y' is not affine in the latents); enumeration (continuous latent nodes (w1, b1, w2, b2)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for w1, b1, w2, b2).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | reported | converged (1001 steps) | 0 divergences; R-hat high: w1[0], w1[1], w1[2], ...; ESS low: w1[0], w1[1], w1[2], ... | multimodal in the weights (unit permutations) but the function values are well mixed: close to the long run it is compared with (a consistency check only, since the reference is itself NUTS) |
| tempered-smc | reported | converged (8 steps) | weights ESS 889 | tempered particles over ten weights: finds several symmetric modes |
| laplace | reported | converged (179 steps) | - | a Gaussian at one mode of a posterior with many symmetric ones: overconfident away from the data |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the weights: overconfident, the output spread too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | fstar[0] | -0.159 | 0.462 | 0.115 | 1.91 | 1.03 | 0.0461 | 0.125 | 0.0433 | 292 |
| nuts | fstar[1] | -0.907 | 0.0655 | -0.0486 | -1.43 | 0.985 | 0.0285 | 0.0536 | 0.00822 | 842 |
| nuts | fstar[2] | 0.318 | 0.264 | 0.0849 | 1.77 | 1 | 0.0425 | 0.0894 | 0.0131 | 435 |
| nuts | fstar[3] | 0.825 | 0.079 | -0.0367 | -0.857 | 0.941 | 0.0537 | 0.102 | 0.022 | 482 |
| nuts | fstar[4] | 0.139 | 0.366 | -0.0128 | -0.251 | 0.972 | 0.0581 | 0.127 | 0.0422 | 364 |
| tempered-smc | fstar[0] | -0.713 | 0.0696 | -1.12 | -228 | 0.155 | 0.746 | 1.2 | 3.08 | 1000 |
| tempered-smc | fstar[1] | -0.926 | 0.0508 | -0.344 | -14.3 | 0.764 | 0.494 | 0.6 | 1.13 | 1000 |
| tempered-smc | fstar[2] | 0.19 | 0.106 | -0.399 | -31.5 | 0.401 | 0.488 | 0.756 | 1.87 | 1000 |
| tempered-smc | fstar[3] | 0.838 | 0.0427 | 0.121 | 7.54 | 0.509 | 0.392 | 0.591 | 1.25 | 1000 |
| tempered-smc | fstar[4] | 0.29 | 0.076 | 0.389 | 60.9 | 0.202 | 0.477 | 0.791 | 2.16 | 1000 |
| laplace | fstar[0] | -0.218 | 0.652 | -0.0159 | -0.695 | 1.45 | 0.0757 | 0.303 | 0.0558 | 4000 |
| laplace | fstar[1] | -0.985 | 0.409 | -1.23 | -12.6 | 6.16 | 0.259 | 2.7 | 0.478 | 4000 |
| laplace | fstar[2] | 0.139 | 0.398 | -0.593 | -24.8 | 1.51 | 0.196 | 0.66 | 0.144 | 4000 |
| laplace | fstar[3] | 0.826 | 0.358 | -0.0223 | -0.33 | 4.27 | 0.195 | 1.71 | 0.335 | 4000 |
| laplace | fstar[4] | 0.144 | 0.558 | -0.000448 | -0.0191 | 1.48 | 0.0757 | 0.315 | 0.0595 | 4000 |
| svi | fstar[0] | -0.581 | 0.0944 | -0.824 | -248 | 0.21 | 0.64 | 0.945 | 3.49 | 4000 |
| svi | fstar[1] | -0.805 | 0.0928 | 1.48 | 66.9 | 1.4 | 0.482 | 1.48 | 0.627 | 4000 |
| svi | fstar[2] | 0.37 | 0.18 | 0.285 | 26.4 | 0.682 | 0.189 | 0.362 | 0.222 | 4000 |
| svi | fstar[3] | 0.693 | 0.0886 | -1.61 | -96.4 | 1.06 | 0.567 | 1.61 | 1.19 | 4000 |
| svi | fstar[4] | 0.398 | 0.0943 | 0.675 | 170 | 0.251 | 0.559 | 0.825 | 2.98 | 4000 |

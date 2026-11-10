# svi/banana

Haario's banana: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 3, rows 5, scalePercent 100), 14.4 s.

**Passed**: 0 method(s) agree with the analytic truth, 1 differ as expected; 2 reported

**Haario's banana** (banana): geometry, curved, density, analytic, two-dimensional, correlated

A curved two-dimensional ridge: x0 ~ N(0, 4), x1 | x0 ~ N(0.5 (x0^2 - 4), 1). Marginals are known; the shape is far from Gaussian.

Truth (analytic): x0 is N(0, 4) exactly; x1 has mean 0 and variance 2 b² a⁴ + 1 = 9, its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, laplace, importance, tempered-smc, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of x1 depends on another node); ep (x1: operator 'square' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); vmp (x1: operator 'square' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); bp (belief propagation sums over finite discrete variables, and 'x0', 'x1' are continuous); gibbs (no discrete or conjugate conditionals); em (observed nodes have no likelihood); particle-filter (the model has no single scan); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); gaussian (the mean of 'x1' is not affine in the latents); enumeration (observed nodes have no likelihood); chain (the model has no single scan); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method         | expectation        | run                    | diagnostics     | why                                                                                                                                           |
| -------------- | ------------------ | ---------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| svi-structured | reported           | completed (2000 steps) | -               | six planar flows bend the Gaussian along the ridge: wider than any Gaussian guide, still narrower than the truth (reverse KL is mode-seeking) |
| svi-structured | expected to differ | completed (1500 steps) | -               | a full-rank normal is a straight ellipse: wrong in the spread of x1 and in its skew                                                           |
| pathfinder     | reported           | converged (15 steps)   | weights ESS 588 | normal approximations along the path, importance weights from the mixture of four paths                                                       |

| method         | quantity | mean    | sd   | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL    | ESS  |
| -------------- | -------- | ------- | ---- | --------------- | ------ | -------- | ------ | ------- | ----- | ---- |
| svi-structured | x0       | -0.0349 | 1.43 | -0.0175         | -1.54  | 0.716    | 0.0923 | 0.207   | 0.244 | 4000 |
| svi-structured | x1       | -0.942  | 1.59 | -0.314          | -37.6  | 0.528    | 0.12   | 0.309   | 0.313 | 4000 |
| svi-structured | x1       | -1.65   | 1    | -0.549          | -104   | 0.334    | 0.319  | 0.544   | 1.04  | 4000 |
| pathfinder     | x0       | -0.0501 | 1.41 | -0.0251         | -0.863 | 0.704    | 0.0918 | 0.212   | 0.438 | 588  |
| pathfinder     | x1       | -1.06   | 1.43 | -0.353          | -18    | 0.475    | 0.177  | 0.349   | 0.48  | 588  |

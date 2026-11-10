# svi/banana

Haario's banana: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 547817e0; seed 0, budget scale 1 (plans 3, rows 5, scalePercent 100), 14.4 s.

**Passed**: 0 method(s) agree with the analytic truth, 1 differ as expected; 2 reported

**Haario's banana** (banana): geometry, curved, density, analytic, two-dimensional, correlated

A curved two-dimensional ridge: x0 ~ N(0, 4), x1 | x0 ~ N(0.5 (x0^2 - 4), 1). Marginals are known; the shape is far from Gaussian.

Truth (analytic): x0 is N(0, 4) exactly; x1 has mean 0 and variance 2 b² a⁴ + 1 = 9, its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep (x1: operator 'square' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); vmp (x1: operator 'square' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); bp (belief propagation sums over finite discrete variables, and 'x0', 'x1' are continuous); gibbs (no discrete or conjugate conditionals); em (observed nodes have no likelihood); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); gaussian (the mean of 'x1' is not affine in the latents); enumeration (observed nodes have no likelihood); chain (a spec has no recurrences yet (the chain is a markov loop of a program)); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| svi-structured | reported | completed (2000 steps) | - | six planar flows bend the Gaussian along the ridge: wider than any Gaussian guide, still narrower than the truth (reverse KL is mode-seeking) |
| svi-structured | expected to differ | completed (1500 steps) | - | a full-rank normal is a straight ellipse: wrong in the spread of x1 and in its skew |
| pathfinder | reported | converged (14 steps) | weights ESS 66 | normal approximations along the path, importance weights from the mixture of four paths |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svi-structured | x0 | -0.0363 | 1.67 | -0.0181 | -1.37 | 0.834 | 0.047 | 0.111 | 0.0755 | 4000 |
| svi-structured | x1 | -0.749 | 1.58 | -0.25 | -30 | 0.527 | 0.112 | 0.248 | 0.344 | 4000 |
| svi-structured | x1 | -1.63 | 1.03 | -0.543 | -100 | 0.342 | 0.31 | 0.537 | 0.985 | 4000 |
| pathfinder | x0 | -0.167 | 1.72 | -0.0834 | -0.787 | 0.86 | 0.0984 | 0.129 | 0.548 | 65.8 |
| pathfinder | x1 | -0.641 | 1.73 | -0.214 | -3 | 0.577 | 0.14 | 0.209 | 0.462 | 65.8 |

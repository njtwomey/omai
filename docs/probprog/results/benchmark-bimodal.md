# benchmark/bimodal

Two Gaussian modes: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit 11026959; seed 0, budget scale 1 (plans 3, rows 6, scalePercent 100), 3.0 s.

**Passed**: 1 method(s) agree with the analytic truth, 1 differ as expected; 1 reported

**Two Gaussian modes** (bimodal): multimodal, geometry, density, analytic, two-dimensional, discrete-latent

An equal mixture of two unit Gaussians at (-3, -3) and (3, 3): the target whose mode weights a single chain cannot find.

Truth (analytic): each coordinate is the even mixture of N(-3, 1) and N(3, 1): a closed-form normal mixture CDF.

Methods whose own `check` says they apply to the compiled program (a target with no observed node is refused at the spec level): nuts, svi, gibbs, laplace, importance. Not applicable: em (no params to estimate); tempered-smc (no single markov loop, and discrete latent sites prevent tempering); particle-filter (no single markov loop); abc-smc (no observed sites); conjugate (no conjugate structure for z, x); gaussian (the program has discrete latent sites); enumeration (continuous latent sites (x)); chain (the program needs exactly one markov loop); abc-rejection (no observed sites).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | reported | converged (2001 steps) | 0 divergences; R-hat high: x[1] | one mode per chain: the weights are those of the chains (R-hat is poor), not one half each |
| gibbs | expected to differ | completed (2300 steps) | - | the component given the point, the point given the component: stuck in the mode it starts in |
| importance | must agree | completed (10 steps) | weights ESS 20000 | no observed node: the prior is the target and the weights are all one, so the draws are exact |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | x[0] | 2.98 | 1.03 | 0.942 | 84.8 | 0.325 | 0.499 | 0.942 | 1.91 | 857 |
| nuts | x[1] | 3 | 0.99 | 0.95 | 91.9 | 0.313 | 0.498 | 0.95 | 1.89 | 916 |
| gibbs | x[0] | 3.04 | 1.02 | 0.963 | 133 | 0.322 | 0.5 | 0.963 | 2.2 | 2000 |
| gibbs | x[1] | 2.99 | 0.997 | 0.946 | 127 | 0.315 | 0.498 | 0.946 | 2.19 | 1800 |
| importance | x[0] | 0.0245 | 3.16 | 0.00773 | 1.09 | 1 | 0.00711 | 0.00859 | 0.000452 | 20000 |
| importance | x[1] | 0.014 | 3.16 | 0.00444 | 0.629 | 0.998 | 0.00432 | 0.00625 | 0.0003 | 20000 |

# svi/bimodal

Two Gaussian modes: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 547817e0; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 2.6 s.

**Passed**: 1 method(s) agree with the analytic truth, 1 differ as expected; 0 reported

**Two Gaussian modes** (bimodal): multimodal, geometry, density, analytic, two-dimensional, discrete-latent

An equal mixture of two unit Gaussians at (-3, -3) and (3, 3): the target whose mode weights a single chain cannot find.

Truth (analytic): each coordinate is the even mixture of N(-3, 1) and N(3, 1): a closed-form normal mixture CDF.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: ep (z: 'z' is categorical: a discrete latent is mapped only for Bernoulli, with a table over scalar nodes; x: depends on 'z', which could not be compiled); vmp (z: 'z' is categorical: a discrete latent is mapped only for Bernoulli, with a table over scalar nodes; x: depends on 'z', which could not be compiled); bp (belief propagation sums over finite discrete variables, and 'x' is continuous); em (observed nodes have no likelihood); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (a spec has no recurrences yet (the filter runs over a markov loop of a program)); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); gaussian ('z' is categorical, not Gaussian); enumeration (observed nodes have no likelihood); chain (a spec has no recurrences yet (the chain is a markov loop of a program)); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| svi-mixture | must agree | completed (1500 steps) | - | a two-component Gaussian mixture guide with its component enumerated starts its components on opposite sides of their centre and finds both modes |
| svi-stl | expected to differ | completed (1500 steps) | - | one Gaussian is mode-seeking: it settles on one of the two modes (or straddles them) and misses the other |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| svi-mixture | x[0] | -0.125 | 3.09 | -0.0395 | -2.55 | 0.978 | 0.0475 | 0.0552 | 0.0457 | 4000 |
| svi-mixture | x[1] | -0.0608 | 3.16 | -0.0192 | -1.22 | 1 | 0.0184 | 0.0208 | 0.00345 | 4000 |
| svi-stl | x[0] | -3 | 1 | -0.95 | -190 | 0.317 | 0.499 | 0.95 | 2.53 | 4000 |

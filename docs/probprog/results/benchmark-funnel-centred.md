# benchmark/funnel-centred

Neal's funnel, centred: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 8.9 s.

**Passed**: 0 method(s) agree with the analytic truth, 2 differ as expected; 0 reported

**Neal's funnel, centred** (funnel-centred): funnel, geometry, divergences, centred, density, analytic, hierarchical-prior

Neal's funnel in ten dimensions, written directly: v ~ N(0, 9) and nine coordinates x_i ~ N(0, e^v). The neck defeats a fixed step size.

Truth (analytic): v is N(0, 9) exactly; x_i | v ~ N(0, e^v) is a scale mixture with mean 0 and sd exp(2.25), its CDF the empirical one of 100000 iid draws of the exact sampler.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, vmp, laplace, importance, tempered-smc, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of x depends on another node); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); ep-multistart (ep-multistart starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable); bp (belief propagation sums over finite discrete variables, and 'v', 'x' are continuous); gibbs (no discrete or conjugate conditionals); em (observed nodes have no likelihood); particle-filter (the model has no single scan); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); gaussian (the scale of 'x' depends on a latent); enumeration (observed nodes have no likelihood); chain (the model has no single scan); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | expected to differ | converged (1001 steps) | 2 divergences; R-hat high: v, x[0], x[1], ...; ESS low: v, x[0], x[1], ... | divergent transitions in the neck and too small a spread of v: the sampler never reaches the narrowest part (the non-centred funnel has none) |
| svi | expected to differ | completed (1500 steps) | - | a mean-field normal cannot be a funnel: v and x are dependent in scale, not in location |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | v | -0.443 | 2.71 | -0.148 | -0.475 | 0.903 | 0.0944 | 0.168 | 0.043 | 8.43 |
| svi | v | 0.211 | 0.496 | 0.0705 | 26.9 | 0.165 | 0.379 | 0.667 | 3.3 | 4000 |
| svi | x[0] | 0.0246 | 1.18 | 0.0026 | 1.32 | 0.124 | 0.087 | 0.17 | 0.705 | 4000 |

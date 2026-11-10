# svi/bimodal

Two Gaussian modes: the variational methods (Pathfinder, sticking the landing, structured, flow and mixture guides, natural gradients) against the truth.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (plans 2, rows 3, scalePercent 100), 2.5 s.

**Passed**: 1 method(s) agree with the analytic truth, 1 differ as expected; 0 reported

**Two Gaussian modes** (bimodal): multimodal, geometry, density, analytic, two-dimensional, discrete-latent

An equal mixture of two unit Gaussians at (-3, -3) and (3, 3): the target whose mode weights a single chain cannot find.

Truth (analytic): each coordinate is the even mixture of N(-3, 1) and N(3, 1): a closed-form normal mixture CDF.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, gibbs, laplace, importance, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of z is categorical, not normal or multivariateNormal); ep (x: arithmetic on the discrete variable 'z': only Gaussian values add up); vmp (x: arithmetic on the discrete variable 'z': only Gaussian values add up); bp (belief propagation sums over finite discrete variables, and 'x' is continuous); em (observed nodes have no likelihood); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (no observed nodes); conjugate (observed nodes have no likelihood); gaussian ('z' is categorical, not Gaussian); enumeration (observed nodes have no likelihood); chain (the model has no single scan); abc-rejection (no observed nodes); svi-conjugate-natural (observed nodes have no likelihood).

| method      | expectation        | run                    | diagnostics | why                                                                                                                                               |
| ----------- | ------------------ | ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| svi-mixture | must agree         | completed (1500 steps) | -           | a two-component Gaussian mixture guide with its component enumerated starts its components on opposite sides of their centre and finds both modes |
| svi-stl     | expected to differ | completed (1500 steps) | -           | one Gaussian is mode-seeking: it settles on one of the two modes (or straddles them) and misses the other                                         |

| method      | quantity | mean   | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL     | ESS  |
| ----------- | -------- | ------ | ----- | --------------- | ------ | -------- | ------ | ------- | ------ | ---- |
| svi-mixture | x[0]     | 0.0806 | 3.28  | 0.0255          | 1.55   | 1.04     | 0.0422 | 0.0689  | 0.0383 | 4000 |
| svi-mixture | x[1]     | -0.177 | 3.13  | -0.0559         | -3.58  | 0.989    | 0.0437 | 0.0564  | 0.0243 | 4000 |
| svi-stl     | x[0]     | 2.99   | 0.988 | 0.945           | 191    | 0.312    | 0.499  | 0.945   | 2.53   | 4000 |

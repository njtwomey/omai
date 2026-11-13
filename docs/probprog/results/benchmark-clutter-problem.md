# benchmark/clutter-problem

The clutter problem: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-13 at commit 2cabaeb2; seed 0, budget scale 1 (plans 3, rows 3, scalePercent 100), 2.9 s.

**Passed**: 0 method(s) agree with the quadrature truth; 3 reported

**The clutter problem** (clutter-problem): robust, mixture, discrete-latent, ep, one-dimensional, multimodal, quadrature

Minka's robust location model with a distractor cluster: observations from a normal around an unknown location mixed with wide clutter, whose one-dimensional posterior has two modes and whose exact form is a quadrature; the EP showcase.

Truth (quadrature): a grid of 12001 values of theta over [-60, 60] (spacing 0.0100), the product of the prior and the n mixture densities, the clutter indicators summed out in closed form; the evidence is the trapezoid integral. Log evidence -36.2704.

Methods whose own `check` says they apply: nuts, hmc, rwm, mala, barker, slice, sgld, sghmc, mclmc, aies, chees, parallel-tempering, ensemble-slice, svi, ep, ep-multistart, vmp, gibbs, laplace, importance, abc-rejection, svi-trace-graph, svi-trace-enum, svi-stl, svi-renyi, svi-structured, svi-mixture, svi-natural, pathfinder. Not applicable: elliptical-slice (the prior of z is bernoulli, not normal or multivariateNormal); ep-grid (ep-grid needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none); bp (belief propagation sums over finite discrete variables, and 'theta' is continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for theta, z); gaussian ('z' is bernoulli, not Gaussian); enumeration (continuous latent nodes (theta)); chain (the model has no single scan); svi-conjugate-natural (no usable conjugate structure for theta, z).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | reported | converged (2001 steps) | 0 divergences; R-hat high: theta; ESS low: theta | NUTS with the clutter indicators summed out: a one-dimensional bimodal posterior, so how the chains share the modes is the finding |
| laplace | reported | converged (6 steps) | - | one normal at the mode it finds: misses the second mode and the mass between them |
| svi | reported | completed (1500 steps) | - | a mean-field normal on theta: one normal for a bimodal posterior |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | theta | 4.84 | 2.52 | -0.671 | -1.19 | 1.09 | 0.305 | 0.671 | 0.192 | 3.75 |
| laplace | theta | 2.81 | 0.463 | -1.55 | -490 | 0.2 | 0.721 | 1.55 | 3.84 | 4000 |
| svi | theta | 2.82 | 0.523 | -1.54 | -432 | 0.226 | 0.72 | 1.54 | 3.84 | 4000 |

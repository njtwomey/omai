# benchmark/clutter-problem

The clutter problem: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 3, rows 3, scalePercent 100), 3.7 s.

**Passed**: 0 method(s) agree with the quadrature truth; 3 reported

**The clutter problem** (clutter-problem): robust, mixture, discrete-latent, ep, one-dimensional, multimodal, quadrature

Minka's robust location model with a distractor cluster: observations from a normal around an unknown location mixed with wide clutter, whose one-dimensional posterior has two modes and whose exact form is a quadrature; the EP showcase.

Truth (quadrature): a grid of 12001 values of theta over [-60, 60] (spacing 0.0100), the product of the prior and the n mixture densities, the clutter indicators summed out in closed form; the evidence is the trapezoid integral. Log evidence -36.2704.

Methods whose own `check` says they apply: nuts, svi, gibbs, laplace, importance, abc-rejection. Not applicable: ep (y: the scale of 'y' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)); vmp (y: the scale of 'y' depends on a random value that is not the precision idiom: write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)); bp (belief propagation sums over finite discrete variables, and 'theta' is continuous); em (no params to estimate); tempered-smc (discrete latent nodes (z) prevent tempering); particle-filter (the model has no single scan); abc-smc (discrete latent nodes are not supported by ABC-SMC); conjugate (no usable conjugate structure for theta, z); gaussian ('z' is bernoulli, not Gaussian); enumeration (continuous latent nodes (theta)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | reported | converged (2001 steps) | 0 divergences; R-hat high: theta; ESS low: theta | NUTS with the clutter indicators summed out: a one-dimensional bimodal posterior, so how the chains share the modes is the finding |
| laplace | reported | converged (8 steps) | - | one normal at the mode it finds: misses the second mode and the mass between them |
| svi | reported | completed (1500 steps) | - | a mean-field normal on theta: one normal for a bimodal posterior |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | theta | 6.55 | 2.31 | 0.0675 | 0.193 | 0.995 | 0.0579 | 0.0716 | 0.0171 | 8.11 |
| laplace | theta | 7.86 | 0.645 | 0.632 | 144 | 0.278 | 0.288 | 0.632 | 1.32 | 4000 |
| svi | theta | 2.98 | 0.472 | -1.47 | -457 | 0.204 | 0.72 | 1.47 | 3.85 | 4000 |

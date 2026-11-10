# benchmark/bayesian-neural-network

Bayesian neural network: the planned inference methods against the exact, quadrature or analytic truth.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (plans 4, rows 20, scalePercent 100), 83.4 s.

**Passed**: 0 method(s) agree with the reference truth; 4 reported

**Bayesian neural network** (bayesian-neural-network): neural-network, regression, non-identifiable, reference, uncertainty

A one-hidden-layer tanh network with Gaussian priors on a noisy sine with a gap in the inputs: no closed form, a stored long-NUTS reference for the output, and the predictive uncertainty that should grow away from the data.

Truth (reference): the mean and standard deviation of the network output at five test inputs from a long NUTS run (four chains, 1000 warm-up and 2500 draws each, key(7), repeated by `bnnReferenceRun`; no divergences, R-hat at most 1.0014 and ESS above 3000 for the five function values); stored, not draws; the weights have no single-mode marginals (hidden-unit permutation and sign symmetries), so the function values are compared.

Methods whose own `check` says they apply: nuts, svi, laplace, importance, tempered-smc, abc-smc, abc-rejection. Not applicable: ep (y: operator 'tanh' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation); fstar: operator 'tanh' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); vmp (y: operator 'tanh' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation); fstar: operator 'tanh' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)); bp (belief propagation sums over finite discrete variables, and 'w1', 'b1', 'w2', 'b2' are continuous); gibbs (no discrete or conjugate conditionals); em (no params to estimate); particle-filter (the model has no single scan); conjugate (no usable conjugate structure for w1, b1, w2, b2); gaussian (the mean of 'y' is not affine in the latents); enumeration (continuous latent nodes (w1, b1, w2, b2)); chain (the model has no single scan).

| method | expectation | run | diagnostics | why |
| --- | --- | --- | --- | --- |
| nuts | reported | converged (1001 steps) | 2 divergences; R-hat high: w1[0], w1[1], w1[2], ...; ESS low: w1[0], w1[1], w1[2], ... | consistent but multimodal in the weights (unit permutations): the function values are what to compare |
| tempered-smc | reported | converged (8 steps) | weights ESS 727 | tempered particles over ten weights: finds several symmetric modes |
| laplace | reported | completed (194 steps) | - | a Gaussian at one mode of a posterior with many symmetric ones: overconfident away from the data |
| svi | reported | completed (1500 steps) | - | a mean-field normal on the weights: overconfident, the output spread too small |

| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nuts | fstar[0] | -0.22 | 0.437 | -0.0202 | -0.332 | 0.971 | 0.0668 | 0.122 | 0.0521 | 257 |
| nuts | fstar[1] | -0.902 | 0.0667 | 0.0285 | 0.851 | 1 | 0.0375 | 0.0383 | 0.0132 | 897 |
| nuts | fstar[2] | 0.297 | 0.255 | 0.0061 | 0.151 | 0.966 | 0.0263 | 0.0451 | 0.00941 | 571 |
| nuts | fstar[3] | 0.824 | 0.0824 | -0.0489 | -1.2 | 0.981 | 0.0658 | 0.121 | 0.0277 | 577 |
| nuts | fstar[4] | 0.188 | 0.365 | 0.118 | 2.28 | 0.97 | 0.119 | 0.224 | 0.0695 | 354 |
| tempered-smc | fstar[0] | -0.72 | 0.146 | -1.13 | -111 | 0.324 | 0.706 | 1.15 | 2.74 | 1000 |
| tempered-smc | fstar[1] | -0.798 | 0.135 | 1.59 | 24.8 | 2.03 | 0.653 | 1.77 | 1.93 | 1000 |
| tempered-smc | fstar[2] | 0.0653 | 0.275 | -0.873 | -26.5 | 1.04 | 0.599 | 0.917 | 1.8 | 1000 |
| tempered-smc | fstar[3] | 0.909 | 0.11 | 0.969 | 23.4 | 1.31 | 0.635 | 1.02 | 1.89 | 1000 |
| tempered-smc | fstar[4] | 0.164 | 0.233 | 0.0531 | 2.72 | 0.618 | 0.32 | 0.484 | 1.78 | 1000 |
| laplace | fstar[0] | -0.21 | 0.648 | 0.000569 | 0.025 | 1.44 | 0.0851 | 0.296 | 0.0533 | 4000 |
| laplace | fstar[1] | -0.97 | 0.408 | -1 | -10.3 | 6.14 | 0.252 | 2.66 | 0.479 | 4000 |
| laplace | fstar[2] | 0.154 | 0.402 | -0.537 | -22.3 | 1.53 | 0.182 | 0.615 | 0.129 | 4000 |
| laplace | fstar[3] | 0.834 | 0.353 | 0.0711 | 1.07 | 4.21 | 0.193 | 1.68 | 0.328 | 4000 |
| laplace | fstar[4] | 0.138 | 0.564 | -0.0175 | -0.738 | 1.5 | 0.0823 | 0.33 | 0.0669 | 4000 |
| svi | fstar[0] | -0.544 | 0.109 | -0.741 | -194 | 0.241 | 0.587 | 0.873 | 3.11 | 4000 |
| svi | fstar[1] | -0.793 | 0.1 | 1.66 | 69.6 | 1.51 | 0.512 | 1.66 | 0.709 | 4000 |
| svi | fstar[2] | 0.368 | 0.166 | 0.276 | 27.7 | 0.63 | 0.208 | 0.382 | 0.321 | 4000 |
| svi | fstar[3] | 0.743 | 0.0986 | -1.02 | -54.8 | 1.17 | 0.364 | 1.02 | 0.388 | 4000 |
| svi | fstar[4] | 0.486 | 0.108 | 0.91 | 200 | 0.287 | 0.618 | 0.978 | 3.03 | 4000 |

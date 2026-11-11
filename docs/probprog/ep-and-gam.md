# Expectation propagation and the Bayesian additive model

A note on why EP did not run on the Bayesian additive model (the `bayesian-gam` page) or on linear regression with a learned
noise level, whether that is a limit of EP or of the setup, and what has changed. Written 2026-11-11; the checks quoted are the
library's own verdicts (`factorGraphVerdict`) and the tests named at the end.

## Short answer

It is a gap in the message-passing library's rules, not a limit of EP. EP is not restricted to Gaussian models: it only needs,
for each factor, the moments of "the factor times the cavities", and where those have no closed form they can be computed by a
low-dimensional quadrature (the library already does that for the logistic and probit links, products and clutter). What was
missing were a few factor rules. One is now written (an unknown noise level); one is the real remaining gap (an unknown
smoothing parameter on a whole vector of coefficients).

## The model, as regression

A GAM with fixed knots is a linear regression on features: `y = B beta + noise`, where the columns of `B` are the spline bases.
The smoothness penalty is a Gaussian prior on `beta` with precision `lambda D` (`D` the penalty matrix, `lambda` the smoothing
parameter). Given `lambda` and the noise level `sigma` the model is linear-Gaussian: the posterior of `beta` is Gaussian in
closed form, and EP, VMP, Laplace and the exact Gaussian method all agree with it. That is the page's `fixed` variant (EP,
VMP and `gaussian` run, within 0.004 response sd of the exact curve).

The difficulty is the hyperparameters. Once `lambda` and `sigma` are unknown the joint posterior of `(beta, lambda, sigma)`
is not Gaussian (it is a scale mixture of Gaussians), so a method has to approximate it: this is where the factor rules matter.

## The three gaps the library had

1. **The noise written as `exp(log sigma)` or with a half-normal sd.** Message passing has no rule for `exp` of a random
   value, and the noise scale has to be written in the precision idiom `1 / sqrt(tau)` with `tau` gamma-distributed (the
   `gaussian-gamma-precision` factor). The default page model is written with log-normal hyperpriors, so nothing applies.
   This is a choice of parameterisation of the model, not of the method.
2. **EP for an unknown noise level.** The `gaussian-gamma-precision` factor had a VMP rule only (its own documentation said
   "expectation propagation would need Student-t moments and is not provided"). **Added.** For an observed value and a
   Gaussian mean it integrates the mean out (`N(y; m, v + 1/lambda)`), computes the moments of the precision on a grid in
   `log lambda`, and gives the mean a Gaussian site and the precision a Gamma site. Checked against the exact posterior on a
   grid: an intercept-only model, and a two-feature linear regression with learned noise, both agree (coefficients to
   about 0.05, the noise precision within 15%); the rule also covers a latent Gaussian (a population of values with an
   unknown spread: abilities, group effects), checked against the exact posterior on a grid; on the page's `gammaNoise` variant EP agrees with the exact reference
   (mean error under 0.03 response sd, band width within 15%).
3. **A random smoothing parameter on a vector.** `lambda` scales the precision of every coefficient at once
   (`beta_j ~ N(0, 1 / (lambda d_j))`). The compiler refused a random scale on a vector ("a joint Gaussian vector has one
   covariance, so its scale must be known"). **Built, 2026-11-11** (both routes of `ep-plan.md` section 3): the factor
   `gaussian-mv-precision-scale` takes the whole coefficient vector (dense, so its correlations are kept) and the one gamma
   smoothing parameter, integrates the vector out against its Gaussian cavity on a grid in `log lambda` and sends the vector the
   moment-matched mixture of the Gaussian conditionals; and `gaussian-gamma-precision` takes a `scale` and `offset` (the
   precision `scale lambda + offset`), the per-coefficient route. The model is `hyper: 'gammaSmooth'` of `bayesianGam` (the
   smoothing parameter and the noise are gamma latents, the coefficients are written in the eigenbasis of the penalty). Against the
   exact reference (two-dimensional quadrature over the smoothing parameter and the noise; sine data, 60 rows, 10 coefficients):

   |                                          | mean curve error (response sd) | band width ratio | log smoothing  | log noise sd    |
   | ---------------------------------------- | ------------------------------ | ---------------- | -------------- | --------------- |
   | exact reference                          |                                | 1                | 2.716 +- 0.524 | -1.688 +- 0.096 |
   | EP, dense vector (route 1, the default)  | 0.004                          | 1.015            | 2.664 +- 0.533 | -1.688 +- 0.102 |
   | EP, one scalar per coefficient (route 2) | 0.005                          | 1.19             | 2.828 +- 0.493 |                 |

   The dense route is accurate to a few percent; the per-coefficient one has the right curve and smoothing parameter and a
   band 19% too wide (the loopy sum over coefficients), and is the form that carries ARD and bases with their own precisions.

## The non-centred forms (2026-11-11, later)

The default model (log-normal hyperpriors) and ARD write the coefficient non-centred, `z / sqrt(precision)` with `z` standard
normal and the precision a known map of a latent (`exp(log lambda) d_j + n_j` of a Gaussian, or `s alpha_j + 1 - s` of a gamma
relevance). That is not a refusal of EP any more: `operators/functions.ts` has two additions. The unary `apply` factor takes a
chain of steps (`exp`, then `mul`, `add`, `sqrt`, the reciprocal) and a gamma input, and the compiler collapses the arithmetic on
the output of an `apply` into one chain from the latent (no Gaussian approximation of the intermediate values; the factors left
unread are pruned). The quotient of a variable by such a chain is the `apply-product` factor `y = z f(x)`: given `x` the factor is
linear in `z`, so the integral over `z` is closed form and what remains is a one-dimensional integral over `x` (a panelled grid
for a Gaussian `x`, a grid in `log x` for a gamma). A first version with a separate `apply` and `product` (a Gaussian
approximation of `f(x)` in between) gave the curve but a smoothing parameter 1.2 to 3.2 log units off and answers that moved
with the damping; the fused factor does not.

Measured on the page's data (sine, 60 rows, 10 coefficients), against the exact reference or a long sampler:

| form                                   | offered (EP) | mean curve error (response sd) | band width ratio | log smoothing          | note                                     |
| -------------------------------------- | ------------ | ------------------------------ | ---------------- | ---------------------- | ---------------------------------------- |
| penalty, `gammaSmooth` (dense vector)  | yes          | 0.004                          | 1.015            | 2.664 +- 0.533 (2.716) | converges in 21 sweeps, 0.65 s           |
| penalty, `logNormal` (non-centred)     | yes          | 0.005                          | 1.17             | 2.783 +- 0.630 (2.533 +- 0.619) | about 80 sweeps, 3 s; never flags convergence |
| penalty, `gammaNoise`, `fixed`         | yes          | under 0.03                     | 0.85 to 1.15     |                        | (tests of the earlier forms)             |
| ARD (the four hyper settings are one model) | yes     | 0.011 at worst against NUTS    | 1.09             |                        | 64 sweeps, 1.8 s; bases in use within 0.05 of NUTS |
| spike and slab, all four               | yes          |                                |                  |                        | (earlier work)                           |

The band is wide because the posterior of a non-centred coefficient comes back as a marginal per coefficient (the dense form's one
vector keeps their correlations): gap `non-centred-scale` in `ep-gaps.ts`. Variational message passing still refuses these
forms (its non-conjugate message for a scale is degenerate).

## Optimising instead

The user's regression view is right: for fixed `lambda` and `sigma` nothing needs approximating, and the classical GAM
answer is to optimise the hyperparameters: choose `lambda` and `sigma` by maximising the marginal likelihood (REML,
`omai-engine/gam`, `outer.ts`) and use the Gaussian posterior at that value. Fully Bayesian versions integrate over them
instead (an INLA-style Laplace approximation on `(log lambda, log sigma)` of the marginal likelihood with `beta`
integrated out exactly). Neither is in the method list of the probabilistic layer: the `laplace` kind there is a Laplace
fit at the joint mode of everything, which for a hierarchical model sits at a degenerate small smoothing parameter (on the
page: log lambda -0.86 +- 2.2 against exact 2.57 +- 0.61, a band 25 times too wide). A method kind that marginalises the
Gaussian block and optimises or Laplace-approximates the hyperparameters would be the cheap, accurate method for this
model; it is a missing method, not a missing theory.

## What the page offers per variant

| prior and hyperpriors                                                        | what runs                                                                                     |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| smoothness penalty, log-normal hyperpriors                                   | the samplers, SVI, Pathfinder, Laplace (poor), importance, and EP (the `apply-product` factor)  |
| penalty, gamma smoothing parameter and gamma noise precision (`gammaSmooth`, the page's default) | the above, plus VMP, and EP first (the dense scale factor), against the exact reference |
| penalty, Gamma noise precision, smoothing fixed (`gammaNoise`)               | the above, plus VMP and EP                                                                    |
| penalty, smoothing and noise fixed (`fixed`)                                 | EP, VMP, the exact Gaussian method (all exact), and the rest                                  |
| spike and slab, ARD                                                          | the samplers and variational methods, and EP (mixture sites; the `apply-product` factor for ARD) |

## Next steps, in order of value

1. ~~The EP rule and compiler mapping for a Gaussian vector with a random precision scale (gap 3)~~: done, see above.
2. A method kind that marginalises the Gaussian block (empirical Bayes or Laplace on the hyperparameters).
3. ~~A Gamma-precision form of the default model~~: done (`gammaSmooth`, now the page's default); and the non-centred form runs under EP anyway.

Tests: `probprog/model/factors.test.ts` ("runs under expectation propagation ...", "a linear regression with an unknown noise
level ..."), `probprog/models/bayesian-gam.test.ts` (the verdicts and the agreement with the exact reference per variant).

## Item response and other hierarchies

The Rasch model `P(y_ij = 1) = sigmoid(theta_i - b_j)` is a logistic regression with a one-dimensional embedding for each
person and each item, and EP runs on it as it stands (factors `prior`, `difference`, `logistic`; verdict ok for EP and VMP).
With the population spread of the abilities learned, the model is a Gaussian with a random precision, so it needs the
precision idiom: write `theta_i ~ N(0, 1 / sqrt(tau))` with `tau` gamma-distributed (a half-normal sd is refused), and the
rule above (now for a latent value as well as an observed one) lets EP and VMP run it.

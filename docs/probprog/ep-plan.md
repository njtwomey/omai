# Expectation propagation: what we have, what is missing, the plan

Written 2026-11-11 after an audit of the factor-graph compiler (`probprog/model/factors.ts`) and the engine's operators
(`message-passing/operators/`). The aim, from the user: EP in all or at least most of the examples and models, with spike and slab,
gates and the spline (GAM) case done properly, and a view of the other distributions we are likely to meet. EP is a showcase
capability and worth real time; a refusal is usually a missing rule or the wrong way of writing the model, not a limit of EP.
Companion note on the GAM case: `ep-and-gam.md`.

## 1. Where we are (measured, not remembered)

The engine has 35 factor operators and eight message families (`gaussian`, `gaussian-mv`, `gamma`, `beta`, `bernoulli`,
`discrete`, `dirichlet`, `normal-gamma`). The compiler maps these likelihoods: `normal`, `bernoulli` (probit, logistic,
greater-than), `binomial`, `poisson` (with an exposure), `categorical`; priors on `normal`, `gamma`, `exponential`, `beta`,
`dirichlet`, `bernoulli`, `categorical`; affine maps of Gaussians; products of two random values; max and min; a gate
`at(components, z)` / `where(z, a, b)` when the components are known values or fresh variables.

Compiling each of the 27 benchmark models and asking for the EP verdict: **EP runs on 9** (beta-Bernoulli, normal-normal,
gamma-Poisson, logistic and probit regression, the Gaussian mixture, TrueSkill, the AR(1) state space, item response scoring).
The other 18 are refused for a small number of reasons:

| blocked by | models |
| --- | --- |
| `exp`, `square`, `tanh` of a random value (no unary-function rule) | poisson regression, funnel (non-centred), banana, Bayesian neural network |
| a prior with no message family: half-normal, half-Cauchy, log-normal | eight schools (both forms), radon, negative binomial, GP (length scale), funnel |
| a likelihood not mapped: negative binomial, Student-t, Laplace | negative binomial regression, (robust regression page) |
| a random scale that is not the precision idiom | eight schools, funnel (centred), stochastic volatility, clutter |
| a constraint (`ordered`) | ordered Gaussian mixture |
| a dense Gaussian vector (`multivariateNormal`) and opaque custom functions (kernel Cholesky, stick breaking) | GP regression (both), stick-breaking mixture |
| a gate over random components, a discrete variable in arithmetic | Gaussian HMM, bimodal |
| a Dirichlet with no declared size (the compiler throws instead of refusing: a bug) | LDA |

Added on 2026-11-11: an EP rule for a Gaussian with a gamma-distributed precision (`gaussian-gamma-precision`), for an
observed or a latent value (linear regression with an unknown noise level, a population with an unknown spread; checked against
the exact posterior on a grid).

## 2. The catalogue of what we need, by family of rule

EP needs, for each factor, the moments of "the factor times the cavities" projected on the family of the variables. Where there
is no closed form it is a one- or two-dimensional quadrature, which is how the logistic link, the product and clutter already
work. So the work is a small number of rules, plus compiler mappings. In order of leverage:

**A. A generic numeric likelihood on a Gaussian linear predictor** (one operator, a registry of log-densities): Student-t and
Cauchy (robust regression), Laplace, Poisson with a log link (`exp` link), negative binomial with a log link, Weibull and
log-normal survival (log link), ordinal probit, binomial with a logistic link, asymmetric Laplace (quantile regression),
zero-inflated forms (with the gate, C). The tilted moments are a 1-D quadrature of `log p(y | u)` against the cavity of `u`, the
template of `operators/links.ts`. One implementation, many likelihoods; the compiler maps `observe y ~ D(g(affine))` for the
`(D, g)` pairs in the registry. This unblocks poisson regression, negative binomial, robust regression and survival models.

**B. Unary functions of a random value** (`apply`: `exp`, `square`, `tanh`, `relu`, `abs`, `softplus`, `sigmoid`): the output is
matched by moments under the cavity of the input, the message back is the tilted moment of the input (monotone functions:
1-D; `relu` has a closed form). This is the probabilistic-backprop construction for neural networks, and it is what the
non-centred funnel (`exp`), the banana (`square`), the Bayesian neural network (`tanh`), log-normal priors and the
stochastic-volatility observation (`N(0, exp(h))`) need. The SV observation is a special case: a Gaussian with variance
`exp(h)`, one variable's tilted moments by quadrature.

**C. Gates and mixtures (Minka and Winn, "Gates")**: a discrete latent `z` selecting among sub-models. Spike and slab is the
simplest gate: `z_j ~ Bernoulli(pi)`, `beta_j | z_j` narrow or wide. Two parts:
  1. *A mixture-of-Gaussians prior factor* on a scalar (closed form: a Gaussian cavity times a K-component Gaussian mixture is a
     K-component Gaussian mixture, so the moments are exact; no quadrature). This is spike and slab, scale mixtures, a
     Student-t as a finite mixture, outlier models, and the `normalSpikeSlab` family of the model layer maps to it directly.
  2. *The gate proper*: the model layer's `gate` node (listed as not built) and its EP/VMP messages, so a branch with its own
     random variables works (the Gaussian HMM, the change-point models, zero-inflation, mixtures of regressions). The
     message to `z` is the ratio of the branches' evidence under their cavities; the branch variables get messages weighted by
     `z`'s cavity.
  Plus `ordered` as a chain of `greater-than` factors (the factor exists, the compiler does not map the constraint).

**D. Priors with no family: half-normal, half-Cauchy, inverse-gamma, log-normal, uniform, truncated normal**: a numeric *prior*
factor (project the density once onto the variable's family: Gamma for a positive scale, Gaussian for a real one), plus the
known conjugate identities where they exist (a half-Cauchy as an inverse-gamma mixture; an inverse-gamma on a variance is a
gamma on a precision). Scales that are not in the precision idiom (`sigma ~ halfNormal`, `exp(log sigma)`) are then handled by
B and D together rather than refused.

**E. Dense Gaussian vectors**: map `latent f ~ multivariateNormal(0, K)` with `K` a data matrix to the engine's `gaussian-mv`
prior (the engine and the gp domain's own EP already work with dense precisions), and make kernel Gram matrices first-class
(the compiler currently sees `rbfCholeskyApply` as an opaque custom function). Kernel hyperparameters stay outside EP at first
(a grid or an outer loop over them, as the gp domain does for its marginal likelihood). This is GP classification, the EP
showcase.

**F. Random scales on vectors** (the GAM, section 3).

**G. Later**: correlated random effects (an LKJ or Wishart covariance: a new family or the Cholesky/correlation
parameterisation with a numeric 1-D prior), stick-breaking and Dirichlet processes (custom functions made first-class),
softmax/multinomial logit (a (K-1)-dimensional tilted moment: a delta-method or Laplace rule), survival with censoring
(truncation factors exist), matrix factorisation (products exist), HMM forward-backward (the `table` factors and BP exist).

## 3. The spline (GAM) case

The user's instinct is right: it should be doable with an accurate approximation. A spline GAM is linear regression on features
`B` with a Gaussian prior of precision `lambda D` on the coefficients. Given `lambda` and the noise it is exactly Gaussian (EP is
exact); the only non-Gaussian part is the smoothing parameter, a single positive scalar that scales the precision of the whole
vector. Two routes, both modest:

1. *Scale-mixture factor on the vector* (the principled one). The factor `f(beta, lambda) = N(beta; 0, (lambda K + P0)^-1)`, with
   `K` the penalty and `P0` a fixed precision for the unpenalised directions. With a Gaussian cavity `N(m, V)` on `beta` and a
   Gamma cavity on `lambda`, integrating `beta` out gives `Gamma(lambda) N(m; 0, V + (lambda K + P0)^-1)`: a function of the
   scalar `lambda` only, so the moments of `lambda` come from a grid in `log lambda` (the same device as the rule added today),
   and `beta`'s moments are the mixture over that grid of the Gaussian conditionals (dense `p x p`, `p` is 10 to 40). The
   Gamma site is a projection, the Gaussian site is dense. Cost `O(p^3)` per grid node, so about a second at worst, `O(p)`
   in the penalty's eigenbasis when the cavity is diagonal. The approximation is: `beta | y` Gaussian, `lambda | y` Gamma, which
   is accurate when the posterior of `log lambda` is close to a log-gamma, as it usually is. The compiler change: allow a random
   scale on a packed vector with a known constant matrix.
2. *Per-coefficient route* (cheaper, covers ARD and basis selection too). Work in the penalty's eigenbasis, where the prior is
   `gamma_j ~ N(0, 1 / (lambda d_j))`: the existing rule with a known constant multiplier `d_j` on the precision (a one-line
   generalisation of `precisionTilt`), with the coefficients as scalar variables (an "unpacked" compilation of the matmul: one
   sum-of-products row per observation) instead of one packed vector. ARD is the same with a precision per coefficient; basis
   selection (spike and slab on the weights) is C1. The coupling of `lambda` across the `p` Gaussians is carried by EP's
   iteration; `p` factors on one Gamma variable is the same structure as the hierarchy that already works.

Recommendation: route 2 first (it reuses the rule just added, unlocks ARD and spike and slab on bases, and needs a compiler
option rather than a new rule), route 1 when a dense, correlated posterior of the coefficients matters. The unknown noise level
is already handled. The rest of the page's model (log-normal hyperpriors) is D and B.

## 4. Other distributions and use cases to plan for

| use case | what it needs | in section 2 |
| --- | --- | --- |
| robust regression, outlier models | Student-t, Cauchy, Laplace, a mixture of inliers and outliers | A, C1 |
| count data: Poisson, negative binomial, zero-inflated | exp link, negative binomial, a gate | A, B, C |
| survival, reliability, censoring | Weibull, log-normal, exponential with a log link; truncation | A, B |
| ordinal and ranking (ratings, Bradley-Terry, Plackett-Luce) | ordinal probit, logistic on differences (IRT is this) | A, existing |
| quantile regression | asymmetric Laplace | A |
| variable selection, sparse regression | spike and slab, ARD, horseshoe (a scale mixture: half-Cauchy via inverse gamma) | C1, D |
| hierarchical models with a learned spread | precision idiom, half-normal and half-Cauchy sds | D, done for the gamma form |
| correlated random effects, varying slopes | LKJ or Wishart | G |
| Gaussian processes: classification, count and survival GPs | dense MVN prior, kernel Gram data, A for the likelihood | E, A |
| Bayesian neural networks | `tanh`, `relu`, `sigmoid` on random values | B |
| state space, stochastic volatility, change points | `exp` variance, scan chains, gates | B, C2 |
| mixtures, HMMs, topic models | gates, `table` / BP, Dirichlet sizes | C2, G |
| factor models, matrix factorisation, recommenders | products of Gaussians (exist), ARD precisions | existing, F |

## 5. Plan

Two layers, infrastructure first, so the rules land on shared machinery rather than one-off operators.

**Layer 1: shared machinery (compiler and registry).** A numeric 1-D likelihood registry and operator (A); the `apply` operator
for unary functions (B); the numeric prior factor (D); a known constant multiplier on a precision; an unpacked compilation of a
matmul for a vector with per-element random scales; `multivariateNormal` with a data covariance mapped to `gaussian-mv`; the
`ordered` constraint as greater-than factors; the Dirichlet-size throw turned into a refusal (the LDA audit error).

**Layer 2: rules.** Student-t, Laplace, Poisson-log, negative binomial, Weibull, ordinal on the registry (A); `exp`, `square`,
`tanh`, `relu` (B); the two-component mixture prior and the K-component generalisation (C1); the gate (C2); the half-normal,
half-Cauchy and log-normal priors (D); the GAM routes (F); the SV observation.

**Verification standard for every rule**: (a) the tilted moments against a closed form or a high-order quadrature; (b) a small
model against the exact grid posterior; (c) a TSDoc `@example`; (d) a row in an audit test (`ep-coverage.test.ts`) that compiles
every benchmark model and records its EP verdict, so the number of models EP runs on (9 of 27 today; target at least 22) is a
tested figure that only goes up; (e) the matching example page offers EP in its dropdown without changing the page.

**Work packages** (three at a time): WP-A the registry, numeric likelihoods and `apply` (A, B); WP-B gates, mixture priors,
`ordered`, spike and slab (C); WP-C dense MVN and GP classification, the scaled precision and unpacked compilation, the GAM
routes (E, F). D is split between A and C. Each ends with its pages offering EP.

## 6. What is half-done in flight

Three example agents were building EP rules for their own pages when the work was paused (Student-t, spike and slab and ARD
for robust and sparse regression; the hierarchical Rasch and the 2PL product for item response; the GP prior mapping and the SV
observation). Their reports say what exists; the work packages above absorb it rather than redo it.

## 7. Status at the end of 2026-11-11 (built by the example agents while the plan was written)

Done, each with tests against an exact or brute-force reference (details in the commits):

- `gaussian-gamma-precision`: EP for a gamma precision on an observed or a latent Gaussian (starts from a flat cavity: a first
  version blocked on one, fixed). Linear regression with unknown noise, a population with an unknown spread, the hierarchical
  Rasch model: EP within 0.1 reference sd of NUTS.
- `mixture-site` (closed form, exact: spike and slab prior, inlier/outlier likelihood), `student-t-site`, `laplace-site`
  (quadrature), and their rows on a packed weight vector (`operators/robust.ts`); model family `normalOutliers`; the
  `messagePassing` family flag. Student-t (df fixed), Laplace, outlier mixture, spike and slab, and ARD (a precision per
  coefficient, unpacked) all agree with long NUTS runs to about 0.15 sd (ARD, spike and slab: a vague-prior workaround for
  improper first cavities: to be replaced by proper initial messages).
- `gaussian-log-scale` (a Gaussian observation whose scale is `exp` of a Gaussian state): EP on stochastic volatility with known
  `mu, phi, sigma` agrees with the exact smoother (0.13 to 0.19 sd). The AR(1) scan maps.
- A dense `multivariateNormal` prior with a constant covariance compiles to a `gaussian-mv` prior: GP classification under EP
  (probit and logistic) agrees with the gp domain's own EP (5e-3) and a long NUTS run.
- Item response: Rasch, hierarchical, under EP and VMP; 2PL with normal discriminations runs under EP (numeric `product`) but is
  inaccurate (a sign symmetry in the posterior); 2PL with log-normal discriminations needs `exp` of a random value.

Done in WP-B (2026-11-11, `agent-pp53-rulesb`), each against an exact or brute-force reference:

- The gate of Minka and Winn (`operators/gate.ts`, `gate`): a discrete variable selects among branch factors, each any operator of
  the registry over its own variables; the message to the variable is the ratio of the branches' evidence under their cavities, to a
  variable the moment-matched mixture over the branches (weighted by the variable's cavity and the evidence); VMP too (expected
  log per branch, responsibility-weighted messages). Compiler: a lookup of known values by a discrete latent (`at(means, z)`,
  `z * 6 - 3`, `gt(t, tau)` for a change point, `where(z > 0.5, 0, theta)`) is a table over `z` (a hidden Markov model's emission) or a
  gate of Gaussian branches over the random variables of its states; a categorical with probabilities `at(A, z)` is a table
  (Markov transition). The Gaussian HMM is exact (forward-backward), the bimodal target matches the mixture moments, the clutter
  problem agrees with the numeric clutter EP, a change point's posterior over the change time agrees with the exact one to 0.12.
- `ordered` (and `positive`, `nonNegative`, `greaterThan`, `lessThan`, `interval` on a Gaussian latent) as comparison factors: the
  ordered Gaussian mixture benchmark runs and matches its quadrature truth (mu[0] -1.870 +- 0.295 against -1.872 +- 0.301).
- The smoothing parameter of a penalised spline, both routes (`ep-and-gam.md`): `gaussian-gamma-precision` with `scale` and
  `offset`, and the dense `gaussian-mv-precision-scale`; the Bayesian GAM `hyper: 'gammaSmooth'` against the exact reference: mean
  curve error 0.004 response sd, band ratio 1.015, log smoothing 2.664 +- 0.533 (exact 2.716 +- 0.524).
- The horseshoe (`gamma-rate`, `gamma-product`: half-Cauchy as gamma hierarchies, a local times a global precision): exact on one
  coefficient (mean 1.8958 against 1.8952), six coefficients within about 0.4 sd of a NUTS run (the shrunk coefficients' sds are about
  half the sampler's). A Student-t likelihood with an unknown scale (a gamma mixing precision per row times the noise precision):
  location 0.616 +- 0.271 against the exact 0.635 +- 0.316, precision 2.370 against 2.364.
- Stochastic volatility with the persistence and the scale inferred needed no new rule (the `product` factor and a gamma precision
  on the latent): phi 0.937 +- 0.17 against NUTS 0.886 +- 0.125, tau 11.5 against 10.5, states within 0.2 sd.
- Start messages on a factor spec (`FactorSpec.start`, carried by the compiled graph and read by the engine's `init`) replace the
  vague prior factors the compiler added for spike and slab and random precisions: the factor's own first update replaces the vague
  message, so the model is no longer changed by them. (The operators' own flat-cavity reads, `cavityOf` in
  `gaussian-gamma-precision` and `widened` in the gate, remain.)
- The benchmark audit: EP runs on 14 of 27 (10 before: gate-ordered mixture, bimodal, Gaussian HMM, clutter).

Still open, in the order of the work packages that follow:

- WP-A (likelihood and function rules): built, see section 8 (the likelihood registry, `apply`, the projected priors, the
  negative binomial with a random dispersion). Remaining of it: model-layer distributions for the ordinal probit and the
  asymmetric Laplace (their rules are in the registry), `sqrt` and `log` of a random value, the GP length scale (a kernel
  matrix that depends on a random value), a half-Cauchy scale through the `gamma-rate` hierarchy instead of its projection.
- WP-B remainder: Student-t with learned degrees of freedom (a random shape of the mixing gamma has no exponential-family rule),
  a gate over a mix of known values and non-Gaussian random components, nested model-layer `gate` nodes with sub-models in
  branches (the engine's `gate` takes any branch operator; the model layer maps lookups of Gaussian forms only), the engine-level
  replacement of the operators' flat-cavity reads by start messages, the multi-coefficient horseshoe's shrunk-coefficient widths.
- The atoms suite and the coverage figure (`epCoverage()` as a library function with a test): done by pp51; the atoms `gate`,
  `ordered-means` and `scaled-precision` were flipped to `runs`.

## 8. WP-A, built (2026-11-11)

With WP-A alone EP ran on 19 of the 27 benchmark models; with WP-B merged it runs on 23 (`epCoverage()`; 28 of 28 atoms). Still refused: the two Gaussian process regressions (a kernel Gram matrix that depends on the random length scale, and an opaque Cholesky function), the stick-breaking mixture (an opaque custom function), LDA (the compiler throws on a Dirichlet with no size). Built, each against an exact or brute-force reference:
- A generic numeric 1-D likelihood registry (`message-passing/operators/likelihoods.ts`: Poisson and negative binomial with a log link,
  Weibull and log-normal survival (right-censored too), binomial with a logistic or probit link, ordinal probit, asymmetric
  Laplace, Student t, Laplace) behind `likelihood-site` and `likelihood-row`, integrated by the zooming panelled quadrature of
  `_tilted.ts`; the compiler maps Poisson (exposure), binomial, negative binomial (known or random dispersion), Weibull, log-normal
  (exact, on the log). Ordinal probit and asymmetric Laplace are in the registry but have no model-layer distribution yet.
- `apply` (`functions.ts`): `exp`, `square`, `tanh`, `relu` (closed form), `softplus`, `abs`, `sigmoid`; `max`/`min` with a known
  operand is a rectifier. Poisson regression, the funnels, the banana and the Bayesian neural network compile.
- Priors without a message family projected once (`projectedPrior`, `priors.ts`): half-normal, half-Cauchy, log-normal, uniform and
  truncated normal, a positive scale on its log; a latent with a random scale is written non-centred, an observed value with a
  random scale is `gaussian-scale`. Eight schools (both forms), the centred funnel and radon compile.
- `toFactorGraph(bound, { packScalars: true })`: scalar normal latents that share a predictor are one joint vector, so EP keeps
  their correlation (without it the marginal sds are too small by 2 to 3 times on Poisson and negative binomial regression).
  The mixtures are written as the gates they are (an explicit switch `z`, so EP reports P(z) and the branches): the gate between
  known values, between random values (select and index), spike and slab (a gate between a narrow and a wide Gaussian, and the
  switch-summed-out mixture site as a second setup) and clutter (a switch per observation). Checked on 36 combinations of observed
  value, inclusion probability and widths: EP's P(slab), mean and sd of the coefficient equal the closed form to 1e-4 (moment
  matching through a gate is exact); the Gaussian EP gives for a spike-plus-slab posterior is the moment-matched projection of a
  two-hump mixture, so it sits between the humps. EP runs on 24 of the 30 atoms; the other six (exp, tanh, Poisson with a log link,
  half-Cauchy, half-normal, the product of two Gaussians) show the library's reason on their pages.

## 9. The prior atlas (2026-11-11)

`ep-priors.md` is the table of every distribution we observe, its parameters, the priors and hyperpriors EP carries (exact or numeric),
how to write them and what is still planned. Built with it: inverse-gamma variances, the normal-inverse-gamma pair as one exact
normal-gamma variable, exponential and gamma observations with a gamma rate, geometric and negative binomial counts with a beta
probability, a beta or Dirichlet with a gamma or log-normal concentration (`dirichlet-concentration`), a Bernoulli or categorical switch
with a beta or Dirichlet probability (a gate with an unknown mixing weight) and its summed-out form (`mixture-weight`, `normalPair`),
a positive latent (log-normal and the like) as a rate of the log-link likelihoods. 21 atom pages (`family: 'priors'` and
`'gate weights'`), all with the model code that ran at the bottom.

## 12. The extended atoms: parameters learned (2026-11-11)

Each basic atom with fixed numbers has a twin `<name>-learned` (`extends: '<basic>'`, `recovery: {...}`) in
`ep-atoms-extended-{conjugate,scales,likelihoods}.ts`: the numbers the basic atom fixes are latents with stated priors, the data are simulated
from a stated truth (knobs: the true values, `seed`, `n`), and `atomCalibration` (`ep-extended.ts`) repeats it over the atom's seeds and
reports per parameter the share of runs whose 90% interval holds the truth. The registry test (`ep-extended.test.ts`) asserts the promised
floors. Coverage over 40 seeds (25 for the negative binomial) at the default knobs:

| atom | parameter: coverage (error in sds, rms) |
| --- | --- |
| gaussian-learned | mean 0.82 (1.10), variance 0.95 (0.79) |
| gamma-precision-learned | mean 0.82 (1.09), precision 0.97 (0.70) |
| shared-precision-learned | precision 0.85 (1.01), mean 0.85 (1.21), the 8 values 0.86 (1.18) |
| beta-bernoulli-learned | strength 1.00 (0.81), the 8 probabilities 0.90 (1.02) |
| gamma-poisson-learned | prior rate 0.90 (0.89), the 8 rates 0.89 (1.04) |
| student-t-learned | location 0.88 (1.01), precision 0.97 (0.87) |
| probit-learned | offset 0.97, slopes 0.93 and 0.88 (0.9 to 1.2) |
| logistic-learned | offset 0.90, slopes 0.82 and 0.78 (shrinkage of fixed weights, 1.3) |
| poisson-log-learned | intercept 0.88, slope 0.85 (1.05, 1.11) |
| poisson-log-dispersion-learned | intercept 0.92, slope 0.96, dispersion 0.96 |
| weibull-learned | intercept 0.97, slope 0.88 |
| skill-difference-learned | spread precision 1.00 (0.53, an informative prior), the 8 skills 0.86 (1.14) |

Gaps found. (1) `laplace-learned` is planned: the library takes a Laplace likelihood with a known scale only ("the scale of 'y' is
random"); the missing rule is a two-variable numeric site over (location, scale) in the pattern of `negative-binomial-dispersion`. The grid
posterior of the model is the reference. (2) Stochastic volatility with an unknown level, persistence and innovation precision compiles and
runs, but is not calibrated (30 seeds of 60 returns: the truth inside the 90% interval for 37% of the levels, 60% of the persistences and 47%
of the precisions, with errors of 10 to 40 posterior sds: the product factor of the persistence and the previous state, with the level as a
third coupled latent, gives overconfident and sometimes improper beliefs, and a run takes seconds), so it is not an atom; it needs a joint
Gaussian rule over (level, persistence) or a block update. (3) A Weibull with a random shape and a negative binomial with a gamma
dispersion are not mapped (the shape and the gamma form of the dispersion have no rule); the dispersion here is the half-normal
numeric site.
## 10. The gaps, as tested data

The open items of this plan (what EP cannot run, where it is less good than it should be, the infrastructure and integration tasks)
are kept as data in `packages/engine/src/lib/bayesian-examples/ep-gaps.ts` and shown at the end of the EP coverage page, each with
why, how it would be closed, its size and its status. A test (`ep-gaps.test.ts`) holds the list to the coverage table: every setup
and benchmark model EP refuses is named by a gap, and a gap that names something EP now runs is stale. Where the code lives: the
rules (`message-passing/operators`), the compiler mappings (`probprog/model/factors.ts`), the model families
(`probprog/model/dists`) and the methods (`probprog/inference`) are in the library proper; the setups, programs, coverage and
recovery helpers are still in `bayesian-examples` and are to move to `probprog/ep` (gap `integration`).
## 11. Robustness: what EP reports and what the variants do (2026-11-11, `agent-pp81-robust`)

**False convergence (found by the sparse regression with outliers).** The engine took "the beliefs moved less than `tol` over a
sweep" for convergence. An update that cannot be computed (an operator throws for an improper or empty cavity: the proposal is
`failed`, or `blocked` when the own message of a port it reads has no density) leaves the old message in place, so a run in which
a share of the updates cannot be computed has a change of zero for want of updates, not because it is at a fixed point.
Reproduction: the sparse program with noise sd 1, seed 3, undamped (the original report, seed 6 at noise 0.5, no longer
reproduces on the compiler as it stands; the mechanism is the same): the wild first sweeps (changes up to 6e3) leave the gamma
noise precision with a cavity that is improper for 20 of the 60 row factors, which are skipped for ever (40 failed or blocked
updates per sweep from then on), the other 40 settle around a precision of 2e-5, every inclusion probability stays at the prior
0.2, and the run reported `converged` after 43 sweeps. Damping 0.2 to 0.7 gives the right answer (precision 0.0986, the four
active coefficients at 0.95 to 1.00) in 29 to 105 sweeps. Over 120 setups and seeds (noise, outlier share, rows, features, active coefficients) this was the only collapse, and in every
other converged run (all of the changepoint and sparse seeds measured) the unresolved count at convergence was zero.

What the engine does now (`message-passing/engine.ts`, tests in `verdict.test.ts`):

- Every sweep counts the updates that could not be computed, or that the improper-message policy dropped or reset
  (`MpState.unresolved`, the tape metric `mp/unresolved`). `mpVerdict(g, state, settings)` is the one place that says how a run
  stands and is what `done`, the posterior's `status` and the tape use.
- `converged`: the beliefs stopped moving and every update was recomputed in the last sweep. `stopped` with the reason `stalled`:
  they stopped moving with updates that cannot be computed (the case above). `failed` with `degenerate`: a fixed point with a
  belief that has no density. `diverged`: a message that is not finite. `completed`: the sweep budget ran out. The reason is on the
  tape (`mp/verdict`, a message with `{ status, reason }`, only for a run whose verdict has a reason) and in the posterior's
  `diagnostics.reason`.
- A restart (`restarts`, a knob of every EP and of its variants; default 1 for EP, 0 for variational and belief propagation): a
  run that stalls or diverges is started again from its start messages with damping `max(damping, 0.5)` (then 0.8, 0.9), counted
  on the tape (`mp/restarts`); the sweep cap counts from the last restart. The sparse case recovers by it (converged, 100 sweeps,
  precision 0.0986, one restart). A run that has been restarted and is stuck on the same kind of update is `converged` with the
  count in the reason ("converged after 1 damped restart, but 3 message updates could not be computed even then"): the stall is
  then not an artifact of its first trajectory but the usual skip of an update with an improper cavity. This is what the
  horseshoe regression (three shrunk coefficients whose precision cavity is improper at the fixed point, and whose marginals agree
  with a long NUTS run) does, and it keeps that test, and every page that reads `converged`, working. With `restarts: 0` the same
  run is `stopped`/`stalled`, which is the honest report when nothing has checked it against a damped run.

**Overconfidence when a regime is unvisited (the 3-state Poisson changepoint).** The compiled graph has the operators `prior`,
`table` (the sticky chain), `select` (`y = c_z`, a point-mass constraint) and `poisson`. Measured against the program's Gibbs
reference (mean regime-probability gap, accuracy of the most probable regime, ratio of EP's rate sds to the sampler's):

| seed | plain EP | damping 0.3 | power 0.5 or alpha 0.5 on the compiled graph |
| --- | --- | --- | --- |
| 1 (all regimes visited) | exact (gap 0.001) | same | power: the same numbers (it has nothing to act on); alpha: refused ("no factor supports a power") |
| 2 | gap 0.26, accuracy 0.70, sd ratios 0.31 / 0.15 / 0.12, 155 sweeps | gap 0.32, accuracy 1.00, 0.31 / 1.22 / 0.09 | as plain EP |
| 5 | gap 0.25, accuracy 1.00, 0.96 / 0.12 / 0.09 | does not converge, gap 0.29 | as plain EP |
| 8 | does not converge, gap 0.22, accuracy 0.79, 0.42 / 0.05 / 0.13 | does not converge | as plain EP |
| 3 (visited) | exact | does not converge (gap 0.5), damping breaks a good run | as plain EP |

Power EP and the alpha divergence do nothing on this graph for a structural reason: the only approximating factor is the
`select`, a point-mass constraint `y = c_z`, and a delta has no power (raising the select to a power was tried: it gives thousands of
improper updates and, before the verdict above, runs that reported `converged`). They do apply to the same model written as one
soft factor per step, a gate of Poisson branches (`z_t` chooses which rate's likelihood holds); the gate of closed-form branches now
takes a power (`operators/gate.ts`: the message to `z` is the branch evidences to the power `1/p`, to a variable the matched mixture
divided by `p`; tested against a grid). On that form (hand-built in the experiment; the compiler does not emit it) with the label
permutation of the modes aligned to the Gibbs run:

| seed | plain EP | power 0.5 | alpha 0.5 | alpha 0.3 |
| --- | --- | --- | --- | --- |
| 1 | gap 0.001, sd ratios 0.96 / 1.00 / 1.01 | same | same | same |
| 2 | gap 0.32, accuracy 1.00, 0.31 / 1.22 / 0.09 | 0.32, 0.31 / 1.19 / 0.09 | 0.32, 0.31 / 1.19 / 0.09 | does not converge, 0.38, 0.43 / 0.10 / 0.09 |
| 5 | does not converge, gap 0.36, accuracy 0.46, 0.25 / 0.15 / 0.16 | does not converge, 0.34, 0.42 | does not converge, 0.34, 0.46 | converged, 0.25, accuracy 0.85, 0.25 / 1.02 / 0.09 |
| 8 | does not converge, gap 0.25, accuracy 0.69, 0.38 / 0.05 / 0.13 | does not converge, 0.24, 0.69 | does not converge, 0.22, 0.75 | does not converge, 0.24, 0.66 |

No variant restores calibrated rates or regime probabilities; the rate sds of the barely visited regimes stay at 0.05 to 0.3 of the
sampler's. Alpha 0.3 on seed 5 is the one improvement (a converged run, accuracy 0.85, one sd ratio restored) and it is not
reproduced on seeds 2 and 8. The cause is not the direction of the divergence: the posterior over which steps belong to a rarely
visited regime is multimodal (and so is the label of the regimes: the same model on the gate form settled in a cyclic relabelling of
the regimes, with the same rates) and a Gaussian-or-gamma message per variable cannot hold it; the three rates are shared by all
the steps, a loop that double-counts, which a power only softens. What would help is another kind of method (Gibbs, SMC, or
several EP starts weighted by their evidence), not a slot of this engine. The test of the program holds plain EP to this limit
(`ep-programs.test.ts`).

**The tournament skills (`ep-program-skills`, sds 0.79 to 0.83 of a long NUTS run).** Plain EP, damping 0.5, power 0.8, 0.5, 0.3
and alpha 0.8, 0.5, 0.3 on seeds 1, 2, 3: the mean sd ratio is 0.790, 0.832, 0.816 for every one of them (at most 0.001 apart), the means agree within 0.08 NUTS sd, the precision of the skills within 6%. The probit factor takes a power; it just does not matter,
because the likelihood of a difference of two skills is smooth and near Gaussian. The variants cost: power EP is 25 times slower
(2 s against 0.08 s) and alpha 200 to 900 times (17 to 70 s). The shortfall is not the divergence either.

**Rendering.** The message player's label for a packed Gaussian wrote its whole precision diagonal
(`N-60(precision diag 2, 4, ...)` across the panel). `describeMessage` (`exposition/model-run-tape.ts`) now writes at most the first
four numbers and the count: `N-60(precision diag 2, 2, 2, 2, ... (60))`, and the same for a long discrete or Dirichlet message and an
unknown family.

## 13. Gates whose components have gamma precisions of their own (2026-11-11)

A Gaussian observation whose mean and precision are chosen by the same discrete variable (`where(z, clutter, inlier)` as the
precision idiom `1 / sqrt(precision)`, `at(tau, z)`, with a switch that is a Bernoulli over a beta or a categorical over a
Dirichlet) compiles to one `gate` per observation whose branches are `gaussian-gamma-precision` factors over the means and the
gamma variables of the components (`probprog/model/factors.ts`, `observedSwitch`; the precision cell is a lookup with
`precisions`, read in a "lookup mode" in which a gate keeps its components as states instead of a `select` each). The operator
gained `logEvidence` (what the gate weighs its branches by). Tests: `probprog/model/gate-precisions.test.ts` (the clutter problem
against a four-dimensional grid with the switches summed out; a two-component mixture) and `operators/precision-scale.test.ts`.

The extended gate atoms run on it, each against an exact or sampled reference (default knobs; EP over the reference's sd):

| atom | reference | EP against it | recovery over seeds (coverage of the 90% intervals) |
| --- | --- | --- | --- |
| clutter-learned | quadrature over four unknowns, switches summed out | centre 1.412 against 1.412 (+- 0.082 against 0.083), inlier precision 3.96 +- 0.88 against 3.97 +- 0.88, clutter precision 0.0995 against 0.0996, share 0.163 against 0.164; responsibilities within 0.004 | 40 seeds: centre 1.00, inlier 0.93, clutter 0.90, share 0.88 |
| gate-random-learned | Gibbs sampler | means within 0.12 sd, precisions within 0.05 sd, share 0.406 +- 0.076 against 0.408 +- 0.083 | 20 seeds: 0.75 to 1.00 |
| mixture-components-learned | Gibbs sampler (means relabelled in order) | means within 0.1 sd, spreads 0.88 to 1.01 of the sampler's | 20 seeds: 0.75 to 1.00; closer components: second mean 0.55 to 0.60, EP's spreads 0.5 to 0.7 of the sampler's (one reading of the labels) |
| spike-and-slab-learned | exact (share and slab precision on a grid) | share 0.417 +- 0.148 against 0.421 +- 0.152, slab precision 0.201 +- 0.093 against 0.201 +- 0.104 | 20 seeds: 1.00 and 1.00 |

Planned: the two summed-out forms (`clutter-mixture-learned`, `spike-and-slab-marginal-learned`): `normalOutliers` and
`normalSpikeSlab` take a known scale, share and slab width; the missing piece is a mixture site with a gamma precision and a beta
weight (inside the gate it exists). The AR(1) twin has no basic atom (the tracker is a program page).

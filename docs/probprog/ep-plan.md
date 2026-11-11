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

Still open, in the order of the work packages that follow:

- WP-A (likelihood and function rules): a generic numeric 1-D likelihood operator and registry (Poisson and negative binomial
  with a log link, Weibull, ordinal probit, binomial logistic; the Student-t and Laplace sites folded into it), `apply` for unary
  functions of a random value (`exp`, `square`, `tanh`, `relu`, `softplus`, `abs`: `exp` for log-normal discriminations,
  Poisson regression, the non-centred funnel; `tanh` and `relu` for the Bayesian neural network), numeric priors (half-normal,
  half-Cauchy, log-normal, uniform, truncated: eight schools, radon, the GP length scale), an EP-friendly negative binomial.
- WP-B (structure and scales): the model-layer `gate` node with EP and VMP messages (Gaussian HMM, change points,
  zero-inflation, mixtures with random components), `ordered` as greater-than factors, a constant multiplier on a precision and
  the penalised spline with a random smoothing parameter (both GAM routes of section 3, with the posterior of the smoothing
  parameter checked against the exact reference of the Bayesian GAM page), the horseshoe in the precision idiom (inverse-gamma
  scale mixture), a heavy-tailed likelihood with an unknown scale (a Gaussian with a gamma precision per row), Student-t with
  learned degrees of freedom, proper initial messages in place of the vague-prior workaround, and stochastic volatility with
  `phi` and `sigma` inferred (a product with a random slope).
- The atoms suite and the coverage figure: done on 2026-11-11 (`bayesian-examples/ep-atoms*.ts`, `epCoverage()` with its test,
  the pocs group `expectation-propagation`). 28 bare setups, one factor each, with an exact or grid reference and a tolerance:
  EP runs on 19 (gate between random values, a random index into random components, spike and slab, clutter, an ordered pair,
  `exp(h/2)` as a scale, max, gamma precision observed and latent, Student-t, Laplace, probit, logistic, a probit on a
  difference of skills, a dense GP prior with probit and logistic likelihoods, and the three conjugate pairs) and is planned on
  9 with the library's reason on each page: the gate with known components (the compiler wants a variable per component), exp,
  tanh and Poisson with a log link (`apply`), half-Cauchy and half-normal (numeric priors), a scaled precision (the GAM),
  ordered means (the `ordered` constraint) and the product of two Gaussians (the compiler accepts it and the first update
  sends an improper message: the run breaks, WP-B proper initial messages). EP runs on 10 of the 27 benchmark models. When a
  rule lands the registry test of the atom it unblocks goes red until the atom is flipped to `runs`, and the coverage test is
  moved up.

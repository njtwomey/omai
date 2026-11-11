# Expectation propagation: the plan

The aim, from the user: EP in all or at least most of the examples and models, with spike and slab, gates and the spline (GAM) case
done properly, and a view of the other distributions we are likely to meet. EP is a showcase capability and worth real time; a refusal
is usually a missing rule or the wrong way of writing the model, not a limit of EP.

Sections 2 to 4 were written before the work and say what was needed and why; nearly all of it is built (the status page says what is
not).

This page is the stable part of the plan: the catalogue of rules by family, the GAM design and the use cases that motivate them (the
section numbers are kept, other documents refer to them). It carries no status: how the library is put together is
[`ep.md`](ep.md); what runs today, the gaps and the counts are [`ep-status.md`](ep-status.md) (generated from the code); what was
built when is [`ep-changelog.md`](ep-changelog.md) (the first audit and the dated sections of every work package). Companion notes:
the GAM case [`ep-and-gam.md`](ep-and-gam.md), the prior atlas [`ep-priors.md`](ep-priors.md), and `notes/` (covariance matrices,
the dense non-centred scale, EP quality).

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
1. _A mixture-of-Gaussians prior factor_ on a scalar (closed form: a Gaussian cavity times a K-component Gaussian mixture is a
   K-component Gaussian mixture, so the moments are exact; no quadrature). This is spike and slab, scale mixtures, a
   Student-t as a finite mixture, outlier models, and the `normalSpikeSlab` family of the model layer maps to it directly.
2. _The gate proper_: the model layer's `gate` node (listed as not built) and its EP/VMP messages, so a branch with its own
   random variables works (the Gaussian HMM, the change-point models, zero-inflation, mixtures of regressions). The
   message to `z` is the ratio of the branches' evidence under their cavities; the branch variables get messages weighted by
   `z`'s cavity.
   Plus `ordered` as a chain of `greater-than` factors (the factor exists, the compiler does not map the constraint).

**D. Priors with no family: half-normal, half-Cauchy, inverse-gamma, log-normal, uniform, truncated normal**: a numeric _prior_
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
1. _Scale-mixture factor on the vector_ (the principled one). The factor `f(beta, lambda) = N(beta; 0, (lambda K + P0)^-1)`, with
   `K` the penalty and `P0` a fixed precision for the unpenalised directions. With a Gaussian cavity `N(m, V)` on `beta` and a
   Gamma cavity on `lambda`, integrating `beta` out gives `Gamma(lambda) N(m; 0, V + (lambda K + P0)^-1)`: a function of the
   scalar `lambda` only, so the moments of `lambda` come from a grid in `log lambda` (the same device as the rule added today),
   and `beta`'s moments are the mixture over that grid of the Gaussian conditionals (dense `p x p`, `p` is 10 to 40). The
   Gamma site is a projection, the Gaussian site is dense. Cost `O(p^3)` per grid node, so about a second at worst, `O(p)`
   in the penalty's eigenbasis when the cavity is diagonal. The approximation is: `beta | y` Gaussian, `lambda | y` Gamma, which
   is accurate when the posterior of `log lambda` is close to a log-gamma, as it usually is. The compiler change: allow a random
   scale on a packed vector with a known constant matrix.
2. _Per-coefficient route_ (cheaper, covers ARD and basis selection too). Work in the penalty's eigenbasis, where the prior is
   `gamma_j ~ N(0, 1 / (lambda d_j))`: the existing rule with a known constant multiplier `d_j` on the precision (a one-line
   generalisation of `precisionTilt`), with the coefficients as scalar variables (an "unpacked" compilation of the matmul: one
   sum-of-products row per observation) instead of one packed vector. ARD is the same with a precision per coefficient; basis
   selection (spike and slab on the weights) is C1. The coupling of `lambda` across the `p` Gaussians is carried by EP's
   iteration; `p` factors on one Gamma variable is the same structure as the hierarchy that already works.

Recommendation: route 2 first (it reuses the rule just added, unlocks ARD and spike and slab on bases, and needs a compiler
option rather than a new rule), route 1 when a dense, correlated posterior of the coefficients matters. The unknown noise level
is already handled. The rest of the page's model (log-normal hyperpriors) is D and B.


## 4. Other distributions and use cases to plan for

| use case                                                    | what it needs                                                                   | in section 2               |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------- |
| robust regression, outlier models                           | Student-t, Cauchy, Laplace, a mixture of inliers and outliers                   | A, C1                      |
| count data: Poisson, negative binomial, zero-inflated       | exp link, negative binomial, a gate                                             | A, B, C                    |
| survival, reliability, censoring                            | Weibull, log-normal, exponential with a log link; truncation                    | A, B                       |
| ordinal and ranking (ratings, Bradley-Terry, Plackett-Luce) | ordinal probit, logistic on differences (IRT is this)                           | A, existing                |
| quantile regression                                         | asymmetric Laplace                                                              | A                          |
| variable selection, sparse regression                       | spike and slab, ARD, horseshoe (a scale mixture: half-Cauchy via inverse gamma) | C1, D                      |
| hierarchical models with a learned spread                   | precision idiom, half-normal and half-Cauchy sds                                | D, done for the gamma form |
| correlated random effects, varying slopes                   | LKJ or Wishart                                                                  | G                          |
| Gaussian processes: classification, count and survival GPs  | dense MVN prior, kernel Gram data, A for the likelihood                         | E, A                       |
| Bayesian neural networks                                    | `tanh`, `relu`, `sigmoid` on random values                                      | B                          |
| state space, stochastic volatility, change points           | `exp` variance, scan chains, gates                                              | B, C2                      |
| mixtures, HMMs, topic models                                | gates, `table` / BP, Dirichlet sizes                                            | C2, G                      |
| factor models, matrix factorisation, recommenders           | products of Gaussians (exist), ARD precisions                                   | existing, F                |


## 5. Verification standard for every rule

(a) The tilted moments against a closed form or a high-order quadrature; (b) a small model against the exact grid posterior (an
atom: a setup, a reference, a tolerance with its reason); (c) a TSDoc `@example`; (d) the coverage measurement (`epCoverage`) says
every benchmark model still runs, and the committed digest (`make ep-digest-check`) says nothing else moved; (e) the matching page
offers EP without changing the page. How to add each kind of piece is in [`ep.md`](ep.md), section 9.

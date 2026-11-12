# EP changelog

Moved here verbatim from `ep-plan.md` on 2026-11-11 so that nothing is lost: sections 7 to 18 of the old plan, a dated log written
by the agents that built the capability (the numbering is the old one: two sections 16, no 17, 12 before 10), followed by the
sections of the first plan that described the world before any of it existed (the audit, the work packages, what was in
flight). Every number in it was true when it was written and none is kept up to date: the current figures are in
[`ep-status.md`](ep-status.md) (generated from the code), the design in [`ep.md`](ep.md), the stable plan in [`ep-plan.md`](ep-plan.md).
The log is frozen: a new piece of work adds its section at the end, dated, or goes in `notes/`.

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
  `gaussian-gamma-precision` and `widened` in the gate, were replaced by one engine policy, section 18.)
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
- Priors without a message family (first projected once with `projectedPrior`, `priors.ts`, since section 18 the exact numeric
  `prior-site`): half-normal, half-Cauchy, log-normal, uniform and truncated normal, a positive scale on its log; a latent with a random scale is written non-centred, an observed value with a
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

| atom                           | parameter: coverage (error in sds, rms)                                      |
| ------------------------------ | ---------------------------------------------------------------------------- |
| gaussian-learned               | mean 0.82 (1.10), variance 0.95 (0.79)                                       |
| gamma-precision-learned        | mean 0.82 (1.09), precision 0.97 (0.70)                                      |
| shared-precision-learned       | precision 0.85 (1.01), mean 0.85 (1.21), the 8 values 0.86 (1.18)            |
| beta-bernoulli-learned         | strength 1.00 (0.81), the 8 probabilities 0.90 (1.02)                        |
| gamma-poisson-learned          | prior rate 0.90 (0.89), the 8 rates 0.89 (1.04)                              |
| student-t-learned              | location 0.88 (1.01), precision 0.97 (0.87)                                  |
| probit-learned                 | offset 0.97, slopes 0.93 and 0.88 (0.9 to 1.2)                               |
| logistic-learned               | offset 0.90, slopes 0.82 and 0.78 (shrinkage of fixed weights, 1.3)          |
| poisson-log-learned            | intercept 0.88, slope 0.85 (1.05, 1.11)                                      |
| poisson-log-dispersion-learned | intercept 0.92, slope 0.96, dispersion 0.96                                  |
| weibull-learned                | intercept 0.97, slope 0.88                                                   |
| skill-difference-learned       | spread precision 1.00 (0.53, an informative prior), the 8 skills 0.86 (1.14) |

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

The open items of this plan (what EP cannot run, where it is less good than it should be, the infrastructure tasks)
are kept as data in `packages/engine/src/lib/probprog/ep/ep-gaps.ts` and shown at the end of the EP coverage page, each with
why, how it would be closed, its size and its status. A test (`ep-gaps.test.ts`) holds the list to the coverage table: every setup
and benchmark model EP refuses is named by a gap, and a gap that names something EP now runs is stale. Where the code lives: the
rules (`message-passing/operators`), the compiler mappings (`probprog/model/factors.ts`), the model families
(`probprog/model/dists`) and the methods (`probprog/inference`) are in the library proper, and so is the suite that runs them:
the setups (the atoms), coverage, recovery, calibration and the gaps themselves are in `probprog/ep` (`omai-engine/probprog/ep`;
2026-11-11, `agent-pp161-move`, which closed the gap `integration`). The example methods of the Bayesian pages (`defineExample`, the per-example knobs and budgets, the views) are not library: they are
pocs code, in `apps/pocs/src/examples/` (moved there 2026-11-11: a demonstration composes library estimators, it is not one). The one
helper the atoms took from the example scaffold (`exampleNumber`) is `configNumber` in `probprog/inference/method.ts`.
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

| seed                    | plain EP                                                          | damping 0.3                                            | power 0.5 or alpha 0.5 on the compiled graph                                                      |
| ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 1 (all regimes visited) | exact (gap 0.001)                                                 | same                                                   | power: the same numbers (it has nothing to act on); alpha: refused ("no factor supports a power") |
| 2                       | gap 0.26, accuracy 0.70, sd ratios 0.31 / 0.15 / 0.12, 155 sweeps | gap 0.32, accuracy 1.00, 0.31 / 1.22 / 0.09            | as plain EP                                                                                       |
| 5                       | gap 0.25, accuracy 1.00, 0.96 / 0.12 / 0.09                       | does not converge, gap 0.29                            | as plain EP                                                                                       |
| 8                       | does not converge, gap 0.22, accuracy 0.79, 0.42 / 0.05 / 0.13    | does not converge                                      | as plain EP                                                                                       |
| 3 (visited)             | exact                                                             | does not converge (gap 0.5), damping breaks a good run | as plain EP                                                                                       |

Power EP and the alpha divergence do nothing on this graph for a structural reason: the only approximating factor is the
`select`, a point-mass constraint `y = c_z`, and a delta has no power (raising the select to a power was tried: it gives thousands of
improper updates and, before the verdict above, runs that reported `converged`). They do apply to the same model written as one
soft factor per step, a gate of Poisson branches (`z_t` chooses which rate's likelihood holds); the gate of closed-form branches now
takes a power (`operators/gate.ts`: the message to `z` is the branch evidences to the power `1/p`, to a variable the matched mixture
divided by `p`; tested against a grid). On that form (hand-built in the experiment; the compiler does not emit it) with the label
permutation of the modes aligned to the Gibbs run:

| seed | plain EP                                                       | power 0.5                     | alpha 0.5                     | alpha 0.3                                          |
| ---- | -------------------------------------------------------------- | ----------------------------- | ----------------------------- | -------------------------------------------------- |
| 1    | gap 0.001, sd ratios 0.96 / 1.00 / 1.01                        | same                          | same                          | same                                               |
| 2    | gap 0.32, accuracy 1.00, 0.31 / 1.22 / 0.09                    | 0.32, 0.31 / 1.19 / 0.09      | 0.32, 0.31 / 1.19 / 0.09      | does not converge, 0.38, 0.43 / 0.10 / 0.09        |
| 5    | does not converge, gap 0.36, accuracy 0.46, 0.25 / 0.15 / 0.16 | does not converge, 0.34, 0.42 | does not converge, 0.34, 0.46 | converged, 0.25, accuracy 0.85, 0.25 / 1.02 / 0.09 |
| 8    | does not converge, gap 0.25, accuracy 0.69, 0.38 / 0.05 / 0.13 | does not converge, 0.24, 0.69 | does not converge, 0.22, 0.75 | does not converge, 0.24, 0.66                      |

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

| atom                       | reference                                          | EP against it                                                                                                                                                                                              | recovery over seeds (coverage of the 90% intervals)                                                                                       |
| -------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| clutter-learned            | quadrature over four unknowns, switches summed out | centre 1.412 against 1.412 (+- 0.082 against 0.083), inlier precision 3.96 +- 0.88 against 3.97 +- 0.88, clutter precision 0.0995 against 0.0996, share 0.163 against 0.164; responsibilities within 0.004 | 40 seeds: centre 1.00, inlier 0.93, clutter 0.90, share 0.88                                                                              |
| gate-random-learned        | Gibbs sampler                                      | means within 0.12 sd, precisions within 0.05 sd, share 0.406 +- 0.076 against 0.408 +- 0.083                                                                                                               | 20 seeds: 0.75 to 1.00                                                                                                                    |
| mixture-components-learned | Gibbs sampler (means relabelled in order)          | means within 0.1 sd, spreads 0.88 to 1.01 of the sampler's                                                                                                                                                 | 20 seeds: 0.75 to 1.00; closer components: second mean 0.55 to 0.60, EP's spreads 0.5 to 0.7 of the sampler's (one reading of the labels) |
| spike-and-slab-learned     | exact (share and slab precision on a grid)         | share 0.417 +- 0.148 against 0.421 +- 0.152, slab precision 0.201 +- 0.093 against 0.201 +- 0.104                                                                                                          | 20 seeds: 1.00 and 1.00                                                                                                                   |

The two summed-out forms (`clutter-mixture-learned`, `spike-and-slab-marginal-learned`) run since section 18. The AR(1) twin has no
basic atom (the tracker is a program page).

## 14. The changepoint program with the rates and the transition matrix learned (2026-11-11, `agent-pp111-cpext`)

Page `ep-program-changepoint-extended`: three regimes, Poisson counts at unknown rates (the gamma priors of the basic page) and an
unknown transition matrix, a Dirichlet per row leaning towards staying (`[4,1,1]` and its rotations, mean stay 0.67). The model
writes the step as `categorical(probs: at(A, prev))` with `A` a Dirichlet latent over the regimes.

**The mapping.** The compiler took only a known matrix there (a `table`). With Dirichlet rows it now builds, per step, one `gate`
(`probprog/model/factors.ts`, `latentCategorical`, `randomTransitionRow`): the state is the previous regime, the variables are the new
regime and the K row variables, and branch `j` is the conjugate `discrete` factor of the new regime on row `j`. The gate's message to
the new regime and to the previous one is then the exact belief-propagation message of the transition table under the cavities of
the rows, and a row hears only the transitions out of its own regime, weighted by the belief that the chain was there (a Dirichlet
moment match of the mixture of conjugate updates). Without this the compiled form was a `select` of Dirichlet variables, which did not
even compile a graph (`dirichlet needs a size`). Test: `probprog/model/transition-rows.test.ts`, a three-state chain of nine Gaussian
readings against the exact posterior by enumeration of the 3^9 paths (rows integrate out as Dirichlet-multinomials): row means within
0.02 and EP's spreads 0.9 to 1.15 of the exact ones, state marginals within 0.05. A matrix with fewer rows than states is refused with its
reason. Not mapped: a sticky model whose row is a function of one Beta stay probability (the diagonal entries of the Dirichlet rows
give the same quantity), a Poisson or other emission whose rate is a known lookup by the state (that needs a table of count
probabilities for a Poisson; the Gaussian lookup has it), and transition rows that are not Dirichlet latents.

**Cost.** Each row variable is shared by every step, so its degree is the number of steps and the engine's belief (a sum of
all its messages) is recomputed for every port of every gate: 100 steps run in about 2 s. Two cheap changes made that tolerable
(`message-passing/families.ts` `combine` as a plain loop, `_scalar.ts` calling the scalar special-function kernels instead of the
tensor ones; same numbers, the Newton loop of the Dirichlet projection is the hot spot). Incremental beliefs per variable would be the
next step; the kernel `polygamma(n, NaN)` recursed for ever and now returns NaN.

**What was measured** (100 steps, stay probability 0.93, truth rates 3, 9, 22, EP damped by half, Gibbs from a data-ordered start;
`ep-program-changepoint-extended/calibration.ts`). Seeds 1 to 24: where every regime has at least 6% of the steps (18 seeds), EP's
regime probabilities are within 0.05 of Gibbs on every one, and the truth is inside EP's 90% interval for the rates in 89%, 89%, 94%
of seeds (error 0.98, 1.21, 0.85 sds, Gibbs 89%, 94%, 94%), for the three stay probabilities 78%, 89%, 89% (Gibbs 72%, 83%, 89%: the
prior pulls them towards 0.67, so even the exact posterior covers a little less than 90%) and for the six moves 83% to 100%; the
error is at most 1.2 sds everywhere. Where a regime is barely visited (6 seeds with fewer than 6 steps in a regime), EP agrees with Gibbs on 1,
two runs do not converge, the coverage of the stay probabilities falls to 50% to 75% and the rate of the thin regime has an error of 2.6 sds.

**Label switching is not the same failure, and damping fixes it.** The three regimes have alike likelihoods up to the rates, so the
posterior has permuted modes, of which only the broad priors (means 4, 12, 30) prefer the labelling the data were made with. Undamped
EP took a swapped labelling on 3 of 30 seeds with all regimes visited (low and middle, or middle and high, exchanged; regime accuracy
0%) and on two more with a regime of under 15 steps; damping by half (about twice the sweeps) removed every one of them, and the
reference sampler had the same problem when started from the prior means (seed 1: it sat in the swapped mode while EP was right),
so it starts from the low, middle and high quantiles of the counts. Flooding also fixes the swaps on the three seeds but swaps seed 1.
An ordered log-rate form (`constraint: ordered`, `exp` into a Poisson) was tried and does not work: 180 sweeps (9 s) on an easy seed,
absurd rates on the swapped ones. A hyperprior tying the rates would make the regimes exchangeable and the swap worse. The barely
visited regimes are untouched by all of this (section 11); they are the case multi-start EP by evidence (`agent-pp101-evidence`)
is for.

## 15. WP-E: the log evidence of every rule, kernel hyperparameters on a grid, EP from several starts (2026-11-11, `agent-pp101-evidence`)

**The log evidence.** Every operator of the registry that has an EP rule now has a `logEvidence` (the audit
`operators/evidence.test.ts` fails on one that has none): by the time this package started all but the gamma-precision rule
(`gaussian-gamma-precision`, whose tilt grid in `log lambda` already computed the integral) and `matmul-constant` had one. Each is now
tested against a brute-force integral of the factor against the unnormalised cavities (`exp(eta . T)`, the engine's convention: the
masses of the Gaussian cavities and the integral of the rest): the gamma-precision rule (a latent value and a known mean, both latent,
both known, with a scale and an offset), `gamma-rate`, `gamma-product`, `gaussian-mv-precision-scale`, the mixture, Laplace and Student-t
sites, the Student-t row, the numeric negative binomial with a random dispersion, the gate and `matmul-constant`; the others
(`apply`, the likelihood sites, the dispersion, the concentration, the Gaussian and conjugate ones) were tested before. The brute
forces agree to 1e-5 to 1e-7 (the grid is the limit). What the model-level audit found was not a missing rule but a missing case: a cavity
that is not proper alone. A defined output that nothing else reads (a flat cavity), or a likelihood's site with a negative precision
as the mean of a Gaussian, made `gaussian`, `dot-constant` and `apply` return NaN and so the whole model's evidence NaN (the `exp`,
`product`, `max`, half-Cauchy, half-normal, skill-difference, log-link Poisson regressions). They now integrate the pair (or the vector
and the output) in natural parameters, which needs only the sum to be proper (`logGaussianIntegral`), and a flat output integrates to one.

**EP's model evidence against the grid evidence of the same model** (`probprog/ep/ep-evidence.test.ts`, 39 atoms; the posterior
and the tape `mp/evidence` carry it, nothing new was needed there): conjugate setups (Gaussian with a normal-gamma pair, beta-Bernoulli,
gamma-Poisson, gamma-exponential, beta-geometric, Dirichlet-categorical, normal-inverse-gamma) within 2e-6; a log-link Poisson, a gate
with a beta weight, the spike and slab, the clutter, the ordered means, a gate of random components, a beta with a random
concentration within 1e-3 (the gate and the clutter are exact: 0.0000); `exp`, `tanh`, log scale, Student-t within 1e-3, the product (0.006)
and the max (0.001) of Gaussians, a Gaussian with a gamma precision 0.024 and 0.038 (the gamma projected from a Student-t shaped tilt),
Laplace 0.04, probit 0.016, logistic 0.019, the skill difference 0.001, a log-normal rate through `exp` 0.003 and 0.004, a Weibull
0.009, a negative binomial with a learned dispersion 0.2. Three limits at the time (the first two are
closed or explained in section 18): the projected priors (half-normal 0.42, half-Cauchy 0.19, uniform 0.09 nats: the graph carried the
projection, not the prior); a fixed point with a factor whose cavity is improper (the learned logistic regression: NaN); an alpha
divergence has no EP energy.

**Kernel hyperparameters (`epGridMethod`, kind `ep-grid`).** A Gaussian process with a random length scale has a covariance that is a
function of a random value, which the graph cannot see through. The method sets the scalar latents a `custom` function reads to each
value of a grid (evenly in the unconstrained coordinate, so a positive one is log-spaced, between the tails of the prior; `doSpec`), the
compiler reads the covariance as a constant matrix, EP runs, and the EP log evidence plus the log prior weights the node; the
posterior is the evidence-weighted mixture (`_site-mixture.ts`: closed-form moments, joint draws, the hyperparameter as draws), a step
is one node in a coarse-to-fine order, and the grid extends where an edge still has mass (budget `maxNodes`). Compiler changes (all in
`probprog/model`): a `custom` function of known values is evaluated, one that is affine in one random vector is read as a matrix by
probing it on the unit vectors (and checked at two other points), and an event vector (a multivariate normal with no named dimension,
observed through a diagonal known covariance) is read as a vector over a dimension of its size (`event-dims.ts`); a node whose custom
calls were all read is no longer reported opaque, so a model that compiled passes `epSpecMethod`'s static check too.

Results on the two benchmarks (`gaussian-process-regression`, event form, and `gaussian-process-non-centred`, `f = L(ell) z`): with a
Gaussian likelihood EP is exact at every node, so the node's evidence equals the textbook marginal likelihood to 1e-8 and the function's
mean and sd to 1e-7 (`probprog/inference/ep-grid.test.ts`). Against the benchmark's own truth (quadrature over the length scale, the
function values the mixture of their exact conditionals) at 31 points on the unconstrained scale (64 ms for the centred form, 400 ms for
the non-centred, which has a 12-dimensional whitened vector): the evidence differs by 4e-7, the length scale 0.9758 +- 0.2571 against
0.9758 +- 0.2571, the function means within 1e-4 and the sds within 1e-4 of the truth's; 9 points are within 0.012 nats and 0.012 sds,
17 points 3e-5. The two benchmark rows flip to `runs` (`epCoverage`: benchmarks 25 of 27, atoms 68 of 72; 23 and 46 at the start of
the package). New atoms: `gp-length-scale` (six points, exact reference by quadrature, `ep-atoms-kernels.ts`) and its extended twin
`gp-probit-learned` (a probit classifier on eight inputs, labels drawn from a function at a stated length scale; over 40 seeds the 90%
interval holds the length scale every time because eight binary labels barely move it from the prior, and the function values in 83 to
98% of the runs, errors 0.9 to 1.15 sds; 3 s for the 40 seeds). Not done: more than a few hyperparameters (the grid is a product; gap
`grid-hyperparameters`), a kernel node in the model layer that would make the custom function unnecessary, a numeric site over the length
scale inside one EP run.

**EP from several starts (`epMultiStartMethod`, kind `ep-multistart`; the gap `multimodal`).** The discrete and Bernoulli messages of the
graph are started from random log-probabilities (sd 2), and, with `exclusions`, from messages that rule out one category of every
discrete variable (a regime nobody visits); the first start is plain EP. Runs that reach the same fixed point (every site mean within 5% of
the larger sd) are one mode, only runs that converged count when any did, the weights are the modes' EP evidence, the posterior is the
mixture of the modes' beliefs (`_site-mixture.ts`), and the model's evidence the log sum of the modes'. A step is one start, so a snapshot
is the mixture so far. Test (`ep-multistart.test.ts`): two regimes with Poisson counts of which only one has data, the posterior by
enumeration of the 2^14 assignments (each assignment's evidence and rates are gamma closed forms): for the empty regime the exact posterior is
17.7 +- 16.9, one run says 23.0 +- 17.1 (a third of an sd off) and the mixture of 30 starts 17.5 +- 16.9; the visited regime is 3.50 +- 1.01 exactly, 3.45 +- 0.49
for one run (the exact spread counts the steps that may belong to the other regime) and 3.45 +- 0.57 for the mixture.

On the changepoint program (`ep-program-changepoint`, 80 steps, stay 0.95, rates 3, 9, 22, 24 starts, Gibbs reference of the page; spreads as
ratios of the sampler's, errors in sampler sds; one run -> the mixture):

| seed            | the rarely visited regime                                                                                             | the other rates                            | modes |
| --------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----- |
| 2               | rate[1] 19.5 +- 1.06 -> 18.8 +- 5.8 (Gibbs 17.5 +- 7.0): sd ratio 0.15 -> 0.83, error 0.28 -> 0.18                    | rate[0] 0.31 -> 0.31, rate[2] 0.12 -> 1.27 | 4     |
| 5               | rate[1] 8.4 +- 0.85 -> 12.3 +- 6.8 (Gibbs 13.2 +- 7.2): sd ratio 0.12 -> 0.94, error -0.66 -> -0.12                   | rate[0] 0.96, rate[2] 0.09 -> 0.75         | 3     |
| 8               | rate[1] 2.9 +- 0.45 -> 8.8 +- 6.0 (Gibbs 10.1 +- 9.2): sd ratio 0.05 -> 0.65, error -0.78 -> -0.15                    | rate[0] 0.21, rate[2] 0.13 (unchanged)     | 1     |
| 1 (all visited) | one run is exact (sd ratios 1.00, 0.97, 1.01) -> the mixture is 4 to 12 times wider: it contains the relabelled modes |                                            | 6     |

So calibration is restored for the regime nobody visits (the spread to 0.65 to 0.94 of the sampler's, the mean within 0.2 sds) and EP's
mixture needs 2 to 6 EP runs' worth of time per start (23 to 47 s for 24 starts of a chain of 80 steps; 3 s on the easy seed). It is not
a fix for everything. (1) The relabellings: the three regimes differ only by their rates and the priors (means 4, 12, 30) prefer one
labelling by 1.4 to 8 nats, so the permuted fixed points carry weight and the mixture is the posterior of the model as written, whose
marginal rate of "regime 1" is then not one number; the data-ordered Gibbs reference stays in one labelling, so against it a case where
one run was exact is worse (seed 1: the truth is inside every interval of the mixture, 3.07 +- 1.92, 10.3 +- 6.0, 19.3 +- 5.8). (2) The
visited rates under label switching on seeds 2 and 8 (the sampler's spread of 6.9 and 9.1 on rate[2] comes from moving between labellings)
are not recovered on seed 8, where the 24 starts reach one converged mode only, and the runs that stop at the sweep cap (collapsed rates
2.5/3.1/13.6 on another seed) are left out. (3) The evidence of a fixed point is a good weight where it is the mass of a mode, and
a poor one for a mode that is an artefact of loops (it is within 1 to 2 nats of the true mode on seed 2). Starts from the data (regimes
ordered by the counts) and a start for each relabelling are the next steps (gap `multimodal`).

## 16. WP-M: gaps closed and coverage raised (2026-11-11, `agent-pp132-maps`)

Four compiler mappings and one engine rule, each with a brute-force or exact reference, an atom or program page, and the coverage floors up
(benchmarks 25 -> 27 of 27, atoms 68 of 72 -> 72 of 76 with the four new ones; the four planned atoms are the rule gaps of other work packages).

| Mapping                                                                                                                                                                                | Compiles to                                                                                                                                                                                                                                                                                                                                                                                                                                           | Reference                                                                                                                                     | EP against it                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LDA-lite (`lda-lite`): a plate of Dirichlet mixtures over documents and of Dirichlet word distributions over topics, `categorical({ probs: phi.at(z) })` read through a discrete index | one conjugate `discrete` factor per token on its document's mixture and one `gate` per token on its topic whose branch `k` is the conjugate `discrete` factor of the observed word on topic `k`'s Dirichlet (`observedRowGate`, the same gate as an unknown transition row with an observed outcome)                                                                                                                                                  | the enumeration of the 2^15 topic assignments with the Dirichlets integrated out (the benchmark's own truth)                                  | full EP in 2 s (35 variables, 35 factors): the 26 marginals within 0.26 truth sds at worst, spreads 0.58 to 1.16 of the truth's (the tokens that share a word share a topic); stochastic EP was not needed. Page `ep-topic-model` (two documents, exact beta mixtures over the 256 assignments): within 0.06 sds    |
| Stick-breaking mixture (`stick-breaking-mixture`)                                                                                                                                      | `ops.stickBreaking(v)` is a first-class model operator (differentiable, so the other engines read it as before: the benchmark's `custom` function is gone), a `sticks` cell in the compiler and one `stick-breaking` factor per label (a new engine operator: the message to the label is the mean weights, to a break the beta matched on the log moments to cavity times `r_j v + (later states)(1 - v) + (earlier states)`, VMP the expected logs) | the 3-D quadrature of the benchmark                                                                                                           | operator against a 3-D integral to 1e-3 (messages and evidence); the three breaks within 0.017 truth sds, spreads 0.89 to 0.97, log evidence 0.13 nats off. The weights `pi` are not a site of the graph (they are read back from the breaks). Page `ep-stick-breaking` (grid over the two breaks): within 0.02 sds |
| Mixtures and switching regressions (gap `gate-node`, direction `mixture-regression`)                                                                                                   | `ops.pick(matmul(X, B), z)`: the prediction of the class a row's discrete latent picks, one `gate` of Gaussian likelihoods per row over the classes' predictions (a known index selects, a known function of a discrete variable is the lookup of the branches)                                                                                                                                                                                       | `mixtureOfRegressions` (new, `validation/enumeration.ts`): the exact posterior by enumerating the assignments, validated against closed forms | coefficient means within 0.14 sds, spreads 0.72 to 1.05 of the exact, class probabilities within 0.12. Page `ep-mixture-regression`                                                                                                                                                                                 |
| Hurdle counts (direction `count-models`)                                                                                                                                               | families `zeroTruncatedPoisson` and `hurdlePoisson` (a zero with probability `zero`, else zero-truncated Poisson); the state of the hurdle is determined by the count, so a zero is a conjugate `bernoulli` observation of the Beta `zero` and a positive count the same observed false plus a `poisson-log-truncated` site (a new row of the likelihood registry) on the log rate                                                                    | a Beta for `zero` (closed form) and a grid over the coefficients                                                                              | `zero` exact to 1e-6, coefficients within 0.004 sds, spreads within 3%. Page `ep-hurdle-poisson`                                                                                                                                                                                                                    |
| Vector-state linear dynamical systems (direction `kalman`)                                                                                                                             | `ops.matvec(F, prev)` and `ops.dot(h, state)`; a scan of multivariate normal states is one Gaussian vector per step, a `gaussian-transition` (new operator: `x ~ N(F w + c, Q)`, messages in information form so a cavity that sees only the position still passes information to the velocity; evidence in the joint's information form) between neighbours and a `linear-regression-row` per reading                                                | the engine's Kalman filter and RTS smoother (`dynamics/filtering`) and the filter's log likelihood                                            | means and sds of every step to 1e-6, the log evidence to 1e-8 (EP is exact on the chain); calibrated over 12 seeds (50 steps): the truth inside the 90% intervals at 87% (position) and 90% (velocity, which is never read), errors 1.07 and 0.98 posterior sds. Page `ep-program-kalman`                           |

Also closed: gap `gate-random-scale` (a mean and a log spread chosen by one discrete variable, `exp(at(logSd, z))`): the observation is one
`gate` whose branch `k` is the `gaussian-scale` factor on the mean and the log scale of state `k`, instead of two selects that lost which
component both came from. Against a 4-D grid the means are within 0.15 sds; the spreads of the means come out at 0.57 to 0.71 of the grid's
(one Gaussian per variable on a posterior that is a mixture over the labels, the `multimodal` limit), those of the log spreads 0.97 to 1.1.

What remains: the explicit `gate` family with different likelihood families per branch (gap `gate-node`: nested distributions are 13 walkers of
the model layer), the weights `pi` as a site, a random concentration on the stick-breaking breaks and components with unknown means in the
Dirichlet-process mixture, a random transition matrix or non-square readings in the vector-state scan, and the four atoms planned on other
work packages (`laplace-learned`, `prior-beta-two-shapes`, `clutter-mixture-learned`, `spike-and-slab-marginal-learned`).

## 16. WP-G: the two-variable numeric site, the multiclass site, the ordinal and quantile distributions (2026-11-11, `agent-pp131-rules`)

**The two-variable numeric site (`message-passing/operators/pairs.ts`, `_grid.ts`).** One operator, `likelihood-pair`, reads a row of a
registry of densities `p(y | x0, x1)` over two random scalars: `laplace-scale`, `asymmetric-laplace-scale`, `student-t-df` (known scale),
`weibull-shape` and `beta-shapes` (a beta outcome with two random shapes). A coordinate is a real value or a positive one that is a
Gaussian on its log (log-normal, half-normal, half-Cauchy and uniform priors, which the compiler already carries on the log) or a gamma
variable (integrated in its log, the density in that coordinate carries the Jacobian), and the message to a gamma port is matched on
`E log x` and `E x`. The integral is a panelled Gauss-Legendre grid in both coordinates (`_grid.ts`: coarse passes that find the window from
the marginal weights, a fine pass over it; panels split at the kinks of the density and graded towards them, which a Laplace needs; a
density may give a column form so what depends on the second coordinate alone is computed once per node). About 4,000 evaluations per
message once the window is found. Tested against a brute-force midpoint integral of both cavities (`operators/pairs.test.ts`): moments of
both ports and the log evidence within 2e-4 for the narrow and gamma cases, 5e-4 for a location cavity much wider than the likelihood. A
beta variable with two random shapes is `beta-two-shapes` (`operators/concentration.ts`), the two-dimensional version of
`dirichlet-concentration` on the same grid: the conjugate integral `B(a + e1, b + e2) / B(a, b)` against the two shape cavities,
messages to `p` matched on `E log p` and `E log (1 - p)`; tested against a triple brute-force integral.

_Compiler (`factors.ts`, `observedPair`, `positivePort`)._ `laplace(mu, scale)`, `asymmetricLaplace(mu, scale, tau)` and `studentT(df, mu, s)`
with a random scale or df, `weibull(shape, exp(affine))` with a random shape and `beta(a, b)` observed map to `likelihood-pair`; a random
positive parameter must be a gamma or exponential latent, a positive latent carried on its log, or `exp` of an affine value, else the
refusal says so. A latent `beta(a, b)` with two different random shapes (each exactly a gamma or log-normal latent) maps to `beta-two-shapes`.
Model tests against exact grids (`probprog/model/pair-factors.test.ts`): Laplace scale, Student-t df and Weibull shape within 0.2 reference
sds and spreads 0.8 to 1.2; the latent beta with two shapes: `p` within 0.01 of the exact mean. Beta outcomes with two shapes keep
the means but their spreads are 0.65 to 0.8 of the exact ones: the two shapes are strongly correlated and one Gaussian each cannot hold that
(gap `loopy-overconfidence`).

_Atoms._ `laplace-learned` (mean z 0.005 and 0.05, sd ratios 1.02 and 0.93; 40 seeds: coverage 0.93 and 0.97), `prior-beta-two-shapes` (p
0.66717 against 0.66710, sd ratio 1.001; the shapes within 0.004 sd), and the new `student-t-df-learned` (gamma df prior; sd ratios 1.0 and
0.94; the prior and the truth chosen so the exact posterior covers 0.96, EP the same) and `weibull-shape-learned` (sd ratios 0.95, 0.975;
coverage 1.00 and 0.95 over 20 seeds). Closed gaps: `laplace-random-scale`, `two-random-shapes`, `learned-df`.

**The multiclass site (`operators/multiclass.ts`).** `multiclass-site`: a categorical outcome whose probabilities are a softmax (or the
multinomial probit) of K Gaussian class scores, one port per class. The tilted mean and variance of every free score is a K-dimensional
integral against the product of the cavities, found by adaptive Gauss-Hermite with a full-covariance proposal (the cavity, then the Gaussian
of the tilted moments, up to three rounds, stopped when the moments settle), 8 nodes per score for K = 3 and up to six free scores; the
integral of a sweep is shared by the K ports and the evidence (one memoised slot). Tested against brute-force integrals for the logit (three
classes, wide cavities, two classes, an observed score) and the probit (three classes within 1.5e-2, grid-limited; two classes within 4e-3).
_Model layer:_ `categorical({ logits }, { over: cls })` takes the class axis as a named event dimension (`DistSpec.eventDims`, honoured by
the checker and by the program form, which lays it out as the last axis) and `categoricalProbit(scores, { over })` is the probit family; the compiler
makes one `multiclass-site` per observation on the K scores. Model tests (`multiclass-factors.test.ts`): a softmax and a probit classifier with
one slope per class against exact three-dimensional grids: means within 0.06 of an exact sd, spreads 0.78 to 0.86 (the three slopes share every
observation). Atoms `softmax-regression` and `softmax-regression-learned` (24 points: spreads 0.55 to 0.8 of the exact, coverage 1.0, 0.94 and 1.0
over 16 seeds; the sweeps converge slowly, so `tol: 1e-4`). Gap `softmax` became `softmax-many-classes` (more than six classes, a multiclass GP).

**Ordinal probit, ordered logistic, the asymmetric Laplace and the draw margin.** Model-layer families `orderedProbit`, `orderedLogistic`
(the library's own distributions, whose fixtures exist) and `asymmetricLaplace`: a new library distribution
(`dists/continuous/heavy-tailed.ts`, scipy `laplace_asymmetric` fixtures for three quantile levels, in the golden, contract and sampling
tests), all mapped to the registry rows (`ordinal-probit`, the new `ordinal-logit`, `asymmetric-laplace`); known cut points only (random ones are
refused with the reason). Model tests against exact grids (`ordinal-quantile-factors.test.ts`). Atoms `ordinal-probit(-learned)`, `quantile-regression(-learned)`
(an explicit grid box, since the Laplace fit of a kinked density finds no curvature: EP equals the exact posterior to three digits on 20 seeds,
which itself covers the slope only 0.85 of the time, so the recovery floor is 0.65) and `skill-draws(-learned)`: the TrueSkill draw margin comes out
naturally as an ordinal probit of the skill difference with cut points `-m/sqrt 2, m/sqrt 2` (EP within 0.1% of the exact posterior on the
tree-like three-player graph, spreads 0.83 to 0.95 in the plated form). Gap `ordinal-asymmetric` became `ordinal-random-cuts` (random cut points, a
Student-t with random df and scale: a third axis of the grid).

**Coverage:** atoms 84 of 86 run (72 of 76 at the merge with main; this package added ten atoms and flipped two); benchmarks 27 of 27. The EP log
evidence against the grid evidence of the same model on the new atoms: beta two shapes 2e-5, ordinal 0.002, Student-t df 0.006, Weibull shape 0.036,
quantile 0.07 and 0.02, draws 0.12 (loops), softmax 0.30 (loops).

## 18. Engine quality: summed-out mixtures with random parameters, one flat-cavity policy, exact priors, the NaN evidence explained (2026-11-11, `agent-pp171-quality`)

**Summed-out mixtures with a random share or random precisions (gap `mixture-site-random-parameters`, closed).** `normalOutliers(loc, scale,
share, width)` with a random scale (the precision idiom of a gamma latent) or a random share (a beta latent), and `normalSpikeSlab(inclusion,
slab, spike)` with a random inclusion probability or random slab or spike widths, compile to the gate of their two components over a
hidden Bernoulli switch (one per observation or element, a conjugate `bernoulli` factor on a beta share, a prior on a known one): the same
graph as the explicit-switch form (`where(z, ...)`), the switch left out of the posterior (`hiddenSwitch`, `observedOutlierSwitch`,
`latentSpikeSlabSwitch` in `factors.ts`; the observation goes through the existing `observedSwitch`, the branches are the existing
`gaussian-gamma-precision` and `gaussian`). The grid machinery of `pairs.ts` was not needed. A known share and scale keep the exact
`mixture-site`; a vector read by a matmul with a random share is refused (one joint covariance), as is a random width. Atoms
`clutter-mixture-learned` (centre 1.4316 +- 0.0903 against the exact 1.4318 +- 0.0903, precision 3.409 +- 0.830 against 3.419 +- 0.858, share
0.1497 +- 0.0540 against 0.1504 +- 0.0561; the log evidence within 0.02 nats of the grid) and `spike-and-slab-marginal-learned` (share 0.4200
+- 0.1492 against 0.4209 +- 0.1517, slab precision 0.2011 +- 0.1021 against 0.2014 +- 0.1039; evidence within 0.014 nats) run and pass their
calibration over 20 and 30 seeds; atoms 86 of 86, and the floor in `ep-coverage.test.ts` is 86. Model tests against two-dimensional grids
(`probprog/model/mixture-random-factors.test.ts`).

**One policy for a flat cavity (gap `improper-start`, closed).** An operator that integrates over its cavities (`gaussian-gamma-precision`,
the gate, `mixture-site`, `prior-site`) declares `needsDensity`; the engine (`readable` in `engine.ts`, the one place every operator
call and every evidence call goes through) shows it a flat Gaussian cavity as a very wide one (`FLAT_CAVITY_PRECISION`, variance 1e6)
and takes the wide cavity back out of the message it returns, so such a factor speaks first on a variable that nothing else has spoken
of, and a flat target is not `blocked`. An improper cavity (negative precision) is still handled by the existing engine policy (an
operator that throws `ImproperMessageError` sends nothing, counted as an improper event). Removed: `cavityOf` (gaussian.ts), `widened`
and its constant (gate.ts), the compiler's start messages on the gate, the scalar `mixture-site` and the precision latent with a known
mean. Kept: the start message of a mixture row on a packed vector (a vector family, not a scalar one). Proof of no behaviour change: all
86 atoms run before and after, beliefs within 1e-9 posterior sds, log evidence within 1e-12 nats, the same number of sweeps
(`message-passing/flat-cavity.test.ts` holds the policy and the list of operators that declare it).

**Exact priors (gap `projected-priors`, closed).** Half-normal, half-Cauchy, uniform and truncated-normal priors are the numeric
`prior-site` operator (`operators/prior-sites.ts`, an entry of the likelihood registry's machinery): its message is the moment match
of the cavity times the density by the panelled quadrature of the likelihood sites, redone at every sweep, on the value or on its log
(a positive scale, with the Jacobian). The first message (a flat cavity) is the projection of the density alone, as before. Tested
against the scan-and-trapezoid of `densityMoments` and against the old one-off projection (`prior-sites.test.ts`). Effect on the atoms with
such priors: the log evidence against the grid evidence of the same model: uniform 0.09 nats off before, now exact to 1e-6 (a
Gaussian likelihood makes the posterior a truncated normal); half-normal 0.42 before, 0.004 now; half-Cauchy 0.17 before, 0.015 now. What
remains is the Gaussian belief of the log scale, not the model. The beliefs moved by up to 0.16 posterior sds (half-normal) and every atom
check still passes. One accuracy trade-off, stated: a dispersion with a half-normal prior whose posterior piles up against zero (the negative
binomial test of `probprog/model/likelihood-factors.test.ts`: mean within 0.06 sds) now has a spread 1.52 of the grid's where the projected
prior gave 0.87 by being narrower; one Gaussian on the log holds the long right tail of a log-normal. Not a new error source: the exact
prior shows what the Gaussian-on-the-log belief costs.

**The NaN evidence of `logistic-learned` (explained).** The evidence of a graph is the sum of the factors' parts, each an integral of the
factor against its cavity. The learned logistic regression compiles to one plated factor for all rows; the scale mixture on the weights
(`gaussian-mv-precision-scale`) sends it a site with a negative precision (the matched Gaussian of a heavy-tailed prior against a sharp
likelihood is wider than the cavity), so the cavity of the plate is indefinite and its integral against the logistic likelihood
diverges: the EP energy is undefined at that fixed point, not wrong. The posterior now says so (`diagnostics.evidenceUndefinedAt`, from
`evidenceDefects`: `factor:y: improper cavity`). With `plates: 'expand'` (a factor per row, each cavity proper) the evidence exists:
-26.520 against the grid's -26.474 (0.046 nats); the weights' posterior means move by 0.002 to 0.016 (a few hundredths of a posterior sd). The remaining NaN of the same kind is any
plated factor with a negative-precision neighbour.

**Property tests that depended on the seed (item 6).** Two property tests of `message-passing/operators.test.ts` failed on some seeds
in the gate: `product: EP to the output and to each factor` (seed 1817728420, and 1.7e-4 against 1e-4 elsewhere) and
`gaussian-log-scale: EP to the log scale matches the brute-force tilted moments and evidence` (seed 1395539355). Both were limits of the
brute-force reference, not rule bugs: the log-scale counterexample (y = -3, loc 0, a sharp peak in the log scale) has the rule's evidence
within 3e-9 of a reference refined to 200 panels of order 24, while the fixed 10-panel reference was 5.0e-5 off, exactly the tolerance;
the product case is the same (a ridge along a b = y that 14 panels resolve to 2e-4). `bruteTilted` (`testing/message-passing.ts`) now adapts:
after a pass it zooms every Gaussian port's interval onto the nodes that carry weight (a panel of margin) and runs again until the
intervals stop shrinking, so a likelihood far narrower than its cavity keeps its nodes. The audit: `operators.test.ts` is the only `message-passing` test file that uses `fast-check` (eight properties; the other seven
passed every seed and the stress run unchanged), the other files that use the reference (`numerical`, `extremum`, `alpha`, `variational`,
`rows`, `probit-slice`, `families`) test fixed cases and pass with the adaptive reference. Both counterexamples are pinned as fixed `examples`. The file
takes `OMAI_FC_SEED` (one fixed seed for every property) and `OMAI_FC_SCALE` (a multiple of the runs): it passes on the seeds 1395539355,
1817728420, 7, 123456 and 987654321 and with 15 times the runs; the old reference fails the pinned log-scale case.

Done for the GAM's remaining forms (2026-11-11, `agent-pp181-gamep`): EP runs on all twelve (prior, hyperprior) combinations of the
Bayesian GAM. `apply` takes a chain of steps (`mul`, `add`, `pow` and the registry, now with `sqrt` and `recip`) and a gamma input;
the compiler collapses arithmetic on an `apply` output into one chain and prunes unread `apply` factors; `z / g(x)` for a chain `g`
is the new fused factor `apply-product` (`y = z f(x)`, closed form in `z`, one integral over `x`). Penalty with log-normal
hyperpriors: curve 0.005 response sd from the exact reference, log smoothing 2.78 against 2.53; ARD: 0.011 sd from NUTS. Remaining:
the posterior of the non-centred coefficients is per-coefficient marginals (band 17% wide), no VMP rule (gap `non-centred-scale`).


---

## Appendix: the first plan, as written on 2026-11-11 (before the work above was done)

The audit and the work packages of the first version of `ep-plan.md`. Superseded: 9 of 27 benchmark models ran then, all 27 do now.

## 1. Where we are (measured, not remembered)

The engine has 35 factor operators and eight message families (`gaussian`, `gaussian-mv`, `gamma`, `beta`, `bernoulli`,
`discrete`, `dirichlet`, `normal-gamma`). The compiler maps these likelihoods: `normal`, `bernoulli` (probit, logistic,
greater-than), `binomial`, `poisson` (with an exposure), `categorical`; priors on `normal`, `gamma`, `exponential`, `beta`,
`dirichlet`, `bernoulli`, `categorical`; affine maps of Gaussians; products of two random values; max and min; a gate
`at(components, z)` / `where(z, a, b)` when the components are known values or fresh variables.

Compiling each of the 27 benchmark models and asking for the EP verdict: **EP runs on 9** (beta-Bernoulli, normal-normal,
gamma-Poisson, logistic and probit regression, the Gaussian mixture, TrueSkill, the AR(1) state space, item response scoring).
The other 18 are refused for a small number of reasons:

| blocked by                                                                                                   | models                                                                          |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `exp`, `square`, `tanh` of a random value (no unary-function rule)                                           | poisson regression, funnel (non-centred), banana, Bayesian neural network       |
| a prior with no message family: half-normal, half-Cauchy, log-normal                                         | eight schools (both forms), radon, negative binomial, GP (length scale), funnel |
| a likelihood not mapped: negative binomial, Student-t, Laplace                                               | negative binomial regression, (robust regression page)                          |
| a random scale that is not the precision idiom                                                               | eight schools, funnel (centred), stochastic volatility, clutter                 |
| a constraint (`ordered`)                                                                                     | ordered Gaussian mixture                                                        |
| a dense Gaussian vector (`multivariateNormal`) and opaque custom functions (kernel Cholesky, stick breaking) | GP regression (both), stick-breaking mixture                                    |
| a gate over random components, a discrete variable in arithmetic                                             | Gaussian HMM, bimodal                                                           |
| a Dirichlet with no declared size (the compiler throws instead of refusing: a bug)                           | LDA                                                                             |

Added on 2026-11-11: an EP rule for a Gaussian with a gamma-distributed precision (`gaussian-gamma-precision`), for an
observed or a latent value (linear regression with an unknown noise level, a population with an unknown spread; checked against
the exact posterior on a grid).


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

## 2026-11-12 Two quality gaps closed (`agent-a9e812-gaps`)

- `beta-two-shapes` honours its power. It was a message rule that declared `power: true` but put the power only on the whole
  integrand (the integral to the power, not the integral of the powered factor) and never divided its site by it. It is now a
  site rule: given the shapes the factor to the power `alpha` times the cavity is a beta with parameters `e1 + alpha (a - 1) + 1` and
  `e2 + alpha (b - 1) + 1` of mass `B(A, B) / B(a, b)^alpha`, the grid integrates that, and `projectSite` divides the site by the power
  as for every powered factor. At power 1 the arithmetic is the one it was (the digest does not move for power 1); tested at power 1
  and 0.5 against a brute-force triple integral (`concentration.test.ts`). Gap `beta-two-shapes-power` removed.
- Improper evidence is caught once, in the engine (`factorEvidence` in `engine/evidence.ts`, which `evidenceOf` and `evidenceDefects`
  both use), for every operator, with the `evidenceOrNaN` that `evidence.improperIsNaN` already used. A factor whose cavity is improper
  has no evidence (`NaN`), never an error that ends the run, and `evidenceUndefinedAt` names it; any other error still propagates.
  `model/bayesian-neural-network` no longer records a thrown error: the EP run completes its 200 sweeps without converging
  (status `stopped`), with no log evidence and the improper factors listed in `diagnostics.evidenceUndefinedAt`. Gap
  `evidence-catch-uneven` removed.

## 2026-11-12 The atlas is read against the code (`agent-a5ac69-atlas`)

- Every row of `ep-priors.md` has a machine-readable `ran by` column (`atom:<name>`, `test:<path>#<text>`, or `not yet`), a `status`
  column that `make ep-report` writes from the coverage (`runs`, `planned`, `broken`) and a prose `note`. `make ep-report-check` fails
  when a status word is stale, an atom does not exist or does not run, a test file or its text is gone, or a row names nothing
  (`scripts/ep-atlas.ts`, `scripts/ep-atlas.test.ts`; `ep-status.md` counts the rows). Gap `docs-drift` removed.
- First reading: 37 rows, 36 run, one `not yet` (the direct `studentT` / `laplace` / `cauchy` prior). Six rows had no atom (they said
  "probed", "tests", "test", "tested", or pointed at `ep-plan.md`): each now names a test file and a text it holds. The row of a mean
  whose spread is a random multiple of the noise precision (`kappa` random) has no test that runs it under EP, only the compiler mapping
  (`gates-categorical.test.ts`) and the gamma-product rule it uses (`observed-scale-mixtures.test.ts`); a probe confirmed it converges,
  and its note says so.

## 2026-11-12 A gate over likelihoods of different families (`agent-a952d6-gate`, gap `gate-node` closed)

- `d.gate(index, [poisson(..), negativeBinomial(..)])` is a distribution family of the model language whose branches are different
  families: the name carries the branch families (`gate(poisson,negativeBinomial)`) and the parameters are the index and each branch's
  own, flattened as `index`, `0.rate`, `1.n`, `1.probs`, so every analysis walks the expressions as it does for any family. The table
  of `distributionFamilies()` answers to any such name; the distribution the other engines read is the library's `mixture` with the
  one-hot weights of the index (a log joint at fixed values is the density of the branch each row's index names). Branches are
  scalar-event families, all discrete or all continuous; fewer than two branches or a nested gate throws.
- The compiler has one more entry in the ordered table of element mappings, `gate-families`: one engine `gate` per element on the
  index (a Bernoulli or categorical latent, one state per branch), whose branch `k` is the factor that branch's family maps to when it
  is observed alone. It reuses the whole table rather than writing the likelihoods again: each branch goes through
  `observedElementMappings` as a synthetic observed node of its family, the one likelihood factor it made is taken out and becomes the
  gate's branch (the value algebra it needed stays), and the gate machinery (`GateBuilder`) is the one the other gates use. A branch
  with no random parameter has no factor to take: its likelihood is a weight of its state, from the library's density of the family, a
  `table` over the index beside the gate (or the only factor when every branch is known). A known index maps the branch it names alone.
- Refusals, with a code and the node and branch named (`_reasons.ts`): an index that is a computed random value
  (`gateIndexNotDiscrete`), states versus branches and an index outside the branches (`support`), a branch the mapping of its own family
  refuses (`gateBranchRefused`, with that reason), and a branch that is not one likelihood factor on its own (`gateBranchNotOneLikelihood`:
  a hurdle, a nested gate or a table). Existing models compile as before: the graph corpus snapshot is unchanged and 30 digest items
  around gates and mixtures are BITS.
- Against references (`factor-graph/gates-families.test.ts`, EP through `runEp` and `epRoute`, grid posterior of the same model with
  the index summed out by the library's mixture): a Poisson against a negative binomial of another mean (10 rows): means within 0.002
  sds, spreads 0.99 to 1.01, log evidence 0.003 nats; a known Poisson against a learned one (the known branch as a weight): means within
  0.001 sds, evidence 0.004; a Gaussian regression on two weights against a constant: means within 0.04 sds, spreads 0.90 to 0.93,
  evidence 0.02.
- What remains: a gate whose branch maps to several factors (a hurdle) or to a gate or table of its own, nested gates, a computed
  random index (a function of a latent that is not a discrete variable), and vector-event branches. Gap `gate-node` removed;
  `mixture-regression` (direction) says what is mapped.

## 2026-11-12 Random cut points and a Student-t with a random scale and degrees of freedom (`agent-a5524e-cuts`, gap `ordinal-random-cuts` closed)

- A three-variable numeric site, `likelihood-triple` (`operators/likelihoods/triples.ts`), a registry of densities `p(y | x0, x1, x2)` read by
  one operator as `likelihood-pair` reads its pairs: `student-t-full` (a random location, log scale and degrees of freedom: a Gaussian or
  gamma port each), `ordinal-probit-cuts` and `ordinal-logit-cuts` (a random predictor and the two cut points that bound the observed
  category; the first and the last category have one cut and ignore the other port). Any port may be a known value. The grid is the one
  the covariance sites use, with a profile of its own (`gridProfiles.triple`), a finer grid along the heavy-tailed coordinate of the
  Student-t (`fineness`), the ordinal rows read from tables of pairs of nodes (`GridIndexed`: a pass costs the pairs, not the triples) and
  one integral per sweep for the messages and the evidence (`memoised`). The ordinal probit with Gaussian cavities does not use the
  grid: given its latent score the predictor and the two cuts are independent (a Kalman update and two truncated Gaussians), so the
  tilted moments are one smooth integral over the score, exact, and the order of the cuts is the event itself. The same rows have no
  variational message where the density is zero out of order (the ordinal ones), and a mean-field message for the Student-t.
- `likelihood-pair` has a row `student-t-scale` (a random scale, known degrees of freedom: a Student-t regression with an unknown scale,
  which was refused). `logDiffPhi` and `logDiffSigmoid` moved to `numerics/_scalar.ts` (three copies, one place).
- Compiler: `d.orderedProbit(predictor, cuts, { over })` and `d.orderedLogistic` take the cut dimension as `categorical` takes its class
  dimension, so the cut points are a latent vector over a dimension of its own (`constraint: { kind: 'ordered' }` makes the prior an ordered
  Gaussian) or any affine form of Gaussian variables over it (a learned draw margin is `margin * sign` with `sign` known data over the
  dimension); one `likelihood-triple` factor per outcome on the predictor and the two cuts around its category, a known cut a known port.
  A Student-t with a random scale and random degrees of freedom is `student-t-full`, a random scale alone `student-t-scale`. Refusals
  removed: `cutPointsRandom`, `shapeAndScaleBothRandom`; added `cutPointsShape` (cut points over more than one dimension). The table
  entries `heavy-tailed` and `ordered-numeric` list the new operator.
- Against references. Operator tests (`triples.test.ts`) against a brute-force product rule: moments and evidence within 3e-3 relative
  for the ordinal and Student-t rows (a wide outlier case against a refined grid within 0.05, the limit of a grid where the scale is
  small and the location a spike); `accuracy.test.ts` row `likelihood-triple` (15 cases): mean 7.9e-4 sds, sd 4.8e-4 (the probit rows
  1e-4 or better, the Student-t 4e-5, the logit on the grid 7.9e-4). Models (`observed-triples.test.ts`): a Student-t with three
  random parameters against a 90-cell grid (the mean of the location within 0.2 sds, spreads 0.75 to 1.25 of the exact for the location
  and scale, the degrees of freedom 0.5 to 1.5: a long right tail), an ordered latent vector of cuts and a slope against a 70-cell grid
  (means within 0.3 sds, spreads 0.75 to 0.95: the slope and the cuts share every row), a learned draw margin against a 180-cell grid.
  Atoms: `ordinal-probit-cuts-learned` (three categories, slope and two ordered cuts; means within 0.03 exact sds, spreads 0.83 to 0.94,
  Kolmogorov distance 0.05; recovery over 20 seeds: coverage 0.80, 0.85, 1.00, rms error 1.25 sds at most; 3 s) and
  `student-t-full-learned` (location, scale and degrees of freedom; means within 0.02 exact sds, spreads 0.95 to 1.01, Kolmogorov distance
  0.03; recovery over 20 seeds: coverage 0.80, 0.95, 0.95, rms error 1.2 sds at most; 6 s, 128 s for the 20 seeds).
- What is not done: the ordinal logit with random cuts runs on the grid (about 25 ms per outcome and sweep against 1 ms for the probit), the
  ordinal rows have no variational message, and the Student-t grid is the limit of its resolution where a small scale makes the location
  a spike against a wide cavity (an error of 0.05 sds in the first sweeps; it settles with the cavities). Gap `ordinal-random-cuts` removed;
  the variational limit is added to `variational-rules`. The digest has two atoms and two variational items more and the registry one
  operator; the 8 existing items checked (ordinal, skill draws, Student-t, quantile, the uniform prior, the gates) are as at the baseline.

## 2026-11-12 A learned covariance with few groups, and Wishart matrices of dimension four (`agent-a3e7a0-matrix`)

- Cause of the gap `hierarchical-matrix-few-groups` (mean too narrow, matrix too wide with three or four groups): a factor per group vector
  (`mvn-wishart`) shares the population mean and the Wishart matrix with every other group, so the graph has loops through the mean and the matrix and
  EP's product-form beliefs lose the dependence of each vector on the mean. Not the family projection (the matrix alone was exact with a plated
  factor or a known mean) and not damping. Measured before the fix against the atom's grid reference (`random-intercepts-slopes-learned`, three to ten
  groups): the mean's spread 0.78 to 0.95 of the reference, the evidence 0.1 nats low at three groups, 12 to 14 sweeps.
- New operator `mvn-wishart-group` (`message-passing/operators/covariance/hierarchy.ts`): the vectors of all the groups, their shared Gaussian mean
  and the Wishart matrix are the ports of ONE factor. Given the matrix everything is Gaussian, so the vectors and the mean integrate out together
  in closed form (`N(m_j; mean, V_j + L^-1)` per cavity, then the mean against its cavity: `c_j`, `P = W_0 + sum (L - L T_j L)`, `g = h_0 + sum L T_j h_j`),
  the Wishart is matched on the Bartlett quadrature whose base measure is the cavity times the conjugate part the vectors' cavity means give it
  (`|L|^((J - 1) / 2)` and their scatter; the remainder is smooth), and each vector and the mean are matched to the mixture of their conditionals over
  the nodes. One tilt serves every port and the evidence of a sweep (the cache key is the cavities to ten digits of their largest entry: cavities that
  are the same to rounding must hit). The variational rule is the sum of the per-vector rules, so the VMP digest items agree to 1e-10. The compiler
  maps a multivariate normal with a Wishart precision and a Gaussian-vector mean as this one factor (`_mappings/matrix.ts`; a known mean stays one
  `mvn-wishart` per vector, exact on its own); `_table.ts` names the operator, the graph corpus has the two lines of the changed atom, the compiler
  page is regenerated.
- Measured after (same atom, three to ten groups, prior degrees of freedom 2.2 to 3, 3 sweeps): the mean's spread 0.96 to 1.00 of the grid
  reference, the matrix 0.99 to 1.00, every group 1.00, means within 0.05 reference sd, the log evidence within 0.002 nats of the exact (a new lazy
  `exactLogEvidence` of the reference, every constant of the model in it). The grid reference itself is a few percent high for the mean's spread at
  three groups: seed 10 of three groups, a graded polar Gauss-Legendre integration of the same model gives 0.6887, EP 0.6877, the atom's three-pass
  midpoint grid 0.7068 (0.7274 on a finer grid: it does not converge there). The atom's tolerance is now measured (`sdRatio` 0.93 to 1.06, not the
  `close` rung). Operator tests against a brute-force reference that does not use its algebra (a grid over the three entries of a 2 by 2 matrix, the
  joint Gaussian of the readings conditioned densely), the exact conjugate limits in dimensions 2 to 4 and importance sampling in dimensions 3 and 4.
- `matrix-larger`: dimension 4 is now allowed (`BARTLETT_ORDER[4] = 3`: 59049 nodes), for `mvn-wishart` and `mvn-wishart-group`. Against importance sampling
  (eight groups of four effects, ESS above 13000): means within 0.03 reference sd, spreads 0.97 to 1.01 of the sampled ones for the mean and the matrix,
  with a random mean (14 s, 3 sweeps) and a known one (15 s, 7 sweeps, one `mvn-wishart` per vector). Dimension 3 with eight groups: 2 s, 3 sweeps (it was 12 s,
  30). Dimension 5 with two nodes per coordinate leaves the matrix's spread 0.85 to 0.99 of the sampled one (5% to 15% short): it stays refused, as does
  three nodes (fourteen million). Not done, and why: an LKJ prior of order 3 and above needs a new site over the partial correlations (three betas whose
  product loses their dependence; six coordinates with the standard deviations), and a model-language form for a covariance made of sds and a correlation
  matrix; a latent bivariate vector with its own sds and correlation would be a group form of `mvn-sd-corr`; the normal-Wishart variable inside a larger model is a
  compiler pattern with its own readback. The gap `hierarchical-matrix-few-groups` is replaced by `correlation-strong-few-pairs`, the beta belief of a
  strong correlation (spread 0.81 to 0.92 of the exact), which this change does not touch; `matrix-larger` lists what remains.
- Searched and reused: Wishart family, `wishartRule` and the Bartlett helpers, `gaussianSide`, `mvn-wishart`'s variational rule (called per vector), the
  atom machinery and `grid3` reference, `posterior3`, the library's `gamma` and `normal` samplers and `spd` for the importance-sampling reference (`testing/covariance-group.ts`).
  Digest: `atom/random-intercepts-slopes-learned` (graph 52 to 43 factors, 14 to 3 sweeps, evidence and beliefs moved toward the reference),
  `vmp/random-intercepts-slopes-learned` (28 sweeps for 25, beliefs within 1e-10) and the registry line (one more operator) re-accepted; `atom/random-intercepts-slopes` gains the reference's evidence only.
## 2026-11-12 Variational message passing for the dense non-centred scale (`agent-a40-vmp`, gap `non-centred-scale` narrowed)

- `gaussian-mv-function-scale` has a variational rule (support `numeric`) and an expected log: with the standard normals integrated
  out the factor is the Gaussian density of the vector given `x`, so the message to the vector is `E[1 / u_j(x)^2]` on the diagonal
  (no linear part) and the message to `x` is the non-conjugate message (`numericMessage`) of `sum_j -log |u_j(x)| - E[beta_j^2] /
  (2 u_j(x)^2)`, on `quadraturePorts` (Gauss-Hermite for a Gaussian `x`, a grid in the log for a gamma one): the quadrature of the
  variational kernel, no second one. Tested against `bruteExpectation` (one coefficient: the fixed point maximises the brute-force bound,
  for a Gaussian and for a gamma `x`; two coefficients: the diagonal message, the natural gradient of the expected log and the bound
  against brute force). On the Bayesian GAM (penalty prior, log-normal hyperpriors, 60 points): curve 0.004 response sd from the exact
  reference, band 0.996, log smoothing 2.58 +- 0.57 against 2.53 +- 0.62, 46 sweeps; the offer lists `vmp` for it.
- Not done: ARD. A VMP rule for `apply-product` as a deterministic factor (`defines` its output; the product of the two beliefs'
  quadratures; it matches the brute-force bound on a scale read by a Gaussian) was written and left out: the compiler ties the scalar
  outputs of ARD to one Gaussian vector with `dot-constant` factors that define each output from the vector, and a variable with two
  definers does not compile, EP included. The gap says what the compiler would have to emit.
- The `evidence.improperIsNaN` flag is gone from the nine operators that set it (`gaussian-gamma-precision`, `gamma-rate`, `gamma-product`,
  `dirichlet-concentration`, `beta-two-shapes`, `gate`, `mvn-sd-corr`, `gaussian-mv-precision-scale`, `gaussian-mv-function-scale`): the
  engine catches an improper cavity for every operator (`factorEvidence`). The option stays in `defineOperator` for a caller of
  `logEvidence` outside the engine; a direct call of those operators on a flat cavity now throws `ImproperMessageError` where it gave `NaN`
  (`evidence.test.ts` says so). 109 digest items around gates, mixtures, gamma precisions, priors and covariances are BITS identical; the
  one item that moves (`atom/gate-random-learned`, evidence by 2.8e-12) moves the same on the commit before, so it is not this change.


## 2026-11-12 The softmax beyond six classes, and a multiclass Gaussian process (`agent-a3ff91-softmax`, gap `softmax-many-classes` narrowed)

`multiclass-site` integrated up to six free scores on a product grid (a node for every combination). It now takes any number of classes
under EP, with the product grid kept up to six (the digest items of the existing softmax atoms are bit-identical) and a new rule beyond
(`operators/links/_multiclass-auxiliary.ts`, chosen by the parameter `integration`: `auto` is the product grid while it reaches, `product`,
`auxiliary`):
- **Softmax**: `S^-p = int t^(p-1) exp(-t S) dt / Gamma(p)` with `S = sum_k exp(z_k)` and `p` the power of the factor. Given the auxiliary `t`
  the factor is a product over the classes, so the mass is one integral over `log t` of the product of `A_k(t) = E_cavity[exp(-t exp(z_k))]`
  (the numerator `exp(p z_y)` shifts the observed class's cavity to `N(m + p v, v)`), and every score's tilted mean and variance are the
  mixture over `t` of its conditional moments (law of total variance, centred). The outer integral is the existing panelled grid (`integrateLine`,
  a new profile `gridProfiles.line` of `numerics/_grid.ts`: 12 panels of 8 nodes after a coarse search of the window, shrunk onto the mass),
  the inner ones a trapezoid rule over the cavity (spectrally accurate for a Gaussian times a smooth step), so the only approximation is the
  quadrature. A known score is a constant (`log S` of the known scores), the power is the shape of the gamma (fractional EP works, down to 0.03).
- **Multinomial probit**: `c = z_y + e` is Gaussian under the observed score's cavity (variance plus one), so the mass is
  `int N(c) prod_j Phi((c - m_j) / sqrt(1 + v_j)) dc` and every other score's moments given `c` are the truncated-Gaussian closed forms; one
  coordinate on the same line grid. The power must be one beyond six scores (a power of the integral is not an integral).
- Variational message passing stays on its product grid: `unsupported('vmp')` says so for more than six classes (it was an unbounded cost).

Checked against: the brute-force integral (`bruteTilted`) for two to four classes (precision and mean term within 1e-3 to 2e-3, the brute force
being the coarser of the two), the product grid at six classes (3e-3, which is the grid's own error: at three wide cavities the exact evidence
of identical scores is `-log 3`, which the new rule gives to 1e-8 and the grid to 1.5e-4), and for any number the identities that hold
for every categorical likelihood against any cavities: the classes' masses add up to one and their tilted means and second moments, weighted by
the masses, add up to the cavities' (softmax of 8, 12, 20 classes: 1e-9, 1e-8, 1e-8; probit of 8, 12, 20: 1e-10 to 1e-9). A softmax of 20
classes is about 3 ms (wide cavities 6 ms) a tilted moment, the probit about 4 ms; the product grid at five classes was 80 to 140 ms.
- `atom/softmax-choice` and `-learned`: twelve choices among eight alternatives with a price and a quality (two shared coefficients), against
  the grid of the two coefficients: plain EP is 0.77 to 0.83 of the exact spread (the means 0.15 exact sds off, the evidence 0.46 nats below the
  grid, as with three classes), and the learned twin recovers its coefficients over 16 seeds.
- `atom/gp-softmax`: a multiclass Gaussian process works without a compiler change: one dense Gaussian vector over the (input, class) pairs
  (`covariance` the kernel between the inputs where the classes agree and zero elsewhere), the scores of an input gathered with `ops.at`, the
  label a categorical over the class dimension: the compiler reads the gather as the scores of the multiclass site. Two inputs and three classes
  against quadrature of the six values: the means within 1e-3 and the spreads within 0.2% of it.
- Not done: variational message passing beyond six classes; a joint site message over the classes of one observation (the site sends each
  score its own Gaussian, so a prior correlation between classes is carried by the dense prior only); a compiler pattern for a plate of dense
  vectors (one process per class) so that the cells need not be written out; the probit with a power beyond six; a reference for the multiclass GP
  at more than two inputs (the posterior has as many dimensions as values).

## 2026-11-12 Loops: tied groups pack hierarchies and tournaments (`agent-a22fa5-packing`, gap `loopy-overconfidence` narrowed)

- `findGroups` (`_plan.ts`) now groups normal latents that are vectors over one dimension or have a prior of their own, when an
  observed node reads them linearly (sums, differences, known multiples, indexing, and a link such as `probit` at the top of a
  parameter; a product of two random values is not linear, so the non-centred `mu + tau z` reads nothing). A group with a scalar beside
  the vectors, or one vector that a row reads at two places, is joined (200 coordinates at most). Known priors keep the old design (the
  prior is on the vector, `latentGrouped` now takes vector members); the others are a tied group: the member is made as it is and each
  element is tied to its coordinate of the vector by an exact `difference` constraint (`tieToGroup`; two definers of one variable are not
  allowed, so the constraint is an observed zero difference), and the vector starts with a vague message in each coordinate (variance
  100, replaced by the factor's first update; without it nothing can speak first). Where a tied graph has no EP energy (the cavity of a
  tie is improper), the evidence is that of the same model compiled without groups (`mp.ts`, method `ep-energy (rows expanded, unpacked)`).
- Against references (`model/factor-graph/tied-groups.test.ts`): radon (exact quadrature) floor effect sd 0.71 -> 0.91 of the exact, county
  intercepts 0.82 to 0.98 -> 0.92 to 0.99, mean of the hierarchy 0.85 -> 0.88, means within 0.11 sd (177 sweeps, was 180); the radon counties
  of the pocs (centred, 12 counties, 4 x 600 NUTS draws): floor effect 0.83 -> 0.96, 24 counties 0.68 -> 0.86; tournament of three players
  (exact grid) 0.74, 0.76, 0.85 -> 0.99, 1.00, 1.00 in 3 sweeps (28), draws tournament 0.91 -> 1.00 and with the games simulated 0.82 to
  0.95 -> 1.01 to 1.06, their evidence 0.12 and 0.23 nats from the grid -> 0.0001 and 0.006; the tournament program with a learned spread
  (8 players, 3 rounds) 0.78 -> 0.97 of NUTS and the truth of `skill-difference-learned` inside the 90% interval 0.87 -> 0.94 (24 seeds).
- Not packed, and why: the multiclass class slopes and the two shapes of a beta outcome (their sites send one message per port, so a joint
  vector would receive no off-diagonal term: both need an operator with a vector port, which belongs to the message-passing operators);
  the 2PL item response model (the logit `a (theta - b)` is a product, bilinear in the latents: 0.04 to 0.7 of the NUTS spread of the
  discriminations, 30 students, 8 questions, unchanged); the non-centred hierarchy (no linear read; the floor effect 0.86 of NUTS, unchanged).
- Digest items moved deliberately: `model/radon`, `model/trueskill-three-players`, `atom/skill-draws`, `atom/skill-draws-learned`,
  `atom/skill-difference-learned`, `program/skills` (spreads move towards the reference; the means within 0.02 sd except the learned
  tournaments' skills, 0.05 sd). The two draws atoms moved to the `plated` rung with evidence within 0.02 nats.

## 2026-11-12 ARD under variational message passing: the scalar products assembled into one vector (`agent-a6a2ec-ard`, gap `non-centred-scale` closed)

- The compiler assembles a non-centred vector with a scale of its own for each coefficient (`z_j / sqrt(alpha_j)`, ARD) with one new
  operator, `gaussian-mv-stack` (`D = (y_0, ..., y_n-1)`, in `linear.ts`), instead of a `dot-constant` tie per coordinate:
  `tiedToVector` (`_mappings/derived.ts`) emits it over the `apply-product` outputs, and the derived mapping declares it in `requires`
  (`operator-contract.test.ts` checks it exists and that what the corpus emits is declared). The stack defines the vector from its
  coordinates, as each `apply-product` defines its scalar, so every variable has one definer and the graph compiles under both rules.
  Under EP the message to the vector is the diagonal of the scalars' cavities and the message to a scalar the marginal, at its
  coordinate, of the vector's cavity with the other scalars' cavities (the `n` unit ties in one factor; exact on a linear Gaussian
  model, tested against the closed-form posterior); the evidence is the log partition of the vector's cavity plus that message. Under
  the mean field the belief of the vector is the diagonal Gaussian of the scalars' beliefs, and the message to a scalar is the expected
  log of the vector's cavity with the other coordinates at their means.
- `apply-product` has its variational rule (the one of the earlier prototype, `definedFunctionMessage`: it defines its output, the
  mean and variance of the pushforward of `z f(x)` to `y`, the non-conjugate messages of the readers' expected log to `z` and `x` on the
  product of the two quadratures; no second quadrature), tested against the brute-force bound for a Gaussian and for a gamma `x`.
- Bayesian GAM, ARD, `bayesianGamMethods('ard')` now offers `vmp`: at k = 10 the curve is 0.029 response sd from EP and 0.027 from a
  sampler (EP is 0.032 from it), the log noise -1.60 +- 0.10 (EP -1.67 +- 0.12, sampler -1.67 +- 0.10), the band 1.39 times the
  sampler's (EP 1.005), 627 sweeps to a change of 1e-5 at damping 0.3 (the default of no damping stalls, status `stopped` with the
  reason). Against the exact posterior of a two-coefficient ARD regression (`models/ard-exact.test.ts`, a grid over the two log
  relevances): EP means within 2e-5 and spreads 1.00001 of exact (the model is a tree of exact factors but for the Gaussian
  projection of the data-dominated posterior), the mean field 0.08 off in a mean of 1.9, the large coefficient's spread 1.28 times
  the exact one and the small one's 1.04.
- EP on the ARD GAM is unchanged in kind (24 sweeps, the sampler test passes within its bounds); no digest item contains an
  `apply-product` or an ARD model, so the items are as before (the 1e-11 drifts of `atom/gamma-precision` and
  `atom/prior-beta-strength-lognormal` are not this change: neither compiles a stack or a product). The registry section of the digest
  has the new operator (67) and `apply-product` with `defines` 0 and a `vmp` rule (support `numeric`), nothing else. The graph corpus
  is unchanged (no ARD model in it).
- The varying-slopes radon model (`radonCountiesDefinition('slopes')`) was refused at compile time: `slope = mu_b + tau_b (rho z_a +
  sqrt(clip(1 - rho^2, 1e-12, 1)) z_b)` has a `clip` of a random value. A clip by known bounds is now a `min` and a `max` step in the
  chain of the value's `apply` factor (`clipAff` in `_symbolic/_functions.ts`; `chainOf` of the `apply` operator knows both steps), the
  same way a rectifier of a known operand is, so `sqrt(clip(1 - rho^2))` is one `apply` from `rho` with no second Gaussian
  approximation (a clip with a random bound is still refused, with the reason). Against the exact grid posterior a clipped curve
  agrees as the rectifier does (`symbolic-functions.test.ts`), the message of a chain with the clip binding matches a fine grid
  (`functions.test.ts`), and EP on the slopes model settles (71 sweeps, 10 counties, one seed) with `mu` and `mu_b` within 0.06 of a
  NUTS sd of the sampler's and their spreads 0.76 to 0.8 times its, the scales `tau` and `tau_b` 0.4 sd low (spreads 0.96 and 0.69) and
  `rho`, barely identified (sd 0.57 against 0.55), -0.06 against -0.35. No digest item moved (no benchmark model clips a random value).

## 2026-11-12 Stochastic volatility with learned parameters: a walk and warm starts in `epGridMethod` (`agent-a3da95-sv`)

Gap `sv-parameters` narrowed (summary rewritten in `ep-gaps.ts`, still open). No second grid method: `epGridMethod` got two knobs and a
rule for nodes that cycle (`inference/ep-grid.ts`).
- `order: 'walk'` (one hyperparameter): the first node is the middle of the range, then one step outwards on the side whose edge node
  weighs more, until both edges are `window` (default 10) nats of log evidence plus log prior below the best, so the tail nodes (a
  persistence of -0.3 or 1.9, the most expensive ones: they need 140 to 200 sweeps) are never run. `warm: true` starts each node from the
  final messages of the nearest settled node (a `messages` field on the node; the start is skipped when the graph has another shape). Both
  default off (`order: 'bisect'`, `warm: false`), so no other grid changes.
- A node that did not settle used to weigh nothing once any node had settled. It now weighs the grid where its evidence plus log prior does
  not exceed the best settled node's (a cycling state in the tail is a fair estimate; one that collapsed overshoots, and still has no weight).
- The settings that matter for this model: damping 0.2 (0.5 settled half as fast and 0.7 collapses the precision to zero), 120 sweeps a node.
  Schedules were tried (`tree` settles a node the serial order does not, but not reliably) and `tol: 1e-3` (faster, but one seed collapses).
- Calibration over 30 seeds of 60 returns (truth -1, 0.9, 11.1), 90% intervals, before (21 nodes, damping 0.5, 22 s a fit) and after
  (walk, warm, damping 0.2, 7 s, 11 nodes of which 8 settled on average): level 0.97 and 0.97, persistence 0.70 and 0.97, precision 1.00 and
  1.00, pooled 0.89 and 0.98. Against a brute-force posterior (a point-mass filter for the likelihood of each (level, persistence, precision)
  on a 26 by 30 by 24 grid with the prior, scratch script; its seed-1 numbers agree with the NUTS run to 0.2 sd): the mean of the persistence
  was off by up to 2.2 sds before and by up to 1.1 after (rms 0.59 and 0.26), the sd ratio of the persistence 0.76 and 0.85 at the median,
  the level's means within 0.35 sd (rms 0.08), the level's sd 1.6 times the reference's, the precision's 1.0.
- What did not work: a grid over the level and the persistence together (earlier, 70 points for the level), power EP (the `power` knob does
  nothing here: no factor of the model takes one), a coarser grid (15 points and a window of 8: persistence coverage 0.77).
- Not fixed, and the reason the gap stays open: (1) EP at a fixed persistence overstates the level's spread (0.50 against the exact
  conditional 0.27 at a persistence of 0.69; the mixture over the persistence makes it 1.6 times the posterior's); (2) EP at low
  persistence, and on a few seeds everywhere, cycles (a persistent oscillation of the shared precision's message at every damping, schedule
  and improper policy tried), so the lower tail of the persistence is cut (seed 27, almost no node settles: the sd is 0.2 of the reference's
  and the persistence is missed). Both are in the message passing of the chain (`message-passing/{numerics,operators}`, out of this work's
  scope), not in the grid.
- Tests: `ep-grid-sv.test.ts` (seed 1 against the brute-force numbers, in a dozen nodes), `ep-grid.test.ts` (warm start and walk on the
  Gaussian process benchmarks, where every node is exact; the weight of a node that did not settle). The EP digest: no item moves because
  of the change (`atom/gp-length-scale` differs from the committed baseline in the last digits, on the unchanged HEAD as well).

## 2026-11-12 Variational message passing: the Student-t rows checked, the ordinal rows split (`agent-acc004-vmprules`, gap `variational-rules` narrowed)

- `student-t-full` (`likelihood-triple`) and `student-t-scale` (`likelihood-pair`) already had a mean-field message (the numeric rule: the density
  is smooth everywhere); what they lacked was a reference. `variational.test.ts` now checks that the fixed point of each is a stationary maximum
  of the brute-force bound (a Gaussian and a gamma scale, three random parameters): the engine's bound is within 5e-4 nats of the brute force
  with two random ports, 3.5e-3 with three (the heavy tails against 24 and 12 Gauss-Hermite nodes), the gradient at the fixed point under 1e-3
  and 6e-3.
- Ordinal rows: what is sound. The density of a category between two cut points is zero where they are out of order, and a Gaussian belief of
  them has mass there, so `E_q[log f]` is minus infinity for every Gaussian `q` (and no lower bound can be finite: it would have to be below
  minus infinity); the family has to change (a truncated or order-by-construction belief) for a bound to exist, so those rows stay refused,
  with the reason. The first and the last category have one cut and a density that is smooth (`log Phi(c - u)`, `log Phi(u - c)`): `noVmp` of a
  triple row is now a function of the outcome and those two run under VMP (`ordinal-probit-cuts`, `ordinal-logit-cuts`, the cut that is not
  there ignored), each checked as a stationary maximum of the brute-force bound. A model with a middle category still has no VMP message
  (every ordinal atom with three categories), and its ordering comes from `greater-than`, a hard step the mean field cannot hold either.
  What would run them: cut points that are ordered by construction (a first cut and positive gaps, `exp` of a Gaussian, so the density is
  smooth over unconstrained beliefs), which needs a compiler pattern for it; not done.

- The softmax and the multinomial probit beyond six classes under VMP (`operators/links/_multiclass-variational.ts`; `multiclass-site` loses its
  `unsupported('vmp')`). Under a mean field the scores' beliefs are independent, so the expected log of the softmax needs no product grid:
  `-log S = int (exp(-t S) - exp(-t)) dt / t` (Frullani, the auxiliary variable of the EP rule) and `E exp(-t S) = exp(-t K) prod_k A_k(t)` with
  `A_k(t) = E_k exp(-t exp(z_k))`, so `E log P = E z_y + int (E exp(-t S) - exp(-t)) ds` in `s = log t`, exact up to a trapezoid rule over
  `s` (step 0.4: the integrand is analytic, decays as `exp(s)` on the left and as a Gaussian tail on the right). The message to a score is
  the same integral with that class's `A_k` replaced by `exp(-t exp(z))` at each of its quadrature nodes, through the new
  `naturalGradientMessage` (the part of `numericMessage` after the average over the other ports, now shared), from one table of the `A_k(t_s)`
  per factor and sweep (`memoised`). Nothing is bounded: the expected log is the product grid's to 1e-8 (3 free scores, and 8 classes with
  6 known, against `bruteExpectation`), the messages are the product grid's to 2e-5 (`integration: 'product'` against `'auxiliary'`, three
  and eight classes, any observed class), and the cost is the number of classes. `integration` picks the rule as for EP; `auto` is the product
  grid up to six free scores and the auxiliary rule beyond (`integration: 'auxiliary'` forces it).
- The multinomial probit is `P(y | z) = int N(c; z_y, 1) prod_j Phi(c - z_j) dc`, the expectation of a product: its log has no separable
  form, so VMP takes a bound. `log int N(c; z_y, 1) exp(G(c)) dc` is convex in `G`, so by Jensen over the scores of the other classes (not the
  observed one's) `E log P >= E_(z_y) log int N(c; z_y, 1) exp(H(c)) dc`, `H(c) = sum_j E log Phi(c - z_j)` (a Gauss-Hermite sum per class at each
  node of a grid of `c`). The gap is the spread of the other scores and is zero when they are known (checked to 1e-6); with one other score
  random it is 0.004 to 0.02 nats (a first version with a Gaussian `q(e)` over the noise, the usual bound, was 0.5 nats off for an improbable
  outcome and was dropped). The messages are exact gradients of the bound (to the observed class's score the bound at each node, to another
  score the expected `log Phi(c - z)` under the optimal density of `c`): they equal central differences of the reported bound to 2e-4. The
  rule is `numeric` in the registry's accounting; for the probit it is a lower bound, not the exact expected log.
- Tests: `multiclass.test.ts` (expected log against `bruteExpectation`, messages against the product grid, messages against finite differences
  of the bound for the softmax and the probit, the probit bound against the brute-force expected log), `variational.test.ts` (eight classes,
  two random scores: the fixed point is a stationary maximum of the brute-force bound). Atoms: `softmax-choice` and `-learned` run under VMP
  (41 and 46 sweeps; against the grid of the two coefficients the means are within 0.17 and 0.19 sds, the spreads 0.71 to 0.77 and 0.65 to 0.88
  of the grid's, the usual mean-field shortfall: plain EP has 0.77 to 0.83), `vmpCoverage` counts 88 of 97.

## 2026-11-12 The multi-start finds the other mode of a mixture: starts on the labels' priors, and mode weights held to references (`agent-a11610-modes`, gap `multimodal` narrowed)

Gap `multimodal` rewritten in `ep-gaps.ts` (still open). No second multi-start: `epMultiStartMethod` (`inference/ep-multistart.ts`) changed in two places.
- **The defect.** A start was set on the discrete ports of the factors that are not priors. In a model whose labels are independent given the
  parameters (a mixture, the clutter problem) each label has one other neighbour besides its row's likelihood, the prior, so the label's cavity
  for the likelihood is the prior's message and the start on the likelihood's own port is divided out: every random, exclusion and relabelling
  start reached the same fixed point (a two-cluster mean: one mode, a relabelled start returning to the first labelling; the clutter problem
  at 7.8 +- 0.7 where the posterior is 6.4 +- 2.3). In a chain (the changepoint program) the neighbours carry the start, which is why the earlier
  measurements found several modes there.
- **The fix.** `discretePorts` now includes the ports of a prior; a start is a message on those (a relabelled start: the relabelled belief of
  the label), and the start is carried through three damped sweeps (damping 0.5) before the run goes on at its own settings, because with
  the default damping of 0 the prior sends its own message again at its first update. `diagnostics.modeWeights` is new (the weights, summing
  to one).
- **References** (`ep-multistart-modes.test.ts`): the clutter problem's location against its quadrature on six data sets, one run -> the
  multi-start: the evidence 0.0 to 3.6 nats off -> within 0.006 nats, the mean 0.0 to 5.5 sd off (the other mode missed: data 2, one run
  at 1.6 where the posterior is 9.0 +- 1.4) -> within 0.03 sd, the sd 0.26 to 0.49 of the truth's (0.95 on data 1, where one mode has all the mass) -> within 4%; the second mode weighs 0.28, 0.001, 0.027,
  0.17, 0.23, 0.026 on the six. A two-cluster mean with unequal priors (12 points, the enumeration of the 4096 assignments gives the mass of
  the labelling with the first mean below the second): 0.559, 0.186 and 0.980 exact, and 1.0 (one mode) before, 0.559, 0.186 and 0.980 now, the
  evidence within 1e-4 nats, the labelled means within 0.02 sd and the sds within 1% of the exact ones. The benchmark mixture with symmetric priors (3 data sets of 40
  points): two modes of equal weight, the sorted means within 0.05 sd and the sds within 4% of the grid (-1.889 +- 0.308 against -1.887 +- 0.313 on the first), the
  evidence of one mode within 0.04 nats of the grid's. Overlapping clusters (ten points between -2 and 2) stay one fixed point (the evidence -19.07 against
  the exact -18.75): the weights are only as good as the fixed points found.
- **The changepoint program** (8 seeds, 12 starts, a Gibbs reference, the truth inside the 90% interval, spreads over the sampler's): the
  `sort` read-out is unchanged to the digit (0.917, 0.72), `keep` finds more labellings (0.958, spread 2.7 -> 3.2 times the sampler's), `permute`
  0.958 and 3.6, times 60, 75 and 57 s for the eight seeds.
- **Labels default, for the owner** (`keep`, `permute`, `sort`; docs/probprog/ep.md section 4 has the table and the recommendation): `keep` gives
  the exact marginals of the model as written; `sort` is the read-out of a data-ordered sampler; `permute` is the one that finds a labelling
  of 2% weight reliably (12 random starts missed it). The default is not changed.
- Not done: continuous symmetries (the sign of a loading: no discrete message to start from), a regime visited by a handful of steps, the
  circular mixture's evidence, the forward-algorithm site (all in the gap's plan).
- Digest: no item runs the multi-start (`ep-digest` items are the route's atoms, models and programs), so none moves.

## 2026-11-12 Stochastic volatility with learned parameters: the chain of a scan as one factor, `gaussian-chain` (`agent-ad2a59-svchain`)

Gap `sv-parameters` rewritten in `ep-gaps.ts` (still open, for the single run with the persistence random). The two defects the walk and warm
start left were both in the message passing of the chain, and neither was the product factor or the `power` knob.
- Diagnosis of the level's spread (0.50 against the exact 0.27 at a persistence of 0.69): not the `product` factor (the persistence is known
  in a grid node) and not the precision (with the precision known as well the level is still 0.455 against 0.237). A linear Gaussian copy of
  the model (a Gaussian reading, the exact joint by a dense inverse) gives the same: 0.549 against 0.288, with the means exact. It is the
  loop of scalar messages through the level, which every step reads: the cavities of the previous state and of the level are taken as
  independent, they are positively correlated through the level itself, so each step tells the level less than it knows. A dense Gaussian
  vector (the states and the level, a rank-one `dot-constant` per step) gives the exact 0.251 but costs 20 s a node (a variable of 180
  edges and 3700 numbers a message: the sums of its messages dominate the engine).
- Diagnosis of the cycling at a low persistence (precision 0.002 or 0.14 against the exact 10.4 at a persistence of 0.1, seed 1): the
  precision is shared by sixty factors that each send it a gamma message from a cavity that does not hold the whole chain, and the messages
  do not settle (also with the dense vector). One factor that integrates the chain against all the cavities, with the precision on a grid in
  its log, has no such messages.
- New operator `gaussian-chain` (`operators/gaussian/chain.ts`, 67 operators): `x_t ~ N(a x_(t-1) + c . u + d, 1 / (scale tau + offset))`
  for the states `x_0 .. x_(n-1)`, drivers `u` and a gamma `tau`, ports `[tau, states, drivers]`. For one precision the chain with the Gaussian
  cavities is a Gaussian whose marginals, log integral included, come from one pass along the chain (a tridiagonal elimination with the drivers
  as a border: linear in `n`, the same numbers as the dense inverse to 1e-15); the precision is integrated by `tiltGrid` over `log tau` against
  its gamma cavity (reused, with `gammaLogWindow` and `centredMoments`), the tilted mean and variance of each variable are the mixture over the
  grid, `E log tau` and `E tau` the precision's, and the ports of one sweep's update share the pass through the engine's memo (2 ms for 60
  states). It is the `gaussian-mv-precision-scale` of a vector, for a chain. No variational rule.
- The compiler (`factor-graph/_chains.ts`, only with `packScalars`, only when the registry has the operator): the steps of an unrolled scan,
  `h[t]` with a known nonzero coefficient on `h[t-1]`, Gaussian drivers (the level) with known coefficients and one gamma precision, are
  collected as the walk maps them and written, after the nodes, as one factor placed where the first step would have been (so the readings meet
  proper cavities); the states keep their variables and layouts. A step that does not continue the chain ends it. Without `packScalars` the
  scalar steps are as before; the digest's `model/stochastic-volatility`, `model/ar1-state-space` and `program/kalman` (known scales) are
  unchanged to the last bit.
- A return near zero is the trap: the log-variance likelihood of a return of 0.004 is an exponential in the state with no curvature, so its
  cavity has almost no precision and a mean at -4000, and a factor that integrates the precision over its whole range lets it run to zero
  (the first run of the chain: the precision at 3e-5). The factor takes a `ridge`: a weak prior of precision `ridge exp(-j / ridge)` on a state
  whose cavity has precision `j`, centred at the state's mean under the chain at the mean of the precision's cavity, so it vanishes for a
  state with a cavity of its own and holds no mean at a settled one. The compiler sets it at 0.05 of the first state's prior precision. A
  ridge of 0.01 still cycled on seed 5 (fifteen returns under 0.1 in sixty); a plain ridge on every state (0.05) moved the level by 0.7 sd.
- At a fixed persistence (seed 1, point-mass filter, level / precision): 0.69: exact -0.886 +- 0.251 and 9.90 +- 5.34, scalar steps -0.883 +-
  0.468 and 10.13 +- 5.36, the chain -0.866 +- 0.254 and 10.28 +- 5.21 (20 sweeps; the evidence differences between the persistences agree with
  the exact ones to 0.07 nats); 0.1: exact -0.800 +- 0.199 and 10.39 +- 5.69, scalar precision 0.14 +- 0.11 (not settled), the chain
  -0.798 +- 0.196 and 10.69 +- 5.61; 0.9: exact -0.963 +- 0.433 and 12.05 +- 5.70, the chain -0.966 +- 0.425 and 12.34 +- 5.60.
- The grid (`epGridMethod(bound, { over: ['phi'] })`, walk, warm, window 10, damping 0 and 60 sweeps instead of 0.2 and 120: the nodes settle in
  14 to 20 sweeps now), 30 seeds of 60 returns, against the brute-force posterior (26 by 30 by 24, scratch script), before (previous agent) and
  after: z of the mean in reference sds, rms (max): level 0.08 (0.35) -> 0.05 (0.18), persistence 0.26 (1.1) -> 0.05 (0.07), precision 0.10 ->
  0.02 (0.04); sd over the reference's, median (range): level 1.6 -> 0.96 (0.88 to 1.07), persistence 0.85 (0.2 on seed 27) -> 1.03 (1.01 to
  1.05), precision 1.0 -> 0.98 (0.95 to 1.00); 90% coverage of the truth: level 0.97 -> 0.80 (the interval was too wide before and is the
  reference's now; 0.80 is what the reference posterior itself covers, to the means within 0.2 sd), persistence 0.97 -> 1.00, precision
  1.00 -> 1.00; the fit 7 s -> 5 s on a busy machine (2.8 s the least), 15 or 16 nodes of which every one settles but the one beyond a
  persistence of one (a chain that explodes has no evidence). The seed that missed the persistence (27) is covered.
- Tests: `operators/gaussian/chain.test.ts` (the pass against the dense joint with drivers, an intercept and a flat cavity; the tilted moments and the
  evidence against a fine grid over the precision; the ridge; the messages and the cache; a chain of two states against `gaussian-gamma-precision`),
  `factor-graph/chains.test.ts` (the compilation, and EP against the point-mass filter at 0.69, 0.1 and 0.9, and the scalar steps' twice-as-wide
  level), `ep-grid-sv.test.ts` (every spread within a tenth of the brute-force one, every node settled). Digest: the registry line (the new
  operator; the grid knobs `order`, `warm`, `window` of the previous change were not in it either), no item moves.
- Not done: the single run with the persistence random (the product factor; the persistence as a port of the chain), a likelihood for a return
  near zero that is not improper (the ridge is a heuristic), a chain with a fixed scale and a random level (the scalar steps are used; the same
  loop), a dense vector with a cheaper belief sum in the engine.


## 2026-11-12 A strong correlation from few pairs, and the group form of the sd-and-correlation site (`agent-a6b87c-corr`, gaps `correlation-strong-few-pairs` and `matrix-larger` narrowed)

Searched first: the correlation family and its mapping (`d.correlation`, `latentCorrelation`, the readback of the beta as a transformed beta), `mvn-sd-corr` and its beta projection, the
numeric prior site and the likelihood sites (for a Gaussian belief on the Fisher transform), the grid (`_grid.ts`: `betaAxis`, `gaussianAxis`), `mvn-wishart-group` and its closed-form evaluator
(reused, moved to `_group.ts`), `posterior3`. Nothing is a second family system: the beliefs are the library's beta and Gaussian.
- `mvn-sd-corr` (`operators/covariance/correlation.ts`, shared parts in `_sdcorr.ts`): the message to the correlation is the beta with the tilted mean and variance of `u = (rho + 1) / 2`
  (`match: 'moments'`, new default) instead of the one with its log moments (`match: 'log-moments'` keeps the KL projection; a point mass falls back to it). With one factor on the
  correlation (a plated likelihood and its beta prior) the cavity is the prior itself and the belief has the exact mean and spread.
  Measured against the exact grid posterior of the correlation (known sds, mean zero, 4 seeds, 2 to 10 pairs; sd ratio / mean z; before, then after): rho 0.3 and 0.5: 0.91 to 0.99 / 0.005 to 0.017
  -> 1.000 / 0.000; rho 0.7: 0.89 to 0.98 / 0.006 to 0.019 -> 1.000 / 0.000; rho 0.9: 0.81 to 0.95 / 0.009 to 0.022 -> 1.000 / 0.000; rho 0.99: 0.54 to 0.92 (a seed at 0.45) / 0.003 to 0.033 -> 0.999 to
  1.000 / at most 0.002; eta 3, 3 to 10 pairs: 0.65 to 1.006 -> 1.000. The evidence is the exact one before and after (within 0.0012 nats; a single factor), two sweeps. The default atom (eight pairs, rho
  about 0.8): sd ratio 0.829 -> 1.000, z 0.022 -> 0.000, evidence unchanged. Not better: the curve. The Kolmogorov distance of the belief's own density from the exact one (oracle fit, 3 seeds) is 0.01 to 0.07 at
  rho 0.7 and below as for the KL beta, 0.05 to 0.14 at 0.9 (KL: 0.05 to 0.09), and 0.14 to 0.4 at 0.99 where the KL beta had 0.12 to 0.15; the 5% and 95% points are 0.1 to 0.2 exact sds off at rho 0.9 and
  above (0.4 at 0.99 from five pairs). The harness KS compares a normal with the belief's moments, so it cannot see it (0.22 at the default, 0.15 to 0.33 at rho 0.9 to 0.99, unchanged by the fix).
- Tried and dropped: a Gaussian on the Fisher transform (`s = logit u = 2 atanh rho`) as the correlation's belief, with a numeric `prior-site` base `logit-beta` for the LKJ prior, matched to the
  KL moments of `s` and to the moments of the correlation. Its curve is the best of the three (distance 0.04 to 0.19), but with the KL moments the spread is no better than the beta's (it is
  the exact spread of `s`, which `tanh` then compresses: 0.56 to 0.99), and with the moments of the correlation the two-site fixed point (prior site and likelihood site, each a Gaussian
  approximation) is off by 1% to 8% in spread, 0.005 to 0.08 nats in evidence, and needs 10 to 60 sweeps at rho 0.99 (some seeds do not settle). Kept in the gap's plan.
- The reference of `bivariate-correlation-learned` (`validation/references/grid3.ts`, `ep/_atoms/covariance.ts`): the atom's grid was a quarter short on the spread of a correlation of 0.95 from eight
  pairs (0.0330 against 0.0429 by an independent grid in the logs and the Fisher transform, stable to five figures from 100 to 200 nodes; EP 0.0417) and its mean mixture 3% to 5% off (the
  heaviest 6000 nodes). `posterior3` has `warp` (`log`, `correlation`) per coordinate and `mixtureNodes`; the learned bivariate atom runs the precisions in their logs and the correlation in
  `2 atanh(rho)` with 60000 mixture nodes (the reference costs 2 s instead of 0.6 s, lazily). Now EP against the reference: rho sd ratio 0.84 to 0.92 -> 0.997 to 1.002 at rho 0.9 from 6 to 10 pairs and 0.84 -> 0.972
  at 0.95 from eight pairs (where EP's own three-dimensional grid is 3% short), the mean's spread 0.98 to 1.007 -> 0.999 to 1.02 (against the wrong reference before). No other atom's digest line moves (identical to the bit).
- New operator `mvn-sd-corr-group` (`operators/covariance/correlation-group.ts`; the evaluator and the tilt key of the Wishart group moved to `_group.ts`, `mvn-wishart-group` unchanged): `x_j ~ N(mean,
  Sigma)` for `J` bivariate vectors sharing a Gaussian mean and the covariance of two gamma precisions and a beta correlation, as ONE factor. Given the three parameters the vectors and the mean integrate
  out in closed form; the grid over the parameters is the one of `mvn-sd-corr`; each vector and the mean are matched to the mixture of their conditionals over the nodes; one tilt per sweep serves all ports (the
  cache key as the Wishart group's). A known mean is one `mvn-sd-corr` per vector. EP only. The compiler maps a latent `bivariateNormal` over `dims: [group, coef]` (`_mappings/matrix.ts`
  `latentBivariateVectors`; the entry `bivariate-normal-refused` of the table is now `bivariate-normal-latent`, the refusal that names the form is for an axis that is not two long or missing), the
  compiler page is regenerated. Four groups of four readings (`bivariate-group.test.ts`): 20 sweeps, 0.9 s, against a grid over the parameters with the vectors and the mean integrated out densely
  (nothing of the factor's algebra): the correlation's spread 0.999 of the exact, the mean's and a vector's 1.000, means within 0.005 sd (the test allows 0.06), each precision's spread 0.98 to 0.99
  (a gamma matched to its log moments). The operator: `J = 1` is `mvn-sd-corr` to 1e-9 (messages and evidence), two and three vectors against a dense joint-Gaussian grid
  (messages to the parameters, the vectors and the mean to 1% and 0.01, the evidence to 0.01 nats).
- Not done: an LKJ prior of order 3 and above and vectors of more than two coordinates with their own sds and correlation (as `matrix-larger` says), a normal-Wishart variable inside a larger model, an
  accuracy row (no operator of the covariance folder has one), an atom for the group form (the compiler test is the check), a VMP rule (the expected log of a beta correlation has no closed form).
- EP digest (`--only` items): `atom/bivariate-correlation` and `atom/bivariate-correlation-learned` moved toward the reference and were merged by line (the `vmp/` lines and every other
  atom are identical to the bit, the registry line gained the operator): rho mean 0.794981 -> 0.798515, sd 0.132244 -> 0.159559, sd ratio 0.828826 -> 1.000022, z 0.021859 -> 0.000294, KS 0.2135 ->
  0.2220 (basic); rho mean 0.373943 -> 0.374300, sd 0.163614 -> 0.165082, sd ratio 1.000818 -> 1.000037 (learned, against the corrected reference). The registry and params tests counted 67 operators on a main that had 68; they count
  71 now (with `gaussian-chain` and this operator).
## 2026-11-12 The evidence of cavities with no density: plates, affine factors, gates, and the alpha-divergence energy (`agent-ab33f2-evidence`, gap `evidence` narrowed)

Gap `evidence` rewritten in `ep-gaps.ts` (still open, now small). One evidence system: `factorEvidence` / `evidenceOf` (`engine/evidence.ts`) and the operators' `logEvidence`; no second one.
- **Measured first.** The learned logistic regression as one plated factor had no EP energy of its own: its plate's cavity (the weights' scale mixture sends a site with
  a negative precision) is indefinite at the fixed point, the engine did not even call the plate there (`readsTarget` blocks a target cavity with no density), so the
  run "converged" with one update that could not be computed (35 sweeps, one damped restart, `unresolved: 1`) and the evidence came from the rows-expanded model
  (-26.5201 against the grid's -26.4738). Eleven digest items had `evidenceUndefinedAt`; by operator: `linear`, `plus`, `times-constant` (the affine ones), the
  gate's inner rows, `product`, `apply`, `sum` and `gaussian-gamma-precision`. The circular mixture's "row gates" are not gates in the compiled graph: they are
  `select` + `linear` (the copy of a period) + `gaussian-gamma-precision` rows.
- **The plate** (`operators/lifts/plates.ts`, `engine/dispatch.ts`, `core/types.ts`): a new trait `improperTarget` says the operator reads its target's cavity only
  through a posterior it makes with its own parts, so a cavity that has no density but is not flat is shown to it (a flat one still blocks). The plate sets it; its inner
  propagation starts, for such a cavity, from the least uniform positive precision on every row that makes the posterior proper (doubling), and the energy no longer asks the
  cavity to be proper (the energy was already written in natural parameters: the posterior's partition plus the rows' terms). A proper cavity starts from no sites as before.
  Against the references: the plated `logistic-learned` now converges in 8 sweeps (was 35 with a restart), its evidence is -26.520135736411973 against the rows-expanded
  model's -26.520135736412023 (5e-14; error to the grid 0.046 as before), `evidenceMethod` is `ep-energy` (not `ep-energy (rows expanded)`), `probit-learned` is unchanged
  to 1e-13. The beliefs of `logistic-learned` moved to the true fixed point (the rows-expanded model's): means 0.005, 0.026, 0.019 sd for the weights, spreads 1.7% to 3.1% narrower, tau mean 0.7133 -> 0.7063.
  This is not toward the grid marginals (`sdRatio` 0.98, 1.01, 0.99, 0.98 -> 0.96, 0.99, 0.96, 0.95; ks 0.006 to 0.027 -> 0.010 to 0.037), and the atom still passes.
  The evidence moved by 6e-13.
- **The affine operators and the Gaussian joint** (`operators/gaussian/linear.ts`): `linear`, `plus`, `difference`, `times-constant`, `sum` and `gaussian-joint` take the integral
  in natural parameters when the cavities are not all proper (the output substituted, or one input eliminated with the Jacobian `1 / |c|` when the output is observed): finite
  whenever the joint precision is positive definite, NaN when it is not. Proper cavities take the old path (bits identical). Tests (`operators/evidence.test.ts`)
  against two-dimensional and one-dimensional integrals with an indefinite cavity: plus (output precision -0.5), linear with an offset and an indefinite input, difference
  and times-constant with an observed output, a joint that is not positive definite (NaN), and the proper case unchanged. Model-level: `banana` has an evidence (0, the exact one of a model without data), `radon` -61.6636 (`plus`), the mixture regression
  -14.0258 (grid -13.6441, 0.38 nats low; its run has 12 updates that could not be computed, so it is not a fixed point), `sparse-student-t` -98.8674; the circular mixture
  went from 229 to 23 factors without an evidence in the data form and 139 to 7 in the means form.
- **The gate** (`operators/discrete/gate.ts`): with a cavity that has no density but is not flat the evidence is the log sum over the branches of the mass of the branch against its
  variables' cavities (its own rule, in natural parameters) times the mass of the cavities it does not read, which must be proper (a variable every branch reads needs none). Test against a
  one-dimensional integral (both branches read an indefinite cavity) and an `ImproperMessageError` when a branch does not read it.
- **The alpha runs** (`engine/evidence.ts`, `engine/_cavities.ts`): the energy of an alpha-divergence run is the power EP energy at power alpha for the factors that take a power
  (the fixed points of the alpha updates are those of power EP with those factors at alpha: tilted `f^a cavity q^(1 - a)`). `alphaEpAlgorithm` now carries `mp/evidence`. On the clutter
  problem (one variable, eight robust rows, quadrature -18.3231): alpha 1, 0.8, 0.5, 0.3, 0.1 give errors +0.0017, +0.0006, -0.0009, -0.0018, -0.0028 nats, and equal a power EP
  run's energy at the same powers to 3e-14 (`alpha.test.ts`, the contract test no longer says "no evidence below alpha 1").
- **Not done.** The `gaussian-gamma-precision` rows (23, 7 and 3 of 50 in the circular forms) and the `product` / `apply` factors of the funnels and the Bayesian neural
  network still have none: against an indefinite cavity their integral truly diverges or is numeric. The circular mixture cannot be multi-started yet. The tied-group fallback in `mp.ts` stays.
- **Digest** (merged lines only, `make ep-digest ARGS="--only ..."`): `atom/logistic-learned`, `atom/mixture-regression` (evidence null -> -14.025781793658), `model/banana`
  (null -> 1.6e-15), `model/radon` (null -> -61.663554203010), `model/funnel-centred` and `model/funnel-non-centred` (undefined factors 10 -> 9), `program/sparse-student-t` (null ->
  -98.867411158437), and the registry line (the `improperTarget` trait of the two plates). 56 items compared in all: every other item (the atoms, models and programs with
  plates, gates, affine factors, `linear-regression-row`) has the same numbers, but `model/bayesian-neural-network`, `program/circular-data`, `-means` (beliefs, damped and skipped counts) and
  four program evidences (3e-11 to 4e-12: `circular-rows`, `cox`, `sparse-gaussian`, `sparse-mixture`) already differed from the baseline on the commit this work started from
  (checked in a second worktree), so their lines are left; their `evidenceUndefinedAt` lists changed here as said above.

## 2026-11-12 The Bayesian neural network is refused by EP, with the reason, instead of run to a state that is not a posterior (`agent-ab3358-bnn`, gap `network-layer-modes`)

No sound fix was found in the effort, so the benchmark's recorded state is made honest: `model/bayesian-neural-network` was an EP run of 200 sweeps
that never converged (digest on this commit's parent: change 347, 8369 improper updates of which 8354 skipped, 71 factors with an improper cavity,
no log evidence, function values -0.55, -1.07, -1.59, -2.41, -3.83 with sds 0.06 to 0.12 against the long NUTS run's -0.21, -0.90, 0.30, 0.83, 0.14
with sds 0.45, 0.066, 0.26, 0.084, 0.38; 3 s a run) and was listed as `runs (ep)`.
- Diagnosis. The weights of a layer are not identified (sign flip of a unit, permutation of units), so the posterior is multimodal and one Gaussian
  per weight has only two kinds of fixed point: the symmetric one (all weight means 0, which is a converged and uninformative state: the function
  value 0 +- 1 at damping 0.8 and 0.9, 0 +- 0.04 with a power below one) and ones with dead units. From the symmetric start the first sweeps keep
  the three units identical to two decimals (weights 0, the spreads shrinking together) and the first dropped updates (from the fifth sweep) are where
  they part; after that one unit lives (input weight near 2, bias 0.4, output weight 0.85) and the other two collapse (input weight 0, output weight with a
  spread of 1). Besides that, about half of the messages of the `tanh` factors to their inputs have no positive precision at the damped end state,
  so the affine factor beside such a unit has no cavity: 32 of the 180 messages of the `linear` factors were not computable.
- Tried, one at a time and combined, each 200 to 600 sweeps (3 to 20 s) on the canonical data (function values at -3, -1.25, 0, 1.25, 3; the
  reference -0.21 +- 0.45, -0.90 +- 0.066, 0.30 +- 0.26, 0.83 +- 0.084, 0.14 +- 0.38): damping 0.3 (not converged: -0.98, -0.89, 0.19, 0.82, 0.84, sds
  0.45, 0.30, 0.32, 0.31, 0.46), 0.5 (-0.91, -0.87, 0.16, 0.80, 0.81, change 3e-3 after 200 sweeps and 1.7e-6 after 400 with `reset`, but 32 updates
  never computable: sds 0.44, 0.36, 0.37, 0.36, 0.44), 0.8 and 0.9 (converged at the symmetric fixed point, no information), the `reset` policy,
  flooding and tree schedules (tree at 0.7: 6.5e-8 after 400 sweeps, 25587 improper updates), power 0.1 to 0.5 on the `apply` factors (degenerate: the
  function 0 +- 0.04, or not settling), a start that breaks the symmetry (messages on the weights' ports of the first row, damped so that they
  survive: one unit lives, not settling, change 0.1 to 0.7), and an affine rule that accepts a cavity of negative precision (a scratch edit of
  `affineMessage`, reverted: damping 0.5 then has no unresolved update and 21 improper events, but the change stays at 0.8; damping 0.7 at 0.14
  after 300 sweeps). Within the data the function values of the best of these agree with the long run (-0.88 to -0.93 at -1.25, 0.80 to 0.87 at 1.25,
  0.16 to 0.28 at 0), the spreads do not (0.2 to 0.45 against 0.066 and 0.084 beside the data), and the far inputs are 1.6 to 1.9 spreads off. One
  hidden unit alone settles (undamped, change 1e-8, 8 updates not computable); two do not.
- The code: `networkLayerUnits` (`model/factor-graph/verdict.ts`: the `apply` factors whose input is an affine map of a random weight that other rows
  read and whose output is multiplied by a random weight that other rows read) and the EP verdict of `factorGraphVerdict` refuses a graph that has
  any, with the reason and the missing feature `beliefs that hold the symmetric modes of a hidden layer`; `epRoute` is then `none` for the benchmark
  network (the grid route does not apply either), the coverage row is `planned`, and gap `network-layer-modes` (new, quality, large) claims it
  through its marker `neural-network layer`; the direction `bnn` is rewritten. Only the benchmark network has units: the funnels, the schools, the
  radon model, the banana, the negative binomial regression, a `tanh` regression of two weights and a nonlinear state space have none, and the
  variational rule is not asked. Tests: `factor-graph/network-layers.test.ts`, `ep-coverage.test.ts`, `ep-gaps.test.ts`.
- What a sound fix needs (in the gap's plan): a fixed point per mode weighted by evidence (the continuous multi-start named under `multimodal`) with a
  run per start that settles, and for that the forward pass of a row as one factor with the weights as the only variables (probabilistic
  backpropagation), so that no hidden variable carries a negative-precision message; the refusal goes when that is checked against the long NUTS run.
- Digest: `model/bayesian-neural-network` moves deliberately (route `ep` to `none`, the verdict, no posterior); no other item is affected (the
  verdict changes only for graphs with a layer); the registry is untouched.

## 2026-11-12 Several hyperparameters on the grid: a walk over the lattice, ARD length scales, a noise sd (`agent-a6eb2d-gridhyp`, gap `grid-hyperparameters` narrowed)

No second grid method: `epGridMethod` (`inference/ep-grid.ts`) gained what the plan asked for, and the gap's summary and plan were rewritten
(`ep-gaps.ts`; still open: five or more axes, hierarchical or non-identical priors, a likelihood that is not Gaussian at several axes).
- `order: 'walk'` over any number of axes: from the middle of the lattice, always the unvisited lattice neighbour of the heaviest node that is
  within `window` nats of the best (a best-first flood fill; with one hyperparameter it is the interval walk of the stochastic volatility work,
  node for node), so the cost follows the volume of the posterior and not the product of the axes. The range still grows where an edge has
  mass (`extend` takes only the wider range in a walk). A walk that the budget cut short while its region has an open edge reports status
  `stopped` and `diagnostics.budgetReached` (a finished-looking `converged` on 4000 nodes was wrong).
- Defaults by number of axes (`epGridKnobsFor(axes)`; the method's own `config`): one axis as before (bisect, 31 points, window 10, 400 nodes,
  no warm start), several axes `walk`, `warm`, 15 points, window 7, 4000 nodes. `runEp` gives a grid run as many steps as its node budget.
- The hyperparameters: the scalar latents a custom function reads (as before) and, only when the model has no EP graph without them, the plated
  vectors it reads, one axis per component (`ell[0]`, `ell[1]`: an ARD kernel with one prior for all its inputs); the order is the model's.
  A vector the function is affine in (the whitened `z` of the non-centred Gaussian process) stays a random vector for EP. A noise sd joins
  the grid by being read by a custom function too (`m.custom('noiseSd', ([s]) => s.value, [noise], {})`; with it inside the EP run instead,
  one axis fewer, a node costs 20 times more, 57 ms, and the evidence is 0.05 nats off and the noise's spread 0.91 of the exact).
- Against a brute-force tensor grid of `N(y; 0, K + sigma^2 I)` in plain arithmetic (`gpPosteriorGivenHyperparameters`, the new reference,
  with `gridMarginals` extended by `logEvidence`, `expect`, `mass` and `mixture`): the atom `gp-hyperparameters` (length scale, variance and
  noise sd on eight points; 948 nodes, 1 to 2 s) is 3e-4 nats from the grid in evidence with means within 0.004 reference sds and spreads
  0.992 to 1.000; an ARD kernel of two inputs with its variance (three axes, 1163 nodes, 1.7 s) 1e-4 nats, means within 0.006 sds, spreads
  0.996, 1.017, 0.995; with a noise sd (four axes, 11 points, window 7, about 2500 nodes, 5 s; measured against a 25-point-per-axis reference in
  a scratch script, and in `ep-grid-hyper.test.ts` against a 21-point one with wider tolerances) 4e-4 nats, means within 0.03 sds, spreads
  0.99 to 1.02. At 15 points and window 7 the three-scalar case is 800 nodes and 0.9 s; the lattice, not the window, sets the error (9
  points: 0.03 nats; 11: 0.005; 13: 0.002; 15: 3e-4). Warm starts do not help where each node is exact (cold was faster: the messages are
  part of the state the session serialises every step, which is quadratic in the nodes).
- New: atom `gp-hyperparameters` (family `kernels`), `epGridKnobsFor`, `gpPosteriorGivenHyperparameters` (`models/gaussian-process.ts`), the
  tests of `inference/ep-grid-hyper.test.ts`. EP digest: two items added (`atom/gp-hyperparameters`, `vmp/gp-hyperparameters`) and the
  registry line (one atom's knobs); the other grid items (`atom/gp-length-scale`, `atom/gp-probit-learned`, `model/gaussian-process-*`,
  `model/stochastic-volatility`) are unchanged at the exact level (`atom/gp-length-scale` differs in the last digits on the unchanged HEAD too).
- Not done: five or more axes (the nodes grow with the volume; a sparse design or a Laplace step is the plan), diagonal neighbours (a ridge
  thin against the lattice's spacing and diagonal to it can be cut; the window of 7 nats makes it unlikely for the Gaussian processes tried),
  hierarchical priors on the grid, a reference at several axes with a non-Gaussian likelihood (each node's evidence is then approximate).

## 2026-11-12 ARD under variational message passing opens damped (`agent-a6eb2d-gridhyp`)

The ARD Bayesian GAM under VMP stalled without damping and the offer listed `vmp` regardless. VMP on this form does run once damped, so the
fix is a default, on the engine side only (`models/bayesian-gam.ts`; the pocs page is untouched and reads the offer):
`bayesianGamFormStarting(prior)` gives the starting knobs a form needs (`ard`: `vmp` at `damping: 0.5`, `maxSweeps: 60`), merged over
`bayesianGamStarting` in `bayesianGamMethods` (so the union's defaults, which a page opens its controls on) and applied to the method's own
config in `bayesianGamMethod` (so a session with no config runs damped too).
- Measured on 60 rows (scratch script, EP as the reference, the same data as the tests): undamped k = 6 stops at sweep 74 with the log noise
  at -46 (collapsed) and k = 10 ends at +0.90 with a spread of 2e-8 (curve 0.42 sd and band 75 times EP's); damping 0.3, 0.4, 0.5 and 0.7
  all reach the same point (k = 6: log noise -1.585, curve 0.083 response sd from EP's, band 1.22 times; k = 10 at 0.5: -1.597, 0.029, 1.39),
  0.1 and 0.2 settle elsewhere (band 1.7 to 2.0 times). The change in the messages is far above any tolerance at 200 sweeps (tol 1e-3 to 1e-6
  stop nowhere; 627 sweeps to 1e-5 at 0.3, as measured earlier), while the log noise is within 0.015 of its 200-sweep value and the curve's distance to EP's within 0.02 sds of it from 30 sweeps, so
  the cap opens at 60 (5.6 s at k = 6, 11 s at k = 10, against 18 and 36 s at 200). The run ends at the cap, status `completed`, never
  `converged`: honest, and the offer keeps `vmp` (it runs, with a band 1.2 to 1.4 times EP's, the mean field of the precisions and the
  coefficients).
- Not made `unavailable`: the verdict of the method is about the model (it compiles and has rules), and now the run is sound by default.
  Other forms keep the method's defaults (`penalty` VMP runs at damping 0.3 in its test, 0 by default, unchanged).
- Tests (`bayesian-gam-ep.test.ts`): the union's defaults for ARD (0.5, 60) and for the penalty form (0, 200); a session with no config on ARD
  stops within 60 sweeps with the curve within 0.1 sd of EP's and the log noise within 0.1, and the undamped run's log noise below -10.

## 2026-11-12 EP and the Gaussian mixture: the symmetric fixed point, and the damped warm-up that breaks it (`agent-a83869-gmm`)

The owner's report: on the pocs Gaussian mixture page (150 rows of two clusters 3 apart, equal weights, ordered means, a shared spread, the
page's `ep` at damping 0) the fitted components were two nearly identical wide Gaussians near 0 (means -0.15 +- 0.24 and 0.32 +- 0.24,
weights 0.51 / 0.49, spread 1.83 +- 0.11 in the data's units; the truth is -1.5, 1.5, 1.0), and `ep-multistart` gave the same. Both ran on the
default knobs of `mixtureExample`; the compiled graph is a `select` per row (`y_i = mu[z_i]`), a `gaussian-scale` per row (the shared log
spread), a `discrete` per row on the Dirichlet weights, and `difference` plus `greater-than` for the ordering; nothing in the graph pulls the
means together.
- Diagnosis. (1) The graph has a symmetric fixed point (both components the same, the labels uninformative: a clean fixed point with a finite
  evidence, -221.80) that undamped EP from flat messages falls into on data seeds 1 and 4 of 8 and on free means 2 of 3. The first undamped
  sweep commits the rows and the shared spread overshoots (0.20 after the second sweep in the standardised units, where the clusters' is
  0.55), then climbs to 1.0 over the next sweeps and the run converges to the blob in 55 to 60 sweeps. (2) The clusters' reading is a fixed
  point of the damped dynamics and stable undamped (started from the damped state at damping 0 it stays), but not a clean one: about a third
  of the rows (the ones between the clusters) get from `select` a message to the value of the row with a negative precision, so the cavity of
  their `gaussian-scale` has no density, the update cannot be computed and is dropped by the improper policy (50 a sweep, forever), and the
  evidence is undefined there. (3) Damping 0.5 throughout, or three damped sweeps and then the run's own damping, reaches the clusters
  in every one of the 14 data sets tried (8 seeds, free means on 3, K = 3 on 3), 1.5 s a run (47 sweeps; 127 at damping 0.5 throughout).
  (4) Why `ep-multistart` was no better: its random and exclusion starts set the labels' messages at random or at one category, and random
  labels average out to uninformative ones, so every such start falls into the blob (the held start of `warmed` in the multi-start
  does too once released undamped); the starts that did reach the clusters (random 4 and 5 in a 6-start run) had no evidence, and the blob
  had one, so the weighing by evidence dropped them. The symmetric reading is thus the only one with a weight.
- Fix. `warmup` (engine setting `EngineSettings.warmup`, option of `epAlgorithm`, knob of `ep` and `ep-multistart`: `auto`, `on`, `off`;
  `auto` is the default): the first three sweeps of a run (and of each restart) are damped by at least a half, then the run goes on at its
  own damping. `auto` applies it where the graph has a `select` with a latent index (`hasExchangeableLabels`), and `epAlgorithm` called
  directly without the option is what it was. `ep-multistart` does not weigh the readings when its first run (the plain start, which now
  reaches the clusters) has no evidence (`diagnostics.unweighed`; `siteMixturePosterior`'s `logEvidence` is optional): the posterior is the
  converged readings that have none, equally. `epSpecMethod`'s fallback evidence by an expanded graph is skipped when the expanded graph is the
  same graph (it re-ran EP from flat messages and attached the blob's evidence, -221.80, to the clusters' state).
- Before and after, `ep` at damping 0, tol 1e-6 (means and spread in the data's units, sd after the +-; time of one run including the
  restart at damping 0.5 that a run with dropped updates gets): default (seed 1) -0.15 +- 0.24, 0.32 +- 0.24, spread 1.83, 60 sweeps, 2.6 s ->
  -1.46 +- 0.14, 1.52 +- 0.13, spread 0.97 +- 0.07, weights 0.48 / 0.52, 177 sweeps (47 + the restart), 6 to 8 s; seed 4 -0.20, 0.26, 1.83 ->
  -1.34 +- 0.13, 1.56 +- 0.15, 0.98 (9 s); free means -0.08, 0.26, 1.83 (0.6 s) -> -1.61 +- 0.12, 1.69 +- 0.12, 0.89 (3 s); K = 3, separation 4, unequal
  weights, equal weights and Old Faithful were already right and are identical to the digit (-4.12, 0.04, 4.21, spread 0.90; -4.15, -0.22, 4.21, 0.87;
  54.74 +- 0.58, 80.20 +- 0.44, 5.30 +- 0.26). `ep-multistart` (6 starts): default -0.16, 0.30, 1.83 (18 s) -> -1.48 +- 0.14, 1.51 +- 0.13, 0.97 (20 s);
  free -0.10, 0.24, 1.83 -> -1.53, 1.59, 0.94; the K = 3 and Old Faithful rows identical. One component: EP refuses it (a Dirichlet over one
  component has no message family), as before; NUTS runs. Against NUTS (4 chains, 300 + 600 draws; the example's reference for the ordered model):
  means -1.40 +- 0.34, 1.42 +- 0.31, spread 1.16 +- 0.20 (default; Gibbs 4000 draws: -1.37 +- 0.37, 1.41 +- 0.35, 1.19 +- 0.22), free -1.43, 1.46, 1.13; K = 3 unequal
  -3.97, 0.03, 4.07 and 1.13, equal -4.00, -0.22, 4.12 and 1.11; Old Faithful 54.60 +- 0.64, 80.09 +- 0.48, 5.93 +- 0.27. EP's means are within 0.1
  of the sampler's (0.15 for K = 3), its sds are 0.4 to 0.9 of the sampler's (the 150-row seed 1 has a posterior twice as wide as EP's 0.14: 0.14 against 0.33), and its spread is 0.8 to 0.9
  of the sampler's (0.97 against 1.16; 0.90 against 1.13): the dropped updates make the clusters' fixed point a little over-confident and low on the spread. That is what EP does
  here, not a defect of the start.
- Tried and left: the natural-parameter rule for `gaussian-scale` with a cavity of negative precision on the value of the row (the integral exists
  when the precisions add up to a positive one, and the evidence was then defined at the clusters), which made the dynamics diverge (the change 1e135 in
  200 sweeps, the spread 1.78 +- 0); its evidence at the dropped-update state was nonsense (+958) before that. So the clusters have no evidence and the readings of a mixture
  cannot be weighed (gap `multimodal`, plan updated).
- Tests: `message-passing/algorithms/warmup.test.ts` (the first sweeps are damped, the same fixed point, none without it, `auto` on a graph without labels),
  `inference/ep-mixture-warmup.test.ts` (the 150-row mixture of the page: the blob without the warm-up, the clusters with it, free means, the multi-start),
  `apps/pocs/src/examples/mixture-method.test.ts` (EP on the page's defaults recovers the means within 0.4 and the spread within 0.25, ordered and free).
  EP digest: four items move and are replaced by their new lines (`atom/gate-random`, `atom/mixture-components`, `atom/spike-and-slab`: three more
  sweeps from the warm-up (5 -> 8, 3 -> 6, 5 -> 8), the beliefs and evidence the same to tolerance; `program/changepoint`: 20 -> 21 sweeps, the evidence 2e-11 away), and the
  registry line (the `warmup` knob of `epMultiStart`); every other item of the mixture, clutter, gate, topic, lda, stick, changepoint, spike, tracker, zero, sparse and hmm
  families is unchanged (`program/sparse-gaussian` and `program/sparse-mixture` differ by 2e-12 and 8e-12 in the evidence on the unchanged base too).
- Not done: a data-informed start of the labels (the warm-up does what is needed on the data tried; a start from sorted values would need the
  observation values from the graph), the evidence at the clusters (above), the restart of a warmed run that stalls (it reruns from flat messages at damping 0.5 and ends at the
  same state: the run costs twice), `gate`-based mixtures (the trigger is a `select`; none of them failed).

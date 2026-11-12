# The numerics step of the rules review (S5c)

The deliberate, non-bit-preserving change of `lib/message-passing` that `review-rules.md` calls 5c (section 6.1, row S5): one window and
zoom policy for the gamma and beta tilts, centred variances, one floor policy that reports a collapse, no 24-point rule in
`negative-binomial-dispersion`, and one shifted-weights helper for the hand-rolled log-sum-exps. Judged, as the review says, by the
error against a high-order reference and not by sameness, under the owner's tolerance policy:

- no atom's posterior z-score or sd ratio against its reference moves by more than 0.01;
- no evidence moves by more than 1e-6 nats;
- the operators' brute-force tests stay green at their current tolerances (none was loosened);
- a change that breaks the policy is reverted or reworked (two were: see "Not done").

## What changed, by commit

| commit | change | bit-identical |
| --- | --- | --- |
| `shiftedSum` | `numerics/_moments.ts`: the largest log weight, the shifted weights, their total added in order, the log mass. About twenty hand-written loops use it (the one-dimensional tilt, the panelled grid, the Gauss-Legendre log grid, the numeric rule, the precision, gamma-scale, concentration and vector-scale grids, the Wishart rule, the rectifier, apply, apply-product, select, gate, the multiclass product rule, the mixture and step sites through the existing `logSumExpOf`, the clutter and region algorithms, the Dirichlet projection, the prior window, the variational quadrature). | yes (`ep-digest-check --bits`, 61 items) |
| accuracy harness | `numerics/accuracy.test.ts`: the table below, its bounds. | n/a |
| negative binomial | `numericOperator` takes an optional `column` form of the log density; with it the default rule is the panelled pair grid of `likelihood-pair` (graded panels, zoomed onto the mass), and what depends on the log dispersion alone is made once per node of it. The 24-point Gauss-Hermite rule is gone. | no |
| centred variances | `centredMoments` (two passes) in the pair grid, the precision tilt, the weighted product, mixture-site, the rectifier and apply-product mixtures, gaussian-scale, the log concentration, the log multiplier and the dense covariance of the scale operators, the variational deterministic function, and the zoom window of the product rule. | no |
| one floor policy | `resolvedVariance`: a variance from grid or mixture moments that is not positive and finite is reported as an `ImproperMessageError` (the engine counts and repairs it) instead of being floored to 1e-300, which handed the next update a point mass of precision 1e300. | no |
| one window and one zoomed grid | `windowProfiles.logTilt` (20 / 20 / 60) for the precision, gamma-scale, concentration and vector-scale tilts, which share `tiltGrid` (a uniform grid zoomed onto the mass). | no |
| shared window finder | `massWindow` of the prior projection becomes `findWindow` (bit for bit); `tiltedByQuadrature` finds the window of a hinted factor by scanning for its mass, and `WindowHint` (a centre, a scale and a reach written by each likelihood) becomes `TiltHint` (kinks, zoom, and for the Student-t alone a window in its own scale). | no |

## Error against a high-order reference, before and after

Each scenario of `numerics/accuracy.test.ts` integrates the exact tilted density of one operator with a composite Gauss-Legendre rule of
high order (600 panels of 12 points in one coordinate, 40 panels of 10 points a coordinate in two, on a window found by scanning) with the
integrand written out in the test, and compares the mean and sd of the target port's member of its family with the operator's message.
`z` is the error of the mean in standard deviations of the reference, `sd` the relative error of the sd; each cell is the worst of the
operator's scenarios (3 to 12: narrow and wide cavities, large means, small shapes, data far from the cavity). "Before" is the library at
the baseline commit with the same harness, "after" is this branch.

| operator | cases | z before | z after | sd before | sd after |
| --- | ---: | ---: | ---: | ---: | ---: |
| apply abs | 2 | 7.10e-15 | 7.10e-15 | 1.89e-15 | 1.89e-15 |
| beta-two-shapes | 9 | 6.24e-4 | 6.24e-4 | 1.19e-3 | 1.19e-3 |
| dirichlet-concentration | 10 | 2.29e-14 | 6.73e-14 | 4.23e-13 | 5.11e-13 |
| gamma-product | 9 | 1.33e-13 | 1.34e-13 | 1.51e-12 | 1.49e-12 |
| gamma-rate | 6 | 1.55e-9 | 1.55e-9 | 4.14e-9 | 4.14e-9 |
| gaussian-gamma-precision | 12 | 5.71e-12 | 7.50e-12 | 1.45e-9 | 8.89e-12 |
| gaussian-mv-function-scale | 3 | 1.17e-12 | 1.17e-12 | 4.98e-12 | 4.98e-12 |
| gaussian-mv-precision-scale | 6 | 3.24e-8 | 3.24e-8 | 2.27e-7 | 2.27e-7 |
| gaussian-scale | 6 | 5.39e-8 | 5.39e-8 | 9.11e-6 | 2.79e-7 |
| greater-than | 3 | 2.73e-11 | 2.73e-11 | 1.12e-8 | 1.12e-8 |
| likelihood-pair | 10 | 9.73e-6 | 9.73e-6 | 3.86e-5 | 3.86e-5 |
| likelihood-site laplace | 3 | 1.49e-6 | 1.49e-6 | 3.72e-6 | 3.72e-6 |
| likelihood-site ordinal-probit | 3 | 3.17e-14 | 3.17e-14 | 9.99e-16 | 9.99e-16 |
| likelihood-site student-t | 3 | 9.22e-6 | 9.22e-6 | 1.39e-3 | 1.39e-3 |
| likelihood-triple (new, `ordinal-cuts` gap) | 15 | n/a | 7.87e-4 | n/a | 4.82e-4 |
| mixture-site | 2 | 2.23e-10 | 1.98e-10 | 3.55e-6 | 2.83e-12 |
| negative-binomial-dispersion | 8 | 3.48e-2 | 1.07e-4 | 1.47e-2 | 1.00e-4 |
| prior-site half-normal (log) | 3 | 6.15e-14 | 1.37e-15 | 2.90e-12 | 2.11e-15 |
| prior-site uniform | 2 | 7.07e-15 | 5.55e-15 | 2.66e-15 | 2.22e-15 |
| product | 3 | 7.11e-12 | 7.11e-12 | 1.78e-8 | 2.66e-15 |

What it says: the rules were already accurate to 1e-5 or better wherever they were not the 24-point rule; the changes that were meant to
improve things do (negative binomial 300 times, the centred sds of the product, mixture-site, gaussian-scale and the precision tilt by 3 to
6 orders of magnitude, the half-normal prior), and the ones meant to be neutral are neutral to roundoff (the 1e-14 to 1e-12 differences
in `dirichlet-concentration` and `gaussian-gamma-precision` z are rounding). The test bounds each operator at about ten times its error
after; `greater-than` and the other step sites are measured but not changed (see below).

## The digest

`make ep-digest-check` over the 90 atoms and models (and their variational twins) the changes can reach, 171 items:
105 bit-identical, 1 exact, 52 within the tolerant level, 13 beyond it (beliefs differ from the baseline by more than 1e-9 relative). The
largest moves of the 13 (old to new), all far inside the policy:

| item | evidence | z-scores and sd ratios |
| --- | --- | --- |
| atom/poisson-log-dispersion-learned | -47.34897065904611 to -47.34897050730459 (1.5e-7) | at most 7.6e-7 (an sd ratio of 1.035973) |
| model/negative-binomial-regression | not a number of the digest | converges in 16 sweeps, not 17 (6528 updates, not 6936); beliefs move by 2e-7 |
| atom/mixture-components-learned | -97.63366151156465 to -97.63366151156619 (1.5e-12) | at most 1.2e-11 |
| atom/poisson-log | -3.568605296484585 to -3.568605296483831 (7.5e-13) | at most 3.9e-11 |
| atom/clutter-learned | -67.47651899820589 to -67.47651899820721 (1.3e-12) | at most 2.2e-11 |
| atom/poisson-log-learned | -50.28775376122098 to -50.28775376122144 (4.6e-13) | at most 1e-14 |
| atom/clutter, atom/mixture-components, model/poisson-regression | at most 1e-12 | beliefs only (no check rows) |
| vmp/half-normal | none | meanZ 0.104805, sdRatio 1.114555 (by 3.4e-9 and 2.1e-9) |
| vmp/laplace, vmp/laplace-learned, vmp/poisson-log, vmp/poisson-log-dispersion-learned | none | at most 1.1e-9 |

Over all 171 items the largest move of an evidence is 1.5e-7, of a z-score 7.2e-8 and of an sd ratio 7.6e-7 (all from the
negative binomial atom); 56 items move at all. No item's improper, skipped or restart counts move, and no sweep count but the negative binomial model's.

## Not done, and why

- **Centred variance in the step operators** (`greater-than`, `is-between`: `regionMoments` in `operators/links/truncation.ts`). With the
  exact variance an inactive constraint (a truncation that cuts nothing) sends an exactly flat message, and `model/gaussian-mixture-ordered`
  then reports its evidence undefined (an improper cavity at `mu#0[0]`; 10 sweeps and no improper message instead of 54 sweeps, 16 skipped
  messages and a restart; beliefs identical to 1e-10) where the rounding noise of the old formula kept it defined (-87.8408). That is the
  engine's flat-site evidence policy (S4), not a numerics question, so the old formula stays and its floor is the collapse report only.
  The remaining sites that floor a variance (`maximum`, `minimum`, `spin`, the unary-function pushforwards, `truncatedNormalMoments`) are
  closed forms, not grid moments, and also stay.
- **One window for the panelled grid** (`_grid.ts`): its gamma axis keeps 12 / 8 / 45 and its beta axis 10 / 10 / 40. With the tilts' 20 / 20 / 60
  the evidence of `atom/bivariate-correlation` (three axes, sixteen to twenty-four nodes each) moved by 1.27e-3 nats: measured and reverted.
- **A window found by scanning for the Student-t**: its tail keeps the mass within 40 nats of the top out to hundreds of scales and one grid
  over that cannot resolve the core (the sd error was 9.5e-2 standard deviations, against 1.4e-3 with the window in its own scale), so it is
  the one likelihood that still says where it lives.
- The tilted moments of a `numericOperator` and of `likelihood-pair` still go through `TiltedMoments` as `[E x, E x^2]` and the engine
  subtracts the square again; fixing that is the contract change of S6 (a site rule returning `{ mean, variance }`).

## Found on the way (not mine, not fixed)

- Two `@example`s fail at the baseline commit already: `applyProduct` (`'4.000'` against `'3.997'`) and `dirichletConcentration`
  (`['0.287', '-0.289']` against `['0.469', '-0.104']`). The accuracy harness agrees with the library on the second (to 1e-14), so it is
  the documented number that is stale.
- `operators/kernel/variational.test.ts` "multiclass-site: three Gaussian scores" exceeds the 5 s test timeout at `nice -n 15` on a busy
  machine, at the baseline commit too.

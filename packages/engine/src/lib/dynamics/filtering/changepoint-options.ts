/**
 * The two slots of Bayesian online changepoint detection as option tables: the segment model (the kind of shift to look
 * for: the mean, the variance, both, the rate of counts, a binary frequency, an AR(1) coefficient, a linear trend), each
 * with the knobs of its own prior, and the hazard (the prior on how long a segment lasts: constant, with a minimum
 * length, Weibull, gamma, negative binomial), each with its own knobs. Each table is one `union` space for controls,
 * grids and tuners, and builds the library's `ConjugateModel` or hazard function from a configuration.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type DictSpace,
  defaults,
  dict,
  int,
  type Knobs,
  knobOption,
  type Option,
  type OptionTable,
  optionTable,
  type RealSpace,
  real,
  type SpaceValue,
  type UnionSpace,
} from '../../space/index.ts'
import { constantHazard } from './changepoint.ts'
import { gammaHazard, minimumSegmentHazard, negativeBinomialHazard, weibullHazard } from './hazards.ts'
import type { ConjugateModel } from './segment-models.ts'

/** What a segment model looks for, for validity rules and selectors. */
export interface SegmentModelProperties {
  /** What changes between segments. */
  readonly shifts: 'mean' | 'variance' | 'mean and variance' | 'rate' | 'frequency' | 'dynamics' | 'trend'
  /** Whether the observations are counts or bits rather than real numbers. */
  readonly discrete: boolean
  /** Whether the predictive reads the previous value or the time (the series' order matters beyond the segments). */
  readonly regression: boolean
}

/** A hazard function of the run length. */
export type Hazard = (runLength: number) => number

/** What a hazard prior on segment lengths does, for validity rules and selectors. */
export interface HazardProperties {
  /** Whether the hazard is the same at every run length (geometric lengths). */
  readonly constant: boolean
  /** Whether a segment can end before it has some minimum length. */
  readonly allowsShort: boolean
}

function model(
  description: string,
  properties: SegmentModelProperties,
  space: DictSpace,
  build: (k: Knobs) => ConjugateModel,
): Option<ConjugateModel, SegmentModelProperties> {
  return knobOption(description, properties, space, build)
}

function hazard(
  description: string,
  properties: HazardProperties,
  space: DictSpace,
  build: (k: Knobs) => Hazard,
): Option<Hazard, HazardProperties> {
  return knobOption(description, properties, space, build)
}

const sigma = real(0.001, 100, { log: true, default: 1, label: 'noise standard deviation' })
const priorMean = real(-1000, 1000, { default: 0, label: 'prior mean' })
const shape = (label: string, def: number): RealSpace => real(0.01, 100, { log: true, default: def, label })

/**
 * The segment models as one slot, named after the shift each looks for: `mean` (Gaussian, known noise), `variance`
 * (Gaussian, known mean), `mean-variance` (normal-gamma), `rate` (Poisson counts), `bernoulli` (bits), `ar` (the AR(1)
 * coefficient, known noise) and `trend` (intercept and slope, known noise). The names of `mean`, `variance`, `rate`,
 * `ar` and `trend` are those of the generator `changepointSeries`' kinds of change.
 *
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.segmentModels.make({ kind: 'bernoulli', alpha: 1, beta: 2 }) // => { kind: 'bernoulli-beta', alpha: 1, beta: 2 }
 * ```
 */
export const segmentModels: OptionTable<ConjugateModel, SegmentModelProperties> = optionTable(
  {
    mean: model(
      'a Gaussian with known noise whose mean shifts, with a normal prior on the mean',
      { shifts: 'mean', discrete: false, regression: false },
      dict({
        sigma,
        priorMean,
        priorStd: real(0.001, 1000, { log: true, default: 3, label: 'prior standard deviation of the mean' }),
      }),
      (k) => ({
        kind: 'normal-known-variance',
        sigma: k('sigma'),
        priorMean: k('priorMean'),
        priorStd: k('priorStd'),
      }),
    ),
    variance: model(
      'a Gaussian with a known mean whose variance shifts, with an inverse-gamma prior on the variance',
      { shifts: 'variance', discrete: false, regression: false },
      dict({
        mean: real(-1000, 1000, { default: 0, label: 'known mean' }),
        alpha: shape('prior shape α', 3),
        beta: shape('prior scale β', 2),
      }),
      (k) => ({ kind: 'normal-known-mean', mean: k('mean'), alpha: k('alpha'), beta: k('beta') }),
    ),
    'mean-variance': model(
      'a Gaussian whose mean and variance both shift, with a normal-gamma prior (Student-t predictive)',
      { shifts: 'mean and variance', discrete: false, regression: false },
      dict({
        mu0: real(-1000, 1000, { default: 0, label: 'prior mean μ₀' }),
        kappa0: shape('prior pseudo-observations κ₀', 0.1),
        alpha0: shape('prior shape α₀', 1),
        beta0: shape('prior scale β₀', 1),
      }),
      (k) => ({
        kind: 'normal-gamma',
        mu0: k('mu0'),
        kappa0: k('kappa0'),
        alpha0: k('alpha0'),
        beta0: k('beta0'),
      }),
    ),
    rate: model(
      'Poisson counts whose rate shifts, with a gamma prior on the rate',
      { shifts: 'rate', discrete: true, regression: false },
      dict({ shape: shape('prior shape', 2), rate: shape('prior rate', 0.4) }),
      (k) => ({ kind: 'poisson-gamma', shape: k('shape'), rate: k('rate') }),
    ),
    bernoulli: model(
      'bits whose frequency of ones shifts, with a beta prior on the frequency',
      { shifts: 'frequency', discrete: true, regression: false },
      dict({ alpha: shape('prior α', 1), beta: shape('prior β', 1) }),
      (k) => ({ kind: 'bernoulli-beta', alpha: k('alpha'), beta: k('beta') }),
    ),
    ar: model(
      'an AR(1) process whose coefficient shifts, with a normal prior on the coefficient and known noise',
      { shifts: 'dynamics', discrete: false, regression: true },
      dict({
        sigma,
        priorStd: real(0.01, 10, { log: true, default: 0.5, label: 'prior standard deviation of the coefficient' }),
      }),
      (k) => ({ kind: 'ar1-known-noise', sigma: k('sigma'), priorStd: k('priorStd') }),
    ),
    trend: model(
      'a line in the time since the segment began, whose intercept and slope shift, with normal priors and known noise',
      { shifts: 'trend', discrete: false, regression: true },
      dict({
        sigma,
        priorMean,
        interceptStd: real(0.001, 1000, { log: true, default: 3, label: 'prior standard deviation of the intercept' }),
        slopeStd: real(0.0001, 100, { log: true, default: 0.1, label: 'prior standard deviation of the slope' }),
      }),
      (k) => ({
        kind: 'trend-known-noise',
        sigma: k('sigma'),
        priorMean: k('priorMean'),
        interceptStd: k('interceptStd'),
        slopeStd: k('slopeStd'),
      }),
    ),
  },
  {
    default: 'mean',
    label: 'kind of shift',
    description: 'What changes between segments, and the prior on the segment model’s parameters.',
  },
)

/**
 * The hazards as one slot: `constant` (geometric lengths), `minimum-segment` (geometric beyond a shortest length),
 * `weibull`, `gamma` and `negative-binomial` lengths. The run length at which a changepoint is likely is the
 * hazard's own, so the choice sets how regular the regimes are expected to be.
 *
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.hazards.make({ kind: 'minimum-segment', rate: 0.1, minSegment: 3 })(2) // => 0
 * ```
 */
export const hazards: OptionTable<Hazard, HazardProperties> = optionTable(
  {
    constant: hazard(
      'every step starts a new segment with the same probability (geometric lengths)',
      { constant: true, allowsShort: true },
      dict({ rate: real(0.0001, 0.5, { log: true, default: 0.01, label: 'changepoint probability per step' }) }),
      (k) => constantHazard(1 / k('rate')),
    ),
    'minimum-segment': hazard(
      'no segment shorter than a minimum, then a constant probability per step',
      { constant: false, allowsShort: false },
      dict({
        rate: real(0.0001, 0.5, { log: true, default: 0.01, label: 'changepoint probability per step' }),
        minSegment: int(1, 10_000, { log: true, default: 5, label: 'shortest segment length' }),
      }),
      (k) => minimumSegmentHazard(k('rate'), k('minSegment')),
    ),
    weibull: hazard(
      'Weibull segment lengths: a shape above one makes a segment likelier to end the older it is',
      { constant: false, allowsShort: true },
      dict({
        shape: real(0.2, 10, { log: true, default: 2, label: 'shape' }),
        scale: real(1, 100_000, { log: true, default: 50, label: 'scale (typical length)' }),
      }),
      (k) => weibullHazard(k('shape'), k('scale')),
    ),
    gamma: hazard(
      'gamma segment lengths, concentrated around shape / rate',
      { constant: false, allowsShort: true },
      dict({
        shape: real(0.2, 100, { log: true, default: 5, label: 'shape' }),
        rate: real(0.0001, 10, { log: true, default: 0.1, label: 'rate' }),
      }),
      (k) => gammaHazard(k('shape'), k('rate')),
    ),
    'negative-binomial': hazard(
      'negative-binomial segment lengths: one plus the failures before the n-th success',
      { constant: false, allowsShort: true },
      dict({
        successes: real(0.2, 1000, { log: true, default: 5, label: 'successes' }),
        p: real(0.001, 0.999, { default: 0.1, label: 'success probability' }),
      }),
      (k) => negativeBinomialHazard(k('successes'), k('p')),
    ),
  },
  {
    default: 'constant',
    label: 'segment length prior',
    description: 'The prior on how long a segment lasts, as the hazard of a changepoint at each run length.',
  },
)

/**
 * A complete value of the `segmentModels` slot: the option's name and its knobs, those not given at their defaults (the
 * form the estimator's hyperparameter takes, and what a control for the union space produces).
 *
 * @param kind - The option: `mean`, `variance`, `mean-variance`, `rate`, `bernoulli`, `ar` or `trend`.
 * @param knobs - Values for some of the option's own knobs.
 * @returns The configuration, `{ kind, ...knobs }` with every knob of the option.
 * @throws MdmlError with code `value` for an option the table lacks.
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.segmentModelValue('rate', { shape: 4 }) // => { kind: 'rate', shape: 4, rate: 0.4 }
 * ```
 */
export function segmentModelValue(kind: string, knobs: Readonly<Record<string, number>> = {}): SpaceValue {
  return optionValue(segmentModels.space, kind, knobs)
}

/**
 * A complete value of the `hazards` slot: the option's name and its knobs, those not given at their defaults.
 *
 * @param kind - The option: `constant`, `minimum-segment`, `weibull`, `gamma` or `negative-binomial`.
 * @param knobs - Values for some of the option's own knobs.
 * @returns The configuration, `{ kind, ...knobs }` with every knob of the option.
 * @throws MdmlError with code `value` for an option the table lacks.
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.hazardValue('minimum-segment', { minSegment: 8 }) // => { kind: 'minimum-segment', rate: 0.01, minSegment: 8 }
 * ```
 */
export function hazardValue(kind: string, knobs: Readonly<Record<string, number>> = {}): SpaceValue {
  return optionValue(hazards.space, kind, knobs)
}

function optionValue(space: UnionSpace, kind: string, knobs: Readonly<Record<string, number>>): SpaceValue {
  const option = space.options[kind]
  if (option === undefined)
    throw new MdmlError('value', `'${kind}' is not one of ${Object.keys(space.options).join(', ')}`)
  const tag = space.tag ?? 'kind'
  for (const name of Object.keys(knobs))
    if (!(name in option.spaces)) throw new MdmlError('value', `'${kind}' has no knob '${name}'`)
  return { [tag]: kind, ...(defaults(option) as Readonly<Record<string, SpaceValue>>), ...knobs }
}

/**
 * The `segmentModels` value of a conjugate model: the option that builds it and its prior as knobs (the inverse of
 * `segmentModels.make`), to point the detector at the model that generated a series.
 *
 * @param model - The conjugate model.
 * @returns The configuration.
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.segmentModelValueOf({ kind: 'bernoulli-beta', alpha: 2, beta: 3 }) // => { kind: 'bernoulli', alpha: 2, beta: 3 }
 * ```
 */
export function segmentModelValueOf(model: ConjugateModel): SpaceValue {
  const { kind, ...knobs } = model
  const name = {
    'normal-known-variance': 'mean',
    'normal-known-mean': 'variance',
    'normal-gamma': 'mean-variance',
    'poisson-gamma': 'rate',
    'bernoulli-beta': 'bernoulli',
    'ar1-known-noise': 'ar',
    'trend-known-noise': 'trend',
  }[kind]
  return { kind: name, ...knobs }
}

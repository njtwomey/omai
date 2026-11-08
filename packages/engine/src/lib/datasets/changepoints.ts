/**
 * Changepoint series with known regime changes (in mean, variance, rate, autoregressive dynamics or trend), for
 * detection methods such as BOCPD, PELT and binary segmentation. It also holds `exactBocpd`, exact Bayesian online
 * changepoint detection (Adams and MacKay, 2007) for a conjugate segment model, so a generated series comes with the
 * true run-length posterior.
 */

import { OmaiError } from '../base/index.ts'
import { type DatasetGenerator, type TimeSeriesData, timeSeries } from '../data/index.ts'
import { bocpd, type ConjugateModel } from '../dynamics/filtering/index.ts'
import { foldIn, gamma, normal, poisson, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { labels, stamp, vector } from './_util.ts'

/** The kinds of regime change {@link changepointSeries} generates. */
export type ChangeKind = 'mean' | 'variance' | 'rate' | 'ar' | 'trend'

/**
 * A conjugate segment model for Bayesian online changepoint detection: the distribution of a segment's observations
 * given its parameter, and the parameter's prior. Every kind has an exact posterior predictive.
 */
export type SegmentModel =
  | { readonly kind: 'mean'; readonly sigma: number; readonly priorMean: number; readonly priorStd: number }
  | { readonly kind: 'variance'; readonly mean: number; readonly alpha: number; readonly beta: number }
  | { readonly kind: 'rate'; readonly shape: number; readonly rate: number }
  | { readonly kind: 'ar'; readonly sigma: number; readonly priorStd: number }
  | {
      readonly kind: 'trend'
      readonly sigma: number
      readonly priorMean: number
      readonly interceptStd: number
      readonly slopeStd: number
    }

/**
 * The conjugate model of dynamics' `bocpd` for a segment model of the generator (`mean` is a Gaussian with known noise,
 * `variance` a Gaussian with known mean, `rate` Poisson counts, `ar` an AR(1) coefficient, `trend` a line).
 *
 * @param model - The segment model.
 * @returns The conjugate model BOCPD runs.
 * @example
 * ```ts
 * import { conjugateModelOf } from 'omai-engine/datasets'
 * conjugateModelOf({ kind: 'rate', shape: 2, rate: 0.5 }) // => { kind: 'poisson-gamma', shape: 2, rate: 0.5 }
 * ```
 */
export function conjugateModelOf(model: SegmentModel): ConjugateModel {
  switch (model.kind) {
    case 'mean':
      return { kind: 'normal-known-variance', sigma: model.sigma, priorMean: model.priorMean, priorStd: model.priorStd }
    case 'variance':
      return { kind: 'normal-known-mean', mean: model.mean, alpha: model.alpha, beta: model.beta }
    case 'rate':
      return { kind: 'poisson-gamma', shape: model.shape, rate: model.rate }
    case 'ar':
      return { kind: 'ar1-known-noise', sigma: model.sigma, priorStd: model.priorStd }
    case 'trend':
      return {
        kind: 'trend-known-noise',
        sigma: model.sigma,
        priorMean: model.priorMean,
        interceptStd: model.interceptStd,
        slopeStd: model.slopeStd,
      }
  }
}

/** What {@link exactBocpd} returns. */
export interface BocpdResult {
  /** `log p(y_t | y_1..y_{t-1})` for every step. */
  readonly logPredictive: readonly number[]
  /** `E[y_t | y_1..y_{t-1}]` for every step. */
  readonly predictiveMean: readonly number[]
  /**
   * `p(r_t = r | y_1..y_t)` for every step: the probability that the current segment holds exactly `r` observations
   * up to and including `y_t` (so entry 0 is always 0).
   */
  readonly runLength: readonly Float64Array[]
  /** The most probable run length at each step. */
  readonly mapRunLength: readonly number[]
  /** `log p(y_1..y_n)`, the exact marginal likelihood under the model. */
  readonly logEvidence: number
}

/**
 * Exact Bayesian online changepoint detection (Adams and MacKay, 2007) for a hazard (constant, or a function of the
 * current run length) and a conjugate segment model: run-length posteriors, one-step predictives and the evidence. The
 * models all run dynamics' `filtering.bocpd` (the one implementation; the oracle's brute-force enumeration over every
 * segmentation is the independent check). Hypotheses whose posterior falls below `e^-40` of the total are dropped,
 * which changes nothing at double precision.
 * It is the exact truth for data drawn by {@link changepointSeries} from the same model.
 *
 * @param y - The series.
 * @param model - The segment model and its prior.
 * @param hazard - The probability of a changepoint before each step, or a function of the run length so far.
 * @returns Run-length posteriors, predictives and the evidence.
 * @throws OmaiError with code `value` for a constant hazard outside (0, 1).
 * @example
 * ```ts
 * import { exactBocpd } from 'omai-engine/datasets'
 * const r = exactBocpd([0, 0.1, 5, 5.2], { kind: 'mean', sigma: 0.5, priorMean: 0, priorStd: 5 }, 0.1)
 * r.mapRunLength // => [1, 2, 1, 2]
 * ```
 */
export function exactBocpd(
  y: readonly number[],
  model: SegmentModel,
  hazard: number | ((runLength: number) => number),
): BocpdResult {
  if (typeof hazard === 'number' && !(hazard > 0 && hazard < 1))
    throw new OmaiError('value', 'exactBocpd: the hazard must be in (0, 1)')
  const r = bocpd(y, conjugateModelOf(model), hazard)
  return {
    logPredictive: r.logPredictive,
    predictiveMean: r.predictiveMean,
    runLength: r.runLength,
    mapRunLength: r.mapRunLength,
    logEvidence: r.logEvidence,
  }
}

/** The segment model behind each kind, from the generator's knobs. */
function modelOf(kind: ChangeKind, noise: number, shift: number, meanRate: number, slope: number): SegmentModel {
  switch (kind) {
    case 'mean':
      return { kind, sigma: noise, priorMean: 0, priorStd: shift }
    case 'variance':
      // Prior mean of the variance is noise^2 (inverse gamma with alpha 3, beta 2 noise^2).
      return { kind, mean: 0, alpha: 3, beta: 2 * noise * noise }
    case 'rate':
      return { kind, shape: 2, rate: 2 / meanRate }
    case 'ar':
      return { kind, sigma: noise, priorStd: 0.5 }
    case 'trend':
      return { kind, sigma: noise, priorMean: 0, interceptStd: shift, slopeStd: slope }
  }
}

/**
 * Tunable changepoint series for detection methods (BOCPD, PELT, binary segmentation): once a segment has
 * `minSegment` steps, a new one starts before each step with probability `hazard` (so segment lengths are
 * `minSegment` plus a geometric number), each segment draws its parameter from a conjugate prior,
 * and the kind decides what changes: the `mean` (Gaussian, known noise), the `variance`, the `rate` of Poisson counts,
 * the coefficient of an AR(1) process (`ar`, regime switches) or the intercept and slope of a linear trend (`trend`,
 * trend breaks). The truth holds the changepoints, every segment's parameters, the model, and the exact BOCPD
 * run-length posterior and predictive (`truth.extra.bocpd()`, computed on first call).
 *
 * @index dataset Series with tunable changepoints for detection methods.
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { changepointSeries } from 'omai-engine/datasets'
 * generate(changepointSeries, { knobs: { n: 100, kind: 'rate' } }).data.values.shape // => [100, 1]
 * ```
 */
export const changepointSeries: DatasetGenerator<TimeSeriesData> = {
  name: 'changepointSeries',
  kind: 'timeseries',
  description:
    'Piecewise-stationary series whose mean, variance, rate, AR coefficient or trend changes at random times.',
  knobs: dict({
    n: int(2, 100_000, { label: 'length', description: 'The number of time steps.', default: 300 }),
    kind: choice(['mean', 'variance', 'rate', 'ar', 'trend'], {
      label: 'kind of change',
      description: 'What changes at each changepoint: the mean, variance, Poisson rate, AR coefficient or trend.',
      default: 'mean',
    }),
    hazard: real(0.0001, 0.5, {
      description: 'The probability of a changepoint at each step (a geometric segment length).',
      default: 0.01,
      label: 'changepoint probability per step',
    }),
    minSegment: int(1, 10_000, {
      description: 'The shortest allowed segment.',
      default: 1,
      label: 'shortest segment length',
    }),
    noise: real(0.001, 100, {
      description: 'The standard deviation of the observation noise.',
      default: 1,
      label: 'observation noise',
    }),
    shift: real(0.001, 100, {
      description: 'The spread of the segment means drawn at each change.',
      default: 3,
      label: 'prior spread of segment means',
    }),
    meanRate: real(0.01, 1000, {
      description: 'The prior mean of the Poisson rates (rate changes).',
      default: 5,
      label: 'prior mean of Poisson rates',
    }),
    slope: real(0.0001, 10, {
      description: 'The spread of the trend slopes (trend changes).',
      default: 0.05,
      label: 'prior spread of trend slopes',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const kind = knobs['kind'] as ChangeKind
    const hazard = knobs['hazard'] as number
    const model = modelOf(
      kind,
      knobs['noise'] as number,
      knobs['shift'] as number,
      knobs['meanRate'] as number,
      knobs['slope'] as number,
    )
    const u = uniform(foldIn(k, 0), [n]).data
    const starts = [0]
    const minSegment = knobs['minSegment'] as number
    for (let t = 1; t < n; t++)
      if (t - (starts[starts.length - 1] ?? 0) >= minSegment && (u[t] ?? 1) < hazard) starts.push(t)
    const hazardOf = (runLength: number) => (runLength >= minSegment ? hazard : 0)
    const z = normal(foldIn(k, 1), [n]).data
    const y = new Float64Array(n)
    const segment = new Int32Array(n)
    const params: { start: number; end: number; parameters: Record<string, number> }[] = []
    const param = new Float64Array(n)
    for (const [s, start] of starts.entries()) {
      const end = starts[s + 1] ?? n
      const ks = foldIn(foldIn(k, 2), s)
      const draw = normal(foldIn(ks, 0), [2]).data
      let p: Record<string, number>
      switch (model.kind) {
        case 'mean':
          p = { mean: model.priorMean + model.priorStd * (draw[0] ?? 0) }
          break
        case 'variance':
          p = { variance: model.beta / (gamma(foldIn(ks, 1), [1], model.alpha).data[0] ?? 1) }
          break
        case 'rate':
          p = { rate: (gamma(foldIn(ks, 1), [1], model.shape).data[0] ?? 1) / model.rate }
          break
        case 'ar':
          p = { phi: model.priorStd * (draw[0] ?? 0) }
          break
        case 'trend':
          p = {
            intercept: model.priorMean + model.interceptStd * (draw[0] ?? 0),
            slope: model.slopeStd * (draw[1] ?? 0),
          }
          break
      }
      params.push({ start, end, parameters: p })
      const counts = model.kind === 'rate' ? poisson(foldIn(ks, 2), p['rate'] ?? 0, [end - start]).data : undefined
      for (let t = start; t < end; t++) {
        segment[t] = s
        const e = z[t] ?? 0
        const prev = t > 0 ? (y[t - 1] ?? 0) : 0
        switch (model.kind) {
          case 'mean':
            y[t] = (p['mean'] ?? 0) + model.sigma * e
            param[t] = p['mean'] ?? 0
            break
          case 'variance':
            y[t] = model.mean + Math.sqrt(p['variance'] ?? 1) * e
            param[t] = p['variance'] ?? 0
            break
          case 'rate':
            y[t] = counts?.[t - start] ?? 0
            param[t] = p['rate'] ?? 0
            break
          case 'ar':
            y[t] = (p['phi'] ?? 0) * prev + model.sigma * e
            param[t] = p['phi'] ?? 0
            break
          case 'trend':
            y[t] = (p['intercept'] ?? 0) + (p['slope'] ?? 0) * (t - start) + model.sigma * e
            param[t] = (p['intercept'] ?? 0) + (p['slope'] ?? 0) * (t - start)
            break
        }
      }
    }
    const series = Array.from(y)
    let cached: BocpdResult | undefined
    const bocpd = (): BocpdResult => {
      if (cached === undefined) cached = exactBocpd(series, model, minSegment > 1 ? hazardOf : hazard)
      return cached
    }
    return {
      data: timeSeries(y, {
        samplingRate: 1,
        segments: segment,
        events: starts.slice(1).map((t) => ({ time: t, label: 'changepoint' })),
        meta: stamp('changepointSeries', changepointSeries.description, knobs),
      }),
      truth: {
        changepoints: starts.slice(1),
        clusters: labels(segment),
        latents: { segments: labels(segment), parameter: vector(param) },
        extra: { model, hazard, minSegment, hazardFunction: hazardOf, segments: params, bocpd },
      },
    }
  },
}

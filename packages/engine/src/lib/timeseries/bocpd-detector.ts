/**
 * Bayesian online changepoint detection as an estimator on the iterative base: the segment model (the kind of shift to
 * look for, with the knobs of its prior), the hazard (the prior on how long a segment lasts) and the pruning and
 * detection thresholds are its hyperparameters, filled from the option tables of `dynamics/filtering`; the fit steps
 * through a series one observation at a time, so its `session` runs live and `fittedAt(t)` is the detector after t
 * observations. The fitted value carries the change probabilities, the MAP run lengths, the evidence and the
 * thresholded changepoints, and its exposition is the run-length posterior over time with the true changepoints
 * overlaid when the data carries them.
 */

import { type Json, OmaiError } from '../base/index.ts'
import type { TimeSeriesData } from '../data/index.ts'
import {
  type BocpdPath,
  type BocpdState,
  bocpdAlgorithm,
  bocpdContract,
  type ConjugateModel,
  hazards,
  type MapChangepoint,
  mapChangepoints,
  segmentModels,
} from '../dynamics/filtering/index.ts'
import { runLengthStrip } from '../exposition/index.ts'
import { type FittedFacts, type IterativeEstimator, iterative, type LearnData, type Verbs } from '../learn/index.ts'
import { dict, type RealSpace, real, type SpaceValue } from '../space/index.ts'
import { asTensor, type Tensor, toFlat } from '../tensor/index.ts'
import { defineContract, derived, Log } from '../trace/index.ts'
import { asSeries } from './_series.ts'

/** A series for the detector: the observations as `x`, and what is known of its truth. */
export interface ChangepointData extends LearnData {
  /** The steps at which a segment truly starts, when known (from the generator's truth or a labelling). */
  readonly changepoints?: readonly number[]
}

/** The hyperparameters of {@link bocpdDetector}. */
export interface BocpdDetectorHyper {
  /** The segment model, a value of `segmentModels.space`: `{ kind: 'mean', sigma, priorMean, priorStd }` and so on. */
  readonly model: SpaceValue
  /** The hazard, a value of `hazards.space`: `{ kind: 'constant', rate }` and so on. */
  readonly hazard: SpaceValue
  /** Log probability below which a run-length hypothesis is dropped (pruning; -40 changes nothing at double precision). */
  readonly threshold: number
  /** The probability the most probable run length must reach to declare a changepoint. */
  readonly changeThreshold: number
}

/** The fitted parameters of {@link bocpdDetector}: what the detector said at every step, as plain numbers. */
export type BocpdParams = {
  readonly model: ConjugateModel
  readonly steps: number
  readonly changeProbability: readonly number[]
  readonly mapRunLength: readonly number[]
  readonly mapProbability: readonly number[]
  readonly logPredictive: readonly number[]
  readonly predictiveMean: readonly number[]
  readonly predictiveStd: readonly number[]
  readonly logEvidence: number
  readonly changeThreshold: number
}

/** What a fitted {@link bocpdDetector} offers beyond its parameters. */
export interface BocpdVerbs extends Verbs, FittedFacts {
  /** The changepoints at a probability threshold (default the fit's): the first observation of each new segment. */
  changepoints(threshold?: number): number[]
  /** The same with the step each was detected at and the probability then. */
  detections(threshold?: number): MapChangepoint[]
}

/** The paths the detector adds to those of the algorithm: the run-length posterior so far, and the detections. */
export type BocpdDetectorPath = BocpdPath | 'model/run-length' | 'model/changepoints'

/**
 * Data for the detector from a series: a plain array, a single-channel time series (its segment labels or its
 * `changepoint` events become the truth) or a tensor.
 *
 * @param series - The observations, one per step.
 * @param truth - The steps at which a segment truly starts, when known (overrides what the series carries).
 * @returns Data the detector fits.
 * @throws OmaiError with code `shape` for a series with more than one channel.
 * @example
 * ```ts
 * import { changepointData } from 'omai-engine/timeseries'
 * changepointData([0, 0, 5, 5], [2]).changepoints // => [2]
 * ```
 */
export function changepointData(
  series: readonly number[] | TimeSeriesData | Tensor,
  truth?: readonly number[],
): ChangepointData {
  if (Array.isArray(series) || !('kind' in series && series.kind === 'timeseries')) {
    const x = asTensor(series as readonly number[] | Tensor, 'float64')
    return { x, ...(truth === undefined ? {} : { changepoints: [...truth] }) }
  }
  const ts = series as TimeSeriesData
  if ((ts.values.shape[1] ?? 1) !== 1) throw new OmaiError('shape', 'changepointData: expected a single-channel series')
  const labels = ts.segments === undefined ? undefined : toFlat(ts.segments)
  const fromLabels = labels?.flatMap((v, t) => (t > 0 && v !== labels[t - 1] ? [t] : []))
  const fromEvents = ts.events
    .filter((e) => e.label === 'changepoint')
    .map((e) => Math.round(e.time * (ts.samplingRate ?? 1)))
  const known = truth ?? fromLabels ?? (ts.events.length > 0 ? fromEvents : undefined)
  return { x: ts.values, ...(known === undefined ? {} : { changepoints: [...known] }) }
}

/** The steps at which a segment truly starts, as the data carries them. */
function truthOf(data: LearnData): readonly number[] | undefined {
  const given = (data as ChangepointData).changepoints
  if (given !== undefined) return given
  if (data.y === undefined) return undefined
  const labels = toFlat(asTensor(data.y))
  return labels.flatMap((v, t) => (t > 0 && v !== labels[t - 1] ? [t] : []))
}

/** The series of the data, checked against what the model can score. */
function seriesOf(data: LearnData, model: ConjugateModel): readonly number[] {
  const y = Array.from(asSeries(data.x, 'bocpdDetector'))
  if (y.some((v) => !Number.isFinite(v))) throw new OmaiError('value', 'bocpdDetector: the series must be finite')
  if (model.kind === 'poisson-gamma' && y.some((v) => v < 0 || !Number.isInteger(v)))
    throw new OmaiError('value', 'bocpdDetector: the rate model scores counts (non-negative integers)')
  if (model.kind === 'bernoulli-beta' && y.some((v) => v !== 0 && v !== 1))
    throw new OmaiError('value', 'bocpdDetector: the bernoulli model scores bits (0 or 1)')
  return y
}

const pruning: RealSpace = real(-100, -2, { default: -40, label: 'pruning threshold (log probability)' })

/**
 * Bayesian online changepoint detection (Adams and MacKay 2007) as an estimator: choose the kind of shift (`model`: the
 * mean, the variance, both, the rate of counts, bits, an AR(1) coefficient, a linear trend, each with its prior), the
 * prior on segment lengths (`hazard`: constant, minimum segment, Weibull, gamma, negative binomial), the pruning
 * `threshold` and the `changeThreshold` at which the most probable run length declares a changepoint. `fit` takes a
 * series (see {@link changepointData}); `session` steps through it live, logging the paths of `bocpdContract` and, as
 * derived paths, the run-length posterior so far and the detections. The fitted value has the change probabilities, MAP
 * run lengths, predictives and evidence per step and `changepoints()`; its exposition is the run-length posterior as a
 * strip with the true changepoints marked when the data carries them.
 *
 * @concept bayesian-changepoint-detection
 * @index estimator Bayesian online changepoint detection.
 * @example
 * ```ts
 * import { bocpdDetector, changepointData } from 'omai-engine/timeseries'
 * const series = [0, 0.1, -0.1, 0, 0.1, 5, 5.2, 4.9, 5.1, 5]
 * const fitted = bocpdDetector
 *   .with({ model: { kind: 'mean', sigma: 0.5, priorMean: 0, priorStd: 5 }, hazard: { kind: 'constant', rate: 0.1 } })
 *   .fit(changepointData(series))
 * fitted.changepoints() // => [5]
 * ```
 */
export const bocpdDetector: IterativeEstimator<
  BocpdDetectorHyper,
  BocpdState,
  BocpdParams,
  BocpdVerbs,
  BocpdDetectorPath
> = iterative<BocpdDetectorHyper, BocpdState, BocpdParams, BocpdVerbs, BocpdDetectorPath>({
  name: 'timeseries/bocpd-detector',
  kind: 'other',
  defaults: {
    model: { kind: 'mean', sigma: 1, priorMean: 0, priorStd: 3 },
    hazard: { kind: 'constant', rate: 0.01 },
    threshold: -40,
    changeThreshold: 0.5,
  },
  hyperSpace: (h) =>
    dict({
      model: segmentModels.space,
      hazard: hazards.space,
      threshold: { ...pruning, default: h.threshold },
      changeThreshold: real(0.05, 0.99, { default: h.changeThreshold, label: 'detection threshold' }),
    }),
  algorithm: (data, h) => {
    const model = segmentModels.make(h.model)
    return bocpdAlgorithm(seriesOf(data, model), model, hazards.make(h.hazard), { threshold: h.threshold })
  },
  maxSteps: (_, data) => asSeries(data.x, 'bocpdDetector').length,
  params: (s, _data, h) => ({
    model: segmentModels.make(h.model),
    steps: s.t,
    changeProbability: s.history.changeProbability,
    mapRunLength: s.history.mapRunLength,
    mapProbability: s.history.mapProbability,
    logPredictive: s.history.logPredictive,
    predictiveMean: s.history.predictiveMean,
    predictiveStd: s.history.predictiveStd,
    logEvidence: s.logEvidence,
    changeThreshold: h.changeThreshold,
  }),
  verbs: (p) => ({
    changepoints: (threshold) => mapChangepoints(p, threshold ?? p.changeThreshold).map((c) => c.index),
    detections: (threshold) => mapChangepoints(p, threshold ?? p.changeThreshold),
  }),
  contract: (data, h) =>
    defineContract({
      ...bocpdContract,
      'model/run-length': derived<BocpdState>({
        description: 'the run-length posterior over time so far, with the true changepoints when known',
        cost: 'dear',
        compute: (s) => runLengthStrip(stripOf(s, truthOf(data))),
      }),
      'model/changepoints': derived<BocpdState>({
        description: 'the changepoints detected so far, with the step and probability at detection',
        compute: (s) => mapChangepoints(s.history, h.changeThreshold) as unknown as Json,
      }),
    }),
  records: (s, data) => [runLengthStrip(stripOf(s, truthOf(data)))],
  logs: {
    'bocpd/change-probability': Log(),
    'bocpd/map-run-length': Log(),
    'bocpd/map-probability': Log(),
    'bocpd/log-evidence': Log(),
    'bocpd/predictive-mean': Log(),
    'bocpd/predictive-std': Log(),
    'bocpd/run-length': Log({ detail: 'full' }),
  },
})

function stripOf(s: BocpdState, truth: readonly number[] | undefined): Parameters<typeof runLengthStrip>[0] {
  const steps = s.history.mapRunLength.length
  return {
    runLength: s.history.runLength,
    mapRunLength: s.history.mapRunLength,
    ...(truth === undefined ? {} : { changepoints: truth.filter((c) => c >= 0 && c < steps) }),
  }
}

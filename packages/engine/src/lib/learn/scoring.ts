/**
 * Scorers, scikit-learn style: functions of a fitted estimator and data where higher is better, with losses negated.
 * `makeScorer` turns a metric into a scorer that calls the right verb (predictions, probabilities or decision scores),
 * `getScorer` looks one up by scikit-learn name, and helpers normalise scoring options for cross-validation and
 * searches.
 */

import { type MetricResult, WhiteboxError } from '../base/index.ts'
import {
  accuracyScore,
  averagePrecisionScore,
  balancedAccuracyScore,
  brierScoreLoss,
  explainedVarianceScore,
  f1Score,
  logLoss,
  maxError,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanSquaredError,
  medianAbsoluteError,
  precisionScore,
  r2Score,
  recallScore,
  rocAucScore,
  rootMeanSquaredError,
} from '../metrics/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Fitted, type LearnData, requireTargets, requireVerb } from './estimator.ts'

/** Which verb a scorer calls: point predictions, class probabilities or decision scores. */
export type ResponseMethod = 'predict' | 'predictProba' | 'decisionFunction'

/**
 * A scorer: a function of a fitted estimator and data where higher is better (sklearn's convention: losses are
 * negated and named `neg_...`). Model selection, inspection and feature selection all speak scorers.
 */
export interface Scorer {
  readonly name: string
  readonly response: ResponseMethod
  readonly score: (fitted: Fitted, data: LearnData) => number
}

/** A metric as {@link makeScorer} accepts it: targets, a response and options, returning a metric result. */
export type ScoringMetric = (yTrue: TensorLike, yPred: TensorLike, options?: never) => MetricResult

/** Options of {@link makeScorer}. */
export interface MakeScorerOptions {
  /** The scorer's name (default the metric's spec name). */
  readonly name?: string
  /** The verb whose output the metric reads (default `'predict'`). */
  readonly response?: ResponseMethod
  /** Whether larger metric values are better (default true); otherwise the score is the negated metric. */
  readonly greaterIsBetter?: boolean
  /** For probabilities of a binary problem, pass the positive column `[n]` (true) or the matrix (default false). */
  readonly positiveColumn?: boolean
  /** Extra options for the metric (sample weights come from the data). */
  readonly metricOptions?: Readonly<Record<string, unknown>>
  /** Pass the fitted classes as the metric's `labels` (for probability matrices; default false). */
  readonly passLabels?: boolean
}

function positive(t: Tensor): Tensor {
  if (t.shape.length !== 2) return t
  const n = t.shape[0] ?? 0
  const k = t.shape[1] ?? 1
  if (k !== 2) return t
  const v = toFlat(t)
  return fromStorage(
    Float64Array.from({ length: n }, (_, i) => v[i * 2 + 1] ?? 0),
    [n],
    'float64',
  )
}

/**
 * Turn a metric into a scorer (sklearn.metrics.make_scorer): the scorer calls the chosen verb of the fitted estimator
 * on `data.x`, compares with `data.y` (weighted by `data.weights`) and negates losses.
 *
 * @param metric - A metric function (from `whitebox-engine/metrics` or your own).
 * @param options - name, response, greaterIsBetter, positiveColumn, passLabels and metricOptions.
 * @returns The scorer.
 * @example
 * ```ts
 * import { makeScorer } from 'whitebox-engine/learn'
 * import { meanAbsoluteError } from 'whitebox-engine/metrics'
 * import { tensor } from 'whitebox-engine/tensor'
 * const scorer = makeScorer(meanAbsoluteError, { greaterIsBetter: false, name: 'neg_mae' })
 * const fitted = { estimator: 'zero', kind: 'regressor', params: {}, predict: () => tensor([0, 0]) }
 * scorer.score(fitted, { x: [[1], [2]], y: [1, 3] }) // => -2
 * ```
 */
export function makeScorer(
  metric: ScoringMetric | ((...args: never[]) => MetricResult),
  options: MakeScorerOptions = {},
): Scorer {
  const response = options.response ?? 'predict'
  const sign = options.greaterIsBetter === false ? -1 : 1
  const spec = (metric as { readonly spec?: { readonly name: string } }).spec
  const name = options.name ?? spec?.name ?? 'score'
  const call = metric as unknown as (t: TensorLike, p: TensorLike, o: Readonly<Record<string, unknown>>) => MetricResult
  return {
    name,
    response,
    score: (fitted, data) => {
      const y = requireTargets(data, name)
      let out = requireVerb(fitted, response, name)(data.x)
      if (response !== 'predict' && options.positiveColumn === true) out = positive(out)
      const extra: Record<string, unknown> = { ...(options.metricOptions ?? {}) }
      if (data.weights !== undefined) extra['sampleWeight'] = data.weights
      if (options.passLabels === true && fitted.classes !== undefined) extra['labels'] = fitted.classes
      if (options.positiveColumn === true && fitted.classes !== undefined && fitted.classes.length === 2) {
        extra['posLabel'] = fitted.classes[1]
      }
      return sign * call(y, out, extra).estimate.value
    },
  }
}

/** The scorer names {@link getScorer} knows (sklearn's). */
export type ScorerName =
  | 'accuracy'
  | 'balanced_accuracy'
  | 'f1'
  | 'f1_macro'
  | 'f1_micro'
  | 'f1_weighted'
  | 'precision'
  | 'recall'
  | 'roc_auc'
  | 'roc_auc_ovr'
  | 'average_precision'
  | 'neg_log_loss'
  | 'neg_brier_score'
  | 'r2'
  | 'explained_variance'
  | 'max_error'
  | 'neg_mean_squared_error'
  | 'neg_root_mean_squared_error'
  | 'neg_mean_absolute_error'
  | 'neg_median_absolute_error'
  | 'neg_mean_absolute_percentage_error'

const SCORERS: Readonly<Record<ScorerName, () => Scorer>> = {
  accuracy: () => makeScorer(accuracyScore, { name: 'accuracy' }),
  balanced_accuracy: () => makeScorer(balancedAccuracyScore, { name: 'balanced_accuracy' }),
  f1: () => makeScorer(f1Score, { name: 'f1' }),
  f1_macro: () => makeScorer(f1Score, { name: 'f1_macro', metricOptions: { average: 'macro' } }),
  f1_micro: () => makeScorer(f1Score, { name: 'f1_micro', metricOptions: { average: 'micro' } }),
  f1_weighted: () => makeScorer(f1Score, { name: 'f1_weighted', metricOptions: { average: 'weighted' } }),
  precision: () => makeScorer(precisionScore, { name: 'precision' }),
  recall: () => makeScorer(recallScore, { name: 'recall' }),
  roc_auc: () => makeScorer(rocAucScore, { name: 'roc_auc', response: 'predictProba', positiveColumn: true }),
  roc_auc_ovr: () =>
    makeScorer(rocAucScore, {
      name: 'roc_auc_ovr',
      response: 'predictProba',
      passLabels: true,
      metricOptions: { multiClass: 'ovr' },
    }),
  average_precision: () =>
    makeScorer(averagePrecisionScore, { name: 'average_precision', response: 'predictProba', positiveColumn: true }),
  neg_log_loss: () =>
    makeScorer(logLoss, { name: 'neg_log_loss', response: 'predictProba', greaterIsBetter: false, passLabels: true }),
  neg_brier_score: () =>
    makeScorer(brierScoreLoss, {
      name: 'neg_brier_score',
      response: 'predictProba',
      greaterIsBetter: false,
      positiveColumn: true,
    }),
  r2: () => makeScorer(r2Score, { name: 'r2' }),
  explained_variance: () => makeScorer(explainedVarianceScore, { name: 'explained_variance' }),
  max_error: () => makeScorer(maxError, { name: 'max_error', greaterIsBetter: false }),
  neg_mean_squared_error: () =>
    makeScorer(meanSquaredError, { name: 'neg_mean_squared_error', greaterIsBetter: false }),
  neg_root_mean_squared_error: () =>
    makeScorer(rootMeanSquaredError, { name: 'neg_root_mean_squared_error', greaterIsBetter: false }),
  neg_mean_absolute_error: () =>
    makeScorer(meanAbsoluteError, { name: 'neg_mean_absolute_error', greaterIsBetter: false }),
  neg_median_absolute_error: () =>
    makeScorer(medianAbsoluteError, { name: 'neg_median_absolute_error', greaterIsBetter: false }),
  neg_mean_absolute_percentage_error: () =>
    makeScorer(meanAbsolutePercentageError, { name: 'neg_mean_absolute_percentage_error', greaterIsBetter: false }),
}

/**
 * A scorer by sklearn name, or the scorer itself.
 *
 * @param scoring - A {@link ScorerName} or a scorer.
 * @returns The scorer.
 * @throws WhiteboxError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { getScorer } from 'whitebox-engine/learn'
 * getScorer('neg_mean_squared_error').name // => 'neg_mean_squared_error'
 * getScorer('nope') // => throws value
 * ```
 */
export function getScorer(scoring: ScorerName | Scorer): Scorer {
  if (typeof scoring !== 'string') return scoring
  const make = SCORERS[scoring]
  if (make === undefined) throw new WhiteboxError('value', `getScorer: unknown scorer ${String(scoring)}`)
  return make()
}

/**
 * The scorer for an option that may be missing: the estimator's own `score` verb by default.
 *
 * @param scoring - A scorer, a scorer name, or undefined for the default.
 * @returns The scorer.
 * @example
 * ```ts
 * import { scorerFor } from 'whitebox-engine/learn'
 * scorerFor(undefined).name // => 'score'
 * ```
 */
export function scorerFor(scoring: ScorerName | Scorer | undefined): Scorer {
  if (scoring !== undefined) return getScorer(scoring)
  return { name: 'score', response: 'predict', score: (fitted, data) => requireVerb(fitted, 'score', 'score')(data) }
}

/** Several scorers by name, as `crossValidate` and the searches accept them. */
export type Scoring = ScorerName | Scorer | readonly (ScorerName | Scorer)[] | undefined

/**
 * Normalise a scoring option to named scorers.
 *
 * @param scoring - One scorer or name, several, or undefined for the estimator's default.
 * @returns The scorers by name, in order.
 * @example
 * ```ts
 * import { scorersOf } from 'whitebox-engine/learn'
 * Object.keys(scorersOf(['r2', 'neg_mean_absolute_error'])) // => ['r2', 'neg_mean_absolute_error']
 * ```
 */
export function scorersOf(scoring: Scoring): Readonly<Record<string, Scorer>> {
  const list: readonly (ScorerName | Scorer | undefined)[] = Array.isArray(scoring)
    ? scoring
    : [scoring as ScorerName | Scorer | undefined]
  const out: Record<string, Scorer> = {}
  for (const s of list) {
    const scorer = scorerFor(s)
    out[scorer.name] = scorer
  }
  return out
}

/**
 * Conformal prediction for any model, as in MAPIE: prediction intervals for regressors and prediction sets for
 * classifiers (LAC and APS scores) with a guaranteed coverage, calibrated on held-out rows, and online conformal
 * prediction (Gibbs and Candès, 2021), which adapts its level as data arrive so coverage holds under drift.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { conformalRecord, type Explained } from '../../lib/exposition/index.ts'
import {
  adaptiveConformalInference,
  type ClassificationScore,
  type ConformalRegressor,
  conformalQuantile,
  crossConformalRegressor,
  type Estimator,
  featureNamesOf,
  intervalCoverage,
  type LearnData,
  matrixView,
  requireVerb,
  splitConformalClassifier,
  splitConformalRegressor,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { columnOf, splitData } from './_split.ts'

/** How {@link conformalRegression} calibrates. */
export type ConformalRegressionMethod = 'split' | 'normalised' | 'cv-plus'

/** Options of {@link conformalRegression}. */
export interface ConformalRegressionOptions {
  /** The miscoverage: intervals cover 1 - alpha of new rows (default 0.1). */
  readonly alpha?: number
  /**
   * `split` (absolute residuals, one width everywhere), `normalised` (residuals divided by a second model of their
   * size, so widths adapt to the noise) or `cv-plus` (Barber et al.'s CV+, no calibration rows lost).
   */
  readonly method?: ConformalRegressionMethod
  /** The shares of rows to fit on, calibrate on and test on (default 0.4, 0.3, 0.3; CV+ fits on the first two). */
  readonly fractions?: readonly [number, number, number]
  /** The feature the intervals are drawn along (default 0). */
  readonly feature?: number
  readonly key?: Key
}

const vector = (v: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(v), [v.length], 'float64')

/**
 * Conformal prediction intervals for any regressor (MAPIE's `MapieRegressor`): fit on some rows, take the conformity
 * scores (absolute residuals, or residuals over a model of their size) of held-out calibration rows, and widen every
 * prediction by their `ceil((1 - alpha)(n + 1))`-th smallest, which covers a new exchangeable row with probability at
 * least `1 - alpha`; CV+ uses cross-validated residuals instead. Exposition: a `conformal` record of the test rows along
 * one feature with their intervals, the truth, the calibration scores and their quantile, and the empirical coverage.
 *
 * @param estimator - A regressor.
 * @param data - Labelled rows.
 * @param options - Miscoverage, method, split fractions, feature and key.
 * @returns The conformal regressor, the test coverage and mean width, with the record.
 */
export function conformalRegression(
  estimator: Estimator,
  data: LearnData,
  options: ConformalRegressionOptions = {},
): Explained<{ readonly regressor: ConformalRegressor; readonly coverage: number; readonly width: number }> {
  const alpha = options.alpha ?? 0.1
  const method = options.method ?? 'split'
  const key = options.key ?? makeKey(0)
  const [fitRows, calRows, test] = splitData(data, options.fractions ?? [0.4, 0.3, 0.3], foldIn(key, 0))
  if (fitRows === undefined || calRows === undefined || test === undefined)
    throw new WhiteboxError('value', 'conformalRegression: no rows')
  let regressor: ConformalRegressor
  let scores: number[] | undefined
  if (method === 'cv-plus') {
    const both = {
      x: [...matrixRows(fitRows.x), ...matrixRows(calRows.x)],
      y: [...targets(fitRows), ...targets(calRows)],
    }
    regressor = crossConformalRegressor(estimator, both, { alpha, cv: 5, method: 'plus', key: foldIn(key, 1) })
  } else {
    const fitted = estimator.fit(fitRows, { key: foldIn(key, 1), explain: false })
    const predict = requireVerb(fitted, 'predict', 'conformalRegression')
    if (method === 'normalised') {
      // A second model of the residuals' size, fitted on the same rows.
      const residuals = toFlat(predict(fitRows.x)).map((p, i) => Math.abs((targets(fitRows)[i] ?? 0) - p))
      const scale = estimator.fit({ x: fitRows.x, y: Array.from(residuals) }, { key: foldIn(key, 2), explain: false })
      const floor = Math.max(1e-6, 0.05 * (residuals.reduce((a, b) => a + b, 0) / Math.max(1, residuals.length)))
      const sizeOf = (x: Parameters<typeof predict>[0]): Tensor =>
        vector(toFlat(requireVerb(scale, 'predict', 'conformalRegression')(x)).map((v) => Math.max(floor, v)))
      regressor = splitConformalRegressor(fitted, calRows, { alpha, score: 'normalized', scale: sizeOf })
      const p = toFlat(predict(calRows.x))
      const s = toFlat(sizeOf(calRows.x))
      scores = targets(calRows).map((y, i) => Math.abs(y - (p[i] ?? 0)) / (s[i] ?? 1))
    } else {
      regressor = splitConformalRegressor(fitted, calRows, { alpha })
      const p = toFlat(predict(calRows.x))
      scores = targets(calRows).map((y, i) => Math.abs(y - (p[i] ?? 0)))
    }
  }
  const intervals = regressor.predictInterval(test.x)
  const y = targets(test)
  const { coverage, width } = intervalCoverage(intervals, y)
  const j = options.feature ?? 0
  const along = columnOf(test.x, j)
  const order = along.map((_, i) => i).sort((a, b) => (along[a] ?? 0) - (along[b] ?? 0))
  const pick = (v: ArrayLike<number>): number[] => order.map((i) => v[i] ?? 0)
  return {
    regressor,
    coverage,
    width,
    exposition: [
      conformalRecord({
        task: 'regression',
        method:
          method === 'cv-plus' ? 'CV+' : method === 'normalised' ? 'normalised split conformal' : 'split conformal',
        alpha,
        coverage,
        ...(scores === undefined ? {} : { scores, quantile: conformalQuantile(scores, alpha) }),
        x: pick(along),
        xLabel: featureNamesOf(test, matrixView(test.x, 'conformalRegression').d)[j] ?? `x${j}`,
        y: pick(y),
        prediction: pick(toFlat(intervals.prediction)),
        lower: pick(toFlat(intervals.lower)),
        upper: pick(toFlat(intervals.upper)),
      }),
    ],
  }
}

function targets(d: LearnData): number[] {
  if (d.y === undefined) throw new WhiteboxError('value', 'conformal: the data needs targets')
  return Array.from(targetVector(d.y, 'conformal'))
}

function matrixRows(x: LearnData['x']): number[][] {
  const m = matrixView(x, 'conformal')
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}

/** Options of {@link conformalClassification}. */
export interface ConformalClassificationOptions {
  readonly alpha?: number
  /** LAC (default), APS or RAPS scores. */
  readonly score?: ClassificationScore
  /** Calibrate per class (Mondrian), so every class is covered (default false). */
  readonly classConditional?: boolean
  readonly fractions?: readonly [number, number, number]
  readonly key?: Key
}

/**
 * Conformal prediction sets for any probabilistic classifier (MAPIE's `MapieClassifier`): the classes whose
 * conformity score (LAC: one minus the class's probability; APS: the probability mass down to it) is within the
 * calibration rows' quantile, so the set holds the true class with probability at least `1 - alpha`. Sets grow where
 * the model is unsure. Exposition: a `conformal` record of the test rows' sets, their sizes and the coverage, with the
 * calibration rows' LAC scores.
 *
 * @param estimator - A classifier with probabilities.
 * @param data - Labelled rows.
 * @param options - Miscoverage, score, class-conditional calibration, split fractions and key.
 * @returns The sets, coverage and mean size, with the record.
 */
export function conformalClassification(
  estimator: Estimator,
  data: LearnData & { readonly classes?: readonly string[] },
  options: ConformalClassificationOptions = {},
): Explained<{ readonly sets: readonly number[][]; readonly coverage: number; readonly meanSize: number }> {
  const alpha = options.alpha ?? 0.1
  const key = options.key ?? makeKey(0)
  const [fitRows, calRows, test] = splitData(data, options.fractions ?? [0.4, 0.3, 0.3], foldIn(key, 0))
  if (fitRows === undefined || calRows === undefined || test === undefined)
    throw new WhiteboxError('value', 'conformalClassification: no rows')
  const fitted = estimator.fit(fitRows, { key: foldIn(key, 1), explain: false })
  const c = splitConformalClassifier(fitted, calRows, {
    alpha,
    score: options.score ?? 'lac',
    classConditional: options.classConditional ?? false,
    key: foldIn(key, 2),
  })
  const sets = c.predictLabels(test.x)
  const classes = c.classes
  const index = new Map(classes.map((v, i) => [v, i]))
  const truth = targets(test).map((v) => index.get(v) ?? -1)
  const setIdx = sets.map((s) => s.map((v) => index.get(v) ?? -1))
  const covered = setIdx.filter((s, i) => s.includes(truth[i] ?? -2)).length
  const coverage = covered / Math.max(1, sets.length)
  // The calibration rows' LAC scores: one minus the true class's probability.
  const proba = toFlat(requireVerb(fitted, 'predictProba', 'conformalClassification')(calRows.x))
  const k = classes.length
  const scores = targets(calRows).map((v, i) => 1 - (proba[i * k + (index.get(v) ?? 0)] ?? 0))
  const q = c.quantile
  return {
    sets,
    coverage,
    meanSize: sets.reduce((s, v) => s + v.length, 0) / Math.max(1, sets.length),
    exposition: [
      conformalRecord({
        task: 'classification',
        method: `${(options.score ?? 'lac').toUpperCase()}${options.classConditional === true ? ' (by class)' : ''}`,
        alpha,
        coverage,
        scores,
        ...(typeof q === 'number' && (options.score ?? 'lac') === 'lac' ? { quantile: q } : {}),
        classes: classes.map((v) => data.classes?.[v] ?? String(v)),
        sets: setIdx,
        truth,
      }),
    ],
  }
}

/** Options of {@link onlineConformal}. */
export interface OnlineConformalOptions {
  readonly alpha?: number
  /** The step of the level update (default 0.01). */
  readonly gamma?: number
  /** Recent scores each quantile uses (default 60). */
  readonly window?: number
  /** The feature the rows arrive in the order of (default 0). */
  readonly feature?: number
  /** The shares of rows to fit on and calibrate on before the stream (default 0.3 and 0.2; the rest arrives). */
  readonly fractions?: readonly [number, number]
  readonly key?: Key
}

/**
 * Conformal prediction as data arrive (Gibbs and Candès, 2021): a regressor fitted once, then rows arriving in order
 * of a feature (time, or a covariate whose noise drifts). Adaptive conformal inference widens its interval after a
 * miss and narrows it after a hit, `alpha_t+1 = alpha_t + gamma (alpha - err_t)`, using a window of recent scores, so its
 * running coverage stays near the target where a split conformal interval of fixed width drifts away. Exposition: a
 * `conformal` record along the stream with the adaptive intervals and, as its stream, the running coverage of both,
 * the working level and the width.
 *
 * @param estimator - A regressor.
 * @param data - Labelled rows.
 * @param options - Miscoverage, step, window, feature, fractions and key.
 * @returns The coverage of both, with the record.
 */
export function onlineConformal(
  estimator: Estimator,
  data: LearnData,
  options: OnlineConformalOptions = {},
): Explained<{ readonly coverage: number; readonly baselineCoverage: number }> {
  const alpha = options.alpha ?? 0.1
  const key = options.key ?? makeKey(0)
  const [a, b] = options.fractions ?? [0.3, 0.2]
  const [fitRows, calRows, stream] = splitData(data, [a, b, 1 - a - b], foldIn(key, 0))
  if (fitRows === undefined || calRows === undefined || stream === undefined)
    throw new WhiteboxError('value', 'onlineConformal: no rows')
  const fitted = estimator.fit(fitRows, { key: foldIn(key, 1), explain: false })
  const predict = (x: LearnData['x']): number[] =>
    Array.from(toFlat(requireVerb(fitted, 'predict', 'onlineConformal')(x)))
  const pCal = predict(calRows.x)
  const calScores = targets(calRows).map((y, i) => Math.abs(y - (pCal[i] ?? 0)))
  const q = conformalQuantile(calScores, alpha)
  const j = options.feature ?? 0
  const along = columnOf(stream.x, j)
  const order = along.map((_, i) => i).sort((p, r) => (along[p] ?? 0) - (along[r] ?? 0))
  const pAll = predict(stream.x)
  const yAll = targets(stream)
  const t = order.map((i) => along[i] ?? 0)
  const pred = order.map((i) => pAll[i] ?? 0)
  const y = order.map((i) => yAll[i] ?? 0)
  const aci = adaptiveConformalInference(pred, y, {
    alpha,
    gamma: options.gamma ?? 0.01,
    window: options.window ?? 60,
    initialScores: calScores,
  })
  const running = (miss: readonly number[]): number[] => {
    let m = 0
    return miss.map((e, i) => {
      m += e
      return 1 - m / (i + 1)
    })
  }
  const baselineMiss = y.map((v, i) => (Math.abs(v - (pred[i] ?? 0)) > q ? 1 : 0))
  const baselineRunning = running(baselineMiss)
  // A level at or beyond the extremes gives an empty or unbounded interval: clip the drawn edges to the data's span.
  const span = Math.max(...y) - Math.min(...y)
  const clip = (v: number, centre: number): number => Math.max(centre - span, Math.min(centre + span, v))
  const lower = aci.lower.map((v, i) => clip(v, pred[i] ?? 0))
  const upper = aci.upper.map((v, i) => clip(v, pred[i] ?? 0))
  const names = featureNamesOf(stream, matrixView(stream.x, 'onlineConformal').d)
  return {
    coverage: aci.coverage,
    baselineCoverage: baselineRunning[baselineRunning.length - 1] ?? Number.NaN,
    exposition: [
      conformalRecord({
        task: 'regression',
        method: 'adaptive conformal inference',
        alpha,
        coverage: aci.coverage,
        scores: calScores,
        quantile: q,
        x: t,
        xLabel: names[j] ?? `x${j}`,
        y,
        prediction: pred,
        lower,
        upper,
        stream: {
          t,
          coverage: running(aci.errors),
          baseline: baselineRunning,
          alphas: [...aci.alphas],
          width: upper.map((u, i) => u - (lower[i] ?? 0)),
        },
      }),
    ],
  }
}

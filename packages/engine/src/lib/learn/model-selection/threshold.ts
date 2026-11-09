/**
 * Decision-threshold tuning for a binary classifier that gives probabilities or scores: `fixedThresholdClassifier`
 * (scikit-learn's `FixedThresholdClassifier`) cuts the response at a chosen value, and `tunedThresholdClassifierCV`
 * (`TunedThresholdClassifierCV`) chooses the cut that maximises a metric of the predicted labels by cross-validation,
 * as scikit-learn does: each fold scores a grid of thresholds over its own response range, the curves are interpolated
 * onto a common grid and averaged, and the best point wins. The classifier is then refitted on all the rows.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  accuracyScore,
  balancedAccuracyScore,
  f1Score,
  jaccardScore,
  matthewsCorrcoef,
  precisionScore,
  recallScore,
} from '../../metrics/index.ts'
import { interp } from '../../numerics/interpolate/index.ts'
import { foldIn } from '../../random/index.ts'
import { choice, dict, int } from '../../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { innerFitOptions, keyFor } from '../_shared.ts'
import {
  defineEstimator,
  type Estimator,
  type Fitted,
  type LearnData,
  requireTargets,
  requireVerb,
} from '../estimator.ts'
import { encodeLabels, rowsOf, takeData } from '../validation.ts'
import { type CvOption, checkCv, splitInputOf } from './cross-validation.ts'
import { splitIndices } from './splitters.ts'

/** Which response of the classifier is cut: its probabilities, its decision scores, or whichever it has (probabilities first). */
export type ThresholdResponse = 'auto' | 'predictProba' | 'decisionFunction'

/** A metric of the labels a threshold gives, higher better: the true labels, the predicted ones, and the positive label. */
export type ThresholdMetric = (yTrue: readonly number[], yPred: readonly number[], posLabel: number) => number

/**
 * The metrics of predicted labels a tuned threshold can maximise, by scikit-learn's scorer name.
 *
 * @example
 * ```ts
 * import { thresholdMetrics } from 'omai-engine/learn/model-selection'
 * Object.keys(thresholdMetrics) // => ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'jaccard', 'matthews_corrcoef']
 * ```
 */
export const thresholdMetrics: Readonly<Record<string, ThresholdMetric>> = {
  accuracy: (t, p) => accuracyScore([...t], [...p]).estimate.value,
  balanced_accuracy: (t, p) => balancedAccuracyScore([...t], [...p]).estimate.value,
  f1: (t, p, pos) => f1Score([...t], [...p], { posLabel: pos }).estimate.value,
  precision: (t, p, pos) => precisionScore([...t], [...p], { posLabel: pos }).estimate.value,
  recall: (t, p, pos) => recallScore([...t], [...p], { posLabel: pos }).estimate.value,
  jaccard: (t, p, pos) => jaccardScore([...t], [...p], { posLabel: pos }).estimate.value,
  matthews_corrcoef: (t, p) => matthewsCorrcoef([...t], [...p]).estimate.value,
}

/** The response of a fitted binary classifier as one score per row, high meaning the positive class. */
function positiveScores(
  fitted: Fitted,
  x: TensorLike,
  method: ThresholdResponse,
  posLabel: number,
): { readonly scores: Float64Array; readonly method: 'predictProba' | 'decisionFunction' } {
  const classes = fitted.classes ?? []
  const use = method === 'auto' ? (fitted.predictProba !== undefined ? 'predictProba' : 'decisionFunction') : method
  const out = requireVerb(fitted, use, 'thresholdClassifier')(x)
  const flat = toFlat(out)
  const n = rowsOf(x)
  const pos = classes.indexOf(posLabel)
  if (pos < 0) throw new OmaiError('value', `thresholdClassifier: ${posLabel} is not a class of the classifier`)
  if (use === 'predictProba') {
    const k = out.shape[1] ?? 1
    return { scores: Float64Array.from({ length: n }, (_, i) => flat[i * k + pos] ?? 0), method: use }
  }
  // A binary decision score is the positive (second) class's; the other class is its negation.
  return { scores: Float64Array.from(flat, (v) => (pos === 1 ? v : -v)), method: use }
}

const labelsAt = (scores: ArrayLike<number>, threshold: number, pos: number, neg: number): number[] =>
  Array.from(scores, (s) => (s >= threshold ? pos : neg))

const linspace = (lo: number, hi: number, num: number): number[] =>
  num === 1 ? [lo] : Array.from({ length: num }, (_, i) => lo + ((hi - lo) * i) / (num - 1))

/** Hyperparameters of {@link fixedThresholdClassifier}. */
export interface FixedThresholdHyper {
  /** The classifier, fitted on the rows. */
  readonly estimator: Estimator
  /** The cut: a number, or `'auto'` (0.5 for probabilities, 0 for decision scores; default). */
  readonly threshold: number | 'auto'
  /** The positive class (default the larger label). */
  readonly posLabel?: number
  readonly responseMethod: ThresholdResponse
}

/** The fitted parameters of a thresholded classifier: the classes, the cut and the inner classifier's parameters. */
export type ThresholdedParams = {
  readonly classes: number[]
  readonly posLabel: number
  readonly threshold: number
  readonly responseMethod: 'predictProba' | 'decisionFunction'
  /** The inner classifier's parameters (see its `revive`). */
  readonly estimator: Json
  /** Tuned only: the thresholds tried and the cross-validated metric at each. */
  readonly curve?: { readonly thresholds: number[]; readonly scores: number[] }
  /** Tuned only: the best cross-validated metric. */
  readonly bestScore?: number
}

/** The verbs of a thresholded classifier: labels by the cut, and the inner classifier's responses. */
export interface ThresholdedVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly decisionFunction?: (x: TensorLike) => Tensor
  readonly classes: readonly number[]
  readonly nFeaturesIn?: number
}

function thresholdedVerbs(p: ThresholdedParams, inner: Fitted): ThresholdedVerbs {
  const neg = p.classes.find((c) => c !== p.posLabel) ?? p.posLabel
  return {
    classes: p.classes,
    predict: (x) => {
      const { scores } = positiveScores(inner, x, p.responseMethod, p.posLabel)
      return fromStorage(Float64Array.from(labelsAt(scores, p.threshold, p.posLabel, neg)), [scores.length], 'float64')
    },
    ...(inner.predictProba === undefined ? {} : { predictProba: inner.predictProba }),
    ...(inner.decisionFunction === undefined ? {} : { decisionFunction: inner.decisionFunction }),
    ...(inner.nFeaturesIn === undefined ? {} : { nFeaturesIn: inner.nFeaturesIn }),
  }
}

function binaryClasses(data: LearnData, who: string): { readonly classes: number[]; readonly codes: Int32Array } {
  const { classes, codes } = encodeLabels(requireTargets(data, who), who)
  if (classes.length !== 2) throw new OmaiError('value', `${who}: needs two classes, got ${classes.length}`)
  return { classes: [...classes], codes }
}

function nestedWith<H extends { readonly estimator: Estimator }>(
  hyper: H,
  changes: Readonly<Record<string, unknown>>,
): H {
  const { estimator, ...rest } = changes as { readonly estimator?: Estimator } & Readonly<Record<string, unknown>>
  const nested = Object.fromEntries(
    Object.entries(rest)
      .filter(([k]) => k.startsWith('estimator.'))
      .map(([k, v]) => [k.slice('estimator.'.length), v]),
  )
  const own = Object.fromEntries(Object.entries(rest).filter(([k]) => !k.startsWith('estimator.')))
  return { ...hyper, ...own, estimator: (estimator ?? hyper.estimator).with(nested) }
}

/**
 * A classifier cut at a fixed threshold (scikit-learn's `FixedThresholdClassifier`): the wrapped binary classifier is
 * fitted as it is, and `predict` returns the positive class where its probability (or decision score) is at least the
 * threshold. The default `'auto'` is 0.5 for probabilities and 0 for decision scores.
 *
 * @param hyper - The classifier, and the threshold, positive label and response.
 * @returns The estimator.
 * @index estimator Cut a binary classifier's probabilities or scores at a fixed threshold.
 * @example
 * ```ts
 * import { defineEstimator } from 'omai-engine/learn'
 * import { fixedThresholdClassifier } from 'omai-engine/learn/model-selection'
 * import { fromStorage, toFlat } from 'omai-engine/tensor'
 * // A classifier whose probability of class 1 is the feature itself.
 * const identity = defineEstimator(
 *   { name: 'p', kind: 'classifier', fit: () => ({}),
 *     methods: () => ({ classes: [0, 1], predict: (x) => x, predictProba: (x) => fromStorage(Float64Array.from(x.flatMap((r) => [1 - r[0], r[0]])), [x.length, 2], 'float64') }) },
 *   {},
 * )
 * const f = fixedThresholdClassifier({ estimator: identity, threshold: 0.3 }).fit({ x: [[0.1], [0.9]], y: [0, 1] })
 * toFlat(f.predict([[0.2], [0.35], [0.8]])) // => [0, 1, 1]
 * ```
 */
export function fixedThresholdClassifier(
  hyper: Partial<FixedThresholdHyper> & { readonly estimator: Estimator },
): Estimator<FixedThresholdHyper, Fitted<ThresholdedParams> & ThresholdedVerbs> {
  const h: FixedThresholdHyper = {
    estimator: hyper.estimator,
    threshold: hyper.threshold ?? 'auto',
    responseMethod: hyper.responseMethod ?? 'auto',
    ...(hyper.posLabel === undefined ? {} : { posLabel: hyper.posLabel }),
  }
  return defineEstimator<FixedThresholdHyper, ThresholdedParams, ThresholdedVerbs>(
    {
      name: 'fixed-threshold-classifier',
      kind: 'classifier',
      fit: (data, hy, options) => {
        const { classes } = binaryClasses(data, 'fixedThresholdClassifier')
        const posLabel = hy.posLabel ?? (classes[1] as number)
        const inner = hy.estimator.fit(
          data,
          innerFitOptions(options, keyFor(options, 'fixedThreshold'), options.parent, true),
        )
        const response = positiveScores(inner, data.x, hy.responseMethod, posLabel).method
        const threshold = hy.threshold === 'auto' ? (response === 'predictProba' ? 0.5 : 0) : hy.threshold
        return { classes, posLabel, threshold, responseMethod: response, estimator: inner.params }
      },
      methods: (params, hy) => thresholdedVerbs(params, hy.estimator.revive(params.estimator)),
      update: (hy, changes) => nestedWith(hy, changes),
    },
    h,
  )
}

/** Hyperparameters of {@link tunedThresholdClassifierCV}. */
export interface TunedThresholdHyper {
  /** The binary classifier. */
  readonly estimator: Estimator
  /** The metric of the labels to maximise: a name of {@link thresholdMetrics} (default `balanced_accuracy`) or a function. */
  readonly scoring: string | ThresholdMetric
  readonly responseMethod: ThresholdResponse
  /** The number of candidate thresholds (evenly spaced over the response range), or the thresholds themselves (default 100). */
  readonly thresholds: number | readonly number[]
  /** The splitter or number of folds (default 5, stratified). */
  readonly cv?: CvOption
  /** The positive class (default the larger label). */
  readonly posLabel?: number
}

/**
 * Choose the decision threshold that maximises a metric, by cross-validation (scikit-learn's
 * `TunedThresholdClassifierCV`): on each fold the classifier is fitted on the training rows and its response on the
 * held-out rows is cut at `thresholds` evenly spaced values over that fold's own response range, scoring the labels
 * with `scoring`; the fold curves are interpolated onto evenly spaced thresholds over the widest range and averaged, and
 * the best one is kept. The classifier is then refitted on all the rows and cut there. `params.curve` is the averaged
 * metric along the thresholds and `params.bestScore` its maximum.
 *
 * @param hyper - The classifier, and the scoring, response, thresholds, cv and positive label.
 * @returns The estimator.
 * @index estimator Choose a binary classifier's decision threshold by cross-validation to maximise a metric.
 * @example
 * ```ts
 * import { gaussianNaiveBayes } from 'omai-engine/learn'
 * import { tunedThresholdClassifierCV } from 'omai-engine/learn/model-selection'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0.1], [0.3], [0.2], [0.4], [0.25], [0.35], [2.1], [2.3], [2.2], [2.4], [2.25], [2.35]]
 * const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
 * const f = tunedThresholdClassifierCV({ estimator: gaussianNaiveBayes(), cv: 3, scoring: 'accuracy' }).fit({ x, y })
 * f.params.bestScore // => 1
 * toFlat(f.predict([[0.2], [2.2]])) // => [0, 1]
 * ```
 */
export function tunedThresholdClassifierCV(
  hyper: Partial<TunedThresholdHyper> & { readonly estimator: Estimator },
): Estimator<TunedThresholdHyper, Fitted<ThresholdedParams> & ThresholdedVerbs> {
  const h: TunedThresholdHyper = {
    estimator: hyper.estimator,
    scoring: hyper.scoring ?? 'balanced_accuracy',
    responseMethod: hyper.responseMethod ?? 'auto',
    thresholds: hyper.thresholds ?? 100,
    ...(hyper.cv === undefined ? {} : { cv: hyper.cv }),
    ...(hyper.posLabel === undefined ? {} : { posLabel: hyper.posLabel }),
  }
  const metricOf = (s: string | ThresholdMetric): ThresholdMetric => {
    if (typeof s === 'function') return s
    const m = thresholdMetrics[s]
    if (m === undefined)
      throw new OmaiError(
        'value',
        `tunedThresholdClassifierCV: unknown scoring ${s}; one of ${Object.keys(thresholdMetrics).join(', ')}`,
      )
    return m
  }
  return defineEstimator<TunedThresholdHyper, ThresholdedParams, ThresholdedVerbs>(
    {
      name: 'tuned-threshold-classifier-cv',
      kind: 'classifier',
      hyperSpace: (current) =>
        dict({
          ...(typeof current.thresholds === 'number'
            ? { thresholds: int(10, 500, { default: current.thresholds, label: 'thresholds' }) }
            : {}),
          scoring: choice(Object.keys(thresholdMetrics) as [string, ...string[]], {
            default: typeof current.scoring === 'string' ? current.scoring : 'balanced_accuracy',
          }),
        }),
      fit: (data, hy, options) => {
        const who = 'tunedThresholdClassifierCV'
        const { classes, codes } = binaryClasses(data, who)
        const posLabel = hy.posLabel ?? (classes[1] as number)
        const negLabel = classes.find((c) => c !== posLabel) ?? posLabel
        const metric = metricOf(hy.scoring)
        const key = keyFor(options, who)
        const splitter = checkCv(hy.cv, { kind: 'classifier' })
        const folds = splitIndices(splitter, splitInputOf(data), foldIn(key, 0))
        const y = Array.from(codes, (c) => classes[c] ?? 0)
        const perFold = folds.map((fold, f) => {
          const fitted = hy.estimator.fit(
            takeData(data, fold.train),
            innerFitOptions(options, foldIn(key, 1 + f), options.parent),
          )
          const test = takeData(data, fold.test)
          const { scores } = positiveScores(fitted, test.x, hy.responseMethod, posLabel)
          const truth = Array.from(fold.test, (i) => y[i] ?? 0)
          const own =
            typeof hy.thresholds === 'number'
              ? linspace(Math.min(...scores), Math.max(...scores), hy.thresholds)
              : [...hy.thresholds]
          return {
            thresholds: own,
            scores: own.map((t) => metric(truth, labelsAt(scores, t, posLabel, negLabel), posLabel)),
          }
        })
        const grid =
          typeof hy.thresholds === 'number'
            ? linspace(
                Math.min(...perFold.map((f) => f.thresholds[0] ?? 0)),
                Math.max(...perFold.map((f) => f.thresholds[f.thresholds.length - 1] ?? 0)),
                hy.thresholds,
              )
            : [...hy.thresholds]
        const mean = grid.map(() => 0)
        for (const f of perFold) {
          const v = toFlat(interp(grid, f.thresholds, f.scores))
          for (const [i, s] of v.entries()) mean[i] = (mean[i] ?? 0) + s / perFold.length
        }
        let best = 0
        for (const [i, s] of mean.entries()) if (s > (mean[best] ?? Number.NEGATIVE_INFINITY)) best = i
        const refit = hy.estimator.fit(
          data,
          innerFitOptions(options, foldIn(key, 1 + folds.length), options.parent, true),
        )
        const response = positiveScores(refit, data.x, hy.responseMethod, posLabel).method
        return {
          classes,
          posLabel,
          threshold: grid[best] ?? 0,
          responseMethod: response,
          estimator: refit.params,
          curve: { thresholds: grid, scores: mean },
          bestScore: mean[best] ?? Number.NaN,
        }
      },
      methods: (params, hy) => thresholdedVerbs(params, hy.estimator.revive(params.estimator)),
      update: (hy, changes) => nestedWith(hy, changes),
    },
    h,
  )
}

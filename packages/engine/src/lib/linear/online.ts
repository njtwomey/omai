/**
 * Online linear learners that look at one example per step: the perceptron family (plain, averaged, voted and pocket,
 * and the kernel perceptron in the dual) and passive-aggressive learners (PA, PA-I and PA-II for classification, and
 * the epsilon-insensitive regressor). Each is an `Algorithm` over the examples in order (or in a fresh random order
 * each epoch) logging `mistakes` per step, and an estimator on the iterative base, so a session shows the boundary
 * moving mistake by mistake and `fittedAt(t)` is the model after `t` examples. Classifiers are binary (the class with
 * the larger label is the positive one); a multiclass problem wraps them in `oneVsRest`.
 */

import { OmaiError } from '../base/index.ts'
import {
  type Kernel,
  type PairwiseKernelMetric,
  type PairwiseKernelOptions,
  pairwiseKernels,
} from '../kernels/index.ts'
import {
  encodeLabels,
  type IterativeEstimator,
  iterative,
  type LearnData,
  type MatrixView,
  matrixView,
  rowsOf,
  targetVector,
} from '../learn/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'

/** The variants of the perceptron. */
export type PerceptronVariant = 'plain' | 'averaged' | 'voted' | 'pocket'

/** The hyperparameters of {@link perceptron}. */
export interface PerceptronHyper {
  readonly variant: PerceptronVariant
  /** The most passes over the data (a pass without a mistake ends the run sooner). */
  readonly maxEpochs: number
  /** A fresh random order each pass (from the run's key) rather than the data's order. */
  readonly shuffle: boolean
  readonly fitIntercept: boolean
}

/** One weight vector of a voted perceptron and the number of examples it survived. */
export type Vote = { readonly w: number[]; readonly b: number; readonly count: number }

/** The state of a perceptron run. */
export type PerceptronState = {
  readonly w: number[]
  readonly b: number
  /** Examples looked at. */
  readonly t: number
  /** Cumulative mistakes (updates). */
  readonly mistakes: number
  readonly epochMistakes: number
  /** Mistakes in the last completed pass, `-1` before the first completes. */
  readonly lastEpochMistakes: number
  /** Averaged: the running mean of the weights and intercept over all steps. */
  readonly avgW: number[]
  readonly avgB: number
  /** Voted: every weight vector so far with its survival count (the last is the current one). */
  readonly votes: Vote[]
  /** Pocket: the consecutive examples the current weights got right. */
  readonly run: number
  /** Pocket: the best weights so far, by training errors, with the run that earned them. */
  readonly pocket: { readonly w: number[]; readonly b: number; readonly run: number; readonly errors: number }
}

/** The fitted parameters of a perceptron or passive-aggressive classifier: plain JSON. */
export type LinearOnlineParams = {
  readonly classes: number[]
  readonly coef: number[]
  readonly intercept: number
  readonly mistakes: number
  readonly examples: number
  /** Voted: the weight vectors with their counts (the prediction is their weighted vote). */
  readonly votes: Vote[]
}

/** What a fitted online linear classifier offers. */
export interface LinearOnlineVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  /** The margin `w.x + b` (a voted perceptron: the share of votes for the positive class, minus the negative's). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

/** The training rows with labels as +1 and -1 (the larger label positive), and the classes. */
function binaryProblem(
  data: LearnData,
  who: string,
): { readonly x: MatrixView; readonly y: Int8Array; readonly classes: readonly number[] } {
  const x = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, x.n)
  if (classes.length > 2) {
    throw new OmaiError('value', `${who}: binary labels only (found ${classes.length} classes); wrap it in oneVsRest`)
  }
  const classes2 = classes.length === 2 ? classes : [classes[0] ?? 0, (classes[0] ?? 0) + 1]
  return { x, y: Int8Array.from(codes, (c) => (c === 1 ? 1 : -1)), classes: classes2 }
}

/** The index of the example looked at in step `t`: the data's order, or a fresh permutation each pass. */
function scheduler(n: number, shuffle: boolean, key: Key): (t: number) => number {
  const cache = new Map<number, Int32Array>()
  return (t) => {
    const k = t % n
    if (!shuffle) return k
    const epoch = Math.floor(t / n)
    let perm = cache.get(epoch)
    if (perm === undefined) {
      perm = Int32Array.from(toFlat(permutation(foldIn(key, epoch), n)))
      cache.clear()
      cache.set(epoch, perm)
    }
    return perm[k] ?? k
  }
}

const dotRow = (w: readonly number[], x: MatrixView, i: number): number =>
  ddot(w, x.data.subarray(i * x.d, (i + 1) * x.d))

const addRow = (w: readonly number[], x: MatrixView, i: number, scale: number): number[] =>
  w.map((v, j) => v + scale * (x.data[i * x.d + j] ?? 0))

function trainingErrors(w: readonly number[], b: number, x: MatrixView, y: Int8Array): number {
  let errors = 0
  for (let i = 0; i < x.n; i++) if ((y[i] ?? 0) * (dotRow(w, x, i) + b) <= 0) errors++
  return errors
}

/**
 * The perceptron as an algorithm that looks at one example per step. On a mistake (`y (w.x + b) <= 0`) the weights move
 * by `y x`. The variants differ in the model they keep: `plain` the last weights; `averaged` the mean of the weights
 * over all steps (Freund and Schapire's cheap approximation of the voted perceptron, scikit-learn's `average=True`);
 * `voted` every weight vector with the number of examples it survived, predicting by their weighted vote; `pocket`
 * with ratchet (Gallant) the weights with the fewest training errors seen, so a non-separable problem keeps its best
 * line rather than the last. It logs the metrics `mistakes`, `epoch` and the variant's own (`votes`,
 * `pocketErrors`), and the tensor `weights`; a pass without a mistake ends the run.
 *
 * @param data - Features and binary labels.
 * @param hyper - The variant, the pass limit, shuffling, the intercept.
 * @param key - Orders the passes when `shuffle`.
 * @returns The algorithm.
 * @reference F. Rosenblatt (1958); Y. Freund and R. Schapire, Large margin classification using the perceptron
 * algorithm, Machine Learning 37 (1999); S. Gallant, Perceptron-based learning algorithms, IEEE Trans. Neural
 * Networks 1 (1990).
 * @example
 * ```ts
 * import { perceptronAlgorithm } from 'omai-engine/linear'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const hyper = { variant: 'plain', maxEpochs: 10, shuffle: false, fitIntercept: true } as const
 * const alg = perceptronAlgorithm({ x: [[0], [1], [3], [4]], y: [0, 0, 1, 1] }, hyper, key(0))
 * runAlgorithm(alg, { maxSteps: 40 }).state.mistakes // => 8
 * ```
 */
export function perceptronAlgorithm(data: LearnData, hyper: PerceptronHyper, key: Key): Algorithm<PerceptronState> {
  const { x, y } = binaryProblem(data, 'perceptron')
  const { n, d } = x
  const at = scheduler(n, hyper.shuffle, key)
  const zeros = new Array<number>(d).fill(0)
  const start: PerceptronState = {
    w: zeros,
    b: 0,
    t: 0,
    mistakes: 0,
    epochMistakes: 0,
    lastEpochMistakes: -1,
    avgW: zeros,
    avgB: 0,
    votes: hyper.variant === 'voted' ? [{ w: zeros, b: 0, count: 0 }] : [],
    run: 0,
    pocket: { w: zeros, b: 0, run: 0, errors: hyper.variant === 'pocket' ? n : 0 },
  }
  const channelsOf = channels({
    metrics: {
      mistakes: { description: 'examples classified wrongly so far (each moved the weights)', direction: 'lower' },
      epoch: { description: 'passes over the data completed' },
      ...(hyper.variant === 'voted' ? { votes: { description: 'weight vectors kept, one per mistake' } } : {}),
      ...(hyper.variant === 'pocket'
        ? { pocketErrors: { description: 'training errors of the pocket weights', direction: 'lower' as const } }
        : {}),
    },
    tensors: { weights: { role: 'param', description: 'the weights shown by the model', shape: [d] } },
  })
  return {
    name: `perceptron-${hyper.variant}`,
    config: { variant: hyper.variant, maxEpochs: hyper.maxEpochs, shuffle: hyper.shuffle },
    schema: channelsOf,
    init: () => start,
    step: (s) => {
      const i = at(s.t)
      const yi = y[i] ?? 1
      const margin = yi * (dotRow(s.w, x, i) + s.b)
      const wrong = margin <= 0
      const w = wrong ? addRow(s.w, x, i, yi) : s.w
      const b = wrong && hyper.fitIntercept ? s.b + yi : s.b
      const t = s.t + 1
      const epochMistakes = s.epochMistakes + (wrong ? 1 : 0)
      const endOfPass = t % n === 0
      const next: PerceptronState = {
        ...s,
        w,
        b,
        t,
        mistakes: s.mistakes + (wrong ? 1 : 0),
        epochMistakes: endOfPass ? 0 : epochMistakes,
        lastEpochMistakes: endOfPass ? epochMistakes : s.lastEpochMistakes,
      }
      switch (hyper.variant) {
        case 'averaged':
          return {
            ...next,
            avgW: s.avgW.map((v, j) => v + ((w[j] ?? 0) - v) / t),
            avgB: s.avgB + (b - s.avgB) / t,
          }
        case 'voted': {
          const votes = s.votes
          if (!wrong)
            return { ...next, votes: votes.map((v, k) => (k === votes.length - 1 ? { ...v, count: v.count + 1 } : v)) }
          return { ...next, votes: [...votes, { w, b, count: 1 }] }
        }
        case 'pocket': {
          if (wrong) return { ...next, run: 0 }
          const run = s.run + 1
          if (run > s.pocket.run) {
            const errors = trainingErrors(w, b, x, y)
            if (errors < s.pocket.errors) return { ...next, run, pocket: { w, b, run, errors } }
          }
          return { ...next, run }
        }
        default:
          return next
      }
    },
    done: (s) =>
      s.lastEpochMistakes === 0 || (hyper.variant === 'pocket' && s.pocket.errors === 0) ? 'converged' : undefined,
    observe: (s) => ({
      metrics: {
        mistakes: s.mistakes,
        epoch: Math.floor(s.t / n),
        ...(hyper.variant === 'voted' ? { votes: s.votes.length } : {}),
        ...(hyper.variant === 'pocket' ? { pocketErrors: s.pocket.errors } : {}),
      },
      tensors: { weights: { role: 'param', value: modelOf(s, hyper.variant).w, shape: [d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PerceptronState,
  }
}

function modelOf(s: PerceptronState, variant: PerceptronVariant): { readonly w: number[]; readonly b: number } {
  if (variant === 'averaged') return { w: s.avgW, b: s.avgB }
  if (variant === 'pocket') return { w: s.pocket.w, b: s.pocket.b }
  return { w: s.w, b: s.b }
}

function onlineVerbs(p: LinearOnlineParams, who: string): LinearOnlineVerbs {
  const d = p.coef.length
  const margins = (x: TensorLike): Float64Array => {
    const m = matrixView(x, who, { nFeatures: d })
    if (p.votes.length > 0) {
      const total = p.votes.reduce((s, v) => s + v.count, 0)
      return Float64Array.from({ length: m.n }, (_, i) => {
        let sum = 0
        for (const v of p.votes) sum += v.count * (dotRow(v.w, m, i) + v.b > 0 ? 1 : -1)
        return total === 0 ? 0 : sum / total
      })
    }
    return Float64Array.from({ length: m.n }, (_, i) => dotRow(p.coef, m, i) + p.intercept)
  }
  const predict = (x: TensorLike): Tensor =>
    fromStorage(
      Float64Array.from(margins(x), (v) => p.classes[v > 0 ? 1 : 0] ?? Number.NaN),
      [margins(x).length],
      'float64',
    )
  return {
    classes: p.classes,
    nFeaturesIn: d,
    predict,
    decisionFunction: (x) => {
      const mg = margins(x)
      return fromStorage(mg, [mg.length], 'float64')
    },
    featureImportances: p.coef.map(Math.abs),
    score: (data) => {
      const pred = toFlat(predict(data.x))
      const truth = targetVector(data.y ?? [], who, pred.length)
      let hits = 0
      for (let i = 0; i < pred.length; i++) if (pred[i] === truth[i]) hits++
      return hits / pred.length
    },
  }
}

const PERCEPTRON_DEFAULTS: PerceptronHyper = { variant: 'plain', maxEpochs: 20, shuffle: false, fitIntercept: true }

/**
 * The perceptron family (`sklearn.linear_model.Perceptron` for the plain variant with `shuffle: false`, and
 * `SGDClassifier(loss='perceptron', average=True)` for the averaged), fitted by {@link perceptronAlgorithm} one example
 * per step: a session shows the boundary moving mistake by mistake, and `fittedAt(t)` is the model after `t`
 * examples. Binary; wrap in `oneVsRest` for more classes.
 *
 * @param hyper - `variant` (`plain`, `averaged`, `voted`, `pocket`), `maxEpochs`, `shuffle`, `fitIntercept`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { perceptron } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = perceptron({ variant: 'averaged' }).fit({ x: [[0], [1], [3], [4]], y: [0, 0, 1, 1] })
 * toFlat(f.predict([[0.5], [3.5]])) // => [0, 1]
 * ```
 */
export function perceptron(
  hyper: Partial<PerceptronHyper> = {},
): IterativeEstimator<PerceptronHyper, PerceptronState, LinearOnlineParams, LinearOnlineVerbs> {
  return iterative<PerceptronHyper, PerceptronState, LinearOnlineParams, LinearOnlineVerbs>(
    {
      name: 'perceptron',
      kind: 'classifier',
      defaults: PERCEPTRON_DEFAULTS,
      algorithm: (data, h, key) => perceptronAlgorithm(data, h, key),
      maxSteps: (h, data) => h.maxEpochs * rowsOf(data.x),
      params: (s, data, h) => {
        const { classes } = binaryProblem(data, 'perceptron')
        const m = modelOf(s, h.variant)
        return {
          classes: [...classes],
          coef: m.w,
          intercept: m.b,
          mistakes: s.mistakes,
          examples: s.t,
          votes: h.variant === 'voted' ? s.votes.filter((v) => v.count > 0) : [],
        }
      },
      verbs: (p) => onlineVerbs(p, 'perceptron'),
      logs: { 'solver/mistakes': Log(), 'solver/weights': Log({ detail: 'full' }) },
    },
    { ...PERCEPTRON_DEFAULTS, ...hyper },
  )
}

/** The hyperparameters of {@link kernelPerceptron}. */
export interface KernelPerceptronHyper {
  /** A kernel of `omai-engine/kernels`, or a scikit-learn metric name with its parameters. */
  readonly kernel: Kernel | PairwiseKernelMetric
  readonly kernelOptions: PairwiseKernelOptions
  readonly maxEpochs: number
  readonly shuffle: boolean
}

/** The state of the kernel perceptron: the mistake count of every training example. */
export type KernelPerceptronState = {
  /** `alpha_i`: how often example `i` was a mistake. */
  readonly alpha: number[]
  /** The intercept `sum alpha_i y_i`. */
  readonly b: number
  readonly t: number
  readonly mistakes: number
  readonly epochMistakes: number
  readonly lastEpochMistakes: number
}

/** The fitted parameters of the kernel perceptron: the support rows and their signed counts. */
export type KernelPerceptronParams = {
  readonly classes: number[]
  /** Training rows that were ever a mistake `[s][d]`. */
  readonly support: number[][]
  /** `alpha_i y_i` for each. */
  readonly coef: number[]
  readonly intercept: number
  readonly mistakes: number
  readonly examples: number
  readonly nFeatures: number
}

/**
 * The kernel perceptron as an algorithm: the perceptron in the dual (Aizerman et al.; Freund and Schapire), the
 * weights being `sum_i alpha_i y_i phi(x_i)` and the score of an example `sum_j alpha_j y_j k(x_j, x_i) + b`; a mistake
 * adds one to the example's count. It logs `mistakes`, `epoch` and `support` (examples with a nonzero count), and the
 * tensor `alpha`.
 *
 * @param data - Features and binary labels.
 * @param hyper - The kernel, its options, the pass limit and shuffling.
 * @param key - Orders the passes when `shuffle`.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { kernelPerceptronAlgorithm } from 'omai-engine/linear'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // XOR is not linearly separable, but an RBF kernel perceptron learns it.
 * const x = [[0, 0], [1, 1], [0, 1], [1, 0]]
 * const hyper = { kernel: 'rbf', kernelOptions: { gamma: 2 }, maxEpochs: 50, shuffle: false } as const
 * const alg = kernelPerceptronAlgorithm({ x, y: [0, 0, 1, 1] }, hyper, key(0))
 * runAlgorithm(alg, { maxSteps: 200 }).state.lastEpochMistakes // => 0
 * ```
 */
export function kernelPerceptronAlgorithm(
  data: LearnData,
  hyper: KernelPerceptronHyper,
  key: Key,
): Algorithm<KernelPerceptronState> {
  const { x, y } = binaryProblem(data, 'kernelPerceptron')
  const { n } = x
  const gram = Float64Array.from(
    toFlat(
      primalOf(pairwiseKernels(fromStorage(x.data, [n, x.d], 'float64'), undefined, hyper.kernel, hyper.kernelOptions)),
    ),
  )
  const at = scheduler(n, hyper.shuffle, key)
  return {
    name: 'kernel-perceptron',
    config: { maxEpochs: hyper.maxEpochs, shuffle: hyper.shuffle },
    schema: channels({
      metrics: {
        mistakes: { description: 'examples classified wrongly so far', direction: 'lower' },
        epoch: { description: 'passes over the data completed' },
        support: { description: 'examples with a nonzero mistake count' },
      },
      tensors: { alpha: { role: 'param', description: 'the mistake count of every example', shape: [n] } },
    }),
    init: () => ({
      alpha: new Array<number>(n).fill(0),
      b: 0,
      t: 0,
      mistakes: 0,
      epochMistakes: 0,
      lastEpochMistakes: -1,
    }),
    step: (s) => {
      const i = at(s.t)
      const yi = y[i] ?? 1
      let score = s.b
      for (let j = 0; j < n; j++)
        if ((s.alpha[j] ?? 0) !== 0) score += (s.alpha[j] ?? 0) * (y[j] ?? 0) * (gram[j * n + i] ?? 0)
      const wrong = yi * score <= 0
      const t = s.t + 1
      const epochMistakes = s.epochMistakes + (wrong ? 1 : 0)
      const endOfPass = t % n === 0
      return {
        alpha: wrong ? s.alpha.map((a, j) => (j === i ? a + 1 : a)) : s.alpha,
        b: wrong ? s.b + yi : s.b,
        t,
        mistakes: s.mistakes + (wrong ? 1 : 0),
        epochMistakes: endOfPass ? 0 : epochMistakes,
        lastEpochMistakes: endOfPass ? epochMistakes : s.lastEpochMistakes,
      }
    },
    done: (s) => (s.lastEpochMistakes === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { mistakes: s.mistakes, epoch: Math.floor(s.t / n), support: s.alpha.filter((a) => a !== 0).length },
      tensors: { alpha: { role: 'param', value: s.alpha, shape: [n] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as KernelPerceptronState,
  }
}

/** What a fitted kernel perceptron offers. */
export interface KernelPerceptronVerbs {
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
}

const KERNEL_PERCEPTRON_DEFAULTS: KernelPerceptronHyper = {
  kernel: 'rbf',
  kernelOptions: {},
  maxEpochs: 50,
  shuffle: false,
}

/**
 * The kernel perceptron: a perceptron whose boundary lives in a kernel's feature space, fitted by
 * {@link kernelPerceptronAlgorithm} one example per step. The model is the support examples and their signed mistake
 * counts. Binary.
 *
 * @param hyper - `kernel`, `kernelOptions`, `maxEpochs`, `shuffle`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { kernelPerceptron } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0, 0], [1, 1], [0, 1], [1, 0]]
 * const f = kernelPerceptron({ kernelOptions: { gamma: 2 } }).fit({ x, y: [0, 0, 1, 1] })
 * toFlat(f.predict(x)) // => [0, 0, 1, 1]
 * ```
 */
export function kernelPerceptron(
  hyper: Partial<KernelPerceptronHyper> = {},
): IterativeEstimator<KernelPerceptronHyper, KernelPerceptronState, KernelPerceptronParams, KernelPerceptronVerbs> {
  return iterative<KernelPerceptronHyper, KernelPerceptronState, KernelPerceptronParams, KernelPerceptronVerbs>(
    {
      name: 'kernel-perceptron',
      kind: 'classifier',
      defaults: KERNEL_PERCEPTRON_DEFAULTS,
      algorithm: (data, h, key) => kernelPerceptronAlgorithm(data, h, key),
      maxSteps: (h, data) => h.maxEpochs * rowsOf(data.x),
      params: (s, data) => {
        const { x, y, classes } = binaryProblem(data, 'kernelPerceptron')
        const support = s.alpha.map((a, i) => (a !== 0 ? i : -1)).filter((i) => i >= 0)
        return {
          classes: [...classes],
          support: support.map((i) => Array.from(x.data.subarray(i * x.d, (i + 1) * x.d))),
          coef: support.map((i) => (s.alpha[i] ?? 0) * (y[i] ?? 0)),
          intercept: s.b,
          mistakes: s.mistakes,
          examples: s.t,
          nFeatures: x.d,
        }
      },
      verbs: (p, h) => {
        const margins = (xs: TensorLike): Float64Array => {
          const m = matrixView(xs, 'kernelPerceptron', { nFeatures: p.nFeatures })
          if (p.support.length === 0) return new Float64Array(m.n).fill(p.intercept)
          const k = toFlat(
            primalOf(
              pairwiseKernels(
                fromStorage(m.data, [m.n, m.d], 'float64'),
                fromStorage(Float64Array.from(p.support.flat()), [p.support.length, m.d], 'float64'),
                h.kernel,
                h.kernelOptions,
              ),
            ),
          )
          return Float64Array.from({ length: m.n }, (_, i) => {
            let s = p.intercept
            for (let j = 0; j < p.support.length; j++) s += (p.coef[j] ?? 0) * (k[i * p.support.length + j] ?? 0)
            return s
          })
        }
        const predict = (xs: TensorLike): Tensor => {
          const mg = margins(xs)
          return fromStorage(
            Float64Array.from(mg, (v) => p.classes[v > 0 ? 1 : 0] ?? Number.NaN),
            [mg.length],
            'float64',
          )
        }
        return {
          classes: p.classes,
          nFeaturesIn: p.nFeatures,
          predict,
          decisionFunction: (xs) => {
            const mg = margins(xs)
            return fromStorage(mg, [mg.length], 'float64')
          },
          score: (data) => {
            const pred = toFlat(predict(data.x))
            const truth = targetVector(data.y ?? [], 'kernelPerceptron', pred.length)
            let hits = 0
            for (let i = 0; i < pred.length; i++) if (pred[i] === truth[i]) hits++
            return hits / pred.length
          },
        }
      },
      logs: { 'solver/mistakes': Log(), 'solver/support': Log(), 'solver/alpha': Log({ detail: 'full' }) },
    },
    { ...KERNEL_PERCEPTRON_DEFAULTS, ...hyper },
  )
}

/** The passive-aggressive variants: PA (no cap), PA-I (step capped at `C`) and PA-II (a softened denominator). */
export type PassiveAggressiveVariant = 'pa' | 'pa1' | 'pa2'

/** The hyperparameters of {@link passiveAggressive} and {@link passiveAggressiveRegressor}. */
export interface PassiveAggressiveHyper {
  readonly variant: PassiveAggressiveVariant
  /** The aggressiveness bound `C` of PA-I and PA-II. */
  readonly c: number
  /** The width of the insensitive tube (regression). */
  readonly epsilon: number
  readonly maxEpochs: number
  readonly shuffle: boolean
  readonly fitIntercept: boolean
}

/** The state of a passive-aggressive run. */
export type PassiveAggressiveState = {
  readonly w: number[]
  readonly b: number
  readonly t: number
  /** Cumulative misclassified examples (classification), or examples outside the tube (regression). */
  readonly mistakes: number
  /** Cumulative loss (hinge, or epsilon-insensitive). */
  readonly loss: number
  /** Updates made (a loss above zero). */
  readonly updates: number
  readonly epochUpdates: number
  readonly lastEpochUpdates: number
}

function passiveStep(
  s: PassiveAggressiveState,
  h: PassiveAggressiveHyper,
  x: MatrixView,
  i: number,
  target: number,
  regression: boolean,
  n: number,
): PassiveAggressiveState {
  const p = dotRow(s.w, x, i) + s.b
  const loss = regression ? Math.max(0, Math.abs(target - p) - h.epsilon) : Math.max(0, 1 - target * p)
  const row = x.data.subarray(i * x.d, (i + 1) * x.d)
  const sq = ddot(row, row)
  const wrong = regression ? loss > 0 : target * p <= 0
  const t = s.t + 1
  let update = 0
  if (loss > 0 && sq > 0) {
    update = h.variant === 'pa' ? loss / sq : h.variant === 'pa1' ? Math.min(h.c, loss / sq) : loss / (sq + 0.5 / h.c)
    update *= regression ? Math.sign(target - p) : target
  }
  const endOfPass = t % n === 0
  const epochUpdates = s.epochUpdates + (loss > 0 ? 1 : 0)
  return {
    w: update === 0 ? s.w : addRow(s.w, x, i, update),
    b: update !== 0 && h.fitIntercept ? s.b + update : s.b,
    t,
    mistakes: s.mistakes + (wrong ? 1 : 0),
    loss: s.loss + loss,
    updates: s.updates + (update === 0 ? 0 : 1),
    epochUpdates: endOfPass ? 0 : epochUpdates,
    lastEpochUpdates: endOfPass ? epochUpdates : s.lastEpochUpdates,
  }
}

function passiveAlgorithm(
  name: string,
  rows: MatrixView,
  targets: ArrayLike<number>,
  h: PassiveAggressiveHyper,
  key: Key,
  regression: boolean,
): Algorithm<PassiveAggressiveState> {
  const n = rows.n
  const at = scheduler(n, h.shuffle, key)
  return {
    name,
    config: { variant: h.variant, c: h.c, epsilon: h.epsilon, maxEpochs: h.maxEpochs },
    schema: channels({
      metrics: {
        mistakes: {
          description: regression ? 'examples outside the tube so far' : 'examples classified wrongly so far',
          direction: 'lower',
        },
        loss: {
          description: regression ? 'cumulative epsilon-insensitive loss' : 'cumulative hinge loss',
          direction: 'lower',
        },
        updates: { description: 'updates made so far (examples with a loss)' },
        epoch: { description: 'passes over the data completed' },
      },
      tensors: { weights: { role: 'param', description: 'the weights', shape: [rows.d] } },
    }),
    init: () => ({
      w: new Array<number>(rows.d).fill(0),
      b: 0,
      t: 0,
      mistakes: 0,
      loss: 0,
      updates: 0,
      epochUpdates: 0,
      lastEpochUpdates: -1,
    }),
    step: (s) => {
      const i = at(s.t)
      return passiveStep(s, h, rows, i, targets[i] ?? 0, regression, n)
    },
    done: (s) => (s.lastEpochUpdates === 0 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { mistakes: s.mistakes, loss: s.loss, updates: s.updates, epoch: Math.floor(s.t / n) },
      tensors: { weights: { role: 'param', value: s.w, shape: [rows.d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PassiveAggressiveState,
  }
}

/**
 * The passive-aggressive classifier as an algorithm (Crammer et al.): for each example, if the hinge loss
 * `max(0, 1 - y (w.x + b))` is zero keep the weights (passive), else move them just enough to satisfy the margin
 * (aggressive): `w += tau y x` with `tau = loss / |x|^2` (PA), `min(C, loss / |x|^2)` (PA-I) or
 * `loss / (|x|^2 + 1 / 2C)` (PA-II). It logs `mistakes`, `loss`, `updates` and `epoch`, and the tensor `weights`.
 *
 * @param data - Features and binary labels.
 * @param hyper - The variant, `c`, the pass limit, shuffling, the intercept.
 * @param key - Orders the passes when `shuffle`.
 * @returns The algorithm.
 * @reference K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz and Y. Singer, Online passive-aggressive
 * algorithms, JMLR 7 (2006).
 * @example
 * ```ts
 * import { passiveAggressiveAlgorithm } from 'omai-engine/linear'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const hyper = { variant: 'pa1', c: 1, epsilon: 0.1, maxEpochs: 5, shuffle: false, fitIntercept: true } as const
 * const alg = passiveAggressiveAlgorithm({ x: [[0], [1], [3], [4]], y: [0, 0, 1, 1] }, hyper, key(0))
 * runAlgorithm(alg, { maxSteps: 20 }).state.mistakes // => 7
 * ```
 */
export function passiveAggressiveAlgorithm(
  data: LearnData,
  hyper: PassiveAggressiveHyper,
  key: Key,
): Algorithm<PassiveAggressiveState> {
  const { x, y } = binaryProblem(data, 'passiveAggressive')
  return passiveAlgorithm('passive-aggressive', x, y, hyper, key, false)
}

const PA_DEFAULTS: PassiveAggressiveHyper = {
  variant: 'pa1',
  c: 1,
  epsilon: 0.1,
  maxEpochs: 20,
  shuffle: false,
  fitIntercept: true,
}

/**
 * The passive-aggressive classifier (`SGDClassifier(learning_rate='pa1')`, scikit-learn's deprecated
 * `PassiveAggressiveClassifier`), fitted by {@link passiveAggressiveAlgorithm} one example per step. Binary.
 *
 * @param hyper - `variant` (`pa`, `pa1`, `pa2`), `c`, `maxEpochs`, `shuffle`, `fitIntercept`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { passiveAggressive } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = passiveAggressive({ variant: 'pa2', c: 0.5 }).fit({ x: [[0], [1], [3], [4]], y: [0, 0, 1, 1] })
 * toFlat(f.predict([[0.5], [3.5]])) // => [0, 1]
 * ```
 */
export function passiveAggressive(
  hyper: Partial<PassiveAggressiveHyper> = {},
): IterativeEstimator<PassiveAggressiveHyper, PassiveAggressiveState, LinearOnlineParams, LinearOnlineVerbs> {
  return iterative<PassiveAggressiveHyper, PassiveAggressiveState, LinearOnlineParams, LinearOnlineVerbs>(
    {
      name: 'passive-aggressive',
      kind: 'classifier',
      defaults: PA_DEFAULTS,
      hyperSpace: (h) => dict({ c: real(Math.min(1e-3, h.c), Math.max(100, h.c), { default: h.c, log: true }) }),
      algorithm: (data, h, key) => passiveAggressiveAlgorithm(data, h, key),
      maxSteps: (h, data) => h.maxEpochs * rowsOf(data.x),
      params: (s, data) => ({
        classes: [...binaryProblem(data, 'passiveAggressive').classes],
        coef: s.w,
        intercept: s.b,
        mistakes: s.mistakes,
        examples: s.t,
        votes: [],
      }),
      verbs: (p) => onlineVerbs(p, 'passiveAggressive'),
      logs: { 'solver/mistakes': Log(), 'solver/loss': Log(), 'solver/weights': Log({ detail: 'full' }) },
    },
    { ...PA_DEFAULTS, ...hyper },
  )
}

/** The fitted parameters of the passive-aggressive regressor: plain JSON. */
export type PassiveRegressorParams = {
  readonly coef: number[]
  readonly intercept: number
  readonly mistakes: number
  readonly examples: number
}

/** What a fitted passive-aggressive regressor offers. */
export interface PassiveRegressorVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

/**
 * The passive-aggressive regressor (`SGDRegressor(loss='epsilon_insensitive', learning_rate='pa1')`): an example inside
 * the epsilon tube is left alone, one outside moves the weights just enough to bring it to the tube's edge (capped
 * by `C` for PA-I, softened for PA-II). One example per step, logging `mistakes` (outside the tube), `loss` and
 * `updates`.
 *
 * @param hyper - `variant`, `c`, `epsilon`, `maxEpochs`, `shuffle`, `fitIntercept`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { passiveAggressiveRegressor } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = passiveAggressiveRegressor({ epsilon: 0.01, maxEpochs: 50 }).fit({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] })
 * Math.round(toFlat(f.predict([[4]]))[0] ?? 0) // => 9
 * ```
 */
export function passiveAggressiveRegressor(
  hyper: Partial<PassiveAggressiveHyper> = {},
): IterativeEstimator<PassiveAggressiveHyper, PassiveAggressiveState, PassiveRegressorParams, PassiveRegressorVerbs> {
  return iterative<PassiveAggressiveHyper, PassiveAggressiveState, PassiveRegressorParams, PassiveRegressorVerbs>(
    {
      name: 'passive-aggressive-regressor',
      kind: 'regressor',
      defaults: PA_DEFAULTS,
      hyperSpace: (h) => dict({ c: real(Math.min(1e-3, h.c), Math.max(100, h.c), { default: h.c, log: true }) }),
      algorithm: (data, h, key) => {
        const x = matrixView(data.x, 'passiveAggressiveRegressor')
        return passiveAlgorithm(
          'passive-aggressive-regressor',
          x,
          targetVector(data.y ?? [], 'passiveAggressiveRegressor', x.n),
          h,
          key,
          true,
        )
      },
      maxSteps: (h, data) => h.maxEpochs * rowsOf(data.x),
      params: (s) => ({ coef: s.w, intercept: s.b, mistakes: s.mistakes, examples: s.t }),
      verbs: (p) => {
        const predict = (xs: TensorLike): Tensor => {
          const m = matrixView(xs, 'passiveAggressiveRegressor', { nFeatures: p.coef.length })
          return fromStorage(
            Float64Array.from({ length: m.n }, (_, i) => dotRow(p.coef, m, i) + p.intercept),
            [m.n],
            'float64',
          )
        }
        return {
          nFeaturesIn: p.coef.length,
          predict,
          featureImportances: p.coef.map(Math.abs),
          score: (data) => {
            const pred = toFlat(predict(data.x))
            const truth = targetVector(data.y ?? [], 'passiveAggressiveRegressor', pred.length)
            const mean = truth.reduce((a, b) => a + b, 0) / truth.length
            let res = 0
            let tot = 0
            for (let i = 0; i < pred.length; i++) {
              res += ((truth[i] ?? 0) - (pred[i] ?? 0)) ** 2
              tot += ((truth[i] ?? 0) - mean) ** 2
            }
            return tot === 0 ? (res === 0 ? 1 : 0) : 1 - res / tot
          },
        }
      },
      logs: { 'solver/mistakes': Log(), 'solver/loss': Log(), 'solver/weights': Log({ detail: 'full' }) },
    },
    { ...PA_DEFAULTS, ...hyper },
  )
}

/**
 * Data attribution: which training rows a model owes its behaviour to. Influence functions, TracIn over checkpoints,
 * Monte Carlo data Shapley, the exact k-nearest-neighbour Shapley values, Beta Shapley and data Banzhaf over any
 * subset utility, and representer point values.
 */

import { grad, hessian } from '../autodiff/index.ts'
import { meanOf, OmaiError } from '../base/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from '../learn/_shared.ts'
import type { Estimator, FitOptions, LearnData } from '../learn/estimator.ts'
import { type Scorer, type Scoring, scorersOf } from '../learn/scoring.ts'
import { matrixView, rowsOf, takeData, targetVector } from '../learn/validation.ts'
import { solve } from '../numerics/linalg/index.ts'
import { lgamma as logGamma } from '../numerics/special/_kernels.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { add, asTensor, fromStorage, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  logShapley,
  type ShapleyPath,
  type ShapleyState,
  shapleyAccumulate,
  shapleyEstimates,
  shapleyStart,
} from './permutation-estimates.ts'

/** A per-example loss of flat parameters `[p]`, one input row `[d]` and its target. */
export type ExampleLoss = (params: Value, x: Value, y: number) => Value

function rows(x: TensorLike, who: string): Tensor[] {
  const m = matrixView(x, who)
  return Array.from({ length: m.n }, (_, i) =>
    fromStorage(Float64Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)), [m.d], 'float64'),
  )
}

function flatParams(params: TensorLike): Tensor {
  const v = Float64Array.from(toFlat(asTensor(params)))
  return fromStorage(v, [v.length], 'float64')
}

/** Options of {@link influenceFunctions}. */
export interface InfluenceOptions {
  /** Added to the Hessian's diagonal (default 0; use a small value when the loss is not strictly convex). */
  readonly damping?: number
}

/** The result of {@link influenceFunctions}. */
export interface InfluenceResult {
  /** influence[test][train]: d L(z_test) / d epsilon when z_train is up-weighted by epsilon (negative helps). */
  readonly influence: readonly (readonly number[])[]
  /** The first-order change in the parameters from removing each training point, (1/n) H^-1 grad L(z): [train][p]. */
  readonly parameterChange: readonly (readonly number[])[]
}

/**
 * Influence functions (Cook and Weisberg 1982; Koh and Liang 2017): the effect of up-weighting a training point on a
 * test loss, I(z, z_test) = -grad L(z_test)^T H^-1 grad L(z), with H the mean Hessian of the training loss at the
 * fitted parameters (exact, by automatic differentiation). Removing z changes the test loss by about -I / n.
 *
 * @param loss - The per-example loss.
 * @param params - The fitted parameters (flat).
 * @param train - Training rows and targets.
 * @param test - Test rows and targets.
 * @param options - damping.
 * @returns Influences and first-order parameter changes.
 * @concept data-valuation
 * @example
 * ```ts
 * import { influenceFunctions } from 'omai-engine/explain'
 * import { mul, square, sub, sum } from 'omai-engine/tensor'
 * // Least squares on a mean: theta = mean(y) = 2. Up-weighting the outlier at 4 raises the test loss at 1 (+4);
 * // up-weighting a point at 1 lowers it (-2).
 * const loss = (p, x, y) => square(sub(sum(p), y))
 * const r = influenceFunctions(loss, [2], { x: [[0], [0], [0]], y: [1, 1, 4] }, { x: [[0]], y: [1] })
 * r.influence[0].map((v) => Math.round(v * 1e9) / 1e9) // => [-2, -2, 4]
 * ```
 */
export function influenceFunctions(
  loss: ExampleLoss,
  params: TensorLike,
  train: LearnData,
  test: LearnData,
  options: InfluenceOptions = {},
): InfluenceResult {
  const theta = flatParams(params)
  const p = theta.shape[0] ?? 0
  const xs = rows(train.x, 'influenceFunctions')
  const ys = targetVector(train.y ?? [], 'influenceFunctions', xs.length)
  const xt = rows(test.x, 'influenceFunctions')
  const yt = targetVector(test.y ?? [], 'influenceFunctions', xt.length)
  const n = xs.length
  const h = new Float64Array(p * p)
  const grads = xs.map((x, i) => {
    const f = (q: Value): Value => loss(q, x, ys[i] ?? 0)
    const hi = toFlat(hessian(f)(theta) as Tensor)
    for (let t = 0; t < p * p; t++) h[t] = (h[t] ?? 0) + (hi[t] ?? 0) / n
    return toFlat(grad(f)(theta) as Tensor)
  })
  for (let a = 0; a < p; a++) h[a * p + a] = (h[a * p + a] ?? 0) + (options.damping ?? 0)
  const hm = fromStorage(h, [p, p], 'float64')
  const hinvG = grads.map((g) => toFlat(solve(hm, fromStorage(Float64Array.from(g), [p], 'float64'))))
  const influence = xt.map((x, j) => {
    const gt = toFlat(grad((q: Value) => loss(q, x, yt[j] ?? 0))(theta) as Tensor)
    return hinvG.map((v) => -v.reduce((s, t, a) => s + t * (gt[a] ?? 0), 0))
  })
  return { influence, parameterChange: hinvG.map((v) => v.map((t) => t / n)) }
}

/** A training checkpoint for {@link tracIn}: parameters and the learning rate used after them. */
export interface Checkpoint {
  readonly params: TensorLike
  readonly learningRate: number
}

/**
 * TracIn (Pruthi et al. 2020): the influence of a training point on a test point traced through training, the sum
 * over checkpoints of learning rate times the dot product of their loss gradients. Positive values are proponents
 * (the training point lowered the test loss), negative ones opponents.
 *
 * @param loss - The per-example loss.
 * @param checkpoints - Parameters saved during training, with learning rates.
 * @param train - Training rows and targets.
 * @param test - Test rows and targets.
 * @returns tracin[test][train].
 * @example
 * ```ts
 * import { tracIn } from 'omai-engine/explain'
 * import { mul, square, sub, sum } from 'omai-engine/tensor'
 * const loss = (p, x, y) => square(sub(sum(mul(p, x)), y))
 * tracIn(loss, [{ params: [0], learningRate: 0.5 }], { x: [[1], [1]], y: [1, -1] }, { x: [[1]], y: [1] }) // => [[2, -2]]
 * ```
 */
export function tracIn(
  loss: ExampleLoss,
  checkpoints: readonly Checkpoint[],
  train: LearnData,
  test: LearnData,
): number[][] {
  if (checkpoints.length === 0) throw new OmaiError('value', 'tracIn: needs at least one checkpoint')
  const xs = rows(train.x, 'tracIn')
  const ys = targetVector(train.y ?? [], 'tracIn', xs.length)
  const xt = rows(test.x, 'tracIn')
  const yt = targetVector(test.y ?? [], 'tracIn', xt.length)
  const out = xt.map(() => new Array<number>(xs.length).fill(0))
  for (const c of checkpoints) {
    const theta = flatParams(c.params)
    const g = xs.map((x, i) => toFlat(grad((q: Value) => loss(q, x, ys[i] ?? 0))(theta) as Tensor))
    xt.forEach((x, j) => {
      const gt = toFlat(grad((q: Value) => loss(q, x, yt[j] ?? 0))(theta) as Tensor)
      const row = out[j] as number[]
      g.forEach((gi, i) => {
        row[i] = (row[i] ?? 0) + c.learningRate * gi.reduce((s, v, a) => s + v * (gt[a] ?? 0), 0)
      })
    })
  }
  return out
}

/** Options of {@link dataShapley}. */
export interface DataShapleyOptions extends FitOptions {
  /** The utility: a scorer on the test data (default the estimator's score). */
  readonly scoring?: Scoring
  /** Permutations (default 100). */
  readonly nPermutations?: number
  /** Stop a permutation once the prefix's utility is within this fraction of the full data's (default 0.01). */
  readonly truncationTolerance?: number
  /** The utility of the empty set and of prefixes the estimator cannot fit (default 0). */
  readonly emptyUtility?: number
}

/** The result of {@link dataShapley}. */
export interface DataShapleyResult {
  /** One value per training point; they sum to about v(all) - v(empty). */
  readonly values: readonly number[]
  /** Standard errors of the Monte Carlo means. */
  readonly standardErrors: readonly number[]
  /** The utility of the full training set. */
  readonly fullUtility: number
}

function dataShapleyRun(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  options: DataShapleyOptions,
): {
  readonly algorithm: Algorithm<ShapleyState, ShapleyPath>
  readonly nPerm: number
  readonly fullUtility: number
} {
  const n = rowsOf(train.x)
  const scorer: Scorer | undefined = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new OmaiError('value', 'dataShapley: no scorer')
  const nPerm = options.nPermutations ?? 100
  if (!(Number.isInteger(nPerm) && nPerm >= 1)) throw new OmaiError('value', 'dataShapley: nPermutations must be >= 1')
  const empty = options.emptyUtility ?? 0
  const key = keyFor(options, 'dataShapley')
  const utility = (idx: readonly number[], opts: FitOptions): number => {
    if (idx.length === 0) return empty
    try {
      return scorer.score(estimator.fit(takeData(train, idx), opts), test)
    } catch {
      return empty
    }
  }
  const fullUtility = utility(
    Array.from({ length: n }, (_, i) => i),
    innerFitOptions(options, foldIn(key, 0), options.parent),
  )
  const tol = options.truncationTolerance ?? 0.01
  const algorithm: Algorithm<ShapleyState, ShapleyPath> = {
    name: 'data-shapley',
    config: { estimator: estimator.name, n, nPermutations: nPerm, truncationTolerance: tol },
    schema: channels({ metrics: { permutations: { description: 'orders sampled' } } }),
    init: () => shapleyStart(n),
    step: (s, ctx) => {
      const order = toFlat(permutation(foldIn(key, 1 + s.count), n))
      const marginal = new Float64Array(n)
      let prev = empty
      let truncated = false
      for (let t = 0; t < n; t++) {
        const i = order[t] ?? 0
        let contribution = 0
        if (!truncated) {
          const v = utility(order.slice(0, t + 1), innerFitOptions(options, foldIn(ctx.key, t), ctx.run))
          contribution = v - prev
          prev = v
          if (Math.abs(fullUtility - v) <= tol * Math.abs(fullUtility)) truncated = true
        }
        marginal[i] = contribution
      }
      const next = shapleyAccumulate(s, marginal)
      logShapley(ctx, next, marginal)
      return next
    },
    done: (s) => (s.count >= nPerm ? 'completed' : undefined),
    observe: (s) => ({ metrics: { permutations: s.count } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ShapleyState,
  }
  return { algorithm, nPerm, fullUtility }
}

/**
 * Data Shapley by truncated Monte Carlo (Ghorbani and Zou 2019) as a stepped algorithm: each step draws a random order
 * of the training data, refits the model on growing prefixes and adds each point's marginal gain in test utility to a
 * running mean and standard error, truncating the order once a prefix is almost as good as the full data. Each fit
 * nests as a child run. The state serialises. The utility of the full data is computed when the algorithm is built.
 *
 * @param estimator - The estimator to refit.
 * @param train - Training data.
 * @param test - The data the utility is measured on.
 * @param options - scoring, nPermutations, truncationTolerance, emptyUtility and key.
 * @returns The algorithm, `nPermutations` steps; read the estimates with `shapleyEstimates`.
 * @throws OmaiError with code `value` for no scorer or fewer than one permutation.
 * @concept data-valuation
 * @example
 * ```ts
 * import { dataShapleyAlgorithm, shapleyEstimates } from 'omai-engine/explain'
 * import { defineEstimator, rowsOf } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { full, item, mean } from 'omai-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }) },
 *   {},
 * )
 * const alg = dataShapleyAlgorithm(meanModel, { x: [[0], [0], [0]], y: [1, 1, 9] }, { x: [[0]], y: [1] }, { scoring: 'neg_mean_absolute_error', nPermutations: 30, emptyUtility: -5 })
 * const v = shapleyEstimates(runAlgorithm(alg, { maxSteps: 30 }).state).values
 * v[2] < v[0] // => true
 * ```
 */
export function dataShapleyAlgorithm(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  options: DataShapleyOptions = {},
): Algorithm<ShapleyState, ShapleyPath> {
  return dataShapleyRun(estimator, train, test, options).algorithm
}

/**
 * Data Shapley by truncated Monte Carlo (Ghorbani and Zou 2019): each training point's value is its mean marginal
 * contribution to the test utility over random orders of the data, refitting the model on growing prefixes and
 * truncating an order once its prefix is almost as good as the full data. This is {@link dataShapleyAlgorithm} run to
 * the end, a traced algorithm with one step per order.
 *
 * @param estimator - The estimator to refit.
 * @param train - Training data.
 * @param test - The data the utility is measured on.
 * @param options - scoring, nPermutations, truncationTolerance, emptyUtility, key and tracing.
 * @returns The values, their standard errors and the full utility.
 * @example
 * ```ts
 * import { dataShapley } from 'omai-engine/explain'
 * import { defineEstimator, rowsOf } from 'omai-engine/learn'
 * import { full, item, mean } from 'omai-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }) },
 *   {},
 * )
 * const r = dataShapley(meanModel, { x: [[0], [0], [0]], y: [1, 1, 9] }, { x: [[0]], y: [1] }, { scoring: 'neg_mean_absolute_error', nPermutations: 30, emptyUtility: -5 })
 * r.values[2] < r.values[0] // => true
 * ```
 */
export function dataShapley(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  options: DataShapleyOptions = {},
): DataShapleyResult {
  const { algorithm, nPerm, fullUtility } = dataShapleyRun(estimator, train, test, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'dataShapley'), maxSteps: nPerm })
  const { values, standardErrors } = shapleyEstimates(state)
  return { values, standardErrors, fullUtility }
}

/**
 * Exact KNN-Shapley (Jia et al. 2019): the Shapley values of training points for the utility of an unweighted k-NN
 * classifier on test points, by the closed-form recursion over training points sorted by distance, in O(N log N) per
 * test point; averaged over the test points.
 *
 * @param trainX - Training rows `[N, d]`.
 * @param trainY - Their labels `[N]`.
 * @param testX - Test rows `[M, d]`.
 * @param testY - Their labels `[M]`.
 * @param k - The number of neighbours (default 5).
 * @returns One value per training point.
 * @example
 * ```ts
 * import { knnShapley } from 'omai-engine/explain'
 * // The near point with the right label is worth most; the wrong label is worth less than nothing.
 * knnShapley([[0], [1], [5]], [1, 0, 1], [[0]], [1], 1).map((v) => Math.round(v * 1e6) / 1e6) // => [0.833333, -0.166667, 0.333333]
 * ```
 */
export function knnShapley(
  trainX: TensorLike,
  trainY: TensorLike,
  testX: TensorLike,
  testY: TensorLike,
  k = 5,
): number[] {
  const tr = matrixView(trainX, 'knnShapley')
  const te = matrixView(testX, 'knnShapley', { nFeatures: tr.d })
  const ytr = targetVector(trainY, 'knnShapley', tr.n)
  const yte = targetVector(testY, 'knnShapley', te.n)
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('value', 'knnShapley: k must be >= 1')
  const N = tr.n
  const total = new Float64Array(N)
  for (let j = 0; j < te.n; j++) {
    const dist = Array.from({ length: N }, (_, i) => {
      let s = 0
      for (let c = 0; c < tr.d; c++) s += ((tr.data[i * tr.d + c] ?? 0) - (te.data[j * te.d + c] ?? 0)) ** 2
      return { i, s }
    }).sort((a, b) => a.s - b.s || a.i - b.i)
    const match = dist.map((e) => (ytr[e.i] === yte[j] ? 1 : 0))
    const s = new Float64Array(N)
    // The farthest point counts only while it is among the k nearest: in min(k, N) of the N subset sizes.
    s[N - 1] = ((match[N - 1] ?? 0) * Math.min(k, N)) / (k * N)
    for (let r = N - 2; r >= 0; r--) {
      s[r] = (s[r + 1] ?? 0) + (((match[r] ?? 0) - (match[r + 1] ?? 0)) / k) * (Math.min(k, r + 1) / (r + 1))
    }
    dist.forEach((e, r) => {
      total[e.i] = (total[e.i] ?? 0) + (s[r] ?? 0) / te.n
    })
  }
  return Array.from(total)
}

/**
 * The k-NN utility KNN-Shapley values (for checks): the mean over test points of the fraction of the k nearest
 * selected training points (fewer when fewer are selected, divided by k) that share the test label.
 *
 * @param trainX - Training rows.
 * @param trainY - Their labels.
 * @param testX - Test rows.
 * @param testY - Their labels.
 * @param k - The number of neighbours.
 * @returns The utility of each subset of training points (a coalition value).
 * @example
 * ```ts
 * import { knnUtility } from 'omai-engine/explain'
 * knnUtility([[0], [1]], [1, 0], [[0]], [1], 1)([true, true]) // => 1
 * ```
 */
export function knnUtility(
  trainX: TensorLike,
  trainY: TensorLike,
  testX: TensorLike,
  testY: TensorLike,
  k = 5,
): (subset: readonly boolean[]) => number {
  const tr = matrixView(trainX, 'knnUtility')
  const te = matrixView(testX, 'knnUtility', { nFeatures: tr.d })
  const ytr = targetVector(trainY, 'knnUtility', tr.n)
  const yte = targetVector(testY, 'knnUtility', te.n)
  return (subset) =>
    meanOf(
      Array.from({ length: te.n }, (_, j) => {
        const chosen = Array.from({ length: tr.n }, (_, i) => i)
          .filter((i) => subset[i] === true)
          .map((i) => {
            let s = 0
            for (let c = 0; c < tr.d; c++) s += ((tr.data[i * tr.d + c] ?? 0) - (te.data[j * te.d + c] ?? 0)) ** 2
            return { i, s }
          })
          .sort((a, b) => a.s - b.s || a.i - b.i)
          .slice(0, k)
        return chosen.filter((e) => ytr[e.i] === yte[j]).length / k
      }),
    )
}

/**
 * The total of a per-example loss over a dataset (to fit parameters with the same loss the valuation uses).
 *
 * @param loss - The per-example loss.
 * @param params - The parameters.
 * @param data - Rows and targets.
 * @returns The summed loss (differentiable in the parameters).
 * @example
 * ```ts
 * import { totalLoss } from 'omai-engine/explain'
 * import { item, square, sub, sum } from 'omai-engine/tensor'
 * item(totalLoss((p, x, y) => square(sub(sum(p), y)), [1], { x: [[0], [0]], y: [0, 3] })) // => 5
 * ```
 */
export function totalLoss(loss: ExampleLoss, params: Value, data: LearnData): Value {
  const xs = rows(data.x, 'totalLoss')
  const ys = targetVector(data.y ?? [], 'totalLoss', xs.length)
  return xs.reduce<Value>(
    (acc, x, i) => add(acc, loss(params, x, ys[i] ?? 0)),
    fromStorage(new Float64Array([0]), [], 'float64'),
  )
}

/** A utility of training subsets: the test score of a model fitted on those rows. */
export type SubsetUtility = (rows: readonly number[]) => number

/**
 * The utility of training subsets for an estimator: fit on the rows and score on the test data, with a fixed
 * utility for the empty set and for subsets the estimator cannot fit (one class only, too few rows).
 *
 * @param estimator - The estimator.
 * @param train - Training data.
 * @param test - The data the utility is measured on.
 * @param options - scoring (default the estimator's score) and emptyUtility (default 0).
 * @returns The utility.
 * @example
 * ```ts
 * import { subsetUtility } from 'omai-engine/explain'
 * import { defineEstimator, rowsOf } from 'omai-engine/learn'
 * import { full, item, mean } from 'omai-engine/tensor'
 * const meanModel = defineEstimator({ name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }) }, {})
 * const u = subsetUtility(meanModel, { x: [[0], [0]], y: [1, 3] }, { x: [[0]], y: [1] }, { scoring: 'neg_mean_absolute_error' })
 * u([0]) // => 0
 * ```
 */
export function subsetUtility(
  estimator: Estimator,
  train: LearnData,
  test: LearnData,
  options: { readonly scoring?: Scoring; readonly emptyUtility?: number } = {},
): SubsetUtility {
  const scorer = Object.values(scorersOf(options.scoring))[0]
  if (scorer === undefined) throw new OmaiError('value', 'subsetUtility: no scorer')
  const empty = options.emptyUtility ?? 0
  return (rows) => {
    if (rows.length === 0) return empty
    try {
      return scorer.score(estimator.fit(takeData(train, rows), { explain: false }), test)
    } catch {
      return empty
    }
  }
}

/** Options of {@link betaShapley}. */
export interface BetaShapleyOptions extends FitOptions {
  /** The Beta(alpha, beta) weights on subset sizes (default 16 and 1, Kwon and Zou's choice; 1 and 1 is Shapley). */
  readonly alpha?: number
  readonly beta?: number
  /** Permutations sampled (default 100). */
  readonly nPermutations?: number
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b)
}

/**
 * Beta Shapley (Kwon and Zou 2022): a semivalue that weights a point's marginal contributions by the size of the
 * subset it joins, w(j) = n C(n - 1, j - 1) B(j + beta - 1, n - j + alpha) / B(alpha, beta), estimated over random
 * permutations (in which the size a point joins is uniform). With alpha > beta small subsets weigh more, where a noisy
 * or mislabelled point's harm shows most; alpha = beta = 1 is data Shapley.
 *
 * @param utility - The utility of training subsets (see {@link subsetUtility}).
 * @param n - The number of training points.
 * @param options - alpha, beta, nPermutations and key.
 * @returns One value per point and the standard errors.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * // An additive utility: each point is worth its own value, whatever the weights.
 * const worth = [1, 2, -1]
 * const u = (rows) => rows.reduce((s, i) => s + worth[i], 0)
 * explain.betaShapley(u, 3, { nPermutations: 20, alpha: 1, beta: 1 }).values.map((v) => Math.round(v * 1e9) / 1e9) // => [1, 2, -1]
 * ```
 */
export function betaShapley(
  utility: SubsetUtility,
  n: number,
  options: BetaShapleyOptions = {},
): { readonly values: readonly number[]; readonly standardErrors: readonly number[] } {
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'betaShapley: needs at least one point')
  const a = options.alpha ?? 16
  const b = options.beta ?? 1
  const logB = logBeta(a, b)
  // The weight of joining a subset of size s (j = s + 1).
  const weight = Array.from({ length: n }, (_, s) => {
    const j = s + 1
    const logChoose = logGamma(n) - logGamma(j) - logGamma(n - j + 1)
    return n * Math.exp(logChoose + logBeta(j + b - 1, n - j + a) - logB)
  })
  const key = keyFor(options, 'betaShapley')
  const nPerm = options.nPermutations ?? 100
  const sum = new Float64Array(n)
  const sumSq = new Float64Array(n)
  for (let t = 0; t < nPerm; t++) {
    const order = toFlat(permutation(foldIn(key, t), n))
    let prev = utility([])
    const prefix: number[] = []
    order.forEach((i, s) => {
      prefix.push(i)
      const cur = utility(prefix)
      const c = (weight[s] ?? 1) * (cur - prev)
      sum[i] = (sum[i] ?? 0) + c
      sumSq[i] = (sumSq[i] ?? 0) + c * c
      prev = cur
    })
  }
  const values = Array.from(sum, (v) => v / nPerm)
  const standardErrors = Array.from(sumSq, (v, i) =>
    nPerm < 2 ? Number.NaN : Math.sqrt(Math.max(0, v / nPerm - (values[i] ?? 0) ** 2) / (nPerm - 1)),
  )
  return { values, standardErrors }
}

/**
 * Data Banzhaf (Wang and Jia 2023) by the maximum sample reuse estimator: draw random training subsets, each point in
 * with probability one half; a point's value is the mean utility of the subsets that hold it minus that of those that
 * do not. Every sample informs every point, and the Banzhaf weighting is the most robust semivalue to noisy utilities.
 *
 * @param utility - The utility of training subsets.
 * @param n - The number of training points.
 * @param options - nSamples (default 200) and key.
 * @returns One value per point.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * const worth = [1, 2, -1]
 * const u = (rows) => rows.reduce((s, i) => s + worth[i], 0)
 * explain.dataBanzhaf(u, 3, { nSamples: 400 }).values.map((v) => Math.round(v)) // => [1, 2, -1]
 * ```
 */
export function dataBanzhaf(
  utility: SubsetUtility,
  n: number,
  options: { readonly nSamples?: number; readonly key?: Key } = {},
): { readonly values: readonly number[] } {
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', 'dataBanzhaf: needs at least one point')
  const key = options.key ?? keyFor({}, 'dataBanzhaf')
  const ns = options.nSamples ?? 200
  const inSum = new Float64Array(n)
  const inCount = new Float64Array(n)
  const outSum = new Float64Array(n)
  const outCount = new Float64Array(n)
  for (let t = 0; t < ns; t++) {
    const u = toFlat(uniform(foldIn(key, t), [n]))
    const rows = Array.from({ length: n }, (_, i) => i).filter((i) => (u[i] ?? 0) < 0.5)
    const value = utility(rows)
    const member = new Set(rows)
    for (let i = 0; i < n; i++) {
      if (member.has(i)) {
        inSum[i] = (inSum[i] ?? 0) + value
        inCount[i] = (inCount[i] ?? 0) + 1
      } else {
        outSum[i] = (outSum[i] ?? 0) + value
        outCount[i] = (outCount[i] ?? 0) + 1
      }
    }
  }
  return {
    values: Array.from({ length: n }, (_, i) =>
      (inCount[i] ?? 0) === 0 || (outCount[i] ?? 0) === 0
        ? 0
        : (inSum[i] ?? 0) / (inCount[i] ?? 1) - (outSum[i] ?? 0) / (outCount[i] ?? 1),
    ),
  }
}

/**
 * Representer point values (Yeh, Kim, Yen and Ravikumar 2018): for a model whose last layer is linear with an L2
 * penalty lambda, at a stationary point the prediction decomposes over the training points as
 * f(x_t) = sum_i alpha_i phi_i . phi_t with alpha_i = -(1 / (2 lambda n)) dL/df(x_i); each training point's
 * representer value for a test point is alpha_i (phi_i . phi_t), positive points excitatory and negative inhibitory.
 *
 * @param trainFeatures - The last layer's inputs for the training points `[n, p]`.
 * @param outputGradients - The loss's gradient with respect to the model's output at each training point `[n]`.
 * @param testFeatures - The last layer's inputs for the test points `[m, p]`.
 * @param lambda - The L2 penalty of the last layer.
 * @returns values[test][train], and alpha.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * const r = explain.representerValues([[1, 0], [0, 1]], [-2, 1], [[1, 1]], 0.25)
 * r.values // => [[2, -1]]
 * ```
 */
export function representerValues(
  trainFeatures: TensorLike,
  outputGradients: readonly number[],
  testFeatures: TensorLike,
  lambda: number,
): { readonly values: readonly (readonly number[])[]; readonly alpha: readonly number[] } {
  const tr = matrixView(trainFeatures, 'representerValues')
  const te = matrixView(testFeatures, 'representerValues', { nFeatures: tr.d })
  if (outputGradients.length !== tr.n)
    throw new OmaiError('shape', 'representerValues: one gradient per training point')
  if (!(lambda > 0)) throw new OmaiError('value', 'representerValues: lambda must be positive')
  const alpha = outputGradients.map((g) => -g / (2 * lambda * tr.n))
  const values = Array.from({ length: te.n }, (_, t) =>
    alpha.map((a, i) => {
      let dot = 0
      for (let j = 0; j < tr.d; j++) dot += (tr.data[i * tr.d + j] ?? 0) * (te.data[t * tr.d + j] ?? 0)
      return a * dot
    }),
  )
  return { values, alpha }
}

/**
 * Training with noisy labels: the loss corrections that use the noise-transition matrix (forward and backward,
 * Patrini et al. 2017), small-loss selection (the rows a model fits best are mostly the clean ones), and co-teaching
 * (Han et al. 2018), where two models each pick their small-loss rows for the other to learn from. Finding which labels are
 * wrong, and estimating the transition matrix, is `weak-supervision.ts`.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import {
  asTensor,
  fromStorage,
  log,
  matmul,
  maximum,
  mul,
  neg,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { accuracyOf } from './_classify.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import { type Estimator, type LearnData, requireTargets, requireVerb } from './estimator.ts'
import { matrixView, takeData } from './validation.ts'

const oneHot = (labels: TensorLike, k: number): Tensor => {
  const y = toFlat(asTensor(labels))
  const out = new Float64Array(y.length * k)
  y.forEach((c, i) => {
    if (!Number.isInteger(c) || c < 0 || c >= k)
      throw new OmaiError('value', `loss correction: label ${c} is not a class index in [0, ${k})`)
    out[i * k + c] = 1
  })
  return fromStorage(out, [y.length, k], 'float64')
}

/**
 * The forward-corrected loss (Patrini et al. 2017): the model's class probabilities are pushed through the noise
 * transition `T[true][noisy]` to the probabilities of the *noisy* label, and the cross-entropy is taken against the given
 * noisy labels, `-log (T^T p)_y`. Minimised, it recovers the classifier of the clean labels when `T` is right. Differentiable in
 * `proba`.
 *
 * @param proba - Class probabilities `[n, k]` of the model.
 * @param labels - The given (noisy) labels as class indices.
 * @param transition - `T[i][j] = P(noisy j | true i)`, rows summing to one.
 * @returns The loss of each row, `[n]`.
 * @example
 * ```ts
 * import { forwardCorrectedLoss } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * // A model sure of class 0, labels noisy with a 20 percent flip: the loss of label 1 is -log 0.2.
 * toFlat(forwardCorrectedLoss([[1, 0]], [1], [[0.8, 0.2], [0.2, 0.8]])) // => [1.6094379124341003]
 * ```
 */
export function forwardCorrectedLoss(
  proba: TensorLike,
  labels: TensorLike,
  transition: readonly (readonly number[])[],
): Value {
  const k = transition.length
  const t = fromStorage(Float64Array.from(transition.flatMap((r) => [...r])), [k, k], 'float64')
  const noisy = matmul(proba, t)
  return neg(log(maximum(sum(mul(noisy, oneHot(labels, k)), -1), 1e-12)))
}

/**
 * The backward-corrected loss (Patrini et al. 2017): the per-class cross-entropies `-log p_j` are multiplied by the
 * inverse of the transition, `(T^-1 l)_y`, an unbiased estimate of the clean loss (it can be negative for a single row).
 * Differentiable in `proba`.
 *
 * @param proba - Class probabilities `[n, k]` of the model.
 * @param labels - The given (noisy) labels as class indices.
 * @param transition - `T[i][j] = P(noisy j | true i)`, rows summing to one and invertible.
 * @returns The loss of each row, `[n]`.
 * @example
 * ```ts
 * import { backwardCorrectedLoss } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * // With no noise (T the identity) the corrected loss is the plain cross-entropy.
 * toFlat(backwardCorrectedLoss([[0.5, 0.5]], [0], [[1, 0], [0, 1]])) // => [0.6931471805599453]
 * ```
 */
export function backwardCorrectedLoss(
  proba: TensorLike,
  labels: TensorLike,
  transition: readonly (readonly number[])[],
): Value {
  const k = transition.length
  const t = fromStorage(Float64Array.from(transition.flatMap((r) => [...r])), [k, k], 'float64')
  const tInv = inv(t)
  const losses = neg(log(maximum(proba, 1e-12)))
  // (T^-1 l)_y = sum_j Tinv[y][j] l_j: row y of the inverse, taken for each row's noisy label.
  const rows = matmul(oneHot(labels, k), tInv)
  return sum(mul(rows, losses), -1)
}

/**
 * Small-loss selection: the indices of the rows with the smallest losses, a fraction `keep` of them (at least one). Deep
 * models fit clean labels before they memorise noisy ones, so the low-loss rows are mostly clean.
 *
 * @param losses - The loss of each row.
 * @param keep - The fraction to keep, in (0, 1].
 * @returns The selected row indices in increasing loss (ties by index).
 * @example
 * ```ts
 * import { smallLossSelection } from 'omai-engine/learn'
 * smallLossSelection([0.5, 2, 0.1, 0.9], 0.5) // => [2, 0]
 * ```
 */
export function smallLossSelection(losses: ArrayLike<number>, keep: number): number[] {
  if (!(keep > 0 && keep <= 1)) throw new OmaiError('value', `smallLossSelection: keep must be in (0, 1], got ${keep}`)
  const n = losses.length
  const count = Math.max(1, Math.ceil(keep * n - 1e-12))
  return Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (losses[a] ?? 0) - (losses[b] ?? 0) || a - b)
    .slice(0, count)
}

/** The hyperparameters of {@link coTeaching}. */
export interface CoTeachingHyper {
  /** The base classifier of both models: it must give class probabilities. */
  readonly base: Estimator
  /** The fraction of rows to forget at the end: the estimated noise rate. */
  readonly forgetRate: number
  /** The rounds over which the fraction forgotten grows from zero to `forgetRate`. */
  readonly rampRounds: number
  readonly maxRounds: number
}

/** The state of {@link coTeachingAlgorithm}. */
export type CoTeachingState = {
  readonly round: number
  /** The fraction of rows each model keeps this round. */
  readonly keep: number
  readonly modelA: Json
  readonly modelB: Json
  /** The rows model A chose (to teach B) and model B chose (to teach A) in the last round. */
  readonly chosenByA: number[]
  readonly chosenByB: number[]
  /** The fraction of the rows both models chose. */
  readonly agreement: number
}

/** The fitted parameters of {@link coTeaching}. */
export type CoTeachingParams = CoTeachingState & { readonly classes: number[]; readonly d: number }

/** What a fitted co-teaching pair offers: model A's verbs, and the rows the pair trusts. */
export interface CoTeachingVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The rows both models chose in the last round: the pair's estimate of the clean ones. */
  readonly cleanRows: readonly number[]
}

/** The cross-entropy of each row's given label under a fitted base, by its class order. */
function rowLosses(proba: Float64Array, k: number, classes: readonly number[], y: ArrayLike<number>): Float64Array {
  return Float64Array.from({ length: y.length }, (_, i) => {
    const c = classes.indexOf(y[i] ?? Number.NaN)
    return c < 0 ? Number.POSITIVE_INFINITY : -Math.log(Math.max(proba[i * k + c] ?? 0, 1e-12))
  })
}

/**
 * Co-teaching as an algorithm (Han et al. 2018): two models, each round each ranks all rows by its own loss and keeps the
 * fraction `1 - forgetRate * min(round / rampRounds, 1)` with the smallest, and each is refitted on the rows the *other*
 * kept (so one model's mistakes are not reinforced by itself). The models start on the two halves of a random split, so
 * they differ. Logs `keep`, `agreement` (the fraction of rows both chose) and `lossA`, `lossB` (their mean loss on the
 * rows chosen for them).
 *
 * @param data - Features and (possibly noisy) class labels.
 * @param hyper - The base, the noise rate and the schedule.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { coTeachingAlgorithm, gaussianNaiveBayes } from 'omai-engine/learn'
 * const alg = coTeachingAlgorithm({ x: [[0], [1], [5], [6]], y: [0, 0, 1, 1] }, { base: gaussianNaiveBayes(), forgetRate: 0.25, rampRounds: 2, maxRounds: 4 })
 * alg.name // => 'co-teaching'
 * ```
 */
export function coTeachingAlgorithm(data: LearnData, hyper: CoTeachingHyper): Algorithm<CoTeachingState> {
  const who = 'coTeaching'
  const m = matrixView(data.x, who)
  const y = Float64Array.from(toFlat(asTensor(requireTargets(data, who))))
  if (y.length !== m.n) throw new OmaiError('shape', `${who}: ${m.n} rows but ${y.length} labels`)
  const all = Array.from({ length: m.n }, (_, i) => i)
  const fitOn = (rows: readonly number[], key: Key): Json =>
    hyper.base.fit(
      {
        ...takeData(data, rows),
        y: fromStorage(
          Float64Array.from(rows, (i) => y[i] ?? 0),
          [rows.length],
          'float64',
        ),
      },
      { key, explain: false },
    ).params
  const lossesOf = (model: Json): Float64Array => {
    const f = hyper.base.revive(model)
    const p = requireVerb(f, 'predictProba', who)(data.x)
    return rowLosses(Float64Array.from(toFlat(p)), p.shape[1] ?? 1, f.classes ?? [], y)
  }
  const keepAt = (round: number): number => 1 - hyper.forgetRate * Math.min(round / Math.max(hyper.rampRounds, 1), 1)
  return {
    name: 'co-teaching',
    config: { base: hyper.base.name, forgetRate: hyper.forgetRate, rampRounds: hyper.rampRounds },
    schema: channels({
      metrics: {
        keep: { description: 'the fraction of rows each model keeps for the other' },
        agreement: { description: 'the fraction of kept rows both models chose', direction: 'higher' },
        lossA: { description: 'model A’s mean loss on the rows model B chose', direction: 'lower' },
        lossB: { description: 'model B’s mean loss on the rows model A chose', direction: 'lower' },
      },
    }),
    init: (key) => {
      const order = Array.from(permutation(foldIn(key, 0), m.n).data)
      const half = Math.ceil(m.n / 2)
      return {
        round: 0,
        keep: 1,
        modelA: fitOn(order.slice(0, half), foldIn(key, 1)),
        modelB: fitOn(order.slice(half), foldIn(key, 2)),
        chosenByA: all,
        chosenByB: all,
        agreement: 1,
      }
    },
    step: (s, ctx) => {
      const round = s.round + 1
      const keep = keepAt(round)
      const chosenByA = smallLossSelection(lossesOf(s.modelA), keep)
      const chosenByB = smallLossSelection(lossesOf(s.modelB), keep)
      const both = new Set(chosenByB)
      return {
        round,
        keep,
        modelA: fitOn(chosenByB, foldIn(ctx.key, 0)),
        modelB: fitOn(chosenByA, foldIn(ctx.key, 1)),
        chosenByA,
        chosenByB,
        agreement: chosenByA.filter((i) => both.has(i)).length / chosenByA.length,
      }
    },
    done: (s) => (s.round >= hyper.maxRounds ? 'completed' : undefined),
    observe: (s) => {
      const la = lossesOf(s.modelA)
      const lb = lossesOf(s.modelB)
      const mean_ = (l: Float64Array, rows: readonly number[]): number =>
        rows.reduce((a, i) => a + (l[i] ?? 0), 0) / Math.max(rows.length, 1)
      return {
        metrics: { keep: s.keep, agreement: s.agreement, lossA: mean_(la, s.chosenByB), lossB: mean_(lb, s.chosenByA) },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CoTeachingState,
  }
}

/**
 * Co-teaching over any probabilistic classifier (Han et al. 2018): the noise rate `forgetRate` of the labels is the one
 * thing to know; two models trade the rows they find easy and so avoid memorising the noisy ones. The fitted value is model
 * A; `cleanRows` are the rows both models kept in the last round.
 *
 * @param base - The base classifier (it must offer `predictProba`).
 * @param hyper - `forgetRate` (default 0.2), `rampRounds` (default 5), `maxRounds` (default 15).
 * @returns The estimator.
 * @example
 * ```ts
 * import { coTeaching, gaussianNaiveBayes } from 'omai-engine/learn'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.2], [0.4], [0.1], [5], [5.2], [5.4], [5.1], [0.3], [5.3]]
 * const y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
 * const f = coTeaching(gaussianNaiveBayes(), { forgetRate: 0.2 }).fit({ x, y })
 * toFlat(f.predict([[0.2], [5.2]])) // => [0, 1]
 * ```
 */
export function coTeaching(
  base: Estimator,
  hyper: Partial<Omit<CoTeachingHyper, 'base'>> = {},
): IterativeEstimator<CoTeachingHyper, CoTeachingState, CoTeachingParams, CoTeachingVerbs> {
  const h: CoTeachingHyper = { base, forgetRate: 0.2, rampRounds: 5, maxRounds: 15, ...hyper }
  return iterative<CoTeachingHyper, CoTeachingState, CoTeachingParams, CoTeachingVerbs>(
    {
      name: 'co-teaching',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          forgetRate: real(0, 0.9, { default: c.forgetRate, label: 'forget rate (the noise rate)' }),
          rampRounds: int(1, 50, { default: c.rampRounds, label: 'ramp rounds' }),
        }),
      algorithm: (data, c) => coTeachingAlgorithm(data, c),
      maxSteps: (c) => c.maxRounds + 1,
      params: (s, data, c) => {
        const f = c.base.revive(s.modelA)
        return { ...s, classes: [...(f.classes ?? [])], d: matrixView(data.x, 'coTeaching').d }
      },
      verbs: (p, c) => {
        const f = c.base.revive(p.modelA)
        const predict = requireVerb(f, 'predict', 'coTeaching')
        const both = new Set(p.chosenByB)
        return {
          predict,
          predictProba: requireVerb(f, 'predictProba', 'coTeaching'),
          score: accuracyOf(predict),
          classes: p.classes,
          nFeaturesIn: p.d,
          cleanRows: p.chosenByA.filter((i) => both.has(i)),
        }
      },
      logs: { 'solver/keep': Log(), 'solver/agreement': Log(), 'solver/lossA': Log(), 'solver/lossB': Log() },
    },
    h,
  )
}

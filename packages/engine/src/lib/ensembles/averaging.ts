/**
 * The core of the averaging ensembles (bagging, random forests, extra trees): a stepped algorithm that adds one member
 * per step, each fitted on a bootstrap sample of the rows, and keeps the bookkeeping that makes the story: the rows
 * every member saw, the running sum of the members' outputs on the training rows, the out-of-bag sum (each row predicted
 * only by the members that never saw it) and the scores of both after each member. What a member is - a tree grown by
 * the grower, any estimator of the registry - is a `AveragingMember`; the paths logged are the same for all of them.
 */

import type { Json } from '../base/index.ts'
import { argmaxOf } from '../base/index.ts'
import { foldIn, type Key, randint, subsample } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type ScalarSource,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  type TensorSource,
  tensor,
} from '../trace/index.ts'
import { scoreOfSum } from './_stages.ts'

/** What an averaging ensemble needs of its members. */
export interface AveragingMember<P extends Json> {
  /** Fit a member on the rows with the given multiplicities (0 for rows left out); returns its parameters. */
  fit(counts: Float64Array, key: Key): P
  /** The member's outputs on the training rows: class probabilities `[n * k]` or values `[n]`. */
  outputs(member: P): Float64Array
  /** Called after each member with all the members so far, to log what only members know (importances). */
  after?(members: readonly P[], ctx: StepContext): void
}

/** Options of {@link averagingAlgorithm}. */
export interface AveragingOptions {
  readonly nEstimators: number
  /**
   * Draw a bootstrap sample (with replacement) per member. Otherwise every member sees every row, or, when
   * `maxSamples` is below the number of rows, a sample without replacement (pasting).
   */
  readonly bootstrap: boolean
  /** Rows per sample, as a count (default all). */
  readonly maxSamples?: number
}

/** The state of {@link averagingAlgorithm}. */
export interface AveragingState<P extends Json> {
  readonly members: readonly P[]
  /** Per member: how many times each training row is in its sample. */
  readonly bags: readonly Int32Array[]
  /** Running sum of the members' outputs over all rows, and over each row's out-of-bag members. */
  readonly sum: Float64Array
  readonly oobSum: Float64Array
  readonly oobCount: Float64Array
  /** After each member: the training score, the out-of-bag score (NaN while fewer than two rows are covered). */
  readonly trainScores: readonly number[]
  readonly oobScores: readonly number[]
}

/** What the algorithm knows of the training targets: class codes (`k` classes) or values (`k` = 0). */
export interface AveragingTarget {
  readonly y: ArrayLike<number>
  readonly k: number
}

/**
 * Members added one per step, each on a bootstrap sample. Logs, when subscribed: `ensemble/member` (members so far),
 * `ensemble/weight` (the member's weight in the average, 1), `ensemble/trainScore` and `ensemble/oobScore` (accuracy for
 * classifiers, R^2 for regressors, of the average so far), `ensemble/oobError` (the out-of-bag error rate, or mean
 * squared error), `ensemble/memberError` (the new member's error on the rows it did not see) and `ensemble/bootstrap`
 * (how often each row is in the member's sample).
 *
 * @param name - The algorithm's name.
 * @param target - The training targets.
 * @param options - Members, bootstrap and sample size.
 * @param member - How to fit a member and read its outputs.
 * @returns The algorithm.
 * @concept ensembles
 * @example
 * ```ts
 * import { averagingAlgorithm } from 'mdml-engine/ensembles'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // Members that each predict the mean of the rows they saw.
 * const y = [1, 2, 3, 4]
 * const member = {
 *   fit: (counts) => ({ m: counts.reduce((s, c, i) => s + c * y[i], 0) / counts.reduce((s, c) => s + c, 0) }),
 *   outputs: (p) => Float64Array.from(y, () => p.m),
 * }
 * const alg = averagingAlgorithm('mean', { y, k: 0 }, { nEstimators: 5, bootstrap: true }, member)
 * runAlgorithm(alg, { maxSteps: 10, seed: 1 }).state.members.length // => 5
 * ```
 */
export function averagingAlgorithm<P extends Json>(
  name: string,
  target: AveragingTarget,
  options: AveragingOptions,
  member: AveragingMember<P>,
): Algorithm<AveragingState<P>> {
  const n = target.y.length
  const k = target.k
  const width = Math.max(k, 1)
  const size = options.maxSamples === undefined ? n : Math.min(n, Math.max(1, Math.round(options.maxSamples)))
  return {
    name,
    config: { nEstimators: options.nEstimators, bootstrap: options.bootstrap, maxSamples: size },
    init: () => ({
      members: [],
      bags: [],
      sum: new Float64Array(n * width),
      oobSum: new Float64Array(n * width),
      oobCount: new Float64Array(n),
      trainScores: [],
      oobScores: [],
    }),
    step: (s, ctx) => {
      const counts = new Float64Array(n)
      if (options.bootstrap)
        for (const i of toFlat(randint(foldIn(ctx.key, 0), [size], 0, n))) counts[i] = (counts[i] ?? 0) + 1
      else if (size < n) for (const i of toFlat(subsample(foldIn(ctx.key, 0), n, size))) counts[i] = 1
      else counts.fill(1)
      const fitted = member.fit(counts, foldIn(ctx.key, 1))
      const out = member.outputs(fitted)
      const sum = Float64Array.from(s.sum)
      const oobSum = Float64Array.from(s.oobSum)
      const oobCount = Float64Array.from(s.oobCount)
      const left: number[] = []
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < width; c++) sum[i * width + c] = (sum[i * width + c] ?? 0) + (out[i * width + c] ?? 0)
        if ((counts[i] ?? 0) > 0) continue
        left.push(i)
        oobCount[i] = (oobCount[i] ?? 0) + 1
        for (let c = 0; c < width; c++) oobSum[i * width + c] = (oobSum[i * width + c] ?? 0) + (out[i * width + c] ?? 0)
      }
      const members = s.members.length + 1
      const covered = Array.from({ length: n }, (_, i) => i).filter((i) => (oobCount[i] ?? 0) > 0)
      const trainScore = scoreOfSum(sum, members, target.y, k)
      const oobScore = options.bootstrap || size < n ? scoreOfSum(oobSum, oobCount, target.y, k, covered) : Number.NaN
      if (ctx.wants('ensemble/member')) ctx.log('ensemble/member', members)
      if (ctx.wants('ensemble/weight')) ctx.log('ensemble/weight', 1)
      if (ctx.wants('ensemble/trainScore')) ctx.log('ensemble/trainScore', trainScore)
      if (ctx.wants('ensemble/oobScore') && Number.isFinite(oobScore)) ctx.log('ensemble/oobScore', oobScore)
      if (ctx.wants('ensemble/bootstrap')) ctx.log('ensemble/bootstrap', counts)
      if (ctx.wants('ensemble/oobError') && covered.length > 0)
        ctx.log('ensemble/oobError', errorOf(oobSum, oobCount, target, covered))
      if (ctx.wants('ensemble/memberError') && left.length > 0)
        ctx.log('ensemble/memberError', errorOf(out, 1, target, left))
      const all = [...s.members, fitted]
      member.after?.(all, ctx)
      return {
        members: all,
        bags: [...s.bags, Int32Array.from(counts)],
        sum,
        oobSum,
        oobCount,
        trainScores: [...s.trainScores, trainScore],
        oobScores: [...s.oobScores, oobScore],
      }
    },
    done: (s) => (s.members.length >= options.nEstimators ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as AveragingState<P>,
  }
}

/** The error rate (classes) or mean squared error (values) of summed outputs over some rows. */
function errorOf(
  sum: ArrayLike<number>,
  total: number | ArrayLike<number>,
  target: AveragingTarget,
  rows: readonly number[],
): number {
  const k = target.k
  const t = (i: number): number => (typeof total === 'number' ? total : (total[i] ?? 1))
  let e = 0
  for (const i of rows) {
    if (k > 0) e += argmaxOf(Array.from({ length: k }, (_, c) => sum[i * k + c] ?? 0)) === (target.y[i] ?? 0) ? 0 : 1
    else e += ((sum[i] ?? 0) / t(i) - (target.y[i] ?? 0)) ** 2
  }
  return e / rows.length
}

/**
 * The paths an averaging ensemble can log, for the contract of an estimator built on {@link averagingAlgorithm}.
 *
 * @param classifier - Whether the scores are accuracies (otherwise R^2 and mean squared errors).
 * @returns The sources.
 * @example
 * ```ts
 * import { averagingSources } from 'mdml-engine/ensembles'
 * Object.keys(averagingSources(true)).includes('ensemble/oobError') // => true
 * ```
 */
export function averagingSources(classifier: boolean): Readonly<Record<string, ScalarSource | TensorSource>> {
  const score = classifier ? 'accuracy' : 'R²'
  const error = classifier ? 'error rate' : 'mean squared error'
  return {
    'ensemble/member': scalar({ description: 'members in the ensemble so far' }),
    'ensemble/weight': scalar({ description: 'the new member’s weight in the average' }),
    'ensemble/trainScore': scalar({ description: `training ${score} of the average so far`, direction: 'higher' }),
    'ensemble/oobScore': scalar({ description: `out-of-bag ${score} of the average so far`, direction: 'higher' }),
    'ensemble/oobError': scalar({ description: `out-of-bag ${error} of the average so far`, direction: 'lower' }),
    'ensemble/memberError': scalar({
      description: `the new member’s ${error} on the rows it did not see`,
      direction: 'lower',
    }),
    'ensemble/bootstrap': tensor({
      role: 'output',
      shape: ['n'],
      description: 'how often each row is in the new member’s sample',
    }),
  }
}

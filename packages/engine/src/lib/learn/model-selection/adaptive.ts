/**
 * Adaptive data analysis: how reusing one holdout set for many adaptively chosen queries overfits to it, and the two
 * mechanisms that keep the answers honest. Thresholdout (Dwork et al. 2015, the reusable holdout) answers each
 * statistical query from the training set unless the holdout disagrees by more than a noisy threshold, and only then
 * releases a noised holdout value; the Ladder (Blum and Hardt 2015) reports a leaderboard score that moves only when a
 * submission improves on the best so far by more than a step. A planted problem (labels that are pure noise, an
 * analyst who combines whatever looks good on the answers) shows each mechanism against naive reuse.
 */

import { MdmlError, minOf } from '../../base/index.ts'
import { foldIn, type Key, laplace, key as makeKey, randint } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import type { LearnData } from '../estimator.ts'
import { matrixView, targetVector } from '../validation.ts'

/** A statistical query: a function of one example (its features and label) whose average over a sample is asked for. */
export type HoldoutQuery = (x: ArrayLike<number>, y: number) => number

/** A sample prepared for queries: row-major features and labels. */
interface Sample {
  readonly n: number
  readonly d: number
  readonly x: Float64Array
  readonly y: Float64Array
}

function sampleOf(who: string, data: LearnData): Sample {
  const m = matrixView(data.x, who)
  if (data.y === undefined) throw new MdmlError('value', `${who}: the data needs labels y`)
  return { n: m.n, d: m.d, x: m.data, y: targetVector(data.y, who, m.n) }
}

/** The average of a query over a sample. */
function average(s: Sample, q: HoldoutQuery): number {
  let total = 0
  for (let i = 0; i < s.n; i++) total += q(s.x.subarray(i * s.d, (i + 1) * s.d), s.y[i] ?? 0)
  return total / s.n
}

/** Options of {@link thresholdoutAlgorithm}. */
export interface ThresholdoutOptions {
  /** The threshold T: the training and holdout answers are taken to agree within it (default 0.04). */
  readonly threshold?: number
  /** The noise scale sigma of the Laplace noise (default threshold / 4). */
  readonly sigma?: number
  /** The budget B: how many times the holdout may be released before the mechanism stops answering (default 100). */
  readonly budget?: number
}

/** State of {@link thresholdoutAlgorithm}: the queries answered so far and what remains of the budget. */
export interface ThresholdoutState {
  /** Queries answered. */
  readonly t: number
  /** Releases of the holdout still allowed. */
  readonly budget: number
  /** The current noisy threshold T + gamma. */
  readonly noisyThreshold: number
  /** The answer to each query (NaN after the budget is spent). */
  readonly answers: readonly number[]
  /** Whether the holdout value was released for each query (otherwise the training value was). */
  readonly released: readonly boolean[]
  /** The training average of each query. */
  readonly trainValues: readonly number[]
  /** The holdout average of each query. */
  readonly holdoutValues: readonly number[]
}

/** The analyst of an adaptive run: it sees the answers so far and returns the next query, or `undefined` to stop. */
export type Analyst = (answers: readonly number[]) => HoldoutQuery | undefined

/**
 * Thresholdout (Dwork, Feldman, Hardt, Pitassi, Reingold and Roth 2015), the reusable holdout, as an algorithm with one
 * step per query. Each query phi is answered by the training average unless |holdout - training| exceeds the noisy
 * threshold T + gamma + eta (gamma ~ Lap(2 sigma) redrawn on every release, eta ~ Lap(4 sigma) per query); then it
 * answers the holdout average plus Lap(sigma) noise and spends one unit of the budget. Answers are accurate for
 * adaptively chosen queries (the holdout is only touched when overfitting to the training set is visible), whereas
 * answering every query from the holdout leaks it. The guarantee needs T and sigma of order the sampling error
 * 1 / sqrt(n) and a budget of at most a few times n.
 *
 * @param train - The training sample (features and labels), which the analyst may use freely.
 * @param holdout - The holdout sample, touched only through the mechanism.
 * @param analyst - A list of queries (non-adaptive), or a function of the answers so far returning the next query.
 * @param options - Threshold, noise scale and budget.
 * @returns The algorithm over {@link ThresholdoutState}.
 * @throws MdmlError with code `value` for samples without labels or of different widths.
 * @concept adaptive-data-analysis
 * @example
 * ```ts
 * import { thresholdoutAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const train = { x: [[1], [2], [3], [4]], y: [0, 0, 0, 0] }
 * const holdout = { x: [[1], [2], [3], [4]], y: [0, 0, 0, 0] }
 * const alg = thresholdoutAlgorithm(train, holdout, [(x) => x[0] as number])
 * runAlgorithm(alg, { maxSteps: 1 }).state.answers // => [2.5]
 * ```
 */
export function thresholdoutAlgorithm(
  train: LearnData,
  holdout: LearnData,
  analyst: readonly HoldoutQuery[] | Analyst,
  options: ThresholdoutOptions = {},
): Algorithm<ThresholdoutState> {
  const tr = sampleOf('thresholdout', train)
  const ho = sampleOf('thresholdout', holdout)
  if (tr.d !== ho.d) throw new MdmlError('shape', 'thresholdout: training and holdout widths differ')
  const T = options.threshold ?? 0.04
  const sigma = options.sigma ?? T / 10
  const B = options.budget ?? 100
  if (!(T >= 0 && sigma > 0 && B >= 1))
    throw new MdmlError('value', 'thresholdout: needs threshold >= 0, sigma > 0, budget >= 1')
  const lap = (k: Key, scale: number): number => toFlat(laplace(k, [1], 0, scale))[0] ?? 0
  const next = (answers: readonly number[]): HoldoutQuery | undefined =>
    typeof analyst === 'function' ? analyst(answers) : analyst[answers.length]
  return {
    name: 'thresholdout',
    config: { threshold: T, sigma, budget: B, train: tr.n, holdout: ho.n },
    schema: channels({
      metrics: {
        answer: { description: 'the mechanism answer to the query' },
        gap: { description: 'holdout average minus training average of the query' },
        budget: { description: 'holdout releases left' },
      },
    }),
    init: (key) => ({
      t: 0,
      budget: B,
      noisyThreshold: T + lap(foldIn(key, 7), 2 * sigma),
      answers: [],
      released: [],
      trainValues: [],
      holdoutValues: [],
    }),
    step: (s, ctx) => {
      const q = next(s.answers)
      if (q === undefined) return s
      const a = average(tr, q)
      const h = average(ho, q)
      let { budget, noisyThreshold } = s
      let answer: number
      let released = false
      if (budget < 1) {
        answer = Number.NaN
      } else if (Math.abs(h - a) > noisyThreshold + lap(foldIn(ctx.key, 0), 4 * sigma)) {
        answer = h + lap(foldIn(ctx.key, 1), sigma)
        released = true
        budget -= 1
        noisyThreshold = T + lap(foldIn(ctx.key, 2), 2 * sigma)
      } else {
        answer = a
      }
      return {
        t: s.t + 1,
        budget,
        noisyThreshold,
        answers: [...s.answers, answer],
        released: [...s.released, released],
        trainValues: [...s.trainValues, a],
        holdoutValues: [...s.holdoutValues, h],
      }
    },
    done: (s) => (next(s.answers) === undefined ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        answer: s.answers[s.answers.length - 1] ?? Number.NaN,
        gap: (s.holdoutValues[s.holdoutValues.length - 1] ?? 0) - (s.trainValues[s.trainValues.length - 1] ?? 0),
        budget: s.budget,
      },
    }),
    serialise: (s) => JSON.parse(JSON.stringify(s, (_, v) => (typeof v === 'number' && Number.isNaN(v) ? null : v))),
    deserialise: (j) => {
      const o = j as unknown as ThresholdoutState
      return { ...o, answers: o.answers.map((v) => v ?? Number.NaN) }
    },
  }
}

/** Options of the run helpers. */
export interface AskOptions extends ThresholdoutOptions {
  /** The key for the mechanism's noise (default `key(0)`). */
  readonly key?: Key
  /** Cap on the number of queries when the analyst is a function (default 100000). */
  readonly maxQueries?: number
}

/**
 * Run Thresholdout to the end of the analyst's queries and return the final state: a convenience over
 * {@link thresholdoutAlgorithm} for scripts that only want the answers.
 *
 * @param train - The training sample.
 * @param holdout - The holdout sample.
 * @param analyst - Queries, or a function of the answers so far.
 * @param options - Threshold, sigma, budget, key and a cap on the number of queries.
 * @returns The final state, with every answer and whether it came from the holdout.
 * @concept adaptive-data-analysis
 * @example
 * ```ts
 * import { thresholdout } from 'mdml-engine/learn/model-selection'
 * const d = { x: [[1], [2], [3], [4]], y: [0, 0, 0, 0] }
 * thresholdout(d, d, [(x) => x[0] as number, (x) => (x[0] as number) ** 2]).answers // => [2.5, 7.5]
 * ```
 */
export function thresholdout(
  train: LearnData,
  holdout: LearnData,
  analyst: readonly HoldoutQuery[] | Analyst,
  options: AskOptions = {},
): ThresholdoutState {
  const { key, maxQueries, ...rest } = options
  const alg = thresholdoutAlgorithm(train, holdout, analyst, rest)
  const limit = typeof analyst === 'function' ? (maxQueries ?? 100000) : analyst.length
  const result = runAlgorithm(alg, { key: key ?? makeKey(0), maxSteps: limit })
  return result.state
}

/** Options of {@link ladderAlgorithm}. */
export interface LadderOptions {
  /** The step size eta: scores are reported to this resolution and move only for improvements larger than it (default 0.01). */
  readonly step?: number
  /** The score to start from (default 1, the worst possible error). */
  readonly initial?: number
}

/** State of {@link ladderAlgorithm}. */
export interface LadderState {
  readonly t: number
  /** The reported leaderboard score after each submission (lower is better). */
  readonly reported: readonly number[]
  /** The true holdout loss of each submission. */
  readonly losses: readonly number[]
}

/**
 * The Ladder mechanism (Blum and Hardt 2015) for a leaderboard of lower-is-better scores: given the holdout loss R of a
 * new submission and the best reported score so far b, report b unchanged unless R < b - eta, in which case report R
 * rounded to the nearest multiple of eta. The reported sequence reveals so little about the holdout that no analyst can
 * overfit it by more than about (log k / n)^(1/3) after k submissions, where the plain leaderboard falls to the
 * boosting attack of {@link leaderboardAttack} with an error of order sqrt(k / n).
 *
 * @param losses - The holdout loss of each submission, in order.
 * @param options - The step size eta and the initial score.
 * @returns The algorithm, one step per submission.
 * @concept adaptive-data-analysis
 * @example
 * ```ts
 * import { ladderAlgorithm } from 'mdml-engine/learn/model-selection'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = ladderAlgorithm([0.5, 0.495, 0.42, 0.43], { step: 0.05 })
 * runAlgorithm(alg, { maxSteps: 4 }).state.reported // => [0.5, 0.5, 0.4, 0.4]
 * ```
 */
export function ladderAlgorithm(losses: readonly number[], options: LadderOptions = {}): Algorithm<LadderState> {
  const eta = options.step ?? 0.01
  const start = options.initial ?? 1
  if (!(eta > 0)) throw new MdmlError('value', 'ladder: step must be positive')
  return {
    name: 'ladder',
    config: { step: eta, initial: start, submissions: losses.length },
    schema: channels({
      metrics: {
        reported: { description: 'the leaderboard score after the submission', direction: 'lower' },
        loss: { description: 'the true holdout loss of the submission', direction: 'lower' },
      },
    }),
    init: () => ({ t: 0, reported: [], losses: [] }),
    step: (s) => {
      const loss = losses[s.t]
      if (loss === undefined) throw new MdmlError('value', 'ladder: no submission for this step')
      const best = s.reported[s.reported.length - 1] ?? start
      const next = loss < best - eta ? Math.round(loss / eta) * eta : best
      return { t: s.t + 1, reported: [...s.reported, next], losses: [...s.losses, loss] }
    },
    done: (s) => (s.t >= losses.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        reported: s.reported[s.reported.length - 1] ?? start,
        loss: s.losses[s.losses.length - 1] ?? Number.NaN,
      },
    }),
    serialise: (s) => ({ t: s.t, reported: [...s.reported], losses: [...s.losses] }),
    deserialise: (j) => j as unknown as LadderState,
  }
}

/** Options of {@link adaptiveOverfittingDemo} and {@link leaderboardAttack}. */
export interface AttackOptions {
  /** Examples in each of the training, holdout and fresh samples (default 1000). */
  readonly n?: number
  /** Number of random sign features, i.e. classifiers the analyst tries (default 1000). */
  readonly features?: number
  readonly key?: Key
  /** Thresholdout parameters (defaults: T = 2 / sqrt(n) and sigma = T / 10; budget 100). */
  readonly threshold?: number
  readonly sigma?: number
  readonly budget?: number
}

/** What the analyst concludes and what is true. */
export interface AttackResult {
  /** The accuracy of the final classifier as the mechanism reports it. */
  readonly reported: number
  /** Its accuracy on the holdout (what reuse has overfitted). */
  readonly holdoutAccuracy: number
  /** Its accuracy on the training sample. */
  readonly trainAccuracy: number
  /** Its accuracy on a fresh sample, the truth: 0.5, since the labels are noise. */
  readonly freshAccuracy: number
  /** Number of classifiers that entered the combination. */
  readonly selected: number
}

function signs(k: Key, n: number, d: number): Float64Array {
  return Float64Array.from(toFlat(randint(k, [n * d], 0, 2)), (v) => 2 * v - 1)
}

function planted(k: Key, n: number, d: number): Sample {
  return { n, d, x: signs(foldIn(k, 0), n, d), y: signs(foldIn(k, 1), n, 1) }
}

const asData = (s: Sample): LearnData => ({
  x: Array.from({ length: s.n }, (_, i) => Array.from(s.x.subarray(i * s.d, (i + 1) * s.d))),
  y: Array.from(s.y),
})

function accuracyOf(s: Sample, w: Float64Array): number {
  let ok = 0
  for (let i = 0; i < s.n; i++) {
    let z = 0
    for (let j = 0; j < s.d; j++) z += (w[j] ?? 0) * (s.x[i * s.d + j] ?? 0)
    if ((z >= 0 ? 1 : -1) === s.y[i]) ok++
  }
  return ok / s.n
}

/**
 * The planted overfitting problem of adaptive data analysis. The labels are independent of the features (every
 * classifier is right half the time), yet an analyst who asks each feature's accuracy as a statistical query,
 * weights the features by the sign of the answers' deviation from one half, and asks the accuracy of the combination
 * obtains a classifier that looks far better than chance whenever the answers come from the holdout. With
 * `'naive'` every answer is the holdout average and the final reported accuracy is inflated; with `'thresholdout'` the
 * answers come from the training sample unless the two disagree, and the final answer is honest (close to one half).
 *
 * @param mechanism - `'naive'` (answer from the holdout), `'train'` (answer from the training sample) or `'thresholdout'`.
 * @param options - Sample size, number of features, key and Thresholdout parameters.
 * @returns The reported accuracy, the accuracy on each sample and on fresh data (the truth, one half).
 * @concept adaptive-data-analysis
 * @example
 * ```ts
 * import { adaptiveOverfittingDemo } from 'mdml-engine/learn/model-selection'
 * const naive = adaptiveOverfittingDemo('naive', { n: 200, features: 150 })
 * naive.reported > naive.freshAccuracy + 0.05 // => true
 * ```
 */
export function adaptiveOverfittingDemo(
  mechanism: 'naive' | 'train' | 'thresholdout',
  options: AttackOptions = {},
): AttackResult {
  const n = options.n ?? 1000
  const d = options.features ?? 1000
  const k = options.key ?? makeKey(0)
  const tr = planted(foldIn(k, 10), n, d)
  const ho = planted(foldIn(k, 11), n, d)
  const fresh = planted(foldIn(k, 12), n, d)
  const T = options.threshold ?? 2 / Math.sqrt(n)
  const featureQueries: HoldoutQuery[] = Array.from({ length: d }, (_, j) => (x, y) => ((x[j] ?? 0) === y ? 1 : 0))
  let answers: readonly number[]
  if (mechanism === 'thresholdout') {
    answers = thresholdout(asData(tr), asData(ho), featureQueries, {
      threshold: T,
      sigma: options.sigma ?? T / 10,
      budget: options.budget ?? 100,
      key: foldIn(k, 13),
    }).answers
  } else {
    const source = mechanism === 'naive' ? ho : tr
    answers = featureQueries.map((q) => average(source, q))
  }
  const w = Float64Array.from(answers, (a) => (Number.isNaN(a) ? 0 : Math.sign(a - 0.5)))
  const finalQuery: HoldoutQuery = (x, y) => {
    let z = 0
    for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (x[j] ?? 0)
    return (z >= 0 ? 1 : -1) === y ? 1 : 0
  }
  let reported: number
  if (mechanism === 'thresholdout') {
    reported =
      thresholdout(asData(tr), asData(ho), [finalQuery], {
        threshold: T,
        sigma: options.sigma ?? T / 10,
        key: foldIn(k, 14),
      }).answers[0] ?? Number.NaN
  } else {
    reported = average(mechanism === 'naive' ? ho : tr, finalQuery)
  }
  return {
    reported,
    holdoutAccuracy: accuracyOf(ho, w),
    trainAccuracy: accuracyOf(tr, w),
    freshAccuracy: accuracyOf(fresh, w),
    selected: w.reduce((a, v) => a + (v === 0 ? 0 : 1), 0),
  }
}

/** The outcome of {@link leaderboardAttack}. */
export interface LeaderboardResult {
  /** The best score the leaderboard shows (error rate, lower is better). */
  readonly reportedError: number
  /** The error of the final submission on the holdout. */
  readonly holdoutError: number
  /** Its error on fresh data, one half in truth. */
  readonly freshError: number
  /** The reported score after each submission. */
  readonly trajectory: readonly number[]
}

/**
 * The boosting attack on a leaderboard (Blum and Hardt 2015, section 4): with labels that are pure noise, submit k
 * random sign vectors, keep the ones whose holdout error was below one half (their sign-flipped complements otherwise),
 * and submit their majority vote. Against a plain leaderboard that shows the exact holdout error the final submission
 * claims an error of about 1/2 - c sqrt(k / n) while its true error is 1/2; against the Ladder the reported score
 * moves in steps of eta only on a genuine improvement, and the attack gains almost nothing.
 *
 * @param mechanism - `'naive'` (the leaderboard reports the holdout error of the best submission) or `'ladder'`.
 * @param options - The size n, the number of submissions (`features`), the Ladder's step through `threshold` (default
 *   1 / sqrt(n)) and the key.
 * @returns The reported, holdout and fresh errors and the reported trajectory.
 * @concept adaptive-data-analysis
 * @example
 * ```ts
 * import { leaderboardAttack } from 'mdml-engine/learn/model-selection'
 * const r = leaderboardAttack('naive', { n: 200, features: 150 })
 * r.reportedError < 0.5 - 0.05 // => true
 * ```
 */
export function leaderboardAttack(mechanism: 'naive' | 'ladder', options: AttackOptions = {}): LeaderboardResult {
  const n = options.n ?? 1000
  const d = options.features ?? 1000
  const k = options.key ?? makeKey(0)
  const ho = planted(foldIn(k, 11), n, d)
  const fresh = planted(foldIn(k, 12), n, d)
  const eta = options.threshold ?? 1 / Math.sqrt(n)
  const errorOf = (s: Sample, w: Float64Array): number => 1 - accuracyOf(s, w)
  const unit = (j: number): Float64Array => Float64Array.from({ length: d }, (_, i) => (i === j ? 1 : 0))
  const single = Array.from({ length: d }, (_, j) => errorOf(ho, unit(j)))
  const ladderOf = (losses: readonly number[]): readonly number[] =>
    runAlgorithm(ladderAlgorithm(losses, { step: eta, initial: 0.5 }), { maxSteps: losses.length }).state.reported
  let w: Float64Array
  let shown: readonly number[]
  if (mechanism === 'naive') {
    // The plain leaderboard shows each submission's holdout error; the analyst keeps a feature for the side it favours.
    w = Float64Array.from(single, (e) => Math.sign(0.5 - e))
    const all = [...single, errorOf(ho, w)]
    shown = all.map((_, i) => Math.min(0.5, minOf(all.slice(0, i + 1))))
  } else {
    // The Ladder shows only the stepped running best; the analyst learns which submissions moved it.
    const first = ladderOf(single)
    w = Float64Array.from(first, (r, j) => (r < (j === 0 ? 0.5 : (first[j - 1] ?? 0.5)) ? 1 : 0))
    shown = ladderOf([...single, errorOf(ho, w)])
  }
  return {
    reportedError: shown[shown.length - 1] ?? 0.5,
    holdoutError: errorOf(ho, w),
    freshError: errorOf(fresh, w),
    trajectory: shown,
  }
}

/**
 * Changepoint detection: locating the times where a series' statistical behaviour changes. Bayesian online changepoint
 * detection (Adams and MacKay), which tracks the posterior over the current run length with a conjugate segment model
 * (`segment-models.ts`) and a hazard, as a steppable algorithm whose state serialises (`bocpdAlgorithm`) and as a
 * function over a whole series (`bocpd`); and the offline methods PELT (exact penalised optimal partitioning) and binary
 * segmentation, as in ruptures.
 */

import { logSumExpOf, MdmlError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { type ConjugateModel, type RunStats, segmentOps } from './segment-models.ts'

/**
 * The constant hazard of a geometric prior on segment lengths with mean lambda: h(r) = 1 / lambda.
 *
 * @param lambda - The expected segment length (> 1).
 * @returns The hazard function of the run length.
 * @throws MdmlError with code `value` for lambda <= 1.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.constantHazard(100)(7) // => 0.01
 * ```
 */
export function constantHazard(lambda: number): (runLength: number) => number {
  if (!(lambda > 1)) throw new MdmlError('value', 'constantHazard: the mean segment length must exceed 1')
  return () => 1 / lambda
}

/**
 * The hazard of a prior on segment lengths given by its survival function: h(r) = (S(r) - S(r + 1)) / S(r). With
 * S(r) = P(T >= r) for the segment length T (the number of observations in a segment), this is P(T = r) / P(T >= r),
 * the probability that a segment which has r observations ends there; Weibull, gamma or negative-binomial lengths are
 * such priors (see `weibullHazard` and its siblings).
 *
 * @param survival - S(r) for r = 0, 1, 2, ...
 * @returns The hazard function.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * // Geometric lengths: the hazard is constant.
 * Math.round(filtering.survivalHazard((r) => 0.9 ** r)(3) * 1e12) / 1e12 // => 0.1
 * ```
 */
export function survivalHazard(survival: (r: number) => number): (runLength: number) => number {
  return (r) => {
    const s = survival(r)
    return s <= 0 ? 1 : Math.min(1, Math.max(0, (s - survival(r + 1)) / s))
  }
}

/** The output of {@link bocpd}. */
export interface BocpdResult {
  /** log p(y_t | y_1..y_{t-1}) per step. */
  readonly logPredictive: readonly number[]
  /** E[y_t | y_1..y_{t-1}] per step. */
  readonly predictiveMean: readonly number[]
  /** The standard deviation of the predictive of y_t given y_1..y_{t-1} (the mixture over the run lengths). */
  readonly predictiveStd: readonly number[]
  /** p(r_t = r | y_1..y_t): r counts the observations of the current segment including y_t (entry 0 is 0). */
  readonly runLength: readonly Float64Array[]
  /** The most probable run length per step. */
  readonly mapRunLength: readonly number[]
  /** p(a segment starts at y_t | y_1..y_t) = p(r_t = 1 | y_1..y_t). */
  readonly changeProbability: readonly number[]
  /** log p(y_1..y_T). */
  readonly logEvidence: number
}

/** One hypothesis of the run-length posterior: its log probability and the run's statistics (see {@link RunStats}). */
export interface BocpdHypothesis extends RunStats {
  readonly logp: number
}

/** What the state keeps of the steps so far (the same fields as {@link BocpdResult}, without the evidence). */
export type BocpdHistory = Omit<BocpdResult, 'logEvidence'>

/**
 * The state of {@link bocpdAlgorithm}: the steps done, the posterior over the run length before the next observation as
 * a list of hypotheses (plain numbers, so `stateToJson` serialises it and a run resumes from a checkpoint), the evidence
 * so far and what every step said.
 */
export interface BocpdState {
  /** Observations consumed. */
  readonly t: number
  /** p(run length) before observation `t`: the log probability and statistics of every hypothesis kept. */
  readonly hyps: readonly BocpdHypothesis[]
  /** log p(y_1..y_t). */
  readonly logEvidence: number
  readonly history: BocpdHistory
}

/** The paths {@link bocpdAlgorithm} logs, as a type. */
export type BocpdPath =
  | 'bocpd/change-probability'
  | 'bocpd/map-run-length'
  | 'bocpd/log-predictive'
  | 'bocpd/log-evidence'
  | 'bocpd/predictive-mean'
  | 'bocpd/predictive-std'
  | 'bocpd/hypotheses'
  | 'bocpd/run-length'

/**
 * What {@link bocpdAlgorithm} logs, by path: the probability that a segment starts at this step, the most probable run
 * length, the log predictive of the observation, the evidence so far, the predictive mean and standard deviation, the
 * number of hypotheses kept, and the run-length posterior (a row per step: entry r is p(r_t = r | y_1..y_t)).
 */
export const bocpdContract: Contract<BocpdState, BocpdPath> = defineContract({
  'bocpd/change-probability': scalar({ description: 'p(a segment starts at this step)' }),
  'bocpd/map-run-length': scalar({ description: 'the most probable run length' }),
  'bocpd/log-predictive': scalar({ description: 'log p(y_t | y_1..y_{t-1})' }),
  'bocpd/log-evidence': scalar({ description: 'log p(y_1..y_t), the evidence so far', direction: 'higher' }),
  'bocpd/predictive-mean': scalar({ description: 'E[y_t | y_1..y_{t-1}]' }),
  'bocpd/predictive-std': scalar({ description: 'the standard deviation of y_t given y_1..y_{t-1}' }),
  'bocpd/hypotheses': scalar({ description: 'run lengths kept after pruning' }),
  'bocpd/run-length': tensor({
    role: 'output',
    shape: ['run length'],
    description: 'p(r_t = r | y_1..y_t) for r = 0, 1, ..., t + 1 (entry 0 is 0)',
  }),
})

/** Options of {@link bocpdAlgorithm}. */
export interface BocpdOptions {
  /** Log probability below which a hypothesis is dropped (default -40, which changes nothing at double precision). */
  readonly threshold?: number
  /**
   * Keep every step's run-length posterior in the state (default true). A long series makes the state quadratic in its
   * length; without them `history.runLength` is empty and the paths still log the rows.
   */
  readonly rows?: boolean
}

const EMPTY_HISTORY: BocpdHistory = {
  logPredictive: [],
  predictiveMean: [],
  predictiveStd: [],
  runLength: [],
  mapRunLength: [],
  changeProbability: [],
}

/**
 * Bayesian online changepoint detection (Adams and MacKay 2007) as a steppable algorithm: each step consumes one
 * observation, propagates the run-length posterior exactly with the conjugate predictive of every run and a hazard
 * (constant or a function of the run length), and drops hypotheses below e^threshold of the total (default -40, which
 * changes nothing at double precision). The state is plain numbers and serialises, so a run resumes from a checkpoint;
 * every step logs the change probability, the MAP run length, the log predictive, the evidence so far, the predictive
 * mean and standard deviation, the number of hypotheses and the run-length row (see {@link bocpdContract}).
 *
 * @param y - The series.
 * @param model - The conjugate segment model and its prior.
 * @param hazard - A constant hazard in (0, 1) or a function of the run length.
 * @param options - Pruning threshold and whether the state keeps the run-length rows.
 * @returns The algorithm; it is done after the last observation.
 * @throws MdmlError with code `value` for a constant hazard outside (0, 1) or a model with an invalid prior.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const model = { kind: 'normal-known-variance', sigma: 0.5, priorMean: 0, priorStd: 5 } as const
 * const alg = filtering.bocpdAlgorithm([0, 0.1, 5, 5.2], model, 0.1)
 * runAlgorithm(alg, { maxSteps: 4 }).state.history.mapRunLength // => [1, 2, 1, 2]
 * ```
 */
export function bocpdAlgorithm(
  y: readonly number[],
  model: ConjugateModel,
  hazard: number | ((runLength: number) => number),
  options: BocpdOptions = {},
): Algorithm<BocpdState, BocpdPath> {
  if (typeof hazard === 'number' && !(hazard > 0 && hazard < 1))
    throw new MdmlError('value', 'bocpd: the hazard must be in (0, 1)')
  const h = typeof hazard === 'number' ? () => hazard : hazard
  const cut = options.threshold ?? -40
  const rows = options.rows ?? true
  const ops = segmentOps(model)
  return {
    name: 'bocpd',
    config: { model: model.kind, steps: y.length, threshold: cut },
    init: () => ({
      t: 0,
      hyps: [{ logp: 0, ...ops.fresh(0) }],
      logEvidence: 0,
      history: EMPTY_HISTORY,
    }),
    step: (s, ctx) => {
      const t = s.t
      const yt = y[t] ?? 0
      const at = { t, prev: t > 0 ? (y[t - 1] ?? 0) : 0 }
      // The mixture over the run lengths before y_t: its mean and, from the second moment, its spread.
      let mean = 0
      let second = 0
      for (const p of s.hyps) {
        const w = Math.exp(p.logp)
        const m = ops.predictiveMean(p, at)
        mean += w * m
        second += w * (ops.predictiveVariance(p, at) + m * m)
      }
      const joint = s.hyps.map((p) => p.logp + ops.logPredictive(p, yt, at))
      const lp = logSumExpOf(joint)
      const posterior = s.hyps.map((p, i) => ({ logp: (joint[i] ?? 0) - lp, ...ops.update(p, yt, at) }))
      const rl = new Float64Array(t + 2)
      let best = 0
      for (const p of posterior) {
        rl[p.length] = Math.exp(p.logp)
        if ((rl[p.length] ?? 0) > (rl[best] ?? 0)) best = p.length
      }
      const std = Math.sqrt(Math.max(0, second - mean * mean))
      const logEvidence = s.logEvidence + lp
      const kept = posterior.filter((p) => p.logp > cut)
      // Before y_(t+1): a run of length r continues with probability 1 - h(r), or a new segment starts with h(r).
      const change = logSumExpOf(kept.map((p) => p.logp + Math.log(h(p.length))))
      const next = kept.flatMap((p) => {
        const stay = 1 - h(p.length)
        return stay > 0 ? [{ ...p, logp: p.logp + Math.log(stay) }] : []
      })
      let hyps: BocpdHypothesis[] =
        change === Number.NEGATIVE_INFINITY ? next : [{ logp: change, ...ops.fresh(t + 1) }, ...next]
      const z = logSumExpOf(hyps.map((p) => p.logp))
      hyps = hyps.map((p) => ({ ...p, logp: p.logp - z }))
      if (ctx.wants('bocpd/change-probability')) ctx.log('bocpd/change-probability', rl[1] ?? 0)
      if (ctx.wants('bocpd/map-run-length')) ctx.log('bocpd/map-run-length', best)
      if (ctx.wants('bocpd/log-predictive')) ctx.log('bocpd/log-predictive', lp)
      if (ctx.wants('bocpd/log-evidence')) ctx.log('bocpd/log-evidence', logEvidence)
      if (ctx.wants('bocpd/predictive-mean')) ctx.log('bocpd/predictive-mean', mean)
      if (ctx.wants('bocpd/predictive-std')) ctx.log('bocpd/predictive-std', std)
      if (ctx.wants('bocpd/hypotheses')) ctx.log('bocpd/hypotheses', hyps.length)
      if (ctx.wants('bocpd/run-length')) ctx.log('bocpd/run-length', rl)
      const past = s.history
      return {
        t: t + 1,
        hyps,
        logEvidence,
        history: {
          logPredictive: [...past.logPredictive, lp],
          predictiveMean: [...past.predictiveMean, mean],
          predictiveStd: [...past.predictiveStd, std],
          runLength: rows ? [...past.runLength, rl] : past.runLength,
          mapRunLength: [...past.mapRunLength, best],
          changeProbability: [...past.changeProbability, rl[1] ?? 0],
        },
      }
    },
    done: (s) => (s.t >= y.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BocpdState,
  }
}

/**
 * Bayesian online changepoint detection over a whole series: {@link bocpdAlgorithm} run to the end.
 *
 * @param y - The series.
 * @param model - The conjugate segment model.
 * @param hazard - A constant hazard in (0, 1) or a function of the run length.
 * @param options - Pruning threshold, whether to keep the run-length rows, and tracing.
 * @returns Run-length posteriors, MAP run lengths, change probabilities, predictives and the evidence.
 * @throws MdmlError with code `value` for a constant hazard outside (0, 1).
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * const r = filtering.bocpd([0, 0.1, 5, 5.2], { kind: 'normal-known-variance', sigma: 0.5, priorMean: 0, priorStd: 5 }, 0.1)
 * r.mapRunLength // => [1, 2, 1, 2]
 * ```
 */
export function bocpd(
  y: readonly number[],
  model: ConjugateModel,
  hazard: number | ((runLength: number) => number),
  options: BocpdOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): BocpdResult {
  const { threshold, rows, ...run } = options
  const algorithm = bocpdAlgorithm(y, model, hazard, {
    ...(threshold === undefined ? {} : { threshold }),
    ...(rows === undefined ? {} : { rows }),
  })
  const st = runAlgorithm(algorithm, { maxSteps: y.length, ...run }).state
  return { ...st.history, logEvidence: st.logEvidence }
}

/** A segment cost for offline changepoint detection. */
export type SegmentCost = 'l2' | 'l1' | 'normal' | 'poisson'

/** The cost of y[start..end) under a cost model, with prefix sums for the closed-form ones. */
function costFunction(
  y: readonly number[],
  cost: SegmentCost,
): { error: (a: number, b: number) => number; minSize: number } {
  const n = y.length
  const s1 = new Float64Array(n + 1)
  const s2 = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    s1[i + 1] = (s1[i] ?? 0) + (y[i] ?? 0)
    s2[i + 1] = (s2[i] ?? 0) + (y[i] ?? 0) ** 2
  }
  const sse = (a: number, b: number): number => {
    const m = b - a
    const sum = (s1[b] ?? 0) - (s1[a] ?? 0)
    return Math.max(0, (s2[b] ?? 0) - (s2[a] ?? 0) - (sum * sum) / m)
  }
  switch (cost) {
    case 'l2':
      return { error: sse, minSize: 1 }
    case 'normal':
      // ruptures' CostNormal in one dimension: n log(var + 1e-6).
      return { error: (a, b) => (b - a) * Math.log(sse(a, b) / (b - a) + 1e-6), minSize: 2 }
    case 'l1':
      return {
        error: (a, b) => {
          const seg = y.slice(a, b).sort((p, q) => p - q)
          const k = seg.length
          const med = k % 2 === 1 ? (seg[(k - 1) / 2] ?? 0) : 0.5 * ((seg[k / 2 - 1] ?? 0) + (seg[k / 2] ?? 0))
          return seg.reduce((acc, v) => acc + Math.abs(v - med), 0)
        },
        minSize: 2,
      }
    case 'poisson':
      // Twice the negative Poisson log-likelihood at the segment's rate, without the log y! terms.
      return {
        error: (a, b) => {
          const m = b - a
          const sum = (s1[b] ?? 0) - (s1[a] ?? 0)
          const rate = sum / m
          return rate === 0 ? 0 : 2 * (m * rate - sum * Math.log(rate))
        },
        minSize: 1,
      }
  }
}

/** Options of {@link pelt} and {@link binarySegmentation}. */
export interface OfflineOptions {
  readonly cost?: SegmentCost
  /** Minimum segment length (default 2, at least the cost's). */
  readonly minSize?: number
  /** Candidate changepoints on a grid of this step (default 1). */
  readonly jump?: number
}

/**
 * PELT, pruned exact linear time optimal partitioning (Killick, Fearnhead and Eckley 2012): the segmentation
 * minimising the sum of segment costs plus a penalty per segment, found by dynamic programming with pruning of
 * candidates that can never be optimal. Unlike ruptures' `Pelt`, a candidate is pruned only once every later segment
 * from the pruning point could meet the minimum length, so the result stays exact when `minSize` exceeds 1.
 *
 * @param y - The series.
 * @param penalty - The penalty per segment (BIC: 2 log n for a mean change at unit variance).
 * @param options - Cost (default l2), minimum segment length and grid step.
 * @returns The segment end indices (sorted, the last is n), as ruptures reports them.
 * @throws MdmlError with code `value` when the series is shorter than one minimum segment.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.pelt([0, 0, 0, 0, 5, 5, 5, 5], 1) // => [4, 8]
 * ```
 */
export function pelt(y: readonly number[], penalty: number, options: OfflineOptions = {}): number[] {
  const n = y.length
  const c = costFunction(y, options.cost ?? 'l2')
  const minSize = Math.max(options.minSize ?? 2, c.minSize)
  const jump = options.jump ?? 1
  if (n < minSize) throw new MdmlError('value', 'pelt: the series is shorter than the minimum segment')
  // F[b]: optimal cost of y[0..b) including a penalty per segment; last[b]: the last changepoint before b.
  const F = new Map<number, number>([[0, 0]])
  const last = new Map<number, number>()
  let admissible: number[] = []
  // PELT's pruning (drop t once F(t) + C(t, b) > F(b)) relies on F(b) + C(b, c) being a valid option for every later
  // end c, which needs c - b >= minSize; so a candidate found prunable at b is only dropped from b + minSize on.
  const prunableAt = new Map<number, number>()
  const ends: number[] = []
  for (let k = 0; k < n; k += jump) if (k >= minSize) ends.push(k)
  ends.push(n)
  for (const b of ends) {
    admissible.push(Math.floor((b - minSize) / jump) * jump)
    const scores: { t: number; v: number }[] = []
    for (const t of admissible) {
      const ft = F.get(t)
      if (ft === undefined) continue
      scores.push({ t, v: ft + c.error(t, b) + penalty })
    }
    let best = scores[0]
    for (const sc of scores) if (best === undefined || sc.v < best.v) best = sc
    if (best === undefined) continue
    F.set(b, best.v)
    last.set(b, best.t)
    for (const sc of scores) if (sc.v > best.v + penalty && !prunableAt.has(sc.t)) prunableAt.set(sc.t, b)
    admissible = admissible.filter((t) => {
      const at = prunableAt.get(t)
      return at === undefined || b + jump < at + minSize
    })
  }
  const out: number[] = []
  let b = n
  while (b > 0) {
    out.push(b)
    b = last.get(b) ?? 0
  }
  return out.reverse()
}

/**
 * Binary segmentation (as ruptures' `Binseg`): repeatedly split the segment whose best single split lowers the cost
 * most, until a number of changepoints is reached or the best gain no longer exceeds a penalty.
 *
 * @param y - The series.
 * @param stop - `{ changepoints }` or `{ penalty }`.
 * @param options - Cost (default l2), minimum segment length and grid step.
 * @returns The segment end indices (sorted, the last is n).
 * @throws MdmlError with code `value` without a stopping rule.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.binarySegmentation([0, 0, 0, 5, 5, 5, 1, 1, 1], { changepoints: 2 }) // => [3, 6, 9]
 * ```
 */
export function binarySegmentation(
  y: readonly number[],
  stop: { readonly changepoints?: number; readonly penalty?: number },
  options: OfflineOptions = {},
): number[] {
  if (stop.changepoints === undefined && stop.penalty === undefined) {
    throw new MdmlError('value', 'binarySegmentation: give a number of changepoints or a penalty')
  }
  const n = y.length
  const c = costFunction(y, options.cost ?? 'l2')
  const minSize = Math.max(options.minSize ?? 2, c.minSize)
  const jump = options.jump ?? 1
  const single = (start: number, end: number): { bkp: number | undefined; gain: number } => {
    const whole = c.error(start, end)
    let best: { bkp: number | undefined; gain: number } = { bkp: undefined, gain: 0 }
    let found = false
    for (let k = start; k < end; k += jump) {
      if (k - start >= minSize && end - k >= minSize) {
        const gain = whole - c.error(start, k) - c.error(k, end)
        // ruptures takes max over (gain, bkp): the larger breakpoint wins ties.
        if (!found || gain > best.gain || (gain === best.gain && k > (best.bkp ?? -1))) best = { bkp: k, gain }
        found = true
      }
    }
    return found ? best : { bkp: undefined, gain: 0 }
  }
  const bkps = [n]
  for (;;) {
    const starts = [0, ...bkps.slice(0, -1)]
    let choice: { bkp: number | undefined; gain: number } = { bkp: undefined, gain: Number.NEGATIVE_INFINITY }
    starts.forEach((s, i) => {
      const r = single(s, bkps[i] ?? n)
      if (r.gain > choice.gain) choice = r
    })
    if (choice.bkp === undefined) break
    const more =
      stop.changepoints !== undefined
        ? bkps.length - 1 < stop.changepoints
        : choice.gain > (stop.penalty ?? Number.POSITIVE_INFINITY)
    if (!more) break
    bkps.push(choice.bkp)
    bkps.sort((a, b) => a - b)
  }
  return bkps
}

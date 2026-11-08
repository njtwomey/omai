/**
 * Changepoint detection: locating the times where a series' statistical behaviour changes. Bayesian online changepoint
 * detection (Adams and MacKay), which tracks the posterior over the current run length with a conjugate segment model
 * (`segment-models.ts`) and a hazard, as a steppable algorithm whose state serialises (`bocpdAlgorithm`) and as a
 * function over a whole series (`bocpd`); and the offline methods PELT (exact penalised optimal partitioning) and binary
 * segmentation, as in ruptures.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
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
import { costOf, type SegmentCost } from './offline-costs.ts'

export type { SegmentCost } from './offline-costs.ts'

import { type ConjugateModel, type RunStats, segmentOps } from './segment-models.ts'

/**
 * The constant hazard of a geometric prior on segment lengths with mean lambda: h(r) = 1 / lambda.
 *
 * @param lambda - The expected segment length (> 1).
 * @returns The hazard function of the run length.
 * @throws OmaiError with code `value` for lambda <= 1.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.constantHazard(100)(7) // => 0.01
 * ```
 */
export function constantHazard(lambda: number): (runLength: number) => number {
  if (!(lambda > 1)) throw new OmaiError('value', 'constantHazard: the mean segment length must exceed 1')
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
 * import { filtering } from 'omai-engine/dynamics'
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
  /** The probability of that most probable run length per step. */
  readonly mapProbability: readonly number[]
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
  | 'bocpd/map-probability'
  | 'bocpd/log-predictive'
  | 'bocpd/log-evidence'
  | 'bocpd/predictive-mean'
  | 'bocpd/predictive-std'
  | 'bocpd/hypotheses'
  | 'bocpd/run-length'

/**
 * What {@link bocpdAlgorithm} logs, by path: the probability that a segment starts at this step, the most probable run
 * length and its probability, the log predictive of the observation, the evidence so far, the predictive mean and
 * standard deviation, the number of hypotheses kept, and the run-length posterior (a row per step: entry r is
 * p(r_t = r | y_1..y_t)).
 */
export const bocpdContract: Contract<BocpdState, BocpdPath> = defineContract({
  'bocpd/change-probability': scalar({ description: 'p(a segment starts at this step)' }),
  'bocpd/map-run-length': scalar({ description: 'the most probable run length' }),
  'bocpd/map-probability': scalar({ description: 'the probability of the most probable run length' }),
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
  mapProbability: [],
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
 * @throws OmaiError with code `value` for a constant hazard outside (0, 1) or a model with an invalid prior.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
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
    throw new OmaiError('value', 'bocpd: the hazard must be in (0, 1)')
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
      if (ctx.wants('bocpd/map-probability')) ctx.log('bocpd/map-probability', rl[best] ?? 0)
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
          mapProbability: [...past.mapProbability, rl[best] ?? 0],
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
 * @throws OmaiError with code `value` for a constant hazard outside (0, 1).
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
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

/** A changepoint read off the run-length posterior by {@link mapChangepoints}. */
export interface MapChangepoint {
  /** The index of the first observation of the new segment. */
  readonly index: number
  /** The step at which the most probable run length first put the start there with enough probability. */
  readonly detectedAt: number
  /** That probability: the posterior mass of the most probable run length at the step of detection. */
  readonly probability: number
}

/**
 * Changepoints from the run-length posterior, thresholded: at every step the most probable run length r puts the start
 * of the current segment at t - r + 1; when its probability reaches the threshold, that start is a changepoint (the
 * first time it is, and never the start of the series). A segment that began long ago is reported once, and a change
 * is reported a few observations after it happened, as soon as the posterior commits to it.
 *
 * @param result - The MAP run lengths and their probabilities (a {@link BocpdResult}, or the state's history).
 * @param threshold - The probability the most probable run length must reach (default 0.5).
 * @returns The changepoints in the order they were detected.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * const model = { kind: 'normal-known-variance', sigma: 0.5, priorMean: 0, priorStd: 5 } as const
 * const r = filtering.bocpd([0, 0.1, -0.1, 0, 5, 5.2, 4.9, 5.1], model, 0.1)
 * filtering.mapChangepoints(r, 0.5).map((c) => c.index) // => [4]
 * ```
 */
export function mapChangepoints(
  result: Pick<BocpdResult, 'mapRunLength' | 'mapProbability'>,
  threshold = 0.5,
): MapChangepoint[] {
  const seen = new Set<number>()
  const out: MapChangepoint[] = []
  for (const [t, r] of result.mapRunLength.entries()) {
    const probability = result.mapProbability[t] ?? 0
    const index = t - r + 1
    if (index < 1 || probability < threshold || seen.has(index)) continue
    seen.add(index)
    out.push({ index, detectedAt: t, probability })
  }
  return out
}

/** Options of {@link pelt} and {@link binarySegmentation}. */
export interface OfflineOptions {
  readonly cost?: SegmentCost
  /** Minimum segment length (default 2, at least the cost's). */
  readonly minSize?: number
  /** Candidate changepoints on a grid of this step (default 1). */
  readonly jump?: number
}

/** The state of {@link peltAlgorithm}: the dynamic program over the candidate ends and the candidates still alive. */
export interface PeltState {
  /** Candidate ends processed. */
  readonly k: number
  /** F[b]: the optimal cost of y[0..b) with a penalty per segment (`null` where b is not a candidate end yet). */
  readonly F: readonly (number | null)[]
  /** last[b]: the start of the last segment of that optimum. */
  readonly last: readonly (number | null)[]
  /** The step at which a candidate became prunable (`null`: not yet). */
  readonly prunableAt: readonly (number | null)[]
  /** The candidate starts still admissible. */
  readonly admissible: readonly number[]
  /** The end processed last. */
  readonly end: number
}

/** The paths {@link peltAlgorithm} logs, as a type. */
export type PeltPath = 'pelt/end' | 'pelt/F' | 'pelt/last' | 'pelt/admissible' | 'pelt/pruned'

/**
 * What {@link peltAlgorithm} logs, by path: the end b being solved, its optimal cost F(b), the start of the last
 * segment of that optimum, the candidate starts still alive and the number the step pruned.
 */
export const peltContract: Contract<PeltState, PeltPath> = defineContract({
  'pelt/end': scalar({ description: 'the end b of the prefix being solved' }),
  'pelt/F': scalar({ description: 'F(b): the optimal penalised cost of y[0..b)', direction: 'lower' }),
  'pelt/last': scalar({ description: 'the start of the last segment of that optimum' }),
  'pelt/admissible': tensor({
    role: 'state',
    shape: ['candidate'],
    description: 'the candidate starts still admissible',
  }),
  'pelt/pruned': scalar({ description: 'candidates dropped at this step' }),
})

/**
 * PELT as a steppable algorithm: each step solves the optimal penalised cost F(b) of one more prefix y[0..b) over the
 * admissible last-segment starts, then prunes the candidates that can never be optimal. The state is plain numbers, so
 * it serialises and a run resumes; read the segmentation off the finished state with {@link peltSegments}.
 *
 * @param y - The series.
 * @param penalty - The penalty per segment.
 * @param options - Cost (default l2), minimum segment length and grid step.
 * @returns The algorithm; it is done after the last candidate end.
 * @throws OmaiError with code `value` when the series is shorter than one minimum segment.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const y = [0, 0, 0, 0, 5, 5, 5, 5]
 * const alg = filtering.peltAlgorithm(y, 1)
 * filtering.peltSegments(runAlgorithm(alg, { maxSteps: 100 }).state, y.length) // => [4, 8]
 * ```
 */
export function peltAlgorithm(
  y: readonly number[],
  penalty: number,
  options: OfflineOptions = {},
): Algorithm<PeltState, PeltPath> {
  const n = y.length
  const c = costOf(y, options.cost ?? 'l2')
  const minSize = Math.max(options.minSize ?? 2, c.minSize)
  const jump = options.jump ?? 1
  if (n < minSize) throw new OmaiError('value', 'pelt: the series is shorter than the minimum segment')
  const ends: number[] = []
  for (let k = 0; k < n; k += jump) if (k >= minSize) ends.push(k)
  ends.push(n)
  return {
    name: 'pelt',
    config: { cost: options.cost ?? 'l2', penalty, minSize, jump, steps: ends.length },
    init: () => ({
      k: 0,
      F: Array.from({ length: n + 1 }, (_, i) => (i === 0 ? 0 : null)),
      last: new Array<number | null>(n + 1).fill(null),
      prunableAt: new Array<number | null>(n + 1).fill(null),
      admissible: [],
      end: 0,
    }),
    step: (s, ctx) => {
      const b = ends[s.k] as number
      const F = [...s.F]
      const last = [...s.last]
      const prunableAt = [...s.prunableAt]
      // PELT's pruning (drop t once F(t) + C(t, b) > F(b)) relies on F(b) + C(b, c) being a valid option for every later
      // end c, which needs c - b >= minSize; so a candidate found prunable at b is only dropped from b + minSize on.
      let admissible = [...s.admissible, Math.floor((b - minSize) / jump) * jump]
      const scores: { t: number; v: number }[] = []
      for (const t of admissible) {
        const ft = F[t]
        if (ft === null || ft === undefined) continue
        scores.push({ t, v: ft + c.error(t, b) + penalty })
      }
      let best = scores[0]
      for (const sc of scores) if (best === undefined || sc.v < best.v) best = sc
      if (best === undefined) return { ...s, k: s.k + 1, admissible, end: b }
      F[b] = best.v
      last[b] = best.t
      for (const sc of scores) if (sc.v > best.v + penalty && prunableAt[sc.t] === null) prunableAt[sc.t] = b
      admissible = admissible.filter((t) => {
        const at = prunableAt[t]
        return at === null || at === undefined || b + jump < at + minSize
      })
      const dropped = s.admissible.length + 1 - admissible.length
      if (ctx.wants('pelt/end')) ctx.log('pelt/end', b)
      if (ctx.wants('pelt/F')) ctx.log('pelt/F', best.v)
      if (ctx.wants('pelt/last')) ctx.log('pelt/last', best.t)
      if (ctx.wants('pelt/admissible')) ctx.log('pelt/admissible', admissible)
      if (ctx.wants('pelt/pruned')) ctx.log('pelt/pruned', dropped)
      return { k: s.k + 1, F, last, prunableAt, admissible, end: b }
    },
    done: (s) => (s.k >= ends.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PeltState,
  }
}

/**
 * The segment ends of a finished {@link peltAlgorithm} run: follow the last-segment starts back from the end of the
 * series.
 *
 * @param state - The state of a run to its end.
 * @param n - The length of the series.
 * @returns The segment end indices (sorted, the last is n).
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.peltSegments({ k: 0, F: [0, null, 1], last: [null, null, 0], prunableAt: [], admissible: [], end: 2 }, 2) // => [2]
 * ```
 */
export function peltSegments(state: Pick<PeltState, 'last'>, n: number): number[] {
  const out: number[] = []
  let b = n
  while (b > 0) {
    out.push(b)
    b = state.last[b] ?? 0
  }
  return out.reverse()
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
 * @throws OmaiError with code `value` when the series is shorter than one minimum segment.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.pelt([0, 0, 0, 0, 5, 5, 5, 5], 1) // => [4, 8]
 * ```
 */
export function pelt(y: readonly number[], penalty: number, options: OfflineOptions = {}): number[] {
  const algorithm = peltAlgorithm(y, penalty, options)
  const n = y.length
  return peltSegments(runAlgorithm(algorithm, { maxSteps: n + 1 }).state, n)
}

/** A best single split of a segment. */
export interface Split {
  /** Where to split (`undefined`: the segment cannot be split). */
  readonly bkp: number | undefined
  /** The drop in cost the split buys. */
  readonly gain: number
}

/** The state of {@link binarySegmentationAlgorithm}: the breakpoints so far and the split it would make next. */
export interface BinsegState {
  /** Splits made. */
  readonly step: number
  /** The segment ends so far (sorted, the last is n). */
  readonly bkps: readonly number[]
  /** The best split over all segments (`undefined` when none can be made). */
  readonly next: Split
  /** The split made last, and its gain. */
  readonly split: number | null
  readonly gain: number
}

/** The paths {@link binarySegmentationAlgorithm} logs, as a type. */
export type BinsegPath = 'binseg/split' | 'binseg/gain' | 'binseg/segments' | 'binseg/next-gain'

/**
 * What {@link binarySegmentationAlgorithm} logs, by path: the breakpoint just added, the cost it saved, the number of
 * segments, and the gain of the best split still on offer (what the stopping rule compares).
 */
export const binsegContract: Contract<BinsegState, BinsegPath> = defineContract({
  'binseg/split': scalar({ description: 'the breakpoint added by this step' }),
  'binseg/gain': scalar({ description: 'the cost this split saved', direction: 'higher' }),
  'binseg/segments': scalar({ description: 'the number of segments after this step' }),
  'binseg/next-gain': scalar({ description: 'the gain of the best split still on offer', direction: 'higher' }),
})

/**
 * Binary segmentation as a steppable algorithm: each step adds the best single split over all segments, until a number
 * of changepoints is reached or the best gain no longer exceeds a penalty. The state holds the breakpoints and the
 * split on offer, so it serialises and the stopping rule is read from it.
 *
 * @param y - The series.
 * @param stop - `{ changepoints }` or `{ penalty }`.
 * @param options - Cost (default l2), minimum segment length and grid step.
 * @returns The algorithm; its state's `bkps` are the segment ends.
 * @throws OmaiError with code `value` without a stopping rule.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = filtering.binarySegmentationAlgorithm([0, 0, 0, 5, 5, 5, 1, 1, 1], { changepoints: 2 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.bkps // => [3, 6, 9]
 * ```
 */
export function binarySegmentationAlgorithm(
  y: readonly number[],
  stop: { readonly changepoints?: number; readonly penalty?: number },
  options: OfflineOptions = {},
): Algorithm<BinsegState, BinsegPath> {
  if (stop.changepoints === undefined && stop.penalty === undefined) {
    throw new OmaiError('value', 'binarySegmentation: give a number of changepoints or a penalty')
  }
  const n = y.length
  const c = costOf(y, options.cost ?? 'l2')
  const minSize = Math.max(options.minSize ?? 2, c.minSize)
  const jump = options.jump ?? 1
  const single = (start: number, end: number): Split => {
    const whole = c.error(start, end)
    let best: Split = { bkp: undefined, gain: 0 }
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
  const bestSplit = (bkps: readonly number[]): Split => {
    const starts = [0, ...bkps.slice(0, -1)]
    let choice: Split = { bkp: undefined, gain: Number.NEGATIVE_INFINITY }
    starts.forEach((s, i) => {
      const r = single(s, bkps[i] ?? n)
      if (r.gain > choice.gain) choice = r
    })
    return choice
  }
  const wanted = (s: BinsegState): boolean =>
    s.next.bkp !== undefined &&
    (stop.changepoints !== undefined
      ? s.bkps.length - 1 < stop.changepoints
      : s.next.gain > (stop.penalty ?? Number.POSITIVE_INFINITY))
  return {
    name: 'binary-segmentation',
    config: {
      cost: options.cost ?? 'l2',
      minSize,
      jump,
      changepoints: stop.changepoints ?? null,
      penalty: stop.penalty ?? null,
    },
    init: () => ({ step: 0, bkps: [n], next: bestSplit([n]), split: null, gain: 0 }),
    step: (s, ctx) => {
      const bkp = s.next.bkp as number
      const bkps = [...s.bkps, bkp].sort((a, b) => a - b)
      const next = bestSplit(bkps)
      if (ctx.wants('binseg/split')) ctx.log('binseg/split', bkp)
      if (ctx.wants('binseg/gain')) ctx.log('binseg/gain', s.next.gain)
      if (ctx.wants('binseg/segments')) ctx.log('binseg/segments', bkps.length)
      if (ctx.wants('binseg/next-gain')) ctx.log('binseg/next-gain', next.gain)
      return { step: s.step + 1, bkps, next, split: bkp, gain: s.next.gain }
    },
    done: (s) => (wanted(s) ? undefined : 'converged'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BinsegState,
  }
}

/**
 * Binary segmentation (as ruptures' `Binseg`): repeatedly split the segment whose best single split lowers the cost
 * most, until a number of changepoints is reached or the best gain no longer exceeds a penalty.
 *
 * @param y - The series.
 * @param stop - `{ changepoints }` or `{ penalty }`.
 * @param options - Cost (default l2), minimum segment length and grid step.
 * @returns The segment end indices (sorted, the last is n).
 * @throws OmaiError with code `value` without a stopping rule.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.binarySegmentation([0, 0, 0, 5, 5, 5, 1, 1, 1], { changepoints: 2 }) // => [3, 6, 9]
 * ```
 */
export function binarySegmentation(
  y: readonly number[],
  stop: { readonly changepoints?: number; readonly penalty?: number },
  options: OfflineOptions = {},
): number[] {
  const algorithm = binarySegmentationAlgorithm(y, stop, options)
  return [...runAlgorithm(algorithm, { maxSteps: y.length }).state.bkps]
}

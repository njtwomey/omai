/**
 * Changepoint detection: locating the times where a series' statistical behaviour changes. Bayesian online changepoint
 * detection (Adams and MacKay), which tracks the posterior over the current run length with conjugate segment models
 * and a hazard, and the offline methods PELT (exact penalised optimal partitioning) and binary segmentation, as in
 * ruptures.
 */

import { logSumExpOf, WhiteboxError } from '../../base/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'

/**
 * A conjugate model of the observations within a segment (with its prior), which gives an exact posterior predictive
 * for Bayesian online changepoint detection: Gaussian with known variance (normal prior on the mean), Gaussian with
 * known mean (inverse-gamma prior on the variance), Gaussian with both unknown (normal-gamma, Student-t predictive),
 * Poisson (gamma prior, negative binomial predictive) and Bernoulli (beta prior).
 */
export type ConjugateModel =
  | {
      readonly kind: 'normal-known-variance'
      readonly sigma: number
      readonly priorMean: number
      readonly priorStd: number
    }
  | { readonly kind: 'normal-known-mean'; readonly mean: number; readonly alpha: number; readonly beta: number }
  | {
      readonly kind: 'normal-gamma'
      readonly mu0: number
      readonly kappa0: number
      readonly alpha0: number
      readonly beta0: number
    }
  | { readonly kind: 'poisson-gamma'; readonly shape: number; readonly rate: number }
  | { readonly kind: 'bernoulli-beta'; readonly alpha: number; readonly beta: number }

const LOG_2PI = Math.log(2 * Math.PI)
const lg = (x: number): number => item(lgamma(x))

function studentLog(y: number, nu: number, loc: number, scale2: number): number {
  return (
    lg((nu + 1) / 2) -
    lg(nu / 2) -
    0.5 * Math.log(nu * Math.PI * scale2) -
    ((nu + 1) / 2) * Math.log1p((y - loc) ** 2 / (nu * scale2))
  )
}

/** The posterior of one run: the predictive log density and mean of the next value, and the update by one value. */
export interface RunPosterior {
  readonly length: number
  logPredictive(y: number): number
  predictiveMean(): number
  update(y: number): RunPosterior
}

/**
 * The prior run of a conjugate model (no observations yet).
 *
 * @param model - The model.
 * @returns The run posterior before any observation.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * filtering.priorRun({ kind: 'bernoulli-beta', alpha: 1, beta: 1 }).predictiveMean() // => 0.5
 * ```
 */
export function priorRun(model: ConjugateModel): RunPosterior {
  switch (model.kind) {
    case 'normal-known-variance': {
      const s2 = model.sigma ** 2
      const make = (length: number, precision: number, weighted: number): RunPosterior => {
        const m = weighted / precision
        const v = 1 / precision + s2
        return {
          length,
          logPredictive: (y) => -0.5 * (LOG_2PI + Math.log(v) + (y - m) ** 2 / v),
          predictiveMean: () => m,
          update: (y) => make(length + 1, precision + 1 / s2, weighted + y / s2),
        }
      }
      return make(0, 1 / model.priorStd ** 2, model.priorMean / model.priorStd ** 2)
    }
    case 'normal-known-mean': {
      const mu = model.mean
      const make = (length: number, a: number, b: number): RunPosterior => ({
        length,
        logPredictive: (y) => studentLog(y, 2 * a, mu, b / a),
        predictiveMean: () => mu,
        update: (y) => make(length + 1, a + 0.5, b + 0.5 * (y - mu) ** 2),
      })
      return make(0, model.alpha, model.beta)
    }
    case 'normal-gamma': {
      const make = (length: number, mu: number, kappa: number, a: number, b: number): RunPosterior => ({
        length,
        logPredictive: (y) => studentLog(y, 2 * a, mu, (b * (kappa + 1)) / (a * kappa)),
        predictiveMean: () => mu,
        update: (y) =>
          make(
            length + 1,
            (kappa * mu + y) / (kappa + 1),
            kappa + 1,
            a + 0.5,
            b + (kappa * (y - mu) ** 2) / (2 * (kappa + 1)),
          ),
      })
      return make(0, model.mu0, model.kappa0, model.alpha0, model.beta0)
    }
    case 'poisson-gamma': {
      const make = (length: number, a: number, b: number): RunPosterior => ({
        length,
        logPredictive: (y) => lg(a + y) - lg(a) - lg(y + 1) + a * Math.log(b / (b + 1)) - y * Math.log(b + 1),
        predictiveMean: () => a / b,
        update: (y) => make(length + 1, a + y, b + 1),
      })
      return make(0, model.shape, model.rate)
    }
    case 'bernoulli-beta': {
      const make = (length: number, a: number, b: number): RunPosterior => ({
        length,
        logPredictive: (y) => Math.log(y > 0.5 ? a / (a + b) : b / (a + b)),
        predictiveMean: () => a / (a + b),
        update: (y) => make(length + 1, a + (y > 0.5 ? 1 : 0), b + (y > 0.5 ? 0 : 1)),
      })
      return make(0, model.alpha, model.beta)
    }
  }
}

/**
 * The constant hazard of a geometric prior on segment lengths with mean lambda: h(r) = 1 / lambda.
 *
 * @param lambda - The expected segment length (> 1).
 * @returns The hazard function of the run length.
 * @throws WhiteboxError with code `value` for lambda <= 1.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * filtering.constantHazard(100)(7) // => 0.01
 * ```
 */
export function constantHazard(lambda: number): (runLength: number) => number {
  if (!(lambda > 1)) throw new WhiteboxError('value', 'constantHazard: the mean segment length must exceed 1')
  return () => 1 / lambda
}

/**
 * The hazard of a discretised Weibull or negative-binomial-like prior on segment lengths: h(r) = P(L = r + 1) /
 * P(L > r) for a length distribution given by its survival function.
 *
 * @param survival - P(L > r) for r = 0, 1, 2, ...
 * @returns The hazard function.
 * @concept hazard-function
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
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
  /** p(r_t = r | y_1..y_t): r counts the observations of the current segment including y_t (entry 0 is 0). */
  readonly runLength: readonly Float64Array[]
  /** The most probable run length per step. */
  readonly mapRunLength: readonly number[]
  /** p(a segment starts at y_t | y_1..y_t) = p(r_t = 1 | y_1..y_t). */
  readonly changeProbability: readonly number[]
  /** log p(y_1..y_T). */
  readonly logEvidence: number
}

interface BocpdState {
  readonly hyps: readonly { readonly logp: number; readonly run: RunPosterior }[]
  readonly out: {
    logPredictive: number[]
    predictiveMean: number[]
    runLength: Float64Array[]
    mapRunLength: number[]
    changeProbability: number[]
  }
  readonly logEvidence: number
}

/**
 * Bayesian online changepoint detection (Adams and MacKay 2007) as a traced algorithm: the run-length posterior is
 * propagated exactly with the conjugate predictive of every run and a hazard (constant or a function of the run
 * length); hypotheses below e^threshold of the total are dropped (default -40, which changes nothing at double
 * precision).
 *
 * @param y - The series.
 * @param model - The conjugate segment model.
 * @param hazard - A constant hazard in (0, 1) or a function of the run length.
 * @param options - Pruning threshold and tracing.
 * @returns Run-length posteriors, MAP run lengths, change probabilities, predictives and the evidence.
 * @throws WhiteboxError with code `value` for a constant hazard outside (0, 1).
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * const r = filtering.bocpd([0, 0.1, 5, 5.2], { kind: 'normal-known-variance', sigma: 0.5, priorMean: 0, priorStd: 5 }, 0.1)
 * r.mapRunLength // => [1, 2, 1, 2]
 * ```
 */
export function bocpd(
  y: readonly number[],
  model: ConjugateModel,
  hazard: number | ((runLength: number) => number),
  options: { readonly threshold?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): BocpdResult {
  if (typeof hazard === 'number' && !(hazard > 0 && hazard < 1))
    throw new WhiteboxError('value', 'bocpd: the hazard must be in (0, 1)')
  const h = typeof hazard === 'number' ? () => hazard : hazard
  const cut = options.threshold ?? -40
  const algorithm: Algorithm<BocpdState> = {
    name: 'bocpd',
    config: { model: model.kind, steps: y.length },
    schema: channels({
      metrics: {
        changeProbability: { description: 'p(a segment starts at this step)' },
        mapRunLength: { description: 'the most probable run length' },
      },
    }),
    init: () => ({
      hyps: [{ logp: 0, run: priorRun(model) }],
      out: { logPredictive: [], predictiveMean: [], runLength: [], mapRunLength: [], changeProbability: [] },
      logEvidence: 0,
    }),
    step: (s, ctx) => {
      const t = ctx.t
      const yt = y[t] ?? 0
      let mean = 0
      for (const p of s.hyps) mean += Math.exp(p.logp) * p.run.predictiveMean()
      const joint = s.hyps.map((p) => p.logp + p.run.logPredictive(yt))
      const lp = logSumExpOf(joint)
      const posterior = s.hyps.map((p, i) => ({ logp: (joint[i] ?? 0) - lp, run: p.run.update(yt) }))
      const rl = new Float64Array(t + 2)
      let best = 0
      for (const p of posterior) {
        rl[p.run.length] = Math.exp(p.logp)
        if ((rl[p.run.length] ?? 0) > (rl[best] ?? 0)) best = p.run.length
      }
      s.out.logPredictive.push(lp)
      s.out.predictiveMean.push(mean)
      s.out.runLength.push(rl)
      s.out.mapRunLength.push(best)
      s.out.changeProbability.push(rl[1] ?? 0)
      const kept = posterior.filter((p) => p.logp > cut)
      const change = logSumExpOf(kept.map((p) => p.logp + Math.log(h(p.run.length))))
      const next = kept.flatMap((p) => {
        const stay = 1 - h(p.run.length)
        return stay > 0 ? [{ logp: p.logp + Math.log(stay), run: p.run }] : []
      })
      let hyps = change === Number.NEGATIVE_INFINITY ? next : [{ logp: change, run: priorRun(model) }, ...next]
      const z = logSumExpOf(hyps.map((p) => p.logp))
      hyps = hyps.map((p) => ({ logp: p.logp - z, run: p.run }))
      return { hyps, out: s.out, logEvidence: s.logEvidence + lp }
    },
    observe: (s) => ({
      metrics: { changeProbability: s.out.changeProbability.at(-1) ?? 0, mapRunLength: s.out.mapRunLength.at(-1) ?? 0 },
    }),
  }
  const st = runAlgorithm(algorithm, { maxSteps: y.length, ...options }).state
  return { ...st.out, logEvidence: st.logEvidence }
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
 * @throws WhiteboxError with code `value` when the series is shorter than one minimum segment.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * filtering.pelt([0, 0, 0, 0, 5, 5, 5, 5], 1) // => [4, 8]
 * ```
 */
export function pelt(y: readonly number[], penalty: number, options: OfflineOptions = {}): number[] {
  const n = y.length
  const c = costFunction(y, options.cost ?? 'l2')
  const minSize = Math.max(options.minSize ?? 2, c.minSize)
  const jump = options.jump ?? 1
  if (n < minSize) throw new WhiteboxError('value', 'pelt: the series is shorter than the minimum segment')
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
 * @throws WhiteboxError with code `value` without a stopping rule.
 * @concept changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * filtering.binarySegmentation([0, 0, 0, 5, 5, 5, 1, 1, 1], { changepoints: 2 }) // => [3, 6, 9]
 * ```
 */
export function binarySegmentation(
  y: readonly number[],
  stop: { readonly changepoints?: number; readonly penalty?: number },
  options: OfflineOptions = {},
): number[] {
  if (stop.changepoints === undefined && stop.penalty === undefined) {
    throw new WhiteboxError('value', 'binarySegmentation: give a number of changepoints or a penalty')
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

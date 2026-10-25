/**
 * Streaming (one-pass) statistics that update with each observation in constant memory: running mean, variance,
 * skewness and kurtosis by Welford's and Pebay's updates with merging of partial states, the P-squared estimator of a
 * quantile, and exponentially weighted moving mean, variance and standard deviation matching pandas' `ewm`.
 */

import { MdmlError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { sample, sortedCopy, vec } from './_util.ts'

/**
 * Running central moments up to the fourth (Welford's algorithm with Pebay's higher-order updates): the count, mean
 * and the sums of powers of deviations M2, M3, M4, plus the extremes. Immutable: {@link push} and {@link merge} return
 * new states.
 */
export interface MomentState {
  readonly n: number
  readonly mean: number
  readonly m2: number
  readonly m3: number
  readonly m4: number
  readonly min: number
  readonly max: number
}

/**
 * The empty moment state.
 *
 * @returns A state with no observations.
 * @example
 * ```ts
 * import { momentState } from 'mdml-engine/stats'
 * momentState().n // => 0
 * ```
 */
export function momentState(): MomentState {
  return {
    n: 0,
    mean: 0,
    m2: 0,
    m3: 0,
    m4: 0,
    min: Number.POSITIVE_INFINITY,
    max: Number.NEGATIVE_INFINITY,
  }
}

/**
 * Add one observation to a moment state (Welford; Pebay 2008 for M3 and M4), numerically stable in one pass.
 *
 * @param state - The state so far.
 * @param x - The new observation.
 * @returns The updated state.
 * @reference P. Pebay, Formulas for robust, one-pass parallel computation of covariances and arbitrary-order
 * statistical moments, Sandia report SAND2008-6212 (2008).
 * @example
 * ```ts
 * import { momentState, push } from 'mdml-engine/stats'
 * push(push(momentState(), 1), 3).mean // => 2
 * ```
 */
export function push(state: MomentState, x: number): MomentState {
  const n1 = state.n
  const n = n1 + 1
  const delta = x - state.mean
  const dn = delta / n
  const dn2 = dn * dn
  const term1 = delta * dn * n1
  return {
    n,
    mean: state.mean + dn,
    m4: state.m4 + term1 * dn2 * (n * n - 3 * n + 3) + 6 * dn2 * state.m2 - 4 * dn * state.m3,
    m3: state.m3 + term1 * dn * (n - 2) - 3 * dn * state.m2,
    m2: state.m2 + term1,
    min: Math.min(state.min, x),
    max: Math.max(state.max, x),
  }
}

/**
 * Add every value of a sample to a moment state.
 *
 * @param state - The state so far.
 * @param xs - The new observations.
 * @returns The updated state.
 * @example
 * ```ts
 * import { momentState, pushAll } from 'mdml-engine/stats'
 * pushAll(momentState(), [1, 2, 3, 4]).m2 // => 5
 * ```
 */
export function pushAll(state: MomentState, xs: TensorLike): MomentState {
  let s = state
  for (const v of sample('pushAll', xs, 0)) s = push(s, v)
  return s
}

/**
 * Combine the moment states of two disjoint samples (Chan et al.'s pairwise update, extended by Pebay), so moments
 * can be computed in parallel chunks.
 *
 * @param a - One state.
 * @param b - The other.
 * @returns The state of the union.
 * @example
 * ```ts
 * import { merge, momentState, pushAll } from 'mdml-engine/stats'
 * merge(pushAll(momentState(), [1, 2]), pushAll(momentState(), [3, 4])).m2 // => 5
 * ```
 */
export function merge(a: MomentState, b: MomentState): MomentState {
  if (a.n === 0) return b
  if (b.n === 0) return a
  const n = a.n + b.n
  const d = b.mean - a.mean
  const d2 = d * d
  const na = a.n
  const nb = b.n
  return {
    n,
    mean: a.mean + (d * nb) / n,
    m2: a.m2 + b.m2 + (d2 * na * nb) / n,
    m3: a.m3 + b.m3 + (d2 * d * na * nb * (na - nb)) / (n * n) + (3 * d * (na * b.m2 - nb * a.m2)) / n,
    m4:
      a.m4 +
      b.m4 +
      (d2 * d2 * na * nb * (na * na - na * nb + nb * nb)) / (n * n * n) +
      (6 * d2 * (na * na * b.m2 + nb * nb * a.m2)) / (n * n) +
      (4 * d * (na * b.m3 - nb * a.m3)) / n,
    min: Math.min(a.min, b.min),
    max: Math.max(a.max, b.max),
  }
}

/** The summary read from a moment state. */
export interface RunningMoments {
  readonly n: number
  readonly mean: number
  readonly variance: number
  readonly std: number
  /** The biased skewness g1, as scipy.stats.skew. */
  readonly skewness: number
  /** The biased excess kurtosis g2, as scipy.stats.kurtosis. */
  readonly kurtosis: number
  readonly min: number
  readonly max: number
}

/**
 * Read the mean, variance, skewness and kurtosis from a moment state.
 *
 * @param state - The state.
 * @param options - ddof of the variance (default 0).
 * @returns The moments (`NaN` where undefined).
 * @example
 * ```ts
 * import { momentState, momentsOf, pushAll } from 'mdml-engine/stats'
 * momentsOf(pushAll(momentState(), [1, 2, 3, 4]), { ddof: 1 }).variance // => 1.6666666666666667
 * ```
 */
export function momentsOf(state: MomentState, options: { readonly ddof?: number } = {}): RunningMoments {
  const { n, m2, m3, m4 } = state
  const ddof = options.ddof ?? 0
  const variance = n > ddof ? m2 / (n - ddof) : Number.NaN
  const pop = n > 0 ? m2 / n : Number.NaN
  return {
    n,
    mean: n > 0 ? state.mean : Number.NaN,
    variance,
    std: Math.sqrt(variance),
    skewness: pop > 0 ? m3 / n / pop ** 1.5 : Number.NaN,
    kurtosis: pop > 0 ? m4 / n / (pop * pop) - 3 : Number.NaN,
    min: state.min,
    max: state.max,
  }
}

/**
 * The state of the P-squared quantile estimator (Jain and Chlamtac 1985): five marker heights and positions that track
 * one quantile in constant memory.
 */
export interface P2State {
  readonly p: number
  readonly n: number
  /** Marker heights (the first observations, sorted, until five have arrived). */
  readonly heights: readonly number[]
  /** Actual marker positions (1-based). */
  readonly positions: readonly number[]
  /** Desired marker positions. */
  readonly desired: readonly number[]
}

/**
 * Start a P-squared estimator of the p-quantile.
 *
 * @param p - The level in (0, 1).
 * @returns The empty state.
 * @reference R. Jain and I. Chlamtac, The P2 algorithm for dynamic calculation of quantiles and histograms without
 * storing observations, Communications of the ACM 28 (1985).
 * @example
 * ```ts
 * import { p2Quantile } from 'mdml-engine/stats'
 * p2Quantile(0.5).n // => 0
 * ```
 */
export function p2Quantile(p: number): P2State {
  if (!(p > 0 && p < 1)) throw new MdmlError('value', 'p2Quantile: p must lie in (0, 1)')
  return { p, n: 0, heights: [], positions: [1, 2, 3, 4, 5], desired: [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5] }
}

/**
 * Add one observation to a P-squared estimator: adjust the extreme markers, shift positions, and move the three middle
 * markers towards their desired positions by piecewise-parabolic (or, failing monotonicity, linear) interpolation.
 *
 * @param state - The estimator.
 * @param x - The observation.
 * @returns The updated estimator.
 * @example
 * ```ts
 * import { p2Estimate, p2Push, p2Quantile } from 'mdml-engine/stats'
 * let s = p2Quantile(0.5)
 * for (const x of [5, 1, 4, 2, 3]) s = p2Push(s, x)
 * p2Estimate(s) // => 3
 * ```
 */
export function p2Push(state: P2State, x: number): P2State {
  const p = state.p
  if (state.n < 5) {
    const heights = [...state.heights, x].sort((a, b) => a - b)
    return { ...state, n: state.n + 1, heights }
  }
  const q = [...state.heights]
  const pos = [...state.positions]
  const want = [...state.desired]
  const increments = [0, p / 2, p, (1 + p) / 2, 1]
  let k: number
  if (x < (q[0] ?? 0)) {
    q[0] = x
    k = 0
  } else if (x >= (q[4] ?? 0)) {
    q[4] = x
    k = 3
  } else {
    k = 0
    while (k < 3 && x >= (q[k + 1] ?? 0)) k++
  }
  for (let i = k + 1; i < 5; i++) pos[i] = (pos[i] ?? 0) + 1
  for (let i = 0; i < 5; i++) want[i] = (want[i] ?? 0) + (increments[i] ?? 0)
  for (let i = 1; i <= 3; i++) {
    const d = (want[i] ?? 0) - (pos[i] ?? 0)
    const ni = pos[i] ?? 0
    const nPrev = pos[i - 1] ?? 0
    const nNext = pos[i + 1] ?? 0
    if ((d >= 1 && nNext - ni > 1) || (d <= -1 && nPrev - ni < -1)) {
      const s = Math.sign(d)
      const qi = q[i] ?? 0
      const qPrev = q[i - 1] ?? 0
      const qNext = q[i + 1] ?? 0
      const parabolic =
        qi +
        (s / (nNext - nPrev)) *
          ((ni - nPrev + s) * ((qNext - qi) / (nNext - ni)) + (nNext - ni - s) * ((qi - qPrev) / (ni - nPrev)))
      if (qPrev < parabolic && parabolic < qNext) q[i] = parabolic
      else {
        const j = i + s
        q[i] = qi + (s * ((q[j] ?? 0) - qi)) / ((pos[j] ?? 0) - ni)
      }
      pos[i] = ni + s
    }
  }
  return { p, n: state.n + 1, heights: q, positions: pos, desired: want }
}

/**
 * The current P-squared estimate: the middle marker, or the exact (linear) sample quantile while fewer than six
 * observations have arrived.
 *
 * @param state - The estimator.
 * @returns The estimated quantile (`NaN` before any observation).
 * @example
 * ```ts
 * import { p2Estimate, p2Push, p2Quantile } from 'mdml-engine/stats'
 * p2Estimate(p2Push(p2Push(p2Quantile(0.5), 1), 2)) // => 1.5
 * ```
 */
export function p2Estimate(state: P2State): number {
  if (state.n === 0) return Number.NaN
  if (state.n <= 5) {
    const s = sortedCopy(state.heights)
    const h = (s.length - 1) * state.p
    const lo = Math.floor(h)
    return (s[lo] ?? 0) + (h - lo) * ((s[Math.min(s.length - 1, lo + 1)] ?? 0) - (s[lo] ?? 0))
  }
  return state.heights[2] ?? Number.NaN
}

/** The smoothing of exponentially weighted moments: exactly one of alpha, span, com (centre of mass) or halflife. */
export interface EwmOptions {
  readonly alpha?: number
  readonly span?: number
  readonly com?: number
  readonly halflife?: number
  /** Divide by the decaying sum of weights (pandas' default true); false uses the recursive form. */
  readonly adjust?: boolean
}

/** The state of an exponentially weighted mean and variance, updated as pandas' `ewm` updates it. */
export interface EwmState {
  readonly alpha: number
  readonly adjust: boolean
  readonly n: number
  readonly mean: number
  /** The biased weighted variance. */
  readonly cov: number
  readonly sumWeights: number
  readonly sumWeights2: number
  readonly oldWeight: number
}

function alphaOf(o: EwmOptions): number {
  const given = [o.alpha, o.span, o.com, o.halflife].filter((v) => v !== undefined).length
  if (given !== 1) throw new MdmlError('value', 'ewm: give exactly one of alpha, span, com or halflife')
  if (o.alpha !== undefined) {
    if (!(o.alpha > 0 && o.alpha <= 1)) throw new MdmlError('value', 'ewm: alpha must lie in (0, 1]')
    return o.alpha
  }
  if (o.span !== undefined) {
    if (!(o.span >= 1)) throw new MdmlError('value', 'ewm: span must be at least 1')
    return 2 / (o.span + 1)
  }
  if (o.com !== undefined) {
    if (!(o.com >= 0)) throw new MdmlError('value', 'ewm: com must be non-negative')
    return 1 / (1 + o.com)
  }
  const h = o.halflife ?? 1
  if (!(h > 0)) throw new MdmlError('value', 'ewm: halflife must be positive')
  return 1 - Math.exp(-Math.LN2 / h)
}

/**
 * Start exponentially weighted moments.
 *
 * @param options - The smoothing (one of alpha, span, com, halflife) and adjust (default true).
 * @returns The empty state.
 * @example
 * ```ts
 * import { ewmState } from 'mdml-engine/stats'
 * ewmState({ span: 3 }).alpha // => 0.5
 * ```
 */
export function ewmState(options: EwmOptions): EwmState {
  return {
    alpha: alphaOf(options),
    adjust: options.adjust ?? true,
    n: 0,
    mean: Number.NaN,
    cov: 0,
    sumWeights: 1,
    sumWeights2: 1,
    oldWeight: 1,
  }
}

/**
 * Add one observation to exponentially weighted moments (pandas' ewm mean and ewmcov recursions).
 *
 * @param state - The state.
 * @param x - The observation.
 * @returns The updated state.
 * @example
 * ```ts
 * import { ewmPush, ewmState } from 'mdml-engine/stats'
 * ewmPush(ewmPush(ewmState({ alpha: 0.5 }), 0), 3).mean // => 2
 * ```
 */
export function ewmPush(state: EwmState, x: number): EwmState {
  if (state.n === 0) return { ...state, n: 1, mean: x }
  const factor = 1 - state.alpha
  const newWeight = state.adjust ? 1 : state.alpha
  let sumWeights = state.sumWeights * factor
  let sumWeights2 = state.sumWeights2 * factor * factor
  let oldWeight = state.oldWeight * factor
  const old = state.mean
  const mean = old !== x ? (oldWeight * old + newWeight * x) / (oldWeight + newWeight) : old
  const cov = (oldWeight * (state.cov + (old - mean) ** 2) + newWeight * (x - mean) ** 2) / (oldWeight + newWeight)
  sumWeights += newWeight
  sumWeights2 += newWeight * newWeight
  oldWeight += newWeight
  if (!state.adjust) {
    sumWeights /= oldWeight
    sumWeights2 /= oldWeight * oldWeight
    oldWeight = 1
  }
  return { ...state, n: state.n + 1, mean, cov, sumWeights, sumWeights2, oldWeight }
}

/** The variance of an EWM state: biased, or debiased by sum(w)^2 / (sum(w)^2 - sum(w^2)). */
function ewmVarianceOf(s: EwmState, bias: boolean): number {
  if (bias) return s.cov
  const num = s.sumWeights * s.sumWeights
  const den = num - s.sumWeights2
  return den > 0 ? (num / den) * s.cov : Number.NaN
}

/**
 * The exponentially weighted moving average of a series (pandas `Series.ewm(...).mean()`).
 *
 * @param x - The series.
 * @param options - The smoothing and adjust.
 * @returns The running average at each point.
 * @example
 * ```ts
 * import { ewmMean } from 'mdml-engine/stats'
 * ewmMean([0, 3, 3], { alpha: 0.5, adjust: false }) // => [0, 1.5, 2.25]
 * ```
 */
export function ewmMean(x: TensorLike, options: EwmOptions): Tensor {
  let s = ewmState(options)
  const out: number[] = []
  for (const v of sample('ewmMean', x, 0)) {
    s = ewmPush(s, v)
    out.push(s.mean)
  }
  return vec(out)
}

/**
 * The exponentially weighted moving variance (pandas `Series.ewm(...).var(bias)`); the first value is NaN unless
 * biased.
 *
 * @param x - The series.
 * @param options - The smoothing, adjust and bias (default false, the debiased variance).
 * @returns The running variance at each point.
 * @example
 * ```ts
 * import { ewmVar } from 'mdml-engine/stats'
 * ewmVar([0, 2], { alpha: 0.5 }) // => [NaN, 2]
 * ```
 */
export function ewmVar(x: TensorLike, options: EwmOptions & { readonly bias?: boolean }): Tensor {
  let s = ewmState(options)
  const out: number[] = []
  for (const v of sample('ewmVar', x, 0)) {
    s = ewmPush(s, v)
    out.push(ewmVarianceOf(s, options.bias ?? false))
  }
  return vec(out)
}

/**
 * The exponentially weighted moving standard deviation (pandas `Series.ewm(...).std(bias)`).
 *
 * @param x - The series.
 * @param options - The smoothing, adjust and bias (default false).
 * @returns The running standard deviation at each point.
 * @example
 * ```ts
 * import { ewmStd } from 'mdml-engine/stats'
 * ewmStd([0, 2], { alpha: 0.5, bias: true }) // => [0, 0.9428090415820634]
 * ```
 */
export function ewmStd(x: TensorLike, options: EwmOptions & { readonly bias?: boolean }): Tensor {
  let s = ewmState(options)
  const out: number[] = []
  for (const v of sample('ewmStd', x, 0)) {
    s = ewmPush(s, v)
    out.push(Math.sqrt(ewmVarianceOf(s, options.bias ?? false)))
  }
  return vec(out)
}

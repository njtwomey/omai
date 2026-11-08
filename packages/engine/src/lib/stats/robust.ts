/**
 * Robust statistics, estimates that resist outliers: trimmed and winsorized means and variances (scipy.stats),
 * M-estimates of location by iteratively reweighted least squares with Huber's or Tukey's biweight weights (as
 * statsmodels RLM), and RANSAC, which fits any model to random minimal subsets and keeps the one with the most inliers,
 * also as a traced algorithm.
 */

import { type Estimate, meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { subsample } from '../random/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { normPpf, sample, sortedCopy, vec } from './_util.ts'
import { median } from './quantiles.ts'

function cutCount(name: string, n: number, proportion: number): number {
  if (!(proportion >= 0 && proportion < 0.5))
    throw new OmaiError('value', `${name}: the proportion must lie in [0, 0.5)`)
  return Math.floor(n * proportion)
}

/**
 * Trim a proportion from each end of a sample (scipy.stats.trimboth): the floor(n p) smallest and largest values are
 * removed.
 *
 * @param x - The sample.
 * @param proportion - The fraction cut from each end, in [0, 0.5).
 * @returns The remaining values, sorted.
 * @example
 * ```ts
 * import { trimboth } from 'omai-engine/stats'
 * trimboth([5, 1, 9, 3, 7], 0.2) // => [3, 5, 7]
 * ```
 */
export function trimboth(x: TensorLike, proportion: number): Tensor {
  const s = sortedCopy(sample('trimboth', x))
  const k = cutCount('trimboth', s.length, proportion)
  return vec(s.subarray(k, s.length - k))
}

/**
 * The trimmed mean (scipy.stats.trim_mean): the mean after trimming a proportion from each end.
 *
 * @param x - The sample.
 * @param proportion - The fraction cut from each end.
 * @returns The trimmed mean.
 * @example
 * ```ts
 * import { trimMean } from 'omai-engine/stats'
 * trimMean([1, 2, 3, 4, 100], 0.2) // => 3
 * ```
 */
export function trimMean(x: TensorLike, proportion: number): number {
  return meanOf(trimboth(x, proportion).data as Float64Array)
}

/**
 * The variance of the trimmed sample (the trimmed values discarded).
 *
 * @param x - The sample.
 * @param proportion - The fraction cut from each end.
 * @param options - ddof (default 1).
 * @returns The trimmed variance.
 * @example
 * ```ts
 * import { trimmedVariance } from 'omai-engine/stats'
 * trimmedVariance([1, 2, 3, 4, 100], 0.2) // => 1
 * ```
 */
export function trimmedVariance(x: TensorLike, proportion: number, options: { readonly ddof?: number } = {}): number {
  return varianceOf(trimboth(x, proportion).data as Float64Array, options.ddof ?? 1)
}

/**
 * Winsorize a sample (scipy.stats.mstats.winsorize): the int(n low) smallest values are replaced by the next smallest
 * and the int(n high) largest by the next largest, in the original order.
 *
 * @param x - The sample.
 * @param limits - The fractions [low, high] to winsorize at each end.
 * @returns The winsorized sample.
 * @example
 * ```ts
 * import { winsorize } from 'omai-engine/stats'
 * winsorize([10, 1, 5, 3, 100], [0.2, 0.2]) // => [10, 3, 5, 3, 10]
 * ```
 */
export function winsorize(x: TensorLike, limits: readonly [number, number]): Tensor {
  const a = sample('winsorize', x)
  const n = a.length
  const [lo, hi] = limits
  if (!(lo >= 0 && hi >= 0 && lo + hi < 1))
    throw new OmaiError('value', 'winsorize: limits must be non-negative and sum below 1')
  const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => (a[p] ?? 0) - (a[q] ?? 0) || p - q)
  const lowIdx = Math.trunc(lo * n)
  const upIdx = n - Math.trunc(hi * n)
  const out = Float64Array.from(a)
  const lowValue = a[order[lowIdx] ?? 0] ?? 0
  const upValue = a[order[upIdx - 1] ?? 0] ?? 0
  for (let r = 0; r < lowIdx; r++) out[order[r] ?? 0] = lowValue
  for (let r = upIdx; r < n; r++) out[order[r] ?? 0] = upValue
  return vec(out)
}

/**
 * The winsorized mean.
 *
 * @param x - The sample.
 * @param limits - The fractions [low, high].
 * @returns The mean of the winsorized sample.
 * @example
 * ```ts
 * import { winsorizedMean } from 'omai-engine/stats'
 * winsorizedMean([10, 1, 5, 3, 100], [0.2, 0.2]) // => 6.2
 * ```
 */
export function winsorizedMean(x: TensorLike, limits: readonly [number, number]): number {
  return meanOf(winsorize(x, limits).data as Float64Array)
}

/**
 * The winsorized variance (the variance of the winsorized sample).
 *
 * @param x - The sample.
 * @param limits - The fractions [low, high].
 * @param options - ddof (default 1).
 * @returns The winsorized variance.
 * @example
 * ```ts
 * import { winsorizedVariance } from 'omai-engine/stats'
 * winsorizedVariance([10, 1, 5, 3, 100], [0.2, 0.2]) // => 12.7
 * ```
 */
export function winsorizedVariance(
  x: TensorLike,
  limits: readonly [number, number],
  options: { readonly ddof?: number } = {},
): number {
  return varianceOf(winsorize(x, limits).data as Float64Array, options.ddof ?? 1)
}

/** A robust M-estimation norm. */
export type RobustNorm = 'huber' | 'tukey'

/** An M-estimate of location with the scale it was standardised by. */
export type MEstimate = Estimate & { readonly scale: number }

/** Options of {@link mEstimateLocation}. */
export interface MEstimateOptions {
  /** Huber's (default) or Tukey's biweight norm. */
  readonly norm?: RobustNorm
  /** Tuning constant (default 1.345 for Huber, 4.685 for Tukey: 95% efficiency at the normal). */
  readonly c?: number
  /** A fixed scale, or (default) the normalised MAD of the residuals, updated at every iteration as statsmodels' RLM. */
  readonly scale?: number
  readonly maxIterations?: number
  readonly tol?: number
}

function weight(norm: RobustNorm, c: number, u: number): number {
  const a = Math.abs(u)
  if (norm === 'huber') return a <= c ? 1 : c / a
  return a <= c ? (1 - (u / c) ** 2) ** 2 : 0
}

/**
 * A robust M-estimate of location by iteratively reweighted least squares (statsmodels RLM on a constant): Huber's or
 * Tukey's biweight weights of the standardised residuals, with the scale fixed or re-estimated as the normalised MAD
 * of the residuals at each step; returns the estimate with its asymptotic (sandwich) standard error.
 *
 * @param x - The sample.
 * @param options - norm, c, scale, maxIterations and tol.
 * @returns The location estimate with its standard error and the final scale.
 * @example
 * ```ts
 * import { mEstimateLocation } from 'omai-engine/stats'
 * mEstimateLocation([1, 2, 3, 4, 100]).value < 4 // => true
 * ```
 */
export function mEstimateLocation(x: TensorLike, options: MEstimateOptions = {}): MEstimate {
  const a = sample('mEstimateLocation', x, 2)
  const norm = options.norm ?? 'huber'
  const c = options.c ?? (norm === 'huber' ? 1.345 : 4.685)
  const k = normPpf(0.75)
  const madScale = (mu: number): number => median(Float64Array.from(a, (v) => Math.abs(v - mu))) / k
  let mu = meanOf(a)
  let s = options.scale ?? madScale(mu)
  const tol = options.tol ?? 1e-12
  for (let it = 0; it < (options.maxIterations ?? 500); it++) {
    if (!(s > 0)) break
    let sw = 0
    let swx = 0
    for (const v of a) {
      const w = weight(norm, c, (v - mu) / s)
      sw += w
      swx += w * v
    }
    const next = sw > 0 ? swx / sw : mu
    const nextScale = options.scale ?? madScale(next)
    const done = Math.abs(next - mu) <= tol * (1 + Math.abs(mu)) && Math.abs(nextScale - s) <= tol * (1 + s)
    mu = next
    s = nextScale
    if (done) break
  }
  // Huber's sandwich variance s^2 mean(psi^2) / mean(psi')^2 / n.
  let psi2 = 0
  let dpsi = 0
  for (const v of a) {
    const u = (v - mu) / s
    const au = Math.abs(u)
    if (norm === 'huber') {
      psi2 += (au <= c ? u : c * Math.sign(u)) ** 2
      dpsi += au <= c ? 1 : 0
    } else if (au <= c) {
      const r = (u / c) ** 2
      psi2 += (u * (1 - r) ** 2) ** 2
      dpsi += (1 - r) * (1 - 5 * r)
    }
  }
  const n = a.length
  const se = Math.sqrt((s * s * (psi2 / n)) / (dpsi / n) ** 2 / n)
  return { value: mu, standardError: se, scale: s, method: `${norm} M-estimate` }
}

/**
 * Huber's M-estimate of location (c = 1.345 by default).
 *
 * @param x - The sample.
 * @param options - c and scale.
 * @returns The estimate.
 * @example
 * ```ts
 * import { huber } from 'omai-engine/stats'
 * huber([1, 2, 3, 4, 5]).value // => 3
 * ```
 */
export function huber(x: TensorLike, options: Omit<MEstimateOptions, 'norm'> = {}): MEstimate {
  return mEstimateLocation(x, { ...options, norm: 'huber' })
}

/**
 * Tukey's biweight M-estimate of location (c = 4.685 by default), which gives gross outliers zero weight.
 *
 * @param x - The sample.
 * @param options - c and scale.
 * @returns The estimate.
 * @example
 * ```ts
 * import { tukeyBiweight } from 'omai-engine/stats'
 * tukeyBiweight([1, 2, 3, 4, 5]).value // => 3
 * ```
 */
export function tukeyBiweight(x: TensorLike, options: Omit<MEstimateOptions, 'norm'> = {}): MEstimate {
  return mEstimateLocation(x, { ...options, norm: 'tukey' })
}

/** A model fitted by RANSAC: fit on a subset, residual of one point. */
export interface RansacModel<P, M> {
  /** Fit a model to points (a minimal sample, then the inliers). */
  readonly fit: (points: readonly P[]) => M
  /** The residual of a point under a model (its absolute value is compared with the threshold). */
  readonly residual: (model: M, point: P) => number
}

/** Options of {@link ransac}. */
export interface RansacOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Points per minimal sample. */
  readonly minSamples: number
  /** Points with |residual| at most this are inliers. */
  readonly residualThreshold: number
  /** The most trials (default 100). */
  readonly maxTrials?: number
  /** Stop once a trial count achieving this probability of an outlier-free sample is reached (default 0.99). */
  readonly stopProbability?: number
}

/** The state of {@link ransacAlgorithm}. */
export interface RansacState<M> {
  readonly trials: number
  readonly bestInliers: readonly number[]
  readonly bestModel: M | undefined
  readonly bestScore: number
  /** The trials needed for the stop probability given the best inlier fraction so far. */
  readonly required: number
}

/**
 * RANSAC (Fischler and Bolles 1981) as a traced algorithm, generic in the points and the model: each step fits a
 * model to a keyed random minimal sample and keeps the model with the most inliers (ties to the smaller inlier
 * residual sum), stopping early once enough trials make an outlier-free sample likely (sklearn's dynamic trial count).
 *
 * @param points - The data points.
 * @param model - fit and residual.
 * @param options - minSamples, residualThreshold, maxTrials and stopProbability.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { ransacAlgorithm } from 'omai-engine/stats'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const mean = { fit: (p: readonly number[]) => p.reduce((s, v) => s + v, 0) / p.length, residual: (m: number, p: number) => p - m }
 * runAlgorithm(ransacAlgorithm([1, 1.1, 0.9, 50], mean, { minSamples: 1, residualThreshold: 0.5 }), { maxSteps: 20 }).state.bestInliers.length // => 3
 * ```
 */
export function ransacAlgorithm<P, M>(
  points: readonly P[],
  model: RansacModel<P, M>,
  options: RansacOptions,
): Algorithm<RansacState<M>> {
  const n = points.length
  const m = options.minSamples
  if (!(Number.isInteger(m) && m >= 1 && m <= n)) throw new OmaiError('value', 'ransac: minSamples must be in [1, n]')
  const maxTrials = options.maxTrials ?? 100
  const stopP = options.stopProbability ?? 0.99
  const thr = options.residualThreshold
  return {
    name: 'ransac',
    config: { minSamples: m, residualThreshold: thr, maxTrials, stopProbability: stopP },
    schema: channels({
      metrics: {
        inliers: { description: 'inliers of the best model so far', direction: 'higher' },
        required: { description: 'trials needed for the stop probability' },
      },
    }),
    init: () => ({
      trials: 0,
      bestInliers: [],
      bestModel: undefined,
      bestScore: Number.POSITIVE_INFINITY,
      required: maxTrials,
    }),
    step: (s, ctx) => {
      const idx = Array.from(subsample(ctx.key, n, m).data)
      const fitted = model.fit(idx.map((i) => points[i] as P))
      const inliers: number[] = []
      let score = 0
      points.forEach((p, i) => {
        const r = Math.abs(model.residual(fitted, p))
        if (r <= thr) {
          inliers.push(i)
          score += r
        }
      })
      const better =
        inliers.length > s.bestInliers.length || (inliers.length === s.bestInliers.length && score < s.bestScore)
      if (!better) return { ...s, trials: s.trials + 1 }
      const w = inliers.length / n
      const required =
        w >= 1 ? 1 : w <= 0 ? maxTrials : Math.min(maxTrials, Math.ceil(Math.log(1 - stopP) / Math.log(1 - w ** m)))
      return { trials: s.trials + 1, bestInliers: inliers, bestModel: fitted, bestScore: score, required }
    },
    done: (s) => (s.trials >= Math.min(maxTrials, s.required) ? 'converged' : undefined),
    observe: (s) => ({ metrics: { inliers: s.bestInliers.length, required: s.required } }),
  }
}

/** The result of {@link ransac}. */
export interface RansacResult<M> {
  /** The model refitted on the inliers of the best trial. */
  readonly model: M
  readonly inliers: readonly number[]
  readonly trials: number
}

/**
 * Robust fitting by RANSAC with any model: the best keyed trial's inliers, refitted.
 *
 * @param points - The data points.
 * @param model - fit and residual.
 * @param options - minSamples, residualThreshold, maxTrials, stopProbability, key or seed, tracer.
 * @returns The refitted model, the inlier indices and the trials used.
 * @example
 * ```ts
 * import { ransac } from 'omai-engine/stats'
 * const mean = { fit: (p: readonly number[]) => p.reduce((s, v) => s + v, 0) / p.length, residual: (m: number, p: number) => p - m }
 * ransac([1, 1.1, 0.9, 50], mean, { minSamples: 1, residualThreshold: 0.5, seed: 0 }).model // => 1
 * ```
 */
export function ransac<P, M>(points: readonly P[], model: RansacModel<P, M>, options: RansacOptions): RansacResult<M> {
  const { minSamples: _m, residualThreshold: _r, maxTrials, stopProbability: _p, ...run } = options
  const r = runAlgorithm(ransacAlgorithm(points, model, options), { ...run, maxSteps: maxTrials ?? 100 })
  const inliers = r.state.bestInliers
  if (inliers.length === 0) throw new OmaiError('convergence', 'ransac: no trial found any inlier')
  return { model: model.fit(inliers.map((i) => points[i] as P)), inliers, trials: r.state.trials }
}

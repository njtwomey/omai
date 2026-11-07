/**
 * Extreme-value statistics, which model the tails of a distribution (Coles 2001): maximum-likelihood fits of the
 * generalised extreme value distribution to block maxima and of the generalised Pareto to threshold exceedances,
 * declustered peaks over a threshold, return levels with delta-method errors (the level exceeded once per period on
 * average), and the mean excess function that guides the choice of threshold.
 */

import { type Estimate, MdmlError, meanOf, varianceOf } from '../base/index.ts'
import { type Distribution, generalisedExtremeValue, generalisedPareto } from '../dists/index.ts'
import { asTensor, exp, item, sum, type Tensor, type TensorLike, take, toFlat, type Value } from '../tensor/index.ts'
import { invSpd } from './_linear.ts'
import { mat, normPpf, sample, sortedCopy, vec } from './_util.ts'
import { type NewtonMleOptions, newtonMle, observedInformation } from './fit.ts'

/** A fitted extreme-value distribution. */
export interface ExtremeFit {
  /** loc, scale and shape (xi, the extreme-value convention; scipy's c is -xi for the GEV). */
  readonly params: { readonly loc: number; readonly scale: number; readonly shape: number }
  /** Standard errors from the observed information (NaN for a fixed parameter). */
  readonly standardErrors: { readonly loc: number; readonly scale: number; readonly shape: number }
  /** Covariance of the fitted parameters (loc, scale, shape; zero rows for a fixed location). */
  readonly covariance: Tensor
  readonly logLikelihood: number
  readonly converged: boolean
  readonly distribution: Distribution
}

function finish(
  make: (loc: Value, scale: Value, shape: Value) => Distribution,
  data: Value,
  loc: number,
  scale: number,
  shape: number,
  fixedLoc: boolean,
  converged: boolean,
): ExtremeFit {
  const free = fixedLoc ? [scale, shape] : [loc, scale, shape]
  const k = free.length
  const ll = (t: Value): Value => {
    const at = (i: number): Value => take(t, [i])
    return fixedLoc
      ? sum(make(asTensor([loc]), at(0), at(1)).logProb(data))
      : sum(make(at(0), at(1), at(2)).logProb(data))
  }
  let cov: Float64Array
  try {
    cov = invSpd(toFlat(observedInformation(ll, free)), k)
  } catch {
    cov = new Float64Array(k * k).fill(Number.NaN)
  }
  const full = new Float64Array(9)
  const offset = fixedLoc ? 1 : 0
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) full[(i + offset) * 3 + (j + offset)] = cov[i * k + j] ?? 0
  const se = (i: number): number => (fixedLoc && i === 0 ? Number.NaN : Math.sqrt(full[i * 3 + i] ?? Number.NaN))
  const logLikelihood = item(ll(asTensor(free)) as Tensor)
  return {
    params: { loc, scale, shape },
    standardErrors: { loc: se(0), scale: se(1), shape: se(2) },
    covariance: mat(full, 3, 3),
    logLikelihood,
    converged,
    distribution: make(asTensor(loc), asTensor(scale), asTensor(shape)),
  }
}

/**
 * Fit the generalised extreme value distribution to block maxima by maximum likelihood (scipy.stats.genextreme.fit,
 * with xi = -c): Newton's method on (loc, log scale, xi) from Gumbel moment estimates, with standard errors from the
 * observed information.
 *
 * @param x - Block maxima.
 * @param options - Newton options.
 * @returns The fit.
 * @example
 * ```ts
 * import { fitGev } from 'mdml-engine/stats'
 * fitGev([1.2, 2.3, 1.9, 3.1, 2.2, 1.7, 2.8, 2.0, 2.5, 4.1]).converged // => true
 * ```
 */
export function fitGev(x: TensorLike, options: NewtonMleOptions = {}): ExtremeFit {
  const a = sample('fitGev', x, 3)
  const data = asTensor(a)
  const s0 = Math.sqrt(6 * varianceOf(a)) / Math.PI || 1
  const m0 = meanOf(a) - 0.5772 * s0
  const ll = (eta: Value): Value =>
    sum(generalisedExtremeValue(take(eta, [0]), exp(take(eta, [1])), take(eta, [2])).logProb(data))
  const r = newtonMle(ll, [m0, Math.log(s0), 0.05], options)
  const eta = toFlat(r.eta)
  return finish(
    generalisedExtremeValue,
    data,
    eta[0] ?? 0,
    Math.exp(eta[1] ?? 0),
    eta[2] ?? 0,
    false,
    r.converged,
  )
}

/**
 * Fit the generalised Pareto distribution to exceedances by maximum likelihood (scipy.stats.genpareto.fit with the
 * location fixed at the threshold): Newton's method on (log scale, xi) from moment estimates.
 *
 * @param x - Observations above the threshold (or exceedances, with threshold 0).
 * @param options - threshold (default 0) and Newton options.
 * @returns The fit (loc is the threshold, with no standard error).
 * @example
 * ```ts
 * import { fitGpd } from 'mdml-engine/stats'
 * fitGpd([0.2, 1.1, 0.5, 2.4, 0.1, 0.8, 1.6, 0.3, 0.9, 3.2]).params.loc // => 0
 * ```
 */
export function fitGpd(x: TensorLike, options: NewtonMleOptions & { readonly threshold?: number } = {}): ExtremeFit {
  const u = options.threshold ?? 0
  const a = sample('fitGpd', x, 3)
  for (const v of a) if (v < u) throw new MdmlError('domain', 'fitGpd: observations must exceed the threshold')
  const data = asTensor(a)
  const e = Float64Array.from(a, (v) => v - u)
  const m = meanOf(e)
  const v = varianceOf(e, 1)
  const xi0 = Math.max(-0.4, Math.min(0.4, 0.5 * (1 - (m * m) / v)))
  const s0 = Math.max(1e-6, 0.5 * m * (1 + (m * m) / v))
  const { threshold: _t, ...run } = options
  const ll = (eta: Value): Value =>
    sum(generalisedPareto(asTensor([u]), exp(take(eta, [0])), take(eta, [1])).logProb(data))
  const r = newtonMle(ll, [Math.log(s0), xi0], run)
  const eta = toFlat(r.eta)
  return finish(generalisedPareto, data, u, Math.exp(eta[0] ?? 0), eta[1] ?? 0, true, r.converged)
}

/** The result of {@link peaksOverThreshold}. */
export interface PeaksOverThreshold {
  readonly threshold: number
  /** Exceedances x - u of the (declustered) peaks. */
  readonly exceedances: Tensor
  /** Indices of the peaks in the series. */
  readonly indices: readonly number[]
  /** Peaks per observation, the exceedance rate zeta of return levels. */
  readonly rate: number
}

/**
 * Peaks over a threshold, declustered by runs (Coles 2001, section 5.3): a cluster ends once `runLength` consecutive
 * observations fall below the threshold, and each cluster is represented by its maximum.
 *
 * @param x - The series.
 * @param threshold - The threshold u.
 * @param options - runLength (default 0: every exceedance is its own peak).
 * @returns The exceedances, their indices and the rate per observation.
 * @example
 * ```ts
 * import { peaksOverThreshold } from 'mdml-engine/stats'
 * peaksOverThreshold([1, 5, 6, 1, 1, 7, 1], 4, { runLength: 1 }).exceedances // => [2, 3]
 * ```
 */
export function peaksOverThreshold(
  x: TensorLike,
  threshold: number,
  options: { readonly runLength?: number } = {},
): PeaksOverThreshold {
  const a = sample('peaksOverThreshold', x, 1)
  const run = options.runLength ?? 0
  const indices: number[] = []
  let clusterMax = -1
  let gap = Number.POSITIVE_INFINITY
  for (let i = 0; i < a.length; i++) {
    const v = a[i] ?? 0
    if (v > threshold) {
      if (clusterMax >= 0 && gap < run && run > 0) {
        if (v > (a[clusterMax] ?? 0)) clusterMax = i
      } else {
        if (clusterMax >= 0) indices.push(clusterMax)
        clusterMax = i
      }
      gap = 0
    } else gap++
  }
  if (clusterMax >= 0) indices.push(clusterMax)
  return {
    threshold,
    exceedances: vec(indices.map((i) => (a[i] ?? 0) - threshold)),
    indices,
    rate: indices.length / a.length,
  }
}

/**
 * The return level of a fitted extreme-value model with its delta-method standard error (Coles 2001): for a GEV fit to
 * block maxima, the level exceeded on average once every `period` blocks; for a GPD fit, the level exceeded once every
 * `period` observations given the exceedance rate.
 *
 * @param fit - A GEV or GPD fit.
 * @param period - The return period (blocks for a GEV, observations for a GPD).
 * @param options - kind (`'gev'`, default, or `'gpd'`), rate (GPD exceedances per observation) and level.
 * @returns The return level with its normal interval.
 * @example
 * ```ts
 * import { returnLevel, type ExtremeFit } from 'mdml-engine/stats'
 * const gumbel = { params: { loc: 0, scale: 1, shape: 0 } } as unknown as ExtremeFit
 * returnLevel(gumbel, 100).value // => 4.600149
 * ```
 */
export function returnLevel(
  fit: ExtremeFit,
  period: number,
  options: { readonly kind?: 'gev' | 'gpd'; readonly rate?: number; readonly level?: number } = {},
): Estimate {
  if (!(period > 1)) throw new MdmlError('value', 'returnLevel: the period must exceed 1')
  const { loc, scale, shape } = fit.params
  const kind = options.kind ?? 'gev'
  let value: number
  let g: [number, number, number]
  if (kind === 'gev') {
    const y = -Math.log(1 - 1 / period)
    if (Math.abs(shape) < 1e-12) {
      value = loc - scale * Math.log(y)
      g = [1, -Math.log(y), (scale * Math.log(y) ** 2) / 2]
    } else {
      const yp = y ** -shape
      value = loc - (scale / shape) * (1 - yp)
      g = [1, -(1 - yp) / shape, (scale * (1 - yp)) / (shape * shape) - (scale / shape) * yp * Math.log(y)]
    }
  } else {
    const rate = options.rate ?? 1
    const m = period * rate
    if (Math.abs(shape) < 1e-12) {
      value = loc + scale * Math.log(m)
      g = [0, Math.log(m), (scale * Math.log(m) ** 2) / 2]
    } else {
      const mx = m ** shape
      value = loc + (scale / shape) * (mx - 1)
      g = [0, (mx - 1) / shape, -(scale / (shape * shape)) * (mx - 1) + (scale / shape) * mx * Math.log(m)]
    }
  }
  if (fit.covariance === undefined) return { value, method: `${kind} return level` }
  const V = toFlat(fit.covariance)
  let variance = 0
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) variance += (g[i] ?? 0) * (V[i * 3 + j] ?? 0) * (g[j] ?? 0)
  const se = Math.sqrt(variance)
  const level = options.level ?? 0.95
  const z = normPpf(0.5 + level / 2)
  return {
    value,
    standardError: se,
    interval: [value - z * se, value + z * se],
    level,
    method: `${kind} return level, delta method`,
  }
}

/** The data of a mean residual life (mean excess) plot. */
export interface MeanExcess {
  readonly thresholds: Tensor
  /** The mean of x - u over x > u. */
  readonly meanExcess: Tensor
  readonly lower: Tensor
  readonly upper: Tensor
  /** Observations above each threshold. */
  readonly counts: Tensor
}

/**
 * The mean excess (mean residual life) function e(u) = E[X - u | X > u] at thresholds, with normal intervals: linear
 * in u above a threshold where a GPD fits (slope xi / (1 - xi)), which guides threshold choice.
 *
 * @param x - The sample.
 * @param options - thresholds (default 50 points between the minimum and the fifth-largest value) and level.
 * @returns The thresholds, mean excesses, interval bounds and counts.
 * @example
 * ```ts
 * import { meanExcess } from 'mdml-engine/stats'
 * meanExcess([1, 2, 3, 4, 5], { thresholds: [0, 2] }).meanExcess // => [3, 2]
 * ```
 */
export function meanExcess(
  x: TensorLike,
  options: { readonly thresholds?: TensorLike; readonly level?: number } = {},
): MeanExcess {
  const s = sortedCopy(sample('meanExcess', x, 2))
  const n = s.length
  let us: number[]
  if (options.thresholds === undefined) {
    const lo = s[0] ?? 0
    const hi = s[Math.max(0, n - 5)] ?? lo
    us = Array.from({ length: 50 }, (_, i) => lo + ((hi - lo) * i) / 49)
  } else us = toFlat(asTensor(options.thresholds))
  const z = normPpf(0.5 + (options.level ?? 0.95) / 2)
  const me: number[] = []
  const lo: number[] = []
  const hi: number[] = []
  const counts: number[] = []
  for (const u of us) {
    const e = Array.from(
      s.filter((v) => v > u),
      (v) => v - u,
    )
    const m = e.length > 0 ? meanOf(e) : Number.NaN
    const half = e.length > 1 ? z * Math.sqrt(varianceOf(e, 1) / e.length) : Number.NaN
    me.push(m)
    lo.push(m - half)
    hi.push(m + half)
    counts.push(e.length)
  }
  return { thresholds: vec(us), meanExcess: vec(me), lower: vec(lo), upper: vec(hi), counts: vec(counts) }
}

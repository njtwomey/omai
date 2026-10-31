/**
 * Regression discontinuity: a treatment that switches on at a cutoff of a running variable, estimated by local
 * polynomial fits on each side of the cutoff (sharp), or by their ratio against the jump in treatment (fuzzy, the
 * instrumental-variable form with the cutoff as the instrument), with triangular, uniform or Epanechnikov kernels, the
 * Imbens-Kalyanaraman bandwidth, a bandwidth sweep for the bias-variance picture and a McCrary-style test of
 * manipulation of the running variable.
 */

import { MdmlError } from '../base/index.ts'
import { matrixView, targetVector } from '../learn/validation.ts'
import type { TensorLike } from '../tensor/index.ts'
import { covariance, normCdf, variance, wls } from './_dense.ts'
import { type EffectEstimate, effectFromSe } from './data.ts'
import { instrumentalVariables } from './instrumental.ts'

/** Kernel weights over the scaled distance `u = (x - cutoff) / h`. */
export type Kernel = 'triangular' | 'uniform' | 'epanechnikov'

const kernelWeight = (kernel: Kernel, u: number): number => {
  if (Math.abs(u) >= 1) return 0
  if (kernel === 'uniform') return 0.5
  if (kernel === 'epanechnikov') return 0.75 * (1 - u * u)
  return 1 - Math.abs(u)
}

/** The data of a regression discontinuity design. */
export interface RdData {
  /** The running variable `[n]`. */
  readonly running: TensorLike
  /** The outcome `[n]`. */
  readonly outcome: TensorLike
  /** The treatment received `[n]`, for a fuzzy design. */
  readonly treatment?: TensorLike
}

/** Options of {@link regressionDiscontinuity}. */
export interface RdOptions {
  /** The cutoff (default 0). */
  readonly cutoff?: number
  /** The bandwidth, or `'ik'` for the Imbens-Kalyanaraman choice (default). */
  readonly bandwidth?: number | 'ik'
  readonly kernel?: Kernel
  /** The order of the local polynomial (default 1). */
  readonly order?: number
  /** `'sharp'` (default) or `'fuzzy'` (needs `treatment`). */
  readonly design?: 'sharp' | 'fuzzy'
  readonly level?: number
}

/** A local polynomial fit on one side. */
export interface SideFit {
  /** The fitted value at the cutoff. */
  readonly intercept: number
  /** The variance of that value (HC1). */
  readonly variance: number
  /** The slope at the cutoff. */
  readonly slope: number
  /** Observations with positive weight. */
  readonly n: number
}

/** The result of {@link regressionDiscontinuity}. */
export interface RdResult {
  readonly effect: EffectEstimate
  readonly bandwidth: number
  readonly cutoff: number
  /** The local fits of the outcome below and above the cutoff. */
  readonly left: SideFit
  readonly right: SideFit
  /** For a fuzzy design: the jump in the probability of treatment at the cutoff, and the outcome's jump. */
  readonly firstStageJump?: number
  readonly outcomeJump?: number
}

interface Sample {
  readonly x: Float64Array
  readonly y: Float64Array
  readonly d?: Float64Array
}

function read(data: RdData, who: string): Sample {
  const x = matrixView(
    Array.from(targetVector(data.running, who), (v) => [v]),
    who,
  ).data
  const y = targetVector(data.outcome, who, x.length)
  return {
    x,
    y,
    ...(data.treatment === undefined ? {} : { d: targetVector(data.treatment, who, x.length) }),
  }
}

/** The local polynomial fit of `y` on `x - cutoff` on one side, with kernel weights. */
function sideFit(
  s: Sample,
  y: Float64Array,
  cutoff: number,
  h: number,
  kernel: Kernel,
  order: number,
  above: boolean,
): SideFit {
  const rows: number[] = []
  const w: number[] = []
  for (let i = 0; i < s.x.length; i++) {
    const v = (s.x[i] ?? 0) - cutoff
    if (above ? v < 0 : v >= 0) continue
    const k = kernelWeight(kernel, v / h)
    if (k > 0) {
      rows.push(i)
      w.push(k)
    }
  }
  const p = order + 1
  if (rows.length <= p)
    throw new MdmlError(
      'value',
      `regressionDiscontinuity: the bandwidth ${h} leaves ${rows.length} observations ${above ? 'above' : 'below'} the cutoff`,
    )
  const a = new Float64Array(rows.length * p)
  const yy = new Float64Array(rows.length)
  rows.forEach((r, i) => {
    for (let j = 0; j < p; j++) a[i * p + j] = ((s.x[r] ?? 0) - cutoff) ** j
    yy[i] = y[r] ?? 0
  })
  const fit = wls(a, rows.length, p, yy, w)
  const cov = covariance(fit, a, 'HC1', w)
  return {
    intercept: fit.beta[0] ?? Number.NaN,
    variance: cov[0] ?? Number.NaN,
    slope: fit.beta[1] ?? Number.NaN,
    n: rows.length,
  }
}

/**
 * The Imbens-Kalyanaraman (2012) bandwidth for a local linear sharp design with a triangular kernel: a rule-of-thumb
 * pilot, a global cubic for the third derivative, local quadratics for the curvature on each side, and the plug-in
 * minimiser of the asymptotic mean squared error with their regularisation term.
 *
 * @param running - The running variable.
 * @param outcome - The outcome.
 * @param cutoff - The cutoff (default 0).
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { imbensKalyanaraman } from 'mdml-engine/causal'
 * const x = Array.from({ length: 400 }, (_, i) => (i - 200) / 50)
 * const y = x.map((v) => 1 + 0.5 * v + (v >= 0 ? 2 : 0) + 0.3 * Math.sin(37 * v))
 * imbensKalyanaraman(x, y) > 0 // => true
 * ```
 */
export function imbensKalyanaraman(running: TensorLike, outcome: TensorLike, cutoff = 0): number {
  const s = read({ running, outcome }, 'imbensKalyanaraman')
  const n = s.x.length
  const sdx = Math.sqrt(variance(s.x, 1))
  const h1 = 1.84 * sdx * n ** (-1 / 5)
  const window = (lo: number, hi: number): number[] =>
    Array.from(s.x.keys()).filter((i) => (s.x[i] ?? 0) >= lo && (s.x[i] ?? 0) < hi)
  const left1 = window(cutoff - h1, cutoff)
  const right1 = window(cutoff, cutoff + h1 + 1e-12)
  const sd2 = (rows: number[]): number =>
    variance(
      rows.map((i) => s.y[i] ?? 0),
      1,
    )
  const fc = (left1.length + right1.length) / (2 * n * h1)
  const sigL = sd2(left1)
  const sigR = sd2(right1)
  // Third derivative from a global cubic with a jump.
  const a = new Float64Array(n * 5)
  for (let i = 0; i < n; i++) {
    const v = (s.x[i] ?? 0) - cutoff
    a.set([1, v >= 0 ? 1 : 0, v, v * v, v * v * v], i * 5)
  }
  const m3 = 6 * (wls(a, n, 5, s.y).beta[4] ?? 0)
  const nl = Array.from(s.x).filter((v) => v < cutoff).length
  const nr = n - nl
  const h2 = (sig: number, count: number): number => 3.56 * (sig / (fc * m3 * m3)) ** (1 / 7) * count ** (-1 / 7)
  const curvature = (above: boolean, h: number): { m2: number; count: number } => {
    const rows = above ? window(cutoff, cutoff + h + 1e-12) : window(cutoff - h, cutoff)
    if (rows.length < 4) return { m2: 0, count: rows.length }
    const q = new Float64Array(rows.length * 3)
    const yy = new Float64Array(rows.length)
    rows.forEach((r, i) => {
      const v = (s.x[r] ?? 0) - cutoff
      q.set([1, v, v * v], i * 3)
      yy[i] = s.y[r] ?? 0
    })
    return { m2: 2 * (wls(q, rows.length, 3, yy).beta[2] ?? 0), count: rows.length }
  }
  const hl = Number.isFinite(h2(sigL, nl)) ? h2(sigL, nl) : h1
  const hr = Number.isFinite(h2(sigR, nr)) ? h2(sigR, nr) : h1
  const cl = curvature(false, hl)
  const cr = curvature(true, hr)
  const reg = (sig: number, count: number, h: number): number => (2160 * sig) / (Math.max(1, count) * h ** 4)
  const denom = (cr.m2 - cl.m2) ** 2 + reg(sigR, cr.count, hr) + reg(sigL, cl.count, hl)
  return 3.4375 * ((sigL + sigR) / (fc * denom)) ** (1 / 5) * n ** (-1 / 5)
}

/**
 * Regression discontinuity: the jump of the outcome at the cutoff estimated by local polynomial fits within a
 * bandwidth on each side. A sharp design (everyone above is treated) reads the jump as the effect at the cutoff; a fuzzy
 * design (the probability of treatment jumps at the cutoff) divides it by the jump in treatment, which is two-stage least
 * squares with the cutoff as the instrument on the kernel-weighted window. The standard error is heteroscedasticity-robust.
 *
 * @param data - Running variable, outcome and (fuzzy) the treatment received.
 * @param options - Cutoff, bandwidth (`'ik'` or a number), kernel, polynomial order and design.
 * @returns The effect with its interval, the bandwidth and the two local fits.
 * @example
 * ```ts
 * import { regressionDiscontinuity } from 'mdml-engine/causal'
 * const running = Array.from({ length: 200 }, (_, i) => (i - 100) / 25)
 * const outcome = running.map((v) => 1 + 0.5 * v + (v >= 0 ? 2 : 0))
 * regressionDiscontinuity({ running, outcome }, { bandwidth: 2 }).effect.estimate.value // => 2
 * ```
 */
export function regressionDiscontinuity(data: RdData, options: RdOptions = {}): RdResult {
  const s = read(data, 'regressionDiscontinuity')
  const cutoff = options.cutoff ?? 0
  const kernel = options.kernel ?? 'triangular'
  const order = options.order ?? 1
  const fuzzy = options.design === 'fuzzy'
  if (fuzzy && s.d === undefined)
    throw new MdmlError('value', 'regressionDiscontinuity: a fuzzy design needs the treatment received')
  const h =
    options.bandwidth === undefined || options.bandwidth === 'ik'
      ? imbensKalyanaraman(data.running, data.outcome, cutoff)
      : options.bandwidth
  const left = sideFit(s, s.y, cutoff, h, kernel, order, false)
  const right = sideFit(s, s.y, cutoff, h, kernel, order, true)
  const jump = right.intercept - left.intercept
  const level = options.level ?? 0.95
  if (!fuzzy || s.d === undefined) {
    const n = left.n + right.n
    return {
      effect: effectFromSe('effect-at-cutoff', jump, Math.sqrt(left.variance + right.variance), n, {
        level,
        method: `sharp regression discontinuity (local order ${order}, ${kernel} kernel, bandwidth ${h.toPrecision(4)})`,
      }),
      bandwidth: h,
      cutoff,
      left,
      right,
    }
  }
  const dl = sideFit(s, s.d, cutoff, h, kernel, order, false)
  const dr = sideFit(s, s.d, cutoff, h, kernel, order, true)
  // Stacked weighted 2SLS: y on d, instrumented by the cutoff, with a polynomial on each side.
  const rows = Array.from(s.x.keys()).filter((i) => kernelWeight(kernel, ((s.x[i] ?? 0) - cutoff) / h) > 0)
  const p = order
  const exog = new Float64Array(rows.length * 2 * p)
  rows.forEach((r, i) => {
    const v = (s.x[r] ?? 0) - cutoff
    const above = v >= 0
    for (let j = 1; j <= p; j++) {
      exog[i * 2 * p + (j - 1)] = above ? 0 : v ** j
      exog[i * 2 * p + p + (j - 1)] = above ? v ** j : 0
    }
  })
  const iv = instrumentalVariables(
    {
      outcome: rows.map((r) => s.y[r] ?? 0),
      endogenous: rows.map((r) => s.d?.[r] ?? 0),
      instruments: rows.map((r) => ((s.x[r] ?? 0) >= cutoff ? 1 : 0)),
      exogenous: matrixShape(exog, rows.length, 2 * p),
    },
    { covariance: 'HC1', weights: rows.map((r) => kernelWeight(kernel, ((s.x[r] ?? 0) - cutoff) / h)), level },
  )
  // The window holds an intercept per side through the exogenous polynomial plus a common intercept; the side-specific
  // intercepts are identified by the instrument and the common one, so the coefficient is the ratio of the jumps.
  return {
    effect: { ...iv.effect, estimand: 'effect-at-cutoff' },
    bandwidth: h,
    cutoff,
    left,
    right,
    firstStageJump: dr.intercept - dl.intercept,
    outcomeJump: jump,
  }
}

function matrixShape(data: Float64Array, n: number, k: number): TensorLike {
  return Array.from({ length: n }, (_, i) => Array.from(data.subarray(i * k, (i + 1) * k)))
}

/** One bandwidth of a {@link rdBandwidthSweep}. */
export interface BandwidthPoint {
  readonly bandwidth: number
  readonly estimate: number
  readonly standardError: number
  readonly interval: readonly [number, number]
  /** Observations inside the window. */
  readonly n: number
}

/**
 * The regression-discontinuity estimate over a grid of bandwidths: narrow windows are noisy and nearly unbiased, wide
 * ones precise and biased by the curvature, which the sweep shows beside the Imbens-Kalyanaraman choice.
 *
 * @param data - As {@link regressionDiscontinuity}.
 * @param bandwidths - The bandwidths to try.
 * @param options - As {@link regressionDiscontinuity} (the bandwidth is ignored).
 * @returns One point per bandwidth.
 * @example
 * ```ts
 * import { rdBandwidthSweep } from 'mdml-engine/causal'
 * const running = Array.from({ length: 200 }, (_, i) => (i - 100) / 25)
 * const outcome = running.map((v) => 1 + 0.5 * v + (v >= 0 ? 2 : 0))
 * rdBandwidthSweep({ running, outcome }, [1, 2]).map((p) => p.n) // => [49, 99]
 * ```
 */
export function rdBandwidthSweep(
  data: RdData,
  bandwidths: readonly number[],
  options: Omit<RdOptions, 'bandwidth'> = {},
): BandwidthPoint[] {
  return bandwidths.map((bandwidth) => {
    const r = regressionDiscontinuity(data, { ...options, bandwidth })
    return {
      bandwidth,
      estimate: r.effect.estimate.value,
      standardError: r.effect.estimate.standardError ?? Number.NaN,
      interval: r.effect.estimate.interval ?? [Number.NaN, Number.NaN],
      n: r.left.n + r.right.n,
    }
  })
}

/** The result of {@link densityTest}. */
export interface DensityTest {
  /** The log difference of the estimated densities at the cutoff, above minus below. */
  readonly logDifference: number
  readonly standardError: number
  readonly statistic: number
  readonly pValue: number
  /** The estimated densities just below and above the cutoff. */
  readonly density: readonly [number, number]
  readonly bandwidth: number
  readonly binWidth: number
}

/**
 * A McCrary-style test of manipulation of the running variable: the data are binned (with the cutoff on a bin edge), the
 * normalised counts smoothed by local linear regression on each side, and the log of the ratio of the two densities at
 * the cutoff compared with zero using McCrary's standard error.
 *
 * @param running - The running variable.
 * @param options - `cutoff` (0), `binWidth` (default `2 sd / sqrt(n)`) and `bandwidth` (default `3.348 sd n^(-1/5)`).
 * @returns The log difference, its error, the z statistic and the p-value.
 * @example
 * ```ts
 * import { densityTest } from 'mdml-engine/causal'
 * const x = Array.from({ length: 1000 }, (_, i) => (i - 500) / 100)
 * densityTest(x, { bandwidth: 1 }).pValue > 0.05 // => true
 * ```
 */
export function densityTest(
  running: TensorLike,
  options: { readonly cutoff?: number; readonly binWidth?: number; readonly bandwidth?: number } = {},
): DensityTest {
  const x = targetVector(running, 'densityTest')
  const n = x.length
  const c = options.cutoff ?? 0
  const sd = Math.sqrt(variance(x, 1))
  const b = options.binWidth ?? (2 * sd) / Math.sqrt(n)
  const h = options.bandwidth ?? 3.348 * sd * n ** (-1 / 5)
  const lo = Math.floor((Math.min(...x) - c) / b)
  const hi = Math.floor((Math.max(...x) - c) / b)
  const counts = new Map<number, number>()
  for (const v of x) counts.set(Math.floor((v - c) / b), (counts.get(Math.floor((v - c) / b)) ?? 0) + 1)
  const side = (above: boolean): number => {
    const pts: number[] = []
    for (let j = lo; j <= hi; j++) {
      const mid = c + (j + 0.5) * b
      if (j >= 0 === above && Math.abs(mid - c) < h) pts.push(j)
    }
    if (pts.length < 3) throw new MdmlError('value', 'densityTest: too few bins inside the bandwidth')
    const a = new Float64Array(pts.length * 2)
    const y = new Float64Array(pts.length)
    const w = new Float64Array(pts.length)
    pts.forEach((j, i) => {
      const mid = c + (j + 0.5) * b
      a.set([1, mid - c], i * 2)
      y[i] = (counts.get(j) ?? 0) / (n * b)
      w[i] = kernelWeight('triangular', (mid - c) / h)
    })
    return wls(a, pts.length, 2, y, w).beta[0] ?? Number.NaN
  }
  const below = side(false)
  const above = side(true)
  const theta = Math.log(above) - Math.log(below)
  const se = Math.sqrt((24 / 5 / (n * h)) * (1 / above + 1 / below))
  const z = theta / se
  return {
    logDifference: theta,
    standardError: se,
    statistic: z,
    pValue: 2 * (1 - normCdf(Math.abs(z))),
    density: [below, above],
    bandwidth: h,
    binWidth: b,
  }
}

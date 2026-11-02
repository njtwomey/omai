/**
 * STL, the seasonal-trend decomposition by loess (Cleveland et al. 1990; statsmodels' `STL` is the reference), and
 * MSTL for several seasonal periods. STL is an `Algorithm` whose step is one inner loop (detrend, smooth the
 * cycle-subseries, low-pass filter, deseasonalise, smooth the trend), with the robustness weights refreshed after every
 * outer pass; MSTL steps through the STL fits of its periods. Both record the residual variance and the components.
 */

import { MdmlError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { asSeries } from './_series.ts'
import type { DecompositionResult } from './decomposition.ts'

/** Options of {@link stl}; the names follow statsmodels' `STL`. */
export interface StlOptions {
  /** The seasonal period (an integer of at least 2). */
  readonly period: number
  /** The length of the seasonal smoother, an odd integer of at least 3 (default 7). */
  readonly seasonal?: number
  /** The length of the trend smoother, odd and above the period (default the smallest odd integer above 1.5 period / (1 - 1.5 / seasonal)). */
  readonly trend?: number
  /** The length of the low-pass filter, odd and above the period (default the smallest odd integer above the period). */
  readonly lowPass?: number
  /** The degrees (0 or 1) of the three loess smoothers (default 1). */
  readonly seasonalDegree?: number
  readonly trendDegree?: number
  readonly lowPassDegree?: number
  /** Robust fitting: weights that down-weight outliers, refreshed over outer passes (default false). */
  readonly robust?: boolean
  /** The loess is evaluated every `jump` points and interpolated between (default 1, every point). */
  readonly seasonalJump?: number
  readonly trendJump?: number
  readonly lowPassJump?: number
  /** Inner loops per pass (default 2 when robust, otherwise 5). */
  readonly innerIterations?: number
  /** Outer (robustness) passes after the first (default 15 when robust, otherwise 0). */
  readonly outerIterations?: number
}

interface Config {
  readonly n: number
  readonly np: number
  readonly ns: number
  readonly nt: number
  readonly nl: number
  readonly isdeg: number
  readonly itdeg: number
  readonly ildeg: number
  readonly nsjump: number
  readonly ntjump: number
  readonly nljump: number
  readonly ni: number
  readonly no: number
}

const isOdd = (v: number): boolean => Number.isInteger(v) && v > 0 && v % 2 === 1

function configOf(n: number, o: StlOptions): Config {
  const np = o.period
  if (!(Number.isInteger(np) && np >= 2))
    throw new MdmlError('value', 'stl: the period must be an integer of at least 2')
  const ns = o.seasonal ?? 7
  if (!(isOdd(ns) && ns >= 3)) throw new MdmlError('value', 'stl: seasonal must be an odd integer of at least 3')
  let nt = o.trend ?? Math.ceil((1.5 * np) / (1 - 1.5 / ns))
  if (o.trend === undefined && nt % 2 === 0) nt += 1
  if (!(isOdd(nt) && nt >= 3 && nt > np)) throw new MdmlError('value', 'stl: trend must be odd and above the period')
  let nl = o.lowPass ?? np + 1
  if (o.lowPass === undefined && nl % 2 === 0) nl += 1
  if (!(isOdd(nl) && nl >= 3 && nl > np)) throw new MdmlError('value', 'stl: lowPass must be odd and above the period')
  const robust = o.robust ?? false
  for (const [name, v] of [
    ['seasonalJump', o.seasonalJump ?? 1],
    ['trendJump', o.trendJump ?? 1],
    ['lowPassJump', o.lowPassJump ?? 1],
  ] as const)
    if (!(Number.isInteger(v) && v >= 1)) throw new MdmlError('value', `stl: ${name} must be a positive integer`)
  if (n < 2 * np) throw new MdmlError('value', 'stl: the series must cover at least two periods')
  return {
    n,
    np,
    ns,
    nt,
    nl,
    isdeg: o.seasonalDegree ?? 1,
    itdeg: o.trendDegree ?? 1,
    ildeg: o.lowPassDegree ?? 1,
    nsjump: o.seasonalJump ?? 1,
    ntjump: o.trendJump ?? 1,
    nljump: o.lowPassJump ?? 1,
    ni: o.innerIterations ?? (robust ? 2 : 5),
    no: o.outerIterations ?? (robust ? 15 : 0),
  }
}

/** The local loess fit at `xs` (1-based) of y over the window `nleft..nright` (1-based); NaN when all weights vanish. */
function est(
  y: Float64Array,
  n: number,
  len: number,
  ideg: number,
  xs: number,
  nleft: number,
  nright: number,
  w: Float64Array,
  userw: boolean,
  rw: Float64Array,
): number {
  const range = n - 1
  let h = Math.max(xs - nleft, nright - xs)
  if (len > n) h += Math.floor((len - n) / 2)
  const h9 = 0.999 * h
  const h1 = 0.001 * h
  let a = 0
  for (let j = nleft - 1; j < nright; j++) {
    w[j] = 0
    const r = Math.abs(j + 1 - xs)
    if (r <= h9) {
      w[j] = r <= h1 ? 1 : (1 - (r / h) ** 3) ** 3
      if (userw) w[j] = (w[j] ?? 0) * (rw[j] ?? 0)
      a += w[j] ?? 0
    }
  }
  if (a <= 0) return Number.NaN
  for (let j = nleft - 1; j < nright; j++) w[j] = (w[j] ?? 0) / a
  if (h > 0 && ideg > 0) {
    a = 0
    for (let j = nleft - 1; j < nright; j++) a += (w[j] ?? 0) * (j + 1)
    let b = xs - a
    let c = 0
    for (let j = nleft - 1; j < nright; j++) c += (w[j] ?? 0) * (j + 1 - a) ** 2
    if (Math.sqrt(c) > 0.001 * range) {
      b /= c
      for (let j = nleft - 1; j < nright; j++) w[j] = (w[j] ?? 0) * (b * (j + 1 - a) + 1)
    }
  }
  let ys = 0
  for (let j = nleft - 1; j < nright; j++) ys += (w[j] ?? 0) * (y[j] ?? 0)
  return ys
}

/** Loess smoothing of y (length n) into ys with window `len`, evaluating every `njump` points and interpolating. */
function ess(
  y: Float64Array,
  n: number,
  len: number,
  ideg: number,
  njump: number,
  userw: boolean,
  rw: Float64Array,
  ys: Float64Array,
  res: Float64Array,
): void {
  if (n < 2) {
    ys[0] = y[0] ?? 0
    return
  }
  const newnj = Math.min(njump, n - 1)
  let nleft = 0
  let nright = 0
  const fit = (i: number): void => {
    const v = est(y, n, len, ideg, i + 1, nleft, nright, res, userw, rw)
    ys[i] = Number.isNaN(v) ? (y[i] ?? 0) : v
  }
  if (len >= n) {
    nleft = 1
    nright = n
    for (let i = 0; i < n; i += newnj) fit(i)
  } else if (newnj === 1) {
    const nsh = Math.floor((len + 2) / 2)
    nleft = 1
    nright = len
    for (let i = 0; i < n; i++) {
      if (i + 1 > nsh && nright !== n) {
        nleft += 1
        nright += 1
      }
      fit(i)
    }
  } else {
    const nsh = Math.floor((len + 1) / 2)
    for (let i = 0; i < n; i += newnj) {
      if (i + 1 < nsh) {
        nleft = 1
        nright = len
      } else if (i + 1 >= n - nsh + 1) {
        nleft = n - len + 1
        nright = n
      } else {
        nleft = i + 1 - nsh + 1
        nright = len + i + 1 - nsh
      }
      fit(i)
    }
  }
  if (newnj === 1) return
  for (let i = 0; i < n - newnj; i += newnj) {
    const delta = ((ys[i + newnj] ?? 0) - (ys[i] ?? 0)) / newnj
    for (let j = i; j < i + newnj; j++) ys[j] = (ys[i] ?? 0) + delta * (j - i)
  }
  const k = Math.floor((n - 1) / newnj) * newnj + 1
  if (k !== n) {
    fit(n - 1)
    if (k !== n - 1) {
      const delta = ((ys[n - 1] ?? 0) - (ys[k - 1] ?? 0)) / (n - k)
      for (let j = k; j < n; j++) ys[j] = (ys[k - 1] ?? 0) + delta * (j + 1 - k)
    }
  }
}

/** Moving average of length `len` of x[0..n), written to ave (n - len + 1 values). */
function movingAverage(x: Float64Array, n: number, len: number, ave: Float64Array): void {
  const newn = n - len + 1
  let v = 0
  for (let i = 0; i < len; i++) v += x[i] ?? 0
  ave[0] = v / len
  for (let j = 1, k = len, m = 0; j < newn; j++, k++, m++) {
    v += (x[k] ?? 0) - (x[m] ?? 0)
    ave[j] = v / len
  }
}

/** One inner loop of STL: updates season and trend in place (the Fortran `stlstp` body). */
function innerLoop(
  y: Float64Array,
  c: Config,
  userw: boolean,
  rw: Float64Array,
  season: Float64Array,
  trend: Float64Array,
): void {
  const { n, np } = c
  const m = n + 2 * np
  const work = Array.from({ length: 5 }, () => new Float64Array(m))
  const w0 = work[0] as Float64Array
  const w1 = work[1] as Float64Array
  const w2 = work[2] as Float64Array
  const w3 = work[3] as Float64Array
  const w4 = work[4] as Float64Array
  for (let i = 0; i < n; i++) w0[i] = (y[i] ?? 0) - (trend[i] ?? 0)
  // Cycle-subseries smoothing (stlss), each subseries extended by one value at both ends.
  for (let j = 0; j < np; j++) {
    const k = Math.floor((n - (j + 1)) / np) + 1
    for (let i = 0; i < k; i++) w2[i] = w0[i * np + j] ?? 0
    if (userw) for (let i = 0; i < k; i++) w4[i] = rw[i * np + j] ?? 0
    // work2[1:] receives the smooth, season serves as the scratch weights.
    ess(w2, k, c.ns, c.isdeg, c.nsjump, userw, w4, w3.subarray(1), season)
    const first = est(w2, k, c.ns, c.isdeg, 0, 1, Math.min(c.ns, k), season, userw, w4)
    w3[0] = Number.isNaN(first) ? (w3[1] ?? 0) : first
    const last = est(w2, k, c.ns, c.isdeg, k + 1, Math.max(1, k - c.ns + 1), k, season, userw, w4)
    w3[k + 1] = Number.isNaN(last) ? (w3[k] ?? 0) : last
    for (let i = 0; i < k + 2; i++) w1[i * np + j] = w3[i] ?? 0
  }
  // Low-pass filter of the smoothed cycle-subseries (stlfts): moving averages np, np, 3, then loess.
  movingAverage(w1, m, np, w2)
  movingAverage(w2, m - np + 1, np, w0)
  movingAverage(w0, m - 2 * np + 2, 3, w2)
  ess(w2, n, c.nl, c.ildeg, c.nljump, false, w3, w0, w4)
  for (let i = 0; i < n; i++) {
    season[i] = (w1[np + i] ?? 0) - (w0[i] ?? 0)
    w0[i] = (y[i] ?? 0) - (season[i] ?? 0)
  }
  ess(w0, n, c.nt, c.itdeg, c.ntjump, userw, rw, trend, w2)
}

/** The bisquare robustness weights of the residuals y - fit, from six times their median absolute value. */
function robustnessWeights(y: Float64Array, fit: Float64Array, rw: Float64Array): void {
  const n = y.length
  for (let i = 0; i < n; i++) rw[i] = Math.abs((y[i] ?? 0) - (fit[i] ?? 0))
  const sorted = Float64Array.from(rw).sort()
  const mid0 = Math.floor(n / 2)
  const mid1 = n - mid0 - 1
  const cmad = 3 * ((sorted[mid0] ?? 0) + (sorted[mid1] ?? 0))
  if (cmad === 0) {
    rw.fill(1)
    return
  }
  const c9 = 0.999 * cmad
  const c1 = 0.001 * cmad
  for (let i = 0; i < n; i++) {
    const r = rw[i] ?? 0
    rw[i] = r <= c1 ? 1 : r <= c9 ? (1 - (r / cmad) ** 2) ** 2 : 0
  }
}

/** The state of {@link stlAlgorithm}. */
export interface StlState {
  /** Inner loops done. */
  readonly t: number
  readonly season: Float64Array
  readonly trend: Float64Array
  /** The robustness weights (all one until the first outer pass completes). */
  readonly weights: Float64Array
  /** Whether the weights are in use. */
  readonly robustPass: boolean
}

/**
 * STL as an algorithm, one inner loop per step: detrend, smooth the cycle-subseries by loess, low-pass filter them,
 * deseasonalise and smooth the trend. After each pass of `innerIterations` loops (and until `outerIterations` passes
 * beyond the first are done) the robustness weights are recomputed from the residuals. It records the residual variance
 * and the seasonal and trend components, and its state serialises.
 *
 * @param y - The series.
 * @param options - The period and the smoothers (see {@link StlOptions}).
 * @returns The algorithm and its total number of steps (`innerIterations * (outerIterations + 1)`).
 * @example
 * ```ts
 * import { stlAlgorithm } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 24 }, (_, t) => t + [2, 0, -2, 0][t % 4])
 * stlAlgorithm(y, { period: 4 }).steps // => 5
 * ```
 */
export function stlAlgorithm(
  y: ArrayLike<number>,
  options: StlOptions,
): { readonly algorithm: Algorithm<StlState>; readonly steps: number } {
  const x = asSeries(Float64Array.from(y), 'stl')
  const c = configOf(x.length, options)
  const total = c.ni * (c.no + 1)
  const algorithm: Algorithm<StlState> = {
    name: 'stl',
    config: { ...c, period: c.np },
    schema: channels({
      metrics: { residualVariance: { description: 'the variance of y - season - trend', direction: 'lower' } },
      tensors: {
        season: { role: 'state', description: 'the seasonal component', shape: [c.n] },
        trend: { role: 'state', description: 'the trend component', shape: [c.n] },
      },
    }),
    init: () => ({
      t: 0,
      season: new Float64Array(c.n),
      trend: new Float64Array(c.n),
      weights: new Float64Array(c.n).fill(1),
      robustPass: false,
    }),
    step: (s) => {
      const season = Float64Array.from(s.season)
      const trend = Float64Array.from(s.trend)
      const weights = Float64Array.from(s.weights)
      innerLoop(x, c, s.robustPass, weights, season, trend)
      const t = s.t + 1
      let robustPass = s.robustPass
      if (t % c.ni === 0 && t / c.ni <= c.no) {
        const fit = Float64Array.from(season, (v, i) => v + (trend[i] ?? 0))
        robustnessWeights(x, fit, weights)
        robustPass = true
      }
      return { t, season, trend, weights, robustPass }
    },
    done: (s) => (s.t >= total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { residualVariance: residualVariance(x, s) },
      tensors: { season: { role: 'state', value: s.season }, trend: { role: 'state', value: s.trend } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StlState,
  }
  return { algorithm, steps: total }
}

function residualVariance(y: Float64Array, s: StlState): number {
  const r = Float64Array.from(y, (v, i) => v - (s.season[i] ?? 0) - (s.trend[i] ?? 0))
  const mean = r.reduce((a, b) => a + b, 0) / r.length
  return r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length
}

/**
 * Seasonal-trend decomposition by loess (statsmodels' `STL(...).fit()`).
 *
 * @param y - The series.
 * @param options - The period and the smoothers.
 * @returns The observed series, trend, seasonal component, residual and robustness weights.
 * @index function STL seasonal-trend decomposition; mstl handles several seasons.
 * @example
 * ```ts
 * import { stl } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 32 }, (_, t) => 10 + t + [3, 0, -3, 0][t % 4])
 * const d = stl(y, { period: 4 })
 * Math.abs((d.seasonal[0] as number) - 3) < 0.5 // => true
 * ```
 */
export function stl(y: ArrayLike<number>, options: StlOptions): DecompositionResult {
  const x = asSeries(Float64Array.from(y), 'stl')
  const { algorithm, steps } = stlAlgorithm(x, options)
  const { state } = runAlgorithm(algorithm, { maxSteps: steps })
  return {
    observed: Array.from(x),
    trend: Array.from(state.trend),
    seasonal: Array.from(state.season),
    resid: Array.from(x, (v, i) => v - (state.season[i] ?? 0) - (state.trend[i] ?? 0)),
    weights: Array.from(state.weights),
  }
}

/** Options of {@link mstl}. */
export interface MstlOptions {
  /** The seasonal periods (unordered; processed shortest first). */
  readonly periods: readonly number[]
  /** The seasonal smoother lengths, one per period (default 7 + 4 i for the i-th shortest period, i = 1, 2, ...). */
  readonly windows?: readonly number[]
  /** Passes over the periods (default 2; 1 for a single period). */
  readonly iterate?: number
  /** A Box-Cox transform with this exponent applied first (0 is the logarithm; default none). */
  readonly lambda?: number
  /** The remaining STL options, applied to every period. */
  readonly stl?: Omit<StlOptions, 'period' | 'seasonal'>
}

/** The result of {@link mstl}: one seasonal component per period (shortest first). */
export interface MstlResult {
  readonly observed: readonly number[]
  readonly trend: readonly number[]
  readonly seasonal: readonly (readonly number[])[]
  readonly resid: readonly number[]
  readonly periods: readonly number[]
}

/** The state of {@link mstlAlgorithm}. */
export interface MstlState {
  /** STL fits done (one per period per pass). */
  readonly t: number
  /** The seasonal component of each period. */
  readonly seasonal: (readonly number[])[]
  /** The series with every seasonal component but the last one fitted removed. */
  readonly deseasonalised: readonly number[]
  readonly trend: readonly number[]
}

function boxCox(y: readonly number[], lambda: number): number[] {
  return y.map((v) => {
    if (!(v > 0)) throw new MdmlError('value', 'mstl: a Box-Cox transform needs positive data')
    return lambda === 0 ? Math.log(v) : (v ** lambda - 1) / lambda
  })
}

/**
 * MSTL (Bandara, Hyndman and Bergmeir) as an algorithm: STL is applied to each seasonal period in turn on the series
 * with the other seasonal components removed, and the passes repeat; one step is one STL fit.
 *
 * @param y - The series.
 * @param options - The periods, windows, passes and STL options.
 * @returns The algorithm, the number of steps (`passes * periods`) and the periods in processing order.
 * @example
 * ```ts
 * import { mstlAlgorithm } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 96 }, (_, t) => Math.sin(t) + Math.sin(t / 4) + 0.01 * t)
 * mstlAlgorithm(y, { periods: [24, 6] }).steps // => 4
 * ```
 */
export function mstlAlgorithm(
  y: ArrayLike<number>,
  options: MstlOptions,
): { readonly algorithm: Algorithm<MstlState>; readonly steps: number; readonly periods: readonly number[] } {
  const raw = Array.from(asSeries(Float64Array.from(y), 'mstl'))
  const x = options.lambda === undefined ? raw : boxCox(raw, options.lambda)
  const n = x.length
  const defaults = options.periods.map((_, i) => 7 + 4 * (i + 1))
  const windows = options.windows ?? defaults
  if (windows.length !== options.periods.length)
    throw new MdmlError('shape', 'mstl: periods and windows must have the same length')
  const pairs = options.periods
    .map((p, i) => [p, windows[i] ?? 7] as const)
    .sort((a, b) => a[0] - b[0])
    .filter(([p]) => p < n / 2)
  if (pairs.length === 0) throw new MdmlError('value', 'mstl: no period shorter than half the series')
  const periods = pairs.map(([p]) => p)
  const passes = pairs.length === 1 ? 1 : (options.iterate ?? 2)
  const total = passes * pairs.length
  const algorithm: Algorithm<MstlState> = {
    name: 'mstl',
    config: { periods, windows: pairs.map(([, w]) => w), passes },
    schema: channels({
      metrics: {
        residualVariance: {
          description: 'the variance of the series less the trend and every seasonal',
          direction: 'lower',
        },
      },
      tensors: { trend: { role: 'state', description: 'the trend', shape: [n] } },
    }),
    init: () => ({
      t: 0,
      seasonal: pairs.map(() => new Array<number>(n).fill(0)),
      deseasonalised: [...x],
      trend: new Array<number>(n).fill(0),
    }),
    step: (s) => {
      const i = s.t % pairs.length
      const [period, window] = pairs[i] as readonly [number, number]
      const seasonal = s.seasonal.map((v) => [...v])
      // Put this period's previous estimate back, decompose, and remove the new estimate.
      const input = s.deseasonalised.map((v, k) => v + (seasonal[i]?.[k] ?? 0))
      const fit = stl(input, { ...(options.stl ?? {}), period, seasonal: window })
      seasonal[i] = [...fit.seasonal]
      const deseasonalised = input.map((v, k) => v - (fit.seasonal[k] ?? 0))
      return { t: s.t + 1, seasonal, deseasonalised, trend: fit.trend }
    },
    done: (s) => (s.t >= total ? 'converged' : undefined),
    observe: (s) => {
      const r = s.deseasonalised.map((v, k) => v - (s.trend[k] ?? 0))
      const mean = r.reduce((a, b) => a + b, 0) / n
      return {
        metrics: { residualVariance: r.reduce((a, b) => a + (b - mean) ** 2, 0) / n },
        tensors: { trend: { role: 'state', value: s.trend } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MstlState,
  }
  return { algorithm, steps: total, periods }
}

/**
 * Multiple seasonal-trend decomposition by loess (statsmodels' `MSTL`).
 *
 * @param y - The series.
 * @param options - The periods and the STL settings.
 * @returns The trend, one seasonal component per period (shortest first) and the residual.
 * @example
 * ```ts
 * import { mstl } from 'mdml-engine/timeseries'
 * const y = Array.from({ length: 96 }, (_, t) => Math.sin((2 * Math.PI * t) / 24) + 0.5 * Math.sin((2 * Math.PI * t) / 6) + 0.02 * t)
 * mstl(y, { periods: [6, 24] }).seasonal.length // => 2
 * ```
 */
export function mstl(y: ArrayLike<number>, options: MstlOptions): MstlResult {
  const { algorithm, steps, periods } = mstlAlgorithm(y, options)
  const { state } = runAlgorithm(algorithm, { maxSteps: steps })
  const observed = Array.from(asSeries(Float64Array.from(y), 'mstl'))
  return {
    observed: options.lambda === undefined ? observed : boxCox(observed, options.lambda),
    trend: state.trend,
    seasonal: state.seasonal,
    resid: state.deseasonalised.map((v, k) => v - (state.trend[k] ?? 0)),
    periods,
  }
}

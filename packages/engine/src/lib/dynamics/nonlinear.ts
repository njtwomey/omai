/**
 * Tools for continuous-time nonlinear dynamics and chaos: Poincare sections by event detection, periodic orbits by
 * shooting with their Floquet multipliers, the Lyapunov spectrum of a flow, and pseudo-arclength continuation of
 * equilibria with fold and Hopf detection. For measured time series it adds delay embedding, correlation dimension and
 * the largest Lyapunov exponent (as nolds computes them) and recurrence quantification analysis.
 */

import { jacfwd } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { eigvals, qr, solve } from '../numerics/linalg/index.ts'
import { fsolve } from '../numerics/roots/index.ts'
import {
  concat,
  matmul as matmulValue,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { floatState, fromNumbers, numbersOf, rms } from './ode/_vec.ts'
import { forwardSensitivities } from './ode/adjoint.ts'
import { event, type VectorField } from './ode/problem.ts'
import { type OdeMethodName, solveIvp } from './ode/solve-ivp.ts'

/** Integrator settings shared by the flow tools. */
export interface FlowOptions {
  readonly method?: OdeMethodName
  readonly rtol?: number
  readonly atol?: number
}

const solverOf = (o: FlowOptions): { method: OdeMethodName; rtol: number; atol: number } => ({
  method: o.method ?? 'DOP853',
  rtol: o.rtol ?? 1e-9,
  atol: o.atol ?? 1e-12,
})

/**
 * A Poincare section: the successive crossings of a trajectory with the surface s(y) = 0 in one direction, located by
 * event detection on the dense output.
 *
 * @param f - The field.
 * @param y0 - The initial state.
 * @param surface - The section function s(y).
 * @param options - Crossing direction (default +1), how many crossings (default 100), a time limit (default 1e4) and
 * the integrator.
 * @returns The crossing times and states `[k, ...]`.
 * @concept poincare-section
 * @example
 * ```ts
 * import { poincareSection } from 'whitebox-engine/dynamics'
 * import { stack, sliceAxis, sum, neg } from 'whitebox-engine/tensor'
 * // The harmonic oscillator returns to x = 0 every 2 pi.
 * const f = (_t, y) => stack([sum(sliceAxis(y, 0, 1, 2)), neg(sum(sliceAxis(y, 0, 0, 1)))])
 * const s = poincareSection(f, [0, 1], (y) => y.data[0] ?? 0, { crossings: 3 })
 * Math.abs((s.t[1] ?? 0) - (s.t[0] ?? 0) - 2 * Math.PI) < 1e-7 // => true
 * ```
 */
export function poincareSection(
  f: VectorField,
  y0: TensorLike,
  surface: (y: Tensor) => number,
  options: FlowOptions & { readonly direction?: -1 | 1; readonly crossings?: number; readonly tEnd?: number } = {},
): { readonly t: readonly number[]; readonly points: Value } {
  const crossings = options.crossings ?? 100
  const sol = solveIvp(f, [0, options.tEnd ?? 1e4], y0, {
    ...solverOf(options),
    events: [event((_t, y) => surface(primalOf(y)), { direction: options.direction ?? 1, terminal: crossings })],
  })
  const ys = sol.yEvents[0] ?? []
  const ts = sol.tEvents[0] ?? []
  return { t: ts, points: ys.length === 0 ? fromNumbers(new Float64Array(0), [0, ...floatState(y0).shape]) : stack(ys) }
}

/** A periodic orbit. */
export interface LimitCycle {
  /** A point on the orbit (satisfying the phase condition). */
  readonly y0: Tensor
  readonly period: number
  /** The Floquet multipliers (one is 1, the others decide stability). */
  readonly multipliers: { readonly re: readonly number[]; readonly im: readonly number[] }
  readonly stable: boolean
  readonly converged: boolean
  readonly residual: number
}

/**
 * Find a periodic orbit by shooting: solve phi_T(y0) = y0 for the state and the period together, with the phase
 * condition that one coordinate keeps its guessed value, by Newton's method (the derivatives of the flow taken
 * through the integrator in rescaled time s = t / T). The Floquet multipliers of the orbit come with it.
 *
 * @param f - An autonomous field.
 * @param guess - A state near the orbit.
 * @param period - The guessed period.
 * @param options - The coordinate fixed by the phase condition (default 0), Newton iterations and the integrator.
 * @returns The orbit, period, multipliers and stability.
 * @concept limit-cycle
 * @example
 * ```ts
 * import { limitCycle } from 'whitebox-engine/dynamics'
 * import { stack, sliceAxis, sum, sub, mul } from 'whitebox-engine/tensor'
 * // van der Pol with mu = 1: period about 6.6633.
 * const f = (_t, y) => { const x = sum(sliceAxis(y, 0, 0, 1)); const v = sum(sliceAxis(y, 0, 1, 2)); return stack([v, sub(mul(sub(1, mul(x, x)), v), x)]) }
 * const c = limitCycle(f, [2, 0], 6.6)
 * Math.abs(c.period - 6.6632868593) < 1e-6 // => true
 * ```
 */
export function limitCycle(
  f: VectorField,
  guess: TensorLike,
  period: number,
  options: FlowOptions & { readonly fixedIndex?: number; readonly maxIterations?: number } = {},
): LimitCycle {
  const g = floatState(guess)
  const n = numbersOf(g).length
  const k = options.fixedIndex ?? 0
  const fixedValue = numbersOf(g)[k] ?? 0
  const solver = { ...solverOf(options), rtol: options.rtol ?? 1e-10, atol: options.atol ?? 1e-12 }
  const residual = (z: Value): Value => {
    const y0 = sliceAxis(z, 0, 0, n)
    const T = reshape(sliceAxis(z, 0, n, n + 1), [])
    const end = solveIvp((s, y) => mul(f(s, y), T), [0, 1], y0, { ...solver, tEval: [1] }).ys.at(-1) as Value
    return concat([sub(end, y0), reshape(sub(sliceAxis(y0, 0, k, k + 1), fixedValue), [1])], 0)
  }
  const z0 = concat([reshape(g, [n]), fromNumbers([period], [1])], 0)
  const r = fsolve(residual, z0, { ftol: 1e-10, maxIterations: options.maxIterations ?? 30 })
  const z = numbersOf(r.x)
  const y0 = fromNumbers(z.slice(0, n), g.shape)
  const T = z[n] ?? period
  const multipliers = floquetMultipliers(f, y0, T, options)
  const nontrivial = multipliers.re
    .map((re, i) => Math.hypot(re, multipliers.im[i] ?? 0))
    .sort((a, b) => Math.abs(b - 1) - Math.abs(a - 1))
    .slice(0, n - 1)
  return {
    y0,
    period: T,
    multipliers,
    stable: nontrivial.every((m) => m < 1),
    converged: r.converged,
    residual: r.history[r.history.length - 1] ?? Number.NaN,
  }
}

/**
 * The Floquet multipliers of a periodic orbit: the eigenvalues of the monodromy matrix d phi_T / d y0, from the
 * forward sensitivities over one period.
 *
 * @param f - The field.
 * @param y0 - A point on the orbit.
 * @param period - Its period.
 * @param options - The integrator.
 * @returns The multipliers as real and imaginary parts.
 * @concept floquet-theory
 * @example
 * ```ts
 * import { floquetMultipliers } from 'whitebox-engine/dynamics'
 * import { stack, sliceAxis, sum, neg } from 'whitebox-engine/tensor'
 * const f = (_t, y) => stack([sum(sliceAxis(y, 0, 1, 2)), neg(sum(sliceAxis(y, 0, 0, 1)))])
 * floquetMultipliers(f, [1, 0], 2 * Math.PI).re.map((v) => Math.round(v * 1e6) / 1e6) // => [1, 1]
 * ```
 */
export function floquetMultipliers(
  f: VectorField,
  y0: TensorLike,
  period: number,
  options: FlowOptions = {},
): { readonly re: number[]; readonly im: number[] } {
  const y = floatState(y0)
  const n = numbersOf(y).length
  const s = forwardSensitivities((t, yy) => f(t, yy), y, [0, period], [0], { ...solverOf(options) })
  const M = fromNumbers(s.dy0.data.slice(n * n), [n, n])
  const ev = eigvals(M)
  return { re: Array.from(ev.re.data), im: Array.from(ev.im.data) }
}

/** Options of {@link lyapunovSpectrum}. */
export interface LyapunovSpectrumOptions extends FlowOptions {
  /** Total integration time after the transient (default 100). */
  readonly tEnd?: number
  /** Time between re-orthonormalisations (default 0.5). */
  readonly interval?: number
  /** Transient time discarded first (default 10). */
  readonly transient?: number
  /** How many exponents (default all). */
  readonly count?: number
}

/**
 * The Lyapunov spectrum of a flow by Benettin's method: integrate the state with k tangent vectors under the
 * variational equation Q' = J(y) Q, re-orthonormalise by QR every interval and average the logs of R's diagonal. For
 * Lorenz (sigma = 10, rho = 28, beta = 8/3) the spectrum is about (0.906, 0, -14.57); the sum equals the mean
 * divergence.
 *
 * @param f - An autonomous field.
 * @param y0 - The initial state.
 * @param options - Duration, interval, transient, count and the integrator.
 * @returns The exponents, largest first, and their running estimates.
 * @concept lyapunov-exponent
 * @example
 * ```ts
 * import { lyapunovSpectrum } from 'whitebox-engine/dynamics'
 * import { mul } from 'whitebox-engine/tensor'
 * // A linear flow y' = diag(1, -2) y has exponents 1 and -2.
 * lyapunovSpectrum((_t, y) => mul(y, [1, -2]), [1, 1], { tEnd: 5, transient: 0 }).exponents.map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function lyapunovSpectrum(
  f: VectorField,
  y0: TensorLike,
  options: LyapunovSpectrumOptions = {},
): { readonly exponents: number[]; readonly history: readonly number[][] } {
  const y = floatState(y0)
  const n = numbersOf(y).length
  const k = Math.min(n, options.count ?? n)
  const shape = y.shape
  const solver = { ...solverOf(options), rtol: options.rtol ?? 1e-8, atol: options.atol ?? 1e-10 }
  let state: Value = y
  if ((options.transient ?? 10) > 0) {
    state = solveIvp(f, [0, options.transient ?? 10], y, { ...solver, tEval: [options.transient ?? 10] }).ys.at(
      -1,
    ) as Value
  }
  const interval = options.interval ?? 0.5
  const steps = Math.max(1, Math.round((options.tEnd ?? 100) / interval))
  const augmented: VectorField = (t, z) => {
    const yy = reshape(sliceAxis(z, 0, 0, n), shape)
    const Q = reshape(sliceAxis(z, 0, n, n + n * k), [n, k])
    const g = (v: Value): Value => reshape(f(t, reshape(v, shape)), [n])
    const J = reshape(jacfwd(g)(reshape(yy, [n])), [n, n])
    return concat([reshape(f(t, yy), [n]), reshape(matmulValue(J, Q), [n * k])], 0)
  }
  let Q = fromNumbers(
    Float64Array.from({ length: n * k }, (_, i) => (Math.floor(i / k) === i % k ? 1 : 0)),
    [n, k],
  )
  const sums = new Float64Array(k)
  const history: number[][] = []
  for (let s = 0; s < steps; s++) {
    const z0 = concat([reshape(state, [n]), reshape(Q, [n * k])], 0)
    const sol = solveIvp(augmented, [s * interval, (s + 1) * interval], z0, { ...solver, tEval: [(s + 1) * interval] })
    if (!sol.success) throw new WhiteboxError('convergence', `lyapunovSpectrum: ${sol.message}`)
    const z = numbersOf(sol.ys.at(-1) as Value)
    state = fromNumbers(z.slice(0, n), shape)
    const { q, r } = qr(fromNumbers(z.slice(n), [n, k]))
    const R = primalOf(r)
    const qm = primalOf(q)
    const signs = Float64Array.from({ length: k }, (_, j) => ((R.data[j * k + j] ?? 1) < 0 ? -1 : 1))
    for (let j = 0; j < k; j++) sums[j] = (sums[j] ?? 0) + Math.log(Math.abs(R.data[j * k + j] ?? 1))
    Q = fromNumbers(
      Float64Array.from(qm.data.slice(0, n * k), (v, idx) => v * (signs[idx % k] ?? 1)),
      [n, k],
    )
    history.push(Array.from(sums, (v) => v / ((s + 1) * interval)))
  }
  return { exponents: Array.from(sums, (v) => v / (steps * interval)), history }
}

/**
 * The delay embedding of a scalar series: rows (x_i, x_{i+lag}, ..., x_{i+(m-1) lag}) (Takens).
 *
 * @param series - The series.
 * @param dimension - The embedding dimension m.
 * @param lag - The delay (default 1).
 * @returns The embedded points.
 * @throws WhiteboxError with code `value` when the series is shorter than (m - 1) lag + 1.
 * @concept delay-embedding
 * @example
 * ```ts
 * import { delayEmbedding } from 'whitebox-engine/dynamics'
 * delayEmbedding([1, 2, 3, 4], 2, 2) // => [[1, 3], [2, 4]]
 * ```
 */
export function delayEmbedding(series: readonly number[], dimension: number, lag = 1): number[][] {
  const minLen = (dimension - 1) * lag + 1
  if (series.length < minLen) {
    throw new WhiteboxError('value', `delayEmbedding: need at least ${minLen} points for dimension ${dimension}`)
  }
  return Array.from({ length: series.length - minLen + 1 }, (_, i) =>
    Array.from({ length: dimension }, (_, j) => series[i + j * lag] ?? 0),
  )
}

function distance(a: readonly number[], b: readonly number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return Math.sqrt(s)
}

function lineFit(x: readonly number[], y: readonly number[]): [number, number] {
  const n = x.length
  const mx = x.reduce((a, b) => a + b, 0) / n
  const my = y.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my)
    sxx += ((x[i] ?? 0) - mx) ** 2
  }
  const slope = sxy / sxx
  return [slope, my - slope * mx]
}

function sampleStd(x: readonly number[]): number {
  const m = x.reduce((a, b) => a + b, 0) / x.length
  return Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1))
}

/**
 * The correlation dimension of a time series by Grassberger and Procaccia, as nolds computes it (`corr_dim` with
 * `fit='poly'`): delay-embed, count pairs closer than r for logarithmically spaced r from 0.1 to 0.5 standard
 * deviations (factor 1.03), and fit the slope of log C(r) against log r.
 *
 * @param series - The time series.
 * @param dimension - The embedding dimension.
 * @param options - Lag (default 1) and explicit radii.
 * @returns The dimension estimate and the curve it was fitted to.
 * @concept correlation-dimension
 * @example
 * ```ts
 * import { correlationDimension } from 'whitebox-engine/dynamics'
 * // A sine fills a closed curve: dimension about 1.
 * const x = Array.from({ length: 400 }, (_, i) => Math.sin(0.1 * i))
 * Math.abs(correlationDimension(x, 2).dimension - 1) < 0.1 // => true
 * ```
 */
export function correlationDimension(
  series: readonly number[],
  dimension: number,
  options: { readonly lag?: number; readonly radii?: readonly number[] } = {},
): { readonly dimension: number; readonly logR: number[]; readonly logC: number[] } {
  let radii = options.radii
  if (radii === undefined) {
    const sd = sampleStd(series)
    const lo = 0.1 * sd
    const hi = 0.5 * sd
    const maxI = Math.floor(Math.log(hi / lo) / Math.log(1.03))
    radii = Array.from({ length: maxI + 1 }, (_, i) => lo * 1.03 ** i)
  }
  const orbit = delayEmbedding(series, dimension, options.lag ?? 1)
  const n = orbit.length
  const dists: number[] = []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) dists.push(distance(orbit[i] as number[], orbit[j] as number[]))
  dists.sort((a, b) => a - b)
  const logR: number[] = []
  const logC: number[] = []
  for (const r of radii) {
    // Pairs with d <= r, counted in both orders as nolds does (the diagonal has d = 0 <= r too).
    let lo = 0
    let hi = dists.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((dists[mid] ?? 0) <= r) lo = mid + 1
      else hi = mid
    }
    const c = (2 * lo + n) / (n * (n - 1))
    if (c !== 0) {
      logR.push(Math.log(r))
      logC.push(Math.log(c))
    }
  }
  return { dimension: logR.length === 0 ? Number.NaN : lineFit(logR, logC)[0], logR, logC }
}

/**
 * The largest Lyapunov exponent of a time series by Rosenstein's method, as nolds computes it (`lyap_r` with
 * `fit='poly'`): delay-embed, pair each point with its nearest neighbour at least `minTsep` steps away, follow the
 * mean log distance of the pairs for `trajectoryLen` steps and fit its slope (per time step, divided by tau).
 *
 * @param series - The time series.
 * @param options - Embedding dimension (default 10), lag and minimum temporal separation (estimated as nolds does
 * when absent), trajectory length (default 20), tau (default 1) and fit offset.
 * @returns The exponent estimate and the divergence curve.
 * @throws WhiteboxError with code `value` when the series is too short.
 * @concept lyapunov-exponent
 * @example
 * ```ts
 * import { largestLyapunovExponent } from 'whitebox-engine/dynamics'
 * let x = 0.3
 * const s = Array.from({ length: 500 }, () => (x = 4 * x * (1 - x)))
 * largestLyapunovExponent(s, { dimension: 3, lag: 1, minTsep: 5, trajectoryLen: 4 }).exponent > 0.4 // => true
 * ```
 */
export function largestLyapunovExponent(
  series: readonly number[],
  options: {
    readonly dimension?: number
    readonly lag?: number
    readonly minTsep?: number
    readonly trajectoryLen?: number
    readonly tau?: number
    readonly fitOffset?: number
  } = {},
): { readonly exponent: number; readonly divergence: number[] } {
  const dimension = options.dimension ?? 10
  const trajectoryLen = options.trajectoryLen ?? 20
  let minTsep = options.minTsep
  let lag = options.lag
  if (minTsep === undefined || lag === undefined) {
    const est = rosensteinDefaults(series, dimension, trajectoryLen)
    minTsep = minTsep ?? est.minTsep
    lag = lag ?? est.lag
  }
  const orbit = delayEmbedding(series, dimension, lag)
  const m = orbit.length
  const ntraj = m - trajectoryLen + 1
  if (ntraj <= 0 || ntraj < 2 * minTsep + 2) {
    throw new WhiteboxError('value', 'largestLyapunovExponent: not enough data for these settings')
  }
  const neighbour = new Int32Array(ntraj)
  for (let i = 0; i < ntraj; i++) {
    let best = -1
    let bestD = Number.POSITIVE_INFINITY
    for (let j = 0; j < ntraj; j++) {
      if (Math.abs(i - j) <= minTsep) continue
      const d = distance(orbit[i] as number[], orbit[j] as number[])
      if (d < bestD) {
        bestD = d
        best = j
      }
    }
    neighbour[i] = best
  }
  const divergence: number[] = []
  const ks: number[] = []
  for (let k = 0; k < trajectoryLen; k++) {
    let s = 0
    let c = 0
    for (let i = 0; i < ntraj; i++) {
      const j = neighbour[i] ?? 0
      const d = distance(orbit[i + k] as number[], orbit[j + k] as number[])
      if (d !== 0) {
        s += Math.log(d)
        c += 1
      }
    }
    if (c > 0) {
      divergence.push(s / c)
      ks.push(k)
    }
  }
  const off = options.fitOffset ?? 0
  const [slope] = lineFit(ks.slice(off), divergence.slice(off))
  return { exponent: slope / (options.tau ?? 1), divergence }
}

function rosensteinDefaults(
  series: readonly number[],
  dimension: number,
  trajectoryLen: number,
): { minTsep: number; lag: number } {
  const n = series.length
  // The mean frequency of the power spectrum of the zero-padded series (length 2n - 1), as nolds estimates it.
  const L = 2 * n - 1
  const half = Math.floor(L / 2) + 1
  let num = 0
  let den = 0
  for (let k = 1; k < half; k++) {
    let re = 0
    let im = 0
    for (let t = 0; t < n; t++) {
      const a = (-2 * Math.PI * k * t) / L
      re += (series[t] ?? 0) * Math.cos(a)
      im += (series[t] ?? 0) * Math.sin(a)
    }
    const p = re * re + im * im
    num += (k / L) * p
    den += p
  }
  let minTsep = Math.ceil(1 / (num / den))
  if (minTsep > 0.25 * n) minTsep = Math.floor(0.25 * n)
  // The lag where the (raw) autocorrelation first falls below (1 - 1/e) of its value at zero.
  const acorr = (lag: number): number => {
    let s = 0
    for (let t = 0; t + lag < n; t++) s += (series[t] ?? 0) * (series[t + lag] ?? 0)
    return s
  }
  const eps = acorr(0) * (1 - 1 / Math.E)
  let lag = 1
  for (let i = 1; i < n; i++) {
    lag = i
    if (acorr(i) < eps) break
    const minLen = (dimension - 1) * i + 1 + trajectoryLen - 1 + 2 * minTsep + 1
    if (Math.max(0, n - minLen) < 20) break
  }
  return { minTsep, lag }
}

/** Recurrence quantification measures (Marwan et al.). */
export interface RecurrenceMeasures {
  readonly recurrenceRate: number
  readonly determinism: number
  readonly averageDiagonal: number
  readonly maxDiagonal: number
  /** Shannon entropy of the diagonal line lengths. */
  readonly entropy: number
  readonly laminarity: number
  readonly trappingTime: number
  /** The recurrence matrix, row-major `[n, n]` of 0/1. */
  readonly matrix: Uint8Array
  readonly n: number
}

/**
 * Recurrence quantification analysis: the recurrence matrix R_ij = [|x_i - x_j| <= eps] of states (or a delay
 * embedded series), and the recurrence rate, determinism, average and longest diagonal line, line-length entropy,
 * laminarity and trapping time, with a Theiler window around the main diagonal.
 *
 * @param states - States `[n, d]` (or a 1-D series with `dimension` and `lag`).
 * @param threshold - The recurrence radius eps.
 * @param options - Minimum line length (default 2), Theiler window (default 1: the main diagonal excluded) and
 * embedding for a series.
 * @returns The measures and the matrix.
 * @concept recurrence-quantification
 * @example
 * ```ts
 * import { recurrenceQuantification } from 'whitebox-engine/dynamics'
 * // A periodic series is fully deterministic.
 * const x = Array.from({ length: 60 }, (_, i) => Math.sin((2 * Math.PI * i) / 10))
 * recurrenceQuantification(x, 0.01, { dimension: 2 }).determinism // => 1
 * ```
 */
export function recurrenceQuantification(
  states: readonly number[] | readonly (readonly number[])[],
  threshold: number,
  options: {
    readonly minLine?: number
    readonly theiler?: number
    readonly dimension?: number
    readonly lag?: number
  } = {},
): RecurrenceMeasures {
  const pts: number[][] =
    typeof states[0] === 'number' || states.length === 0
      ? delayEmbedding(states as readonly number[], options.dimension ?? 1, options.lag ?? 1)
      : (states as readonly (readonly number[])[]).map((r) => [...r])
  const n = pts.length
  const w = options.theiler ?? 1
  const lmin = options.minLine ?? 2
  const R = new Uint8Array(n * n)
  let rec = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (Math.abs(i - j) < w) continue
      total += 1
      if (distance(pts[i] as number[], pts[j] as number[]) <= threshold) {
        R[i * n + j] = 1
        rec += 1
      }
    }
  }
  // Diagonal lines (upper triangle, doubled by symmetry) and vertical lines.
  const diag = new Map<number, number>()
  for (let k = w; k < n; k++) {
    let run = 0
    for (let i = 0; i + k <= n; i++) {
      const on = i + k < n && R[i * n + i + k] === 1
      if (on) run += 1
      else if (run > 0) {
        diag.set(run, (diag.get(run) ?? 0) + 2)
        run = 0
      }
    }
  }
  const vert = new Map<number, number>()
  for (let j = 0; j < n; j++) {
    let run = 0
    for (let i = 0; i <= n; i++) {
      const on = i < n && R[i * n + j] === 1
      if (on) run += 1
      else if (run > 0) {
        vert.set(run, (vert.get(run) ?? 0) + 1)
        run = 0
      }
    }
  }
  const summary = (
    h: Map<number, number>,
  ): { points: number; long: number; count: number; max: number; entropy: number } => {
    let points = 0
    let long = 0
    let count = 0
    let max = 0
    for (const [l, c] of h) {
      points += l * c
      if (l >= lmin) {
        long += l * c
        count += c
        max = Math.max(max, l)
      }
    }
    let entropy = 0
    for (const [l, c] of h) {
      if (l < lmin || count === 0) continue
      const p = c / count
      entropy -= p * Math.log(p)
    }
    return { points, long, count, max, entropy }
  }
  const d = summary(diag)
  const v = summary(vert)
  return {
    recurrenceRate: total === 0 ? 0 : rec / total,
    determinism: d.points === 0 ? 0 : d.long / d.points,
    averageDiagonal: d.count === 0 ? 0 : d.long / d.count,
    maxDiagonal: d.max,
    entropy: d.entropy,
    laminarity: v.points === 0 ? 0 : v.long / v.points,
    trappingTime: v.count === 0 ? 0 : v.long / v.count,
    matrix: R,
    n,
  }
}

/** A point of a continued branch of equilibria. */
export interface BranchPoint {
  readonly x: Tensor
  readonly p: number
  readonly stable: boolean
  readonly eigenvalues: { readonly re: readonly number[]; readonly im: readonly number[] }
}

/** A bifurcation detected along a branch. */
export interface Bifurcation {
  readonly type: 'fold' | 'hopf'
  readonly x: Tensor
  readonly p: number
  /** The index of the branch point just after it. */
  readonly index: number
}

/** The state of {@link continuationAlgorithm}. */
export interface ContinuationState {
  readonly z: Float64Array
  readonly tangent: Float64Array
  readonly points: BranchPoint[]
  readonly bifurcations: Bifurcation[]
  readonly failed: boolean
}

/** Options of {@link continueEquilibria}. */
export interface ContinuationOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Arclength step (default 0.05). */
  readonly ds?: number
  /** Number of continuation steps (default 100). */
  readonly steps?: number
  /** Stop outside this parameter range. */
  readonly pRange?: readonly [number, number]
  /** Initial direction of the parameter (default +1). */
  readonly direction?: 1 | -1
  /** Newton tolerance (default 1e-10). */
  readonly tol?: number
}

function analyse(
  F: (x: Value, p: Value) => Value,
  z: Float64Array,
  n: number,
): { J: Float64Array; ev: { re: number[]; im: number[] } } {
  const g = (w: Value): Value => reshape(F(sliceAxis(w, 0, 0, n), reshape(sliceAxis(w, 0, n, n + 1), [])), [n])
  const J = numbersOf(primalOf(reshape(jacfwd(g)(fromNumbers(z, [n + 1])), [n, n + 1])))
  const Jx = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Jx[i * n + j] = J[i * (n + 1) + j] ?? 0
  const e = eigvals(fromNumbers(Jx, [n, n]))
  return { J, ev: { re: Array.from(e.re.data), im: Array.from(e.im.data) } }
}

function hopfTest(ev: { re: number[]; im: number[] }): number {
  // The largest real part among complex eigenvalues (sign changes at a Hopf point).
  let best = Number.NEGATIVE_INFINITY
  ev.re.forEach((re, i) => {
    if (Math.abs(ev.im[i] ?? 0) > 1e-8) best = Math.max(best, re)
  })
  return best
}

/**
 * Pseudo-arclength continuation of equilibria F(x, p) = 0 in one parameter as a traced algorithm (one point per step):
 * Euler predictor along the tangent, Newton corrector on the extended system with the arclength condition, and
 * detection of folds (the parameter component of the tangent changes sign) and Hopf points (a complex pair crosses the
 * imaginary axis), located by linear interpolation.
 *
 * @param F - The field F(x, p) for x `[n]` and scalar p (built from operations).
 * @param x0 - An equilibrium at p0.
 * @param p0 - The starting parameter.
 * @param options - Step, number of steps, parameter range, direction, tolerance and tracing.
 * @returns The algorithm and its step budget.
 * @throws WhiteboxError with code `convergence` when x0 is not an equilibrium.
 * @example
 * ```ts
 * import { continuationAlgorithm } from 'whitebox-engine/dynamics'
 * import { add, mul } from 'whitebox-engine/tensor'
 * continuationAlgorithm((x, p) => add(mul(x, x), p), [-1], -1, { steps: 5 }).steps // => 5
 * ```
 */
export function continuationAlgorithm(
  F: (x: Value, p: Value) => Value,
  x0: TensorLike,
  p0: number,
  options: ContinuationOptions = {},
): { readonly algorithm: Algorithm<ContinuationState>; readonly steps: number } {
  const xv = floatState(x0)
  const n = numbersOf(xv).length
  const ds = options.ds ?? 0.05
  const tol = options.tol ?? 1e-10
  const steps = options.steps ?? 100
  const corrected = fsolve((v) => reshape(F(v, fromNumbers([p0], [])), [n]), reshape(xv, [n]), { ftol: tol })
  if (!corrected.converged) throw new WhiteboxError('convergence', 'continueEquilibria: x0 is not near an equilibrium')
  const z0 = Float64Array.from([...numbersOf(corrected.x), p0])
  const tangentAt = (J: Float64Array, prev: Float64Array | undefined): Float64Array => {
    // Solve [J; prev^T] t = [0; 1] (or with e_p as the last row at the start).
    const A = new Float64Array((n + 1) * (n + 1))
    A.set(J, 0)
    const last = prev ?? Float64Array.from({ length: n + 1 }, (_, i) => (i === n ? 1 : 0))
    A.set(last, n * (n + 1))
    const rhs = new Float64Array(n + 1)
    rhs[n] = 1
    const t = numbersOf(primalOf(solve(fromNumbers(A, [n + 1, n + 1]), fromNumbers(rhs, [n + 1]))))
    const norm = Math.sqrt(t.reduce((s, v) => s + v * v, 0))
    return Float64Array.from(t, (v) => v / norm)
  }
  const point = (z: Float64Array): BranchPoint => {
    const { ev } = analyse(F, z, n)
    return {
      x: fromNumbers(z.slice(0, n), xv.shape),
      p: z[n] ?? 0,
      stable: ev.re.every((v) => v < 0),
      eigenvalues: ev,
    }
  }
  const algorithm: Algorithm<ContinuationState> = {
    name: 'continuation',
    config: { ds, steps, p0 },
    schema: channels({
      metrics: { p: { description: 'the parameter' }, norm: { description: '|x|' } },
      messages: {
        bifurcation: { description: 'a fold or Hopf point', payload: { type: 'fold or hopf', p: 'parameter' } },
      },
    }),
    init: () => {
      const { J } = analyse(F, z0, n)
      let t = tangentAt(J, undefined)
      if (Math.sign(t[n] ?? 1) !== (options.direction ?? 1)) t = Float64Array.from(t, (v) => -v)
      return { z: z0, tangent: t, points: [point(z0)], bifurcations: [], failed: false }
    },
    step: (s) => {
      const pred = Float64Array.from(s.z, (v, i) => v + ds * (s.tangent[i] ?? 0))
      const tangent = s.tangent
      const ext = (w: Value): Value =>
        concat(
          [
            reshape(F(sliceAxis(w, 0, 0, n), reshape(sliceAxis(w, 0, n, n + 1), [])), [n]),
            reshape(sumDot(sub(w, fromNumbers(pred, [n + 1])), tangent), [1]),
          ],
          0,
        )
      const r = fsolve(ext, fromNumbers(pred, [n + 1]), { ftol: tol, maxIterations: 20 })
      if (!r.converged) return { ...s, failed: true }
      const z = Float64Array.from(numbersOf(r.x))
      const { J, ev } = analyse(F, z, n)
      let t = tangentAt(J, tangent)
      if (t.reduce((acc, v, i) => acc + v * (tangent[i] ?? 0), 0) < 0) t = Float64Array.from(t, (v) => -v)
      const pt = point(z)
      const bifurcations = [...s.bifurcations]
      const interp = (a: number, b: number): number => (a === b ? 0.5 : a / (a - b))
      if (Math.sign(t[n] ?? 0) !== Math.sign(tangent[n] ?? 0) && (t[n] ?? 0) !== 0) {
        const w = interp(tangent[n] ?? 0, t[n] ?? 0)
        const zf = Float64Array.from(z, (v, i) => (s.z[i] ?? 0) + w * (v - (s.z[i] ?? 0)))
        bifurcations.push({
          type: 'fold',
          x: fromNumbers(zf.slice(0, n), xv.shape),
          p: zf[n] ?? 0,
          index: s.points.length,
        })
      }
      const h0 = hopfTest(analyse(F, s.z, n).ev)
      const h1 = hopfTest(ev)
      if (Number.isFinite(h0) && Number.isFinite(h1) && Math.sign(h0) !== Math.sign(h1) && h1 !== 0) {
        const w = interp(h0, h1)
        const zh = Float64Array.from(z, (v, i) => (s.z[i] ?? 0) + w * (v - (s.z[i] ?? 0)))
        bifurcations.push({
          type: 'hopf',
          x: fromNumbers(zh.slice(0, n), xv.shape),
          p: zh[n] ?? 0,
          index: s.points.length,
        })
      }
      s.points.push(pt)
      return { z, tangent: t, points: s.points, bifurcations, failed: false }
    },
    done: (s) => {
      if (s.failed) return 'failed'
      const p = s.z[n] ?? 0
      if (options.pRange !== undefined && (p < options.pRange[0] || p > options.pRange[1])) return 'stopped'
      return undefined
    },
    observe: (s) => ({
      metrics: { p: s.z[n] ?? 0, norm: rms(s.z.slice(0, n)) * Math.sqrt(n) },
      messages: s.bifurcations.slice(-1).map((b) => ({ kind: 'bifurcation', payload: { type: b.type, p: b.p } })),
    }),
  }
  return { algorithm, steps }
}

function sumDot(v: Value, w: Float64Array): Value {
  return reshape(matmulValue(reshape(v, [1, -1]), fromNumbers(w, [w.length, 1])), [])
}

/**
 * Continue a branch of equilibria in a parameter and report its folds (saddle-node points) and Hopf points (see
 * {@link continuationAlgorithm}).
 *
 * @param F - The field F(x, p).
 * @param x0 - An equilibrium at p0.
 * @param p0 - The starting parameter.
 * @param options - Step, number of steps, parameter range, direction, tolerance and tracing.
 * @returns The branch with stability, and the bifurcations found.
 * @concept bifurcation-continuation
 * @example
 * ```ts
 * import { continueEquilibria } from 'whitebox-engine/dynamics'
 * import { add, mul } from 'whitebox-engine/tensor'
 * // x' = p + x^2: a fold at p = 0, x = 0.
 * const r = continueEquilibria((x, p) => add(mul(x, x), p), [-1], -1, { ds: 0.1, steps: 30 })
 * Math.abs(r.bifurcations[0]?.p ?? 1) < 1e-2 // => true
 * ```
 */
export function continueEquilibria(
  F: (x: Value, p: Value) => Value,
  x0: TensorLike,
  p0: number,
  options: ContinuationOptions = {},
): { readonly branch: readonly BranchPoint[]; readonly bifurcations: readonly Bifurcation[]; readonly status: string } {
  const { algorithm, steps } = continuationAlgorithm(F, x0, p0, options)
  const run = runAlgorithm(algorithm, { maxSteps: steps, ...options })
  return { branch: run.state.points, bifurcations: run.state.bifurcations, status: run.status }
}

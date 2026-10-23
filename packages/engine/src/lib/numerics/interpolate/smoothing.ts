import { WhiteboxError } from '../../base/index.ts'
import {
  concreteOrThrow,
  fromStorage,
  matmul,
  primalOf,
  promoteFloat,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { inv, solve } from '../linalg/index.ts'
import type { PPoly } from './piecewise.ts'
import { type BSpline, cubicSpline, makeLsqSpline } from './splines.ts'

/** Result of {@link makeSmoothingSpline}. */
export interface SmoothingSpline {
  readonly spline: PPoly
  /** The penalty used (chosen by GCV when not given). */
  readonly lam: number
  /** The generalised cross-validation score at lam. */
  readonly gcv: number
  /** The effective degrees of freedom, trace of the smoother matrix. */
  readonly degreesOfFreedom: number
}

/**
 * The cubic smoothing spline (scipy.interpolate.make_smoothing_spline): the natural cubic spline g minimising
 * sum w_i (y_i - g(x_i))^2 + lam * integral g''^2, computed by Reinsch's method: the fitted values solve
 * (W + lam Q R^{-1} Q^T) g = W y, then g is interpolated with natural ends. Without lam, it is chosen by generalised
 * cross-validation (golden-section search in log lam). Linear in y, so differentiable in the data.
 *
 * @param x - Points `[n]`, strictly increasing (n >= 5).
 * @param y - Values `[n]`.
 * @param options - Weights and the penalty.
 * @returns The spline, the penalty, its GCV score and degrees of freedom.
 * @concept smoothing-splines
 * @example
 * ```ts
 * import { makeSmoothingSpline } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * // A straight line is never penalised.
 * Math.abs(item(makeSmoothingSpline([0, 1, 2, 3, 4, 5], [1, 3, 5, 7, 9, 11], { lam: 10 }).spline.evaluate(2.5)) - 6) < 1e-10 // => true
 * ```
 */
export function makeSmoothingSpline(
  x: TensorLike,
  y: TensorLike,
  options: { readonly w?: readonly number[]; readonly lam?: number } = {},
): SmoothingSpline {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, 'makeSmoothingSpline'))
  const [yv] = promoteFloat(y) as [Value]
  const n = xs.length
  if (n < 5) throw new WhiteboxError('value', 'makeSmoothingSpline: at least five points are needed')
  if (yv.shape.length !== 1 || yv.shape[0] !== n) throw new WhiteboxError('shape', 'makeSmoothingSpline: y must be [n]')
  const w = options.w ?? new Array<number>(n).fill(1)
  const h = xs.slice(1).map((v, i) => v - (xs[i] ?? 0))
  if (h.some((v) => !(v > 0))) throw new WhiteboxError('value', 'makeSmoothingSpline: x must be strictly increasing')
  // Q (n x n-2) and R (n-2 x n-2) of Green and Silverman; K = Q R^{-1} Q^T is the roughness penalty matrix.
  const p = n - 2
  const q = new Float64Array(n * p)
  const r = new Float64Array(p * p)
  for (let j = 0; j < p; j++) {
    const h0 = h[j] ?? 1
    const h1 = h[j + 1] ?? 1
    q[j * p + j] = 1 / h0
    q[(j + 1) * p + j] = -1 / h0 - 1 / h1
    q[(j + 2) * p + j] = 1 / h1
    r[j * p + j] = (h0 + h1) / 3
    if (j + 1 < p) {
      r[j * p + j + 1] = h1 / 6
      r[(j + 1) * p + j] = h1 / 6
    }
  }
  const qt = fromStorage(q, [n, p], 'float64')
  const k = toFlat(matmul(matmul(qt, inv(fromStorage(r, [p, p], 'float64'))), transpose(q, n, p)))
  const smoother = (lam: number): Float64Array => {
    // S = (W + lam K)^{-1} W.
    const lhs = new Float64Array(n * n)
    const wm = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) lhs[i * n + j] = lam * (k[i * n + j] ?? 0)
      lhs[i * n + i] = (lhs[i * n + i] ?? 0) + (w[i] ?? 1)
      wm[i * n + i] = w[i] ?? 1
    }
    return Float64Array.from(toFlat(solve(fromStorage(lhs, [n, n], 'float64'), fromStorage(wm, [n, n], 'float64'))))
  }
  const yp = toFlat(primalOf(yv))
  const score = (lam: number): { gcv: number; df: number; s: Float64Array } => {
    const s = smoother(lam)
    let rss = 0
    let trace = 0
    for (let i = 0; i < n; i++) {
      let fit = 0
      for (let j = 0; j < n; j++) fit += (s[i * n + j] ?? 0) * (yp[j] ?? 0)
      rss += (w[i] ?? 1) * ((yp[i] ?? 0) - fit) ** 2
      trace += s[i * n + i] ?? 0
    }
    return { gcv: rss / n / (1 - trace / n) ** 2, df: trace, s }
  }
  let lam = options.lam
  if (lam === undefined) {
    // Golden-section search on log10 lam over a range scaled to the data spacing.
    const scale = Math.log10(((xs[n - 1] ?? 0) - (xs[0] ?? 0)) ** 3 / n)
    let a = scale - 12
    let b = scale + 6
    const phi = (Math.sqrt(5) - 1) / 2
    let c = b - phi * (b - a)
    let d = a + phi * (b - a)
    let fc = score(10 ** c).gcv
    let fd = score(10 ** d).gcv
    for (let it = 0; it < 80 && b - a > 1e-6; it++) {
      if (fc < fd) {
        b = d
        d = c
        fd = fc
        c = b - phi * (b - a)
        fc = score(10 ** c).gcv
      } else {
        a = c
        c = d
        fc = fd
        d = a + phi * (b - a)
        fd = score(10 ** d).gcv
      }
    }
    lam = 10 ** ((a + b) / 2)
  }
  const { gcv, df, s } = score(lam)
  const fitted = matmul(fromStorage(s, [n, n], 'float64'), yv)
  return { spline: cubicSpline(xs, fitted, 'natural'), lam, gcv, degreesOfFreedom: df }
}

function transpose(q: Float64Array, n: number, p: number): Tensor {
  const t = new Float64Array(p * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) t[j * n + i] = q[i * p + j] ?? 0
  return fromStorage(t, [p, n], 'float64')
}

/** Result of {@link univariateSpline}. */
export interface UnivariateSplineResult {
  readonly spline: BSpline
  /** The weighted residual sum of squares. */
  readonly residual: number
  /** The interior knots chosen. */
  readonly interiorKnots: readonly number[]
}

/**
 * A smoothing spline with automatic knots in the manner of FITPACK's curfit (scipy.interpolate.UnivariateSpline): start
 * from the least-squares polynomial (no interior knots) and insert a knot in the span with the largest residual until
 * the weighted residual sum of squares, sum (w_i (y_i - s(x_i)))^2, is at most the smoothing factor s. FITPACK then
 * also tunes a smoothing parameter on the final knots; this keeps the least-squares fit on them. s = 0 interpolates
 * (knots at the data, not-a-knot).
 *
 * @param x - Points `[m]`, increasing.
 * @param y - Values `[m]`.
 * @param options - Degree k (default 3), smoothing factor s (default m, as FITPACK with unit weights), weights.
 * @returns The spline, its residual and knots.
 * @example
 * ```ts
 * import { univariateSpline } from 'whitebox-engine/numerics/interpolate'
 * const xs = Array.from({ length: 30 }, (_, i) => i / 5)
 * univariateSpline(xs, xs.map(Math.sin), { s: 1e-4 }).residual <= 1e-4 // => true
 * ```
 */
export function univariateSpline(
  x: TensorLike,
  y: TensorLike,
  options: { readonly k?: number; readonly s?: number; readonly w?: readonly number[] } = {},
): UnivariateSplineResult {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, 'univariateSpline'))
  const [yv] = promoteFloat(y) as [Value]
  const m = xs.length
  const k = options.k ?? 3
  const s = options.s ?? m
  const w = options.w ?? new Array<number>(m).fill(1)
  if (m <= k) throw new WhiteboxError('value', `univariateSpline: degree ${k} needs more than ${k} points`)
  const lo = xs[0] ?? 0
  const hi = xs[m - 1] ?? 0
  const yp = toFlat(primalOf(yv))
  const fit = (interior: readonly number[]): { spline: BSpline; residuals: number[] } => {
    const t = [...new Array<number>(k + 1).fill(lo), ...interior, ...new Array<number>(k + 1).fill(hi)]
    const spline = makeLsqSpline(xs, yv, t, k, w)
    const values = toFlat(primalOf(spline.evaluate(xs)))
    return { spline, residuals: values.map((v, i) => ((w[i] ?? 1) * ((yp[i] ?? 0) - v)) ** 2) }
  }
  let interior: number[] = []
  // The most interior knots: one fewer than needed to interpolate.
  const maxInterior = m - k - 1
  let current = fit(interior)
  let total = current.residuals.reduce((a, b) => a + b, 0)
  while (total > s && interior.length < maxInterior) {
    // The span between consecutive knots with the largest residual sum, split at its middle data point.
    const all = [lo, ...interior, hi]
    let best = -1
    let bestSum = -1
    let split = Number.NaN
    for (let j = 0; j + 1 < all.length; j++) {
      const inside = xs.map((v, i) => ({ v, i })).filter(({ v }) => v > (all[j] ?? 0) && v < (all[j + 1] ?? 0))
      if (inside.length === 0) continue
      const sumHere = inside.reduce((acc, { i }) => acc + (current.residuals[i] ?? 0), 0)
      if (sumHere > bestSum) {
        bestSum = sumHere
        best = j
        split = inside[Math.floor(inside.length / 2)]?.v ?? Number.NaN
      }
    }
    if (best < 0 || Number.isNaN(split)) break
    interior = [...interior, split].sort((a, b) => a - b)
    current = fit(interior)
    total = current.residuals.reduce((a, b) => a + b, 0)
  }
  if (total > s && interior.length >= maxInterior && s === 0) {
    // Interpolation: the not-a-knot knots at the data.
    interior = xs.slice(Math.floor((k + 1) / 2), m - Math.floor((k + 1) / 2))
    current = fit(interior)
    total = current.residuals.reduce((a, b) => a + b, 0)
  }
  return { spline: current.spline, residual: total, interiorKnots: interior }
}

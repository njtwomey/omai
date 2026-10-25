/**
 * Penalised regression bases for the smooth terms of generalised additive models, following mgcv and patsy: B-splines,
 * natural splines and P-splines, cubic and cyclic cubic regression splines, thin-plate, Duchon and Gaussian-process
 * smooths, random effects, factor-smooth interactions and tensor products. Each `SmoothBasis` carries its model matrix,
 * its penalty matrices and a way to evaluate it at new data.
 */

import { WhiteboxError } from '../../base/index.ts'
import { bspline } from '../../numerics/interpolate/index.ts'
import { dgesv, dsyev } from '../../numerics/linalg/index.ts'
import { gamma as gammaFn } from '../../numerics/special/index.ts'
import { asTensor, item, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { matmul, transpose } from '../_linear.ts'
import { mat, type Rows, rowsOf, sample, sortedCopy } from '../_util.ts'
import { quantile } from '../quantiles.ts'
import { constraintNullSpace, derivativePenalty, differencePenalty, sumToZeroConstraint } from './penalties.ts'

/**
 * A penalised regression basis for one smooth term (mgcv's smooth.construct): the model matrix at the data, the
 * penalties (beta^T S_j beta) and the dimension of the unpenalised null space, with a function evaluating the basis at
 * new points.
 */
export interface SmoothBasis {
  readonly kind: string
  /** The n x k model matrix at the data. */
  readonly X: Tensor
  /** The k x k penalty matrices. */
  readonly penalties: readonly Tensor[]
  readonly nullSpaceDimension: number
  /** The knots (or centres) the basis was built on. */
  readonly knots: Tensor
  /** The model matrix at new points (a sample `[M]` for one covariate, `[M, D]` for several). */
  readonly evaluate: (x: TensorLike) => Tensor
}

/** A knot placement rule. */
export type KnotRule = 'quantile' | 'uniform'

/**
 * Place k knots over the range of a covariate, both boundaries included: at equally spaced quantiles of the distinct
 * values (patsy and mgcv's default) or equally spaced.
 *
 * @param x - The covariate.
 * @param k - The number of knots (at least 2).
 * @param options - rule (default `'quantile'`).
 * @returns The knots, ascending.
 * @example
 * ```ts
 * import { placeKnots } from 'whitebox-engine/stats/smoothing'
 * placeKnots([0, 1, 2, 3, 4], 3) // => [0, 2, 4]
 * placeKnots([0, 1, 10], 3, { rule: 'uniform' }) // => [0, 5, 10]
 * ```
 */
export function placeKnots(x: TensorLike, k: number, options: { readonly rule?: KnotRule } = {}): Tensor {
  if (!(Number.isInteger(k) && k >= 2)) throw new WhiteboxError('value', 'placeKnots: k must be an integer >= 2')
  const s = sortedCopy(sample('placeKnots', x))
  const lo = s[0] ?? 0
  const hi = s[s.length - 1] ?? 0
  if (options.rule === 'uniform')
    return asTensor(Array.from({ length: k }, (_, i) => lo + ((hi - lo) * i) / (k - 1))) as Tensor
  const u = Float64Array.from(new Set(s))
  return asTensor(
    Array.from({ length: k }, (_, i) => (i === 0 ? lo : i === k - 1 ? hi : quantile(u, i / (k - 1)))),
  ) as Tensor
}

function designOf(
  knots: readonly number[],
  degree: number,
  x: ArrayLike<number>,
): { data: Float64Array; cols: number } {
  const n = knots.length - degree - 1
  const spline = bspline(
    knots,
    Array.from({ length: n }, () => 0),
    degree,
  )
  return { data: Float64Array.from(toFlat(spline.designMatrix(Array.from(x)) as Tensor)), cols: n }
}

function dropFirstColumn(d: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows * (cols - 1))
  for (let i = 0; i < rows; i++) for (let j = 1; j < cols; j++) out[i * (cols - 1) + (j - 1)] = d[i * cols + j] ?? 0
  return out
}

/** Solve A X = B for X (A n x n, B n x m), column by column. */
function solveColumns(A: Float64Array, B: Float64Array, n: number, m: number): Float64Array {
  const X = new Float64Array(n * m)
  for (let c = 0; c < m; c++) {
    const col = dgesv(
      A,
      Float64Array.from({ length: n }, (_, i) => B[i * m + c] ?? 0),
      n,
    )
    for (let i = 0; i < n; i++) X[i * m + c] = col[i] ?? 0
  }
  return X
}

function congruence(S: Tensor, Z: Float64Array, k: number, m: number): Tensor {
  // Z^T S Z for Z k x m.
  const s = toFlat(S)
  return mat(matmul(matmul(transpose(Z, k, m), s, m, k, k), Z, m, k, m), m, m)
}

/** Options of {@link bs}. */
export interface BsOptions {
  /** Degrees of freedom (columns); sets the number of interior knots at quantiles. */
  readonly df?: number
  /** Interior knots (instead of df). */
  readonly knots?: readonly number[]
  readonly degree?: number
  /** Keep the first basis function (default false, as patsy and R). */
  readonly intercept?: boolean
  readonly boundaryKnots?: readonly [number, number]
}

/**
 * A B-spline basis (patsy's bs, R's splines::bs): interior knots at quantiles, boundary knots repeated degree + 1
 * times, the first column dropped unless `intercept`; with the integrated squared second-derivative penalty.
 *
 * @param x - The covariate.
 * @param options - df or knots, degree (default 3), intercept and boundaryKnots.
 * @returns The basis.
 * @example
 * ```ts
 * import { bs } from 'whitebox-engine/stats/smoothing'
 * bs([0, 1, 2, 3, 4, 5], { df: 4 }).X.shape // => [6, 4]
 * ```
 */
export function bs(x: TensorLike, options: BsOptions = {}): SmoothBasis {
  const a = sample('bs', x, 2)
  const degree = options.degree ?? 3
  const order = degree + 1
  const intercept = options.intercept ?? false
  let inner: number[]
  if (options.knots !== undefined) inner = [...options.knots]
  else {
    if (options.df === undefined) throw new WhiteboxError('value', 'bs: give df or knots')
    const nInner = options.df - order + (intercept ? 0 : 1)
    if (nInner < 0) throw new WhiteboxError('value', 'bs: df is too small for the degree')
    inner = Array.from({ length: nInner }, (_, i) => quantile(a, (i + 1) / (nInner + 1)))
  }
  const [lo, hi] = options.boundaryKnots ?? [
    a.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY),
    a.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY),
  ]
  const knots = [...new Array<number>(order).fill(lo), ...inner, ...new Array<number>(order).fill(hi)].sort(
    (p, q) => p - q,
  )
  const evaluate = (q: TensorLike): Tensor => {
    const v = sample('bs', q, 0)
    const { data, cols } = designOf(knots, degree, v)
    return intercept ? mat(data, v.length, cols) : mat(dropFirstColumn(data, v.length, cols), v.length, cols - 1)
  }
  const X = evaluate(a)
  let S = derivativePenalty(knots, degree, Math.min(2, degree))
  const k = knots.length - degree - 1
  if (!intercept) {
    const Z = new Float64Array(k * (k - 1))
    for (let j = 1; j < k; j++) Z[j * (k - 1) + (j - 1)] = 1
    S = congruence(S, Z, k, k - 1)
  }
  return {
    kind: 'bs',
    X,
    penalties: [S],
    nullSpaceDimension: intercept ? 2 : 1,
    knots: asTensor(knots) as Tensor,
    evaluate,
  }
}

/**
 * A natural cubic spline basis (R's splines::ns): cubic B-splines constrained to zero second derivative at the
 * boundary knots (so the fit is linear beyond them), via the Householder QR of the constraints; linear extrapolation
 * outside the boundary knots.
 *
 * @param x - The covariate.
 * @param options - df or knots, intercept (default false) and boundaryKnots.
 * @returns The basis (with the integrated squared second-derivative penalty).
 * @example
 * ```ts
 * import { ns } from 'whitebox-engine/stats/smoothing'
 * ns([0, 1, 2, 3, 4, 5], { df: 3 }).X.shape // => [6, 3]
 * ```
 */
export function ns(x: TensorLike, options: Omit<BsOptions, 'degree'> = {}): SmoothBasis {
  const a = sample('ns', x, 2)
  const intercept = options.intercept ?? false
  let inner: number[]
  if (options.knots !== undefined) inner = [...options.knots]
  else {
    if (options.df === undefined) throw new WhiteboxError('value', 'ns: give df or knots')
    const nInner = options.df - 1 - (intercept ? 1 : 0)
    if (nInner < 0) throw new WhiteboxError('value', 'ns: df is too small')
    inner = Array.from({ length: nInner }, (_, i) => quantile(a, (i + 1) / (nInner + 1)))
  }
  const [lo, hi] = options.boundaryKnots ?? [
    a.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY),
    a.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY),
  ]
  const knots = [lo, lo, lo, lo, ...inner, hi, hi, hi, hi]
  const k = knots.length - 4
  const spline = bspline(
    knots,
    Array.from({ length: k }, () => 0),
    3,
  )
  const identity = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))
  const second = Float64Array.from(toFlat(bspline(knots, identity, 3).derivative(2).evaluate([lo, hi]) as Tensor))
  const first = toFlat(bspline(knots, identity, 3).derivative(1).evaluate([lo, hi]) as Tensor)
  const k0 = intercept ? 0 : 1
  const kk = k - k0
  const C = new Float64Array(2 * kk)
  for (let r = 0; r < 2; r++) for (let j = 0; j < kk; j++) C[r * kk + j] = second[r * k + j + k0] ?? 0
  const Z = Float64Array.from(toFlat(constraintNullSpace(C, kk)))
  const m = kk - 2
  const evaluate = (q: TensorLike): Tensor => {
    const v = sample('ns', q, 0)
    const inside = Float64Array.from(v, (t) => Math.min(hi, Math.max(lo, t)))
    const B = Float64Array.from(toFlat(spline.designMatrix(Array.from(inside)) as Tensor))
    // Linear extrapolation: the basis at the nearest boundary plus the slope times the distance.
    const out = new Float64Array(v.length * m)
    for (let i = 0; i < v.length; i++) {
      const t = v[i] ?? 0
      const side = t < lo ? 0 : t > hi ? 1 : -1
      const dist = side === 0 ? t - lo : side === 1 ? t - hi : 0
      for (let c = 0; c < m; c++) {
        let s = 0
        for (let j = 0; j < kk; j++) {
          const b = (B[i * k + j + k0] ?? 0) + (side >= 0 ? dist * (first[side * k + j + k0] ?? 0) : 0)
          s += b * (Z[j * m + c] ?? 0)
        }
        out[i * m + c] = s
      }
    }
    return mat(out, v.length, m)
  }
  const Sfull = derivativePenalty(knots, 3, 2)
  const Zfull = new Float64Array(k * m)
  for (let j = 0; j < kk; j++) for (let c = 0; c < m; c++) Zfull[(j + k0) * m + c] = Z[j * m + c] ?? 0
  return {
    kind: 'ns',
    X: evaluate(a),
    penalties: [congruence(Sfull, Zfull, k, m)],
    nullSpaceDimension: intercept ? 2 : 1,
    knots: asTensor(knots) as Tensor,
    evaluate,
  }
}

/** Options of {@link ps}. */
export interface PsOptions {
  /** The basis dimension (default 10). */
  readonly k?: number
  readonly degree?: number
  /** The difference penalty order (default 2). */
  readonly order?: number
}

/**
 * A P-spline basis (Eilers and Marx 1996; mgcv's "ps"): B-splines on equally spaced knots extending degree spans
 * beyond the data, with a difference penalty on adjacent coefficients.
 *
 * @param x - The covariate.
 * @param options - k (default 10), degree (default 3) and order (default 2).
 * @returns The basis.
 * @example
 * ```ts
 * import { ps } from 'whitebox-engine/stats/smoothing'
 * ps([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], { k: 6 }).nullSpaceDimension // => 2
 * ```
 */
export function ps(x: TensorLike, options: PsOptions = {}): SmoothBasis {
  const a = sample('ps', x, 2)
  const degree = options.degree ?? 3
  const k = options.k ?? 10
  const order = options.order ?? 2
  const nseg = k - degree
  if (nseg < 1) throw new WhiteboxError('value', 'ps: k must exceed the degree')
  const lo = a.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY)
  const hi = a.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY)
  const dx = (hi - lo) / nseg
  const knots = Array.from({ length: k + degree + 1 }, (_, i) => lo + (i - degree) * dx)
  const evaluate = (q: TensorLike): Tensor => {
    const v = sample('ps', q, 0)
    const { data, cols } = designOf(knots, degree, v)
    return mat(data, v.length, cols)
  }
  return {
    kind: 'ps',
    X: evaluate(a),
    penalties: [differencePenalty(k, order)],
    nullSpaceDimension: order,
    knots: asTensor(knots) as Tensor,
    evaluate,
  }
}

// patsy's _get_natural_f: the map from function values at the knots to second derivatives of the natural spline.
function naturalF(knots: readonly number[]): { F: Float64Array; B: Float64Array; D: Float64Array } {
  const k = knots.length
  const h = knots.slice(1).map((v, i) => v - (knots[i] ?? 0))
  const m = k - 2
  const B = new Float64Array(m * m)
  const D = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    B[i * m + i] = ((h[i] ?? 0) + (h[i + 1] ?? 0)) / 3
    if (i + 1 < m) {
      B[i * m + i + 1] = (h[i + 1] ?? 0) / 6
      B[(i + 1) * m + i] = (h[i + 1] ?? 0) / 6
    }
    D[i * k + i] = 1 / (h[i] ?? 1)
    D[i * k + i + 2] = 1 / (h[i + 1] ?? 1)
    D[i * k + i + 1] = -(D[i * k + i] ?? 0) - (D[i * k + i + 2] ?? 0)
  }
  const F = new Float64Array(k * k)
  if (m > 0) {
    const sol = solveColumns(B, D, m, k)
    for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) F[(i + 1) * k + j] = sol[i * k + j] ?? 0
  }
  return { F, B, D }
}

// patsy's _get_cyclic_f, with its B and D.
function cyclicF(knots: readonly number[]): { F: Float64Array; B: Float64Array; D: Float64Array } {
  const h = knots.slice(1).map((v, i) => v - (knots[i] ?? 0))
  const n = knots.length - 1
  const B = new Float64Array(n * n)
  const D = new Float64Array(n * n)
  B[0] = ((h[n - 1] ?? 0) + (h[0] ?? 0)) / 3
  B[n - 1] = (h[n - 1] ?? 0) / 6
  B[(n - 1) * n] = (h[n - 1] ?? 0) / 6
  D[0] = -1 / (h[0] ?? 1) - 1 / (h[n - 1] ?? 1)
  D[n - 1] = 1 / (h[n - 1] ?? 1)
  D[(n - 1) * n] = 1 / (h[n - 1] ?? 1)
  for (let i = 1; i < n; i++) {
    B[i * n + i] = ((h[i - 1] ?? 0) + (h[i] ?? 0)) / 3
    B[i * n + i - 1] = (h[i - 1] ?? 0) / 6
    B[(i - 1) * n + i] = (h[i - 1] ?? 0) / 6
    D[i * n + i] = -1 / (h[i - 1] ?? 1) - 1 / (h[i] ?? 1)
    D[i * n + i - 1] = 1 / (h[i - 1] ?? 1)
    D[(i - 1) * n + i] = 1 / (h[i - 1] ?? 1)
  }
  return { F: solveColumns(B, D, n, n), B, D }
}

function crsDesign(knots: readonly number[], x: Float64Array, cyclic: boolean): Float64Array {
  const K = knots.length
  const lo = knots[0] ?? 0
  const hi = knots[K - 1] ?? 0
  const n = cyclic ? K - 1 : K
  const F = (cyclic ? cyclicF(knots) : naturalF(knots)).F
  const out = new Float64Array(x.length * n)
  for (let r = 0; r < x.length; r++) {
    let v = x[r] ?? 0
    if (cyclic) {
      if (v > hi) v = lo + ((v - hi) % (hi - lo))
      else if (v < lo) v = hi - ((lo - v) % (hi - lo))
    }
    // patsy's _find_knots_lower_bounds: searchsorted(knots, x) - 1, clipped to [0, K - 2].
    let j = 0
    while (j < K && (knots[j] ?? 0) < v) j++
    j = Math.min(K - 2, Math.max(0, j - 1))
    const h = (knots[j + 1] ?? 0) - (knots[j] ?? 0)
    const right = (knots[j + 1] ?? 0) - v
    const left = v - (knots[j] ?? 0)
    const ajm = right / h
    const ajp = left / h
    const cjm = (v > hi ? 0 : (right * right * right) / (6 * h)) - (h * right) / 6
    const cjp = (v < lo ? 0 : (left * left * left) / (6 * h)) - (h * left) / 6
    const j1 = cyclic && j + 1 === n ? 0 : j + 1
    out[r * n + j] = (out[r * n + j] ?? 0) + ajm
    out[r * n + j1] = (out[r * n + j1] ?? 0) + ajp
    for (let c = 0; c < n; c++)
      out[r * n + c] = (out[r * n + c] ?? 0) + cjm * (F[j * n + c] ?? 0) + cjp * (F[j1 * n + c] ?? 0)
  }
  return out
}

/** Options of {@link cr} and {@link cc}. */
export interface CubicRegressionOptions {
  /** The basis dimension (default 10). */
  readonly k?: number
  /** The knots (default k at quantiles of the distinct values, both ends included). */
  readonly knots?: readonly number[]
}

/**
 * A cubic regression spline basis (mgcv's "cr", patsy's cr): coefficients are the spline's values at the knots, the
 * spline is natural, and the penalty D^T B^-1 D is the integrated squared second derivative.
 *
 * @param x - The covariate.
 * @param options - k (default 10) or knots.
 * @returns The basis (null space: linear functions).
 * @example
 * ```ts
 * import { cr } from 'whitebox-engine/stats/smoothing'
 * cr([0, 1, 2, 3, 4], { k: 3 }).X // => [[1, 0, 0], [0.40625, 0.6875, -0.09375], [0, 1, 0], [-0.09375, 0.6875, 0.40625], [0, 0, 1]]
 * ```
 */
export function cr(x: TensorLike, options: CubicRegressionOptions = {}): SmoothBasis {
  const a = sample('cr', x, 2)
  const knots =
    options.knots === undefined ? toFlat(placeKnots(a, options.k ?? 10)) : [...options.knots].sort((p, q) => p - q)
  const K = knots.length
  const { B, D } = naturalF(knots)
  const m = K - 2
  // S = D^T B^-1 D.
  const BinvD = solveColumns(B, D, m, K)
  const S = matmul(transpose(D, m, K), BinvD, K, m, K)
  const evaluate = (q: TensorLike): Tensor => {
    const v = sample('cr', q, 0)
    return mat(crsDesign(knots, v, false), v.length, K)
  }
  return {
    kind: 'cr',
    X: evaluate(a),
    penalties: [mat(S, K, K)],
    nullSpaceDimension: 2,
    knots: asTensor(knots) as Tensor,
    evaluate,
  }
}

/**
 * A cyclic cubic regression spline basis (mgcv's "cc", patsy's cc): as {@link cr} but periodic on [first knot, last
 * knot], with k - 1 coefficients and the cyclic second-derivative penalty.
 *
 * @param x - The covariate.
 * @param options - k (default 10) or knots.
 * @returns The basis (null space: constants).
 * @example
 * ```ts
 * import { cc } from 'whitebox-engine/stats/smoothing'
 * cc([0, 1, 2, 3, 4, 5, 6], { k: 5 }).X.shape // => [7, 4]
 * ```
 */
export function cc(x: TensorLike, options: CubicRegressionOptions = {}): SmoothBasis {
  const a = sample('cc', x, 2)
  const knots =
    options.knots === undefined ? toFlat(placeKnots(a, options.k ?? 10)) : [...options.knots].sort((p, q) => p - q)
  const n = knots.length - 1
  const { B, D } = cyclicF(knots)
  const S = matmul(transpose(D, n, n), solveColumns(B, D, n, n), n, n, n)
  const evaluate = (q: TensorLike): Tensor => {
    const v = sample('cc', q, 0)
    return mat(crsDesign(knots, v, true), v.length, n)
  }
  return {
    kind: 'cc',
    X: evaluate(a),
    penalties: [mat(S, n, n)],
    nullSpaceDimension: 1,
    knots: asTensor(knots) as Tensor,
    evaluate,
  }
}

/** The radial kernel of a thin-plate (or Duchon) spline of order m in d dimensions (Wood 2003, eq. 2). */
function thinPlateKernel(m: number, d: number, s = 0): (r: number) => number {
  const p = 2 * m + 2 * s - d
  // The log term appears when the power is an even integer (for thin plates: when d is even).
  const even = Number.isInteger(p) && p % 2 === 0
  let c: number
  if (s === 0) {
    if (d % 2 === 0)
      c = (-1) ** (m + 1 + d / 2) / (2 ** (2 * m - 1) * Math.PI ** (d / 2) * factorial(m - 1) * factorial(m - d / 2))
    else c = item(gammaFn(d / 2 - m)) / (2 ** (2 * m) * Math.PI ** (d / 2) * factorial(m - 1))
  } else c = 1
  return (r) => (r <= 0 ? 0 : even ? c * r ** p * Math.log(r) : c * r ** p)
}

function factorial(n: number): number {
  let r = 1
  for (let i = 2; i <= n; i++) r *= i
  return r
}

/** The monomials of total degree below m in d variables (the thin-plate null space). */
function monomials(m: number, d: number): number[][] {
  const out: number[][] = []
  const rec = (prefix: number[], left: number): void => {
    if (prefix.length === d) {
      out.push(prefix)
      return
    }
    for (let e = 0; e <= left; e++) rec([...prefix, e], left - e)
  }
  rec([], m - 1)
  return out.sort((a, b) => a.reduce((s, v) => s + v, 0) - b.reduce((s, v) => s + v, 0))
}

/** Options of the radial-basis smooths ({@link tp}, {@link ts}, {@link duchon}, {@link gp}). */
export interface RadialOptions {
  /** The basis dimension (default 10). */
  readonly k?: number
  /** The penalty order (default the smallest with 2m > d: m = floor((d + 1) / 2) + 1). */
  readonly m?: number
  /** Centres (default the data points). */
  readonly centres?: TensorLike
}

function radialBasis(
  kind: string,
  x: TensorLike,
  kernel: (r: number) => number,
  nullPowers: number[][],
  k: number,
  centresIn: TensorLike | undefined,
): SmoothBasis {
  const data = rowsOf(kind, x)
  const centres = centresIn === undefined ? data : rowsOf(kind, centresIn)
  const n = centres.rows
  const d = data.cols
  const M = nullPowers.length
  if (!(k > M && k <= n))
    throw new WhiteboxError(
      'value',
      `${kind}: k must exceed the null-space dimension ${M} and not the number of centres`,
    )
  const dist = (a: Rows, i: number, b: Rows, j: number): number => {
    let s = 0
    for (let c = 0; c < d; c++) s += ((a.data[i * d + c] ?? 0) - (b.data[j * d + c] ?? 0)) ** 2
    return Math.sqrt(s)
  }
  const E = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) E[i * n + j] = kernel(dist(centres, i, centres, j))
  const poly = (m: Rows, i: number): number[] =>
    nullPowers.map((pw) => pw.reduce((s, e, c) => s * (m.data[i * d + c] ?? 0) ** e, 1))
  const T = new Float64Array(n * M)
  for (let i = 0; i < n; i++)
    poly(centres, i).forEach((v, j) => {
      T[i * M + j] = v
    })
  // Truncated eigen-decomposition: the k - M... k eigenvalues of largest magnitude.
  const { w, v } = dsyev(E, n)
  const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => Math.abs(w[q] ?? 0) - Math.abs(w[p] ?? 0))
  const keep = order.slice(0, k)
  const Uk = new Float64Array(n * k)
  const Dk = new Float64Array(k)
  keep.forEach((c, j) => {
    Dk[j] = w[c] ?? 0
    for (let i = 0; i < n; i++) Uk[i * k + j] = v[i * n + c] ?? 0
  })
  // Absorb T^T Uk delta = 0: Z is the null space of (Uk^T T)^T.
  const UtT = matmul(transpose(Uk, n, k), T, k, n, M)
  const Z = Float64Array.from(toFlat(constraintNullSpace(transpose(UtT, k, M), k)))
  const q = k - M
  const UkZ = matmul(Uk, Z, n, k, q)
  const DZ = Float64Array.from({ length: k * q }, (_, idx) => (Dk[Math.floor(idx / q)] ?? 0) * (Z[idx] ?? 0))
  const S = new Float64Array(k * k)
  const ZDZ = matmul(transpose(Z, k, q), DZ, q, k, q)
  // A conditionally positive definite kernel gives a positive semidefinite constrained penalty; kernels whose constant
  // carries the opposite sign (Duchon's) are flipped.
  const ev = dsyev(ZDZ, q).w
  const flip = Math.abs(ev[0] ?? 0) > Math.abs(ev[q - 1] ?? 0) ? -1 : 1
  for (let i = 0; i < q; i++) for (let j = 0; j < q; j++) S[i * k + j] = flip * (ZDZ[i * q + j] ?? 0)
  const evaluate = (query: TensorLike): Tensor => {
    const pts = rowsOf(kind, query)
    if (pts.cols !== d) throw new WhiteboxError('shape', `${kind}: points must have dimension ${d}`)
    const out = new Float64Array(pts.rows * k)
    for (let r = 0; r < pts.rows; r++) {
      const e = Float64Array.from({ length: n }, (_, i) => kernel(dist(pts, r, centres, i)))
      for (let c = 0; c < q; c++) {
        let s = 0
        for (let i = 0; i < n; i++) s += (e[i] ?? 0) * (UkZ[i * q + c] ?? 0)
        out[r * k + c] = s
      }
      poly(pts, r).forEach((val, j) => {
        out[r * k + q + j] = val
      })
    }
    return mat(out, pts.rows, k)
  }
  return {
    kind,
    X: evaluate(data.cols === 1 ? Array.from(data.data) : x),
    penalties: [mat(S, k, k)],
    nullSpaceDimension: M,
    knots: mat(centres.data, centres.rows, d),
    evaluate,
  }
}

function defaultOrder(d: number): number {
  return Math.floor((d + 1) / 2) + 1
}

/**
 * A thin-plate regression spline basis (Wood 2003; mgcv's "tp"): the k leading eigenvectors of the thin-plate kernel
 * matrix at the centres, constrained orthogonal to the polynomial null space of degree below m, which is appended;
 * the penalty is the thin-plate wiggliness on the eigen basis. Isotropic in any number of covariates.
 *
 * @param x - The covariate(s) `[N]` or `[N, D]`.
 * @param options - k (default 10), m and centres.
 * @returns The basis.
 * @reference S. N. Wood, Thin plate regression splines, Journal of the Royal Statistical Society B 65 (2003).
 * @example
 * ```ts
 * import { tp } from 'whitebox-engine/stats/smoothing'
 * tp([0, 1, 2, 3, 4, 5, 6, 7], { k: 5 }).nullSpaceDimension // => 2
 * ```
 */
export function tp(x: TensorLike, options: RadialOptions = {}): SmoothBasis {
  const d = rowsOf('tp', x).cols
  const m = options.m ?? defaultOrder(d)
  if (!(2 * m > d)) throw new WhiteboxError('value', 'tp: the order m must satisfy 2m > d')
  return radialBasis('tp', x, thinPlateKernel(m, d), monomials(m, d), options.k ?? 10, options.centres)
}

/**
 * A thin-plate regression spline with shrinkage (mgcv's "ts"): {@link tp} with the null space also penalised (by a
 * small multiple of the smallest positive eigenvalue), so the term can be shrunk out of a model entirely.
 *
 * @param x - The covariate(s).
 * @param options - k, m, centres and epsilon (default 0.1).
 * @returns The basis, with a full-rank penalty.
 * @example
 * ```ts
 * import { ts } from 'whitebox-engine/stats/smoothing'
 * ts([0, 1, 2, 3, 4, 5, 6, 7], { k: 5 }).nullSpaceDimension // => 0
 * ```
 */
export function ts(x: TensorLike, options: RadialOptions & { readonly epsilon?: number } = {}): SmoothBasis {
  const base = tp(x, options)
  const k = base.X.shape[1] ?? 0
  const S = toFlat(base.penalties[0] ?? mat([], 0, 0))
  const M = base.nullSpaceDimension
  let smallest = Number.POSITIVE_INFINITY
  for (const e of dsyev(S, k).w) if (e > 1e-9) smallest = Math.min(smallest, e)
  const out = Float64Array.from(S)
  for (let i = k - M; i < k; i++) out[i * k + i] = (out[i * k + i] ?? 0) + (options.epsilon ?? 0.1) * smallest
  return { ...base, kind: 'ts', penalties: [mat(out, k, k)], nullSpaceDimension: 0 }
}

/**
 * A Duchon spline basis (Duchon 1977; mgcv's "ds"): the thin-plate construction with the kernel r^(2m + 2s - d)
 * (times log r when that power is even), penalising a fractional derivative; s = 0 and 2m > d is the thin plate.
 *
 * @param x - The covariate(s).
 * @param options - k, m (default 2), s (default (d - 1) / 2, mgcv's default for m = 2) and centres.
 * @returns The basis.
 * @example
 * ```ts
 * import { duchon } from 'whitebox-engine/stats/smoothing'
 * duchon([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [0.5, 1.5]], { k: 6 }).nullSpaceDimension // => 3
 * ```
 */
export function duchon(x: TensorLike, options: RadialOptions & { readonly s?: number } = {}): SmoothBasis {
  const d = rowsOf('duchon', x).cols
  const m = options.m ?? 2
  const s = options.s ?? (d - 1) / 2
  if (!(m + s > d / 2 && s < d / 2)) throw new WhiteboxError('value', 'duchon: needs m + s > d / 2 and s < d / 2')
  return radialBasis('ds', x, thinPlateKernel(m, d, s === 0 ? 0 : s), monomials(m, d), options.k ?? 10, options.centres)
}

/**
 * A Gaussian-process smooth basis (Kammann and Wand 2003; mgcv's "gp"): the thin-plate eigen construction with a
 * stationary covariance kernel (Matern 3/2 by default, range rho, default the largest distance between centres) and a
 * linear null space.
 *
 * @param x - The covariate(s).
 * @param options - k, centres, rho and kernel (`'matern32'`, `'matern52'`, `'exponential'`, `'squared-exponential'`).
 * @returns The basis.
 * @example
 * ```ts
 * import { gp } from 'whitebox-engine/stats/smoothing'
 * gp([0, 1, 2, 3, 4, 5, 6, 7], { k: 5 }).nullSpaceDimension // => 2
 * ```
 */
export function gp(
  x: TensorLike,
  options: RadialOptions & {
    readonly rho?: number
    readonly kernel?: 'matern32' | 'matern52' | 'exponential' | 'squared-exponential'
  } = {},
): SmoothBasis {
  const data = rowsOf('gp', x)
  const centres = options.centres === undefined ? data : rowsOf('gp', options.centres)
  const d = data.cols
  let rho = options.rho ?? 0
  if (options.rho === undefined)
    for (let i = 0; i < centres.rows; i++)
      for (let j = i + 1; j < centres.rows; j++) {
        let s = 0
        for (let c = 0; c < d; c++) s += ((centres.data[i * d + c] ?? 0) - (centres.data[j * d + c] ?? 0)) ** 2
        rho = Math.max(rho, Math.sqrt(s))
      }
  if (!(rho > 0)) throw new WhiteboxError('value', 'gp: the range rho must be positive')
  const kind = options.kernel ?? 'matern32'
  const kernel = (r: number): number => {
    const u = r / rho
    if (kind === 'exponential') return Math.exp(-u)
    if (kind === 'squared-exponential') return Math.exp(-u * u)
    if (kind === 'matern52') return (1 + Math.sqrt(5) * u + (5 * u * u) / 3) * Math.exp(-Math.sqrt(5) * u)
    return (1 + Math.sqrt(3) * u) * Math.exp(-Math.sqrt(3) * u)
  }
  return radialBasis('gp', x, kernel, monomials(2, d), options.k ?? 10, options.centres)
}

/**
 * A random-effect basis (mgcv's "re"): one indicator column per factor level, penalised by the identity, so the
 * smoothing parameter is the inverse of the random-effect variance.
 *
 * @param factor - The level of each observation (any values).
 * @returns The basis (no null space).
 * @example
 * ```ts
 * import { re } from 'whitebox-engine/stats/smoothing'
 * re(['a', 'b', 'a']).X // => [[1, 0], [0, 1], [1, 0]]
 * ```
 */
export function re(factor: readonly (string | number)[]): SmoothBasis {
  const levels = Array.from(new Set(factor)).sort((a, b) => String(a).localeCompare(String(b), 'en', { numeric: true }))
  const L = levels.length
  const at = new Map(levels.map((v, i) => [v, i]))
  const indicator = (f: readonly (string | number)[]): Tensor => {
    const out = new Float64Array(f.length * L)
    f.forEach((v, i) => {
      const j = at.get(v)
      if (j !== undefined) out[i * L + j] = 1
    })
    return mat(out, f.length, L)
  }
  const I = new Float64Array(L * L)
  for (let i = 0; i < L; i++) I[i * L + i] = 1
  return {
    kind: 're',
    X: indicator(factor),
    penalties: [mat(I, L, L)],
    nullSpaceDimension: 0,
    knots: asTensor([]) as Tensor,
    evaluate: (q) => indicator(toFlat(asTensor(q))),
  }
}

/**
 * A factor-smooth interaction basis (mgcv's "fs"): one copy of a smooth per factor level, all sharing the smooth's
 * penalty and a second penalty on their null spaces, so the level curves are shrunk towards each other like random
 * effects.
 *
 * @param x - The covariate.
 * @param factor - The level of each observation.
 * @param options - k of the underlying cubic regression spline (default 5).
 * @returns The basis (penalties: the smooth's and the null spaces'); `evaluate` takes rows [x, level index] with the
 *   levels in sorted order.
 * @example
 * ```ts
 * import { fs } from 'whitebox-engine/stats/smoothing'
 * fs([0, 1, 2, 0, 1, 2], ['a', 'a', 'a', 'b', 'b', 'b'], { k: 3 }).X.shape // => [6, 6]
 * ```
 */
export function fs(
  x: TensorLike,
  factor: readonly (string | number)[],
  options: { readonly k?: number } = {},
): SmoothBasis {
  const a = sample('fs', x, 2)
  if (factor.length !== a.length) throw new WhiteboxError('shape', 'fs: the factor must match the covariate')
  const base = cr(a, { k: options.k ?? 5 })
  const k = base.X.shape[1] ?? 0
  const levels = Array.from(new Set(factor)).sort((p, q) => String(p).localeCompare(String(q), 'en', { numeric: true }))
  const L = levels.length
  const at = new Map(levels.map((v, i) => [v, i]))
  const S = toFlat(base.penalties[0] ?? mat([], 0, 0))
  const { w, v } = dsyev(S, k)
  let top = 0
  for (const e of w) top = Math.max(top, Math.abs(e))
  const P0 = new Float64Array(k * k)
  for (let c = 0; c < k; c++) {
    if (Math.abs(w[c] ?? 0) > 1e-9 * top) continue
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) P0[i * k + j] = (P0[i * k + j] ?? 0) + (v[i * k + c] ?? 0) * (v[j * k + c] ?? 0)
  }
  const K = k * L
  const blocks = (B: ArrayLike<number>): Tensor => {
    const out = new Float64Array(K * K)
    for (let l = 0; l < L; l++)
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) out[(l * k + i) * K + l * k + j] = B[i * k + j] ?? 0
    return mat(out, K, K)
  }
  const design = (xs: Float64Array, f: readonly (string | number)[]): Tensor => {
    const Bx = toFlat(base.evaluate(xs))
    const out = new Float64Array(xs.length * K)
    for (let i = 0; i < xs.length; i++) {
      const l = at.get(f[i] ?? '')
      if (l === undefined) continue
      for (let j = 0; j < k; j++) out[i * K + l * k + j] = Bx[i * k + j] ?? 0
    }
    return mat(out, xs.length, K)
  }
  return {
    kind: 'fs',
    X: design(a, factor),
    penalties: [blocks(S), blocks(P0)],
    nullSpaceDimension: 0,
    knots: base.knots,
    evaluate: (q) => {
      // Rows of [x, level index] (levels in sorted order).
      const pts = rowsOf('fs', q)
      if (pts.cols !== 2) throw new WhiteboxError('shape', 'fs: evaluate points as rows [x, level index]')
      const xs = Float64Array.from({ length: pts.rows }, (_, i) => pts.data[2 * i] ?? 0)
      const f = Array.from({ length: pts.rows }, (_, i) => levels[pts.data[2 * i + 1] ?? -1] ?? '')
      return design(xs, f)
    },
  }
}

/** The row-wise tensor (Kronecker) product of model matrices (mgcv's tensor.prod.model.matrix column order). */
function rowTensor(
  mats: readonly { data: ArrayLike<number>; cols: number }[],
  n: number,
): { data: Float64Array; cols: number } {
  let acc = { data: Float64Array.from({ length: n }, () => 1), cols: 1 }
  for (const m of mats) {
    const cols = acc.cols * m.cols
    const out = new Float64Array(n * cols)
    for (let i = 0; i < n; i++)
      for (let a = 0; a < acc.cols; a++)
        for (let b = 0; b < m.cols; b++)
          out[i * cols + a * m.cols + b] = (acc.data[i * acc.cols + a] ?? 0) * (m.data[i * m.cols + b] ?? 0)
    acc = { data: out, cols }
  }
  return acc
}

function kronIdentity(S: ArrayLike<number>, k: number, before: number, after: number): Float64Array {
  // I_before (x) S (x) I_after.
  const K = before * k * after
  const out = new Float64Array(K * K)
  for (let p = 0; p < before; p++)
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        for (let q = 0; q < after; q++) out[((p * k + i) * after + q) * K + (p * k + j) * after + q] = S[i * k + j] ?? 0
  return out
}

/**
 * A tensor-product smooth (mgcv's "te", patsy's te): the row-wise Kronecker product of marginal bases, with one penalty
 * per margin, S_j (x) I on the others; invariant to the covariates' relative scaling.
 *
 * @param margins - The marginal bases (built on the same observations).
 * @returns The basis; `evaluate` takes `[M, D]` points with one column per margin.
 * @example
 * ```ts
 * import { cr, te } from 'whitebox-engine/stats/smoothing'
 * const x = [0, 1, 2, 3, 4, 5]
 * te([cr(x, { k: 3 }), cr(x.map((v) => v * v), { k: 4 })]).X.shape // => [6, 12]
 * ```
 */
export function te(margins: readonly SmoothBasis[]): SmoothBasis {
  return tensorProduct('te', margins, false)
}

/**
 * A tensor-product interaction (mgcv's "ti"): {@link te} on marginal bases with their sum-to-zero constraints absorbed
 * first, so the term excludes the main effects and can be added to a model that has them.
 *
 * @param margins - The marginal bases.
 * @returns The basis.
 * @example
 * ```ts
 * import { cr, ti } from 'whitebox-engine/stats/smoothing'
 * const x = [0, 1, 2, 3, 4, 5]
 * ti([cr(x, { k: 3 }), cr(x.map((v) => v * v), { k: 4 })]).X.shape // => [6, 6]
 * ```
 */
export function ti(margins: readonly SmoothBasis[]): SmoothBasis {
  return tensorProduct('ti', margins, true)
}

function tensorProduct(kind: string, margins: readonly SmoothBasis[], constrained: boolean): SmoothBasis {
  if (margins.length < 1) throw new WhiteboxError('value', `${kind}: needs at least one margin`)
  const n = margins[0]?.X.shape[0] ?? 0
  for (const m of margins)
    if ((m.X.shape[0] ?? 0) !== n) throw new WhiteboxError('shape', `${kind}: margins differ in observations`)
  const parts = margins.map((m) => {
    const k = m.X.shape[1] ?? 0
    if (!constrained) return { m, k, Z: undefined as Float64Array | undefined, kk: k }
    const Z = Float64Array.from(toFlat(sumToZeroConstraint(m.X)))
    return { m, k, Z, kk: k - 1 }
  })
  const marginal = (p: (typeof parts)[number], X: Tensor, rows: number): { data: Float64Array; cols: number } => {
    const d = Float64Array.from(toFlat(X))
    if (p.Z === undefined) return { data: d, cols: p.k }
    return { data: matmul(d, p.Z, rows, p.k, p.kk), cols: p.kk }
  }
  const X = rowTensor(
    parts.map((p) => marginal(p, p.m.X, n)),
    n,
  )
  const sizes = parts.map((p) => p.kk)
  const penalties = parts.map((p, j) => {
    const S0 = Float64Array.from(toFlat(p.m.penalties[0] ?? mat([], 0, 0)))
    const S = p.Z === undefined ? S0 : Float64Array.from(toFlat(congruence(mat(S0, p.k, p.k), p.Z, p.k, p.kk)))
    const before = sizes.slice(0, j).reduce((s, v) => s * v, 1)
    const after = sizes.slice(j + 1).reduce((s, v) => s * v, 1)
    return mat(kronIdentity(S, p.kk, before, after), X.cols, X.cols)
  })
  const nullDim = parts.reduce(
    (s, p) => s * (p.Z === undefined ? p.m.nullSpaceDimension : Math.max(0, p.m.nullSpaceDimension - 1)),
    1,
  )
  return {
    kind,
    X: mat(X.data, n, X.cols),
    penalties,
    nullSpaceDimension: nullDim,
    knots: asTensor([]) as Tensor,
    evaluate: (q) => {
      const pts = rowsOf(kind, q)
      if (pts.cols !== margins.length) throw new WhiteboxError('shape', `${kind}: points need one column per margin`)
      const ms = parts.map((p, j) =>
        marginal(
          p,
          p.m.evaluate(Float64Array.from({ length: pts.rows }, (_, i) => pts.data[i * pts.cols + j] ?? 0)),
          pts.rows,
        ),
      )
      const out = rowTensor(ms, pts.rows)
      return mat(out.data, pts.rows, out.cols)
    },
  }
}

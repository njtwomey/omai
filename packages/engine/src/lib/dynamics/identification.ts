/**
 * System identification: estimating a dynamical model from measured inputs and outputs. Polynomial models
 * (least-squares ARX, ARMAX by the prediction-error method), subspace identification of state-space models (N4SID),
 * dynamic mode decomposition and its extended form approximating the Koopman operator, SINDy, which finds sparse
 * governing equations from a library of candidate terms, and recursive least squares, which updates the estimate one
 * sample at a time (with a forgetting factor to follow a changing system).
 */

import { type Json, MdmlError } from '../base/index.ts'
import { ddot, dgemv, eig, lstsq, pinv, solve, svd } from '../numerics/linalg/index.ts'
import { leastSquares } from '../optim/index.ts'
import { type Key, key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  eye,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { fromNumbers, numbersOf } from './ode/_vec.ts'

const series = (x: TensorLike): Float64Array => Float64Array.from(numbersOf(primalOf(asTensor(x))))

/** An identified ARX or ARMAX model A(q) y = B(q) q^-nk u + C(q) e. */
export interface PolynomialModel {
  /** a_1 .. a_na (A = 1 + a_1 q^-1 + ...). */
  readonly a: number[]
  /** b_1 .. b_nb (B = b_1 + b_2 q^-1 + ..., delayed by nk). */
  readonly b: number[]
  /** c_1 .. c_nc (ARMAX only). */
  readonly c: number[]
  readonly nk: number
  /** The one-step prediction errors. */
  readonly residuals: number[]
  /** The residual variance. */
  readonly noiseVariance: number
}

/**
 * Least-squares ARX identification (MATLAB's arx): fit y_t + a_1 y_{t-1} + ... + a_na y_{t-na} =
 * b_1 u_{t-nk} + ... + b_nb u_{t-nk-nb+1} + e_t by ordinary least squares on the regression from the first
 * index where every lag exists.
 *
 * @param y - Output series.
 * @param u - Input series.
 * @param orders - na, nb and the delay nk (default 1).
 * @returns The polynomial coefficients and residuals.
 * @throws MdmlError with code `value` for mismatched lengths or too few samples.
 * @concept system-identification
 * @example
 * ```ts
 * import { arx } from 'mdml-engine/dynamics'
 * // y_t = 0.5 y_{t-1} + 2 u_{t-1}: a_1 = -0.5, b_1 = 2.
 * const u = [1, 0, 0, 1, 1, 0, 1, 0]
 * const y = [0]
 * for (let t = 1; t < u.length; t++) y.push(0.5 * (y[t - 1] ?? 0) + 2 * (u[t - 1] ?? 0))
 * arx(y, u, { na: 1, nb: 1 }).b.map((v) => Math.round(v * 1e10) / 1e10) // => [2]
 * ```
 */
export function arx(
  y: TensorLike,
  u: TensorLike,
  orders: { readonly na: number; readonly nb: number; readonly nk?: number },
): PolynomialModel {
  const ys = series(y)
  const us = series(u)
  if (ys.length !== us.length) throw new MdmlError('value', 'arx: y and u need the same length')
  const { na, nb } = orders
  const nk = orders.nk ?? 1
  const start = Math.max(na, nk + nb - 1)
  const rows = ys.length - start
  if (rows <= na + nb) throw new MdmlError('value', 'arx: too few samples for these orders')
  const X = new Float64Array(rows * (na + nb))
  const Y = new Float64Array(rows)
  for (let r = 0; r < rows; r++) {
    const t = start + r
    for (let i = 0; i < na; i++) X[r * (na + nb) + i] = -(ys[t - 1 - i] ?? 0)
    for (let j = 0; j < nb; j++) X[r * (na + nb) + na + j] = us[t - nk - j] ?? 0
    Y[r] = ys[t] ?? 0
  }
  const theta = numbersOf(primalOf(lstsq(fromNumbers(X, [rows, na + nb]), fromNumbers(Y, [rows]))))
  const resid = Array.from({ length: rows }, (_, r) => {
    let s = Y[r] ?? 0
    for (let k = 0; k < na + nb; k++) s -= (X[r * (na + nb) + k] ?? 0) * (theta[k] ?? 0)
    return s
  })
  return {
    a: Array.from(theta.slice(0, na)),
    b: Array.from(theta.slice(na)),
    c: [],
    nk,
    residuals: resid,
    noiseVariance: resid.reduce((acc, e) => acc + e * e, 0) / Math.max(1, rows - na - nb),
  }
}

/**
 * The prediction-error method for any model whose one-step prediction errors are written with operations: minimise
 * the sum of squared errors by Levenberg-Marquardt (Jacobians by automatic differentiation).
 *
 * @param errors - The prediction errors as a function of the parameters `[k]`.
 * @param theta0 - The starting parameters.
 * @param options - Cap on residual evaluations (default 100 per parameter, as scipy).
 * @returns The parameters and the final cost.
 * @concept prediction-error-method
 * @example
 * ```ts
 * import { predictionError } from 'mdml-engine/dynamics'
 * import { sub, mul } from 'mdml-engine/tensor'
 * const r = predictionError((th) => sub(mul(th, [1, 2, 3]), [2, 4, 6]), [0])
 * Math.round((r.theta[0] ?? 0) * 1e8) / 1e8 // => 2
 * ```
 */
export function predictionError(
  errors: (theta: Value) => Value,
  theta0: TensorLike,
  options: { readonly maxEvaluations?: number } = {},
): { readonly theta: number[]; readonly cost: number; readonly converged: boolean } {
  const r = leastSquares((th) => errors(th as Value), asTensor(theta0), options)
  return { theta: toFlat(r.x as Tensor), cost: r.cost, converged: r.converged }
}

/**
 * ARMAX identification by the prediction-error method: the errors e_t = y_t + sum a_i y_{t-i} - sum b_j u_{t-nk-j+1}
 * - sum c_k e_{t-k} are minimised in squares, starting from the ARX estimate with C = 1.
 *
 * @param y - Output series.
 * @param u - Input series.
 * @param orders - na, nb, nc and the delay nk (default 1).
 * @returns The polynomial coefficients and residuals.
 * @concept prediction-error-method
 * @example
 * ```ts
 * import { armax } from 'mdml-engine/dynamics'
 * const u = Array.from({ length: 40 }, (_, t) => Math.sin(0.7 * t) + (t % 3 === 0 ? 1 : 0))
 * const y = [0]
 * for (let t = 1; t < 40; t++) y.push(0.6 * (y[t - 1] ?? 0) + (u[t - 1] ?? 0))
 * Math.round(armax(y, u, { na: 1, nb: 1, nc: 1 }).a[0] * 1e6) / 1e6 // => -0.6
 * ```
 */
export function armax(
  y: TensorLike,
  u: TensorLike,
  orders: { readonly na: number; readonly nb: number; readonly nc: number; readonly nk?: number },
): PolynomialModel {
  const start = arx(y, u, orders)
  const ys = series(y)
  const us = series(u)
  const { na, nb, nc } = orders
  const nk = orders.nk ?? 1
  const t0 = Math.max(na, nk + nb - 1, nc)
  const n = ys.length
  const errors = (theta: Value): Value => {
    const a = sliceAxis(theta, 0, 0, na)
    const b = sliceAxis(theta, 0, na, na + nb)
    const c = sliceAxis(theta, 0, na + nb, na + nb + nc)
    const es: Value[] = []
    for (let t = t0; t < n; t++) {
      const yl = fromNumbers(
        Array.from({ length: na }, (_, i) => ys[t - 1 - i] ?? 0),
        [na],
      )
      const ul = fromNumbers(
        Array.from({ length: nb }, (_, j) => us[t - nk - j] ?? 0),
        [nb],
      )
      let e: Value = sub(
        add(fromNumbers([ys[t] ?? 0], [1]), reshape(matmul(reshape(a, [1, -1]), reshape(yl, [-1, 1])), [1])),
        reshape(matmul(reshape(b, [1, -1]), reshape(ul, [-1, 1])), [1]),
      )
      if (nc > 0) {
        const past = Array.from({ length: nc }, (_, k) => es[es.length - 1 - k] ?? fromNumbers([0], [1]))
        e = sub(e, reshape(matmul(reshape(c, [1, -1]), reshape(concat(past, 0), [-1, 1])), [1]))
      }
      es.push(e)
    }
    return concat(es, 0)
  }
  const theta0 = fromNumbers([...start.a, ...start.b, ...new Array<number>(nc).fill(0)], [na + nb + nc])
  const r = predictionError(errors, theta0)
  const resid = toFlat(errors(fromNumbers(r.theta, [r.theta.length])))
  return {
    a: r.theta.slice(0, na),
    b: r.theta.slice(na, na + nb),
    c: r.theta.slice(na + nb),
    nk,
    residuals: resid,
    noiseVariance: resid.reduce((acc, e) => acc + e * e, 0) / Math.max(1, resid.length - na - nb - nc),
  }
}

/** An identified state-space model x' = A x + B u, y = C x + D u. */
export interface StateSpaceEstimate {
  readonly A: Tensor
  readonly B: Tensor
  readonly C: Tensor
  readonly D: Tensor
  /** The singular values of the projection (their gap suggests the order). */
  readonly singularValues: number[]
}

function blockHankel(data: Float64Array, dim: number, rows: number, cols: number, offset: number): Float64Array {
  const out = new Float64Array(rows * dim * cols)
  for (let i = 0; i < rows; i++) {
    for (let d = 0; d < dim; d++) {
      for (let j = 0; j < cols; j++) out[(i * dim + d) * cols + j] = data[(offset + i + j) * dim + d] ?? 0
    }
  }
  return out
}

/**
 * Subspace identification in the manner of N4SID (Van Overschee and De Moor 1994): block Hankel matrices of past and
 * future inputs and outputs, the oblique projection of the future outputs onto the past along the future inputs, an
 * SVD for the order and the extended observability matrix, state sequences from it, and A, B, C, D by least squares
 * on the state equations.
 *
 * @param y - Outputs `[T, p]` (or `[T]`).
 * @param u - Inputs `[T, m]` (or `[T]`).
 * @param options - Model order and the number of block rows (default max(order + 1, 2 order)).
 * @returns A, B, C, D and the singular values.
 * @throws MdmlError with code `value` for too few samples.
 * @concept subspace-identification
 * @example
 * ```ts
 * import { n4sid } from 'mdml-engine/dynamics'
 * const u = Array.from({ length: 120 }, (_, t) => Math.sin(0.3 * t) + Math.cos(1.1 * t))
 * let x = 0
 * const y = u.map((v) => { const out = x; x = 0.8 * x + v; return out })
 * Math.round(n4sid(y, u, { order: 1 }).A.data[0] * 1e6) / 1e6 // => 0.8
 * ```
 */
export function n4sid(
  y: TensorLike,
  u: TensorLike,
  options: { readonly order: number; readonly blockRows?: number },
): StateSpaceEstimate {
  const Y = asTensor(y)
  const U = asTensor(u)
  const T = Y.shape[0] ?? 0
  const p = Y.shape.length === 1 ? 1 : (Y.shape[1] ?? 1)
  const m = U.shape.length === 1 ? 1 : (U.shape[1] ?? 1)
  const n = options.order
  const i = options.blockRows ?? Math.max(n + 1, 2 * n)
  const j = T - 2 * i + 1
  if (j < 2 * i * (m + p)) throw new MdmlError('value', 'n4sid: too few samples for these block rows')
  const yd = series(Y)
  const ud = series(U)
  const Up = fromNumbers(blockHankel(ud, m, i, j, 0), [i * m, j])
  const Uf = fromNumbers(blockHankel(ud, m, i, j, i), [i * m, j])
  const Yp = fromNumbers(blockHankel(yd, p, i, j, 0), [i * p, j])
  const Yf = fromNumbers(blockHankel(yd, p, i, j, i), [i * p, j])
  const Wp = concat([Up, Yp], 0)
  // Projection onto the orthogonal complement of the rows of Uf.
  const proj = (M: Value): Value =>
    sub(M, matmul(matmul(M, transpose(Uf)), matmul(pinv(matmul(Uf, transpose(Uf))), Uf)))
  const YfP = proj(Yf)
  const WpP = proj(Wp)
  const Oi = matmul(matmul(YfP, pinv(WpP)), Wp)
  const { u: Us, s } = svd(Oi)
  const sv = toFlat(primalOf(s))
  const U1 = sliceAxis(Us, 1, 0, n)
  const S1 = sv.slice(0, n)
  const Gamma = mul(U1, fromNumbers(S1.map(Math.sqrt), [1, n]))
  // States X_i = Gamma^+ O_i and X_{i+1} = Gamma_-^+ O_{i+1}, with O_{i+1} from the shifted Hankel split.
  const Xi = matmul(pinv(Gamma), Oi)
  const Up1 = fromNumbers(blockHankel(ud, m, i + 1, j, 0), [(i + 1) * m, j])
  const Uf1 = fromNumbers(blockHankel(ud, m, i - 1, j, i + 1), [(i - 1) * m, j])
  const Yp1 = fromNumbers(blockHankel(yd, p, i + 1, j, 0), [(i + 1) * p, j])
  const Yf1 = fromNumbers(blockHankel(yd, p, i - 1, j, i + 1), [(i - 1) * p, j])
  const Wp1 = concat([Up1, Yp1], 0)
  const proj1 = (M: Value): Value =>
    sub(M, matmul(matmul(M, transpose(Uf1)), matmul(pinv(matmul(Uf1, transpose(Uf1))), Uf1)))
  const Oi1 = matmul(matmul(proj1(Yf1), pinv(proj1(Wp1))), Wp1)
  const GammaMinus = sliceAxis(Gamma, 0, 0, (i - 1) * p)
  const Xi1 = matmul(pinv(GammaMinus), Oi1)
  // [X_{i+1}; Y_i] = [A B; C D] [X_i; U_i]
  const Yi = sliceAxis(Yf, 0, 0, p)
  const Ui = sliceAxis(Uf, 0, 0, m)
  const lhs = concat([Xi1, Yi], 0)
  const rhs = concat([Xi, Ui], 0)
  const theta = transpose(lstsq(transpose(rhs), transpose(lhs)))
  const th = primalOf(theta)
  const block = (r0: number, r1: number, c0: number, c1: number): Tensor =>
    primalOf(sliceAxis(sliceAxis(th, 0, r0, r1), 1, c0, c1))
  return {
    A: block(0, n, 0, n),
    B: block(0, n, n, n + m),
    C: block(n, n + p, 0, n),
    D: block(n, n + p, n, n + m),
    singularValues: sv,
  }
}

/** The result of {@link dmd}. */
export interface DmdResult {
  /** DMD eigenvalues (discrete-time). */
  readonly eigenvalues: { readonly re: number[]; readonly im: number[] }
  /** Modes `[n, r]` as real and imaginary parts. */
  readonly modes: { readonly re: Tensor; readonly im: Tensor }
  /** The reduced operator U^T X2 V S^-1 `[r, r]`. */
  readonly operator: Tensor
  /** Mode amplitudes from the first snapshot. */
  readonly amplitudes: { readonly re: number[]; readonly im: number[] }
}

/**
 * Exact dynamic mode decomposition (Schmid 2010; Tu et al. 2014): with snapshots X = [x_0 .. x_{m-1}] and
 * X' = [x_1 .. x_m], the rank-r operator A_tilde = U* X' V S^-1 from the truncated SVD of X, its eigenvalues, the exact
 * modes X' V S^-1 W and their amplitudes from x_0.
 *
 * @param snapshots - States as columns `[n, m + 1]`.
 * @param options - The rank (default the numerical rank).
 * @returns Eigenvalues, modes, the reduced operator and amplitudes.
 * @concept dynamic-mode-decomposition
 * @example
 * ```ts
 * import { dmd } from 'mdml-engine/dynamics'
 * // Snapshots of x_{k+1} = diag(0.9, 0.5) x_k: the eigenvalues are 0.9 and 0.5.
 * const cols = [[1, 1]]
 * for (let k = 0; k < 5; k++) cols.push([0.9 * cols[k][0], 0.5 * cols[k][1]])
 * dmd([cols.map((c) => c[0]), cols.map((c) => c[1])]).eigenvalues.re.map((v) => Math.round(v * 1e10) / 1e10).sort() // => [0.5, 0.9]
 * ```
 */
export function dmd(snapshots: TensorLike, options: { readonly rank?: number } = {}): DmdResult {
  const X = asTensor(snapshots)
  const m = (X.shape[1] ?? 1) - 1
  const X1 = sliceAxis(X, 1, 0, m)
  const X2 = sliceAxis(X, 1, 1, m + 1)
  const { u, s, vh } = svd(X1)
  const sv = toFlat(primalOf(s))
  const tol = Math.max(X1.shape[0] ?? 0, m) * Number.EPSILON * (sv[0] ?? 0)
  const r = options.rank ?? sv.filter((v) => v > tol).length
  const Ur = sliceAxis(u, 1, 0, r)
  const Vr = transpose(sliceAxis(vh, 0, 0, r))
  const SrInv = fromNumbers(
    sv.slice(0, r).map((v) => 1 / v),
    [1, r],
  )
  const XVS = mul(matmul(X2, Vr), SrInv)
  const At = primalOf(matmul(transpose(Ur), XVS))
  const e = eig(At)
  const re = toFlat(e.values.re)
  const im = toFlat(e.values.im)
  // Modes Phi = X' V S^-1 W (complex W).
  const Phr = primalOf(matmul(XVS, e.vectors.re))
  const Phi = primalOf(matmul(XVS, e.vectors.im))
  // Amplitudes: least squares Phi b = x_0 through the real embedding.
  const n = X.shape[0] ?? 0
  const x0 = toFlat(sliceAxis(X, 1, 0, 1))
  const big = concat([concat([Phr, mul(Phi, -1)], 1), concat([Phi, Phr], 1)], 0)
  const b = toFlat(lstsq(big, fromNumbers([...x0, ...new Array<number>(n).fill(0)], [2 * n])))
  return {
    eigenvalues: { re, im },
    modes: { re: Phr, im: Phi },
    operator: At,
    amplitudes: { re: b.slice(0, r), im: b.slice(r) },
  }
}

/** A dictionary of observables for extended DMD: one state `[n]` to the features `[k]`. */
export type Dictionary = (x: Value) => Value

/**
 * The monomials of a state up to a degree (1, x_i, x_i x_j, ...) in scikit-learn's (and pysindy's) order: by degree,
 * then lexicographically by exponent pattern.
 *
 * @param n - State dimension.
 * @param degree - The highest degree.
 * @param options - Whether to include the constant (default true).
 * @returns The dictionary and the feature names.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { polynomialLibrary } from 'mdml-engine/dynamics'
 * polynomialLibrary(2, 2).names // => ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
 * ```
 */
export function polynomialLibrary(
  n: number,
  degree: number,
  options: { readonly bias?: boolean } = {},
): { readonly features: Dictionary; readonly names: string[]; readonly size: number } {
  const terms: number[][] = []
  if (options.bias !== false) terms.push([])
  const build = (deg: number, start: number, acc: number[]): void => {
    if (acc.length === deg) {
      terms.push([...acc])
      return
    }
    for (let i = start; i < n; i++) build(deg, i, [...acc, i])
  }
  for (let d = 1; d <= degree; d++) build(d, 0, [])
  const names = terms.map((t) => {
    if (t.length === 0) return '1'
    const counts = new Map<number, number>()
    for (const i of t) counts.set(i, (counts.get(i) ?? 0) + 1)
    return [...counts.entries()].map(([i, c]) => (c === 1 ? `x${i}` : `x${i}^${c}`)).join(' ')
  })
  return {
    names,
    size: terms.length,
    features: (x) => {
      const xs = Array.from({ length: n }, (_, i) => sliceAxis(reshape(x, [-1]), 0, i, i + 1))
      return concat(
        terms.map((t) => t.reduce<Value>((acc, i) => mul(acc, xs[i] as Value), fromNumbers([1], [1]))),
        0,
      )
    },
  }
}

/**
 * Extended DMD (Williams, Kevrekidis and Rowley 2015): a finite-dimensional approximation of the Koopman operator on
 * a dictionary of observables, K = G^+ A with G = Psi_X^T Psi_X and A = Psi_X^T Psi_Y, and its eigenvalues.
 *
 * @param x - States `[m, n]`.
 * @param y - Their images `[m, n]` (the next states).
 * @param dictionary - The observables.
 * @returns The Koopman matrix `[k, k]` (acting on feature row vectors: psi(y) ~ psi(x) K) and its eigenvalues.
 * @concept koopman-operator
 * @example
 * ```ts
 * import { edmd, polynomialLibrary } from 'mdml-engine/dynamics'
 * // x' = 0.5 x on monomials 1, x, x^2: Koopman eigenvalues 1, 0.5 and 0.25.
 * const xs = [[0.3], [1.1], [-0.7], [2]]
 * const r = edmd(xs, xs.map(([v]) => [0.5 * v]), polynomialLibrary(1, 2).features)
 * r.eigenvalues.re.map((v) => Math.round(v * 1e10) / 1e10).sort() // => [0.25, 0.5, 1]
 * ```
 */
export function edmd(
  x: TensorLike,
  y: TensorLike,
  dictionary: Dictionary,
): { readonly koopman: Tensor; readonly eigenvalues: { readonly re: number[]; readonly im: number[] } } {
  const X = asTensor(x)
  const Y = asTensor(y)
  const m = X.shape[0] ?? 0
  const rows = (M: Value): Value =>
    stack(Array.from({ length: m }, (_, i) => dictionary(reshape(sliceAxis(M, 0, i, i + 1), [-1]))))
  const PX = rows(X)
  const PY = rows(Y)
  const G = matmul(transpose(PX), PX)
  const A = matmul(transpose(PX), PY)
  const K = primalOf(matmul(pinv(G), A))
  const e = eig(K)
  return { koopman: K, eigenvalues: { re: toFlat(e.values.re), im: toFlat(e.values.im) } }
}

/** Options of {@link sindy}. */
export interface SindyOptions {
  /** Coefficient threshold (default 0.1, as pysindy's STLSQ). */
  readonly threshold?: number
  /** Ridge weight (default 0.05). */
  readonly alpha?: number
  readonly maxIterations?: number
  /** Refit the selected terms by least squares (default true). */
  readonly unbias?: boolean
}

/**
 * Sequentially thresholded least squares (pysindy's STLSQ) on a feature matrix: start from least squares, then
 * repeatedly fit a ridge regression on the active features of each target and zero the coefficients below the
 * threshold, until the supports stop changing; finally refit the support by least squares.
 *
 * @param theta - Features `[m, k]`.
 * @param targets - Targets `[m, n]` (e.g. time derivatives).
 * @param options - Threshold, ridge weight, iteration cap and unbiasing.
 * @returns The coefficients `[n, k]` (one row per target, as pysindy's coef_).
 * @concept sparse-regression
 * @example
 * ```ts
 * import { stlsq } from 'mdml-engine/dynamics'
 * // The second feature is irrelevant: thresholding removes it and the refit gives 2 exactly.
 * Array.from(stlsq([[1, 1], [1, -1], [1, 1], [1, -1]], [[2], [2], [2], [2]]).data) // => [2, 0]
 * ```
 */
export function stlsq(theta: TensorLike, targets: TensorLike, options: SindyOptions = {}): Tensor {
  const Th = primalOf(asTensor(theta))
  const Yt = primalOf(asTensor(targets))
  const m = Th.shape[0] ?? 0
  const k = Th.shape[1] ?? 0
  const nT = Yt.shape.length === 1 ? 1 : (Yt.shape[1] ?? 1)
  const thr = options.threshold ?? 0.1
  const alpha = options.alpha ?? 0.05
  const maxIter = options.maxIterations ?? 20
  const colOf = (i: number): Float64Array => Float64Array.from({ length: m }, (_, r) => Yt.data[r * nT + i] ?? 0)
  const sub2 = (cols: readonly number[]): Tensor => {
    const out = new Float64Array(m * cols.length)
    for (let r = 0; r < m; r++)
      cols.forEach((c, j) => {
        out[r * cols.length + j] = Th.data[r * k + c] ?? 0
      })
    return fromNumbers(out, [m, cols.length])
  }
  const ridge = (cols: readonly number[], y: Float64Array, a: number): number[] => {
    const A = sub2(cols)
    const At = transpose(A)
    const lhs = add(matmul(At, A), mul(eye(cols.length), a))
    return toFlat(solve(lhs, matmul(At, fromNumbers(y, [m, 1]))))
  }
  let coef = Array.from({ length: nT }, (_, i) => toFlat(lstsq(Th, colOf(i))))
  let ind = Array.from({ length: nT }, () => new Array<boolean>(k).fill(true))
  const initialSelected = nT * k
  let prev: boolean[][] | undefined
  for (let it = 0; it < maxIter; it++) {
    if (ind.every((r) => r.every((v) => !v))) break
    coef = ind.map((mask, i) => {
      const active = mask.map((v, c) => (v ? c : -1)).filter((c) => c >= 0)
      if (active.length === 0) return new Array<number>(k).fill(0)
      const w = ridge(active, colOf(i), alpha)
      const full = new Array<number>(k).fill(0)
      active.forEach((c, j) => {
        full[c] = w[j] ?? 0
      })
      return full.map((v) => (Math.abs(v) >= thr ? v : 0))
    })
    ind = coef.map((row) => row.map((v) => v !== 0))
    const selected = ind.reduce((acc, r) => acc + r.filter(Boolean).length, 0)
    const same = prev !== undefined && ind.every((r, i) => r.every((v, c) => v === prev?.[i]?.[c]))
    if (selected === initialSelected || same) break
    prev = ind.map((r) => [...r])
  }
  if (options.unbias !== false) {
    coef = coef.map((row, i) => {
      const active = row.map((v, c) => (Math.abs(v) > 1e-14 ? c : -1)).filter((c) => c >= 0)
      if (active.length === 0) return row
      const w = toFlat(lstsq(sub2(active), colOf(i)))
      const full = new Array<number>(k).fill(0)
      active.forEach((c, j) => {
        full[c] = w[j] ?? 0
      })
      return full
    })
  }
  return fromNumbers(coef.flat(), [nT, k])
}

/**
 * SINDy, sparse identification of nonlinear dynamics (Brunton, Proctor and Kutz 2016): evaluate a library of candidate
 * functions on the states, estimate the time derivatives (central differences when not given), and find a sparse
 * combination by STLSQ.
 *
 * @param x - States `[m, n]` sampled every dt.
 * @param options - Time step (or derivatives), library degree (default 2) and STLSQ options.
 * @returns The coefficients `[n, k]`, the feature names and a readable equation per state (coefficients rounded to
 * six decimals).
 * @concept sindy
 * @example
 * ```ts
 * import { sindy } from 'mdml-engine/dynamics'
 * // The harmonic oscillator x0' = 2 x1, x1' = -2 x0 sampled over one period, with a linear library.
 * const xs = Array.from({ length: 64 }, (_, i) => [Math.cos((i * Math.PI) / 32), -Math.sin((i * Math.PI) / 32)])
 * const xdot = xs.map(([a, b]) => [2 * b, -2 * a])
 * sindy(xs, { derivatives: xdot, degree: 1 }).equations // => ["x0' = 2 x1", "x1' = -2 x0"]
 * ```
 */
export function sindy(
  x: TensorLike,
  options: SindyOptions & { readonly dt?: number; readonly derivatives?: TensorLike; readonly degree?: number },
): { readonly coefficients: Tensor; readonly names: readonly string[]; readonly equations: readonly string[] } {
  const X = primalOf(asTensor(x))
  const m = X.shape[0] ?? 0
  const n = X.shape[1] ?? 1
  let D: Tensor
  if (options.derivatives !== undefined) D = primalOf(asTensor(options.derivatives))
  else {
    const dt = options.dt
    if (dt === undefined) throw new MdmlError('value', 'sindy: give dt or the derivatives')
    const out = new Float64Array(m * n)
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) {
        const v = (k: number): number => X.data[k * n + c] ?? 0
        out[r * n + c] =
          r === 0
            ? (-3 * v(0) + 4 * v(1) - v(2)) / (2 * dt)
            : r === m - 1
              ? (3 * v(m - 1) - 4 * v(m - 2) + v(m - 3)) / (2 * dt)
              : (v(r + 1) - v(r - 1)) / (2 * dt)
      }
    }
    D = fromNumbers(out, [m, n])
  }
  const lib = polynomialLibrary(n, options.degree ?? 2)
  const Theta = stack(Array.from({ length: m }, (_, r) => lib.features(reshape(sliceAxis(X, 0, r, r + 1), [-1]))))
  const coef = stlsq(Theta, D, options)
  const equations = Array.from({ length: n }, (_, i) => {
    const terms = lib.names
      .map((name, c) => ({ name, v: Math.round((coef.data[i * lib.size + c] ?? 0) * 1e6) / 1e6 }))
      .filter((t) => t.v !== 0)
      .map((t) => `${t.v}${t.name === '1' ? '' : ` ${t.name}`}`)
    return `x${i}' = ${terms.length === 0 ? '0' : terms.join(' + ')}`
  })
  return { coefficients: coef, names: lib.names, equations }
}

/** The regression rows and targets of an ARX model: `phi_t = [-y_{t-1} .. -y_{t-na}, u_{t-nk} .. u_{t-nk-nb+1}]`, target `y_t`. */
export interface Regression {
  /** One row per usable sample, `[rows][na + nb]`. */
  readonly phi: readonly (readonly number[])[]
  /** The matching outputs. */
  readonly y: readonly number[]
}

/**
 * The regression of an ARX model on a data record, the rows {@link arx} solves in one go and {@link rlsAlgorithm} takes one
 * at a time.
 *
 * @param y - Output series.
 * @param u - Input series.
 * @param orders - na, nb and the delay nk (default 1).
 * @returns The regressors and targets, from the first index where every lag exists.
 * @throws MdmlError with code `value` for mismatched lengths.
 * @concept system-identification
 * @example
 * ```ts
 * import { arxRegression } from 'mdml-engine/dynamics'
 * arxRegression([1, 2, 3, 4], [5, 6, 7, 8], { na: 1, nb: 1 }).phi // => [[-1, 5], [-2, 6], [-3, 7]]
 * ```
 */
export function arxRegression(
  y: TensorLike,
  u: TensorLike,
  orders: { readonly na: number; readonly nb: number; readonly nk?: number },
): Regression {
  const ys = series(y)
  const us = series(u)
  if (ys.length !== us.length) throw new MdmlError('value', 'arxRegression: y and u need the same length')
  const { na, nb } = orders
  const nk = orders.nk ?? 1
  const start = Math.max(na, nk + nb - 1)
  const phi: number[][] = []
  const target: number[] = []
  for (let t = start; t < ys.length; t++) {
    phi.push([
      ...Array.from({ length: na }, (_, i) => -(ys[t - 1 - i] ?? 0)),
      ...Array.from({ length: nb }, (_, j) => us[t - nk - j] ?? 0),
    ])
    target.push(ys[t] ?? 0)
  }
  return { phi, y: target }
}

/** The state of {@link rlsAlgorithm}. */
export interface RlsState {
  /** The parameter estimate. */
  readonly theta: Float64Array
  /** The covariance `P` (flat `[d, d]`): the inverse of the forgetting-weighted information. */
  readonly covariance: Float64Array
  /** Samples used. */
  readonly t: number
  /** The a-priori prediction error of the last sample, y - phi . theta before the update. */
  readonly error: number
  /** The gain vector of the last update. */
  readonly gain: Float64Array
}

/**
 * One recursive-least-squares update with forgetting factor `lambda`: the gain K = P phi / (lambda + phi' P phi), the
 * estimate theta + K (y - phi' theta), and P replaced by (P - K phi' P) / lambda, computed in the Joseph form so that it stays
 * symmetric and positive definite. After t samples the estimate is the weighted least squares of the prior and the samples,
 * sample k weighted lambda^(t-k): with lambda = 1 the batch fit; with lambda < 1 old samples are discounted, so the estimate
 * follows a system that changes.
 *
 * @param state - The estimate and covariance.
 * @param phi - The regressor of the new sample.
 * @param y - Its output.
 * @param lambda - The forgetting factor, in (0, 1].
 * @returns The updated state.
 * @concept recursive-least-squares
 * @example
 * ```ts
 * import { rlsUpdate } from 'mdml-engine/dynamics'
 * const s0 = { theta: Float64Array.of(0), covariance: Float64Array.of(100), t: 0, error: 0, gain: Float64Array.of(0) }
 * // One sample of y = 2 phi at phi = 1 moves a weak prior almost all the way to 2.
 * Math.round(rlsUpdate(s0, [1], 2, 1).theta[0] * 1000) / 1000 // => 1.98
 * ```
 */
export function rlsUpdate(state: RlsState, phi: ArrayLike<number>, y: number, lambda: number): RlsState {
  const d = state.theta.length
  const pPhi = dgemv(state.covariance, d, d, phi)
  const denominator = lambda + ddot(phi, pPhi)
  const gain = pPhi.map((v) => v / denominator)
  const error = y - ddot(phi, state.theta)
  const theta = state.theta.map((v, i) => v + (gain[i] ?? 0) * error)
  // P+ = (I - K phi') (P / lambda) (I - K phi')' + K K': the Joseph form, symmetric and positive semi-definite by
  // construction, where the textbook (P - K phi' P) / lambda loses both as the covariance shrinks.
  const m = new Float64Array(d * d)
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) m[i * d + j] = (i === j ? 1 : 0) - (gain[i] ?? 0) * (phi[j] ?? 0)
  const mp = new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    for (let k = 0; k < d; k++) {
      const mik = m[i * d + k] ?? 0
      for (let j = 0; j < d; j++) mp[i * d + j] = (mp[i * d + j] ?? 0) + mik * (state.covariance[k * d + j] ?? 0)
    }
  }
  const covariance = new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let v = 0
      for (let k = 0; k < d; k++) v += (mp[i * d + k] ?? 0) * (m[j * d + k] ?? 0)
      covariance[i * d + j] = v / lambda + (gain[i] ?? 0) * (gain[j] ?? 0)
    }
  }
  return { theta, covariance, t: state.t + 1, error, gain }
}

/** The paths of {@link rlsAlgorithm}. */
export type RlsPath = 'fit/parameters' | 'fit/error' | 'fit/gain' | 'fit/covarianceTrace'

/** What a recursive-least-squares run logs. */
export const rlsContract: Contract<RlsState, RlsPath> = defineContract({
  'fit/parameters': tensorSource({
    role: 'param',
    shape: ['d'],
    description: 'the parameter estimate after the sample',
  }),
  'fit/error': scalar({ description: 'the a-priori prediction error of the sample', direction: 'lower' }),
  'fit/gain': tensorSource({
    role: 'grad',
    shape: ['d'],
    description: 'the update gain: how far the error moves each parameter',
  }),
  'fit/covarianceTrace': scalar({
    description: 'the trace of P: the uncertainty left in the estimate',
    direction: 'lower',
  }),
})

/** Options of {@link rlsAlgorithm}. */
export interface RlsOptions {
  /** The forgetting factor, in (0, 1] (default 1: nothing is forgotten). */
  readonly forgetting?: number
  /** The prior covariance is `initialVariance` times the identity (default 1000: a vague prior). */
  readonly initialVariance?: number
  /** The starting estimate (default zeros). */
  readonly theta0?: ArrayLike<number>
}

/**
 * Recursive least squares as a stepped algorithm: one sample per step, the estimate moving by the gain times the prediction
 * error, with a forgetting factor that lets it follow a changing system. Logs the parameters, the a-priori error, the gain
 * and the trace of the covariance. The regression of an ARX model on a record comes from {@link arxRegression}; any linear
 * regression works.
 *
 * @param regression - The regressors and targets, one row per sample.
 * @param options - Forgetting, the prior and the start.
 * @returns The algorithm, which ends (completed) after the last sample.
 * @throws MdmlError with code `value` for a forgetting factor outside (0, 1] or rows of different sizes.
 * @concept recursive-least-squares
 * @example
 * ```ts
 * import { rlsAlgorithm } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // y = 3 phi exactly: the estimate is 3 after a few samples.
 * const phi = [[1], [2], [3], [4]]
 * const alg = rlsAlgorithm({ phi, y: [3, 6, 9, 12] })
 * Math.round(runAlgorithm(alg, { maxSteps: 10 }).state.theta[0] * 1e3) / 1e3 // => 3
 * ```
 */
export function rlsAlgorithm(regression: Regression, options: RlsOptions = {}): Algorithm<RlsState, RlsPath> {
  const lambda = options.forgetting ?? 1
  if (!(lambda > 0 && lambda <= 1)) throw new MdmlError('value', 'rlsAlgorithm: the forgetting factor is in (0, 1]')
  const d = regression.phi[0]?.length ?? 0
  if (d === 0 || regression.phi.some((r) => r.length !== d) || regression.y.length !== regression.phi.length) {
    throw new MdmlError('value', 'rlsAlgorithm: give rows of equal size, one target per row')
  }
  const delta = options.initialVariance ?? 1000
  const n = regression.phi.length
  return {
    name: 'recursiveLeastSquares',
    config: { forgetting: lambda, initialVariance: delta, samples: n, size: d },
    init: () => {
      const covariance = new Float64Array(d * d)
      for (let i = 0; i < d; i++) covariance[i * d + i] = delta
      return {
        theta: Float64Array.from(options.theta0 ?? new Float64Array(d)),
        covariance,
        t: 0,
        error: 0,
        gain: new Float64Array(d),
      }
    },
    step: (s, ctx) => {
      const next = rlsUpdate(s, regression.phi[s.t] as number[], regression.y[s.t] as number, lambda)
      ctx.log('fit/parameters', next.theta)
      ctx.log('fit/error', next.error)
      ctx.log('fit/gain', next.gain)
      let trace = 0
      for (let i = 0; i < d; i++) trace += next.covariance[i * d + i] ?? 0
      ctx.log('fit/covarianceTrace', trace)
      return next
    },
    done: (s) => (s.t >= n ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as RlsState,
  }
}

/** A planted ARX system with its truth: the polynomials, and (optionally) a step at which they change. */
export interface ArxPlant {
  /** a_1 .. a_na of A(q) y = B(q) u + e (y_t = -a_1 y_{t-1} - ... + b_1 u_{t-nk} + ...). */
  readonly a: readonly number[]
  /** b_1 .. b_nb. */
  readonly b: readonly number[]
  readonly nk?: number
}

/**
 * Simulate an ARX plant driven by an input record, with white measurement noise from a key, the coefficients switching
 * to another set at a given sample: data whose truth is known at every step, for studying how identification converges and
 * how a forgetting factor tracks a change.
 *
 * @param plant - The coefficients before the change.
 * @param u - The input series.
 * @param options - Noise standard deviation (default 0), its key, and the change (the sample and the new plant).
 * @returns The output series, and the true coefficient vector `[a, b]` at each sample.
 * @concept system-identification
 * @example
 * ```ts
 * import { simulateArx } from 'mdml-engine/dynamics'
 * // y_t = 0.5 y_{t-1} + 2 u_{t-1}.
 * simulateArx({ a: [-0.5], b: [2] }, [1, 0, 0, 1]).y // => [0, 2, 1, 0.5]
 * ```
 */
export function simulateArx(
  plant: ArxPlant,
  u: TensorLike,
  options: {
    readonly noise?: number
    readonly key?: Key
    readonly change?: { readonly at: number; readonly to: ArxPlant }
  } = {},
): { readonly y: number[]; readonly truth: number[][] } {
  const us = series(u)
  const noise = options.noise ?? 0
  const e = noise > 0 ? Array.from(normal(options.key ?? key(0), [us.length], { scale: noise }).data) : []
  const y: number[] = []
  const truth: number[][] = []
  for (let t = 0; t < us.length; t++) {
    const p = options.change !== undefined && t >= options.change.at ? options.change.to : plant
    const nk = p.nk ?? 1
    let v = e[t] ?? 0
    p.a.forEach((a, i) => {
      v -= a * (y[t - 1 - i] ?? 0)
    })
    p.b.forEach((b, j) => {
      v += b * (us[t - nk - j] ?? 0)
    })
    y.push(v)
    truth.push([...p.a, ...p.b])
  }
  return { y, truth }
}

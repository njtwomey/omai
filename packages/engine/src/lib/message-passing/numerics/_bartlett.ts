// Small dense algebra for the hot loops of the matrix sites (a Cholesky factor in place, a solve, a log determinant, on
// row-major Float64Arrays of order two to six: no tensors, no allocation per call), and the Bartlett quadrature of a Wishart
// distribution: nodes and weights that integrate a smooth function of a Wishart matrix exactly up to a polynomial degree, from
// the decomposition `L = C A A' C'` with `A` lower triangular, `A_ii^2 ~ chi-square(nu - i)` and `A_ij ~ N(0, 1)` below the
// diagonal, all independent, so a product of generalised Gauss-Laguerre and Gauss-Hermite rules is a rule for the matrix.
import { OmaiError } from '../../base/index.ts'
import { gaussHermite, gaussLaguerre } from '../../numerics/quadrature/index.ts'
import { div, mul, sum } from '../../tensor/index.ts'
import { float64Of } from './_arrays.ts'
import { digammaOf } from './_scalar.ts'

/**
 * The Cholesky factor of a symmetric matrix, overwritten into its lower triangle (the upper triangle is left alone).
 *
 * @param a - The matrix, row-major `n x n`; the lower triangle becomes the factor.
 * @param n - Its order.
 * @returns False when the matrix is not positive definite.
 */
export function choleskyInPlace(a: Float64Array, n: number): boolean {
  for (let j = 0; j < n; j++) {
    let s = a[j * n + j] as number
    for (let k = 0; k < j; k++) s -= (a[j * n + k] as number) ** 2
    if (!(s > 0) || !Number.isFinite(s)) return false
    const d = Math.sqrt(s)
    a[j * n + j] = d
    for (let i = j + 1; i < n; i++) {
      let t = a[i * n + j] as number
      for (let k = 0; k < j; k++) t -= (a[i * n + k] as number) * (a[j * n + k] as number)
      a[i * n + j] = t / d
    }
  }
  return true
}

/** The log determinant from a Cholesky factor in place. */
export function choleskyLogDet(l: Float64Array, n: number): number {
  let s = 0
  for (let i = 0; i < n; i++) s += Math.log(l[i * n + i] as number)
  return 2 * s
}

/** Solve `A x = b` in place (`b` becomes `x`) from the Cholesky factor of `A`. */
export function choleskySolveInPlace(l: Float64Array, n: number, b: Float64Array): void {
  for (let i = 0; i < n; i++) {
    let s = b[i] as number
    for (let k = 0; k < i; k++) s -= (l[i * n + k] as number) * (b[k] as number)
    b[i] = s / (l[i * n + i] as number)
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i] as number
    for (let k = i + 1; k < n; k++) s -= (l[k * n + i] as number) * (b[k] as number)
    b[i] = s / (l[i * n + i] as number)
  }
}

/** The inverse of a matrix from its Cholesky factor, row-major. */
export function choleskyInverse(l: Float64Array, n: number): Float64Array {
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const out = new Float64Array(n * n)
  const col = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
    col.fill(0)
    col[j] = 1
    choleskySolveInPlace(l, n, col)
    for (let i = 0; i < n; i++) out[i * n + j] = col[i] as number
  }
  return out
}

/** One-dimensional rules by (kind, order, shape): the generalised Laguerre rule of a gamma weight and the Gauss-Hermite rule of a normal. */
const rules = new Map<string, { readonly x: Float64Array; readonly w: Float64Array }>()

/**
 * The nodes and weights of order `order` for `E f(g)`, `g ~ Gamma(shape, 1)` (weights sum to one).
 *
 * @param order - Number of nodes.
 * @param shape - The gamma shape (positive).
 * @returns The rule.
 */
function gammaRule(order: number, shape: number): { readonly x: Float64Array; readonly w: Float64Array } {
  const key = `g${order}:${shape}`
  let rule = rules.get(key)
  if (rule === undefined) {
    const r = gaussLaguerre(order, shape - 1)
    const w = r.weights
    rule = { x: float64Of(r.nodes), w: float64Of(div(w, sum(w))) }
    rules.set(key, rule)
  }
  return rule
}

/** The nodes and weights of order `order` for `E f(z)`, `z ~ N(0, 1)` (weights sum to one). */
function normalRule(order: number): { readonly x: Float64Array; readonly w: Float64Array } {
  const key = `n${order}`
  let rule = rules.get(key)
  if (rule === undefined) {
    const r = gaussHermite(order)
    const w = r.weights
    rule = { x: float64Of(mul(r.nodes, Math.SQRT2)), w: float64Of(div(w, sum(w))) }
    rules.set(key, rule)
  }
  return rule
}

/**
 * The order of the one-dimensional rules by matrix dimension: the product has `order^(d (d + 1) / 2)` nodes (59049 at d = 4 with three each; the
 * operators move the conjugate part of their factor into the base measure, so the remainder the rule integrates is smooth and three nodes
 * match an importance-sampling reference to about 3% in the spread at d = 4, where two nodes (d = 5) leave the spread 5% to 15% short).
 */
export const BARTLETT_ORDER: Readonly<Record<number, number>> = { 1: 40, 2: 12, 3: 5, 4: 3 }

/** A quadrature rule for a Wishart distribution. */
export interface WishartRule {
  readonly count: number
  /** The weights, summing to one. */
  readonly weight: Float64Array
  /** `log |L|` at each node. */
  readonly logDet: Float64Array
  /**
   * The quadrature's error for `E log |L|` under the weights themselves (the rule is exact for polynomials, and a logarithm is not):
   * the mean of the nodes' log determinants minus the exact `psi_d(nu / 2) + d log 2 + log |S|`. A tilted mean of `log |L|` over the
   * nodes is corrected by it, which removes the bias a smooth tilt inherits.
   */
  readonly logDetBias: number
  /** The matrices, node after node (`d x d` row-major each). */
  readonly lambda: Float64Array
}

/**
 * The Bartlett quadrature rule of the Wishart distribution `W_d(nu, S)`.
 *
 * @param nu - Degrees of freedom (above `d - 1`).
 * @param scaleChol - The lower Cholesky factor of the scale `S`, row-major.
 * @param d - The dimension (1 to 4).
 * @param order - Nodes per coordinate (default `BARTLETT_ORDER[d]`).
 * @returns The nodes (matrices and their log determinants) and the weights.
 * @throws OmaiError with code `not-implemented` for a dimension above four.
 */
export function wishartRule(nu: number, scaleChol: Float64Array, d: number, order?: number): WishartRule {
  const n = order ?? BARTLETT_ORDER[d]
  if (n === undefined)
    throw new OmaiError('not-implemented', `the Wishart quadrature is built for dimensions up to 4, got ${d}`)
  // banned-ok: non-numeric: one quadrature rule per diagonal entry
  const diag = Array.from({ length: d }, (_, i) => gammaRule(n, (nu - i) / 2))
  const normal = normalRule(n)
  const off = (d * (d - 1)) / 2
  const dims = d + off
  const count = n ** dims
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const weight = new Float64Array(count)
  const logDet = new Float64Array(count)
  const lambda = new Float64Array(count * d * d)
  let logC = 0
  for (let i = 0; i < d; i++) logC += Math.log(scaleChol[i * d + i] as number)
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const index = new Int32Array(dims)
  const a = new Float64Array(d * d)
  const t = new Float64Array(d * d)
  for (let k = 0; k < count; k++) {
    let rest = k
    for (let m = 0; m < dims; m++) {
      index[m] = rest % n
      rest = Math.floor(rest / n)
    }
    let w = 1
    let ld = 0
    // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
    a.fill(0)
    for (let i = 0; i < d; i++) {
      const r = diag[i] as { x: Float64Array; w: Float64Array }
      const q = index[i] as number
      w *= r.w[q] as number
      a[i * d + i] = Math.sqrt(2 * (r.x[q] as number))
      ld += Math.log(2 * (r.x[q] as number))
    }
    let m = d
    for (let i = 1; i < d; i++)
      for (let j = 0; j < i; j++) {
        const q = index[m++] as number
        w *= normal.w[q] as number
        a[i * d + j] = normal.x[q] as number
      }
    // T = C A (both lower triangular), L = T T'
    for (let i = 0; i < d; i++)
      for (let j = 0; j <= i; j++) {
        let s = 0
        for (let p = j; p <= i; p++) s += (scaleChol[i * d + p] as number) * (a[p * d + j] as number)
        t[i * d + j] = s
      }
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) {
        let s = 0
        for (let p = 0; p <= Math.min(i, j); p++) s += (t[i * d + p] as number) * (t[j * d + p] as number)
        lambda[k * d * d + i * d + j] = s
      }
    weight[k] = w
    logDet[k] = ld + 2 * logC
  }
  let exact = 2 * logC
  for (let i = 0; i < d; i++) exact += digammaOf((nu - i) / 2) + Math.LN2
  let mean = 0
  for (let k = 0; k < count; k++) mean += (weight[k] as number) * (logDet[k] as number)
  return { count, weight, logDet, logDetBias: mean - exact, lambda }
}

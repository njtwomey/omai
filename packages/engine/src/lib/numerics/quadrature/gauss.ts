/**
 * Gaussian quadrature rules, nodes and weights exact for polynomials up to degree 2n - 1 against a weight:
 * Gauss-Legendre, Hermite, Laguerre and Jacobi (as numpy and scipy.special), fixed double-exponential rules for
 * integrating many integrands at once, fixed-order quadrature over an interval, and normal expectations by
 * Gauss-Hermite.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  fromStorage,
  item,
  mul,
  promoteFloat,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { eigh } from '../linalg/index.ts'
import { lbeta } from '../special/gamma.ts'

/** Nodes and weights of a quadrature rule. */
export interface QuadratureRule {
  readonly nodes: Tensor
  readonly weights: Tensor
}

/**
 * The Golub-Welsch algorithm: the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix of the
 * orthogonal polynomials' recurrence, and the weights mu0 times the squared first components of its eigenvectors.
 */
function golubWelsch(n: number, a: (k: number) => number, b: (k: number) => number, mu0: number): QuadratureRule {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'quadrature rules need n >= 1')
  const j = new Float64Array(n * n)
  for (let k = 0; k < n; k++) {
    j[k * n + k] = a(k)
    if (k + 1 < n) {
      j[k * n + k + 1] = b(k + 1)
      j[(k + 1) * n + k] = b(k + 1)
    }
  }
  const { eigenvalues, eigenvectors } = eigh(fromStorage(j, [n, n], 'float64'))
  const v = (eigenvectors as Tensor).data
  const weights = Float64Array.from({ length: n }, (_, k) => mu0 * (v[k] ?? 0) ** 2)
  return { nodes: eigenvalues as Tensor, weights: fromStorage(weights, [n], 'float64') }
}

/** Legendre polynomial P_n and its derivative at x, by the three-term recurrence. */
function legendre(n: number, x: number): [number, number] {
  let p0 = 1
  let p1 = x
  if (n === 0) return [1, 0]
  for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k]
  return [p1, (n * (x * p1 - p0)) / (x * x - 1)]
}

/**
 * Gauss-Legendre nodes and weights on [-1, 1] (numpy.polynomial.legendre.leggauss): exact for polynomials of degree
 * 2n - 1. Nodes come from Golub-Welsch, polished by Newton steps on the recurrence, with weights 2 / ((1 - x^2) P'(x)^2).
 *
 * @param n - Number of points.
 * @returns The rule.
 * @concept gaussian-quadrature
 * @example
 * ```ts
 * import { gaussLegendre } from 'mdml-engine/numerics/quadrature'
 * Array.from(gaussLegendre(2).nodes.data).map((x) => Math.round(x * 1e12) / 1e12) // => [-0.57735026919, 0.57735026919]
 * ```
 */
export function gaussLegendre(n: number): QuadratureRule {
  const rule = golubWelsch(
    n,
    () => 0,
    (k) => k / Math.sqrt(4 * k * k - 1),
    2,
  )
  const nodes = Float64Array.from(rule.nodes.data)
  const weights = new Float64Array(n)
  nodes.forEach((x0, i) => {
    let x = x0
    for (let it = 0; it < 3; it++) {
      const [p, dp] = legendre(n, x)
      if (dp !== 0) x -= p / dp
    }
    const [, dp] = legendre(n, x)
    nodes[i] = x
    weights[i] = 2 / ((1 - x * x) * dp * dp)
  })
  return { nodes: fromStorage(nodes, [n], 'float64'), weights: fromStorage(weights, [n], 'float64') }
}

/**
 * Gauss-Hermite nodes and weights for the weight exp(-x^2) on the real line (numpy.polynomial.hermite.hermgauss).
 *
 * @param n - Number of points.
 * @returns The rule.
 * @example
 * ```ts
 * import { gaussHermite } from 'mdml-engine/numerics/quadrature'
 * Math.abs(Array.from(gaussHermite(5).weights.data).reduce((a, b) => a + b) - Math.sqrt(Math.PI)) < 1e-13 // => true
 * ```
 */
export function gaussHermite(n: number): QuadratureRule {
  return golubWelsch(
    n,
    () => 0,
    (k) => Math.sqrt(k / 2),
    Math.sqrt(Math.PI),
  )
}

/**
 * Gauss-Laguerre nodes and weights for the weight exp(-x) on [0, inf) (numpy.polynomial.laguerre.laggauss).
 *
 * @param n - Number of points.
 * @returns The rule.
 * @example
 * ```ts
 * import { gaussLaguerre } from 'mdml-engine/numerics/quadrature'
 * Math.abs(Array.from(gaussLaguerre(4).weights.data).reduce((a, b) => a + b) - 1) < 1e-13 // => true
 * ```
 */
export function gaussLaguerre(n: number): QuadratureRule {
  return golubWelsch(
    n,
    (k) => 2 * k + 1,
    (k) => k,
    1,
  )
}

/**
 * Gauss-Jacobi nodes and weights for the weight (1 - x)^alpha (1 + x)^beta on [-1, 1] (scipy.special.roots_jacobi).
 *
 * @param n - Number of points.
 * @param alpha - Exponent at 1 (> -1).
 * @param beta - Exponent at -1 (> -1).
 * @returns The rule.
 * @example
 * ```ts
 * import { gaussJacobi } from 'mdml-engine/numerics/quadrature'
 * gaussJacobi(3, 0.5, 0.5).nodes.shape // => [3]
 * ```
 */
export function gaussJacobi(n: number, alpha: number, beta: number): QuadratureRule {
  if (!(alpha > -1 && beta > -1)) throw new MdmlError('value', 'gaussJacobi: alpha and beta must exceed -1')
  const ab = alpha + beta
  const a = (k: number): number => {
    const d = (2 * k + ab) * (2 * k + ab + 2)
    return d === 0 ? (beta - alpha) / (ab + 2) : (beta * beta - alpha * alpha) / d
  }
  const b = (k: number): number => {
    const s = 2 * k + ab
    const num = 4 * k * (k + alpha) * (k + beta) * (k + ab)
    const den = s * s * (s + 1) * (s - 1)
    return k === 1 && ab === -1
      ? Math.sqrt((4 * (1 + alpha) * (1 + beta)) / ((2 + ab) ** 2 * (3 + ab)))
      : Math.sqrt(num / den)
  }
  const mu0 = Math.exp((ab + 1) * Math.LN2 + lbetaNumber(alpha + 1, beta + 1))
  return golubWelsch(n, a, b, mu0)
}

function lbetaNumber(a: number, b: number): number {
  return item(lbeta(a, b))
}

/** The range a double-exponential rule integrates over: [-1, 1], [0, inf) or the real line. */
export type DoubleExponentialRange = 'finite' | 'halfLine' | 'real'

/**
 * A fixed double-exponential (Takahasi-Mori) rule as nodes and weights, for integrating many integrands at once (a
 * batch of densities, say): tanh-sinh x = tanh((pi/2) sinh t) on [-1, 1], exp-sinh x = exp((pi/2) sinh t) on
 * [0, inf) and sinh-sinh x = sinh((pi/2) sinh t) on the real line, with the trapezoidal rule in t on [-4, 4]. Nodes
 * that round to an endpoint and weights that underflow are dropped. Convergence is double-exponential for integrands
 * analytic inside the range, endpoint singularities included; shift and scale the nodes to the integrand's bulk.
 *
 * @param range - `finite`, `halfLine` or `real`.
 * @param h - Step in t (default 1/32: 257 points before dropping).
 * @returns The rule.
 * @example
 * ```ts
 * import { doubleExponentialRule } from 'mdml-engine/numerics/quadrature'
 * import { exp, item, mul, neg, square, sum } from 'mdml-engine/tensor'
 * const { nodes, weights } = doubleExponentialRule('real')
 * Math.abs(item(sum(mul(weights, exp(neg(square(nodes)))))) - Math.sqrt(Math.PI)) < 1e-14 // => true
 * ```
 */
export function doubleExponentialRule(range: DoubleExponentialRange, h: number = 1 / 32): QuadratureRule {
  if (!(h > 0)) throw new MdmlError('value', 'doubleExponentialRule: the step must be positive')
  const xs: number[] = []
  const ws: number[] = []
  const steps = Math.round(4 / h)
  for (let i = -steps; i <= steps; i++) {
    const t = i * h
    const u = (Math.PI / 2) * Math.sinh(t)
    const du = (Math.PI / 2) * Math.cosh(t)
    let x: number
    let w: number
    if (range === 'finite') {
      x = Math.tanh(u)
      w = (h * du) / Math.cosh(u) ** 2
      if (Math.abs(x) >= 1) continue
    } else if (range === 'halfLine') {
      x = Math.exp(u)
      w = h * du * x
      if (x === 0 || !Number.isFinite(x)) continue
    } else {
      x = Math.sinh(u)
      w = h * du * Math.cosh(u)
      if (!Number.isFinite(x)) continue
    }
    if (!(w > 0 && Number.isFinite(w))) continue
    xs.push(x)
    ws.push(w)
  }
  return {
    nodes: fromStorage(Float64Array.from(xs), [xs.length], 'float64'),
    weights: fromStorage(Float64Array.from(ws), [ws.length], 'float64'),
  }
}

/**
 * Fixed-order Gauss-Legendre quadrature of f over [a, b] (scipy.integrate.fixed_quad). The integrand is applied to the
 * vector of nodes at once; the result is differentiable in a, b and anything f closes over.
 *
 * @param f - The integrand, vectorised over a 1-D value of nodes.
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @param n - Number of points (default 5).
 * @returns The estimate.
 * @concept gaussian-quadrature
 * @example
 * ```ts
 * import { fixedQuad } from 'mdml-engine/numerics/quadrature'
 * import { item, square } from 'mdml-engine/tensor'
 * Math.abs(item(fixedQuad((x) => square(x), 0, 3)) - 9) < 1e-13 // => true
 * ```
 */
export function fixedQuad(f: (x: Value) => Value, a: TensorLike, b: TensorLike, n = 5): Value {
  const [lo, hi] = promoteFloat(a, b) as [Value, Value]
  const { nodes, weights } = gaussLegendre(n)
  const half = mul(add(hi, mul(lo, -1)), 0.5)
  const mid = mul(add(hi, lo), 0.5)
  const x = add(mid, mul(half, nodes))
  return mul(half, sum(mul(weights, f(x))))
}

/**
 * E[f(X)] for X ~ N(mean, std^2) by Gauss-Hermite quadrature: (1 / sqrt(pi)) sum w_i f(mean + sqrt(2) std x_i).
 * Differentiable in the mean, the standard deviation and anything f closes over.
 *
 * @param f - The function, vectorised over a 1-D value.
 * @param mean - Mean.
 * @param std - Standard deviation.
 * @param n - Number of points (default 20).
 * @returns The expectation.
 * @example
 * ```ts
 * import { normalExpectation } from 'mdml-engine/numerics/quadrature'
 * import { item, square } from 'mdml-engine/tensor'
 * Math.abs(item(normalExpectation((x) => square(x), 1, 2)) - 5) < 1e-12 // => true
 * ```
 */
export function normalExpectation(f: (x: Value) => Value, mean: TensorLike, std: TensorLike, n = 20): Value {
  const [mu, sigma] = promoteFloat(mean, std) as [Value, Value]
  const { nodes, weights } = gaussHermite(n)
  const x = add(mu, mul(mul(sigma, Math.SQRT2), nodes))
  return mul(sum(mul(weights, f(x))), 1 / Math.sqrt(Math.PI))
}

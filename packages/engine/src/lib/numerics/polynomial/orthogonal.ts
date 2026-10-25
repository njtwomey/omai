/**
 * Orthogonal polynomial series, numpy.polynomial style, for the Legendre, Chebyshev, Hermite (both conventions),
 * Laguerre and Jacobi families: evaluation by Clenshaw's recurrence, pseudo-Vandermonde matrices, least-squares fits,
 * the Gauss quadrature rule of each family, and conversion to and from the power basis.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  concat,
  mul,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { gaussHermite, gaussJacobi, gaussLaguerre, gaussLegendre, type QuadratureRule } from '../quadrature/index.ts'
import { polyFromRoots, weightedFit } from './power.ts'

/** The orthogonal polynomial families (numpy.polynomial and scipy.special naming). */
export type Family = 'legendre' | 'chebyshev' | 'hermite' | 'hermite_e' | 'laguerre' | 'jacobi'

/** Jacobi parameters (alpha, beta > -1); ignored by the other families. */
export interface JacobiParameters {
  readonly alpha?: number
  readonly beta?: number
}

/** The three-term recurrence P_{n+1} = (a_n x + b_n) P_n - c_n P_{n-1}, P_0 = 1, P_{-1} = 0. */
function recurrence(family: Family, p: JacobiParameters): (n: number) => readonly [number, number, number] {
  switch (family) {
    case 'legendre':
      return (n) => [(2 * n + 1) / (n + 1), 0, n / (n + 1)]
    case 'chebyshev':
      return (n) => [n === 0 ? 1 : 2, 0, 1]
    case 'hermite':
      return (n) => [2, 0, 2 * n]
    case 'hermite_e':
      return (n) => [1, 0, n]
    case 'laguerre':
      return (n) => [-1 / (n + 1), (2 * n + 1) / (n + 1), n / (n + 1)]
    case 'jacobi': {
      const a = p.alpha ?? 0
      const b = p.beta ?? 0
      if (!(a > -1 && b > -1)) throw new MdmlError('value', 'jacobi: alpha and beta must exceed -1')
      return (n) => {
        if (n === 0) return [(a + b + 2) / 2, (a - b) / 2, 0]
        const s = 2 * n + a + b
        const d = 2 * (n + 1) * (n + a + b + 1) * s
        return [((s + 1) * (s + 2) * s) / d, ((s + 1) * (a * a - b * b)) / d, (2 * (n + a) * (n + b) * (s + 2)) / d]
      }
    }
  }
}

/**
 * Evaluate a series sum_k c_k P_k(x) in an orthogonal family by Clenshaw's recurrence (numpy's legval, chebval,
 * hermval, hermeval, lagval; scipy.special.eval_jacobi for single terms). Differentiable in x and c.
 *
 * @param family - The family.
 * @param x - Points.
 * @param c - Series coefficients, lowest degree first.
 * @param params - Jacobi's alpha and beta.
 * @returns Values, the shape of x.
 * @concept orthogonal-polynomials
 * @example
 * ```ts
 * import { orthoval } from 'mdml-engine/numerics/polynomial'
 * import { toArray } from 'mdml-engine/tensor'
 * // T_2(x) = 2x^2 - 1.
 * toArray(orthoval('chebyshev', [0, 0.5, 1], [0, 0, 1])) // => [-1, -0.5, 1]
 * ```
 */
export function orthoval(family: Family, x: TensorLike, c: TensorLike, params: JacobiParameters = {}): Value {
  const [xv, cv] = promoteFloat(x, c) as [Value, Value]
  const k = cv.shape[0] ?? 0
  const rec = recurrence(family, params)
  const coef = (i: number): Value => reshape(sliceAxis(cv, 0, i, i + 1), [])
  // b_k = c_k + (a_k x + b'_k) b_{k+1} - c_{k+1} b_{k+2}; the sum is b_0.
  let b1: Value = mul(xv, 0)
  let b2: Value = mul(xv, 0)
  for (let i = k - 1; i >= 0; i--) {
    const [a, b] = rec(i)
    const [, , cNext] = rec(i + 1)
    const next = sub(add(coef(i), mul(add(mul(xv, a), b), b1)), mul(b2, cNext))
    b2 = b1
    b1 = next
  }
  return b1
}

/**
 * The pseudo-Vandermonde matrix V[i, j] = P_j(x_i), j = 0..deg (numpy's legvander, chebvander, ...).
 *
 * @param family - The family.
 * @param x - Points `[m]`.
 * @param deg - Degree.
 * @param params - Jacobi parameters.
 * @returns `[m, deg + 1]`.
 * @example
 * ```ts
 * import { orthovander } from 'mdml-engine/numerics/polynomial'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(orthovander('hermite_e', [2], 2)) // => [[1, 2, 3]]
 * ```
 */
export function orthovander(family: Family, x: TensorLike, deg: number, params: JacobiParameters = {}): Value {
  const [xv] = promoteFloat(x) as [Value]
  const col = reshape(xv, [-1, 1])
  const rec = recurrence(family, params)
  const columns: Value[] = [add(mul(col, 0), 1)]
  for (let n = 0; n < deg; n++) {
    const [a, b, c] = rec(n)
    const prev = columns[n - 1]
    const next = mul(add(mul(col, a), b), columns[n] as Value)
    columns.push(prev === undefined ? next : sub(next, mul(prev, c)))
  }
  return concat(columns, 1)
}

/**
 * The least-squares fit of a series of degree deg (numpy's legfit, chebfit, hermfit, hermefit, lagfit): minimises
 * sum (w_i (y_i - f(x_i)))^2. Differentiable in x, y and w.
 *
 * @param family - The family.
 * @param x - Points `[m]`.
 * @param y - Values `[m]`.
 * @param deg - Degree.
 * @param options - Weights and Jacobi parameters.
 * @returns Series coefficients `[deg + 1]`.
 * @example
 * ```ts
 * import { orthofit } from 'mdml-engine/numerics/polynomial'
 * import { toArray } from 'mdml-engine/tensor'
 * // x^2 = (P_0 + 2 P_2) / 3 in Legendre polynomials.
 * toArray(orthofit('legendre', [-1, -0.5, 0, 0.5, 1], [1, 0.25, 0, 0.25, 1], 2)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.333333333333, 0, 0.666666666667]
 * ```
 */
export function orthofit(
  family: Family,
  x: TensorLike,
  y: TensorLike,
  deg: number,
  options: JacobiParameters & { readonly w?: TensorLike } = {},
): Value {
  return weightedFit(orthovander(family, x, deg, options), y, options.w)
}

/**
 * The Gauss quadrature rule of a family (numpy's leggauss, chebgauss, hermgauss, hermegauss, laggauss;
 * scipy.special.roots_jacobi): exact for polynomials of degree 2n - 1 against the family's weight.
 *
 * @param family - The family.
 * @param n - Number of points.
 * @param params - Jacobi parameters.
 * @returns Nodes and weights.
 * @example
 * ```ts
 * import { orthogauss } from 'mdml-engine/numerics/polynomial'
 * // Chebyshev weights are all pi / n.
 * Math.abs(orthogauss('chebyshev', 4).weights.data[0] - Math.PI / 4) < 1e-14 // => true
 * ```
 */
export function orthogauss(family: Family, n: number, params: JacobiParameters = {}): QuadratureRule {
  switch (family) {
    case 'legendre':
      return gaussLegendre(n)
    case 'chebyshev':
      return gaussJacobi(n, -0.5, -0.5)
    case 'hermite':
      return gaussHermite(n)
    case 'hermite_e': {
      // Weight exp(-x^2 / 2): nodes scaled by sqrt(2), weights by sqrt(2).
      const r = gaussHermite(n)
      return {
        nodes: mul(r.nodes, Math.SQRT2) as QuadratureRule['nodes'],
        weights: mul(r.weights, Math.SQRT2) as QuadratureRule['weights'],
      }
    }
    case 'laguerre':
      return gaussLaguerre(n)
    case 'jacobi':
      return gaussJacobi(n, params.alpha ?? 0, params.beta ?? 0)
  }
}

/**
 * Convert series coefficients to power-basis coefficients (numpy's leg2poly, cheb2poly, herm2poly, herme2poly,
 * lag2poly).
 *
 * @param family - The family.
 * @param c - Series coefficients, lowest first.
 * @param params - Jacobi parameters.
 * @returns Power coefficients, lowest first.
 * @example
 * ```ts
 * import { orthoToPower } from 'mdml-engine/numerics/polynomial'
 * orthoToPower('chebyshev', [0, 0, 0, 1]) // => [0, -3, 0, 4]
 * ```
 */
export function orthoToPower(family: Family, c: readonly number[], params: JacobiParameters = {}): number[] {
  const basis = basisInPower(family, c.length - 1, params)
  const out = new Array<number>(c.length).fill(0)
  c.forEach((ck, k) => {
    ;(basis[k] as number[]).forEach((v, j) => {
      out[j] = (out[j] ?? 0) + ck * v
    })
  })
  return out
}

/**
 * Convert power-basis coefficients to series coefficients (numpy's poly2leg, poly2cheb, ...), by back substitution
 * against the triangular change of basis.
 *
 * @param family - The family.
 * @param c - Power coefficients, lowest first.
 * @param params - Jacobi parameters.
 * @returns Series coefficients, lowest first.
 * @example
 * ```ts
 * import { powerToOrtho } from 'mdml-engine/numerics/polynomial'
 * powerToOrtho('chebyshev', [0, -3, 0, 4]) // => [0, 0, 0, 1]
 * ```
 */
export function powerToOrtho(family: Family, c: readonly number[], params: JacobiParameters = {}): number[] {
  const deg = c.length - 1
  const basis = basisInPower(family, deg, params)
  const rest = [...c]
  const out = new Array<number>(c.length).fill(0)
  for (let k = deg; k >= 0; k--) {
    const pk = basis[k] as number[]
    const coefficient = (rest[k] ?? 0) / (pk[k] ?? 1)
    out[k] = coefficient
    pk.forEach((v, j) => {
      rest[j] = (rest[j] ?? 0) - coefficient * v
    })
  }
  return out.map((v) => (Math.abs(v) < 1e-14 ? 0 : v))
}

/** P_0..P_deg in power coefficients, by the recurrence. */
function basisInPower(family: Family, deg: number, params: JacobiParameters): number[][] {
  const rec = recurrence(family, params)
  const basis: number[][] = [[1]]
  for (let n = 0; n < deg; n++) {
    const [a, b, c] = rec(n)
    const pn = basis[n] as number[]
    const prev = basis[n - 1] ?? []
    const next = new Array<number>(n + 2).fill(0)
    pn.forEach((v, j) => {
      next[j + 1] = (next[j + 1] ?? 0) + a * v
      next[j] = (next[j] ?? 0) + b * v
    })
    prev.forEach((v, j) => {
      next[j] = (next[j] ?? 0) - c * v
    })
    basis.push(next)
  }
  return basis
}

/**
 * The degree-n polynomial of a family with roots at the Gauss nodes, as a check of the rule: monic up to scale.
 *
 * @param family - The family.
 * @param n - Degree.
 * @param params - Jacobi parameters.
 * @returns Power coefficients of the monic polynomial with the Gauss nodes as roots.
 * @example
 * ```ts
 * import { monicFromNodes } from 'mdml-engine/numerics/polynomial'
 * monicFromNodes('chebyshev', 2).map((v) => Math.round(v * 1e12) / 1e12) // => [-0.5, 0, 1]
 * ```
 */
export function monicFromNodes(family: Family, n: number, params: JacobiParameters = {}): number[] {
  return polyFromRoots(Array.from(orthogauss(family, n, params).nodes.data))
}

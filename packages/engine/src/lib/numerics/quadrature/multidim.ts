/**
 * Integration in several dimensions: iterated adaptive quadrature with limits that may depend on outer variables
 * (scipy's `nquad`, `dblquad` and `tplquad`), and cubature rules on boxes, tensor-product Gauss rules and Smolyak
 * sparse grids, which need far fewer points in higher dimensions.
 */

import { MdmlError } from '../../base/index.ts'
import {
  concat,
  fromStorage,
  mul,
  primalOf,
  reshape,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { type QuadOptions, quad } from './adaptive.ts'
import { gaussLegendre } from './gauss.ts'

/** Limits of one variable: fixed, or a function of the outer variables (x_{k+1}, ..., x_{n-1}, as scipy.nquad). */
export type Range = readonly [number, number] | ((...outer: number[]) => readonly [number, number])

/**
 * Iterated adaptive quadrature over n variables (scipy.integrate.nquad): `ranges[0]` is the innermost variable x0, and
 * each range may depend on the variables outside it. The integrand receives the innermost variable as a 1-D value of
 * points and the outer variables as scalar values. Differentiable in the integrand's parameters.
 *
 * @param f - The integrand f(x0, x1, ..., x_{n-1}).
 * @param ranges - One range per variable, innermost first.
 * @param options - Options passed to every one-dimensional {@link quad}.
 * @returns The integral.
 * @concept quadrature
 * @index function Iterated adaptive integration over several variables.
 * @example
 * ```ts
 * import { nquad } from 'mdml-engine/numerics/quadrature'
 * import { item, mul } from 'mdml-engine/tensor'
 * Math.abs(item(nquad((x, y) => mul(x, y), [[0, 1], [0, 2]])) - 1) < 1e-12 // => true
 * ```
 */
export function nquad(f: (...xs: Value[]) => Value, ranges: readonly Range[], options: QuadOptions = {}): Value {
  if (ranges.length === 0) throw new MdmlError('value', 'nquad: at least one range is needed')
  const limits = (k: number, outer: readonly number[]): readonly [number, number] => {
    const r = ranges[k] as Range
    return typeof r === 'function' ? r(...outer) : r
  }
  // Integrate variable k with the variables k+1.. fixed at `outer`.
  const integrate = (k: number, outer: readonly number[]): Value => {
    const [lo, hi] = limits(k, outer)
    const fixed = outer.map((v) => tensor(v))
    if (k === 0) return quad((x) => f(x, ...fixed), lo, hi, options).value
    return quad(
      (xs) => {
        const nodes = toFlat(primalOf(xs))
        return concat(nodes.map((x) => reshape(integrate(k - 1, [x, ...outer]), [1])))
      },
      lo,
      hi,
      options,
    ).value
  }
  return integrate(ranges.length - 1, [])
}

/**
 * A double integral (scipy.integrate.dblquad): the integral over x in [a, b] and y in [g(x), h(x)] of f(y, x).
 *
 * @param f - The integrand f(y, x) (y a 1-D value of points, x a scalar value).
 * @param a - Lower x limit.
 * @param b - Upper x limit.
 * @param g - Lower y limit (number or function of x).
 * @param h - Upper y limit (number or function of x).
 * @param options - Options of the one-dimensional quadratures.
 * @returns The integral.
 * @example
 * ```ts
 * import { dblquad } from 'mdml-engine/numerics/quadrature'
 * import { item, mul } from 'mdml-engine/tensor'
 * // x y over the triangle 0 <= y <= x <= 1: 1/8.
 * Math.abs(item(dblquad((y, x) => mul(x, y), 0, 1, 0, (x) => x)) - 0.125) < 1e-12 // => true
 * ```
 */
export function dblquad(
  f: (y: Value, x: Value) => Value,
  a: number,
  b: number,
  g: number | ((x: number) => number),
  h: number | ((x: number) => number),
  options: QuadOptions = {},
): Value {
  const at = (v: number | ((x: number) => number), x: number): number => (typeof v === 'function' ? v(x) : v)
  return nquad(f, [(x: number) => [at(g, x), at(h, x)], [a, b]], options)
}

/**
 * A triple integral (scipy.integrate.tplquad): x in [a, b], y in [g(x), h(x)], z in [q(x, y), r(x, y)] of f(z, y, x).
 *
 * @param f - The integrand f(z, y, x).
 * @param a - Lower x limit.
 * @param b - Upper x limit.
 * @param g - Lower y limit.
 * @param h - Upper y limit.
 * @param q - Lower z limit (number or function of x and y).
 * @param r - Upper z limit.
 * @param options - Options of the one-dimensional quadratures.
 * @returns The integral.
 * @example
 * ```ts
 * import { tplquad } from 'mdml-engine/numerics/quadrature'
 * import { item, mul } from 'mdml-engine/tensor'
 * // x y z over the unit cube: 1/8.
 * Math.abs(item(tplquad((z, y, x) => mul(mul(z, y), x), 0, 1, 0, 1, 0, 1)) - 0.125) < 1e-12 // => true
 * ```
 */
export function tplquad(
  f: (z: Value, y: Value, x: Value) => Value,
  a: number,
  b: number,
  g: number | ((x: number) => number),
  h: number | ((x: number) => number),
  q: number | ((x: number, y: number) => number),
  r: number | ((x: number, y: number) => number),
  options: QuadOptions = {},
): Value {
  const at1 = (v: number | ((x: number) => number), x: number): number => (typeof v === 'function' ? v(x) : v)
  const at2 = (v: number | ((x: number, y: number) => number), x: number, y: number): number =>
    typeof v === 'function' ? v(x, y) : v
  // nquad passes the outer variables innermost-first: (y, x) for z, (x) for y.
  return nquad(
    f,
    [(y: number, x: number) => [at2(q, x, y), at2(r, x, y)], (x: number) => [at1(g, x), at1(h, x)], [a, b]],
    options,
  )
}

/** A cubature rule on R^d: points `[m, d]` and weights `[m]`. */
export interface CubatureRule {
  readonly points: Tensor
  readonly weights: Tensor
}

/** Map a rule on [-1, 1]^d to a box. */
function toBox(points: number[][], weights: number[], box: readonly (readonly [number, number])[]): CubatureRule {
  const d = box.length
  const m = weights.length
  const p = new Float64Array(m * d)
  const w = new Float64Array(m)
  let jac = 1
  for (const [lo, hi] of box) jac *= (hi - lo) / 2
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < d; k++) {
      const [lo, hi] = box[k] as readonly [number, number]
      p[i * d + k] = (lo + hi) / 2 + ((hi - lo) / 2) * (points[i]?.[k] ?? 0)
    }
    w[i] = (weights[i] ?? 0) * jac
  }
  return { points: fromStorage(p, [m, d], 'float64'), weights: fromStorage(w, [m], 'float64') }
}

function gaussOnInterval(n: number): { x: number[]; w: number[] } {
  const r = gaussLegendre(n)
  return { x: Array.from(r.nodes.data), w: Array.from(r.weights.data) }
}

/**
 * The tensor-product Gauss-Legendre rule on a box: n^d points, exact for polynomials of degree 2n - 1 in each variable.
 *
 * @param box - The interval of each variable.
 * @param n - Points per variable.
 * @returns The rule.
 * @example
 * ```ts
 * import { productGaussRule } from 'mdml-engine/numerics/quadrature'
 * productGaussRule([[0, 1], [0, 1]], 3).points.shape // => [9, 2]
 * ```
 */
export function productGaussRule(box: readonly (readonly [number, number])[], n: number): CubatureRule {
  const { x, w } = gaussOnInterval(n)
  let points: number[][] = [[]]
  let weights: number[] = [1]
  for (let k = 0; k < box.length; k++) {
    const nextP: number[][] = []
    const nextW: number[] = []
    points.forEach((p, i) => {
      x.forEach((xi, j) => {
        nextP.push([...p, xi])
        nextW.push((weights[i] ?? 0) * (w[j] ?? 0))
      })
    })
    points = nextP
    weights = nextW
  }
  return toBox(points, weights, box)
}

function binomial(n: number, k: number): number {
  let r = 1
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i
  return r
}

/**
 * A Smolyak sparse grid on a box: the combination of tensor products of Gauss-Legendre rules U_l (2l - 1 points) over
 * levels with q - d + 1 <= |l| <= q, q = level + d - 1, weighted (-1)^(q - |l|) C(d - 1, q - |l|). Far fewer points
 * than the full product in high dimension, exact for total-degree polynomials up to 2 level - 1.
 *
 * @param box - The interval of each variable.
 * @param level - The Smolyak level (>= 1).
 * @returns The rule (points may repeat across the combined grids).
 * @concept sparse-grids
 * @example
 * ```ts
 * import { sparseGridRule } from 'mdml-engine/numerics/quadrature'
 * Array.from(sparseGridRule([[0, 1], [0, 1], [0, 1]], 2).weights.data).reduce((a, b) => a + b) // => 1
 * ```
 */
export function sparseGridRule(box: readonly (readonly [number, number])[], level: number): CubatureRule {
  const d = box.length
  const q = level + d - 1
  const points: number[][] = []
  const weights: number[] = []
  const levels = (remaining: number, k: number, prefix: number[]): void => {
    if (k === d) {
      const total = prefix.reduce((s, v) => s + v, 0)
      if (total < q - d + 1 || total > q) return
      const coefficient = (-1) ** (q - total) * binomial(d - 1, q - total)
      let grid: { p: number[]; w: number }[] = [{ p: [], w: coefficient }]
      for (const l of prefix) {
        const { x, w } = gaussOnInterval(2 * l - 1)
        grid = grid.flatMap((g) => x.map((xi, j) => ({ p: [...g.p, xi], w: g.w * (w[j] ?? 0) })))
      }
      for (const g of grid) {
        points.push(g.p)
        weights.push(g.w)
      }
      return
    }
    for (let l = 1; l <= remaining - (d - k - 1); l++) levels(remaining - l, k + 1, [...prefix, l])
  }
  levels(q, 0, [])
  return toBox(points, weights, box)
}

/**
 * Apply a cubature rule: sum_i w_i f(p_i). Differentiable in f's parameters.
 *
 * @param f - The integrand on points `[m, d]`, returning `[m]`.
 * @param rule - The rule.
 * @returns The estimate.
 * @example
 * ```ts
 * import { cubature, productGaussRule } from 'mdml-engine/numerics/quadrature'
 * import { item, prod } from 'mdml-engine/tensor'
 * Math.abs(item(cubature((p) => prod(p, -1), productGaussRule([[0, 1], [0, 2]], 2))) - 1) < 1e-13 // => true
 * ```
 */
export function cubature(f: (points: Value) => Value, rule: CubatureRule): Value {
  return sum(mul(rule.weights, f(rule.points)))
}

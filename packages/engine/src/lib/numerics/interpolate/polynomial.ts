/**
 * Polynomial interpolation through points: the stable barycentric form (scipy's `BarycentricInterpolator`) and its
 * weights, the Lagrange power-basis form kept for teaching, Chebyshev and equispaced nodes, Chebyshev interpolants that
 * avoid the Runge phenomenon, Lebesgue constants, and Runge's function itself.
 */

import { maxOf, minOf, WhiteboxError } from '../../base/index.ts'
import {
  add,
  concreteOrThrow,
  div,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'

/** A barycentric polynomial interpolant. */
export interface Barycentric {
  readonly nodes: Tensor
  /** The barycentric weights w_j = 1 / prod_{k != j} (x_j - x_k), rescaled. */
  readonly weights: Tensor
  /** Values at query points `[q]` (or a scalar). */
  readonly evaluate: (x: TensorLike) => Value
}

/**
 * Polynomial interpolation in the second (true) barycentric form (scipy.interpolate.BarycentricInterpolator):
 * p(x) = sum_j (w_j y_j / (x - x_j)) / sum_j (w_j / (x - x_j)), stable and O(n) per point. Differentiable in the values
 * and the query points; exact at the nodes.
 *
 * @param x - Distinct nodes `[n]`.
 * @param y - Values `[n]`.
 * @param weights - Optional precomputed weights (for example Chebyshev weights).
 * @returns The interpolant.
 * @concept polynomial-interpolation
 * @example
 * ```ts
 * import { barycentric } from 'whitebox-engine/numerics/interpolate'
 * import { item } from 'whitebox-engine/tensor'
 * item(barycentric([0, 1, 2], [1, 3, 7]).evaluate(1.5)) // => 4.75
 * ```
 */
export function barycentric(x: TensorLike, y: TensorLike, weights?: readonly number[]): Barycentric {
  const xs = toFlat(concreteOrThrow(promoteFloat(x)[0] as Tensor, 'barycentric'))
  const [yv] = promoteFloat(y) as [Value]
  const n = xs.length
  if (yv.shape[0] !== n) throw new WhiteboxError('shape', 'barycentric: x and y must have the same length')
  let w = weights === undefined ? barycentricWeights(xs) : [...weights]
  const scale = maxOf(w.map(Math.abs))
  w = w.map((v) => v / scale)
  const nodes = fromStorage(Float64Array.from(xs), [n], 'float64')
  const wt = fromStorage(Float64Array.from(w), [n], 'float64')
  const evaluate = (query: TensorLike): Value => {
    const [q] = promoteFloat(query) as [Value]
    const qf = reshape(q, [-1, 1])
    const qp = toFlat(primalOf(qf))
    // Exact hits: use the data there (and a dummy difference of 1 to keep the arithmetic finite).
    const hit = new Float64Array(qp.length * n)
    const hitAny = new Uint8Array(qp.length)
    qp.forEach((v, r) => {
      xs.forEach((node, j) => {
        if (v === node) {
          hit[r * n + j] = 1
          hitAny[r] = 1
        }
      })
    })
    const hitMask = fromStorage(
      Uint8Array.from(hit, (v) => v),
      [qp.length, n],
      'bool',
    )
    const diff = where(hitMask, 1, sub(qf, reshape(nodes, [1, n])))
    const terms = div(reshape(wt, [1, n]), diff)
    const yrow = reshape(yv, [1, n])
    const smooth = div(sum(mul(terms, yrow), 1), sum(terms, 1))
    const exact = sum(mul(fromStorage(hit, [qp.length, n], 'float64'), yrow), 1)
    const out = where(fromStorage(hitAny, [qp.length], 'bool'), exact, smooth)
    return reshape(out, q.shape)
  }
  return { nodes, weights: wt, evaluate }
}

/** Barycentric weights 1 / prod_{k != j} (x_j - x_k), computed with a scaling that avoids overflow. */
export function barycentricWeights(x: readonly number[]): number[] {
  const n = x.length
  const span = maxOf(x) - minOf(x) || 1
  // Dividing each difference by span / 4 keeps the products in range (the capacity of the interval).
  const c = span / 4
  return x.map((xj, j) => {
    let p = 1
    for (let k = 0; k < n; k++) if (k !== j) p *= (xj - (x[k] ?? 0)) / c
    if (p === 0) throw new WhiteboxError('value', 'barycentricWeights: nodes must be distinct')
    return 1 / p
  })
}

/**
 * The Lagrange interpolating polynomial's power-basis coefficients, highest first (scipy.interpolate.lagrange):
 * numerically unstable beyond about 20 points, kept for teaching.
 *
 * @param x - Distinct nodes.
 * @param y - Values.
 * @returns Coefficients, highest power first.
 * @example
 * ```ts
 * import { lagrange } from 'whitebox-engine/numerics/interpolate'
 * lagrange([0, 1, 2], [1, 3, 7]) // => [1, 1, 1]
 * ```
 */
export function lagrange(x: readonly number[], y: readonly number[]): number[] {
  const n = x.length
  let total = new Array<number>(n).fill(0)
  for (let j = 0; j < n; j++) {
    // y_j prod_{k != j} (t - x_k) / (x_j - x_k), built by repeated multiplication by (t - x_k).
    let basis = [1]
    let denom = 1
    for (let k = 0; k < n; k++) {
      if (k === j) continue
      const next = new Array<number>(basis.length + 1).fill(0)
      basis.forEach((c, i) => {
        next[i] = (next[i] ?? 0) + c
        next[i + 1] = (next[i + 1] ?? 0) - c * (x[k] ?? 0)
      })
      basis = next
      denom *= (x[j] ?? 0) - (x[k] ?? 0)
    }
    const scale = (y[j] ?? 0) / denom
    total = total.map((t, i) => t + scale * (basis[i] ?? 0))
  }
  return total.map((v) => (Math.abs(v) < 1e-13 * Math.max(1, maxOf(total.map(Math.abs))) ? 0 : v))
}

/**
 * Chebyshev nodes on [a, b]: of the first kind cos((2j + 1) pi / 2n) (roots of T_n), or of the second kind
 * cos(j pi / (n - 1)) (extrema, including the ends), in increasing order.
 *
 * @param n - Number of nodes.
 * @param interval - The interval (default [-1, 1]).
 * @param kind - 1 or 2 (default 2).
 * @returns The nodes.
 * @example
 * ```ts
 * import { chebyshevNodes } from 'whitebox-engine/numerics/interpolate'
 * chebyshevNodes(3).map((v) => Math.round(v * 1e12) / 1e12) // => [-1, 0, 1]
 * ```
 */
export function chebyshevNodes(n: number, interval: readonly [number, number] = [-1, 1], kind: 1 | 2 = 2): number[] {
  const [a, b] = interval
  const nodes = Array.from({ length: n }, (_, j) =>
    kind === 1 ? Math.cos(((2 * j + 1) * Math.PI) / (2 * n)) : n === 1 ? 0 : Math.cos((j * Math.PI) / (n - 1)),
  )
  return nodes.map((t) => (a + b) / 2 + ((b - a) / 2) * t).reverse()
}

/** Equally spaced nodes on [a, b], including the ends. */
export function equispacedNodes(n: number, interval: readonly [number, number] = [-1, 1]): number[] {
  const [a, b] = interval
  return Array.from({ length: n }, (_, j) => (n === 1 ? (a + b) / 2 : a + ((b - a) * j) / (n - 1)))
}

/**
 * Interpolate a function at Chebyshev points of the second kind, with their closed-form barycentric weights
 * (-1)^j, halved at the ends: near-best polynomial approximation, free of the Runge phenomenon.
 *
 * @param f - The function on a 1-D value of points.
 * @param n - Number of points.
 * @param interval - The interval (default [-1, 1]).
 * @returns The interpolant.
 * @example
 * ```ts
 * import { chebyshevInterpolant } from 'whitebox-engine/numerics/interpolate'
 * import { exp, item } from 'whitebox-engine/tensor'
 * Math.abs(item(chebyshevInterpolant((x) => exp(x), 16).evaluate(0.3)) - Math.exp(0.3)) < 1e-14 // => true
 * ```
 */
export function chebyshevInterpolant(
  f: (x: Value) => Value,
  n: number,
  interval: readonly [number, number] = [-1, 1],
): Barycentric {
  const nodes = chebyshevNodes(n, interval, 2)
  const weights = nodes.map((_, j) => (j % 2 === 0 ? 1 : -1) * (j === 0 || j === n - 1 ? 0.5 : 1))
  return barycentric(nodes, f(fromStorage(Float64Array.from(nodes), [n], 'float64')), weights)
}

/**
 * The Lebesgue constant of a node set, max over x of sum_j |l_j(x)|, estimated on a fine grid: interpolation error is
 * at most (1 + Lambda) times the best approximation error. It grows like 2^n / (e n log n) for equispaced nodes and like
 * (2 / pi) log n for Chebyshev nodes.
 *
 * @param nodes - The nodes.
 * @param samples - Grid size for the maximum (default 2000).
 * @returns The estimate.
 * @example
 * ```ts
 * import { chebyshevNodes, lebesgueConstant } from 'whitebox-engine/numerics/interpolate'
 * lebesgueConstant(chebyshevNodes(10)) < 2.5 // => true
 * ```
 */
export function lebesgueConstant(nodes: readonly number[], samples = 2000): number {
  const w = barycentricWeights(nodes)
  const a = minOf(nodes)
  const b = maxOf(nodes)
  let best = 0
  for (let s = 0; s <= samples; s++) {
    const x = a + ((b - a) * s) / samples
    let num = 0
    let den = 0
    let hit = false
    for (let j = 0; j < nodes.length; j++) {
      const d = x - (nodes[j] ?? 0)
      if (d === 0) {
        hit = true
        break
      }
      num += Math.abs((w[j] ?? 0) / d)
      den += (w[j] ?? 0) / d
    }
    best = Math.max(best, hit ? 1 : num / Math.abs(den))
  }
  return best
}

/**
 * Runge's function 1 / (1 + 25 x^2): equispaced polynomial interpolation of it diverges near the ends as n grows.
 *
 * @param x - Points.
 * @returns Values.
 * @example
 * ```ts
 * import { runge } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(runge([0, 0.2])) // => [1, 0.5]
 * ```
 */
export function runge(x: TensorLike): Value {
  const [v] = promoteFloat(x) as [Value]
  return div(1, add(mul(mul(v, v), 25), 1))
}

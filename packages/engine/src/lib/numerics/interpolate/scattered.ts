/**
 * Interpolation of scattered data in any dimension: radial basis functions with a polynomial tail (scipy's
 * `RBFInterpolator`), Shepard's inverse-distance weighting, and Sibson's natural-neighbour interpolation of planar data
 * from Voronoi cell areas.
 */

import { maxOf, minOf, WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  concreteOrThrow,
  div,
  exp,
  fromStorage,
  log,
  matmul,
  mul,
  neg,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { circumcentre, delaunay, type Point2, toPoints } from '../geometry/index.ts'
import { solve } from '../linalg/index.ts'

/** Radial basis kernels of scipy.interpolate.RBFInterpolator, as functions of r = epsilon * distance. */
export type RbfKernel =
  | 'linear'
  | 'thin_plate_spline'
  | 'cubic'
  | 'quintic'
  | 'multiquadric'
  | 'inverse_multiquadric'
  | 'inverse_quadratic'
  | 'gaussian'

const MIN_DEGREE: Readonly<Record<RbfKernel, number>> = {
  linear: 0,
  thin_plate_spline: 1,
  cubic: 1,
  quintic: 2,
  multiquadric: 0,
  inverse_multiquadric: -1,
  inverse_quadratic: -1,
  gaussian: -1,
}
const SCALE_INVARIANT = new Set<RbfKernel>(['linear', 'thin_plate_spline', 'cubic', 'quintic'])

/** The kernel on squared scaled distances (written on r^2 so that it is smooth where it should be). */
function kernelOf(name: RbfKernel, r2: Value): Value {
  const zero = fromStorage(
    Uint8Array.from(toFlat(primalOf(r2)), (v) => (v === 0 ? 1 : 0)),
    r2.shape,
    'bool',
  )
  const safe = where(zero, 1, r2)
  switch (name) {
    case 'linear':
      return where(zero, 0, neg(sqrt(safe)))
    case 'thin_plate_spline':
      return where(zero, 0, mul(mul(safe, log(safe)), 0.5))
    case 'cubic':
      return where(zero, 0, pow(safe, 1.5))
    case 'quintic':
      return where(zero, 0, neg(pow(safe, 2.5)))
    case 'multiquadric':
      return neg(sqrt(add(r2, 1)))
    case 'inverse_multiquadric':
      return div(1, sqrt(add(r2, 1)))
    case 'inverse_quadratic':
      return div(1, add(r2, 1))
    case 'gaussian':
      return exp(neg(r2))
  }
}

/** Exponent tuples of all monomials in d variables of total degree <= degree. */
function monomials(d: number, degree: number): number[][] {
  const out: number[][] = []
  const rec = (prefix: number[], left: number): void => {
    if (prefix.length === d) {
      out.push(prefix)
      return
    }
    for (let e = 0; e <= left; e++) rec([...prefix, e], left - e)
  }
  if (degree >= 0) rec([], degree)
  return out.sort((a, b) => a.reduce((s, v) => s + v, 0) - b.reduce((s, v) => s + v, 0))
}

/** Options of {@link rbfInterpolator}. */
export interface RbfOptions {
  /** The kernel (default thin_plate_spline). */
  readonly kernel?: RbfKernel
  /** The shape parameter (default 1; required for kernels that are not scale invariant). */
  readonly epsilon?: number
  /** Smoothing (0 interpolates). */
  readonly smoothing?: number
  /** Degree of the added polynomial (default max(minimum degree of the kernel, 0), as scipy; -1 for none). */
  readonly degree?: number
}

/**
 * Radial basis function interpolation (scipy.interpolate.RBFInterpolator, global): f(x) = sum_i a_i phi(eps |x - y_i|)
 * + p(x), with the polynomial p of the chosen degree and the side conditions P^T a = 0; the coefficients solve
 * [[K + smoothing I, P], [P^T, 0]] [a; b] = [d; 0]. Monomials use scipy's shifted and scaled coordinates. Linear in
 * the data values and differentiable in the query points.
 *
 * @param points - Data sites `[n, d]`.
 * @param values - Data values `[n, ...]`.
 * @param options - Kernel, shape parameter, smoothing and polynomial degree.
 * @returns The interpolating function on query points `[q, d]`.
 * @concept radial-basis-functions
 * @example
 * ```ts
 * import { rbfInterpolator } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * // Thin-plate splines with a linear term reproduce linear functions.
 * const f = rbfInterpolator([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.3]], [0, 1, 2, 3, 1.1])
 * toArray(f([[0.25, 0.75]])).map((v) => Math.round(v * 1e10) / 1e10) // => [1.75]
 * ```
 */
export function rbfInterpolator(
  points: TensorLike,
  values: TensorLike,
  options: RbfOptions = {},
): (query: TensorLike) => Value {
  const y = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'rbfInterpolator')
  if (y.shape.length !== 2) throw new WhiteboxError('shape', 'rbfInterpolator: points must be [n, d]')
  const [v] = promoteFloat(values) as [Value]
  const n = y.shape[0] ?? 0
  const d = y.shape[1] ?? 0
  if (v.shape[0] !== n) throw new WhiteboxError('shape', 'rbfInterpolator: one value per point')
  const kernel = options.kernel ?? 'thin_plate_spline'
  if (!SCALE_INVARIANT.has(kernel) && options.epsilon === undefined)
    throw new WhiteboxError('value', `rbfInterpolator: kernel ${kernel} needs epsilon`)
  const epsilon = options.epsilon ?? 1
  const minDegree = MIN_DEGREE[kernel]
  const degree = options.degree ?? Math.max(minDegree, 0)
  if (degree !== -1 && degree < minDegree)
    throw new WhiteboxError('value', `rbfInterpolator: kernel ${kernel} needs degree >= ${minDegree}`)
  const powers = monomials(d, degree)
  const yf = toFlat(y)
  const mins = Array.from({ length: d }, (_, k) => minOf(Array.from({ length: n }, (_, i) => yf[i * d + k] ?? 0)))
  const maxs = Array.from({ length: d }, (_, k) => maxOf(Array.from({ length: n }, (_, i) => yf[i * d + k] ?? 0)))
  const shift = mins.map((lo, k) => (lo + (maxs[k] ?? 0)) / 2)
  const scale = mins.map((lo, k) => ((maxs[k] ?? 0) - lo) / 2 || 1)
  const polynomial = (x: Value): Value => {
    const m = x.shape[0] ?? 0
    const scaled = div(
      sub(x, reshape(fromStorage(Float64Array.from(shift), [d], 'float64'), [1, d])),
      reshape(fromStorage(Float64Array.from(scale), [d], 'float64'), [1, d]),
    )
    const columns = powers.map((ex) => {
      let col: Value = fromStorage(new Float64Array(m).fill(1), [m], 'float64')
      ex.forEach((e, k) => {
        if (e > 0) col = mul(col, pow(reshape(sliceColumn(scaled, k), [m]), e))
      })
      return reshape(col, [m, 1])
    })
    return columns.length === 0 ? fromStorage(new Float64Array(0), [m, 0], 'float64') : concat(columns, 1)
  }
  const sqDist = (a: Value, b: Value): Value => {
    const diff = sub(reshape(a, [-1, 1, d]), reshape(b, [1, -1, d]))
    return mul(sum(mul(diff, diff), 2), epsilon * epsilon)
  }
  const k = toFlat(kernelOf(kernel, sqDist(y, y)))
  const p = toFlat(polynomial(y))
  const np = powers.length
  const size = n + np
  const a = new Float64Array(size * size)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) a[i * size + j] = k[i * n + j] ?? 0
    a[i * size + i] = (a[i * size + i] ?? 0) + (options.smoothing ?? 0)
    for (let j = 0; j < np; j++) {
      a[i * size + n + j] = p[i * np + j] ?? 0
      a[(n + j) * size + i] = p[i * np + j] ?? 0
    }
  }
  const trailing = v.shape.slice(1)
  const flat = reshape(v, [n, -1])
  const rhs = concat(
    [flat, fromStorage(new Float64Array(np * (flat.shape[1] ?? 1)), [np, flat.shape[1] ?? 1], 'float64')],
    0,
  )
  const coefficients = solve(fromStorage(a, [size, size], 'float64'), rhs)
  const weights = sliceRows(coefficients, 0, n)
  const poly = sliceRows(coefficients, n, size)
  return (query) => {
    const [q0] = promoteFloat(query) as [Value]
    const q = q0.shape.length === 1 ? reshape(q0, [1, d]) : q0
    const m = q.shape[0] ?? 0
    let out: Value = matmul(kernelOf(kernel, sqDist(q, y)), weights)
    if (np > 0) out = add(out, matmul(polynomial(q), poly))
    const shaped = reshape(out, [m, ...trailing])
    return q0.shape.length === 1 ? reshape(shaped, trailing) : shaped
  }
}

function sliceColumn(x: Value, k: number): Value {
  return sliceAxis(x, 1, k, k + 1)
}

function sliceRows(x: Value, start: number, stop: number): Value {
  return sliceAxis(x, 0, start, stop)
}

/**
 * Inverse-distance weighting (Shepard): f(x) = sum_i w_i v_i / sum_i w_i with w_i = |x - y_i|^-p, exact at the data
 * sites. Linear in the values.
 *
 * @param points - Data sites `[n, d]`.
 * @param values - Values `[n]`.
 * @param power - The exponent p (default 2).
 * @returns The interpolating function on query points `[q, d]`.
 * @example
 * ```ts
 * import { inverseDistanceWeighting } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(inverseDistanceWeighting([[0, 0], [2, 0]], [1, 3])([[1, 0], [0, 0]])) // => [2, 1]
 * ```
 */
export function inverseDistanceWeighting(
  points: TensorLike,
  values: TensorLike,
  power = 2,
): (query: TensorLike) => Value {
  const y = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'inverseDistanceWeighting')
  const [v] = promoteFloat(values) as [Value]
  const n = y.shape[0] ?? 0
  const d = y.shape[1] ?? 0
  return (query) => {
    const [q] = promoteFloat(query) as [Value]
    const m = q.shape[0] ?? 0
    const diff = sub(reshape(q, [m, 1, d]), reshape(y, [1, n, d]))
    const d2 = sum(mul(diff, diff), 2)
    const d2p = toFlat(primalOf(d2))
    const hits = new Float64Array(m * n)
    const anyHit = new Uint8Array(m)
    d2p.forEach((val, idx) => {
      if (val === 0) {
        hits[idx] = 1
        anyHit[Math.floor(idx / n)] = 1
      }
    })
    const zero = fromStorage(Uint8Array.from(hits), [m, n], 'bool')
    const w = where(zero, 0, pow(where(zero, 1, d2), -power / 2))
    const row = reshape(v, [1, n])
    const smooth = div(sum(mul(w, row), 1), sum(w, 1))
    const exact = div(
      sum(mul(fromStorage(hits, [m, n], 'float64'), row), 1),
      add(sum(fromStorage(hits, [m, n], 'float64'), 1), 1e-300),
    )
    return where(fromStorage(anyHit, [m], 'bool'), exact, smooth)
  }
}

/** Twice the area of a polygon given by its vertices in any order around a convex shape (sorted by angle). */
function convexArea(vertices: readonly Point2[]): number {
  const cx = vertices.reduce((s, p) => s + p[0], 0) / vertices.length
  const cy = vertices.reduce((s, p) => s + p[1], 0) / vertices.length
  const sorted = [...vertices].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx))
  let area = 0
  for (let i = 0; i < sorted.length; i++) {
    const p = sorted[i] as Point2
    const q = sorted[(i + 1) % sorted.length] as Point2
    area += p[0] * q[1] - q[0] * p[1]
  }
  return Math.abs(area) / 2
}

/**
 * Sibson's natural-neighbour coordinates of a query point among planar data sites: the share of the area its Voronoi
 * cell would take from each site's cell if it were inserted. The weights are non-negative, sum to 1, reproduce linear
 * functions, and vary continuously (smoothly away from the sites). NaN outside the convex hull.
 *
 * @param points - Data sites `[n, 2]`.
 * @param query - Query points `[q, 2]`.
 * @returns The weights `[q, n]`.
 * @concept natural-neighbour-interpolation
 * @example
 * ```ts
 * import { naturalNeighbourWeights } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(naturalNeighbourWeights([[0, 0], [2, 0], [0, 2], [2, 2]], [[1, 1]]))[0].map((v) => Math.round(v * 1e12) / 1e12) // => [0.25, 0.25, 0.25, 0.25]
 * ```
 */
export function naturalNeighbourWeights(
  points: TensorLike | readonly Point2[],
  query: TensorLike | readonly Point2[],
): Tensor {
  const tri = delaunay(points)
  const pts = tri.points
  const n = pts.length
  const qs = toPoints(query)
  const out = new Float64Array(qs.length * n)
  qs.forEach((q, r) => {
    const exact = pts.findIndex((p) => p[0] === q[0] && p[1] === q[1])
    if (exact >= 0) {
      out[r * n + exact] = 1
      return
    }
    if (tri.findSimplex(q) === -1) {
      out.fill(Number.NaN, r * n, (r + 1) * n)
      return
    }
    // The cavity: triangles whose circumcircles contain q.
    const cavity = tri.simplices
      .map((t, i) => ({ t, i, c: circumcentre(pts[t[0]] as Point2, pts[t[1]] as Point2, pts[t[2]] as Point2) }))
      .filter(({ t, c }) => {
        const a = pts[t[0]] as Point2
        return (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2 < (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
      })
    // Its boundary edges (counter-clockwise), chained into the cycle of natural neighbours.
    const edges = new Map<string, [number, number]>()
    for (const { t } of cavity) {
      for (let v = 0; v < 3; v++) {
        const a = t[v] as number
        const b = t[(v + 1) % 3] as number
        const reverse = `${b},${a}`
        if (edges.has(reverse)) edges.delete(reverse)
        else edges.set(`${a},${b}`, [a, b])
      }
    }
    const next = new Map<number, number>()
    for (const [a, b] of edges.values()) next.set(a, b)
    const ring: number[] = []
    let start = edges.values().next().value?.[0] ?? 0
    for (let guard = 0; guard <= edges.size; guard++) {
      ring.push(start)
      start = next.get(start) ?? start
      if (start === ring[0]) break
    }
    const k = ring.length
    const g = ring.map((b, j) => circumcentre(q, pts[b] as Point2, pts[ring[(j + 1) % k] as number] as Point2))
    let total = 0
    const areas = ring.map((b, j) => {
      const own = cavity.filter(({ t }) => t.includes(b)).map(({ c }) => c)
      const area = convexArea([g[(j - 1 + k) % k] as Point2, ...own, g[j] as Point2])
      total += area
      return area
    })
    ring.forEach((b, j) => {
      out[r * n + b] = (areas[j] ?? 0) / total
    })
  })
  return fromStorage(out, [qs.length, n], 'float64')
}

/**
 * Natural-neighbour (Sibson) interpolation of planar scattered data: the values weighted by
 * {@link naturalNeighbourWeights}. Linear in the values; NaN outside the convex hull.
 *
 * @param points - Data sites `[n, 2]`.
 * @param values - Values `[n]`.
 * @returns The interpolating function on query points `[q, 2]`.
 * @example
 * ```ts
 * import { naturalNeighbour } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(naturalNeighbour([[0, 0], [2, 0], [0, 2], [2, 2]], [0, 2, 4, 6])([[1, 1]])).map((v) => Math.round(v * 1e12) / 1e12) // => [3]
 * ```
 */
export function naturalNeighbour(
  points: TensorLike | readonly Point2[],
  values: TensorLike,
): (query: TensorLike | readonly Point2[]) => Value {
  const [v] = promoteFloat(values) as [Value]
  return (query) => matmul(naturalNeighbourWeights(points, query), v)
}

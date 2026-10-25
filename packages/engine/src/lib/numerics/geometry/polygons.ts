/**
 * Polygons and convex hulls: planar hulls by Andrew's monotone chain and 3-D hulls by the incremental algorithm (as
 * scipy's `ConvexHull`), polygon area, centroid and perimeter, convexity and winding-number point-in-polygon tests,
 * segment intersection, Sutherland-Hodgman clipping, and Minkowski sums of convex polygons.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  concreteOrThrow,
  div,
  mul,
  promoteFloat,
  reshape,
  roll,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { orient, type Point2, toPoints } from './triangulation.ts'

/** A point in space. */
export type Point3 = readonly [number, number, number]

/**
 * The convex hull of planar points by Andrew's monotone chain (scipy.spatial.ConvexHull in 2-D): vertex indices in
 * counter-clockwise order, starting from the lowest-leftmost point; collinear boundary points are dropped.
 *
 * @param points - Points `[n, 2]`.
 * @returns Hull vertex indices.
 * @concept convex-hull
 * @example
 * ```ts
 * import { convexHull } from 'mdml-engine/numerics/geometry'
 * convexHull([[0, 0], [1, 1], [2, 0], [1, 0.5], [1, -1]]) // => [4, 2, 1, 0]
 * ```
 */
export function convexHull(points: TensorLike | readonly Point2[]): number[] {
  const pts = toPoints(points)
  if (pts.length < 3) return pts.map((_, i) => i)
  const order = pts
    .map((_, i) => i)
    .sort((a, b) => (pts[a]?.[0] ?? 0) - (pts[b]?.[0] ?? 0) || (pts[a]?.[1] ?? 0) - (pts[b]?.[1] ?? 0))
  const at = (i: number): Point2 => pts[i] as Point2
  const chain = (indices: readonly number[]): number[] => {
    const out: number[] = []
    for (const i of indices) {
      while (
        out.length >= 2 &&
        orient(at(out[out.length - 2] as number), at(out[out.length - 1] as number), at(i)) <= 0
      )
        out.pop()
      out.push(i)
    }
    return out
  }
  const lower = chain(order)
  const upper = chain([...order].reverse())
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)]
  // Start from the lowest point (then leftmost), as a canonical order.
  let start = 0
  hull.forEach((v, i) => {
    const p = at(v)
    const q = at(hull[start] as number)
    if (p[1] < q[1] || (p[1] === q[1] && p[0] < q[0])) start = i
  })
  return [...hull.slice(start), ...hull.slice(0, start)]
}

/** A triangulated convex hull in three dimensions. */
export interface ConvexHull3 {
  /** Faces as vertex-index triples, oriented outwards (counter-clockwise seen from outside). */
  readonly faces: readonly (readonly [number, number, number])[]
  /** The hull's vertices, sorted. */
  readonly vertices: readonly number[]
  readonly volume: number
  readonly area: number
}

/**
 * The convex hull of points in space by the incremental algorithm: start from a tetrahedron, and for each point outside
 * the current hull remove the faces it sees and connect it to the horizon (scipy.spatial.ConvexHull in 3-D).
 *
 * @param points - Points `[n, 3]`, not all coplanar.
 * @returns Faces, vertices, volume and surface area.
 * @example
 * ```ts
 * import { convexHull3 } from 'mdml-engine/numerics/geometry'
 * convexHull3([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]]).volume // => 0.16666666666666666
 * ```
 */
export function convexHull3(points: TensorLike | readonly Point3[]): ConvexHull3 {
  const pts: Point3[] =
    Array.isArray(points) && Array.isArray(points[0])
      ? (points as readonly Point3[]).map((p) => [p[0], p[1], p[2]])
      : (() => {
          const t = concreteOrThrow(promoteFloat(points as TensorLike)[0] as Tensor, 'convexHull3')
          if (t.shape.length !== 2 || t.shape[1] !== 3)
            throw new MdmlError('shape', 'convexHull3: points must be [n, 3]')
          const f = toFlat(t)
          return Array.from(
            { length: t.shape[0] ?? 0 },
            (_, i): Point3 => [f[3 * i] ?? 0, f[3 * i + 1] ?? 0, f[3 * i + 2] ?? 0],
          )
        })()
  const n = pts.length
  const sub3 = (a: Point3, b: Point3): Point3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
  const cross = (a: Point3, b: Point3): Point3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
  const dot3 = (a: Point3, b: Point3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const at = (i: number): Point3 => pts[i] as Point3
  const volume6 = (a: number, b: number, c: number, d: number): number =>
    dot3(cross(sub3(at(b), at(a)), sub3(at(c), at(a))), sub3(at(d), at(a)))
  const scale = Math.max(1e-300, ...pts.flatMap((p) => p.map(Math.abs)))
  const eps = 1e-12 * scale ** 3
  // An initial tetrahedron from extreme points.
  let i0 = 0
  for (let i = 1; i < n; i++) if (at(i)[0] < at(i0)[0]) i0 = i
  let i1 = i0 === 0 ? 1 : 0
  for (let i = 0; i < n; i++) {
    const d = sub3(at(i), at(i0))
    const e = sub3(at(i1), at(i0))
    if (dot3(d, d) > dot3(e, e)) i1 = i
  }
  let i2 = -1
  let best = 0
  for (let i = 0; i < n; i++) {
    const c = cross(sub3(at(i1), at(i0)), sub3(at(i), at(i0)))
    if (dot3(c, c) > best) {
      best = dot3(c, c)
      i2 = i
    }
  }
  let i3 = -1
  best = 0
  for (let i = 0; i < n; i++) {
    const v = Math.abs(volume6(i0, i1, i2, i))
    if (v > best) {
      best = v
      i3 = i
    }
  }
  if (i2 < 0 || i3 < 0 || best <= eps) throw new MdmlError('value', 'convexHull3: the points are coplanar')
  type Face = [number, number, number]
  const outward = (f: Face, inside: number): Face => (volume6(f[0], f[1], f[2], inside) > 0 ? [f[0], f[2], f[1]] : f)
  let faces: Face[] = [
    outward([i0, i1, i2], i3),
    outward([i0, i1, i3], i2),
    outward([i0, i2, i3], i1),
    outward([i1, i2, i3], i0),
  ]
  for (let p = 0; p < n; p++) {
    if (p === i0 || p === i1 || p === i2 || p === i3) continue
    const visible = faces.map((f) => volume6(f[0], f[1], f[2], p) > eps)
    if (!visible.some(Boolean)) continue
    // Horizon: directed edges of visible faces whose reverse is not on a visible face.
    const edges = new Set<string>()
    faces.forEach((f, k) => {
      if (!visible[k]) return
      for (let e = 0; e < 3; e++) edges.add(`${f[e]},${f[(e + 1) % 3]}`)
    })
    const kept = faces.filter((_, k) => !visible[k])
    for (const key of edges) {
      const [a, b] = key.split(',').map(Number) as [number, number]
      if (!edges.has(`${b},${a}`)) kept.push([a, b, p])
    }
    faces = kept
  }
  let volume = 0
  let area = 0
  const origin = at(faces[0]?.[0] ?? 0)
  for (const [a, b, c] of faces) {
    const cr = cross(sub3(at(b), at(a)), sub3(at(c), at(a)))
    area += Math.sqrt(dot3(cr, cr)) / 2
    volume += dot3(cross(sub3(at(b), origin), sub3(at(c), origin)), sub3(at(a), origin)) / 6
  }
  return { faces, vertices: [...new Set(faces.flat())].sort((x, y) => x - y), volume: Math.abs(volume), area }
}

function vertices(polygon: TensorLike): Value {
  const [v] = promoteFloat(polygon) as [Value]
  if (v.shape.length !== 2 || v.shape[1] !== 2) throw new MdmlError('shape', 'polygon vertices must be [n, 2]')
  return v
}

/**
 * The signed area of a simple polygon by the shoelace formula: positive for counter-clockwise vertices. Built from
 * operations, so differentiable in the vertices.
 *
 * @param polygon - Vertices `[n, 2]` in order (not repeating the first).
 * @returns The signed area.
 * @example
 * ```ts
 * import { polygonArea } from 'mdml-engine/numerics/geometry'
 * import { item } from 'mdml-engine/tensor'
 * item(polygonArea([[0, 0], [2, 0], [2, 1], [0, 1]])) // => 2
 * ```
 */
export function polygonArea(polygon: TensorLike): Value {
  const v = vertices(polygon)
  const x = reshape(sliceAxis(v, 1, 0, 1), [-1])
  const y = reshape(sliceAxis(v, 1, 1, 2), [-1])
  return mul(sum(sub(mul(x, roll(y, -1)), mul(roll(x, -1), y))), 0.5)
}

/**
 * The centroid of a simple polygon (of its area, not its vertices), differentiable in the vertices.
 *
 * @param polygon - Vertices `[n, 2]`.
 * @returns The centroid `[2]`.
 * @example
 * ```ts
 * import { polygonCentroid } from 'mdml-engine/numerics/geometry'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(polygonCentroid([[0, 0], [4, 0], [4, 2], [0, 2]])) // => [2, 1]
 * ```
 */
export function polygonCentroid(polygon: TensorLike): Value {
  const v = vertices(polygon)
  const x = reshape(sliceAxis(v, 1, 0, 1), [-1])
  const y = reshape(sliceAxis(v, 1, 1, 2), [-1])
  const xn = roll(x, -1)
  const yn = roll(y, -1)
  const cross = sub(mul(x, yn), mul(xn, y))
  const a6 = mul(sum(cross), 3)
  const cx = div(sum(mul(add(x, xn), cross)), a6)
  const cy = div(sum(mul(add(y, yn), cross)), a6)
  return stack([cx, cy])
}

/**
 * The perimeter of a polygon.
 *
 * @param polygon - Vertices `[n, 2]`.
 * @returns The perimeter.
 * @example
 * ```ts
 * import { polygonPerimeter } from 'mdml-engine/numerics/geometry'
 * import { item } from 'mdml-engine/tensor'
 * item(polygonPerimeter([[0, 0], [3, 0], [3, 4]])) // => 12
 * ```
 */
export function polygonPerimeter(polygon: TensorLike): Value {
  const v = vertices(polygon)
  const d = sub(roll(v, -1, 0), v)
  return sum(sqrt(sum(mul(d, d), 1)))
}

/** Whether a polygon is convex (all turns the same way). */
export function isConvex(polygon: TensorLike | readonly Point2[]): boolean {
  const p = toPoints(polygon)
  let sign = 0
  for (let i = 0; i < p.length; i++) {
    const o = orient(p[i] as Point2, p[(i + 1) % p.length] as Point2, p[(i + 2) % p.length] as Point2)
    if (o === 0) continue
    if (sign === 0) sign = Math.sign(o)
    else if (Math.sign(o) !== sign) return false
  }
  return true
}

/**
 * The winding number of a closed polygon around each point (Sunday's algorithm): 0 outside; +1 inside a
 * counter-clockwise loop.
 *
 * @param points - Query points `[q, 2]`.
 * @param polygon - Vertices `[n, 2]`.
 * @returns Winding numbers.
 * @example
 * ```ts
 * import { windingNumber } from 'mdml-engine/numerics/geometry'
 * windingNumber([[0.5, 0.5], [2, 2]], [[0, 0], [1, 0], [1, 1], [0, 1]]) // => [1, 0]
 * ```
 */
export function windingNumber(
  points: TensorLike | readonly Point2[],
  polygon: TensorLike | readonly Point2[],
): number[] {
  const qs = toPoints(points)
  const poly = toPoints(polygon)
  return qs.map((q) => {
    let w = 0
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i] as Point2
      const b = poly[(i + 1) % poly.length] as Point2
      if (a[1] <= q[1]) {
        if (b[1] > q[1] && orient(a, b, q) > 0) w++
      } else if (b[1] <= q[1] && orient(a, b, q) < 0) w--
    }
    return w
  })
}

/** Whether q lies on segment ab (within a relative tolerance). */
function onSegment(a: Point2, b: Point2, q: Point2, tol: number): boolean {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1])
  if (Math.abs(orient(a, b, q)) > tol * Math.max(len, 1) * Math.max(len, 1)) return false
  return (
    q[0] >= Math.min(a[0], b[0]) - tol &&
    q[0] <= Math.max(a[0], b[0]) + tol &&
    q[1] >= Math.min(a[1], b[1]) - tol &&
    q[1] <= Math.max(a[1], b[1]) + tol
  )
}

/**
 * Point-in-polygon classification: 1 inside, 0 on the boundary, -1 outside (non-zero winding rule, so self-overlapping
 * loops count as inside).
 *
 * @param points - Query points `[q, 2]`.
 * @param polygon - Vertices `[n, 2]`.
 * @param tol - Boundary tolerance (default 1e-12).
 * @returns The classification per point.
 * @example
 * ```ts
 * import { pointInPolygon } from 'mdml-engine/numerics/geometry'
 * pointInPolygon([[0.5, 0.5], [1, 0.5], [3, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]]) // => [1, 0, -1]
 * ```
 */
export function pointInPolygon(
  points: TensorLike | readonly Point2[],
  polygon: TensorLike | readonly Point2[],
  tol = 1e-12,
): number[] {
  const qs = toPoints(points)
  const poly = toPoints(polygon)
  const w = windingNumber(qs, poly)
  return qs.map((q, k) => {
    for (let i = 0; i < poly.length; i++)
      if (onSegment(poly[i] as Point2, poly[(i + 1) % poly.length] as Point2, q, tol)) return 0
    return w[k] !== 0 ? 1 : -1
  })
}

/**
 * The intersection point of segments ab and cd, or null (parallel or not meeting).
 *
 * @param a - First segment start.
 * @param b - First segment end.
 * @param c - Second segment start.
 * @param d - Second segment end.
 * @returns The intersection.
 * @example
 * ```ts
 * import { segmentIntersection } from 'mdml-engine/numerics/geometry'
 * segmentIntersection([0, 0], [2, 2], [0, 2], [2, 0]) // => [1, 1]
 * ```
 */
export function segmentIntersection(a: Point2, b: Point2, c: Point2, d: Point2): [number, number] | null {
  const r: Point2 = [b[0] - a[0], b[1] - a[1]]
  const s: Point2 = [d[0] - c[0], d[1] - c[1]]
  const denom = r[0] * s[1] - r[1] * s[0]
  if (denom === 0) return null
  const t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
  const u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / denom
  if (t < 0 || t > 1 || u < 0 || u > 1) return null
  return [a[0] + t * r[0], a[1] + t * r[1]]
}

/**
 * Clip a polygon by a convex polygon (Sutherland-Hodgman): the part of the subject inside the clip region, as one
 * polygon (a non-convex subject whose intersection has several parts is returned joined along the clip boundary).
 *
 * @param subject - Vertices `[n, 2]`.
 * @param clip - Convex clip polygon `[m, 2]`, either orientation.
 * @returns The clipped vertices (empty when disjoint).
 * @example
 * ```ts
 * import { polygonClip } from 'mdml-engine/numerics/geometry'
 * polygonClip([[0, 0], [2, 0], [2, 2], [0, 2]], [[1, 1], [3, 1], [3, 3], [1, 3]]) // => [[1, 1], [2, 1], [2, 2], [1, 2]]
 * ```
 */
export function polygonClip(
  subject: TensorLike | readonly Point2[],
  clip: TensorLike | readonly Point2[],
): [number, number][] {
  let output: Point2[] = toPoints(subject)
  let window = toPoints(clip)
  if (!isConvex(window)) throw new MdmlError('value', 'polygonClip: the clip polygon must be convex')
  let area = 0
  for (let i = 0; i < window.length; i++) {
    const p = window[i] as Point2
    const q = window[(i + 1) % window.length] as Point2
    area += p[0] * q[1] - q[0] * p[1]
  }
  if (area < 0) window = [...window].reverse()
  for (let i = 0; i < window.length && output.length > 0; i++) {
    const a = window[i] as Point2
    const b = window[(i + 1) % window.length] as Point2
    const inside = (p: Point2): boolean => orient(a, b, p) >= 0
    const crossing = (p: Point2, q: Point2): Point2 => {
      const r: Point2 = [q[0] - p[0], q[1] - p[1]]
      const e: Point2 = [b[0] - a[0], b[1] - a[1]]
      const t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / (r[0] * e[1] - r[1] * e[0])
      return [p[0] + t * r[0], p[1] + t * r[1]]
    }
    const input = output
    output = []
    input.forEach((current, k) => {
      const previous = input[(k + input.length - 1) % input.length] as Point2
      if (inside(current)) {
        if (!inside(previous)) output.push(crossing(previous, current))
        output.push(current)
      } else if (inside(previous)) output.push(crossing(previous, current))
    })
  }
  return output.map((p) => [p[0], p[1]])
}

/**
 * The Minkowski sum of two convex polygons by merging their edges in angular order (O(n + m)): the set
 * {p + q : p in P, q in Q}, as a counter-clockwise polygon.
 *
 * @param p - Convex polygon `[n, 2]`.
 * @param q - Convex polygon `[m, 2]`.
 * @returns The sum's vertices, counter-clockwise from the lowest point.
 * @example
 * ```ts
 * import { minkowskiSum } from 'mdml-engine/numerics/geometry'
 * minkowskiSum([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [1, 1], [0, 1]]).length // => 5
 * ```
 */
export function minkowskiSum(p: TensorLike | readonly Point2[], q: TensorLike | readonly Point2[]): [number, number][] {
  const prepare = (poly: Point2[]): Point2[] => {
    const hull = convexHull(poly)
    return hull.map((i) => poly[i] as Point2)
  }
  const a = prepare(toPoints(p))
  const b = prepare(toPoints(q))
  const out: [number, number][] = []
  let i = 0
  let j = 0
  while (i < a.length || j < b.length) {
    const pa = a[i % a.length] as Point2
    const pb = b[j % b.length] as Point2
    out.push([pa[0] + pb[0], pa[1] + pb[1]])
    const ea: Point2 = [(a[(i + 1) % a.length] as Point2)[0] - pa[0], (a[(i + 1) % a.length] as Point2)[1] - pa[1]]
    const eb: Point2 = [(b[(j + 1) % b.length] as Point2)[0] - pb[0], (b[(j + 1) % b.length] as Point2)[1] - pb[1]]
    const c = ea[0] * eb[1] - ea[1] * eb[0]
    if (j >= b.length || (i < a.length && c > 0)) i++
    else if (i >= a.length || c < 0) j++
    else {
      i++
      j++
    }
  }
  // Drop collinear points left by parallel edges.
  return out.filter(
    (v, k) => orient(out[(k + out.length - 1) % out.length] as Point2, v, out[(k + 1) % out.length] as Point2) !== 0,
  )
}

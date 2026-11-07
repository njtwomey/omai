/**
 * Planar triangulation and its relatives: the orientation and in-circle predicates, Delaunay triangulation by the
 * Bowyer-Watson algorithm (as scipy's `Delaunay`), barycentric coordinates, Voronoi diagrams as the Delaunay dual, and
 * alpha shapes, the boundary of the triangles below a circumradius.
 */

import { MdmlError, type Vec2 } from '../../base/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'

/**
 * A point in the plane, `[x, y]`: the position type of polygons, triangulations, contours, outlines, rasterisation and
 * the interaction overlays of mdml-render, so a signature says it takes a place and not any pair. A `Vec2` of `base`.
 */
export type Point2 = Vec2

/** A Delaunay triangulation of planar points. */
export interface Delaunay {
  readonly points: readonly Point2[]
  /** Triangles as vertex indices, counter-clockwise. */
  readonly simplices: readonly (readonly [number, number, number])[]
  /** For each triangle, the triangle across the edge opposite each vertex (-1 on the hull), as scipy. */
  readonly neighbors: readonly (readonly [number, number, number])[]
  /** Hull edges as vertex pairs. */
  readonly hull: readonly (readonly [number, number])[]
  /** The triangle containing p (-1 outside the hull). */
  readonly findSimplex: (p: Point2) => number
}

/** Twice the signed area of (a, b, c): positive when counter-clockwise. */
export function orient(a: Point2, b: Point2, c: Point2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

const IN_CIRCLE_BOUND = (10 + 96 * 2 ** -53) * 2 ** -53

const bits = new DataView(new ArrayBuffer(8))

/** A finite double as `mantissa * 2^exponent`, the mantissa an integer. */
function decompose(x: number): { readonly mantissa: bigint; readonly exponent: number } {
  bits.setFloat64(0, x)
  const raw = bits.getBigUint64(0)
  const biased = Number((raw >> 52n) & 0x7ffn)
  const fraction = raw & 0xfffffffffffffn
  const magnitude = biased === 0 ? fraction : fraction | (1n << 52n)
  return { mantissa: raw >> 63n === 1n ? -magnitude : magnitude, exponent: (biased === 0 ? 1 : biased) - 1075 }
}

/** The sign of the in-circle determinant, exactly (integer arithmetic on the doubles' own mantissas). */
function inCircleSign(a: Point2, b: Point2, c: Point2, d: Point2): number {
  const parts = [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]].map(decompose)
  const lowest = Math.min(...parts.map((p) => p.exponent))
  const [ax, ay, bx, by, cx, cy, dx, dy] = parts.map((p) => p.mantissa << BigInt(p.exponent - lowest)) as [
    bigint,
    bigint,
    bigint,
    bigint,
    bigint,
    bigint,
    bigint,
    bigint,
  ]
  const adx = ax - dx
  const ady = ay - dy
  const bdx = bx - dx
  const bdy = by - dy
  const cdx = cx - dx
  const cdy = cy - dy
  const ad = adx * adx + ady * ady
  const bd = bdx * bdx + bdy * bdy
  const cd = cdx * cdx + cdy * cdy
  const det = adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx)
  return det > 0n ? 1 : det < 0n ? -1 : 0
}

/**
 * Positive when d lies strictly inside the circumcircle of the counter-clockwise triangle (a, b, c), negative when
 * outside, zero when on it, with the sign exact for any finite points: the determinant is evaluated in floating point
 * and, when it is within Shewchuk's error bound of zero, again in integers, so that cocircular points (a regular polygon)
 * are never decided by rounding noise.
 */
export function inCircle(a: Point2, b: Point2, c: Point2, d: Point2): number {
  const adx = a[0] - d[0]
  const ady = a[1] - d[1]
  const bdx = b[0] - d[0]
  const bdy = b[1] - d[1]
  const cdx = c[0] - d[0]
  const cdy = c[1] - d[1]
  const ad = adx * adx + ady * ady
  const bd = bdx * bdx + bdy * bdy
  const cd = cdx * cdx + cdy * cdy
  const det = adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx)
  const permanent =
    (Math.abs(bdx * cdy) + Math.abs(cdx * bdy)) * ad +
    (Math.abs(cdx * ady) + Math.abs(adx * cdy)) * bd +
    (Math.abs(adx * bdy) + Math.abs(bdx * ady)) * cd
  if (Math.abs(det) > IN_CIRCLE_BOUND * permanent) return det
  return inCircleSign(a, b, c, d) * (Math.abs(det) || Number.MIN_VALUE)
}

/** The circumcentre of a triangle. */
export function circumcentre(a: Point2, b: Point2, c: Point2): Point2 {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
  const a2 = a[0] * a[0] + a[1] * a[1]
  const b2 = b[0] * b[0] + b[1] * b[1]
  const c2 = c[0] * c[0] + c[1] * c[1]
  return [
    (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
    (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
  ]
}

/** Points `[n, 2]` (tensor or nested arrays) as pairs. */
export function toPoints(points: TensorLike | readonly Point2[]): Point2[] {
  if (Array.isArray(points) && (points.length === 0 || Array.isArray(points[0])))
    return (points as readonly Point2[]).map((p) => [p[0], p[1]])
  const t = concreteOrThrow(promoteFloat(points as TensorLike)[0] as Tensor, 'points')
  if (t.shape.length !== 2 || t.shape[1] !== 2) throw new MdmlError('shape', 'points must be [n, 2]')
  const flat = toFlat(t)
  return Array.from({ length: t.shape[0] ?? 0 }, (_, i) => [flat[2 * i] ?? 0, flat[2 * i + 1] ?? 0])
}

/**
 * The Delaunay triangulation of planar points (scipy.spatial.Delaunay in 2-D) by the Bowyer-Watson algorithm: insert
 * points one at a time (in a spatially sorted order), remove the triangles whose circumcircles contain the new point and
 * re-triangulate the cavity. No triangle's circumcircle contains another point; the minimum angle is maximised.
 *
 * @param points - Points `[n, 2]`, at least three and not all collinear.
 * @returns The triangulation with neighbours, hull and point location.
 * @concept delaunay
 * @index function Delaunay triangulation of planar points.
 * @example
 * ```ts
 * import { delaunay } from 'mdml-engine/numerics/geometry'
 * delaunay([[0, 0], [1, 0], [0, 1], [1, 1]]).simplices.length // => 2
 * ```
 */
export function delaunay(points: TensorLike | readonly Point2[]): Delaunay {
  const pts = toPoints(points)
  const n = pts.length
  if (n < 3) throw new MdmlError('value', 'delaunay: at least three points are needed')
  if (pts.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y)))
    throw new MdmlError('value', 'delaunay: the points must be finite')
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const [x, y] of pts) {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
  }
  const span = Math.max(maxX - minX, maxY - minY) || 1
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  // Work in normalised coordinates, with a super-triangle far outside.
  const work: Point2[] = pts.map(([x, y]) => [(x - cx) / span, (y - cy) / span])
  const big = 1e4
  work.push([-big, -big], [big, -big], [0, big])
  const s0 = n
  const s1 = n + 1
  const s2 = n + 2
  type Tri = [a: number, b: number, c: number]
  let tris: Tri[] = [[s0, s1, s2]]
  let alive: boolean[] = [true]
  const at = (i: number): Point2 => work[i] as Point2
  // Insert in a Hilbert-like (row-snake) order so that the walk stays local.
  const grid = Math.max(1, Math.round(Math.sqrt(n / 4)))
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const [ax, ay] = at(a)
    const [bx, by] = at(b)
    const ra = Math.min(grid - 1, Math.floor((ay + 0.5) * grid))
    const rb = Math.min(grid - 1, Math.floor((by + 0.5) * grid))
    if (ra !== rb) return ra - rb
    return ra % 2 === 0 ? ax - bx : bx - ax
  })
  for (const p of order) {
    const point = at(p)
    const bad: number[] = []
    tris.forEach((t, i) => {
      if (alive[i] && inCircle(at(t[0]), at(t[1]), at(t[2]), point) > 0) bad.push(i)
    })
    // The cavity boundary: edges of bad triangles not shared with another bad triangle.
    const edgeCount = new Map<string, [number, number]>()
    const seen = new Map<string, number>()
    for (const i of bad) {
      const t = tris[i] as Tri
      for (const [a, b] of [
        [t[0], t[1]],
        [t[1], t[2]],
        [t[2], t[0]],
      ] as const) {
        const key = a < b ? `${a},${b}` : `${b},${a}`
        seen.set(key, (seen.get(key) ?? 0) + 1)
        edgeCount.set(key, [a, b])
      }
      alive[i] = false
    }
    for (const [key, [a, b]] of edgeCount) {
      if (seen.get(key) !== 1) continue
      tris.push([a, b, p])
      alive.push(true)
    }
    if (tris.length > 4 * n + 64 && alive.filter(Boolean).length < tris.length / 2) {
      tris = tris.filter((_, i) => alive[i])
      alive = tris.map(() => true)
    }
  }
  const simplices = tris
    .filter((t, i) => alive[i] && t[0] < n && t[1] < n && t[2] < n)
    .map((t): Tri => (orient(at(t[0]), at(t[1]), at(t[2])) > 0 ? t : [t[0], t[2], t[1]]))
  if (simplices.length === 0) throw new MdmlError('value', 'delaunay: the points are collinear')
  // Neighbours across the edge opposite each vertex.
  const edgeOwner = new Map<string, { tri: number; vertex: number }[]>()
  simplices.forEach((t, ti) => {
    for (let v = 0; v < 3; v++) {
      const a = t[(v + 1) % 3] as number
      const b = t[(v + 2) % 3] as number
      const key = a < b ? `${a},${b}` : `${b},${a}`
      const list = edgeOwner.get(key) ?? []
      list.push({ tri: ti, vertex: v })
      edgeOwner.set(key, list)
    }
  })
  const neighbors = simplices.map((): [number, number, number] => [-1, -1, -1])
  const hull: [number, number][] = []
  for (const [key, owners] of edgeOwner) {
    if (owners.length === 2) {
      const [o1, o2] = owners as [{ tri: number; vertex: number }, { tri: number; vertex: number }]
      ;(neighbors[o1.tri] as [number, number, number])[o1.vertex] = o2.tri
      ;(neighbors[o2.tri] as [number, number, number])[o2.vertex] = o1.tri
    } else {
      const [a, b] = key.split(',').map(Number) as [number, number]
      hull.push([a, b])
    }
  }
  const findSimplex = (p: Point2): number => {
    const q: Point2 = [(p[0] - cx) / span, (p[1] - cy) / span]
    const tol = -1e-12
    // A walk from the first triangle, falling back to a scan when it leaves the hull.
    let current = 0
    for (let steps = 0; steps < simplices.length + 3; steps++) {
      const t = simplices[current] as Tri
      let moved = false
      for (let v = 0; v < 3; v++) {
        const a = at(t[(v + 1) % 3] as number)
        const b = at(t[(v + 2) % 3] as number)
        if (orient(a, b, q) < tol) {
          const next = (neighbors[current] as [number, number, number])[v]
          if (next === -1) return scan(q)
          current = next as number
          moved = true
          break
        }
      }
      if (!moved) return current
    }
    return scan(q)
  }
  const scan = (q: Point2): number =>
    simplices.findIndex(
      (t) =>
        orient(at(t[0]), at(t[1]), q) >= -1e-12 &&
        orient(at(t[1]), at(t[2]), q) >= -1e-12 &&
        orient(at(t[2]), at(t[0]), q) >= -1e-12,
    )
  return { points: pts, simplices, neighbors, hull, findSimplex }
}

/**
 * Barycentric coordinates of p in the triangle (a, b, c): weights summing to 1 with p = l0 a + l1 b + l2 c, all
 * non-negative inside the triangle.
 *
 * @param triangle - The vertices.
 * @param p - The point.
 * @returns The three weights.
 * @example
 * ```ts
 * import { barycentricCoordinates } from 'mdml-engine/numerics/geometry'
 * barycentricCoordinates([[0, 0], [1, 0], [0, 1]], [0.25, 0.25]) // => [0.5, 0.25, 0.25]
 * ```
 */
export function barycentricCoordinates(
  triangle: readonly [Point2, Point2, Point2],
  p: Point2,
): [number, number, number] {
  const [a, b, c] = triangle
  const area = orient(a, b, c)
  if (area === 0) throw new MdmlError('value', 'barycentricCoordinates: degenerate triangle')
  const l1 = orient(p, c, a) / area
  const l2 = orient(p, a, b) / area
  return [1 - l1 - l2, l1, l2]
}

/** A Voronoi diagram, dual to a Delaunay triangulation. */
export interface Voronoi {
  /** Voronoi vertices: the circumcentres of the Delaunay triangles. */
  readonly vertices: readonly Point2[]
  /** For each input point, its cell's vertex indices in counter-clockwise order, with -1 marking an unbounded cell. */
  readonly regions: readonly (readonly number[])[]
  /** Pairs of input points whose cells share an edge, with the edge's vertex indices (-1 for a ray to infinity). */
  readonly ridges: readonly {
    readonly points: readonly [number, number]
    readonly vertices: readonly [number, number]
  }[]
}

/**
 * The Voronoi diagram of planar points (scipy.spatial.Voronoi): the cells of points nearest each input point, built as
 * the dual of the Delaunay triangulation.
 *
 * @param points - Points `[n, 2]`.
 * @returns Vertices, regions and ridges.
 * @concept voronoi
 * @example
 * ```ts
 * import { voronoi } from 'mdml-engine/numerics/geometry'
 * voronoi([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1]]).regions[4].length // => 4
 * ```
 */
export function voronoi(points: TensorLike | readonly Point2[]): Voronoi {
  const tri = delaunay(points)
  const pts = tri.points
  const vertices = tri.simplices.map(([a, b, c]) => circumcentre(pts[a] as Point2, pts[b] as Point2, pts[c] as Point2))
  const incident: number[][] = pts.map(() => [])
  tri.simplices.forEach((t, ti) => {
    for (const v of t) incident[v]?.push(ti)
  })
  const onHull = new Set(tri.hull.flat())
  const regions = pts.map((p, i) => {
    const cells = (incident[i] ?? []).slice()
    // Counter-clockwise by angle of the circumcentres around the point.
    cells.sort((a, b) => {
      const va = vertices[a] as Point2
      const vb = vertices[b] as Point2
      return Math.atan2(va[1] - p[1], va[0] - p[0]) - Math.atan2(vb[1] - p[1], vb[0] - p[0])
    })
    return onHull.has(i) ? [-1, ...cells] : cells
  })
  const ridges: { points: [number, number]; vertices: [number, number] }[] = []
  const done = new Set<string>()
  tri.simplices.forEach((t, ti) => {
    for (let v = 0; v < 3; v++) {
      const a = t[(v + 1) % 3] as number
      const b = t[(v + 2) % 3] as number
      const key = a < b ? `${a},${b}` : `${b},${a}`
      if (done.has(key)) continue
      done.add(key)
      const other = (tri.neighbors[ti] as readonly number[])[v] ?? -1
      ridges.push({
        points: a < b ? [a, b] : [b, a],
        vertices: other === -1 ? [-1, ti] : [Math.min(ti, other), Math.max(ti, other)],
      })
    }
  })
  return { vertices, regions, ridges }
}

/**
 * The alpha shape of planar points: the union of Delaunay triangles whose circumradius is below alpha, returned as its
 * boundary edges (edges belonging to exactly one kept triangle). Large alpha gives the convex hull; small alpha follows
 * concavities and holes.
 *
 * @param points - Points `[n, 2]`.
 * @param alpha - The circumradius threshold.
 * @returns The boundary edges and the kept triangles.
 * @example
 * ```ts
 * import { alphaShape } from 'mdml-engine/numerics/geometry'
 * alphaShape([[0, 0], [1, 0], [0, 1], [1, 1]], 10).edges.length // => 4
 * ```
 */
export function alphaShape(
  points: TensorLike | readonly Point2[],
  alpha: number,
): {
  readonly edges: readonly (readonly [number, number])[]
  readonly triangles: readonly (readonly [number, number, number])[]
} {
  const tri = delaunay(points)
  const pts = tri.points
  const triangles = tri.simplices.filter(([a, b, c]) => {
    const pa = pts[a] as Point2
    const centre = circumcentre(pa, pts[b] as Point2, pts[c] as Point2)
    return Math.hypot(centre[0] - pa[0], centre[1] - pa[1]) < alpha
  })
  const count = new Map<string, number>()
  for (const t of triangles) {
    for (let v = 0; v < 3; v++) {
      const a = t[v] as number
      const b = t[(v + 1) % 3] as number
      const key = a < b ? `${a},${b}` : `${b},${a}`
      count.set(key, (count.get(key) ?? 0) + 1)
    }
  }
  const edges = [...count].filter(([, c]) => c === 1).map(([key]) => key.split(',').map(Number) as [number, number])
  return { edges, triangles }
}

/** Simplices as an int32 tensor `[m, 3]`. */
export function simplexTensor(tri: Delaunay): Tensor {
  return fromStorage(Int32Array.from(tri.simplices.flat()), [tri.simplices.length, 3], 'int32')
}

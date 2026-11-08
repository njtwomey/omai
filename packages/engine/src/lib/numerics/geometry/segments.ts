/**
 * Line segments as obstacles: closest points on a segment, where a ray first hits one, when a moving circle first
 * touches one, and whether two segments touch (collinear overlap included). These are the collision primitives behind
 * continuous motion in worlds, where walls are segments, and anything else that moves points or discs among segments.
 */

import { orient, type Point2 } from './triangulation.ts'

/** A contact with a segment: the time (or ray parameter) of first contact and the unit normal there. */
export interface SegmentContact {
  /** The ray parameter or fraction of the motion at first contact; `Infinity` when there is none. */
  readonly t: number
  /** The unit contact normal, pointing from the segment towards the moving point or circle (`[0, 0]` on a miss). */
  readonly normal: [number, number]
}

/**
 * The point of the closed segment ab closest to p, and its distance from p (a degenerate segment is its point a).
 *
 * @param p - The query point.
 * @param a - Segment start.
 * @param b - Segment end.
 * @returns The closest point and the Euclidean distance to it.
 * @concept collision-detection
 * @example
 * ```ts
 * import { closestPointOnSegment } from 'omai-engine/numerics/geometry'
 * closestPointOnSegment([1, 1], [0, 0], [2, 0]) // => { point: [1, 0], distance: 1 }
 * closestPointOnSegment([3, 0], [0, 0], [2, 0]).distance // => 1
 * ```
 */
export function closestPointOnSegment(
  p: Point2,
  a: Point2,
  b: Point2,
): { readonly point: [number, number]; readonly distance: number } {
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const len2 = ex * ex + ey * ey
  const u = len2 === 0 ? 0 : Math.min(1, Math.max(0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2))
  const point: [number, number] = [a[0] + u * ex, a[1] + u * ey]
  return { point, distance: Math.hypot(p[0] - point[0], p[1] - point[1]) }
}

/**
 * Where the ray `o + t d` (t >= 0) first meets the closed segment ab, with the segment's unit normal facing the ray's
 * origin. A ray parallel to the segment never hits it.
 *
 * @param o - The ray origin.
 * @param d - The ray direction (any length; `t` is in units of it).
 * @param a - Segment start.
 * @param b - Segment end.
 * @returns The ray parameter and normal, or `t = Infinity` on a miss.
 * @concept collision-detection
 * @example
 * ```ts
 * import { raySegmentHit } from 'omai-engine/numerics/geometry'
 * raySegmentHit([0, 0], [1, 0], [2, -1], [2, 1]).t // => 2
 * raySegmentHit([0, 0], [-1, 0], [2, -1], [2, 1]).t // => Infinity
 * ```
 */
export function raySegmentHit(o: Point2, d: Point2, a: Point2, b: Point2): SegmentContact {
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const denom = d[0] * ey - d[1] * ex
  const none: SegmentContact = { t: Number.POSITIVE_INFINITY, normal: [0, 0] }
  if (denom === 0) return none
  const ax = a[0] - o[0]
  const ay = a[1] - o[1]
  const t = (ax * ey - ay * ex) / denom
  const u = (ax * d[1] - ay * d[0]) / denom
  if (t < 0 || u < 0 || u > 1) return none
  const len = Math.hypot(ex, ey)
  let nx = -ey / len
  let ny = ex / len
  if (nx * d[0] + ny * d[1] > 0) {
    nx = -nx
    ny = -ny
  }
  return { t, normal: [nx, ny] }
}

/**
 * The first time `t` in `[0, 1]` at which a circle of radius `r`, moving from centre `p` by the displacement `d`, touches
 * the closed segment ab: against the segment's side (offset by `r`) or either endpoint (a circle-point quadratic). The
 * normal points from the segment towards the circle. A circle already touching and moving closer contacts at `t = 0`;
 * one moving away does not contact.
 *
 * @param p - The circle's centre at the start of the motion.
 * @param d - The displacement over the motion.
 * @param r - The circle's radius.
 * @param a - Segment start.
 * @param b - Segment end.
 * @returns The contact time and normal, or `t = Infinity` when the circle does not touch the segment.
 * @concept collision-detection
 * @example
 * ```ts
 * import { sweptCircleHit } from 'omai-engine/numerics/geometry'
 * sweptCircleHit([0, 0], [4, 0], 0.5, [2, -1], [2, 1]).t // => 0.375
 * sweptCircleHit([0, 0], [1, 0], 0.5, [2, -1], [2, 1]).t // => Infinity
 * ```
 */
export function sweptCircleHit(p: Point2, d: Point2, r: number, a: Point2, b: Point2): SegmentContact {
  let best = Number.POSITIVE_INFINITY
  let normal: [number, number] = [0, 0]
  const ex = b[0] - a[0]
  const ey = b[1] - a[1]
  const len = Math.hypot(ex, ey)
  if (len > 0) {
    let nx = -ey / len
    let ny = ex / len
    let s = (p[0] - a[0]) * nx + (p[1] - a[1]) * ny
    if (s < 0) {
      nx = -nx
      ny = -ny
      s = -s
    }
    const vn = d[0] * nx + d[1] * ny
    if (vn < 0) {
      const t = Math.max(0, (s - r) / -vn)
      if (t <= 1) {
        const cx = p[0] + t * d[0] - a[0]
        const cy = p[1] + t * d[1] - a[1]
        const u = (cx * ex + cy * ey) / (len * len)
        if (u >= 0 && u <= 1 && t < best) {
          best = t
          normal = [nx, ny]
        }
      }
    }
  }
  for (const e of [a, b]) {
    const fx = p[0] - e[0]
    const fy = p[1] - e[1]
    const qa = d[0] * d[0] + d[1] * d[1]
    const qb = 2 * (fx * d[0] + fy * d[1])
    const qc = fx * fx + fy * fy - r * r
    if (qb >= 0 || qa === 0) continue
    let t: number
    if (qc <= 0) t = 0
    else {
      const disc = qb * qb - 4 * qa * qc
      if (disc < 0) continue
      t = (-qb - Math.sqrt(disc)) / (2 * qa)
    }
    if (t <= 1 && t < best) {
      const cx = p[0] + t * d[0] - e[0]
      const cy = p[1] + t * d[1] - e[1]
      const n = Math.hypot(cx, cy)
      best = t
      normal = n > 0 ? [cx / n, cy / n] : [-d[0] / Math.sqrt(qa), -d[1] / Math.sqrt(qa)]
    }
  }
  return { t: best, normal }
}

/**
 * Whether the closed segments ab and cd share a point, touching and collinear overlap included. Each orientation
 * within a relative 1e-12 of zero (relative to the lengths it multiplies) counts as collinear, so segments through
 * rounded corners (the vertices of hexagons, say) are judged consistently.
 *
 * @param a - First segment start.
 * @param b - First segment end.
 * @param c - Second segment start.
 * @param d - Second segment end.
 * @returns True when the segments intersect or touch.
 * @concept collision-detection
 * @example
 * ```ts
 * import { segmentsTouch } from 'omai-engine/numerics/geometry'
 * segmentsTouch([0, 0], [2, 2], [0, 2], [2, 0]) // => true
 * segmentsTouch([0, 0], [1, 0], [1, 0], [1, 5]) // => true
 * segmentsTouch([0, 0], [1, 0], [2, 0], [3, 0]) // => false
 * ```
 */
export function segmentsTouch(a: Point2, b: Point2, c: Point2, d: Point2): boolean {
  const side = (p: Point2, q: Point2, r: Point2): number => {
    const o = orient(p, q, r)
    const tol = 1e-12 * Math.hypot(q[0] - p[0], q[1] - p[1]) * Math.hypot(r[0] - p[0], r[1] - p[1])
    return o > tol ? 1 : o < -tol ? -1 : 0
  }
  const o1 = side(a, b, c)
  const o2 = side(a, b, d)
  const o3 = side(c, d, a)
  const o4 = side(c, d, b)
  if (o1 * o2 > 0 || o3 * o4 > 0) return false
  if (o1 !== 0 || o2 !== 0 || o3 !== 0 || o4 !== 0) return true
  // Collinear: the projections on both axes must overlap.
  const overlap = (p: number, q: number, r: number, s: number): boolean =>
    Math.max(Math.min(p, q), Math.min(r, s)) <= Math.min(Math.max(p, q), Math.max(r, s)) + 1e-12
  return overlap(a[0], b[0], c[0], d[0]) && overlap(a[1], b[1], c[1], d[1])
}

/**
 * Shapes fitted to point sets, OpenCV style: the minimum-area rotated rectangle by rotating calipers (`minAreaRect`,
 * in OpenCV's single precision and angle convention) and its corners (`boxPoints`), the minimum enclosing circle, the
 * algebraic least-squares ellipse of `fitEllipse`, and `fitLine` by least squares or by reweighting under OpenCV's
 * robust distances.
 */

// TODO: fitEllipseDirect and fitEllipseAMS.

import { MdmlError, type Vec2 } from '../base/index.ts'
import { dgesvd } from '../numerics/linalg/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { f32, type PointsLike, pointsOf } from './_image.ts'
import type { GeomPath } from './greedy-contract.ts'
import { convexHull } from './hull.ts'

/** A rotated rectangle (cv::RotatedRect): centre, size [width, height] and angle in degrees. */
export interface RotatedRect {
  readonly center: [number, number]
  readonly size: [number, number]
  readonly angle: number
}

type P = Vec2

/** OpenCV's rotating calipers for the minimum-area rectangle of a counter-clockwise hull, in single precision. */
function calipers(points: readonly P[]): [number, number, number, number, number, number] {
  const n = points.length
  const vx = new Float64Array(n)
  const vy = new Float64Array(n)
  const invLen = new Float64Array(n)
  let left = 0
  let bottom = 0
  let right = 0
  let top = 0
  let p0 = points[0] as P
  let leftX = p0[0]
  let rightX = p0[0]
  let topY = p0[1]
  let bottomY = p0[1]
  for (let i = 0; i < n; i++) {
    if (p0[0] < leftX) {
      leftX = p0[0]
      left = i
    }
    if (p0[0] > rightX) {
      rightX = p0[0]
      right = i
    }
    if (p0[1] > topY) {
      topY = p0[1]
      top = i
    }
    if (p0[1] < bottomY) {
      bottomY = p0[1]
      bottom = i
    }
    const p = points[i + 1 < n ? i + 1 : 0] as P
    const dx = p[0] - p0[0]
    const dy = p[1] - p0[1]
    vx[i] = f32(dx)
    vy[i] = f32(dy)
    invLen[i] = f32(1 / Math.sqrt(dx * dx + dy * dy))
    p0 = p
  }
  const seq = [bottom, right, top, left]
  let minArea = Number.MAX_VALUE
  let baseA = 1
  let baseB = 0
  const best = { left: 0, a: 0, width: 0, b: 0, height: 0, bottom: 0 }
  const rot = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
  ]
  for (let k = 0; k < n; k++) {
    const s0 = seq[0] ?? 0
    const s1 = seq[1] ?? 0
    const s2 = seq[2] ?? 0
    const s3 = seq[3] ?? 0
    rot[0] = [vx[s0] ?? 0, vy[s0] ?? 0]
    rot[1] = [vy[s1] ?? 0, -(vx[s1] ?? 0)]
    rot[2] = [-(vx[s2] ?? 0), -(vy[s2] ?? 0)]
    rot[3] = [-(vy[s3] ?? 0), vx[s3] ?? 0]
    let main = 0
    for (let i = 1; i < 4; i++) {
      // firstVecIsRight(rot[i], rot[main]): rot[i] turned 90 degrees clockwise has a negative dot with rot[main].
      const a = rot[i] as number[]
      const b = rot[main] as number[]
      if (f32(f32((a[1] ?? 0) * (b[0] ?? 0)) + f32(-(a[0] ?? 0) * (b[1] ?? 0))) < 0) main = i
    }
    const pi = seq[main] ?? 0
    const lx = f32((vx[pi] ?? 0) * (invLen[pi] ?? 0))
    const ly = f32((vy[pi] ?? 0) * (invLen[pi] ?? 0))
    if (main === 0) [baseA, baseB] = [lx, ly]
    else if (main === 1) [baseA, baseB] = [ly, -lx]
    else if (main === 2) [baseA, baseB] = [-lx, -ly]
    else [baseA, baseB] = [-ly, lx]
    seq[main] = ((seq[main] ?? 0) + 1) % n
    const q1 = points[seq[1] ?? 0] as P
    const q3 = points[seq[3] ?? 0] as P
    const q0 = points[seq[0] ?? 0] as P
    const q2 = points[seq[2] ?? 0] as P
    const width = f32(f32(f32(q1[0] - q3[0]) * baseA) + f32(f32(q1[1] - q3[1]) * baseB))
    const height = f32(f32(-f32(q2[0] - q0[0]) * baseB) + f32(f32(q2[1] - q0[1]) * baseA))
    const area = f32(width * height)
    if (area <= minArea) {
      minArea = area
      best.left = seq[3] ?? 0
      best.a = baseA
      best.width = width
      best.b = baseB
      best.height = height
      best.bottom = seq[0] ?? 0
    }
  }
  const A1 = best.a
  const B1 = best.b
  const A2 = -best.b
  const B2 = best.a
  const pl = points[best.left] as P
  const pb = points[best.bottom] as P
  const C1 = f32(f32(A1 * pl[0]) + f32(pl[1] * B1))
  const C2 = f32(f32(A2 * pb[0]) + f32(pb[1] * B2))
  const idet = f32(1 / f32(f32(A1 * B2) - f32(A2 * B1)))
  const px = f32(f32(f32(C1 * B2) - f32(C2 * B1)) * idet)
  const py = f32(f32(f32(A1 * C2) - f32(A2 * C1)) * idet)
  return [px, py, f32(A1 * best.width), f32(B1 * best.width), f32(A2 * best.height), f32(B2 * best.height)]
}

/**
 * The minimum-area rotated rectangle enclosing points (Freeman and Shapira; cv2.minAreaRect): rotating calipers
 * over the convex hull, one side flush with a hull edge, computed in single precision as OpenCV does. The angle is
 * OpenCV's: in [-90, 0) degrees, the direction of the first side.
 *
 * @param points - Points `[n, 2]`.
 * @returns The rectangle.
 * @concept minimum-bounding-box
 * @example
 * ```ts
 * import { minAreaRect } from 'mdml-engine/vision'
 * minAreaRect([[0, 0], [4, 0], [4, 2], [0, 2]]) // => { center: [2, 1], size: [2, 4], angle: -90 }
 * ```
 */
export function minAreaRect(points: PointsLike): RotatedRect {
  const pts = pointsOf(points, 'minAreaRect')
  const hull = convexHull(pts).map((i) => pts[i] as P)
  const n = hull.length
  let angle = -Math.PI / 2
  let center: [number, number] = [0, 0]
  let size: [number, number] = [0, 0]
  if (n > 2) {
    const [px, py, ax, ay, bx, by] = calipers(hull)
    center = [f32(px + f32(f32(ax + bx) * 0.5)), f32(py + f32(f32(ay + by) * 0.5))]
    size = [f32(Math.sqrt(bx * bx + by * by)), f32(Math.sqrt(ax * ax + ay * ay))]
    if (ax === 0 && ay > 0) size = [size[1], size[0]]
    else angle = -Math.atan2(ax, ay)
  } else if (n === 2) {
    const [a, b] = hull as [P, P]
    center = [f32(f32(a[0] + b[0]) * 0.5), f32(f32(a[1] + b[1]) * 0.5)]
    const dx = a[0] - b[0]
    const dy = a[1] - b[1]
    size = [0, f32(Math.sqrt(dx * dx + dy * dy))]
    if (dx === 0) size = [size[1], size[0]]
    else if (dy < 0) {
      angle = Math.atan2(dy, dx)
      size = [size[1], size[0]]
    } else if (dy > 0) angle = -Math.atan2(dx, dy)
  } else if (n === 1) center = [(hull[0] as P)[0], (hull[0] as P)[1]]
  return { center, size, angle: f32((angle * 180) / Math.PI) }
}

/**
 * The four corners of a rotated rectangle (cv2.boxPoints): bottom-left, top-left, top-right, bottom-right in
 * OpenCV's order (for a y-down image the "bottom" is the larger y).
 *
 * @param box - The rectangle.
 * @returns Four corners (x, y).
 * @concept minimum-bounding-box
 * @example
 * ```ts
 * import { boxPoints } from 'mdml-engine/vision'
 * boxPoints({ center: [2, 1], size: [4, 2], angle: 0 }) // => [[0, 2], [0, 0], [4, 0], [4, 2]]
 * ```
 */
export function boxPoints(box: RotatedRect): [number, number][] {
  const a = (box.angle * Math.PI) / 180
  const b = Math.cos(a) * 0.5
  const c = Math.sin(a) * 0.5
  const [cx, cy] = box.center
  const [w, h] = box.size
  const p0: [number, number] = [cx - c * h - b * w, cy + b * h - c * w]
  const p1: [number, number] = [cx + c * h - b * w, cy - b * h - c * w]
  const p2: [number, number] = [2 * cx - p0[0], 2 * cy - p0[1]]
  const p3: [number, number] = [2 * cx - p1[0], 2 * cy - p1[1]]
  return [p0, p1, p2, p3]
}

/** The circle through three points, or undefined for (nearly) collinear points. */
function circle3(a: P, b: P, c: P): { c: [number, number]; r: number } | undefined {
  const v1x = b[0] - a[0]
  const v1y = b[1] - a[1]
  const v2x = c[0] - a[0]
  const v2y = c[1] - a[1]
  const c1 = ((a[0] + b[0]) / 2) * v1x + ((a[1] + b[1]) / 2) * v1y
  const c2 = ((a[0] + c[0]) / 2) * v2x + ((a[1] + c[1]) / 2) * v2y
  const det = v1x * v2y - v1y * v2x
  if (Math.abs(det) <= 1e-12 * Math.max(1, v1x * v1x + v1y * v1y, v2x * v2x + v2y * v2y)) return undefined
  const cx = (c1 * v2y - c2 * v1y) / det
  const cy = (v1x * c2 - v2x * c1) / det
  return { c: [cx, cy], r: Math.hypot(cx - a[0], cy - a[1]) }
}

/** The state of {@link minEnclosingCircleAlgorithm}: the circle so far (without the final inflation) and the next point. */
export interface EnclosingCircleState {
  readonly center: [number, number]
  readonly radius: number
  /** The next point to test against the circle. */
  readonly next: number
  readonly count: number
}

/**
 * The smallest enclosing circle (cv2.minEnclosingCircle) as a stepped algorithm, one point per step: Welzl's incremental
 * construction starts from the circle on the first two points and, for each further point outside the circle, rebuilds it
 * on that point and the earlier ones (a circle on two points, then three), in double precision. It logs the circle and the
 * points tested, and is done (`converged`) when every point lies inside. The state serialises; the radius is inflated by
 * 1e-4, as OpenCV's is, by {@link enclosingCircleOf}.
 *
 * @param points - Points `[n, 2]`.
 * @returns The algorithm.
 * @concept minimum-enclosing-circle
 * @example
 * ```ts
 * import { enclosingCircleOf, minEnclosingCircleAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * enclosingCircleOf(runAlgorithm(minEnclosingCircleAlgorithm([[0, 0], [4, 0], [2, 1]]), { maxSteps: 10 }).state).center // => [2, 0]
 * ```
 */
export function minEnclosingCircleAlgorithm(points: PointsLike): Algorithm<EnclosingCircleState, GeomPath> {
  const pts = pointsOf(points, 'minEnclosingCircle')
  const n = pts.length
  const inside = (c: [number, number], r: number, p: P): boolean =>
    Math.hypot(p[0] - c[0], p[1] - c[1]) <= r + 1e-12 * Math.max(1, r)
  const two = (a: P, b: P): { c: [number, number]; r: number } => ({
    c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
    r: Math.hypot(a[0] - b[0], a[1] - b[1]) / 2,
  })
  return {
    name: 'vision/min-enclosing-circle',
    config: { points: n },
    init: () => {
      if (n === 0) return { center: [0, 0], radius: 0, next: 0, count: 0 }
      if (n === 1) return { center: [(pts[0] as P)[0], (pts[0] as P)[1]], radius: 0, next: 1, count: 1 }
      const { c, r } = two(pts[0] as P, pts[1] as P)
      return { center: c, radius: r, next: 2, count: 2 }
    },
    step: (s, ctx) => {
      const i = s.next
      const pi = pts[i] as P
      let c = s.center
      let r = s.radius
      if (!inside(c, r, pi)) {
        ;({ c, r } = two(pts[0] as P, pi))
        for (let j = 1; j < i; j++) {
          const pj = pts[j] as P
          if (inside(c, r, pj)) continue
          ;({ c, r } = two(pi, pj))
          for (let k = 0; k < j; k++) {
            const pk = pts[k] as P
            if (inside(c, r, pk)) continue
            const t = circle3(pi, pj, pk)
            if (t !== undefined) ({ c, r } = t)
          }
        }
      }
      ctx.log('geom/circle', [c[0], c[1], r])
      ctx.log('geom/points', i + 1)
      return { center: c, radius: r, next: i + 1, count: i + 1 }
    },
    done: (s) => (s.next >= n ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EnclosingCircleState,
  }
}

/**
 * The circle of a finished enclosing-circle state, with OpenCV's inflation of the radius by 1e-4 (so every point lies
 * strictly inside; an empty point set gives radius 0).
 *
 * @param state - A state of {@link minEnclosingCircleAlgorithm}.
 * @returns The centre and radius.
 * @concept minimum-enclosing-circle
 * @example
 * ```ts
 * import { enclosingCircleOf, minEnclosingCircleAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * enclosingCircleOf(runAlgorithm(minEnclosingCircleAlgorithm([[1, 1]]), { maxSteps: 5 }).state).radius // => 0.0001
 * ```
 */
export function enclosingCircleOf(state: EnclosingCircleState): {
  readonly center: [number, number]
  readonly radius: number
} {
  return { center: state.center, radius: state.count === 0 ? 0 : state.radius + 1e-4 }
}

/**
 * The smallest circle enclosing points (cv2.minEnclosingCircle): Welzl's incremental construction (a circle on two
 * points, then three) computed in double precision; the radius is inflated by 1e-4 as OpenCV's is, so every point
 * lies strictly inside. {@link minEnclosingCircleAlgorithm} run to the end.
 *
 * @param points - Points `[n, 2]`.
 * @returns The centre and radius.
 * @concept minimum-enclosing-circle
 * @example
 * ```ts
 * import { minEnclosingCircle } from 'mdml-engine/vision'
 * const { center, radius } = minEnclosingCircle([[0, 0], [4, 0], [2, 1]])
 * [center, Math.round(radius * 1e3) / 1e3] // => [[2, 0], 2]
 * ```
 */
export function minEnclosingCircle(points: PointsLike): { readonly center: [number, number]; readonly radius: number } {
  return enclosingCircleOf(
    runAlgorithm(minEnclosingCircleAlgorithm(points), { maxSteps: Number.MAX_SAFE_INTEGER }).state,
  )
}

/** Least squares by the SVD (OpenCV's SVBackSubst / solve with DECOMP_SVD): argmin |A x - b|, A m x n row-major. */
function svdSolve(a: Float64Array, b: Float64Array, m: number, n: number): Float64Array {
  const { u, s, v } = dgesvd(a, m, n)
  const k = s.length
  const x = new Float64Array(n)
  const tol = Number.EPSILON * Math.max(m, n) * (s[0] ?? 0)
  for (let c = 0; c < k; c++) {
    const sc = s[c] ?? 0
    if (sc <= tol) continue
    let ub = 0
    for (let i = 0; i < m; i++) ub += (u[i * k + c] ?? 0) * (b[i] ?? 0)
    const w = ub / sc
    for (let j = 0; j < n; j++) x[j] = (x[j] ?? 0) + w * (v[j * k + c] ?? 0)
  }
  return x
}

/**
 * The ellipse best fitting points in the algebraic least-squares sense (cv2.fitEllipse, Weiss's formulation): a
 * general conic fitted to the centred, scaled points, its centre from the conic's gradient, then the axes and angle
 * refitted about that centre. Returned as OpenCV's rotated rectangle: the full axis lengths with width <= height and
 * the angle in degrees.
 *
 * @param points - At least five points `[n, 2]`.
 * @returns The ellipse as a rotated rectangle.
 * @throws MdmlError with code `value` for fewer than five points.
 * @concept ellipse-fitting
 * @example
 * ```ts
 * import { fitEllipse } from 'mdml-engine/vision'
 * const pts = Array.from({ length: 12 }, (_, i) => [3 * Math.cos((Math.PI * i) / 6) + 5, Math.sin((Math.PI * i) / 6) + 2])
 * const e = fitEllipse(pts)
 * [e.center, e.size, e.angle].flat().map((v) => Math.round(v * 1e4) / 1e4) // => [5, 2, 2, 6, 90]
 * ```
 */
export function fitEllipse(points: PointsLike): RotatedRect {
  const pts = pointsOf(points, 'fitEllipse').map(([x, y]): P => [f32(x), f32(y)])
  const n = pts.length
  if (n < 5) throw new MdmlError('value', 'fitEllipse: at least five points are needed')
  let cx = 0
  let cy = 0
  for (const [x, y] of pts) {
    cx = f32(cx + x)
    cy = f32(cy + y)
  }
  cx = f32(cx / n)
  cy = f32(cy / n)
  let spread = 0
  for (const [x, y] of pts) spread += Math.abs(f32(x - cx)) + Math.abs(f32(y - cy))
  const scale = 100 / Math.max(spread, 1.1920928955078125e-7)
  const A = new Float64Array(5 * n)
  const b = new Float64Array(n).fill(10000)
  pts.forEach(([x, y], i) => {
    const px = f32(x - cx) * scale
    const py = f32(y - cy) * scale
    A.set([-px * px, -py * py, -px * py, px, py], 5 * i)
  })
  const g = svdSolve(A, b, n, 5)
  const centre = svdSolve(
    Float64Array.from([2 * (g[0] ?? 0), g[2] ?? 0, g[2] ?? 0, 2 * (g[1] ?? 0)]),
    Float64Array.from([g[3] ?? 0, g[4] ?? 0]),
    2,
    2,
  )
  const r0 = centre[0] ?? 0
  const r1 = centre[1] ?? 0
  const A3 = new Float64Array(3 * n)
  pts.forEach(([x, y], i) => {
    const px = f32(x - cx) * scale - r0
    const py = f32(y - cy) * scale - r1
    A3.set([px * px, py * py, px * py], 3 * i)
  })
  const q = svdSolve(A3, new Float64Array(n).fill(1), n, 3)
  const q0 = q[0] ?? 0
  const q1 = q[1] ?? 0
  const q2 = q[2] ?? 0
  const theta = -0.5 * Math.atan2(q2, q1 - q0)
  const t = Math.abs(q2) > 1e-8 ? q2 / Math.sin(-2 * theta) : q1 - q0
  let ra = Math.abs(q0 + q1 - t)
  if (ra > 1e-8) ra = Math.sqrt(2 / ra)
  let rb = Math.abs(q0 + q1 + t)
  if (rb > 1e-8) rb = Math.sqrt(2 / rb)
  let width = f32((ra * 2) / scale)
  let height = f32((rb * 2) / scale)
  let angle = 0
  if (width > height) {
    ;[width, height] = [height, width]
    angle = f32(90 + (theta * 180) / Math.PI)
  }
  if (angle < -180) angle += 360
  if (angle > 360) angle -= 360
  return { center: [f32(f32(r0 / scale) + cx), f32(f32(r1 / scale) + cy)], size: [width, height], angle }
}

/** The distances of {@link fitLine} (cv2.DIST_*): least squares or a robust M-estimator. */
export type LineDistance = 'l2' | 'l1' | 'l12' | 'fair' | 'welsch' | 'huber'

const WEIGHTS: Readonly<Record<Exclude<LineDistance, 'l2'>, (r: number, c: number) => number>> = {
  l1: (r) => 1 / Math.max(Math.abs(r), 1e-6),
  l12: (r) => 1 / Math.sqrt(1 + (r * r) / 2),
  fair: (r, c) => 1 / (1 + r / c),
  welsch: (r, c) => Math.exp(-((r / c) ** 2)),
  huber: (r, c) => (r < c ? 1 : c / r),
}

const DEFAULT_C: Readonly<Record<Exclude<LineDistance, 'l2'>, number>> = {
  l1: 0,
  l12: 0,
  fair: 1.3998,
  welsch: 2.9846,
  huber: 1.345,
}

/** The weighted total least-squares line through points: [vx, vy, x0, y0] (OpenCV's fitLine2D_wods). */
function weightedLine(pts: readonly P[], w: Float64Array | undefined): [number, number, number, number] {
  let sx = 0
  let sy = 0
  let sxx = 0
  let syy = 0
  let sxy = 0
  let sw = 0
  pts.forEach(([x, y], i) => {
    const wi = w === undefined ? 1 : (w[i] ?? 0)
    sx += wi * x
    sy += wi * y
    sxx += wi * f32(x * x)
    syy += wi * f32(y * y)
    sxy += wi * f32(x * y)
    sw += wi
  })
  const mx = sx / sw
  const my = sy / sw
  const dxx = sxx / sw - mx * mx
  const dyy = syy / sw - my * my
  const dxy = sxy / sw - mx * my
  const t = f32(f32(Math.atan2(2 * dxy, dxx - dyy)) / 2)
  return [f32(Math.cos(t)), f32(Math.sin(t)), f32(mx), f32(my)]
}

/**
 * Fit a line to points (cv2.fitLine in 2-D): 'l2' is the total least-squares line (the principal axis through the
 * centroid, as OpenCV computes it); the robust distances reweight the points by OpenCV's weight functions of their
 * distances to the current line and refit, from the least-squares line, until the line stops moving (OpenCV restarts
 * from random subsets instead, so robust fits agree with it only approximately).
 *
 * @param points - Points `[n, 2]`.
 * @param options - Distance (default 'l2'), its parameter (0: OpenCV's default), and the radial and angular accuracy
 *   (defaults 0.01).
 * @returns The line [vx, vy, x0, y0]: a unit direction and a point on it.
 * @concept line-fitting
 * @example
 * ```ts
 * import { fitLine } from 'mdml-engine/vision'
 * fitLine([[0, 1], [1, 3], [2, 5]]).map((v) => Math.round(v * 1e4) / 1e4) // => [0.4472, 0.8944, 1, 3]
 * ```
 */
export function fitLine(
  points: PointsLike,
  options: {
    readonly distance?: LineDistance
    readonly param?: number
    readonly reps?: number
    readonly aeps?: number
  } = {},
): [number, number, number, number] {
  const pts = pointsOf(points, 'fitLine')
  if (pts.length < 2) throw new MdmlError('value', 'fitLine: at least two points are needed')
  const dist = options.distance ?? 'l2'
  let line = weightedLine(pts, undefined)
  if (dist === 'l2') return line
  const weight = WEIGHTS[dist]
  if (weight === undefined) throw new MdmlError('value', `fitLine: unknown distance ${String(dist)}`)
  const c = options.param !== undefined && options.param > 0 ? options.param : DEFAULT_C[dist]
  const reps = options.reps ?? 0.01
  const aeps = options.aeps ?? 0.01
  const w = new Float64Array(pts.length)
  for (let it = 0; it < 30; it++) {
    const [vx, vy, x0, y0] = line
    pts.forEach(([x, y], i) => {
      w[i] = weight(Math.abs(vy * (x - x0) - vx * (y - y0)), c)
    })
    const next = weightedLine(pts, w)
    const cos = Math.min(1, Math.max(-1, next[0] * vx + next[1] * vy))
    const moved = Math.max(Math.abs(next[2] - x0), Math.abs(next[3] - y0))
    line = next
    if (Math.abs(Math.acos(cos)) < aeps && moved < reps) break
  }
  return line
}

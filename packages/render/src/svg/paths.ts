/**
 * SVG path data shared by `sceneToSvg` and `SceneView`: polylines and polygons, smooth curves through points
 * (Catmull-Rom splines as cubic Bézier segments), and arrow-head triangles.
 */

import type { Point2 } from 'omai-engine/numerics/geometry'

/**
 * SVG path data of a polyline or polygon through points.
 *
 * @param points - The points.
 * @param closed - Close the path (default false).
 * @returns The path data.
 * @example
 * ```ts
 * import { straightPathData } from 'omai-render/svg'
 * straightPathData([[0, 0], [1, 2]], true) // => 'M0,0L1,2Z'
 * ```
 */
export function straightPathData(points: readonly Point2[], closed = false): string {
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join('')
  return closed && points.length > 0 ? `${d}Z` : d
}

/**
 * SVG path data of a polyline with rounded corners: each interior vertex becomes a cubic curve (a quarter circle at a
 * right angle) between points `radius` back along its two segments, the radius limited to half the shorter segment.
 *
 * @param points - The points.
 * @param radius - The corner radius (a radius of 0 or less gives {@link straightPathData}).
 * @param closed - Close the path (default false).
 * @returns The path data.
 * @example
 * ```ts
 * import { roundedPathData } from 'omai-render/svg'
 * roundedPathData([[0, 0], [10, 0], [10, 10]], 4) // => 'M0,0L6,0C8.2092,0 10,1.7908 10,4L10,10'
 * ```
 */
export function roundedPathData(points: readonly Point2[], radius: number, closed = false): string {
  const n = points.length
  if (n < 3 || !(radius > 0)) return straightPathData(points, closed)
  const r = (v: number): number => Math.round(v * 1e4) / 1e4
  const k = 0.5523
  let d = ''
  const first = closed ? 0 : 1
  const last = closed ? n : n - 1
  const at = (i: number): Point2 => points[(i + n) % n] as Point2
  let moved = false
  const corner = (i: number): void => {
    const [u, v, w] = [at(i - 1), at(i), at(i + 1)]
    const [lu, lw] = [Math.hypot(u[0] - v[0], u[1] - v[1]), Math.hypot(w[0] - v[0], w[1] - v[1])]
    const e = Math.min(radius, lu / 2, lw / 2)
    if (!(e > 1e-9)) {
      d += `${moved ? 'L' : 'M'}${r(v[0])},${r(v[1])}`
      moved = true
      return
    }
    const a: Point2 = [v[0] + ((u[0] - v[0]) / lu) * e, v[1] + ((u[1] - v[1]) / lu) * e]
    const b: Point2 = [v[0] + ((w[0] - v[0]) / lw) * e, v[1] + ((w[1] - v[1]) / lw) * e]
    const c1 = [a[0] + (v[0] - a[0]) * k, a[1] + (v[1] - a[1]) * k]
    const c2 = [b[0] + (v[0] - b[0]) * k, b[1] + (v[1] - b[1]) * k]
    d += `${moved ? 'L' : 'M'}${r(a[0])},${r(a[1])}C${r(c1[0] as number)},${r(c1[1] as number)} ${r(c2[0] as number)},${r(c2[1] as number)} ${r(b[0])},${r(b[1])}`
    moved = true
  }
  if (!closed) {
    const p0 = points[0] as Point2
    d = `M${p0[0]},${p0[1]}`
    moved = true
  }
  for (let i = first; i < last; i++) corner(i)
  if (!closed) {
    const pn = points[n - 1] as Point2
    d += `L${pn[0]},${pn[1]}`
  }
  return closed ? `${d}Z` : d
}

/**
 * SVG path data of a smooth curve through points (Catmull-Rom as cubic Béziers), open or closed; fewer than three
 * points give straight segments.
 *
 * @param points - The points.
 * @param closed - Close the curve (default false).
 * @returns The path data.
 * @example
 * ```ts
 * import { smoothPathData } from 'omai-render/svg'
 * smoothPathData([[0, 0], [1, 1], [2, 0]]).startsWith('M0,0C') // => true
 * ```
 */
export function smoothPathData(points: readonly Point2[], closed = false): string {
  const n = points.length
  if (n < 3) return straightPathData(points, closed)
  const at = (i: number): Point2 => points[closed ? (i + n) % n : Math.min(Math.max(i, 0), n - 1)] as Point2
  const first = at(0)
  let d = `M${first[0]},${first[1]}`
  const segments = closed ? n : n - 1
  for (let i = 0; i < segments; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6]
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6]
    d += `C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${p2[0]},${p2[1]}`
  }
  return closed ? `${d}Z` : d
}

/**
 * The triangle of an arrow head at `to`, pointing away from `from`, as an SVG `points` string.
 *
 * @param from - A point behind the tip.
 * @param to - The tip.
 * @param size - The head's length.
 * @returns The polygon's points.
 * @example
 * ```ts
 * import { arrowHeadPoints } from 'omai-render/svg'
 * arrowHeadPoints([0, 0], [10, 0], 4) // => '10,0 6,1.8 6,-1.8'
 * ```
 */
export function arrowHeadPoints(from: Point2, to: Point2, size: number): string {
  const dx = to[0] - from[0]
  const dy = to[1] - from[1]
  const len = Math.hypot(dx, dy) || 1
  const [ux, uy] = [dx / len, dy / len]
  const bx = to[0] - ux * size
  const by = to[1] - uy * size
  const w = size * 0.45
  const r = (v: number): number => Math.round(v * 1e6) / 1e6
  return `${r(to[0])},${r(to[1])} ${r(bx - uy * w)},${r(by + ux * w)} ${r(bx + uy * w)},${r(by - ux * w)}`
}

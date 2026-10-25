/**
 * SVG path data shared by `sceneToSvg` and `SceneView`: polylines and polygons, smooth curves through points
 * (Catmull-Rom splines as cubic Bézier segments), and arrow-head triangles.
 */

/** A point `[x, y]`. */
type Pt = readonly [number, number]

/**
 * SVG path data of a polyline or polygon through points.
 *
 * @param points - The points.
 * @param closed - Close the path (default false).
 * @returns The path data.
 * @example
 * ```ts
 * import { straightPathData } from 'mdml-render/svg'
 * straightPathData([[0, 0], [1, 2]], true) // => 'M0,0L1,2Z'
 * ```
 */
export function straightPathData(points: readonly Pt[], closed = false): string {
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join('')
  return closed && points.length > 0 ? `${d}Z` : d
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
 * import { smoothPathData } from 'mdml-render/svg'
 * smoothPathData([[0, 0], [1, 1], [2, 0]]).startsWith('M0,0C') // => true
 * ```
 */
export function smoothPathData(points: readonly Pt[], closed = false): string {
  const n = points.length
  if (n < 3) return straightPathData(points, closed)
  const at = (i: number): Pt => points[closed ? (i + n) % n : Math.min(Math.max(i, 0), n - 1)] as Pt
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
 * import { arrowHeadPoints } from 'mdml-render/svg'
 * arrowHeadPoints([0, 0], [10, 0], 4) // => '10,0 6,1.8 6,-1.8'
 * ```
 */
export function arrowHeadPoints(from: Pt, to: Pt, size: number): string {
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

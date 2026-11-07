/**
 * Closed outlines from drawings: SVG path data and elements flattened to polylines, several strokes stitched into one
 * closed path, and paths resampled uniformly by arc length and normalised (centred, unit radius), the form a Fourier
 * series of a curve, a morph or a shape descriptor needs.
 */

import { MdmlError } from '../base/index.ts'
import type { Point2 } from '../numerics/geometry/index.ts'

/** One stroke of a drawing: its points, and whether it closes on itself. */
export interface Stroke {
  readonly points: readonly Point2[]
  readonly closed: boolean
}

/** One SVG element: a tag (`path`, `circle`, `ellipse`, `line`, `rect`, `polyline`, `polygon`) and its attributes. */
export type SvgElement = readonly [string, Readonly<Record<string, string>>]

const dist = (a: Point2, b: Point2): number => Math.hypot(a[0] - b[0], a[1] - b[1])

const ARITY: Readonly<Record<string, number>> = { M: 2, L: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, T: 2, A: 7, Z: 0 }
const NUMBER = /^[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/

// The commands of path data with their argument groups. Arc flags may be written without separators (`a2 2 0 012 2`).
function* tokens(d: string): Generator<[string, number[]]> {
  const parts = d.match(/[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*/g) ?? []
  for (const part of parts) {
    const cmd = part[0] ?? 'Z'
    const up = cmd.toUpperCase()
    const arity = ARITY[up] ?? 0
    if (arity === 0) {
      yield [cmd, []]
      continue
    }
    const body = part.slice(1)
    const args: number[] = []
    let i = 0
    while (i < body.length) {
      const ch = body[i] ?? ''
      if (' ,\t\n\r'.includes(ch)) {
        i++
        continue
      }
      if (up === 'A' && (args.length % 7 === 3 || args.length % 7 === 4)) {
        args.push(Number(ch))
        i++
        continue
      }
      const m = NUMBER.exec(body.slice(i))
      if (m === null) throw new MdmlError('value', `flattenSvgPath: bad number at '${body.slice(i, i + 12)}'`)
      args.push(Number(m[0]))
      i += m[0].length
    }
    for (let k = 0; k < args.length; k += arity) {
      // Extra pairs after a moveto are implicit linetos.
      const c = k > 0 && up === 'M' ? (cmd === 'm' ? 'l' : 'L') : cmd
      yield [c, args.slice(k, k + arity)]
    }
  }
}

function cubic(p0: Point2, p1: Point2, p2: Point2, p3: Point2, k = 64): Point2[] {
  const out: Point2[] = []
  for (let i = 1; i <= k; i++) {
    const t = i / k
    const a = (1 - t) ** 3
    const b = 3 * (1 - t) ** 2 * t
    const c = 3 * (1 - t) * t ** 2
    const d = t ** 3
    out.push([a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]])
  }
  return out
}

// An SVG elliptical arc by the endpoint-to-centre conversion (SVG 1.1, appendix F.6.5), then sampled.
function arc(p0: Point2, rx0: number, ry0: number, phi: number, large: boolean, sweep: boolean, p1: Point2): Point2[] {
  if (rx0 === 0 || ry0 === 0 || dist(p0, p1) < 1e-12) return [p1]
  let rx = Math.abs(rx0)
  let ry = Math.abs(ry0)
  const c = Math.cos((phi * Math.PI) / 180)
  const s = Math.sin((phi * Math.PI) / 180)
  const dx = (p0[0] - p1[0]) / 2
  const dy = (p0[1] - p1[1]) / 2
  const x1 = c * dx + s * dy
  const y1 = -s * dx + c * dy
  const lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
  if (lam > 1) {
    rx *= Math.sqrt(lam)
    ry *= Math.sqrt(lam)
  }
  const num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
  const den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
  const coef = Math.sqrt(Math.max(num / den, 0)) * (large === sweep ? -1 : 1)
  const cx1 = (coef * rx * y1) / ry
  const cy1 = (-coef * ry * x1) / rx
  const cx = c * cx1 - s * cy1 + (p0[0] + p1[0]) / 2
  const cy = s * cx1 + c * cy1 + (p0[1] + p1[1]) / 2
  const a0 = Math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
  let da = Math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx) - a0
  if (sweep && da < 0) da += 2 * Math.PI
  else if (!sweep && da > 0) da -= 2 * Math.PI
  const k = Math.max(8, Math.floor((Math.abs(da) / (2 * Math.PI)) * 256))
  const out: Point2[] = []
  for (let i = 1; i <= k; i++) {
    const a = a0 + (da * i) / k
    const ex = rx * Math.cos(a)
    const ey = ry * Math.sin(a)
    out.push([c * ex - s * ey + cx, s * ex + c * ey + cy])
  }
  return out
}

/**
 * Flatten SVG path data into its subpaths: lines as they are, cubic and quadratic Béziers and elliptical arcs sampled
 * finely, relative commands resolved, and `S` and `T` reflecting the last control point.
 *
 * @param d - The path data (the `d` attribute).
 * @returns One stroke per subpath, closed where it ends with `Z`.
 * @throws MdmlError with code `value` for a malformed number.
 * @example
 * ```ts
 * import { flattenSvgPath } from 'mdml-engine/graphics'
 * flattenSvgPath('M0 0H10V10Z').map((s) => [s.points.length, s.closed]) // => [[3, true]]
 * ```
 */
export function flattenSvgPath(d: string): Stroke[] {
  const strokes: Stroke[] = []
  let pts: Point2[] = []
  let cur: Point2 = [0, 0]
  let start: Point2 = [0, 0]
  let ctrl: Point2 | null = null
  let prev = ''
  const flush = (closed: boolean): void => {
    if (pts.length > 1) strokes.push({ points: pts, closed })
  }
  for (const [cmd, a] of tokens(d)) {
    const up = cmd.toUpperCase()
    const ox = cmd === up ? 0 : cur[0]
    const oy = cmd === up ? 0 : cur[1]
    const at = (i: number): Point2 => [ox + (a[i] ?? 0), oy + (a[i + 1] ?? 0)]
    const reflect = (kinds: string): Point2 =>
      ctrl !== null && kinds.includes(prev) ? [2 * cur[0] - ctrl[0], 2 * cur[1] - ctrl[1]] : cur
    if (up === 'M') {
      flush(false)
      cur = at(0)
      start = cur
      pts = [cur]
    } else if (up === 'Z') {
      flush(true)
      cur = start
      pts = [cur]
    } else if (up === 'L') {
      cur = at(0)
      pts.push(cur)
    } else if (up === 'H') {
      cur = [ox + (a[0] ?? 0), cur[1]]
      pts.push(cur)
    } else if (up === 'V') {
      cur = [cur[0], oy + (a[0] ?? 0)]
      pts.push(cur)
    } else if (up === 'C' || up === 'S') {
      const c1 = up === 'C' ? at(0) : reflect('CS')
      const c2 = up === 'C' ? at(2) : at(0)
      const end = up === 'C' ? at(4) : at(2)
      pts.push(...cubic(cur, c1, c2, end))
      ctrl = c2
      cur = end
    } else if (up === 'Q' || up === 'T') {
      const q = up === 'Q' ? at(0) : reflect('QT')
      const end = up === 'Q' ? at(2) : at(0)
      const lift = (p: Point2): Point2 => [p[0] + (2 / 3) * (q[0] - p[0]), p[1] + (2 / 3) * (q[1] - p[1])]
      pts.push(...cubic(cur, lift(cur), lift(end), end))
      ctrl = q
      cur = end
    } else if (up === 'A') {
      const end = at(5)
      pts.push(...arc(cur, a[0] ?? 0, a[1] ?? 0, a[2] ?? 0, (a[3] ?? 0) !== 0, (a[4] ?? 0) !== 0, end))
      cur = end
    }
    prev = up
  }
  flush(false)
  return strokes
}

function ellipse(cx: number, cy: number, rx: number, ry: number): Point2[] {
  return Array.from({ length: 256 }, (_, i): Point2 => {
    const a = (2 * Math.PI * i) / 256
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)]
  })
}

/**
 * Flatten SVG elements (paths, circles, ellipses, lines, rounded rectangles, polylines and polygons) into strokes.
 *
 * @param elements - The elements, each a tag and its attributes.
 * @returns The strokes, in element order.
 * @throws MdmlError with code `value` for an unsupported element.
 * @example
 * ```ts
 * import { flattenSvg } from 'mdml-engine/graphics'
 * flattenSvg([['circle', { cx: '0', cy: '0', r: '1' }], ['line', { x1: '0', y1: '0', x2: '1', y2: '0' }]]).map((s) => s.closed) // => [true, false]
 * ```
 */
export function flattenSvg(elements: readonly SvgElement[]): Stroke[] {
  return elements.flatMap(([tag, attrs]): Stroke[] => {
    const f = (k: string): number => Number(attrs[k] ?? 0)
    switch (tag) {
      case 'path':
        return flattenSvgPath(attrs['d'] ?? '')
      case 'circle':
        return [{ points: ellipse(f('cx'), f('cy'), f('r'), f('r')), closed: true }]
      case 'ellipse':
        return [{ points: ellipse(f('cx'), f('cy'), f('rx'), f('ry')), closed: true }]
      case 'line':
        return [
          {
            points: [
              [f('x1'), f('y1')],
              [f('x2'), f('y2')],
            ],
            closed: false,
          },
        ]
      case 'rect': {
        const [x, y, w, h, r] = [f('x'), f('y'), f('width'), f('height'), f('rx')]
        return flattenSvgPath(
          `M${x + r} ${y}H${x + w - r}A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + h - r}` +
            `A${r} ${r} 0 0 1 ${x + w - r} ${y + h}H${x + r}A${r} ${r} 0 0 1 ${x} ${y + h - r}V${y + r}` +
            `A${r} ${r} 0 0 1 ${x + r} ${y}Z`,
        )
      }
      case 'polyline':
      case 'polygon': {
        const v = (attrs['points'] ?? '')
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(Number)
        const points = Array.from(
          { length: Math.floor(v.length / 2) },
          (_, i): Point2 => [v[2 * i] ?? 0, v[2 * i + 1] ?? 0],
        )
        return [{ points, closed: tag === 'polygon' }]
      }
      default:
        throw new MdmlError('value', `flattenSvg: unsupported element '${tag}'`)
    }
  })
}

const pathLength = (p: readonly Point2[]): number => p.reduce((s, q, i) => (i > 0 ? s + dist(p[i - 1] ?? q, q) : 0), 0)

// A closed loop through a stroke: a closed contour as it is, an open stroke traced out and back.
const loop = ({ points, closed }: Stroke): Point2[] =>
  closed ? [...points] : [...points, ...points.slice(1, -1).reverse()]

/**
 * Join strokes into one closed path. The longest loop is the trunk; each remaining loop is attached, nearest first, by
 * a bridge from the closest point of the path so far to the closest point of the loop, and the path runs once round
 * the loop and back along the bridge, so each join adds only a doubled straight segment. Open strokes are traced out
 * and back.
 *
 * @param strokes - The strokes.
 * @returns The closed path (its last point joins back to its first), or an empty path for no strokes.
 * @example
 * ```ts
 * import { stitchStrokes } from 'mdml-engine/graphics'
 * stitchStrokes([{ points: [[0, 0], [1, 0]], closed: false }]) // => [[0, 0], [1, 0]]
 * ```
 */
export function stitchStrokes(strokes: readonly Stroke[]): Point2[] {
  const loops = strokes.filter((s) => s.points.length > 0).map(loop)
  if (loops.length === 0) return []
  const lengths = loops.map(pathLength)
  let path = loops.splice(lengths.indexOf(Math.max(...lengths)), 1)[0] ?? []
  while (loops.length > 0) {
    let best = { d: Number.POSITIVE_INFINITY, i: 0, j: 0, k: 0 }
    for (const [k, l] of loops.entries())
      for (let i = 0; i < path.length; i++)
        for (let j = 0; j < l.length; j++) {
          const p = path[i] ?? [0, 0]
          const q = l[j] ?? [0, 0]
          const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
          if (d < best.d) best = { d, i, j, k }
        }
    const l = loops.splice(best.k, 1)[0] ?? []
    const tour = [...l.slice(best.j), ...l.slice(0, best.j), l[best.j] ?? [0, 0]]
    path = [...path.slice(0, best.i + 1), ...tour, ...path.slice(best.i)]
  }
  return path
}

/**
 * Points spaced uniformly by arc length along the closed path through `points` (the last point joins the first).
 *
 * @param points - The path.
 * @param n - How many points.
 * @returns `n` points, the first at the path's start.
 * @throws MdmlError with code `value` for `n < 1`.
 * @example
 * ```ts
 * import { resampleClosed } from 'mdml-engine/graphics'
 * resampleClosed([[0, 0], [2, 0], [2, 2], [0, 2]], 4) // => [[0, 0], [2, 0], [2, 2], [0, 2]]
 * ```
 */
export function resampleClosed(points: readonly Point2[], n: number): Point2[] {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'resampleClosed: n must be a positive integer')
  const p = points.filter((q, i) => i === 0 || dist(points[i - 1] ?? q, q) > 1e-12)
  const first = p[0] ?? [0, 0]
  if (p.length < 2) return Array.from({ length: n }, (): Point2 => [first[0], first[1]])
  const ring = [...p, first]
  const s = [0]
  for (let i = 1; i < ring.length; i++) s.push((s[i - 1] ?? 0) + dist(ring[i - 1] ?? first, ring[i] ?? first))
  const total = s[s.length - 1] ?? 0
  const out: Point2[] = []
  let j = 0
  for (let i = 0; i < n; i++) {
    const t = (total * i) / n
    while ((s[j + 1] ?? total) < t) j++
    const s0 = s[j] ?? 0
    const s1 = s[j + 1] ?? s0
    const u = s1 > s0 ? (t - s0) / (s1 - s0) : 0
    const a = ring[j] ?? first
    const b = ring[j + 1] ?? first
    out.push([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])])
  }
  return out
}

/**
 * Centre points on their centroid and scale them to unit maximum radius, flipping y first when the source has y
 * pointing down (SVG, images, fonts in screen coordinates).
 *
 * @param points - The points.
 * @param flipY - Whether to flip y (default false).
 * @returns The normalised points.
 * @example
 * ```ts
 * import { normaliseOutline } from 'mdml-engine/graphics'
 * normaliseOutline([[0, 0], [4, 0]]) // => [[-1, 0], [1, 0]]
 * ```
 */
export function normaliseOutline(points: readonly Point2[], flipY = false): Point2[] {
  const sy = flipY ? -1 : 1
  const n = Math.max(1, points.length)
  const mx = points.reduce((s, p) => s + p[0], 0) / n
  const my = points.reduce((s, p) => s + sy * p[1], 0) / n
  const centred = points.map(([x, y]): Point2 => [x - mx, sy * y - my])
  const r = Math.max(0, ...centred.map(([x, y]) => Math.hypot(x, y))) || 1
  return centred.map(([x, y]): Point2 => [x / r, y / r])
}

/**
 * The outline of SVG elements (y down) as one closed path of `n` points, uniform in arc length, centred, of unit
 * radius and y up.
 *
 * @param elements - The SVG elements.
 * @param n - How many points.
 * @returns The outline.
 * @example
 * ```ts
 * import { svgOutline } from 'mdml-engine/graphics'
 * svgOutline([['rect', { x: '0', y: '0', width: '2', height: '2' }]], 4).length // => 4
 * ```
 */
export function svgOutline(elements: readonly SvgElement[], n: number): Point2[] {
  return normaliseOutline(resampleClosed(stitchStrokes(flattenSvg(elements)), n), true)
}

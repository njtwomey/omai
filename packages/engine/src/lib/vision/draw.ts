/**
 * Drawing into images for exposition, OpenCV style but functional (each call returns a new image): lines by OpenCV's
 * Bresenham line iterator (8- or 4-connected) and anti-aliased lines by Wu's algorithm, rectangles, circles and
 * ellipses (outlined or filled), polylines and filled polygons by even-odd scanlines through pixel centres, and the
 * composite views `drawContours`, `drawKeypoints` and `drawMatches`. Thick strokes are drawn with a round brush; text
 * is out of scope.
 */

import { OmaiError } from '../base/index.ts'
import type { Point2 } from '../numerics/geometry/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Raster, rasterOf, rasterTensor } from './_image.ts'
import type { Contour } from './contours.ts'
import type { Keypoint } from './fast.ts'
import type { DMatch } from './matching.ts'

/** A colour: one value for every channel, or one per channel. */
export type Colour = number | readonly number[]

/** Options of the stroked shapes. */
export interface StrokeOptions {
  /** Line width in pixels (default 1); a negative thickness fills closed shapes. */
  readonly thickness?: number
  /** 8- or 4-connected lines, or 'aa' for anti-aliased (default 8). */
  readonly lineType?: 4 | 8 | 'aa'
}

/** A mutable canvas over a copy of the image's samples. */
class Canvas {
  readonly data: Float64Array
  constructor(
    private readonly r: Raster,
    colour: Colour,
  ) {
    this.data = Float64Array.from(r.data)
    this.colour = Array.from({ length: r.c }, (_, k) =>
      typeof colour === 'number' ? colour : (colour[k] ?? colour[0] ?? 0),
    )
  }
  readonly colour: number[]
  get h(): number {
    return this.r.h
  }
  get w(): number {
    return this.r.w
  }
  /** Set a pixel (blended with weight alpha in [0, 1]); outside the image nothing happens. */
  set(x: number, y: number, alpha = 1): void {
    if (x < 0 || y < 0 || x >= this.r.w || y >= this.r.h) return
    const base = (y * this.r.w + x) * this.r.c
    for (let k = 0; k < this.r.c; k++) {
      const old = this.data[base + k] ?? 0
      this.data[base + k] = alpha >= 1 ? (this.colour[k] ?? 0) : old + alpha * ((this.colour[k] ?? 0) - old)
    }
  }
  hline(x1: number, x2: number, y: number): void {
    for (let x = Math.max(0, x1); x <= Math.min(this.r.w - 1, x2); x++) this.set(x, y)
  }
  disc(cx: number, cy: number, radius: number): void {
    const r2 = radius * radius
    for (let dy = -Math.floor(radius); dy <= Math.floor(radius); dy++) {
      const half = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)))
      this.hline(cx - half, cx + half, cy + dy)
    }
  }
  tensor(): Tensor {
    return rasterTensor({ ...this.r, data: this.data })
  }
}

/**
 * The pixels of a line segment by OpenCV's LineIterator: Bresenham's algorithm between the rounded end points, run
 * left to right, stepping 8-connected (diagonals allowed) or 4-connected.
 *
 * @param p1 - One end (x, y).
 * @param p2 - The other end.
 * @param connectivity - 8 (default) or 4.
 * @returns The pixels (x, y) in order.
 * @concept rasterisation
 * @example
 * ```ts
 * import { linePixels } from 'omai-engine/vision'
 * linePixels([0, 0], [3, 1]) // => [[0, 0], [1, 0], [2, 1], [3, 1]]
 * ```
 */
export function linePixels(
  p1: readonly [number, number],
  p2: readonly [number, number],
  connectivity: 4 | 8 = 8,
): [number, number][] {
  let [x1, y1] = [Math.round(p1[0]), Math.round(p1[1])]
  let [x2, y2] = [Math.round(p2[0]), Math.round(p2[1])]
  if (x2 < x1) {
    ;[x1, y1, x2, y2] = [x2, y2, x1, y1]
  }
  let dx = x2 - x1
  let dy = y2 - y1
  let sx = 1
  let sy = 1
  if (dy < 0) {
    dy = -dy
    sy = -1
  }
  const vert = dy > dx
  if (vert) {
    ;[dx, dy] = [dy, dx]
    ;[sx, sy] = [sy, sx]
  }
  let err: number
  let plusDelta: number
  const minusDelta = -(dy + dy)
  // Each step moves by `minus` and, when the error is negative, also by `plus` (in the major and minor axes).
  let minus: [number, number] = [sx, 0]
  let plus: [number, number] = [0, sy]
  let count: number
  if (connectivity === 8) {
    err = dx - (dy + dy)
    plusDelta = dx + dx
    count = dx + 1
  } else {
    err = 0
    plusDelta = dx + dx + (dy + dy)
    plus = [-sx, sy]
    count = dx + dy + 1
  }
  if (vert) {
    minus = [minus[1], minus[0]]
    plus = [plus[1], plus[0]]
  }
  const out: [number, number][] = []
  let x = x1
  let y = y1
  for (let i = 0; i < count; i++) {
    out.push([x, y])
    const neg = err < 0
    err += minusDelta + (neg ? plusDelta : 0)
    x += minus[0] + (neg ? plus[0] : 0)
    y += minus[1] + (neg ? plus[1] : 0)
  }
  return out
}

function stroke(c: Canvas, a: Point2, b: Point2, options: StrokeOptions): void {
  const t = options.thickness ?? 1
  const type = options.lineType ?? 8
  if (type === 'aa' && t <= 1) {
    wu(c, a, b)
    return
  }
  const pts = linePixels(a, b, type === 4 ? 4 : 8)
  if (t <= 1) for (const [x, y] of pts) c.set(x, y)
  else for (const [x, y] of pts) c.disc(x, y, t / 2)
}

/** Xiaolin Wu's anti-aliased line: two pixels per step, weighted by the fractional distance. */
function wu(c: Canvas, a: Point2, b: Point2): void {
  let [x0, y0] = a
  let [x1, y1] = b
  const steep = Math.abs(y1 - y0) > Math.abs(x1 - x0)
  if (steep) {
    ;[x0, y0, x1, y1] = [y0, x0, y1, x1]
  }
  if (x0 > x1) {
    ;[x0, y0, x1, y1] = [x1, y1, x0, y0]
  }
  const dx = x1 - x0
  const gradient = dx === 0 ? 1 : (y1 - y0) / dx
  const plot = (x: number, y: number, w: number): void => (steep ? c.set(y, x, w) : c.set(x, y, w))
  let y = y0
  for (let x = Math.round(x0); x <= Math.round(x1); x++) {
    const fy = Math.floor(y)
    const frac = y - fy
    plot(x, fy, 1 - frac)
    plot(x, fy + 1, frac)
    y += gradient
  }
}

/** Fill a polygon by even-odd scanlines through pixel centres. */
function fill(c: Canvas, polys: readonly (readonly Point2[])[]): void {
  let y0 = Number.POSITIVE_INFINITY
  let y1 = Number.NEGATIVE_INFINITY
  for (const poly of polys)
    for (const [, y] of poly) {
      y0 = Math.min(y0, y)
      y1 = Math.max(y1, y)
    }
  for (let y = Math.max(0, Math.ceil(y0)); y <= Math.min(c.h - 1, Math.floor(y1)); y++) {
    const xs: number[] = []
    for (const poly of polys)
      for (let i = 0; i < poly.length; i++) {
        const [ax, ay] = poly[i] as Point2
        const [bx, by] = poly[(i + 1) % poly.length] as Point2
        if (ay === by) continue
        if ((y >= ay && y < by) || (y >= by && y < ay)) xs.push(ax + ((y - ay) * (bx - ax)) / (by - ay))
      }
    xs.sort((p, q) => p - q)
    for (let k = 0; k + 1 < xs.length; k += 2) c.hline(Math.ceil(xs[k] ?? 0), Math.floor(xs[k + 1] ?? 0), y)
  }
  // Boundary pixels, so thin polygons do not vanish.
  for (const poly of polys)
    for (let i = 0; i < poly.length; i++) stroke(c, poly[i] as Point2, poly[(i + 1) % poly.length] as Point2, {})
}

/**
 * Draw a line segment (cv2.line): OpenCV's Bresenham iterator for 8- or 4-connected lines between rounded end points
 * (pixels outside the image are skipped rather than clipped), Wu's algorithm for anti-aliased ones, a round brush for
 * thick ones.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param p1 - Start (x, y).
 * @param p2 - End (x, y).
 * @param colour - The colour.
 * @param options - Thickness and line type.
 * @returns A new image with the line drawn.
 * @concept rasterisation
 * @example
 * ```ts
 * import { line } from 'omai-engine/vision'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(line([[0, 0, 0], [0, 0, 0]], [0, 0], [2, 1], 1)) // => [[1, 1, 0], [0, 0, 1]]
 * ```
 */
export function line(
  x: TensorLike,
  p1: readonly [number, number],
  p2: readonly [number, number],
  colour: Colour,
  options: StrokeOptions = {},
): Tensor {
  const c = new Canvas(rasterOf(x, 'line'), colour)
  stroke(c, p1, p2, options)
  return c.tensor()
}

/**
 * Draw an open or closed polyline (cv2.polylines), or fill it when the thickness is negative.
 *
 * @param x - Image.
 * @param points - Vertices (x, y).
 * @param closed - Join the last vertex to the first.
 * @param colour - The colour.
 * @param options - Thickness and line type.
 * @returns A new image with the polyline drawn.
 * @concept rasterisation
 * @example
 * ```ts
 * import { polylines } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(polylines(Array.from({ length: 4 }, () => [0, 0, 0, 0]), [[0, 0], [3, 0], [3, 3]], false, 1)).reduce((a, b) => a + b, 0) // => 7
 * ```
 */
export function polylines(
  x: TensorLike,
  points: readonly (readonly [number, number])[],
  closed: boolean,
  colour: Colour,
  options: StrokeOptions = {},
): Tensor {
  const c = new Canvas(rasterOf(x, 'polylines'), colour)
  if ((options.thickness ?? 1) < 0) fill(c, [points])
  else {
    const n = points.length
    for (let i = 0; i + 1 < n; i++) stroke(c, points[i] as Point2, points[i + 1] as Point2, options)
    if (closed && n > 2) stroke(c, points[n - 1] as Point2, points[0] as Point2, options)
  }
  return c.tensor()
}

/**
 * Fill polygons (cv2.fillPoly) by the even-odd rule: pixels whose centres lie inside an odd number of the polygons'
 * crossings, plus the polygons' outlines.
 *
 * @param x - Image.
 * @param polygons - Polygons, each a list of vertices (x, y).
 * @param colour - The colour.
 * @returns A new image with the polygons filled.
 * @concept rasterisation
 * @example
 * ```ts
 * import { fillPoly } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(fillPoly(Array.from({ length: 5 }, () => new Array(5).fill(0)), [[[1, 1], [3, 1], [3, 3], [1, 3]]], 1)).reduce((a, b) => a + b, 0) // => 9
 * ```
 */
export function fillPoly(
  x: TensorLike,
  polygons: readonly (readonly (readonly [number, number])[])[],
  colour: Colour,
): Tensor {
  const c = new Canvas(rasterOf(x, 'fillPoly'), colour)
  fill(c, polygons)
  return c.tensor()
}

/**
 * Draw a rectangle (cv2.rectangle) from one corner to the opposite, outlined or filled (negative thickness).
 *
 * @param x - Image.
 * @param p1 - One corner (x, y).
 * @param p2 - The opposite corner.
 * @param colour - The colour.
 * @param options - Thickness and line type.
 * @returns A new image with the rectangle drawn.
 * @concept rasterisation
 * @example
 * ```ts
 * import { rectangle } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(rectangle(Array.from({ length: 5 }, () => new Array(5).fill(0)), [1, 1], [3, 3], 1)).reduce((a, b) => a + b, 0) // => 8
 * ```
 */
export function rectangle(
  x: TensorLike,
  p1: readonly [number, number],
  p2: readonly [number, number],
  colour: Colour,
  options: StrokeOptions = {},
): Tensor {
  const corners: Point2[] = [
    [p1[0], p1[1]],
    [p2[0], p1[1]],
    [p2[0], p2[1]],
    [p1[0], p2[1]],
  ]
  if ((options.thickness ?? 1) < 0) {
    const c = new Canvas(rasterOf(x, 'rectangle'), colour)
    const [xa, xb] = [Math.min(p1[0], p2[0]), Math.max(p1[0], p2[0])]
    for (let y = Math.min(p1[1], p2[1]); y <= Math.max(p1[1], p2[1]); y++) c.hline(Math.round(xa), Math.round(xb), y)
    return c.tensor()
  }
  return polylines(x, corners, true, colour, options)
}

/**
 * The vertices of an elliptic arc as a polygon (as cv2.ellipse2Poly): points every delta degrees from the start to
 * the end angle on the ellipse with the given half-axes, rotated by angle about the centre and rounded, repeats
 * dropped.
 *
 * @param center - The centre (x, y).
 * @param axes - The half-axes.
 * @param angle - The rotation in degrees.
 * @param arcStart - The start angle in degrees (default 0).
 * @param arcEnd - The end angle in degrees (default 360).
 * @param delta - The angular step in degrees (default 5).
 * @returns The vertices (x, y).
 * @concept rasterisation
 * @example
 * ```ts
 * import { ellipsePoints } from 'omai-engine/vision'
 * ellipsePoints([0, 0], [2, 1], 0, 0, 180, 90) // => [[2, 0], [0, 1], [-2, 0]]
 * ```
 */
export function ellipsePoints(
  center: readonly [number, number],
  axes: readonly [number, number],
  angle: number,
  arcStart = 0,
  arcEnd = 360,
  delta = 5,
): [number, number][] {
  const a = (angle * Math.PI) / 180
  const ca = Math.cos(a)
  const sa = Math.sin(a)
  const out: [number, number][] = []
  for (let t = arcStart; ; t += delta) {
    const tt = Math.min(t, arcEnd)
    const th = (tt * Math.PI) / 180
    const ex = axes[0] * Math.cos(th)
    const ey = axes[1] * Math.sin(th)
    const p: [number, number] = [Math.round(center[0] + ex * ca - ey * sa), Math.round(center[1] + ex * sa + ey * ca)]
    const last = out[out.length - 1]
    if (last === undefined || last[0] !== p[0] || last[1] !== p[1]) out.push(p)
    if (tt >= arcEnd) break
  }
  return out
}

/**
 * Draw a circle (cv2.circle): the outline of radius r around the centre, or a filled disc for a negative thickness.
 *
 * @param x - Image.
 * @param center - The centre (x, y).
 * @param radius - The radius.
 * @param colour - The colour.
 * @param options - Thickness and line type.
 * @returns A new image with the circle drawn.
 * @concept rasterisation
 * @example
 * ```ts
 * import { circle } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(circle(Array.from({ length: 7 }, () => new Array(7).fill(0)), [3, 3], 2, 1, { thickness: -1 })).reduce((a, b) => a + b, 0) // => 13
 * ```
 */
export function circle(
  x: TensorLike,
  center: readonly [number, number],
  radius: number,
  colour: Colour,
  options: StrokeOptions = {},
): Tensor {
  if (!(radius >= 0)) throw new OmaiError('value', 'circle: the radius must be non-negative')
  const c = new Canvas(rasterOf(x, 'circle'), colour)
  const cx = Math.round(center[0])
  const cy = Math.round(center[1])
  if ((options.thickness ?? 1) < 0) {
    c.disc(cx, cy, radius)
    return c.tensor()
  }
  // The midpoint circle, its eight octants, drawn with the brush.
  let px = Math.round(radius)
  let py = 0
  let err = 1 - px
  const t = options.thickness ?? 1
  const plot = (u: number, v: number): void => (t <= 1 ? c.set(u, v) : c.disc(u, v, t / 2))
  while (px >= py) {
    for (const [u, v] of [
      [px, py],
      [py, px],
      [-py, px],
      [-px, py],
      [-px, -py],
      [-py, -px],
      [py, -px],
      [px, -py],
    ] as const)
      plot(cx + u, cy + v)
    py++
    if (err < 0) err += 2 * py + 1
    else {
      px--
      err += 2 * (py - px) + 1
    }
  }
  return c.tensor()
}

/**
 * Draw an ellipse or elliptic arc (cv2.ellipse) through its polygon approximation, outlined or filled.
 *
 * @param x - Image.
 * @param center - The centre (x, y).
 * @param axes - The half-axes.
 * @param angle - The rotation in degrees.
 * @param colour - The colour.
 * @param options - The arc (start and end angles in degrees, default the whole ellipse), thickness and line type.
 * @returns A new image with the ellipse drawn.
 * @concept rasterisation
 * @example
 * ```ts
 * import { ellipse } from 'omai-engine/vision'
 * ellipse(Array.from({ length: 9 }, () => new Array(9).fill(0)), [4, 4], [3, 2], 30, 1).shape // => [9, 9]
 * ```
 */
export function ellipse(
  x: TensorLike,
  center: readonly [number, number],
  axes: readonly [number, number],
  angle: number,
  colour: Colour,
  options: StrokeOptions & { readonly startAngle?: number; readonly endAngle?: number } = {},
): Tensor {
  const pts = ellipsePoints(center, axes, angle, options.startAngle ?? 0, options.endAngle ?? 360)
  const full = (options.endAngle ?? 360) - (options.startAngle ?? 0) >= 360
  return polylines(x, pts, full, colour, options)
}

/**
 * Draw contours (cv2.drawContours): every contour, or the one at an index, as a closed polyline or filled (negative
 * thickness).
 *
 * @param x - Image.
 * @param contours - Contours from {@link findContours}.
 * @param index - The contour to draw, or -1 for all.
 * @param colour - The colour.
 * @param options - Thickness and line type.
 * @returns A new image with the contours drawn.
 * @concept contours
 * @example
 * ```ts
 * import { drawContours } from 'omai-engine/vision'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(drawContours(Array.from({ length: 4 }, () => new Array(4).fill(0)), [[[1, 1], [2, 1], [2, 2], [1, 2]]], -1, 1)).reduce((a, b) => a + b, 0) // => 4
 * ```
 */
export function drawContours(
  x: TensorLike,
  contours: readonly Contour[],
  index: number,
  colour: Colour,
  options: StrokeOptions = {},
): Tensor {
  const chosen = index < 0 ? contours : [contours[index] ?? []]
  if ((options.thickness ?? 1) < 0) return fillPoly(x, chosen, colour)
  let img: TensorLike = x
  for (const c of chosen) img = polylines(img, c, true, colour, options)
  return img as Tensor
}

/**
 * Draw keypoints (cv2.drawKeypoints): a circle of the keypoint's radius (at least 3) around each, and its orientation
 * as a radius when it has one.
 *
 * @param x - Image.
 * @param keypoints - The keypoints.
 * @param colour - The colour (default 1).
 * @returns A new image with the keypoints drawn.
 * @concept keypoints
 * @example
 * ```ts
 * import { drawKeypoints } from 'omai-engine/vision'
 * drawKeypoints(Array.from({ length: 9 }, () => new Array(9).fill(0)), [{ x: 4, y: 4, size: 4, angle: 0, response: 1, octave: 0 }]).shape // => [9, 9]
 * ```
 */
export function drawKeypoints(x: TensorLike, keypoints: readonly Keypoint[], colour: Colour = 1): Tensor {
  let img: TensorLike = x
  for (const k of keypoints) {
    const r = Math.max(3, Math.round(k.size / 2))
    img = circle(img, [k.x, k.y], r, colour)
    if (k.angle >= 0) {
      const a = (k.angle * Math.PI) / 180
      img = line(img, [k.x, k.y], [k.x + r * Math.cos(a), k.y + r * Math.sin(a)], colour)
    }
  }
  return img as Tensor
}

/**
 * Draw matches between two images side by side (cv2.drawMatches): the images on one canvas (the shorter padded with
 * zeros, a grey image beside a colour one made colour), each matched keypoint circled and joined by a line.
 *
 * @param x1 - The first image.
 * @param keypoints1 - Its keypoints (the query side).
 * @param x2 - The second image.
 * @param keypoints2 - Its keypoints (the train side).
 * @param matches - The matches.
 * @param colour - The colour (default 1).
 * @returns The canvas `[max(H1, H2), W1 + W2]` (channels as the inputs).
 * @concept descriptor-matching
 * @example
 * ```ts
 * import { drawMatches } from 'omai-engine/vision'
 * const a = Array.from({ length: 6 }, () => new Array(6).fill(0))
 * const k = [{ x: 2, y: 2, size: 2, angle: -1, response: 1, octave: 0 }]
 * drawMatches(a, k, a, k, [{ queryIdx: 0, trainIdx: 0, distance: 0 }]).shape // => [6, 12]
 * ```
 */
export function drawMatches(
  x1: TensorLike,
  keypoints1: readonly Keypoint[],
  x2: TensorLike,
  keypoints2: readonly Keypoint[],
  matches: readonly DMatch[],
  colour: Colour = 1,
): Tensor {
  const a = rasterOf(x1, 'drawMatches')
  const b = rasterOf(x2, 'drawMatches')
  const c = Math.max(a.c, b.c)
  const h = Math.max(a.h, b.h)
  const w = a.w + b.w
  const data = new Float64Array(h * w * c)
  const paste = (r: Raster, ox: number): void => {
    for (let y = 0; y < r.h; y++)
      for (let xx = 0; xx < r.w; xx++)
        for (let k = 0; k < c; k++)
          data[(y * w + xx + ox) * c + k] = r.data[(y * r.w + xx) * r.c + (r.c === 1 ? 0 : k)] ?? 0
  }
  paste(a, 0)
  paste(b, a.w)
  let img: TensorLike = rasterTensor({ h, w, c, rank: c === 1 ? a.rank : 3, data })
  for (const m of matches) {
    const p = keypoints1[m.queryIdx]
    const q = keypoints2[m.trainIdx]
    if (p === undefined || q === undefined) continue
    const qx = q.x + a.w
    img = circle(img, [p.x, p.y], Math.max(3, Math.round(p.size / 2)), colour)
    img = circle(img, [qx, q.y], Math.max(3, Math.round(q.size / 2)), colour)
    img = line(img, [p.x, p.y], [qx, q.y], colour)
  }
  return img as Tensor
}

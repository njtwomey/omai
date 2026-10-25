/**
 * A small software rasteriser that draws a scene into RGBA pixels without a browser, used for pixel observations in
 * reinforcement learning, animation frames in Node and image tests.
 */

import { clamp, MdmlError } from '../base/index.ts'
import { palettes, parsePaint, type Rgba } from './paint.ts'
import { projectLayers } from './projection.ts'
import type { Paint, Pixels, Scene, SceneNode, Style, ThemeToken, Transform } from './types.ts'

type Point = readonly [number, number]
/** A 2-D affine map `[a, b, c, d, e, f]`: `(x, y) -> (a x + c y + e, b x + d y + f)`. */
type Affine = readonly [number, number, number, number, number, number]

function apply(m: Affine, [x, y]: Point): Point {
  return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}

function compose(m: Affine, t: Transform): Affine {
  const s = t.scale ?? 1
  const theta = ((t.rotate ?? 0) * Math.PI) / 180
  const c = s * Math.cos(theta)
  const sn = s * Math.sin(theta)
  const [tx, ty] = t.translate ?? [0, 0]
  // m * [c -sn tx; sn c ty; 0 0 1]
  return [
    m[0] * c + m[2] * sn,
    m[1] * c + m[3] * sn,
    -m[0] * sn + m[2] * c,
    -m[1] * sn + m[3] * c,
    m[0] * tx + m[2] * ty + m[4],
    m[1] * tx + m[3] * ty + m[5],
  ]
}

function linearScale(m: Affine): number {
  return Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2]))
}

/** A canvas of premultiplied RGBA floats (0 to 1). */
interface Canvas {
  readonly width: number
  readonly height: number
  readonly data: Float32Array
}

/** Pixels covered by a shape, inside a clipped bounding box. */
interface Mask {
  readonly x0: number
  readonly y0: number
  readonly w: number
  readonly h: number
  readonly bits: Uint8Array
}

function maskFor(canvas: Canvas, rings: readonly (readonly Point[])[], pad = 0): Mask | undefined {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const ring of rings) {
    for (const [x, y] of ring) {
      if (x < x0) x0 = x
      if (y < y0) y0 = y
      if (x > x1) x1 = x
      if (y > y1) y1 = y
    }
  }
  const ix0 = Math.max(0, Math.floor(x0 - pad))
  const iy0 = Math.max(0, Math.floor(y0 - pad))
  const ix1 = Math.min(canvas.width, Math.ceil(x1 + pad))
  const iy1 = Math.min(canvas.height, Math.ceil(y1 + pad))
  if (!(ix1 > ix0 && iy1 > iy0)) return undefined
  const w = ix1 - ix0
  const h = iy1 - iy0
  return { x0: ix0, y0: iy0, w, h, bits: new Uint8Array(w * h) }
}

// Scanline fill with the nonzero winding rule, sampling pixel centres; covered pixels are set in the mask.
function scan(mask: Mask, rings: readonly (readonly Point[])[]): void {
  const crossings: { x: number; dir: number }[] = []
  for (let j = 0; j < mask.h; j++) {
    const yc = mask.y0 + j + 0.5
    crossings.length = 0
    for (const ring of rings) {
      const n = ring.length
      for (let i = 0; i < n; i++) {
        const p = ring[i] as Point
        const q = ring[(i + 1) % n] as Point
        const up = p[1] <= yc && q[1] > yc
        const down = q[1] <= yc && p[1] > yc
        if (!up && !down) continue
        crossings.push({ x: p[0] + ((yc - p[1]) * (q[0] - p[0])) / (q[1] - p[1]), dir: up ? 1 : -1 })
      }
    }
    if (crossings.length < 2) continue
    crossings.sort((a, b) => a.x - b.x)
    let wind = 0
    let start = 0
    for (const c of crossings) {
      const before = wind
      wind += c.dir
      if (before === 0 && wind !== 0) start = c.x
      else if (before !== 0 && wind === 0) {
        const from = Math.max(0, Math.ceil(start - 0.5) - mask.x0)
        const to = Math.min(mask.w, Math.ceil(c.x - 0.5) - mask.x0)
        for (let i = from; i < to; i++) mask.bits[j * mask.w + i] = 1
      }
    }
  }
}

function composite(canvas: Canvas, mask: Mask, colour: Rgba, opacity: number): void {
  const a = colour[3] * opacity
  if (a <= 0) return
  const r = (colour[0] / 255) * a
  const g = (colour[1] / 255) * a
  const b = (colour[2] / 255) * a
  for (let j = 0; j < mask.h; j++) {
    for (let i = 0; i < mask.w; i++) {
      if (mask.bits[j * mask.w + i] === 0) continue
      const k = 4 * ((mask.y0 + j) * canvas.width + mask.x0 + i)
      const keep = 1 - a
      canvas.data[k] = r + (canvas.data[k] as number) * keep
      canvas.data[k + 1] = g + (canvas.data[k + 1] as number) * keep
      canvas.data[k + 2] = b + (canvas.data[k + 2] as number) * keep
      canvas.data[k + 3] = a + (canvas.data[k + 3] as number) * keep
    }
  }
}

function ellipseRing(cx: number, cy: number, rx: number, ry: number, pixelRadius: number): Point[] {
  const n = clamp(Math.ceil(pixelRadius * 1.5), 12, 360)
  const out: Point[] = []
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n
    out.push([cx + rx * Math.cos(t), cy + ry * Math.sin(t)])
  }
  return out
}

function roundedRect(x: number, y: number, w: number, h: number, radius: number): Point[] {
  const r = Math.min(radius, w / 2, h / 2)
  if (!(r > 0)) {
    return [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ]
  }
  const out: Point[] = []
  const corners: readonly (readonly [number, number, number])[] = [
    [x + w - r, y + r, -Math.PI / 2],
    [x + w - r, y + h - r, 0],
    [x + r, y + h - r, Math.PI / 2],
    [x + r, y + r, Math.PI],
  ]
  for (const [cx, cy, start] of corners) {
    for (let i = 0; i <= 8; i++) {
      const t = start + (i * Math.PI) / 16
      out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)])
    }
  }
  return out
}

// Uniform Catmull-Rom spline through the points, 8 samples per span.
function smoothed(points: readonly Point[], closed: boolean): Point[] {
  const n = points.length
  if (n < 3) return [...points]
  const at = (i: number): Point =>
    closed ? (points[((i % n) + n) % n] as Point) : (points[clamp(i, 0, n - 1)] as Point)
  const spans = closed ? n : n - 1
  const out: Point[] = []
  for (let i = 0; i < spans; i++) {
    const p0 = at(i - 1)
    const p1 = at(i)
    const p2 = at(i + 1)
    const p3 = at(i + 2)
    for (let s = 0; s < 8; s++) {
      const t = s / 8
      const t2 = t * t
      const t3 = t2 * t
      const coord = (k: 0 | 1): number =>
        0.5 *
        (2 * p1[k] +
          (p2[k] - p0[k]) * t +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
          (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3)
      out.push([coord(0), coord(1)])
    }
  }
  if (!closed) out.push(points[n - 1] as Point)
  return out
}

// Split a polyline into dashes by arc length.
function dashed(points: readonly Point[], closed: boolean, pattern: readonly number[]): Point[][] {
  const total = pattern.reduce((s, x) => s + Math.max(0, x), 0)
  if (pattern.length === 0 || !(total > 0)) return [[...points]]
  const pts = closed && points.length > 0 ? [...points, points[0] as Point] : [...points]
  const out: Point[][] = []
  let index = 0
  let left = Math.max(0, pattern[0] ?? 0)
  let on = true
  let current: Point[] = [pts[0] as Point]
  for (let i = 0; i + 1 < pts.length; i++) {
    let [ax, ay] = pts[i] as Point
    const [bx, by] = pts[i + 1] as Point
    let length = Math.hypot(bx - ax, by - ay)
    while (length > left) {
      const f = left / length
      const cut: Point = [ax + f * (bx - ax), ay + f * (by - ay)]
      if (on) {
        current.push(cut)
        out.push(current)
      }
      current = [cut]
      on = !on
      index = (index + 1) % pattern.length
      ;[ax, ay] = cut
      length -= left
      left = Math.max(0, pattern[index] ?? 0)
    }
    left -= length
    if (on) current.push([bx, by])
  }
  if (on && current.length > 1) out.push(current)
  return out
}

interface Draw {
  readonly canvas: Canvas
  readonly palette: Readonly<Record<ThemeToken, string>>
}

function strokePieces(polyline: readonly Point[], closed: boolean, width: number): Point[][] {
  const half = width / 2
  const pieces: Point[][] = []
  const n = polyline.length
  const segments = closed ? n : n - 1
  for (let i = 0; i < segments; i++) {
    const [ax, ay] = polyline[i] as Point
    const [bx, by] = polyline[(i + 1) % n] as Point
    const len = Math.hypot(bx - ax, by - ay)
    if (len === 0) continue
    const nx = (-(by - ay) / len) * half
    const ny = ((bx - ax) / len) * half
    pieces.push([
      [ax + nx, ay + ny],
      [bx + nx, by + ny],
      [bx - nx, by - ny],
      [ax - nx, ay - ny],
    ])
  }
  // Round joins, so thick polylines have no notches at their corners.
  if (width >= 2) {
    const first = closed ? 0 : 1
    const last = closed ? n : n - 1
    for (let i = first; i < last; i++) {
      const [x, y] = polyline[i] as Point
      pieces.push(ellipseRing(x, y, half, half, half))
    }
  }
  return pieces
}

// A triangular head of the given length (pixels) at the end of a polyline; the shaft is shortened in place.
function arrowHead(polyline: Point[], size: number): Point[] | undefined {
  const n = polyline.length
  if (n < 2) return undefined
  const [bx, by] = polyline[n - 1] as Point
  const [ax, ay] = polyline[n - 2] as Point
  const len = Math.hypot(bx - ax, by - ay)
  if (len === 0) return undefined
  const ux = (bx - ax) / len
  const uy = (by - ay) / len
  const back = Math.min(size, len)
  // Shorten the shaft so its end hides under the head.
  polyline[n - 1] = [bx - ux * back * 0.8, by - uy * back * 0.8]
  const baseX = bx - ux * back
  const baseY = by - uy * back
  const half = 0.4 * size
  return [
    [bx, by],
    [baseX - uy * half, baseY + ux * half],
    [baseX + uy * half, baseY - ux * half],
  ]
}

function stroke(
  draw: Draw,
  polyline: readonly Point[],
  closed: boolean,
  style: Style,
  paint: Paint,
  m: Affine,
  opacity: number,
  head: boolean,
): void {
  const colour = parsePaint(paint, draw.palette)
  if (colour === undefined) return
  const k = linearScale(m)
  const width = (style.strokeWidth ?? 1) * k
  if (!(width > 0)) return
  const mapped = polyline.map((p) => apply(m, p))
  const pieces: Point[][] = []
  const dash = style.dash?.map((d) => d * k) ?? []
  const lines = dash.length > 0 ? dashed(mapped, closed, dash) : [mapped]
  const isClosed = closed && dash.length === 0
  // The head is 3 * strokeWidth + 4 scene units long.
  const tip =
    head && !isClosed ? arrowHead(lines[lines.length - 1] ?? [], (3 * (style.strokeWidth ?? 1) + 4) * k) : undefined
  if (tip !== undefined) pieces.push(tip)
  for (const l of lines) pieces.push(...strokePieces(l, isClosed, width))
  const mask = maskFor(draw.canvas, pieces, 1)
  if (mask === undefined) return
  for (const piece of pieces) scan(mask, [piece])
  composite(draw.canvas, mask, colour, opacity)
}

function fill(draw: Draw, ring: readonly Point[], paint: Paint, m: Affine, opacity: number): void {
  const colour = parsePaint(paint, draw.palette)
  if (colour === undefined) return
  const mapped = ring.map((p) => apply(m, p))
  const mask = maskFor(draw.canvas, [mapped])
  if (mask === undefined) return
  scan(mask, [mapped])
  composite(draw.canvas, mask, colour, opacity)
}

function shape(draw: Draw, ring: readonly Point[], closed: boolean, style: Style, m: Affine, opacity: number): void {
  const o = opacity * (style.opacity ?? 1)
  const fillPaint = style.fill ?? (closed && style.stroke === undefined ? 'fg' : 'none')
  if (closed) fill(draw, ring, fillPaint, m, o)
  const strokePaint = style.stroke ?? (closed ? 'none' : 'fg')
  stroke(draw, ring, closed, style, strokePaint, m, o, false)
}

function drawImage(draw: Draw, node: Extract<SceneNode, { kind: 'image' }>, m: Affine, opacity: number): void {
  const { canvas } = draw
  const { pixels } = node
  const local: readonly Point[] = [
    [node.x, node.y],
    [node.x + node.width, node.y],
    [node.x + node.width, node.y + node.height],
    [node.x, node.y + node.height],
  ]
  const corners = local.map((p) => apply(m, p))
  const mask = maskFor(canvas, [corners])
  const det = m[0] * m[3] - m[1] * m[2]
  if (mask === undefined || det === 0 || node.width === 0 || node.height === 0) return
  const o = opacity
  for (let j = 0; j < mask.h; j++) {
    for (let i = 0; i < mask.w; i++) {
      // Invert the affine map at the pixel centre to find the image coordinate.
      const px = mask.x0 + i + 0.5 - m[4]
      const py = mask.y0 + j + 0.5 - m[5]
      const lx = (m[3] * px - m[2] * py) / det
      const ly = (-m[1] * px + m[0] * py) / det
      const u = ((lx - node.x) / node.width) * pixels.width
      const v = ((ly - node.y) / node.height) * pixels.height
      if (!(u >= 0 && u < pixels.width && v >= 0 && v < pixels.height)) continue
      const sample = (c: number): number => {
        if (node.smoothing !== true) return pixels.rgba[4 * (Math.floor(v) * pixels.width + Math.floor(u)) + c] ?? 0
        const x = clamp(u - 0.5, 0, pixels.width - 1)
        const y = clamp(v - 0.5, 0, pixels.height - 1)
        const x0 = Math.floor(x)
        const y0 = Math.floor(y)
        const x1 = Math.min(pixels.width - 1, x0 + 1)
        const y1 = Math.min(pixels.height - 1, y0 + 1)
        const fx = x - x0
        const fy = y - y0
        const at = (xx: number, yy: number): number => pixels.rgba[4 * (yy * pixels.width + xx) + c] ?? 0
        return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x1, y0)) + fy * ((1 - fx) * at(x0, y1) + fx * at(x1, y1))
      }
      const a = (sample(3) / 255) * o
      if (a <= 0) continue
      const k = 4 * ((mask.y0 + j) * canvas.width + mask.x0 + i)
      const keep = 1 - a
      for (let c = 0; c < 3; c++) canvas.data[k + c] = (sample(c) / 255) * a + (canvas.data[k + c] as number) * keep
      canvas.data[k + 3] = a + (canvas.data[k + 3] as number) * keep
    }
  }
}

function drawNode(draw: Draw, node: SceneNode, m: Affine, opacity: number): void {
  switch (node.kind) {
    case 'rect':
      shape(draw, roundedRect(node.x, node.y, node.width, node.height, node.radius ?? 0), true, node, m, opacity)
      return
    case 'circle':
      shape(draw, ellipseRing(node.cx, node.cy, node.r, node.r, node.r * linearScale(m)), true, node, m, opacity)
      return
    case 'ellipse': {
      const r = Math.max(node.rx, node.ry) * linearScale(m)
      shape(draw, ellipseRing(node.cx, node.cy, node.rx, node.ry, r), true, node, m, opacity)
      return
    }
    case 'line': {
      const o = opacity * (node.opacity ?? 1)
      const pts: Point[] = [
        [node.x1, node.y1],
        [node.x2, node.y2],
      ]
      stroke(draw, pts, false, node, node.stroke ?? 'fg', m, o, node.arrow === true)
      return
    }
    case 'path': {
      const closed = node.closed === true
      const pts = node.smooth === true ? smoothed(node.points, closed) : node.points
      if (node.arrow === true && !closed) {
        const o = opacity * (node.opacity ?? 1)
        if (node.fill !== undefined) fill(draw, pts, node.fill, m, o)
        stroke(draw, pts, false, node, node.stroke ?? 'fg', m, o, true)
        return
      }
      shape(draw, pts, closed, node, m, opacity)
      return
    }
    case 'image':
      drawImage(draw, node, m, opacity)
      return
    case 'group': {
      const inner = compose(m, node)
      const o = opacity * (node.opacity ?? 1)
      for (const child of node.children) drawNode(draw, child, inner, o)
      return
    }
    case 'layer3d':
      for (const child of projectLayers({ kind: 'scene', width: 0, height: 0, children: [node] }).children) {
        drawNode(draw, child, m, opacity)
      }
      return
    case 'text':
    case 'tex':
      // Text needs fonts, which a headless rasteriser does not have; it is skipped.
      return
  }
}

/** Options for {@link rasterise}. */
export interface RasterOptions {
  /** Output width in pixels (default the scene width, rounded). */
  readonly width?: number
  /** Output height in pixels (default the scene height, rounded). */
  readonly height?: number
  /** Theme for token colours (default `light`). */
  readonly mode?: 'light' | 'dark'
  /** Token colours overriding the mode's palette. */
  readonly palette?: Partial<Record<ThemeToken, string>>
  /** Background (default the scene's, else `bg`); `none` leaves it transparent. */
  readonly background?: Paint
  /** Supersampling factor per axis for antialiasing (default 1: crisp, exact pixel coverage at pixel centres). */
  readonly samples?: number
}

/**
 * Draw a scene into RGBA pixels with a small software rasteriser, for pixel observations in reinforcement learning,
 * animation frames in Node and image tests. Shapes are filled with the nonzero rule by sampling pixel centres, strokes
 * have butt caps and round joins, curves are flattened to polygons, smooth paths are Catmull-Rom splines, images are
 * sampled nearest-neighbour (bilinear when `smoothing`), and 3-D layers are projected first. Text and TeX labels are
 * skipped, since they need fonts. The result is deterministic: the same scene gives the same bytes everywhere.
 *
 * @param s - The scene.
 * @param options - Output size, theme and palette, background and supersampling.
 * @returns Straight RGBA pixels.
 * @throws MdmlError with code `value` for a non-positive output size or an invalid colour.
 * @concept scene-graph
 * @example
 * ```ts
 * import { rasterise, rect, scene } from 'mdml-engine/scene'
 * const px = rasterise(scene(4, 4, [rect(0, 0, 2, 4, { fill: '#ff0000' })]), { background: '#000000' })
 * Array.from(px.rgba.slice(0, 4)) // => [255, 0, 0, 255]
 * Array.from(px.rgba.slice(8, 12)) // => [0, 0, 0, 255]
 * ```
 */
export function rasterise(s: Scene, options: RasterOptions = {}): Pixels {
  const width = options.width ?? Math.round(s.width)
  const height = options.height ?? Math.round(s.height)
  const k = Math.max(1, Math.round(options.samples ?? 1))
  if (!(width > 0 && height > 0 && Number.isInteger(width) && Number.isInteger(height))) {
    throw new MdmlError('value', `rasterise: need a positive integer size, got ${width} by ${height}`)
  }
  const palette = { ...palettes[options.mode ?? 'light'], ...options.palette }
  const canvas: Canvas = { width: width * k, height: height * k, data: new Float32Array(width * height * k * k * 4) }
  const draw: Draw = { canvas, palette }
  const background = parsePaint(options.background ?? s.background ?? 'bg', palette)
  if (background !== undefined) {
    const full: Mask = {
      x0: 0,
      y0: 0,
      w: canvas.width,
      h: canvas.height,
      bits: new Uint8Array(canvas.width * canvas.height).fill(1),
    }
    composite(canvas, full, background, 1)
  }
  const sx = s.width > 0 ? (width * k) / s.width : 1
  const sy = s.height > 0 ? (height * k) / s.height : 1
  const base: Affine = [sx, 0, 0, sy, 0, 0]
  for (const node of s.children) drawNode(draw, node, base, 1)
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      for (let j = 0; j < k; j++) {
        for (let i = 0; i < k; i++) {
          const q = 4 * ((y * k + j) * canvas.width + x * k + i)
          r += canvas.data[q] as number
          g += canvas.data[q + 1] as number
          b += canvas.data[q + 2] as number
          a += canvas.data[q + 3] as number
        }
      }
      const p = 4 * (y * width + x)
      if (a > 0) {
        rgba[p] = Math.round((255 * r) / a)
        rgba[p + 1] = Math.round((255 * g) / a)
        rgba[p + 2] = Math.round((255 * b) / a)
      }
      rgba[p + 3] = Math.round((255 * a) / (k * k))
    }
  }
  return { width, height, rgba }
}

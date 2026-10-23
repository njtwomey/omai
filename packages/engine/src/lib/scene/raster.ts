import type { Paint, Scene, SceneNode } from './types.ts'

/** An RGBA image: `width * height * 4` bytes, row-major. */
export interface Raster {
  readonly width: number
  readonly height: number
  readonly rgba: Uint8ClampedArray
}

/** Options of {@link rasterise}. */
export interface RasterOptions {
  /** Output width in pixels (default the scene width). */
  readonly width?: number
  /** Output height in pixels (default the scene height). */
  readonly height?: number
  /** Theme for the theme tokens (default `'light'`). */
  readonly theme?: 'light' | 'dark'
  /** Samples per pixel along each axis for anti-aliasing (default 1: one sample at the pixel centre). */
  readonly supersample?: number
}

type Rgba = readonly [number, number, number, number]

// The tab10 palette (matplotlib), in order.
const TAB10: readonly Rgba[] = [
  [31, 119, 180, 1],
  [255, 127, 14, 1],
  [44, 160, 44, 1],
  [214, 39, 40, 1],
  [148, 103, 189, 1],
  [140, 86, 75, 1],
  [227, 119, 194, 1],
  [127, 127, 127, 1],
  [188, 189, 34, 1],
  [23, 190, 207, 1],
]

const THEMES: Readonly<Record<'light' | 'dark', Readonly<Record<string, Rgba>>>> = {
  light: {
    fg: [33, 33, 33, 1],
    bg: [255, 255, 255, 1],
    surface: [245, 245, 245, 1],
    muted: [140, 140, 140, 1],
    grid: [224, 224, 224, 1],
    accent: [31, 119, 180, 1],
  },
  dark: {
    fg: [235, 235, 235, 1],
    bg: [18, 18, 18, 1],
    surface: [36, 36, 36, 1],
    muted: [150, 150, 150, 1],
    grid: [60, 60, 60, 1],
    accent: [100, 170, 230, 1],
  },
}

/**
 * The RGBA colour of a paint (theme tokens, tab10 colours, `#rgb`, `#rrggbb`, `#rrggbbaa` and `rgba(...)`), or
 * `undefined` for `'none'`.
 *
 * @param paint - The paint.
 * @param theme - The theme for tokens (default light).
 * @returns `[r, g, b, alpha]` with channels 0 to 255 and alpha 0 to 1; an unparseable colour falls back to the
 *   theme's foreground.
 * @example
 * ```ts
 * import { paintColour } from 'whitebox-engine/scene'
 * paintColour('#ff0000') // => [255, 0, 0, 1]
 * ```
 */
export function paintColour(paint: Paint, theme: 'light' | 'dark' = 'light'): Rgba | undefined {
  if (paint === 'none') return undefined
  const token = THEMES[theme][paint]
  if (token !== undefined) return token
  if (paint.startsWith('tab')) {
    const c = TAB10[Number(paint.slice(3))]
    if (c !== undefined) return c
  }
  if (paint.startsWith('#')) {
    const hex = paint.slice(1)
    const full =
      hex.length === 3
        ? hex
            .split('')
            .map((c) => c + c)
            .join('')
        : hex
    if (/^[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(full)) {
      const n = (i: number): number => Number.parseInt(full.slice(i, i + 2), 16)
      return [n(0), n(2), n(4), full.length === 8 ? n(6) / 255 : 1]
    }
  }
  const m = /^rgba\(([^)]*)\)$/.exec(paint)
  if (m !== null) {
    const parts = (m[1] ?? '').split(',').map((p) => Number(p.trim()))
    if (parts.length === 4 && parts.every((p) => Number.isFinite(p))) {
      return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0, parts[3] ?? 1]
    }
  }
  return THEMES[theme]['fg']
}

// An affine map [a c e; b d f] from local coordinates to output pixels.
type Affine = readonly [number, number, number, number, number, number]

function compose(m: Affine, n: Affine): Affine {
  return [
    m[0] * n[0] + m[2] * n[1],
    m[1] * n[0] + m[3] * n[1],
    m[0] * n[2] + m[2] * n[3],
    m[1] * n[2] + m[3] * n[3],
    m[0] * n[4] + m[2] * n[5] + m[4],
    m[1] * n[4] + m[3] * n[5] + m[5],
  ]
}

function invert(m: Affine): Affine {
  const det = m[0] * m[3] - m[1] * m[2]
  const a = m[3] / det
  const b = -m[1] / det
  const c = -m[2] / det
  const d = m[0] / det
  return [a, b, c, d, -(a * m[4] + c * m[5]), -(b * m[4] + d * m[5])]
}

function apply(m: Affine, x: number, y: number): [number, number] {
  return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]
}

interface Canvas {
  readonly width: number
  readonly height: number
  readonly ss: number
  // Accumulated colour per sample (premultiplied by nothing: straight over-compositing per sample).
  readonly r: Float64Array
  readonly g: Float64Array
  readonly b: Float64Array
}

function blend(cv: Canvas, i: number, c: Rgba, alpha: number): void {
  const a = c[3] * alpha
  cv.r[i] = (cv.r[i] ?? 0) * (1 - a) + c[0] * a
  cv.g[i] = (cv.g[i] ?? 0) * (1 - a) + c[1] * a
  cv.b[i] = (cv.b[i] ?? 0) * (1 - a) + c[2] * a
}

// Fill every sample whose local coordinates satisfy `inside`, within the local bounding box.
function fillShape(
  cv: Canvas,
  m: Affine,
  box: readonly [number, number, number, number],
  inside: (x: number, y: number) => boolean,
  c: Rgba,
  alpha: number,
): void {
  const corners = [
    apply(m, box[0], box[1]),
    apply(m, box[2], box[1]),
    apply(m, box[0], box[3]),
    apply(m, box[2], box[3]),
  ]
  const sw = cv.width * cv.ss
  const sh = cv.height * cv.ss
  let x0 = Infinity
  let x1 = -Infinity
  let y0 = Infinity
  let y1 = -Infinity
  for (const [px, py] of corners) {
    x0 = Math.min(x0, px)
    x1 = Math.max(x1, px)
    y0 = Math.min(y0, py)
    y1 = Math.max(y1, py)
  }
  const inv = invert(m)
  const ix0 = Math.max(0, Math.floor(x0 * cv.ss))
  const ix1 = Math.min(sw - 1, Math.ceil(x1 * cv.ss))
  const iy0 = Math.max(0, Math.floor(y0 * cv.ss))
  const iy1 = Math.min(sh - 1, Math.ceil(y1 * cv.ss))
  for (let iy = iy0; iy <= iy1; iy++) {
    for (let ix = ix0; ix <= ix1; ix++) {
      const [lx, ly] = apply(inv, (ix + 0.5) / cv.ss, (iy + 0.5) / cv.ss)
      if (inside(lx, ly)) blend(cv, iy * sw + ix, c, alpha)
    }
  }
}

function segmentDistance(x: number, y: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1
  const dy = y2 - y1
  const len2 = dx * dx + dy * dy
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / len2))
  return Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
}

function strokeSegments(
  cv: Canvas,
  m: Affine,
  points: readonly (readonly [number, number])[],
  closed: boolean,
  width: number,
  c: Rgba,
  alpha: number,
): void {
  const half = Math.max(width, 1e-9) / 2
  const n = points.length
  const count = closed ? n : n - 1
  for (let i = 0; i < count; i++) {
    const [x1, y1] = points[i] ?? [0, 0]
    const [x2, y2] = points[(i + 1) % n] ?? [0, 0]
    fillShape(
      cv,
      m,
      [Math.min(x1, x2) - half, Math.min(y1, y2) - half, Math.max(x1, x2) + half, Math.max(y1, y2) + half],
      (x, y) => segmentDistance(x, y, x1, y1, x2, y2) <= half,
      c,
      alpha,
    )
  }
}

function insidePolygon(points: readonly (readonly [number, number])[], x: number, y: number): boolean {
  let inside = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i] ?? [0, 0]
    const [xj, yj] = points[j] ?? [0, 0]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function polygonBox(points: readonly (readonly [number, number])[]): [number, number, number, number] {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [x, y] of points) {
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  }
  return [x0, y0, x1, y1]
}

function draw(cv: Canvas, node: SceneNode, m: Affine, alpha: number, theme: 'light' | 'dark'): void {
  const colour = (p: Paint | undefined): Rgba | undefined => (p === undefined ? undefined : paintColour(p, theme))
  if (node.kind === 'group') {
    const s = node.scale ?? 1
    const rad = ((node.rotate ?? 0) * Math.PI) / 180
    const [tx, ty] = node.translate ?? [0, 0]
    const local: Affine = [s * Math.cos(rad), s * Math.sin(rad), -s * Math.sin(rad), s * Math.cos(rad), tx, ty]
    const next = compose(m, local)
    for (const child of node.children) draw(cv, child, next, alpha * (node.opacity ?? 1), theme)
    return
  }
  if (node.kind === 'text' || node.kind === 'tex') return
  if (node.kind === 'image') {
    const { pixels } = node
    const inv = invert(m)
    const sw = cv.width * cv.ss
    const sh = cv.height * cv.ss
    for (let iy = 0; iy < sh; iy++) {
      for (let ix = 0; ix < sw; ix++) {
        const [lx, ly] = apply(inv, (ix + 0.5) / cv.ss, (iy + 0.5) / cv.ss)
        const u = (lx - node.x) / node.width
        const v = (ly - node.y) / node.height
        if (u < 0 || u >= 1 || v < 0 || v >= 1) continue
        const px = Math.floor(u * pixels.width)
        const py = Math.floor(v * pixels.height)
        const o = (py * pixels.width + px) * 4
        const c: Rgba = [
          pixels.rgba[o] ?? 0,
          pixels.rgba[o + 1] ?? 0,
          pixels.rgba[o + 2] ?? 0,
          (pixels.rgba[o + 3] ?? 255) / 255,
        ]
        blend(cv, iy * sw + ix, c, alpha)
      }
    }
    return
  }
  const a = alpha * (node.opacity ?? 1)
  const fill = colour(node.fill)
  const stroke = colour(node.stroke)
  const sw = node.strokeWidth ?? 1
  switch (node.kind) {
    case 'rect': {
      const { x, y, width, height } = node
      if (fill !== undefined) {
        fillShape(
          cv,
          m,
          [x, y, x + width, y + height],
          (px, py) => px >= x && px <= x + width && py >= y && py <= y + height,
          fill,
          a,
        )
      }
      if (stroke !== undefined) {
        const pts: [number, number][] = [
          [x, y],
          [x + width, y],
          [x + width, y + height],
          [x, y + height],
        ]
        strokeSegments(cv, m, pts, true, sw, stroke, a)
      }
      return
    }
    case 'circle':
    case 'ellipse': {
      const rx = node.kind === 'circle' ? node.r : node.rx
      const ry = node.kind === 'circle' ? node.r : node.ry
      const { cx, cy } = node
      const box: [number, number, number, number] = [cx - rx - sw, cy - ry - sw, cx + rx + sw, cy + ry + sw]
      const level = (px: number, py: number): number => Math.hypot((px - cx) / rx, (py - cy) / ry)
      if (fill !== undefined) fillShape(cv, m, box, (px, py) => level(px, py) <= 1, fill, a)
      if (stroke !== undefined) {
        const r = Math.min(rx, ry)
        fillShape(cv, m, box, (px, py) => Math.abs(level(px, py) - 1) * r <= sw / 2, stroke, a)
      }
      return
    }
    case 'line': {
      const c = stroke ?? fill
      if (c !== undefined) {
        strokeSegments(
          cv,
          m,
          [
            [node.x1, node.y1],
            [node.x2, node.y2],
          ],
          false,
          sw,
          c,
          a,
        )
      }
      return
    }
    case 'path': {
      const closed = node.closed ?? false
      if (fill !== undefined && closed && node.points.length >= 3) {
        fillShape(cv, m, polygonBox(node.points), (px, py) => insidePolygon(node.points, px, py), fill, a)
      }
      if (stroke !== undefined && node.points.length >= 2) strokeSegments(cv, m, node.points, closed, sw, stroke, a)
      return
    }
  }
}

/**
 * Rasterise a scene to RGBA pixels with a small software renderer: rectangles, circles, ellipses, lines, paths and
 * polygons (filled by the even-odd rule), images and groups with their transforms. Text and TeX labels are skipped.
 * Used for pixel observations in reinforcement learning and for frames in Node.
 *
 * @param s - The scene.
 * @param options - Output size, theme and supersampling.
 * @returns The image.
 * @concept scene-graph
 * @example
 * ```ts
 * import { rasterise, rect, scene } from 'whitebox-engine/scene'
 * const img = rasterise(scene(4, 4, [rect(0, 0, 2, 4, { fill: '#ff0000' })]))
 * Array.from(img.rgba.slice(0, 4)) // => [255, 0, 0, 255]
 * ```
 */
export function rasterise(s: Scene, options: RasterOptions = {}): Raster {
  const width = options.width ?? Math.round(s.width)
  const height = options.height ?? Math.round(s.height)
  const ss = Math.max(1, Math.floor(options.supersample ?? 1))
  const theme = options.theme ?? 'light'
  const n = width * ss * height * ss
  const cv: Canvas = { width, height, ss, r: new Float64Array(n), g: new Float64Array(n), b: new Float64Array(n) }
  const bg = paintColour(s.background ?? 'bg', theme) ?? [255, 255, 255, 1]
  cv.r.fill(bg[0])
  cv.g.fill(bg[1])
  cv.b.fill(bg[2])
  const m: Affine = [width / s.width, 0, 0, height / s.height, 0, 0]
  for (const node of s.children) draw(cv, node, m, 1, theme)
  const rgba = new Uint8ClampedArray(width * height * 4)
  const sw = width * ss
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0
      let g = 0
      let b = 0
      for (let dy = 0; dy < ss; dy++) {
        for (let dx = 0; dx < ss; dx++) {
          const i = (y * ss + dy) * sw + x * ss + dx
          r += cv.r[i] ?? 0
          g += cv.g[i] ?? 0
          b += cv.b[i] ?? 0
        }
      }
      const o = (y * width + x) * 4
      const k = ss * ss
      rgba[o] = Math.round(r / k)
      rgba[o + 1] = Math.round(g / k)
      rgba[o + 2] = Math.round(b / k)
      rgba[o + 3] = 255
    }
  }
  return { width, height, rgba }
}

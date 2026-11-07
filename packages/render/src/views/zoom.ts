/**
 * The maths of a zoomable picture, with no DOM: a viewport is a rectangle of data coordinates, and zooming about a
 * point, panning, clamping to a domain, mapping a pixel to data and back, the rectangle a minimap draws, the value of
 * a grid under the cursor and the raster of a grid along a colour ramp are all pure functions of it. The zoom panel
 * only feeds them pointer events.
 */

import type { DecisionSurface } from 'mdml-engine/exposition'
import type { Pixels } from 'mdml-engine/scene'
import { parseColour, sampleRamp, sequentialRamp } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'

/** A rectangle of data coordinates: `x` from left to right, `y` from bottom to top. */
export interface Viewport {
  readonly x: readonly [number, number]
  readonly y: readonly [number, number]
}

/**
 * Whether two viewports are the same rectangle.
 *
 * @param a - A viewport.
 * @param b - Another.
 * @returns `true` when equal.
 * @example
 * ```ts
 * import { sameViewport } from 'mdml-render/views'
 * sameViewport({ x: [0, 1], y: [0, 1] }, { x: [0, 1], y: [0, 1] }) // => true
 * ```
 */
export function sameViewport(a: Viewport, b: Viewport): boolean {
  return a.x[0] === b.x[0] && a.x[1] === b.x[1] && a.y[0] === b.y[0] && a.y[1] === b.y[1]
}

/**
 * The viewport kept inside a domain: translated back when it sticks out, and shrunk to the domain when it is larger.
 *
 * @param view - The viewport.
 * @param domain - The rectangle the picture exists in.
 * @returns The clamped viewport.
 * @example
 * ```ts
 * import { clampViewport } from 'mdml-render/views'
 * clampViewport({ x: [-3, -1], y: [0, 1] }, { x: [-2, 1], y: [-1, 1] }).x // => [-2, 0]
 * ```
 */
export function clampViewport(view: Viewport, domain: Viewport): Viewport {
  const fit = (v: readonly [number, number], d: readonly [number, number]): [number, number] => {
    const span = Math.min(v[1] - v[0], d[1] - d[0])
    const lo = Math.min(Math.max(v[0], d[0]), d[1] - span)
    return [lo, lo + span]
  }
  return { x: fit(view.x, domain.x), y: fit(view.y, domain.y) }
}

/**
 * Zoom about a point of the data: the point stays where it is on screen, and each span is divided by `factor`
 * (above one zooms in). The spans never fall below `minSpan` (as a fraction of `domain`, default 1e-12 of it) nor,
 * given a domain, rise above it.
 *
 * @param view - The viewport.
 * @param focus - The data point that stays fixed, as `[x, y]`.
 * @param factor - How much to zoom in by.
 * @param domain - Optional: the rectangle the viewport is clamped to.
 * @returns The new viewport.
 * @example
 * ```ts
 * import { zoomViewport } from 'mdml-render/views'
 * zoomViewport({ x: [0, 4], y: [0, 2] }, [0, 0], 2) // => { x: [0, 2], y: [0, 1] }
 * ```
 */
export function zoomViewport(
  view: Viewport,
  focus: readonly [number, number],
  factor: number,
  domain?: Viewport,
): Viewport {
  const f = factor > 0 && Number.isFinite(factor) ? factor : 1
  const scale = (v: readonly [number, number], c: number): [number, number] => [c - (c - v[0]) / f, c + (v[1] - c) / f]
  const next = { x: scale(view.x, focus[0]), y: scale(view.y, focus[1]) }
  // A span that would collapse below the floating-point resolution of its own position is left as it was.
  const tooSmall = (v: readonly [number, number]): boolean =>
    v[1] - v[0] < 1e-13 * Math.max(1, Math.abs(v[0]), Math.abs(v[1]))
  const bounded = tooSmall(next.x) || tooSmall(next.y) ? view : next
  return domain === undefined ? bounded : clampViewport(bounded, domain)
}

/**
 * Pan by a fraction of the viewport's own span (positive `dx` moves the picture's content to the left, the viewport to
 * the right; positive `dy` moves the viewport up).
 *
 * @param view - The viewport.
 * @param dx - The horizontal shift, in viewport widths.
 * @param dy - The vertical shift, in viewport heights.
 * @param domain - Optional: the rectangle the viewport is clamped to.
 * @returns The new viewport.
 * @example
 * ```ts
 * import { panViewport } from 'mdml-render/views'
 * panViewport({ x: [0, 4], y: [0, 2] }, 0.5, -1) // => { x: [2, 6], y: [-2, 0] }
 * ```
 */
export function panViewport(view: Viewport, dx: number, dy: number, domain?: Viewport): Viewport {
  const wx = view.x[1] - view.x[0]
  const wy = view.y[1] - view.y[0]
  const next: Viewport = {
    x: [view.x[0] + dx * wx, view.x[1] + dx * wx],
    y: [view.y[0] + dy * wy, view.y[1] + dy * wy],
  }
  return domain === undefined ? next : clampViewport(next, domain)
}

/**
 * The data point under a pixel of a picture drawn over a viewport (pixel row 0 at the top, data `y` upward).
 *
 * @param view - The viewport the picture shows.
 * @param size - The picture's width and height in pixels.
 * @param px - The pixel's x.
 * @param py - The pixel's y, from the top.
 * @returns The data point.
 * @example
 * ```ts
 * import { pixelToData } from 'mdml-render/views'
 * pixelToData({ x: [0, 4], y: [0, 2] }, [400, 200], 100, 50) // => [1, 1.5]
 * ```
 */
export function pixelToData(view: Viewport, size: readonly [number, number], px: number, py: number): [number, number] {
  return [view.x[0] + (px / size[0]) * (view.x[1] - view.x[0]), view.y[1] - (py / size[1]) * (view.y[1] - view.y[0])]
}

/**
 * The pixel of a data point (the inverse of {@link pixelToData}).
 *
 * @param view - The viewport the picture shows.
 * @param size - The picture's width and height in pixels.
 * @param x - The data x.
 * @param y - The data y.
 * @returns The pixel as `[px, py]`, `py` from the top.
 * @example
 * ```ts
 * import { dataToPixel } from 'mdml-render/views'
 * dataToPixel({ x: [0, 4], y: [0, 2] }, [400, 200], 1, 1.5) // => [100, 50]
 * ```
 */
export function dataToPixel(view: Viewport, size: readonly [number, number], x: number, y: number): [number, number] {
  return [((x - view.x[0]) / (view.x[1] - view.x[0])) * size[0], ((view.y[1] - y) / (view.y[1] - view.y[0])) * size[1]]
}

/**
 * The rectangle of the viewport in a minimap of the domain, as fractions of the minimap (`top` from the top).
 *
 * @param view - The viewport.
 * @param domain - The whole picture.
 * @returns Left, top, width and height as fractions in [0, 1].
 * @example
 * ```ts
 * import { minimapRect } from 'mdml-render/views'
 * minimapRect({ x: [0, 1], y: [0, 1] }, { x: [-1, 1], y: [-1, 1] }) // => { left: 0.5, top: 0, width: 0.5, height: 0.5 }
 * ```
 */
export function minimapRect(
  view: Viewport,
  domain: Viewport,
): { left: number; top: number; width: number; height: number } {
  const dx = domain.x[1] - domain.x[0]
  const dy = domain.y[1] - domain.y[0]
  const unit = (v: number): number => Math.min(1, Math.max(0, v))
  const left = unit((view.x[0] - domain.x[0]) / dx)
  const right = unit((view.x[1] - domain.x[0]) / dx)
  const top = unit((domain.y[1] - view.y[1]) / dy)
  const bottom = unit((domain.y[1] - view.y[0]) / dy)
  return { left, top, width: right - left, height: bottom - top }
}

/** What lies under the cursor on a grid: the nearest cell and its value. */
export interface GridCell {
  readonly i: number
  readonly j: number
  readonly value: number
}

/**
 * The cell of a surface nearest to a data point, with its value; `undefined` outside the surface's extent (half a cell
 * of slack) or when the surface has no `values`.
 *
 * @param surface - A decision surface with `values[i][j]` at (`x[j]`, `y[i]`).
 * @param x - The data x.
 * @param y - The data y.
 * @returns The cell.
 * @example
 * ```ts
 * import { decisionSurface } from 'mdml-engine/exposition'
 * import { cellAt } from 'mdml-render/views'
 * const s = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0, 1], values: [[1, 2], [3, 4]] })
 * cellAt(s, 0.9, 0.1)?.value // => 2
 * ```
 */
export function cellAt(surface: DecisionSurface, x: number, y: number): GridCell | undefined {
  const values = surface.values
  if (values === undefined) return undefined
  const nearest = (axis: readonly number[], v: number): number | undefined => {
    const n = axis.length
    if (n === 0) return undefined
    const lo = axis[0] ?? 0
    const hi = axis[n - 1] ?? lo
    const half = n > 1 ? Math.abs(hi - lo) / (n - 1) / 2 : 0
    if (v < Math.min(lo, hi) - half || v > Math.max(lo, hi) + half) return undefined
    const k = n > 1 ? Math.round(((v - lo) / (hi - lo)) * (n - 1)) : 0
    return Math.min(n - 1, Math.max(0, k))
  }
  const j = nearest(surface.x, x)
  const i = nearest(surface.y, y)
  const value = i === undefined || j === undefined ? undefined : values[i]?.[j]
  return i === undefined || j === undefined || value === undefined ? undefined : { i, j, value }
}

/**
 * The extent of a surface as a viewport: from the first to the last of its coordinates.
 *
 * @param surface - A decision surface.
 * @returns The viewport.
 * @example
 * ```ts
 * import { decisionSurface } from 'mdml-engine/exposition'
 * import { surfaceViewport } from 'mdml-render/views'
 * surfaceViewport(decisionSurface({ features: ['a', 'b'], x: [0, 2], y: [1, 3] })) // => { x: [0, 2], y: [1, 3] }
 * ```
 */
export function surfaceViewport(surface: DecisionSurface): Viewport {
  const ends = (a: readonly number[]): [number, number] => [a[0] ?? 0, a[a.length - 1] ?? 1]
  return { x: ends(surface.x), y: ends(surface.y) }
}

/**
 * The finite minimum and maximum of a grid.
 *
 * @param values - The grid, one array per row.
 * @returns The minimum and maximum.
 */
export function gridRange(values: readonly (readonly number[])[]): [number, number] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const row of values)
    for (const v of row)
      if (Number.isFinite(v)) {
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
  return lo <= hi ? [lo, hi] : [0, 1]
}

/**
 * A surface's `values` as pixels along a colour ramp (row 0 of the picture at the top, so the last `y` row first);
 * values outside `range` clamp to its ends, and a value that is not finite is drawn in the background grey.
 *
 * @param surface - A decision surface with `values`.
 * @param options - The colour mode, the value range mapped to the ramp's ends (default the values' own), and the ramp.
 * @returns The pixels (one per grid cell).
 * @example
 * ```ts
 * import { decisionSurface } from 'mdml-engine/exposition'
 * import { surfaceRaster } from 'mdml-render/views'
 * const s = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0, 1], values: [[0, 1], [1, 0]] })
 * surfaceRaster(s).width // => 2
 * ```
 */
export function surfaceRaster(
  surface: DecisionSurface,
  options: {
    readonly mode?: ColourMode
    readonly range?: readonly [number, number]
    readonly ramp?: readonly string[]
  } = {},
): Pixels {
  const values = surface.values ?? []
  const rows = values.length
  const cols = surface.x.length
  const [lo, hi] = options.range ?? gridRange(values)
  const stops = options.ramp ?? sequentialRamp('viridis', options.mode ?? 'light')
  const lut = sampleRamp(stops, 256).map(parseColour)
  const rgba = new Uint8ClampedArray(rows * cols * 4)
  const span = hi > lo ? hi - lo : 1
  for (let r = 0; r < rows; r++) {
    const row = values[rows - 1 - r] ?? []
    for (let c = 0; c < cols; c++) {
      const v = row[c]
      const p = 4 * (r * cols + c)
      if (v === undefined || !Number.isFinite(v)) {
        rgba.set([128, 128, 128, 255], p)
        continue
      }
      const colour = lut[Math.round(Math.min(1, Math.max(0, (v - lo) / span)) * 255)] ?? [0, 0, 0, 1]
      rgba.set([colour[0] ?? 0, colour[1] ?? 0, colour[2] ?? 0, 255], p)
    }
  }
  return { width: cols, height: rows, rgba }
}

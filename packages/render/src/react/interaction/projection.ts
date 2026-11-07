/**
 * Projections between a figure's data coordinates and the pixels of its drawing area, which overlays use to place
 * handles and read the pointer: a linear projection from ranges and margins, and the projection of one grid of a live
 * ECharts chart.
 */

import type { Point2 } from 'mdml-engine/numerics/geometry'

/**
 * The map between a figure's data coordinates and the pixels of its drawing area (origin top left, y down), which
 * interaction overlays use to place handles and read the pointer.
 */
export interface Projection {
  toPixel(p: Point2): Point2
  toData(p: Point2): Point2
}

/** Space left around a plot area, in pixels. */
export interface Margins {
  readonly top: number
  readonly right: number
  readonly bottom: number
  readonly left: number
}

/**
 * A linear projection: the data ranges `x` and `y` fill the area of `width` by `height` pixels inside `margins`, with
 * y increasing upwards.
 *
 * @param options - The data ranges, the size in pixels and the margins (default none).
 * @returns The projection.
 */
export function linearProjection(options: {
  readonly x: Point2
  readonly y: Point2
  readonly width: number
  readonly height: number
  readonly margins?: Partial<Margins>
}): Projection {
  const { x, y, width, height } = options
  const m = { top: 0, right: 0, bottom: 0, left: 0, ...options.margins }
  const w = width - m.left - m.right
  const h = height - m.top - m.bottom
  const [x0, x1] = x
  const [y0, y1] = y
  return {
    toPixel: ([u, v]) => [m.left + ((u - x0) / (x1 - x0)) * w, m.top + ((y1 - v) / (y1 - y0)) * h],
    toData: ([px, py]) => [x0 + ((px - m.left) / w) * (x1 - x0), y1 - ((py - m.top) / h) * (y1 - y0)],
  }
}

/** The part of an ECharts instance a projection needs (structural, so this module does not import ECharts). */
export interface PixelConverter {
  convertToPixel(finder: { readonly gridIndex: number }, value: number[]): number[] | number
  convertFromPixel(finder: { readonly gridIndex: number }, value: number[]): number[] | number
}

/**
 * The projection of one grid (one set of axes) of a live ECharts chart, for overlays laid over the chart. While the
 * chart is between an option change and its next redraw, ECharts cannot convert; the projection then answers with the
 * last linear map it saw (exact for linear axes), so a drag never jumps to nowhere.
 *
 * @param chart - The chart instance (or anything with ECharts' `convertToPixel` and `convertFromPixel`).
 * @param gridIndex - Which grid (default 0).
 * @returns The projection.
 */
export function chartProjection(chart: PixelConverter, gridIndex = 0): Projection {
  const finder = { gridIndex }
  const pair = (v: unknown): Point2 | null =>
    Array.isArray(v) && Number.isFinite(v[0]) && Number.isFinite(v[1]) ? [v[0] as number, v[1] as number] : null
  // The last good linear map, pixel = origin + scale * data, per axis.
  let map: { readonly origin: Point2; readonly scale: Point2 } | null = null
  const learn = (): void => {
    const a = pair(chart.convertToPixel(finder, [0, 0]))
    const b = pair(chart.convertToPixel(finder, [1, 1]))
    if (a !== null && b !== null && b[0] !== a[0] && b[1] !== a[1])
      map = { origin: a, scale: [b[0] - a[0], b[1] - a[1]] }
  }
  const none: Point2 = [Number.NaN, Number.NaN]
  return {
    toPixel: (p) => {
      const out = pair(chart.convertToPixel(finder, [p[0], p[1]]))
      if (out !== null) {
        learn()
        return out
      }
      const m = map as { readonly origin: Point2; readonly scale: Point2 } | null
      return m === null ? none : [m.origin[0] + m.scale[0] * p[0], m.origin[1] + m.scale[1] * p[1]]
    },
    toData: (p) => {
      const out = pair(chart.convertFromPixel(finder, [p[0], p[1]]))
      if (out !== null) {
        learn()
        return out
      }
      const m = map as { readonly origin: Point2; readonly scale: Point2 } | null
      return m === null ? none : [(p[0] - m.origin[0]) / m.scale[0], (p[1] - m.origin[1]) / m.scale[1]]
    },
  }
}

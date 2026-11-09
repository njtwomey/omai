/**
 * Projections between a figure's data coordinates and the pixels of its drawing area, which overlays use to place
 * handles and read the pointer: a linear projection from ranges and margins, and the projection of one grid of a live
 * ECharts chart.
 */

import type { Point2 } from 'omai-engine/numerics/geometry'

/**
 * The map between a figure's data coordinates and the pixels of its drawing area (origin top left, y down), which
 * interaction overlays use to place handles and read the pointer.
 */
export interface Projection {
  toPixel(p: Point2): Point2
  toData(p: Point2): Point2
  /** {@link toPixel} for a point given as two numbers: the pixel of the data point `(x, y)`. */
  project(x: number, y: number): Point2
  /** {@link toData} for a pixel given as two numbers: the data point at pixel `(px, py)` of the drawing area. */
  unproject(px: number, py: number): Point2
  /**
   * The plotting box in pixels (origin top left): where the axes are inside the drawing area, so a caller can place its
   * own marks with the chart's geometry. `null` while it is not known (a chart between two redraws).
   */
  readonly rect?: PlotBox | null
}

/** A rectangle in pixels: the plotting box of a projection. */
export interface PlotBox {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
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
  const toPixel = ([u, v]: Point2): Point2 => [m.left + ((u - x0) / (x1 - x0)) * w, m.top + ((y1 - v) / (y1 - y0)) * h]
  const toData = ([px, py]: Point2): Point2 => [
    x0 + ((px - m.left) / w) * (x1 - x0),
    y1 - ((py - m.top) / h) * (y1 - y0),
  ]
  return {
    rect: { x: m.left, y: m.top, width: w, height: h },
    toPixel,
    toData,
    project: (x, y) => toPixel([x, y]),
    unproject: (px, py) => toData([px, py]),
  }
}

/** The part of an ECharts instance a projection needs (structural, so this module does not import ECharts). */
export interface PixelConverter {
  convertToPixel(finder: { readonly gridIndex: number }, value: number[]): number[] | number
  convertFromPixel(finder: { readonly gridIndex: number }, value: number[]): number[] | number
  /** The chart's current option, from which the grid's box and axis ranges are read (optional: without it `rect` is `null`). */
  getOption?(): unknown
  /**
   * Subscribe to a chart event (optional). With it the projection reads the option once and again after each
   * `finished` (a redraw); without it, it asks `getOption` on every call and compares the answer by identity.
   */
  on?(event: string, handler: () => void): unknown
}

/** What a projection reads from a chart's option: the grid's box and, when the axes allow it, the exact linear map. */
interface Geometry {
  readonly rect: PlotBox
  /** The map computed from the option alone (both axes numeric value axes), or `null` when ECharts must convert. */
  readonly linear: Projection | null
  /** Changes whenever the box or the ranges do. */
  readonly key: string
}

type Dict = Readonly<Record<string, unknown>>

// The entry of an option's `xAxis` or `yAxis` (an array, or one object) that belongs to the grid.
function axisOf(axes: unknown, gridIndex: number): Dict | undefined {
  const list = (Array.isArray(axes) ? axes : axes === undefined ? [] : [axes]) as readonly Dict[]
  return list.find((a) => ((a['gridIndex'] as number | undefined) ?? 0) === gridIndex)
}

// A value axis with numeric limits: its range from the first pixel (left, bottom) to the last, swapped for an inverse axis.
function limitsOf(axis: Dict | undefined): Point2 | undefined {
  if (axis === undefined || axis['type'] !== 'value') return undefined
  const { min, max } = axis
  if (typeof min !== 'number' || typeof max !== 'number' || !Number.isFinite(min) || !Number.isFinite(max))
    return undefined
  if (!(max > min)) return undefined
  return axis['inverse'] === true ? [max, min] : [min, max]
}

function geometryOf(option: unknown, gridIndex: number): Geometry | null {
  const o = option as {
    readonly grid?: unknown
    readonly xAxis?: unknown
    readonly yAxis?: unknown
    readonly dataZoom?: unknown
  }
  const grids = (Array.isArray(o?.grid) ? o.grid : o?.grid === undefined ? [] : [o.grid]) as readonly Dict[]
  const g = grids[gridIndex]
  const n = (k: string): number | undefined => (typeof g?.[k] === 'number' ? (g[k] as number) : undefined)
  const [x, y, width, height] = [n('left'), n('top'), n('width'), n('height')]
  if (x === undefined || y === undefined || width === undefined || height === undefined) return null
  const rect: PlotBox = { x, y, width, height }
  // Zoom moves the axes' extent without changing the option, and a grid that contains its labels is smaller than
  // its box: in both cases only ECharts knows the map. So does ECharts 6's `outerBoundsMode: 'auto'` (the default it
  // merges into a hand-written grid), which moves the box inside the one named to keep the axis labels in the figure:
  // the box is where the option says only when the mode is 'none' (or not stated, as in an option not yet merged).
  const mode = g?.['outerBoundsMode']
  const pinned = mode === 'none' || (mode === undefined && g?.['containLabel'] !== true)
  const zoomed = Array.isArray(o.dataZoom) ? o.dataZoom.length > 0 : o.dataZoom !== undefined
  const xl = limitsOf(axisOf(o.xAxis, gridIndex))
  const yl = limitsOf(axisOf(o.yAxis, gridIndex))
  const exact = !zoomed && pinned && width > 0 && height > 0 && xl !== undefined && yl !== undefined
  const linear = exact
    ? linearProjection({
        x: xl,
        y: yl,
        width: x + width,
        height: y + height,
        margins: { left: x, top: y },
      })
    : null
  return { rect, linear, key: JSON.stringify([rect, exact ? [xl, yl] : null]) }
}

/**
 * The projection of one grid (one set of axes) of a live ECharts chart, for overlays laid over the chart.
 *
 * When the chart's option places the grid with numbers (`left`, `top`, `width`, `height`), says that ECharts will not
 * move the box (`outerBoundsMode: 'none'`, which omai's cartesian builders write), and the grid's axes are value axes
 * with numeric `min` and `max` (also `inverse`), the map is computed from the option alone and ECharts is never asked:
 * converting through ECharts logs a warning per call while it is between a change of option and its redraw, which a
 * drag turns into dozens. Log, category and time axes, auto-ranged axes, a grid that contains its labels, a chart with
 * `dataZoom` and a grid whose `outerBoundsMode` is `'auto'` (ECharts 6's default for a hand-written grid, which shrinks
 * the plotting box to keep the labels in the figure, so the option's numbers are not where the axes are) or `'same'`
 * go through `convertToPixel` and `convertFromPixel`; for a hand-written grid that means ECharts' warnings while it is
 * between redraws. Add `outerBoundsMode: 'none'` to such a grid when its margins leave room for the labels. While the chart is between an option
 * change and its next redraw ECharts cannot convert; the projection then answers with the last linear map it saw (exact
 * for linear axes), so a drag never jumps to nowhere.
 *
 * The option is read lazily and kept: a chart that reports events (`on`) is read once and again after each redraw
 * (`finished`), so a drag costs no more than arithmetic; one that does not is asked on every call and the answer is
 * compared by identity. `onChange` is called after a redraw that moved the box or the ranges, so a host can draw its
 * handles again.
 *
 * From a chart handle (the instance `Chart`'s `onInit` gives): `const p = chartProjection(chart)`, then
 * `p.project(x, y)` is the pixel of a data point and `p.unproject(px, py)` the data point under a pixel, in the pixels of
 * the chart's drawing area (origin top left), and `p.rect` is the plotting box.
 *
 * @param chart - The chart instance (or anything with ECharts' `convertToPixel` and `convertFromPixel`).
 * @param gridIndex - Which grid (default 0).
 * @param onChange - Called when a redraw changes the grid's box or the axes' ranges (needs `chart.on`).
 * @returns The projection; its `rect` is the grid's plotting box in pixels, read from the chart's option.
 * @example
 * ```ts
 * import { chartProjection } from 'omai-render/react'
 * const option = {
 *   grid: [{ left: 50, top: 10, width: 200, height: 100, outerBoundsMode: 'none' }],
 *   xAxis: [{ type: 'value', min: 0, max: 10 }],
 *   yAxis: [{ type: 'value', min: 0, max: 1 }],
 * }
 * const never = (): never => {
 *   throw new Error('not asked')
 * }
 * const p = chartProjection({ convertToPixel: never, convertFromPixel: never, getOption: () => option })
 * p.project(5, 0.5) // => [150, 60]
 * p.unproject(150, 60) // => [5, 0.5]
 * ```
 */
export function chartProjection(chart: PixelConverter, gridIndex = 0, onChange?: () => void): Projection {
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
  // The geometry read from the option, kept until a redraw (with events) or until the option object changes (without).
  let cache: { readonly option: unknown; readonly geometry: Geometry | null } | undefined
  const subscribed = typeof chart.on === 'function'
  const read = (): Geometry | null => {
    const option = chart.getOption?.()
    cache = { option, geometry: option === undefined ? null : geometryOf(option, gridIndex) }
    return cache.geometry
  }
  const geometry = (): Geometry | null => {
    if (chart.getOption === undefined) return null
    if (subscribed) return cache === undefined ? read() : cache.geometry
    return cache !== undefined && cache.option === chart.getOption() ? cache.geometry : read()
  }
  if (subscribed)
    chart.on?.('finished', () => {
      const before = cache?.geometry?.key
      cache = undefined
      if (onChange !== undefined && read()?.key !== before) onChange()
    })
  const none: Point2 = [Number.NaN, Number.NaN]
  const toPixel = (p: Point2): Point2 => {
    const exact = geometry()?.linear
    if (exact !== null && exact !== undefined) return exact.toPixel(p)
    const out = pair(chart.convertToPixel(finder, [p[0], p[1]]))
    if (out !== null) {
      learn()
      return out
    }
    const m = map as { readonly origin: Point2; readonly scale: Point2 } | null
    return m === null ? none : [m.origin[0] + m.scale[0] * p[0], m.origin[1] + m.scale[1] * p[1]]
  }
  const toData = (p: Point2): Point2 => {
    const exact = geometry()?.linear
    if (exact !== null && exact !== undefined) return exact.toData(p)
    const out = pair(chart.convertFromPixel(finder, [p[0], p[1]]))
    if (out !== null) {
      learn()
      return out
    }
    const m = map as { readonly origin: Point2; readonly scale: Point2 } | null
    return m === null ? none : [(p[0] - m.origin[0]) / m.scale[0], (p[1] - m.origin[1]) / m.scale[1]]
  }
  return {
    // The grid of the chart's option, in pixels (omai's charts place every grid with numbers).
    get rect(): PlotBox | null {
      return geometry()?.rect ?? null
    },
    toPixel,
    toData,
    project: (x, y) => toPixel([x, y]),
    unproject: (px, py) => toData([px, py]),
  }
}

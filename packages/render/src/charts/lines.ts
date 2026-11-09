/**
 * Line layers: curves drawn as lines, steps or areas with gaps where values are not finite, filled bands between two
 * curves (confidence bands, fill-between), curves read from a tensor, and `lineChart` for curves on one pair of axes.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { OmaiError } from 'omai-engine/base'
import type { Point2 } from 'omai-engine/numerics/geometry'
import { asTensor, type TensorLike, toFlat } from 'omai-engine/tensor'
import { extentOf, unionRange } from '../figure/limits.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { colourFor, dashType, modeOf, numbers } from './_data.ts'
import { deferMode, isModeless } from './_mode.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'

/** A curve: y values against x values (default 0, 1, 2, ...), and how to draw it. */
export interface Curve {
  readonly x?: TensorLike | readonly number[]
  readonly y: TensorLike | readonly number[]
  readonly name?: string
  /** A CSS colour or theme token (default the next tab10 colour). */
  readonly colour?: string
  readonly width?: number
  readonly dash?: 'solid' | 'dashed' | 'dotted' | readonly number[]
  /** Draw as steps, changing value at the start, middle or end of each interval. */
  readonly step?: 'start' | 'middle' | 'end'
  /** Fill down to zero (an area chart), with this opacity (default 0.2 when `true`). */
  readonly area?: boolean | number
  readonly markers?: boolean
  readonly smooth?: boolean
  readonly opacity?: number
}

/** Options of the line layers. */
export interface LineOptions {
  readonly mode?: ColourMode
  /** The tab10 index of the first curve's default colour (default 0). */
  readonly colourOffset?: number
}

function pairs(c: Curve): [number, number][] {
  const y = numbers(c.y)
  const x = c.x === undefined ? y.map((_, i) => i) : numbers(c.x)
  if (x.length !== y.length) throw new OmaiError('shape', 'lines: x and y must have the same length')
  return y.map((v, i) => [x[i] ?? i, v])
}

/**
 * Curves as a layer: lines, steps or areas, with gaps where y is not finite.
 *
 * @param curves - The curves.
 * @param options - Colour mode and colour offset.
 * @returns The layer.
 * @index function A layer of lines, steps or areas to combine: `lines([{ y: [1, 3, 2] }])`.
 * @example
 * ```ts
 * import { lines } from 'omai-render/charts'
 * const l = lines([{ y: [1, 3, 2], name: 'loss' }])
 * l.extent // => { x: [0, 2], y: [1, 3] }
 * l.series[0].name // => 'loss'
 * ```
 */
export function lines(curves: readonly Curve[], options: LineOptions = {}): Layer {
  if (isModeless(options)) return deferMode((mode) => lines(curves, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const offset = options.colourOffset ?? 0
  const xs: [number, number][] = []
  const series: SeriesOption[] = curves.map((c, i) => {
    const data = pairs(c).map(([x, y]): [number, number | null] => [x, Number.isFinite(y) ? y : null])
    for (const [x, y] of data) if (y !== null && Number.isFinite(x)) xs.push([x, y])
    const colour = colourFor(c.colour, mode, offset + i)
    const area = c.area === undefined || c.area === false ? undefined : c.area === true ? 0.2 : c.area
    return {
      type: 'line',
      ...(c.name === undefined ? {} : { name: c.name }),
      data,
      showSymbol: c.markers === true,
      symbolSize: 5,
      connectNulls: false,
      smooth: c.smooth === true,
      ...(c.step === undefined ? {} : { step: c.step }),
      color: colour,
      lineStyle: { color: colour, width: c.width ?? 2, type: dashType(c.dash), opacity: c.opacity ?? 1 },
      ...(area === undefined ? {} : { areaStyle: { color: colour, opacity: area } }),
      emphasis: { disabled: true },
      clip: true,
    } as SeriesOption
  })
  const yExtent = extentOf(xs.map((p) => p[1]))
  const hasArea = curves.some((c) => c.area !== undefined && c.area !== false)
  return layer(series, {
    ...(xs.length > 0 ? { x: extentOf(xs.map((p) => p[0])) ?? [0, 1] } : {}),
    ...(yExtent === undefined ? {} : { y: hasArea ? (unionRange([yExtent, [0, 0]]) ?? yExtent) : yExtent }),
  })
}

/** A filled band between two curves over the same x values (a confidence band, a fill-between). */
export interface Band {
  readonly x: TensorLike | readonly number[]
  readonly lower: TensorLike | readonly number[]
  readonly upper: TensorLike | readonly number[]
  readonly name?: string
  readonly colour?: string
  /** Fill opacity (default 0.2). */
  readonly opacity?: number
  /** Also stroke the two edges (default false). */
  readonly edges?: boolean
}

interface PolygonApi {
  coord(p: readonly number[]): number[]
}

/**
 * Bands as a layer: each band is a filled polygon between its lower and upper curves (fill-between), split where
 * either is not finite. Drawn as one polygon per band, so it handles crossings and negative values exactly.
 *
 * @param bands - The bands.
 * @param options - Colour mode and colour offset.
 * @returns The layer.
 * @index function A filled band between two curves: `band([{ x, lower, upper }])`.
 * @example
 * ```ts
 * import { band } from 'omai-render/charts'
 * band([{ x: [0, 1, 2], lower: [0, 0, 1], upper: [1, 2, 2] }]).extent // => { x: [0, 2], y: [0, 2] }
 * ```
 */
export function band(bands: readonly Band[], options: LineOptions = {}): Layer {
  if (isModeless(options)) return deferMode((mode) => band(bands, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const offset = options.colourOffset ?? 0
  const all: Point2[] = []
  const series: SeriesOption[] = []
  bands.forEach((b, i) => {
    const x = numbers(b.x)
    const lo = numbers(b.lower)
    const hi = numbers(b.upper)
    if (lo.length !== x.length || hi.length !== x.length)
      throw new OmaiError('shape', 'band: x, lower and upper must have the same length')
    const polygons: Point2[][] = []
    let run: number[] = []
    const flush = (): void => {
      if (run.length >= 2) {
        const top = run.map((k): Point2 => [x[k] ?? 0, hi[k] ?? 0])
        const bottom = run.map((k): Point2 => [x[k] ?? 0, lo[k] ?? 0]).reverse()
        polygons.push([...top, ...bottom])
      }
      run = []
    }
    x.forEach((xv, k) => {
      if (Number.isFinite(xv) && Number.isFinite(lo[k]) && Number.isFinite(hi[k])) run.push(k)
      else flush()
    })
    flush()
    for (const p of polygons) all.push(...p)
    const colour = colourFor(b.colour, mode, offset + i)
    const opacity = b.opacity ?? 0.2
    series.push({
      type: 'custom',
      name: b.name ?? `__band${i}`,
      data: polygons.map((_, k) => [k]),
      renderItem: (params: { dataIndex: number }, api: PolygonApi) => ({
        type: 'polygon',
        shape: { points: (polygons[params.dataIndex] ?? []).map((p) => api.coord(p)) },
        style: {
          fill: colour,
          opacity,
          ...(b.edges === true ? { stroke: colour, lineWidth: 1 } : {}),
        },
        silent: true,
      }),
      color: colour,
      clip: true,
      silent: true,
      z: 1,
      tooltip: { show: false },
    } as unknown as SeriesOption)
  })
  return layer(series, {
    ...(all.length > 0
      ? { x: extentOf(all.map((p) => p[0])) ?? [0, 1], y: extentOf(all.map((p) => p[1])) ?? [0, 1] }
      : {}),
  })
}

/** A filled polygon in data coordinates (a rule's box, a leaf's region, an ellipse). */
export interface Polygon {
  /** The vertices, in order; the polygon closes itself. */
  readonly points: readonly (readonly [number, number])[]
  readonly name?: string
  readonly colour?: string
  /** Fill opacity (default 0.15). */
  readonly opacity?: number
  /** Stroke the outline (default true). */
  readonly edges?: boolean
  readonly dash?: 'solid' | 'dashed' | 'dotted'
}

/**
 * Polygons as a layer: each a filled shape with an optional outline, in data coordinates, so a region of the plane (a
 * box, a leaf, an ellipse) combines with any other layer.
 *
 * @param shapes - The polygons.
 * @param options - Colour mode and colour offset.
 * @returns The layer.
 * @example
 * ```ts
 * import { polygons } from 'omai-render/charts'
 * polygons([{ points: [[0, 0], [2, 0], [1, 3]] }]).extent // => { x: [0, 2], y: [0, 3] }
 * ```
 */
export function polygons(shapes: readonly Polygon[], options: LineOptions = {}): Layer {
  if (isModeless(options)) return deferMode((mode) => polygons(shapes, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const offset = options.colourOffset ?? 0
  const all: Point2[] = []
  const series: SeriesOption[] = []
  shapes.forEach((s, i) => {
    const pts = s.points.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))
    if (pts.length < 3) return
    all.push(...pts)
    const colour = colourFor(s.colour, mode, offset + i)
    const opacity = s.opacity ?? 0.15
    series.push({
      type: 'custom',
      name: s.name ?? `__polygon${i}`,
      data: [[0]],
      renderItem: (_: unknown, api: PolygonApi) => {
        const shape = { points: pts.map((p) => api.coord(p)) }
        return {
          type: 'group',
          silent: true,
          children: [
            { type: 'polygon', shape, style: { fill: colour, opacity, stroke: 'none' } },
            ...(s.edges === false
              ? []
              : [
                  {
                    type: 'polygon',
                    shape,
                    style: {
                      fill: 'none',
                      stroke: colour,
                      lineWidth: 1.5,
                      ...(s.dash === undefined || s.dash === 'solid'
                        ? {}
                        : { lineDash: s.dash === 'dashed' ? [5, 4] : [2, 3] }),
                    },
                  },
                ]),
          ],
        }
      },
      color: colour,
      clip: true,
      silent: true,
      z: 1,
      tooltip: { show: false },
    } as unknown as SeriesOption)
  })
  return layer(
    series,
    all.length > 0 ? { x: extentOf(all.map((p) => p[0])) ?? [0, 1], y: extentOf(all.map((p) => p[1])) ?? [0, 1] } : {},
  )
}

/**
 * Curves from a tensor: a 1-D tensor is one curve against its index, a tensor of shape `[n, 2]` (n other than 2) is
 * points `(x, y)`, and any other `[k, n]` tensor is k curves against the index.
 *
 * @param t - The tensor.
 * @param options - Names for the curves.
 * @returns The curves.
 * @example
 * ```ts
 * import { curvesFromTensor } from 'omai-render/charts'
 * curvesFromTensor([1, 2, 3]).length // => 1
 * curvesFromTensor([[0, 1], [1, 3], [2, 5]])[0].y // => [1, 3, 5]
 * curvesFromTensor([[1, 2, 3], [4, 5, 6]], { names: ['a', 'b'] })[1].name // => 'b'
 * ```
 */
export function curvesFromTensor(t: TensorLike, options: { readonly names?: readonly string[] } = {}): Curve[] {
  const v = asTensor(t)
  const shape = 'shape' in v ? v.shape : []
  const flat = toFlat(v)
  const named = (i: number): { name?: string } => {
    const name = options.names?.[i]
    return name === undefined ? {} : { name }
  }
  if (shape.length <= 1) return [{ y: flat, ...named(0) }]
  if (shape.length !== 2) throw new OmaiError('shape', 'curvesFromTensor: give a 1-D or 2-D tensor')
  const [k = 0, n = 0] = shape
  if (n === 2 && k !== 2)
    return [{ x: flat.filter((_, i) => i % 2 === 0), y: flat.filter((_, i) => i % 2 === 1), ...named(0) }]
  return Array.from({ length: k }, (_, i) => ({ y: flat.slice(i * n, (i + 1) * n), ...named(i) }))
}

/**
 * A line chart: curves on one pair of axes, with an axis tooltip.
 *
 * @param curves - The curves.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @index function A line chart of curves: `lineChart([{ y: [3, 1, 2], name: 'loss' }])`.
 * @example
 * ```ts
 * import { lineChart } from 'omai-render/charts'
 * const o = lineChart([{ y: [3, 1, 2], name: 'a' }, { y: [1, 2, 3], name: 'b' }])
 * o.legend.show // => true
 * ```
 */
export function lineChart(curves: readonly Curve[], options: PlotOptions = {}): EChartsOption {
  return cartesian([lines(curves, { mode: modeOf(options) })], {
    tooltip: 'axis',
    ...options,
  })
}

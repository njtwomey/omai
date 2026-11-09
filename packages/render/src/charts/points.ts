/**
 * Scatter plots: point sets coloured by group (tab10, with a marker shape per group so identity does not rest on colour
 * alone), bubble sizes by area, continuous colours along viridis and labels, plus rugs along an axis and
 * `scatterChart`.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf, minOf, OmaiError } from 'omai-engine/base'
import type { TensorLike } from 'omai-engine/tensor'
import { extentOf } from '../figure/limits.ts'
import { colourScale, sequentialRamp } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { colourFor, modeOf, numbers } from './_data.ts'
import { deferMode, isModeless } from './_mode.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'

/** Marker shapes, so group identity need not rely on colour alone. */
export const MARKERS: readonly string[] = ['circle', 'rect', 'triangle', 'diamond', 'pin', 'roundRect', 'arrow']

/** Points to scatter: coordinates, and optionally groups, sizes, continuous colours and labels. */
export interface PointSet {
  /** x coordinates, or `[n, 2]` points when `y` is omitted. */
  readonly x: TensorLike | readonly number[]
  readonly y?: TensorLike | readonly number[]
  /** A group per point (class labels): each group is a series with its own tab10 colour, and marker with `shapes`. */
  readonly groups?: readonly (string | number)[]
  /** Names of numeric groups, by group value. */
  readonly groupNames?: Readonly<Record<string, string>>
  /** A size per point (bubbles): mapped by area onto `sizeRange`. */
  readonly sizes?: readonly number[]
  readonly sizeRange?: readonly [number, number]
  /** A value per point coloured along a sequential ramp (instead of groups). */
  readonly values?: readonly number[]
  /** A text label per point, drawn beside it (with `symbol: 'none'`, the label alone marks the point). */
  readonly labels?: readonly string[]
  /** Label size in pixels (default the figure's type size). */
  readonly labelSize?: number
  /** Where a label sits against its point (default `right`); `inside` centres it on the point. */
  readonly labelPosition?: 'right' | 'top' | 'left' | 'bottom' | 'inside'
  /** Label colour (default the secondary text colour): a token name or any CSS colour. */
  readonly labelColour?: string
  /** Label weight (default normal) and CSS font family (default the figure's). */
  readonly labelWeight?: 'normal' | 'bold' | number
  readonly labelFamily?: string
  readonly name?: string
  readonly colour?: string
  /** Marker size in pixels (default 7). */
  readonly size?: number
  readonly opacity?: number
  readonly symbol?: string
  /** Vary marker shapes by group (default false). */
  readonly shapes?: boolean
  /** Outline markers with the background colour (default true), so overlapping points stay distinct. */
  readonly outline?: boolean
}

function coordinates(p: PointSet): { x: number[]; y: number[] } {
  if (p.y !== undefined) {
    const x = numbers(p.x)
    const y = numbers(p.y)
    if (x.length !== y.length) throw new OmaiError('shape', 'points: x and y must have the same length')
    return { x, y }
  }
  const flat = numbers(p.x)
  if (flat.length % 2 !== 0) throw new OmaiError('shape', 'points: give x and y, or [n, 2] points')
  return { x: flat.filter((_, i) => i % 2 === 0), y: flat.filter((_, i) => i % 2 === 1) }
}

/**
 * Scatter points as a layer: one series per group (tab10 colours in group order), bubble sizes by area, continuous
 * colours along viridis, and labels.
 *
 * @param sets - The point sets.
 * @param options - Colour mode and the tab10 offset of the first group.
 * @returns The layer.
 * @example
 * ```ts
 * import { points } from 'omai-render/charts'
 * const l = points([{ x: [0, 1, 2], y: [1, 0, 1], groups: [0, 1, 0], groupNames: { 0: 'a', 1: 'b' } }])
 * l.series.map((s) => s.name) // => ['a', 'b']
 * l.extent // => { x: [0, 2], y: [0, 1] }
 * ```
 */
export function points(
  sets: readonly PointSet[],
  options: { readonly mode?: ColourMode; readonly colourOffset?: number } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => points(sets, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const bg = tokens(mode).bg
  let slot = options.colourOffset ?? 0
  const allX: number[] = []
  const allY: number[] = []
  const series: SeriesOption[] = []
  for (const p of sets) {
    const { x, y } = coordinates(p)
    allX.push(...x)
    allY.push(...y)
    const n = x.length
    // A symbol of 'none' draws no label in ECharts: the marker is a circle of size zero instead.
    const hidden = p.symbol === 'none'
    const size = hidden ? 0 : (p.size ?? 7)
    const [smin, smax] = p.sizeRange ?? [4, 28]
    const sizeOf = (() => {
      if (p.sizes === undefined) return (): number => size
      const sz = p.sizes
      const lo = minOf(sz.map(Math.abs))
      const hi = maxOf(sz.map(Math.abs))
      return (i: number): number => {
        const a = Math.abs(sz[i] ?? 0)
        const f = hi > lo ? (a - lo) / (hi - lo) : 0.5
        return Math.sqrt(smin ** 2 + f * (smax ** 2 - smin ** 2))
      }
    })()
    const valueColour =
      p.values === undefined
        ? undefined
        : colourScale(sequentialRamp('viridis'), (extentOf(p.values) ?? [0, 1]) as [number, number])
    const groups = p.groups ?? new Array<string | number>(n).fill('__all')
    if (groups.length !== n) throw new OmaiError('shape', 'points: give one group per point')
    const order: (string | number)[] = []
    for (const g of groups) if (!order.includes(g)) order.push(g)
    if (p.groups !== undefined) order.sort((a, b) => (typeof a === 'number' && typeof b === 'number' ? a - b : 0))
    order.forEach((g, gi) => {
      const colour = colourFor(p.colour, mode, slot)
      slot += 1
      const data: Record<string, unknown>[] = []
      for (let i = 0; i < n; i++) {
        if (groups[i] !== g) continue
        data.push({
          value: [x[i], y[i]],
          ...(p.sizes === undefined || hidden ? {} : { symbolSize: sizeOf(i) }),
          ...(valueColour === undefined ? {} : { itemStyle: { color: valueColour(p.values?.[i] ?? 0) } }),
          ...(p.labels?.[i] === undefined
            ? {}
            : { label: { show: true, formatter: p.labels[i], position: p.labelPosition ?? 'right' } }),
          index: i,
        })
      }
      const name =
        g === '__all' ? (p.name ?? `__points${series.length}`) : (p.groupNames?.[String(g)] ?? p.name ?? String(g))
      series.push({
        type: 'scatter',
        name,
        data,
        symbol: hidden
          ? 'circle'
          : (p.symbol ?? (p.shapes === true ? (MARKERS[gi % MARKERS.length] ?? 'circle') : 'circle')),
        symbolSize: size,
        color: colour,
        itemStyle: {
          color: colour,
          opacity: p.opacity ?? 0.9,
          ...(p.outline === false ? {} : { borderColor: bg, borderWidth: 0.8 }),
        },
        label: {
          color: p.labelColour === undefined ? tokens(mode).fgSecondary : colourFor(p.labelColour, mode),
          ...(p.labelSize === undefined ? {} : { fontSize: p.labelSize }),
          ...(p.labelWeight === undefined ? {} : { fontWeight: p.labelWeight }),
          ...(p.labelFamily === undefined ? {} : { fontFamily: p.labelFamily }),
        },
        emphasis: { scale: 1.3 },
        clip: true,
      } as SeriesOption)
    })
  }
  const xe = extentOf(allX)
  const ye = extentOf(allY)
  return layer(series, { ...(xe === undefined ? {} : { x: xe }), ...(ye === undefined ? {} : { y: ye }) })
}

interface RugApi {
  coord(p: readonly number[]): number[]
  value(dim: number): number
}

/**
 * A rug: a short tick per value along the bottom (x) or left (y) edge of the plot.
 *
 * @param values - The values.
 * @param options - Axis (default x), tick length in pixels (default 8), colour, opacity and colour mode.
 * @returns The layer.
 * @example
 * ```ts
 * import { rug } from 'omai-render/charts'
 * rug([0.5, 1.5]).extent // => { x: [0.5, 1.5] }
 * ```
 */
export function rug(
  values: TensorLike | readonly number[],
  options: {
    readonly axis?: 'x' | 'y'
    readonly length?: number
    readonly colour?: string
    readonly opacity?: number
    readonly mode?: ColourMode
    readonly name?: string
  } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => rug(values, { ...options, mode }))
  const v = numbers(values).filter(Number.isFinite)
  const axis = options.axis ?? 'x'
  const length = options.length ?? 8
  const colour = colourFor(options.colour ?? 'fg', options.mode ?? 'light')
  const series = {
    type: 'custom',
    name: options.name ?? '__rug',
    data: v.map((u) => [u]),
    encode: axis === 'x' ? { x: 0 } : { y: 0 },
    renderItem: (
      params: { coordSys: { x: number; y: number; width: number; height: number } },
      api: RugApi,
    ): Record<string, unknown> => {
      const c = params.coordSys
      const u = api.value(0)
      if (axis === 'x') {
        const [px = 0] = api.coord([u, 0])
        return {
          type: 'line',
          shape: { x1: px, y1: c.y + c.height, x2: px, y2: c.y + c.height - length },
          style: { stroke: colour, opacity: options.opacity ?? 0.6, lineWidth: 1 },
        }
      }
      const [, py = 0] = api.coord([0, u])
      return {
        type: 'line',
        shape: { x1: c.x, y1: py, x2: c.x + length, y2: py },
        style: { stroke: colour, opacity: options.opacity ?? 0.6, lineWidth: 1 },
      }
    },
    silent: true,
    tooltip: { show: false },
    z: 3,
  } as unknown as SeriesOption
  const e = extentOf(v)
  return layer([series], e === undefined ? {} : axis === 'x' ? { x: e } : { y: e })
}

/**
 * A scatter chart of point sets.
 *
 * @param sets - The point sets.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @index function A scatter plot coloured by group: `scatterChart([{ x, y, groups }])`.
 * @example
 * ```ts
 * import { scatterChart } from 'omai-render/charts'
 * scatterChart([{ x: [[0, 0], [1, 1]] }], { aspect: 'equal' }).series.length // => 1
 * ```
 */
export function scatterChart(sets: readonly PointSet[], options: PlotOptions = {}): EChartsOption {
  return cartesian([points(sets, { mode: modeOf(options) })], options)
}

/**
 * Error bars in x and y: for each point a horizontal bar of ± its x error and a vertical bar of ± its y error, with
 * small caps, beside the points themselves.
 *
 * @param data - The points and their errors (either may be absent).
 * @param options - Colour mode, a name, a colour and the caps' size in data units (default a 60th of the range).
 * @returns The layer.
 * @example
 * ```ts
 * import { errorBars } from 'omai-render/charts'
 * errorBars([{ x: 1, y: 2, yErr: 0.5 }]).extent.y // => [1.5, 2.5]
 * ```
 */
export function errorBars(
  data: readonly { readonly x: number; readonly y: number; readonly xErr?: number; readonly yErr?: number }[],
  options: { readonly mode?: ColourMode; readonly name?: string; readonly colour?: string; readonly cap?: number } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => errorBars(data, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const colour = colourFor(options.colour, mode, 0)
  const xs = data.flatMap((d) => [d.x - (d.xErr ?? 0), d.x + (d.xErr ?? 0)])
  const ys = data.flatMap((d) => [d.y - (d.yErr ?? 0), d.y + (d.yErr ?? 0)])
  const xr = extentOf(xs) ?? [0, 1]
  const yr = extentOf(ys) ?? [0, 1]
  const capX = options.cap ?? (xr[1] - xr[0]) / 60
  const capY = options.cap ?? (yr[1] - yr[0]) / 60
  const segs: [number, number][] = []
  const gap: [number, number] = [Number.NaN, Number.NaN]
  for (const d of data) {
    if (d.xErr !== undefined) {
      const [a, b] = [d.x - d.xErr, d.x + d.xErr]
      segs.push([a, d.y], [b, d.y], gap, [a, d.y - capY], [a, d.y + capY], gap, [b, d.y - capY], [b, d.y + capY], gap)
    }
    if (d.yErr !== undefined) {
      const [a, b] = [d.y - d.yErr, d.y + d.yErr]
      segs.push([d.x, a], [d.x, b], gap, [d.x - capX, a], [d.x + capX, a], gap, [d.x - capX, b], [d.x + capX, b], gap)
    }
  }
  const series: SeriesOption[] = [
    {
      type: 'line',
      name: `__${options.name ?? 'errors'}`,
      data: segs.map(([x, y]) => (Number.isNaN(x) ? '-' : [x, y])),
      connectNulls: false,
      showSymbol: false,
      lineStyle: { color: colour, width: 1.25 },
      silent: true,
      animation: false,
    } as SeriesOption,
    {
      type: 'scatter',
      name: options.name ?? 'points',
      data: data.map((d) => [d.x, d.y]),
      itemStyle: { color: colour },
      symbolSize: 7,
    },
  ]
  return layer(series, { x: xr, y: yr })
}

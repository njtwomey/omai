/**
 * Bar charts as layers and charts: grouped or stacked bars over categories (vertical or horizontal, with error bars),
 * histograms binned by the engine's `stats.histogram` and drawn as adjoining bars on a value axis, and Pareto charts
 * (sorted bars with the cumulative share as a line on a second axis).
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf, OmaiError } from 'omai-engine/base'
import { type HistogramOptions, histogram as histogramOf } from 'omai-engine/stats'
import { type TensorLike, toFlat } from 'omai-engine/tensor'
import { extentOf } from '../figure/limits.ts'
import { formatNumber } from '../theme/format.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { colourFor, modeOf, numbers } from './_data.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'

/** A named group of bar heights, one per category. */
export interface BarGroup {
  readonly name?: string
  readonly values: TensorLike | readonly number[]
  readonly colour?: string
  /** Error bar half-widths, one per category. */
  readonly errors?: readonly number[]
  /** Asymmetric error bars: the low and high end per category (a confidence interval); overrides `errors`. */
  readonly intervals?: readonly (readonly [number, number])[]
}

/** Options of {@link bars}. */
export interface BarOptions {
  readonly mode?: ColourMode
  /** Stack the groups rather than placing them side by side (default false). */
  readonly stacked?: boolean
  /** Horizontal bars, categories on the y axis (default false). */
  readonly horizontal?: boolean
  /** Write each bar's value on it (default false). */
  readonly labels?: boolean
  /** Size of those value labels in pixels (default the figure's type size). */
  readonly labelSize?: number
  readonly colourOffset?: number
}

const BAR_GAP = 0.3
const CATEGORY_GAP = 0.2
const BAR_MAX = 48

/**
 * The offset, in pixels from the category's centre, of group `i`'s bar when `n` groups sit side by side in a band of
 * the given width (ECharts' layout with this file's bar gap, category gap and maximum bar width).
 *
 * @param band - The category band's width in pixels.
 * @param n - The number of groups.
 * @param i - The group.
 * @returns The offset of the bar's centre.
 * @example
 * ```ts
 * import { barOffset } from 'omai-render/charts'
 * barOffset(100, 1, 0) // => 0
 * barOffset(100, 2, 0) < 0 && barOffset(100, 2, 1) > 0 // => true
 * ```
 */
export function barOffset(band: number, n: number, i: number): number {
  if (n <= 1) return 0
  const raw = (band * (1 - CATEGORY_GAP)) / (n + (n - 1) * BAR_GAP)
  const w = Math.min(BAR_MAX, raw)
  const total = n * w + (n - 1) * BAR_GAP * w
  return -total / 2 + i * w * (1 + BAR_GAP) + w / 2
}

interface ErrorApi {
  coord(p: readonly (number | string)[]): number[]
  value(dim: number): number
  size(p: readonly number[]): number[]
}

/**
 * Bars over categories: one series per group, side by side or stacked, vertical or horizontal, with optional error
 * bars.
 *
 * @param categories - The category names.
 * @param groups - The groups (or a single array of heights).
 * @param options - Stacking, orientation, labels and colour mode.
 * @returns The layer, with the categories on the x (or y) axis.
 * @throws OmaiError with code `shape` when a group does not have one value per category.
 * @example
 * ```ts
 * import { bars } from 'omai-render/charts'
 * const l = bars(['a', 'b'], [{ name: 'n', values: [3, 5] }])
 * l.xCategories // => ['a', 'b']
 * l.extent // => { y: [0, 5], baseline: 'y' }
 * ```
 */
export function bars(
  categories: readonly string[],
  groups: readonly BarGroup[] | TensorLike | readonly number[],
  options: BarOptions = {},
): Layer {
  const mode = options.mode ?? 'light'
  const list: readonly BarGroup[] =
    Array.isArray(groups) && groups.length > 0 && typeof groups[0] === 'object' && !Array.isArray(groups[0])
      ? (groups as readonly BarGroup[])
      : [{ values: groups as TensorLike }]
  const horizontal = options.horizontal === true
  const totals = new Array<number>(categories.length).fill(0)
  const negatives = new Array<number>(categories.length).fill(0)
  const heights: number[] = []
  const series: SeriesOption[] = []
  // Where each group's stack starts in each category (positive and negative values stack separately, as in ECharts).
  const stackBase: number[][] = []
  {
    const up = new Array<number>(categories.length).fill(0)
    const down = new Array<number>(categories.length).fill(0)
    for (const g of list) {
      const v = numbers(g.values)
      stackBase.push(
        options.stacked === true ? v.map((h, k) => (h >= 0 ? (up[k] ?? 0) : (down[k] ?? 0))) : v.map(() => 0),
      )
      v.forEach((h, k) => {
        if (h >= 0) up[k] = (up[k] ?? 0) + h
        else down[k] = (down[k] ?? 0) + h
      })
    }
  }
  list.forEach((g, i) => {
    const v = numbers(g.values)
    if (v.length !== categories.length) throw new OmaiError('shape', 'bars: give one value per category')
    const colour = colourFor(g.colour, mode, (options.colourOffset ?? 0) + i)
    v.forEach((h, k) => {
      if (h >= 0) totals[k] = (totals[k] ?? 0) + h
      else negatives[k] = (negatives[k] ?? 0) + h
      const iv = g.intervals?.[k]
      if (iv !== undefined) heights.push(iv[0], iv[1])
      else heights.push(h + (g.errors?.[k] ?? 0), h - (g.errors?.[k] ?? 0))
    })
    series.push({
      type: 'bar',
      name: g.name ?? (list.length === 1 ? '__bars' : `group ${i + 1}`),
      data: v,
      color: colour,
      itemStyle: {
        color: colour,
        borderRadius: options.stacked === true ? 0 : horizontal ? [0, 2, 2, 0] : [2, 2, 0, 0],
      },
      ...(options.stacked === true ? { stack: 'total' } : {}),
      ...(options.labels === true
        ? {
            label: {
              show: true,
              position: horizontal ? 'right' : 'top',
              ...(options.labelSize === undefined ? {} : { fontSize: options.labelSize }),
              // Short numbers, not the raw float (0.8068181818181818).
              formatter: (p: { readonly value?: unknown }): string => {
                const h = Array.isArray(p.value) ? p.value[p.value.length - 1] : p.value
                return typeof h === 'number' ? formatNumber(h, { digits: 3 }) : String(h ?? '')
              },
            },
          }
        : {}),
      barMaxWidth: BAR_MAX,
      barGap: `${BAR_GAP * 100}%`,
      barCategoryGap: `${CATEGORY_GAP * 100}%`,
      emphasis: { focus: 'series' },
    } as SeriesOption)
    if (g.errors !== undefined || g.intervals !== undefined) {
      const errors = g.errors
      const intervals = g.intervals
      // Each bar's [category, low, high] (no bar where an interval is missing or not finite).
      const ends = v.flatMap((h, k) => {
        const iv = intervals?.[k]
        if (intervals !== undefined)
          return iv !== undefined && Number.isFinite(iv[0]) && Number.isFinite(iv[1]) ? [[k, iv[0], iv[1]]] : []
        const top = (stackBase[i]?.[k] ?? 0) + h
        const e = errors?.[k] ?? 0
        return [[k, top - e, top + e]]
      })
      series.push({
        type: 'custom',
        name: `__errors${i}`,
        data: ends,
        renderItem: (_params: unknown, api: ErrorApi): Record<string, unknown> => {
          const k = api.value(0)
          const lo = api.value(1)
          const hi = api.value(2)
          const cat = categories[k] ?? k
          const a0 = api.coord(horizontal ? [lo, cat] : [cat, lo])
          const b0 = api.coord(horizontal ? [hi, cat] : [cat, hi])
          // Grouped bars sit beside each other in the category band; put the error bar on this group's bar.
          const band = horizontal ? (api.size([0, 1])[1] ?? 0) : (api.size([1, 0])[0] ?? 0)
          const shift = options.stacked === true ? 0 : barOffset(band, list.length, i)
          const a = horizontal ? [a0[0], (a0[1] ?? 0) + shift] : [(a0[0] ?? 0) + shift, a0[1]]
          const b = horizontal ? [b0[0], (b0[1] ?? 0) + shift] : [(b0[0] ?? 0) + shift, b0[1]]
          const w = 4
          const lines = horizontal
            ? [
                [a[0], (a[1] ?? 0) - w, a[0], (a[1] ?? 0) + w],
                [a[0], a[1], b[0], b[1]],
                [b[0], (b[1] ?? 0) - w, b[0], (b[1] ?? 0) + w],
              ]
            : [
                [(a[0] ?? 0) - w, a[1], (a[0] ?? 0) + w, a[1]],
                [a[0], a[1], b[0], b[1]],
                [(b[0] ?? 0) - w, b[1], (b[0] ?? 0) + w, b[1]],
              ]
          return {
            type: 'group',
            children: lines.map(([x1, y1, x2, y2]) => ({
              type: 'line',
              shape: { x1, y1, x2, y2 },
              style: { stroke: colourFor('fg', mode), lineWidth: 1.2 },
            })),
          }
        },
        silent: true,
        tooltip: { show: false },
        z: 4,
      } as unknown as SeriesOption)
    }
  })
  const values = options.stacked === true ? [...totals, ...negatives, 0] : [...heights, 0]
  const e = extentOf(values) ?? [0, 1]
  return horizontal
    ? layer(series, { x: e, baseline: 'x' }, { yCategories: categories })
    : layer(series, { y: e, baseline: 'y' }, { xCategories: categories })
}

/** Options of {@link histogram}. */
export interface HistogramLayerOptions extends HistogramOptions {
  /** Normalise to a density that integrates to one (default false: counts). */
  readonly density?: boolean
  /** Accumulate the bins (default false). */
  readonly cumulative?: boolean
  readonly name?: string
  readonly colour?: string
  readonly opacity?: number
  /** Draw the outline of the histogram only (a step curve) rather than filled bars. */
  readonly outline?: boolean
  /** Bins along the y axis, bars growing to the right (a marginal histogram beside a scatter plot). */
  readonly horizontal?: boolean
  readonly mode?: ColourMode
  readonly colourOffset?: number
}

interface RectApi {
  coord(p: readonly number[]): number[]
  value(dim: number): number
}

/**
 * A histogram of values as a layer: bins from omai-engine's `stats.histogram` (counts, densities, the automatic bin
 * rules, explicit edges, weights), drawn as adjoining bars on a value axis, so uneven bins keep their widths.
 *
 * @param values - The data.
 * @param options - Bins, range, weights, density, cumulative, outline and colour.
 * @returns The layer.
 * @index function A histogram layer of values: `histogram(values, { bins: 20 })`.
 * @example
 * ```ts
 * import { histogram } from 'omai-render/charts'
 * const l = histogram([0, 1, 1, 2, 3], { bins: 3 })
 * l.extent // => { x: [0, 3], y: [0, 2], baseline: 'y' }
 * ```
 */
export function histogram(values: TensorLike | readonly number[], options: HistogramLayerOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const { density = false, cumulative = false } = options
  const horizontal = options.horizontal === true
  const h = histogramOf(values as TensorLike, {
    ...(options.bins === undefined ? {} : { bins: options.bins }),
    ...(options.range === undefined ? {} : { range: options.range }),
    ...(options.weights === undefined ? {} : { weights: options.weights }),
    density: density && !cumulative,
  })
  const edges = toFlat(h.edges)
  let counts = toFlat(h.counts)
  if (cumulative) {
    let acc = 0
    const total = counts.reduce((a, b) => a + b, 0)
    counts = counts.map((c) => {
      acc += c
      return density && total > 0 ? acc / total : acc
    })
  }
  const colour = colourFor(options.colour, mode, options.colourOffset ?? 0)
  const data = counts.map((c, i) => [edges[i] ?? 0, edges[i + 1] ?? 0, c])
  const name = options.name ?? '__histogram'
  const series: SeriesOption =
    options.outline === true
      ? ({
          type: 'line',
          name,
          step: horizontal ? 'start' : 'end',
          data: [...data.map(([a, , c]) => [a, c]), [edges[edges.length - 1] ?? 0, counts[counts.length - 1] ?? 0]].map(
            ([a, c]) => (horizontal ? [c, a] : [a, c]),
          ),
          showSymbol: false,
          color: colour,
          lineStyle: { color: colour, width: 1.5 },
        } as SeriesOption)
      : ({
          type: 'custom',
          name,
          data,
          encode: horizontal ? { y: [0, 1], x: 2, tooltip: [0, 1, 2] } : { x: [0, 1], y: 2, tooltip: [0, 1, 2] },
          dimensions: ['from', 'to', density ? 'density' : 'count'],
          renderItem: (_params: unknown, api: RectApi): Record<string, unknown> => {
            const at = (edge: number, count: number): number[] => api.coord(horizontal ? [count, edge] : [edge, count])
            const [ax = 0, ay = 0] = at(api.value(0), 0)
            const [bx = 0, by = 0] = at(api.value(1), api.value(2))
            const gap = 0.5
            const shape = horizontal
              ? { x: ax, y: by + gap, width: Math.max(0, bx - ax), height: Math.max(0, ay - by - gap) }
              : { x: ax, y: by, width: Math.max(0, bx - ax - gap), height: Math.max(0, ay - by) }
            return { type: 'rect', shape, style: { fill: colour, opacity: options.opacity ?? 0.75 } }
          },
          color: colour,
          clip: true,
        } as unknown as SeriesOption)
  const span: [number, number] = [edges[0] ?? 0, edges[edges.length - 1] ?? 1]
  const height: [number, number] = [0, Math.max(0, maxOf(counts))]
  return layer([series], horizontal ? { x: height, y: span, baseline: 'x' } : { x: span, y: height, baseline: 'y' })
}

/**
 * Pareto layers: categories sorted by value as bars, and the cumulative share as a line on the secondary axis (give
 * the plot `y2: { limits: [0, 1], format: 'percent' }`).
 *
 * @param categories - The category names.
 * @param values - Their values (non-negative).
 * @param options - Colour mode.
 * @returns The bar layer and the cumulative-share layer.
 * @example
 * ```ts
 * import { pareto } from 'omai-render/charts'
 * pareto(['a', 'b'], [1, 3])[1].series[0].data // => [0.75, 1]
 * ```
 */
export function pareto(
  categories: readonly string[],
  values: readonly number[],
  options: { readonly mode?: ColourMode } = {},
): Layer[] {
  const mode = options.mode ?? 'light'
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0])
  const sorted = order.map(([, i]) => categories[i] ?? String(i))
  const v = order.map(([x]) => x)
  const total = v.reduce((a, b) => a + b, 0)
  let acc = 0
  const share = v.map((x) => {
    acc += x
    return total > 0 ? acc / total : 0
  })
  const line = layer(
    [
      {
        type: 'line',
        name: 'cumulative share',
        data: share,
        showSymbol: true,
        symbolSize: 5,
        color: colourFor('tab1', mode),
        lineStyle: { color: colourFor('tab1', mode), width: 2 },
      } as SeriesOption,
    ],
    { y2: [0, 1] },
    { secondary: true, xCategories: sorted },
  )
  return [bars(sorted, v, { mode }), line]
}

/**
 * A Pareto chart: categories sorted by value as bars, with the cumulative share as a line on a secondary axis.
 *
 * @param categories - The category names.
 * @param values - Their values (non-negative).
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { paretoChart } from 'omai-render/charts'
 * const o = paretoChart(['a', 'b', 'c'], [1, 5, 2])
 * o.xAxis[0].data // => ['b', 'c', 'a']
 * ```
 */
export function paretoChart(
  categories: readonly string[],
  values: readonly number[],
  options: PlotOptions = {},
): EChartsOption {
  return cartesian(pareto(categories, values, { mode: modeOf(options) }), {
    y2: { limits: [0, 1], format: 'percent' },
    tooltip: 'axis',
    ...options,
  })
}

/**
 * A bar chart.
 *
 * @param categories - The category names.
 * @param groups - The groups, or one array of heights.
 * @param options - Plot options and bar options.
 * @returns The ECharts option.
 * @index function A bar chart over categories: `barChart(['a', 'b'], [3, 5])`.
 * @example
 * ```ts
 * import { barChart } from 'omai-render/charts'
 * barChart(['x', 'y'], [1, 2], { horizontal: true }).yAxis[0].data // => ['x', 'y']
 * ```
 */
export function barChart(
  categories: readonly string[],
  groups: readonly BarGroup[] | TensorLike | readonly number[],
  options: PlotOptions & Omit<BarOptions, 'mode'> = {},
): EChartsOption {
  const { stacked, horizontal, labels, colourOffset, ...plot } = options
  const l = bars(categories, groups, {
    mode: modeOf(options),
    ...(stacked === undefined ? {} : { stacked }),
    ...(horizontal === undefined ? {} : { horizontal }),
    ...(labels === undefined ? {} : { labels }),
    ...(colourOffset === undefined ? {} : { colourOffset }),
  })
  return cartesian([l], { tooltip: 'axis', ...plot })
}

/**
 * A waterfall chart: a running total built up step by step, each step a floating bar from the total before it to
 * the total after (rises and falls in different colours), with the final total as a full bar.
 *
 * @param steps - The steps in order: names and signed amounts.
 * @param options - Plot options, the start's name and value (default none) and the total's name (default 'total').
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { waterfallChart } from 'omai-render/charts'
 * waterfallChart([{ name: 'a', value: 3 }, { name: 'b', value: -1 }]).series.length // => 3
 * ```
 */
export function waterfallChart(
  steps: readonly { readonly name: string; readonly value: number }[],
  options: PlotOptions & {
    readonly start?: { readonly name: string; readonly value: number }
    readonly total?: string
  } = {},
): EChartsOption {
  const { start, total = 'total', ...plot } = options
  const mode = modeOf(plot)
  const names = [...(start === undefined ? [] : [start.name]), ...steps.map((s) => s.name), total]
  let running = start?.value ?? 0
  const base: (number | null)[] = start === undefined ? [] : [0]
  const rise: (number | null)[] = start === undefined ? [] : [start.value]
  const fall: (number | null)[] = start === undefined ? [] : [null]
  for (const s of steps) {
    const next = running + s.value
    base.push(Math.min(running, next))
    rise.push(s.value >= 0 ? s.value : null)
    fall.push(s.value < 0 ? -s.value : null)
    running = next
  }
  base.push(0)
  rise.push(running)
  fall.push(null)
  const series: SeriesOption[] = [
    { type: 'bar', name: '__base', stack: 'w', data: base, itemStyle: { color: 'transparent' }, silent: true },
    { type: 'bar', name: 'rise', stack: 'w', data: rise, itemStyle: { color: colourFor(undefined, mode, 2) } },
    { type: 'bar', name: 'fall', stack: 'w', data: fall, itemStyle: { color: colourFor(undefined, mode, 3) } },
  ]
  const tops = base.map((b, i) => (b ?? 0) + (rise[i] ?? 0) + (fall[i] ?? 0))
  return cartesian([layer(series, { y: extentOf([0, ...tops]) ?? [0, 1] }, { xCategories: names })], plot)
}

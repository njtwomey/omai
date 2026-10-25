/**
 * The cartesian plot every chart builder shares: a `Layer` is ECharts series with the data extent they cover (and
 * optionally a colour bar or category labels), so layers from any builder combine, and `cartesian` draws layers on one
 * pair of axes, with ranges from the limit policies and the data aspect, log axes, legend, tooltip and colour bars.
 * `panelParts` and `chartFrame` are the pieces subplots reuse.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { type DataAspect, fitDataAspect, type Range } from '../figure/aspect.ts'
import { type Figure, figure as makeFigure, plotArea, type Rect } from '../figure/figure.ts'
import { type Limits, legibleRange, padRange, resolveLimits, unionRange } from '../figure/limits.ts'
import { axisStyle, baseOption } from '../theme/echarts.ts'
import { formatNumber, formatPercent, formatPower } from '../theme/format.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { type ColourSelect, SELECT_KEY } from './selection.ts'

/** The data extent of a layer, used by the axes' limit policies. `tight` asks for no padding (images, fields). */
export interface Extent {
  readonly x?: Range
  readonly y?: Range
  /** The extent on the secondary (right-hand) y axis. */
  readonly y2?: Range
  readonly tight?: boolean
  /** The axis on which the layer stands on a zero baseline (bars, histograms, densities): its range keeps zero. */
  readonly baseline?: 'x' | 'y'
}

/**
 * A layer of a cartesian plot: ECharts series (without axis indices; the plot assigns them) with the data extent they
 * cover, and optionally a colour bar (a continuous visual map), category labels for either axis, and whether the series
 * use the secondary y axis. Every chart builder returns layers, so layers from different builders combine in one plot.
 */
export interface Layer {
  readonly kind: 'layer'
  readonly series: readonly SeriesOption[]
  readonly extent: Extent
  /** A colour bar: a continuous visual map for the layer's series at index `target` (default 0). */
  readonly colourBar?: ColourBar
  readonly xCategories?: readonly string[]
  readonly yCategories?: readonly string[]
  readonly secondary?: boolean
}

/** A colour bar shown beside a plot. */
export interface ColourBar {
  readonly min: number
  readonly max: number
  readonly colours: readonly string[]
  readonly label?: string
  /** The index, within the layer's series, of the series the bar describes (default 0). */
  readonly target?: number
  /** Discrete levels (a piecewise bar) instead of a continuous ramp. */
  readonly levels?: readonly number[]
  /**
   * How the layer answers a range selection on the bar (dragging a continuous bar's handles, clicking a piecewise bar's
   * pieces). Without it the bar is a plain legend: no handles, no selectable pieces.
   */
  readonly select?: ColourSelect
}

/**
 * Make a layer.
 *
 * @param series - The ECharts series.
 * @param extent - The data extent.
 * @param extra - Colour bar, categories and the secondary axis flag.
 * @returns The layer.
 * @example
 * ```ts
 * import { layer } from 'mdml-render/charts'
 * layer([{ type: 'line', data: [[0, 1], [1, 2]] }], { x: [0, 1], y: [1, 2] }).series.length // => 1
 * ```
 */
export function layer(
  series: readonly SeriesOption[],
  extent: Extent = {},
  extra: Omit<Partial<Layer>, 'kind' | 'series' | 'extent'> = {},
): Layer {
  return { kind: 'layer', series, extent, ...extra }
}

/** The scale of an axis. */
export type AxisScale = 'linear' | 'log' | 'category' | 'time'

/** How an axis is labelled and ranged. */
export interface AxisSpec {
  readonly label?: string
  readonly scale?: AxisScale
  /** A limit policy or a fixed `[low, high]` (default: auto). */
  readonly limits?: Limits
  /** Show tick labels and the axis line (default true; split lines stay). */
  readonly show?: boolean
  readonly splitLine?: boolean
  readonly inverse?: boolean
  readonly format?: 'number' | 'percent' | 'power' | ((value: number) => string)
}

/** Options shared by every cartesian chart. */
export interface PlotOptions {
  readonly mode?: ColourMode
  /** The figure the chart fills (default `figure({ mode })`). */
  readonly figure?: Figure
  readonly title?: string
  readonly x?: AxisSpec
  readonly y?: AxisSpec
  /** The secondary (right-hand) y axis, used by layers marked `secondary`. */
  readonly y2?: AxisSpec
  /** The data aspect (default auto); `equal` widens a range so units are square in the plotting area. */
  readonly aspect?: DataAspect
  /** Show a legend of the named series (default: when more than one series is named). */
  readonly legend?: boolean
  /** Tooltip trigger (default `item`). */
  readonly tooltip?: 'axis' | 'item' | false
  /** The ranges shown before, for `expand` limit policies. */
  readonly previous?: { readonly x?: Range; readonly y?: Range }
  /** Wheel and drag zoom inside the plot (default false). */
  readonly zoom?: boolean
}

/** One placed panel: the ECharts components of a set of layers on one grid. */
export interface PanelParts {
  readonly grid: Readonly<Record<string, unknown>>
  readonly xAxis: Readonly<Record<string, unknown>>
  readonly yAxis: readonly Readonly<Record<string, unknown>>[]
  readonly series: readonly SeriesOption[]
  readonly visualMap: readonly Readonly<Record<string, unknown>>[]
  readonly ranges: { readonly x?: Range; readonly y?: Range }
  readonly legendNames: readonly string[]
}

function formatter(spec: AxisSpec | undefined, scale: AxisScale): (value: number) => string {
  const f = spec?.format
  if (typeof f === 'function') return f
  if (f === 'percent') return (v) => formatPercent(v, 0)
  if (f === 'power' || (f === undefined && scale === 'log')) return formatPower
  return (v) => formatNumber(v)
}

// ECharts' own rounding of a tick interval (numberUtil.nice with round): 1, 2, 3, 5 or 10 times a power of ten.
function echartsNice(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) return 1
  const e = Math.floor(Math.log10(x))
  const f = x / 10 ** e
  const n = f < 1.5 ? 1 : f < 2.5 ? 2 : f < 4 ? 3 : f < 7 ? 5 : 10
  return n * 10 ** e
}

// How many ticks a linear axis asks ECharts for (about one per 70 px across, 50 px up), and the labels at the ends
// shown only where the ends fall on ECharts' ticks (a padded range ends between ticks).
function linearTicks(
  r: Range,
  pixels: number,
  dim: 'x' | 'y',
): { readonly splitNumber: number; readonly showMinLabel: boolean; readonly showMaxLabel: boolean } {
  const splitNumber = Math.max(2, Math.min(10, Math.round(pixels / (dim === 'x' ? 70 : 50))))
  const step = echartsNice((r[1] - r[0]) / splitNumber)
  const onTick = (v: number): boolean => Math.abs(v / step - Math.round(v / step)) < 1e-6
  return { splitNumber, showMinLabel: onTick(r[0]), showMaxLabel: onTick(r[1]) }
}

/**
 * The range of a log axis under a limit policy: padding and `nice` (whole decades) work on the logarithms, and a fitted
 * range (auto or expand) ends on 1, 2 or 5 times a power of ten. Non-positive values are ignored.
 *
 * @param limits - The policy, or a fixed range.
 * @param extent - The data's extent.
 * @param previous - The range shown before (for `expand`).
 * @returns The range.
 * @example
 * ```ts
 * import { logRange } from 'mdml-render/charts'
 * logRange({ kind: 'auto', pad: 0.05 }, [0.25, 1.6]) // => [0.2, 2]
 * logRange({ kind: 'auto', nice: true }, [0.25, 1.6]) // => [0.1, 10]
 * ```
 */
export function logRange(limits: Limits, extent: Range | undefined, previous?: Range): Range {
  const positive =
    extent === undefined ? undefined : ([Math.max(extent[0], 1e-300), Math.max(extent[1], 1e-300)] as const)
  const toLog = (r: Range | undefined): Range | undefined =>
    r === undefined ? undefined : [Math.log10(Math.max(r[0], 1e-300)), Math.log10(Math.max(r[1], 1e-300))]
  const policy: Limits = Array.isArray(limits) ? (toLog(limits as Range) ?? [0, 1]) : limits
  // Nice rounding on a log axis means whole decades, not the linear 1-2-5 rule.
  const decades = !Array.isArray(policy) && 'nice' in policy && policy.nice === true
  const plain: Limits =
    decades && !Array.isArray(policy) && (policy.kind === 'auto' || policy.kind === 'expand')
      ? { ...policy, nice: false, pad: policy.pad ?? 0 }
      : policy
  const r = resolveLimits(plain, toLog(positive), toLog(previous))
  if (decades) return [10 ** Math.floor(r[0] + 1e-9), 10 ** Math.ceil(r[1] - 1e-9)]
  // A fitted range ends on 1, 2 or 5 times a power of ten, so its end labels read 0.2 and 2 rather than 0.249 and 1.668.
  const fitted = 'kind' in policy && (policy.kind === 'auto' || policy.kind === 'expand')
  return fitted ? [niceDown(10 ** r[0]), niceUp(10 ** r[1])] : [10 ** r[0], 10 ** r[1]]
}

function niceDown(v: number): number {
  const e = Math.floor(Math.log10(v) + 1e-9)
  const f = v / 10 ** e
  return (f >= 5 - 1e-9 ? 5 : f >= 2 - 1e-9 ? 2 : 1) * 10 ** e
}

function niceUp(v: number): number {
  const e = Math.floor(Math.log10(v) - 1e-9)
  const f = v / 10 ** e
  return (f <= 1 + 1e-9 ? 1 : f <= 2 + 1e-9 ? 2 : f <= 5 + 1e-9 ? 5 : 10) * 10 ** e
}

/**
 * The ECharts components of a set of layers placed in a rectangle: one grid, its axes (ranges from the limit policies
 * and the data aspect), its series with axis indices, and its colour bars. Used by {@link cartesian} and by
 * `subplots` in compose.ts.
 *
 * @param layers - The layers.
 * @param options - Axes, aspect and colour mode.
 * @param rect - The plotting area in pixels.
 * @param index - The grid's index (and its x axis's).
 * @param yIndex - The index of its first y axis.
 * @param seriesIndex - The global index of its first series (for colour bars).
 * @returns The parts.
 * @example
 * ```ts
 * import { layer, panelParts } from 'mdml-render/charts'
 * const p = panelParts([layer([{ type: 'line', data: [[0, 0], [2, 4]] }], { x: [0, 2], y: [0, 4], tight: true })], {}, { x: 50, y: 20, width: 300, height: 200 }, 0, 0, 0)
 * p.ranges.x // => [0, 2]
 * ```
 */
export function panelParts(
  layers: readonly Layer[],
  options: Omit<PlotOptions, 'figure' | 'title' | 'legend' | 'tooltip' | 'zoom'>,
  rect: Rect,
  index: number,
  yIndex: number,
  seriesIndex: number,
): PanelParts {
  const mode = options.mode ?? 'light'
  const t = tokens(mode)
  const style = axisStyle(mode)
  const tight = layers.length > 0 && layers.every((l) => l.extent.tight === true)
  const pad = tight ? 0 : 0.04
  const xCats = layers.find((l) => l.xCategories !== undefined)?.xCategories
  const yCats = layers.find((l) => l.yCategories !== undefined)?.yCategories
  const xScale: AxisScale = xCats !== undefined ? 'category' : (options.x?.scale ?? 'linear')
  const yScale: AxisScale = yCats !== undefined ? 'category' : (options.y?.scale ?? 'linear')
  const primary = layers.filter((l) => l.secondary !== true)
  const xExtent = unionRange(layers.map((l) => l.extent.x))
  const yExtent = unionRange(primary.map((l) => l.extent.y))
  const y2Extent = unionRange(layers.filter((l) => l.secondary === true).map((l) => l.extent.y2 ?? l.extent.y))
  const range = (
    scale: AxisScale,
    spec: AxisSpec | undefined,
    extent: Range | undefined,
    prev?: Range,
    dim?: 'x' | 'y',
  ): Range | undefined => {
    if (scale === 'category') return undefined
    const zero = dim !== undefined && layers.some((l) => l.extent.baseline === dim)
    const given = spec?.limits
    // A fixed range of one point (a path with one penalty) is widened, not refused.
    const limits =
      given !== undefined && Array.isArray(given) && !((given[1] as number) > (given[0] as number))
        ? padRange(given as Range)
        : (given ?? { kind: 'auto' as const, pad, includeZero: zero && scale === 'linear' })
    return scale === 'log' ? logRange(limits, extent, prev) : legibleRange(resolveLimits(limits, extent, prev))
  }
  let xr = range(xScale, options.x, xExtent, options.previous?.x, 'x')
  let yr = range(yScale, options.y, yExtent, options.previous?.y, 'y')
  if (xr !== undefined && yr !== undefined && xScale === 'linear' && yScale === 'linear') {
    const fitted = fitDataAspect(xr, yr, rect, options.aspect ?? 'auto')
    xr = fitted.x
    yr = fitted.y
  }
  const axis = (
    dim: 'x' | 'y',
    scale: AxisScale,
    spec: AxisSpec | undefined,
    r: Range | undefined,
    cats: readonly string[] | undefined,
    gridIndex: number,
    position?: 'right',
  ): Record<string, unknown> => {
    const show = spec?.show ?? true
    const ticks =
      r !== undefined && scale === 'linear' ? linearTicks(r, dim === 'x' ? rect.width : rect.height, dim) : undefined
    return {
      ...style,
      type: scale === 'linear' ? 'value' : scale === 'category' ? 'category' : scale,
      gridIndex,
      ...(position === undefined ? {} : { position }),
      ...(spec?.label === undefined || !show ? {} : { name: spec.label }),
      ...(cats === undefined ? {} : { data: [...cats] }),
      ...(r === undefined ? {} : { min: r[0], max: r[1] }),
      ...(ticks === undefined ? {} : { splitNumber: ticks.splitNumber }),
      ...(spec?.inverse === true ? { inverse: true } : {}),
      ...(scale === 'log' ? { logBase: 10 } : {}),
      nameGap: dim === 'x' ? style.nameGap : Math.round(style.nameGap * 1.5),
      axisLine: { ...style.axisLine, show, onZero: false },
      axisLabel: {
        ...style.axisLabel,
        show,
        ...(ticks === undefined ? {} : { showMinLabel: ticks.showMinLabel, showMaxLabel: ticks.showMaxLabel }),
        ...(scale === 'category' || scale === 'time' ? {} : { formatter: formatter(spec, scale) }),
      },
      splitLine: { ...style.splitLine, show: spec?.splitLine ?? (scale !== 'category' && position === undefined) },
      ...(scale === 'category' ? { boundaryGap: true } : {}),
    }
  }
  const yAxes: Record<string, unknown>[] = [axis('y', yScale, options.y, yr, yCats, index)]
  const hasSecondary = layers.some((l) => l.secondary === true)
  if (hasSecondary) {
    const y2Scale = options.y2?.scale ?? 'linear'
    yAxes.push(axis('y', y2Scale, options.y2, range(y2Scale, options.y2, y2Extent), undefined, index, 'right'))
  }
  const series: SeriesOption[] = []
  const visualMap: Record<string, unknown>[] = []
  const legendNames: string[] = []
  for (const l of layers) {
    const first = seriesIndex + series.length
    const select = l.colourBar?.select
    for (const [i, s] of l.series.entries()) {
      const yAxisIndex = l.secondary === true && hasSecondary ? yIndex + 1 : yIndex
      // The series a colour bar's selection recolours is found by id.
      const id = select !== undefined && i === select.series ? { id: selectableId(first + i) } : {}
      const placed = { ...s, ...id, xAxisIndex: index, yAxisIndex } as SeriesOption
      series.push(placed)
      const name = (s as { name?: unknown }).name
      if (typeof name === 'string' && !name.startsWith('__') && !legendNames.includes(name)) legendNames.push(name)
    }
    const bar = l.colourBar
    if (bar !== undefined) {
      const common = {
        ...(select === undefined
          ? {}
          : {
              id: `__colourbar-${first + (bar.target ?? 0)}`,
              [SELECT_KEY]: { series: selectableId(first + select.series), recolour: select.recolour },
            }),
        seriesIndex: first + (bar.target ?? 0),
        orient: 'vertical',
        left: rect.x + rect.width + 12,
        top: rect.y,
        itemWidth: 10,
        itemHeight: Math.max(40, rect.height - 24),
        textStyle: { color: t.muted, fontSize: 10, fontFamily: t.font },
        ...(bar.label === undefined ? {} : { textGap: 6 }),
        formatter: (v: number | string) => (typeof v === 'number' ? formatNumber(v, { digits: 3 }) : v),
      }
      visualMap.push(
        bar.levels === undefined
          ? {
              ...common,
              type: 'continuous',
              min: bar.min,
              max: bar.max,
              // Calculable where the layer answers a selection: the handles carry the ends' values, and dragging them
              // dims the values outside. A plain legend writes its ends' values instead.
              ...(select === undefined
                ? { text: [endLabel(bar.max, bar.label), formatNumber(bar.min, { digits: 3 })] }
                : bar.label === undefined
                  ? {}
                  : { text: [bar.label, ''] }),
              calculable: select !== undefined,
              inRange: { color: [...bar.colours] },
            }
          : {
              ...common,
              type: 'piecewise',
              pieces: bar.levels.slice(0, -1).map((lo, i) => ({
                min: lo,
                max: bar.levels?.[i + 1] ?? lo,
                color: bar.colours[i] ?? bar.colours[bar.colours.length - 1],
              })),
              // One piece per band, stacked to fill the bar's height at most (not each as tall as the bar).
              itemHeight: Math.max(
                8,
                Math.min(24, Math.floor((rect.height - 24) / Math.max(1, bar.levels.length - 1)) - 2),
              ),
              itemGap: 2,
              // Hidden bands show in the grid's colour, visible on either theme.
              inactiveColor: t.grid,
              selectedMode: select === undefined ? false : 'multiple',
            },
      )
    }
  }
  return {
    grid: {
      left: rect.x,
      top: rect.y,
      width: rect.width,
      height: rect.height,
      containLabel: false,
      show: false,
      borderColor: t.grid,
    },
    xAxis: axis('x', xScale, options.x, xr, xCats, index),
    yAxis: yAxes,
    series,
    visualMap,
    ranges: { ...(xr === undefined ? {} : { x: xr }), ...(yr === undefined ? {} : { y: yr }) },
    legendNames,
  }
}

function selectableId(index: number): string {
  return `__selectable-${index}`
}

function endLabel(max: number, label: string | undefined): string {
  const v = formatNumber(max, { digits: 3 })
  return label === undefined ? v : `${label}\n${v}`
}

/**
 * The frame of a chart option: the base option for the mode, a title, a legend and a tooltip. Used by
 * {@link cartesian} and `subplots`.
 *
 * @param fig - The figure.
 * @param options - Title, legend names and tooltip trigger.
 * @returns The option's top-level fields.
 * @example
 * ```ts
 * import { chartFrame } from 'mdml-render/charts'
 * import { figure } from 'mdml-render/figure'
 * chartFrame(figure(), { title: 'Loss' }).title // => { text: 'Loss', left: 8, top: 4, textStyle: { fontSize: 13, fontWeight: 600, color: '#1f2328' } }
 * ```
 */
export function chartFrame(
  fig: Figure,
  options: {
    readonly title?: string
    readonly legendNames?: readonly string[]
    readonly legend?: boolean
    readonly tooltip?: 'axis' | 'item' | false
  },
): Record<string, unknown> {
  const base = baseOption(fig.mode, fig.fontSize)
  const t = tokens(fig.mode)
  const names = options.legendNames ?? []
  const showLegend = options.legend ?? names.length > 1
  const tooltip = options.tooltip ?? 'item'
  const title = options.title ?? fig.title
  return {
    ...base,
    ...(title === undefined
      ? {}
      : {
          title: {
            text: title,
            left: 8,
            top: 4,
            textStyle: { fontSize: fig.fontSize + 2, fontWeight: 600, color: t.fg },
          },
        }),
    legend: { ...base.legend, show: showLegend && names.length > 0, data: [...names], right: fig.padding.right },
    tooltip:
      tooltip === false
        ? { show: false }
        : {
            ...base.tooltip,
            trigger: tooltip,
            ...(tooltip === 'axis' ? { axisPointer: { type: 'line', snap: false } } : {}),
            valueFormatter: (v: unknown) => (typeof v === 'number' ? formatNumber(v) : String(v)),
          },
  }
}

/**
 * Draw layers on one pair of axes: the figure's plotting area, axes from the limit policies and the data aspect,
 * legend, tooltip and colour bars. Every single-panel chart builder is `cartesian` of its layers, so layers from
 * different builders combine freely (`cartesian([band(...), lines(...), points(...)])`).
 *
 * @param layers - The layers, drawn in order.
 * @param options - Figure, title, axes, aspect, legend, tooltip and zoom.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { cartesian, lines } from 'mdml-render/charts'
 * const option = cartesian([lines([{ x: [0, 1, 2], y: [0, 1, 4] }])], { x: { label: 'x', limits: [0, 2] } })
 * option.xAxis[0].min // => 0
 * ```
 */
export function cartesian(layers: readonly Layer[], options: PlotOptions = {}): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const hasBar = layers.some((l) => l.colourBar !== undefined)
  const hasY2 = layers.some((l) => l.secondary === true)
  const area = plotArea(fig)
  const rect: Rect = { ...area, width: area.width - (hasBar ? 64 : 0) - (hasY2 ? 40 : 0) }
  const parts = panelParts(layers, { ...options, mode }, rect, 0, 0, 0)
  const option: Record<string, unknown> = {
    ...chartFrame(
      { ...fig, mode },
      {
        ...(options.title === undefined ? {} : { title: options.title }),
        legendNames: parts.legendNames,
        ...(options.legend === undefined ? {} : { legend: options.legend }),
        ...(options.tooltip === undefined ? {} : { tooltip: options.tooltip }),
      },
    ),
    grid: [parts.grid],
    xAxis: [parts.xAxis],
    yAxis: [...parts.yAxis],
    series: [...parts.series],
    ...(parts.visualMap.length > 0 ? { visualMap: [...parts.visualMap] } : {}),
    ...(options.zoom === true
      ? {
          dataZoom: [
            { type: 'inside', xAxisIndex: [0], filterMode: 'none' },
            { type: 'inside', yAxisIndex: [0], filterMode: 'none' },
          ],
        }
      : {}),
  }
  return option as EChartsOption
}

/**
 * Charts of tables with many columns: parallel coordinates (one axis per column, one polyline per row) with axis
 * brushing that filters rows, radar charts, and scatter matrices (every pair of columns as a scatter, histograms on the
 * diagonal), plus the layout helpers overlays need to place themselves on these charts.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { MdmlError } from 'mdml-engine/base'
import type { Range } from '../figure/aspect.ts'
import { type Figure, figure as makeFigure, type Rect } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { extentOf, niceRange, padRange } from '../figure/limits.ts'
import { formatNumber } from '../theme/format.ts'
import { categoricalColour, colourScale, sequentialRamp } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { histogram } from './bars.ts'
import { type Panel, subplotRects, subplots } from './compose.ts'
import { chartFrame } from './plot.ts'
import { points } from './points.ts'

/**
 * Rows of numbers under named columns (one value per column per row), with an optional group per row (classes,
 * clusters) and names for numeric groups: the input of every multivariate chart.
 */
export interface Table {
  readonly columns: readonly string[]
  readonly rows: readonly (readonly number[])[]
  readonly groups?: readonly (string | number)[]
  readonly groupNames?: Readonly<Record<string, string>>
}

/** A brush on some of a table's columns: the interval kept on each, by column name. */
export type AxisBrush = Readonly<Record<string, Range>>

function check(data: Table, name: string): void {
  const n = data.columns.length
  if (n === 0) throw new MdmlError('shape', `${name}: the table has no columns`)
  for (const r of data.rows)
    if (r.length !== n) throw new MdmlError('shape', `${name}: every row needs one value per column (${n})`)
  if (data.groups !== undefined && data.groups.length !== data.rows.length)
    throw new MdmlError('shape', `${name}: give one group per row`)
}

function groupOrder(data: Table): (string | number)[] {
  const order: (string | number)[] = []
  for (const g of data.groups ?? []) if (!order.includes(g)) order.push(g)
  return order.sort((a, b) => (typeof a === 'number' && typeof b === 'number' ? a - b : 0))
}

const groupName = (data: Table, g: string | number): string => data.groupNames?.[String(g)] ?? String(g)

/**
 * The rows inside a brush: those whose value on every brushed column lies in that column's interval (unknown columns
 * are ignored; an empty brush keeps every row).
 *
 * @param data - The table.
 * @param brush - Intervals by column.
 * @returns The indices of the rows kept, in order.
 * @example
 * ```ts
 * import { brushRows } from 'mdml-render/charts'
 * const t = { columns: ['a', 'b'], rows: [[0, 5], [1, 6], [2, 7]] }
 * brushRows(t, { a: [0.5, 3] }) // => [1, 2]
 * brushRows(t, { a: [0.5, 3], b: [0, 6] }) // => [1]
 * ```
 */
export function brushRows(data: Table, brush: AxisBrush): number[] {
  const tests = Object.entries(brush).flatMap(([col, [lo, hi]]) => {
    const j = data.columns.indexOf(col)
    return j < 0 ? [] : [[j, Math.min(lo, hi), Math.max(lo, hi)] as const]
  })
  const out: number[] = []
  data.rows.forEach((r, i) => {
    if (tests.every(([j, lo, hi]) => (r[j] ?? Number.NaN) >= lo && (r[j] ?? Number.NaN) <= hi)) out.push(i)
  })
  return out
}

/**
 * A brush updated from an ECharts `axisareaselected` event of a parallel-coordinates chart (whose axis ids are the
 * column names, as {@link parallelCoordinates} sets them): the event's last interval replaces that column's, and an
 * empty selection (a click on the axis) clears it.
 *
 * @param brush - The brush before.
 * @param event - The event's parameters (`parallelAxisId` and `intervals`).
 * @returns The brush after.
 * @example
 * ```ts
 * import { brushFromAxisEvent } from 'mdml-render/charts'
 * brushFromAxisEvent({}, { parallelAxisId: 'a', intervals: [[3, 1]] }) // => { a: [1, 3] }
 * brushFromAxisEvent({ a: [1, 3] }, { parallelAxisId: 'a', intervals: [] }) // => {}
 * ```
 */
export function brushFromAxisEvent(brush: AxisBrush, event: unknown): AxisBrush {
  const e = event as { readonly parallelAxisId?: unknown; readonly intervals?: unknown }
  if (typeof e?.parallelAxisId !== 'string' || !Array.isArray(e.intervals)) return brush
  const { [e.parallelAxisId]: _, ...rest } = brush
  const last = e.intervals.at(-1) as unknown
  if (!Array.isArray(last) || last.length !== 2) return rest
  const [a, b] = last as [number, number]
  return { ...rest, [e.parallelAxisId]: [Math.min(a, b), Math.max(a, b)] }
}

/** Options of {@link parallelCoordinates}. */
export interface ParallelOptions {
  readonly mode?: ColourMode
  /** The figure the chart fills (default 640 by 400). */
  readonly figure?: Figure
  readonly title?: string
  /** Intervals kept on some axes: rows outside are drawn faint and grey, and each interval is marked on its axis. */
  readonly brush?: AxisBrush
  /** Fixed axis ranges by column (default each column's nice range). */
  readonly ranges?: Readonly<Record<string, Range>>
  /** Colour rows along viridis by this column's value instead of by group. */
  readonly colourBy?: string
  /** Line opacity (default 0.6) and width (default 1.2). */
  readonly opacity?: number
  readonly width?: number
  /** Show the legend of groups (default: with more than one group). */
  readonly legend?: boolean
  /** The axes' order, by column name (default the table's; columns left out keep their relative order at the end). */
  readonly order?: readonly string[]
  /** Columns whose axis runs downwards (largest value at the bottom). */
  readonly inverted?: readonly string[]
}

/**
 * A table with its columns in a given order: the named columns first, in that order, then the rest as they were.
 *
 * @param data - The table.
 * @param order - Column names.
 * @returns The table with its columns (and every row's values) permuted.
 * @example
 * ```ts
 * import { reorderColumns } from 'mdml-render/charts'
 * reorderColumns({ columns: ['a', 'b', 'c'], rows: [[1, 2, 3]] }, ['c', 'a']).rows // => [[3, 1, 2]]
 * ```
 */
export function reorderColumns(data: Table, order: readonly string[]): Table {
  const first = order.filter((c) => data.columns.includes(c))
  const columns = [...new Set([...first, ...data.columns])]
  const index = columns.map((c) => data.columns.indexOf(c))
  return { ...data, columns, rows: data.rows.map((r) => index.map((j) => r[j] ?? 0)) }
}

/** Where a parallel-coordinates chart puts its axes, in pixels of its figure. */
export interface ParallelLayout {
  readonly left: number
  readonly top: number
  readonly width: number
  readonly height: number
  /** The x position of each axis. */
  readonly axisX: readonly number[]
  /** Each axis's value range (bottom to top). */
  readonly ranges: readonly Range[]
}

/**
 * The placement of a parallel-coordinates chart's axes in its figure (as {@link parallelCoordinates} draws them), for
 * overlays and annotations.
 *
 * @param data - The table.
 * @param options - The figure and fixed ranges.
 * @returns The plotting box, each axis's x and each axis's range.
 * @example
 * ```ts
 * import { parallelLayout } from 'mdml-render/charts'
 * const l = parallelLayout({ columns: ['a', 'b', 'c'], rows: [[0, 1, 2], [1, 3, 5]] })
 * l.axisX.length // => 3
 * l.ranges[1] // => [1, 3]
 * ```
 */
export function parallelLayout(
  data: Table,
  options: Pick<ParallelOptions, 'figure' | 'ranges' | 'legend'> = {},
): ParallelLayout {
  const fig = options.figure ?? makeFigure()
  const left = 48
  const right = 48
  const top = 58
  const bottom = 28
  const width = Math.max(10, fig.width - left - right)
  const height = Math.max(10, fig.height - top - bottom)
  const n = data.columns.length
  const ranges = data.columns.map((c, j): Range => {
    const fixed = options.ranges?.[c]
    if (fixed !== undefined) return fixed
    const e = extentOf(data.rows.map((r) => r[j] ?? Number.NaN)) ?? [0, 1]
    return e[0] === e[1] ? [e[0] - 0.5, e[1] + 0.5] : niceRange(e)
  })
  return {
    left,
    top,
    width,
    height,
    axisX: data.columns.map((_, j) => (n === 1 ? left + width / 2 : left + (j * width) / (n - 1))),
    ranges,
  }
}

/**
 * Parallel coordinates: one vertical axis per column and one polyline per row, coloured by group (tab10) or along
 * viridis by a column. Drag along an axis to brush it (ECharts' area select, reported by the `axisareaselected` event;
 * turn it into a brush with {@link brushFromAxisEvent}); with `brush` given, rows outside every interval fade and the
 * intervals are drawn on their axes, so the picture is the same when exported.
 *
 * @param data - The table.
 * @param options - Figure, title, brush, ranges, colouring, line style and legend.
 * @returns The ECharts option.
 * @throws MdmlError with code `shape` when rows do not match the columns.
 * @index function Parallel coordinates of a table: `parallelCoordinates({ columns, rows })`.
 * @example
 * ```ts
 * import { parallelCoordinates } from 'mdml-render/charts'
 * const t = { columns: ['a', 'b'], rows: [[0, 5], [1, 6], [2, 7]], groups: [0, 0, 1] }
 * const o = parallelCoordinates(t, { brush: { a: [0.5, 3] } })
 * o.parallelAxis.map((a) => a.id) // => ['a', 'b']
 * o.series.map((s) => s.data.length) // => [1, 1, 1]
 * ```
 */
export function parallelCoordinates(table: Table, options: ParallelOptions = {}): EChartsOption {
  check(table, 'parallelCoordinates')
  const data = options.order === undefined ? table : reorderColumns(table, options.order)
  const inverted = new Set(options.inverted ?? [])
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const layout = parallelLayout(data, { ...options, figure: fig })
  const brush = options.brush ?? {}
  const brushed = Object.keys(brush).some((c) => data.columns.includes(c))
  const kept = new Set(brushed ? brushRows(data, brush) : data.rows.map((_, i) => i))
  const opacity = options.opacity ?? 0.6
  const lineWidth = options.width ?? 1.2
  const series: SeriesOption[] = []
  const faded = data.rows.filter((_, i) => !kept.has(i))
  if (faded.length > 0)
    series.push({
      type: 'parallel',
      name: '__outside',
      data: faded.map((r) => [...r]),
      lineStyle: { color: t.muted, opacity: 0.12, width: 1 },
      silent: true,
      z: 1,
    } as SeriesOption)
  const legendNames: string[] = []
  const colourCol = options.colourBy === undefined ? -1 : data.columns.indexOf(options.colourBy)
  if (colourCol >= 0) {
    const scale = colourScale(sequentialRamp('viridis', mode), layout.ranges[colourCol] as [number, number])
    series.push({
      type: 'parallel',
      name: '__rows',
      data: data.rows.flatMap((r, i) =>
        kept.has(i) ? [{ value: [...r], lineStyle: { color: scale(r[colourCol] ?? 0) } }] : [],
      ),
      lineStyle: { opacity, width: lineWidth },
      z: 2,
    } as SeriesOption)
  } else {
    const order = data.groups === undefined ? ['__all'] : groupOrder(data)
    order.forEach((g, k) => {
      const name = g === '__all' ? '__rows' : groupName(data, g)
      if (g !== '__all') legendNames.push(name)
      series.push({
        type: 'parallel',
        name,
        data: data.rows.flatMap((r, i) => (kept.has(i) && (g === '__all' || data.groups?.[i] === g) ? [[...r]] : [])),
        lineStyle: { color: categoricalColour(k), opacity, width: lineWidth },
        z: 2,
      } as SeriesOption)
    })
  }
  const frame = chartFrame(
    { ...fig, mode },
    {
      legendNames,
      ...(options.legend === undefined ? {} : { legend: options.legend }),
      ...(options.title === undefined ? {} : { title: options.title }),
      tooltip: false,
    },
  )
  const graphic = data.columns.flatMap((c, j) => {
    const b = brush[c]
    const r = layout.ranges[j]
    const x = layout.axisX[j]
    if (b === undefined || r === undefined || x === undefined) return []
    const share = (v: number): number => (Math.min(Math.max(v, r[0]), r[1]) - r[0]) / (r[1] - r[0] || 1)
    const yOf = (v: number): number => layout.top + layout.height * (inverted.has(c) ? share(v) : 1 - share(v))
    const ya = yOf(b[0])
    const yb = yOf(b[1])
    const [y0, y1] = [Math.min(ya, yb), Math.max(ya, yb)]
    const shape = { x: x - 8, y: y0, width: 16, height: Math.max(1, y1 - y0) }
    return [
      { type: 'rect', silent: true, z: 10, shape, style: { fill: t.accent, opacity: 0.25 } },
      { type: 'rect', silent: true, z: 11, shape, style: { fill: 'none', stroke: t.accent, lineWidth: 1.5 } },
    ]
  })
  const axisCommon = {
    nameLocation: 'end',
    nameGap: 12,
    nameTextStyle: { color: t.fg, fontSize: fig.fontSize, fontFamily: t.font },
    axisLine: { show: true, lineStyle: { color: t.axis } },
    axisTick: { show: false },
    axisLabel: {
      color: t.muted,
      fontSize: fig.fontSize - 1,
      fontFamily: t.font,
      formatter: (v: number) => formatNumber(v),
    },
    splitLine: { show: false },
    realtime: false,
    areaSelectStyle: { width: 16, borderWidth: 1, borderColor: t.accent, color: t.accent, opacity: 0.22 },
  }
  const option: Record<string, unknown> = {
    ...frame,
    legend: { ...(frame['legend'] as object), top: 4 },
    parallel: {
      left: layout.left,
      top: layout.top,
      width: layout.width,
      height: layout.height,
      parallelAxisDefault: axisCommon,
    },
    parallelAxis: data.columns.map((c, j) => ({
      ...axisCommon,
      id: c,
      dim: j,
      name: c,
      min: layout.ranges[j]?.[0],
      max: layout.ranges[j]?.[1],
      inverse: inverted.has(c),
      // Names stay on top whichever way the axis runs.
      ...(inverted.has(c) ? { nameLocation: 'start' } : {}),
    })),
    series,
    ...(graphic.length > 0 ? { graphic } : {}),
  }
  return option as EChartsOption
}

/** A series of a radar chart: a name and one value per indicator. */
export interface RadarSeries {
  readonly name: string
  readonly values: readonly number[]
  readonly colour?: string
}

/** Options of {@link radarChart}. */
export interface RadarOptions {
  /** The spokes: names with their ranges (default 0 to the nice maximum of the values on that spoke). */
  readonly indicators: readonly (string | { readonly name: string; readonly min?: number; readonly max?: number })[]
  readonly mode?: ColourMode
  readonly figure?: Figure
  readonly title?: string
  /** Fill each polygon (default true). */
  readonly fill?: boolean
  /** The web's shape (default `polygon`). */
  readonly shape?: 'polygon' | 'circle'
  /** A series to bring forward: the others fade. */
  readonly highlight?: string
  readonly legend?: boolean
}

/**
 * A radar (spider) chart: one spoke per indicator, one closed polygon per series, tab10 colours in order.
 *
 * @param series - The series.
 * @param options - Indicators (and their ranges), figure, fill, shape, a highlighted series and the legend.
 * @returns The ECharts option.
 * @throws MdmlError with code `shape` when a series does not have one value per indicator.
 * @example
 * ```ts
 * import { radarChart } from 'mdml-render/charts'
 * const o = radarChart([{ name: 'a', values: [1, 4, 2] }], { indicators: ['x', 'y', 'z'] })
 * o.radar.indicator.map((i) => i.max) // => [1, 4, 2]
 * ```
 */
export function radarChart(series: readonly RadarSeries[], options: RadarOptions): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const k = options.indicators.length
  for (const s of series)
    if (s.values.length !== k) throw new MdmlError('shape', `radarChart: '${s.name}' needs ${k} values`)
  const indicator = options.indicators.map((ind, j) => {
    const spec = typeof ind === 'string' ? { name: ind } : ind
    const top = Math.max(0, ...series.map((s) => s.values[j] ?? 0))
    const max = spec.max ?? (top > 0 ? niceRange([0, top])[1] : 1)
    return { name: spec.name, max, ...(spec.min === undefined ? {} : { min: spec.min }) }
  })
  const fill = options.fill ?? true
  const legendNames = series.map((s) => s.name)
  const frame = chartFrame(
    { ...fig, mode },
    {
      legendNames,
      ...(options.legend === undefined ? {} : { legend: options.legend }),
      ...(options.title === undefined ? {} : { title: options.title }),
      tooltip: 'item',
    },
  )
  const radius = Math.max(20, Math.min(fig.width, fig.height - 56) / 2 - 44)
  const option: Record<string, unknown> = {
    ...frame,
    legend: { ...(frame['legend'] as object), top: 4 },
    radar: {
      indicator,
      shape: options.shape ?? 'polygon',
      center: [fig.width / 2, (fig.height + 24) / 2],
      radius,
      axisName: { color: t.fg, fontSize: fig.fontSize, fontFamily: t.font },
      axisLine: { lineStyle: { color: t.grid } },
      splitLine: { lineStyle: { color: t.grid } },
      splitArea: { show: false },
    },
    series: [
      {
        type: 'radar',
        symbol: 'circle',
        symbolSize: 5,
        data: series.map((s, i) => {
          const colour = s.colour ?? categoricalColour(i)
          const dim = options.highlight !== undefined && options.highlight !== s.name
          return {
            name: s.name,
            value: [...s.values],
            itemStyle: { color: colour, opacity: dim ? 0.25 : 1 },
            lineStyle: { color: colour, width: dim ? 1 : 2, opacity: dim ? 0.3 : 1 },
            ...(fill ? { areaStyle: { color: colour, opacity: dim ? 0.04 : 0.18 } } : {}),
            z: dim ? 1 : 2,
          }
        }),
      },
    ],
  }
  return option as EChartsOption
}

/** Options of {@link scatterMatrix}. */
export interface ScatterMatrixOptions {
  readonly mode?: ColourMode
  readonly figure?: Figure
  readonly title?: string
  /** Rows to highlight (a linked selection): the others are drawn faint and grey. */
  readonly selection?: readonly number[]
  /** What the diagonal shows (default `histogram`, by group, with the column's name). */
  readonly diagonal?: 'histogram' | 'label'
  /** Gap between panels in pixels (default 8). */
  readonly gap?: number
  /** Marker size in pixels (default 5). */
  readonly size?: number
  /** Histogram bins on the diagonal (default 12). */
  readonly bins?: number
}

/** One panel of a scatter matrix: where it sits, which columns it plots, and its axis ranges. */
export interface ScatterMatrixCell {
  readonly id: string
  readonly row: number
  readonly col: number
  /** The column on the x axis (`col`) and on the y axis (`row`). */
  readonly x: string
  readonly y: string
  readonly rect: Rect
  readonly xRange: Range
  readonly yRange: Range
  readonly diagonal: boolean
}

function matrixFigure(options: ScatterMatrixOptions): Figure {
  return (
    options.figure ??
    makeFigure({ mode: options.mode ?? 'light', aspect: 1, width: 520, padding: { top: 28, right: 16 } })
  )
}

function columnRanges(data: Table): Range[] {
  return data.columns.map((_, j) => {
    const e = extentOf(data.rows.map((r) => r[j] ?? Number.NaN)) ?? [0, 1]
    // Padded, so points on the extremes are not cut by the panel's edge.
    return e[0] === e[1] ? [e[0] - 0.5, e[1] + 0.5] : padRange(e, 0.06)
  })
}

/**
 * The panels of a scatter matrix as {@link scatterMatrix} places them in its figure: for overlays such as a brush on
 * any panel (each cell's rectangle and ranges give a linear projection).
 *
 * @param data - The table.
 * @param options - Figure and gap.
 * @returns One cell per pair of columns, row by row.
 * @example
 * ```ts
 * import { scatterMatrixCells } from 'mdml-render/charts'
 * const cells = scatterMatrixCells({ columns: ['a', 'b'], rows: [[0, 1], [2, 3]] })
 * cells.map((c) => [c.x, c.y, c.diagonal]) // => [['a', 'a', true], ['b', 'a', false], ['a', 'b', false], ['b', 'b', true]]
 * ```
 */
export function scatterMatrixCells(data: Table, options: ScatterMatrixOptions = {}): ScatterMatrixCell[] {
  check(data, 'scatterMatrixCells')
  const fig = matrixFigure(options)
  const n = data.columns.length
  const gap = options.gap ?? 8
  const grid = gridSpec(n, n, { wspace: gap, hspace: gap })
  const rects = subplotRects(grid, {}, fig)
  const ranges = columnRanges(data)
  return grid.cells.map((c) => ({
    id: c.id,
    row: c.row,
    col: c.col,
    x: data.columns[c.col] ?? '',
    y: data.columns[c.row] ?? '',
    rect: rects[c.id] as Rect,
    xRange: ranges[c.col] as Range,
    yRange: ranges[c.row] as Range,
    diagonal: c.row === c.col,
  }))
}

/**
 * A scatter matrix (pairs plot): every pair of columns as a scatter panel coloured by group, the diagonal a histogram
 * of each column by group (or just its name), ticks only on the outer panels, and the column names on the diagonal.
 * With `selection`, the selected rows keep their colours in every panel and the rest fade, so a brush in one panel
 * shows the same rows everywhere ({@link scatterMatrixCells} places such a brush).
 *
 * @param data - The table.
 * @param options - Figure, title, selection, diagonal, gap, marker size and bins.
 * @returns The ECharts option, one grid per panel.
 * @throws MdmlError with code `shape` when rows do not match the columns.
 * @example
 * ```ts
 * import { scatterMatrix } from 'mdml-render/charts'
 * const o = scatterMatrix({ columns: ['a', 'b', 'c'], rows: [[0, 1, 2], [2, 3, 1], [1, 1, 1]] })
 * o.grid.length // => 9
 * ```
 */
export function scatterMatrix(data: Table, options: ScatterMatrixOptions = {}): EChartsOption {
  check(data, 'scatterMatrix')
  const fig = matrixFigure(options)
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const n = data.columns.length
  const gap = options.gap ?? 8
  const grid = gridSpec(n, n, { wspace: gap, hspace: gap })
  const ranges = columnRanges(data)
  const order = data.groups === undefined ? ['__all'] : groupOrder(data)
  const selected = options.selection === undefined ? undefined : new Set(options.selection)
  const inGroup = (i: number, g: string | number): boolean => g === '__all' || data.groups?.[i] === g
  const size = options.size ?? 5
  const panels: Record<string, Panel> = {}
  for (const c of grid.cells) {
    const xr = ranges[c.col] as Range
    const yr = ranges[c.row] as Range
    const outer = {
      x: { limits: xr, show: c.row === n - 1, splitLine: false },
      y: { limits: yr, show: c.col === 0 && c.row !== 0, splitLine: false },
    }
    if (c.row === c.col) {
      if (options.diagonal === 'label') {
        panels[c.id] = { layers: [], options: { x: { ...outer.x, limits: xr }, y: { limits: [0, 1], show: false } } }
        continue
      }
      const values = (keep: (i: number) => boolean): number[] =>
        data.rows.flatMap((r, i) => (keep(i) ? [r[c.col] ?? Number.NaN] : []))
      const hist = { bins: options.bins ?? 12, range: xr, mode }
      const layers =
        selected === undefined
          ? order.map((g, k) =>
              histogram(
                values((i) => inGroup(i, g)),
                {
                  ...hist,
                  name: g === '__all' ? '__hist' : groupName(data, g),
                  colour: categoricalColour(k),
                  opacity: order.length > 1 ? 0.45 : 0.7,
                },
              ),
            )
          : [
              histogram(
                values(() => true),
                { ...hist, name: '__all', colour: t.muted, opacity: 0.25 },
              ),
              ...order.flatMap((g, k) => {
                const v = values((i) => selected.has(i) && inGroup(i, g))
                return v.length === 0
                  ? []
                  : [
                      histogram(v, {
                        ...hist,
                        name: g === '__all' ? '__selected' : groupName(data, g),
                        colour: categoricalColour(k),
                        opacity: order.length > 1 ? 0.6 : 0.8,
                      }),
                    ]
              }),
            ]
      // Headroom above the bars for the column's name.
      const y = { show: false, splitLine: false, limits: { kind: 'auto' as const, pad: 0.35, includeZero: true } }
      panels[c.id] = { layers, options: { x: outer.x, y } }
      continue
    }
    const set = (keep: (i: number) => boolean): { x: number[]; y: number[] } => {
      const x: number[] = []
      const y: number[] = []
      data.rows.forEach((r, i) => {
        if (!keep(i)) return
        x.push(r[c.col] ?? Number.NaN)
        y.push(r[c.row] ?? Number.NaN)
      })
      return { x, y }
    }
    const sets = [
      ...(selected === undefined
        ? []
        : [{ ...set((i) => !selected.has(i)), name: '__faded', colour: t.muted, opacity: 0.25, size, outline: false }]),
      ...order.map((g, k) => ({
        ...set((i) => (selected === undefined || selected.has(i)) && inGroup(i, g)),
        name: g === '__all' ? '__points' : groupName(data, g),
        colour: categoricalColour(k),
        size,
        opacity: 0.85,
      })),
    ]
    panels[c.id] = { layers: [points(sets, { mode })], options: outer }
  }
  const option = subplots(grid, panels, {
    figure: { ...fig, mode },
    mode,
    tooltip: false,
    legend: false,
    ...(options.title === undefined ? {} : { title: options.title }),
  }) as Record<string, unknown>
  // Column names inside the diagonal panels, and a legend of the groups at the top.
  const rects = subplotRects(grid, panels, fig)
  const names = data.columns.map((name, j) => {
    const r = rects[`r${j}c${j}`] as Rect
    return {
      type: 'text',
      silent: true,
      z: 10,
      left: r.x + 6,
      top: r.y + 4,
      style: { text: name, fill: t.fg, font: `600 ${fig.fontSize}px ${t.font}` },
    }
  })
  const groups = order.filter((g) => g !== '__all').map((g) => groupName(data, g))
  // A group with no selected rows has no series; an empty one keeps it in the legend.
  const series = option['series'] as Record<string, unknown>[]
  const missing = groups.flatMap((name) =>
    series.some((s) => s['name'] === name)
      ? []
      : [
          {
            type: 'scatter',
            name,
            data: [],
            xAxisIndex: 0,
            yAxisIndex: 0,
            itemStyle: { color: categoricalColour(order.findIndex((g) => groupName(data, g) === name)) },
          },
        ],
  )
  return {
    ...option,
    series: [...series, ...missing],
    graphic: [...((option['graphic'] as unknown[] | undefined) ?? []), ...names],
    ...(groups.length > 1
      ? {
          legend: {
            ...(chartFrame({ ...fig, mode }, { legendNames: groups })['legend'] as object),
            show: true,
            top: 4,
            data: groups,
          },
        }
      : {}),
  } as EChartsOption
}

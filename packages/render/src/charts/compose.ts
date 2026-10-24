import type { EChartsOption, SeriesOption } from 'echarts'
import { WhiteboxError } from 'whitebox-engine/base'
import type { Range } from '../figure/aspect.ts'
import { type Figure, figure as makeFigure, plotArea, type Rect } from '../figure/figure.ts'
import { type CellSpec, type GridSpec, layoutGrid, linkedGroups, sharedGroups } from '../figure/grid.ts'
import { legibleRange, padRange, resolveLimits, unionRange } from '../figure/limits.ts'
import { tokens } from '../theme/tokens.ts'
import { chartFrame, type Layer, logRange, type PlotOptions, panelParts } from './plot.ts'

/** What one axes cell of a subplot grid draws: layers and per-panel axis options. */
export interface Panel {
  readonly layers: readonly Layer[]
  readonly options?: Pick<PlotOptions, 'x' | 'y' | 'y2' | 'aspect' | 'previous'> & { readonly title?: string }
}

/** Options of {@link subplots}. */
export interface SubplotOptions {
  readonly figure?: Figure
  readonly mode?: PlotOptions['mode']
  readonly title?: string
  readonly legend?: boolean
  readonly tooltip?: 'axis' | 'item' | false
  /** Link hover across axes that are shared or linked in the grid (default true). */
  readonly link?: boolean
  /** Wheel and drag zoom, moving linked x axes together (default false). */
  readonly zoom?: boolean
  /** Draw text and TeX cells as chart text (default true); `FigureGrid` overlays them as HTML instead. */
  readonly textCells?: boolean
}

/**
 * Where {@link subplots} places each cell of a grid in a figure, in pixels: the gridspec laid out in the figure's
 * plotting area, less room for a colour bar when a panel has one.
 *
 * @param grid - The grid.
 * @param panels - The panels by cell id.
 * @param fig - The figure.
 * @returns Each cell's rectangle by id.
 * @example
 * ```ts
 * import { subplotRects } from 'whitebox-render/charts'
 * import { figure, gridSpec } from 'whitebox-render/figure'
 * subplotRects(gridSpec(1, 1), {}, figure())['r0c0'] // => { x: 56, y: 28, width: 564, height: 328 }
 * ```
 */
export function subplotRects(
  grid: GridSpec,
  panels: Readonly<Record<string, Panel>>,
  fig: Figure,
): Record<string, Rect> {
  const hasBar = Object.values(panels).some((p) => p.layers.some((l) => l.colourBar !== undefined))
  const area = plotArea(fig)
  return layoutGrid(grid, { ...area, width: area.width - (hasBar ? 64 : 0) })
}

function bottomOf(c: CellSpec): number {
  return c.row + (c.rowSpan ?? 1)
}

/**
 * Lay panels out on a grid specification as one ECharts option: each axes cell gets its own grid and axes placed by
 * the gridspec (spans, ratios, gaps down to zero), shared axes get one range (the union of their data) with inner tick
 * labels hidden, linked axes share hover and zoom, and text cells hold labels. Marginal plots are `withMarginals()`
 * with panels `main`, `top` and `right`.
 *
 * @param grid - The grid.
 * @param panels - Panels by cell id (cells without a panel stay empty).
 * @param options - Figure, title, legend, tooltip, linking and zoom.
 * @returns The ECharts option.
 * @throws WhiteboxError with code `value` when a panel names a cell that is not in the grid.
 * @example
 * ```ts
 * import { histogram, points, subplots } from 'whitebox-render/charts'
 * import { withMarginals } from 'whitebox-render/figure'
 * const x = [0, 1, 2, 3]
 * const y = [1, 3, 2, 4]
 * const o = subplots(withMarginals(), {
 *   main: { layers: [points([{ x, y }])] },
 *   top: { layers: [histogram(x, { bins: 2 })] },
 *   right: { layers: [histogram(y, { bins: 2 })] },
 * })
 * o.grid.length // => 3
 * o.xAxis[1].axisLabel.show // => false
 * ```
 */
export function subplots(
  grid: GridSpec,
  panels: Readonly<Record<string, Panel>>,
  options: SubplotOptions = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const cells = new Map(grid.cells.map((c) => [c.id, c]))
  for (const id of Object.keys(panels))
    if (!cells.has(id)) throw new WhiteboxError('value', `subplots: no cell '${id}' in the grid`)
  const rects = subplotRects(grid, panels, fig)

  // Shared axes: one range from the union of the group's data, under the first panel's limit policy.
  const sharedRange = (axis: 'x' | 'y'): Map<string, Range> => {
    const out = new Map<string, Range>()
    for (const group of sharedGroups(grid, axis)) {
      const members = group.filter((id) => panels[id] !== undefined)
      const ext = unionRange(members.flatMap((id) => (panels[id]?.layers ?? []).map((l) => l.extent[axis])))
      const first = panels[members[0] ?? '']?.options?.[axis]?.limits
      const tight = members.every((id) => (panels[id]?.layers ?? []).every((l) => l.extent.tight === true))
      const policy =
        first !== undefined && Array.isArray(first) && !((first[1] as number) > (first[0] as number))
          ? padRange(first as Range)
          : (first ?? { kind: 'auto' as const, pad: tight ? 0 : 0.04 })
      const log = panels[members[0] ?? '']?.options?.[axis]?.scale === 'log'
      const r = log ? logRange(policy, ext) : legibleRange(resolveLimits(policy, ext))
      for (const id of members) out.set(id, r)
    }
    return out
  }
  const sx = sharedRange('x')
  const sy = sharedRange('y')
  // In a shared group, only the bottom-most cell shows x tick labels and the left-most shows y tick labels.
  const hide = (axis: 'x' | 'y'): Set<string> => {
    const out = new Set<string>()
    for (const group of sharedGroups(grid, axis)) {
      const members = group.map((id) => cells.get(id)).filter((c): c is CellSpec => c !== undefined)
      const keep = axis === 'x' ? Math.max(...members.map(bottomOf)) : Math.min(...members.map((c) => c.col))
      for (const c of members) if ((axis === 'x' ? bottomOf(c) : c.col) !== keep) out.add(c.id)
    }
    return out
  }
  const hideX = hide('x')
  const hideY = hide('y')

  const grids: Record<string, unknown>[] = []
  const xAxes: Record<string, unknown>[] = []
  const yAxes: Record<string, unknown>[] = []
  const series: SeriesOption[] = []
  const visualMap: Record<string, unknown>[] = []
  const titles: Record<string, unknown>[] = []
  const legendNames: string[] = []
  const legends: Record<string, unknown>[] = []
  const axisIndex = new Map<string, { x: number; y: number }>()
  const graphics: Record<string, unknown>[] = []
  for (const cell of grid.cells) {
    const rect = rects[cell.id]
    if (rect === undefined) continue
    if (cell.kind === 'text' || cell.kind === 'tex') {
      if (options.textCells === false) continue
      // TODO: typeset `tex` cells with KaTeX (as an SVG image) rather than showing the TeX source.
      graphics.push({
        type: 'text',
        left: rect.x + rect.width / 2,
        top: rect.y + rect.height / 2,
        style: {
          text: cell.content ?? '',
          fill: t.fg,
          font: `${fig.fontSize + 1}px ${t.font}`,
          align: 'center',
          verticalAlign: 'middle',
        },
      })
      continue
    }
    const panel = panels[cell.id]
    if (panel === undefined || cell.kind === 'empty') continue
    const po = panel.options ?? {}
    const xr = sx.get(cell.id)
    const yr = sy.get(cell.id)
    const parts = panelParts(
      panel.layers,
      {
        mode,
        ...po,
        x: { ...po.x, ...(xr === undefined ? {} : { limits: xr }), ...(hideX.has(cell.id) ? { show: false } : {}) },
        y: { ...po.y, ...(yr === undefined ? {} : { limits: yr }), ...(hideY.has(cell.id) ? { show: false } : {}) },
      },
      rect,
      grids.length,
      yAxes.length,
      series.length,
    )
    axisIndex.set(cell.id, { x: grids.length, y: yAxes.length })
    grids.push({ ...parts.grid })
    // Panels that touch (gaps under 12 px) would print their edge tick labels on top of each other: drop one.
    const touchingAbove = grid.hspace < 12 && cell.row > 0
    const touchingRight = grid.wspace < 12 && cell.col + (cell.colSpan ?? 1) < grid.cols
    const dropMax = (axis: Readonly<Record<string, unknown>>, drop: boolean): Record<string, unknown> =>
      drop ? { ...axis, axisLabel: { ...(axis['axisLabel'] as object), showMaxLabel: false } } : { ...axis }
    xAxes.push(dropMax(parts.xAxis, touchingRight))
    yAxes.push(...parts.yAxis.map((a, i) => dropMax(a, i === 0 && touchingAbove)))
    const names = parts.legendNames
    if (names.length > 0 && (options.legend ?? names.length > 1))
      legends.push({
        show: true,
        data: [...names],
        // Above the panel, in its title's row, where there is room; else inside its top right corner.
        top: (cell.row === 0 ? fig.padding.top : grid.hspace) >= 20 ? Math.max(0, rect.y - 18) : rect.y + 2,
        right: fig.width - rect.x - rect.width + 4,
      })
    series.push(...parts.series)
    visualMap.push(...parts.visualMap)
    for (const n of parts.legendNames) if (!legendNames.includes(n)) legendNames.push(n)
    if (po.title !== undefined)
      titles.push({
        text: po.title,
        left: rect.x,
        top: Math.max(0, rect.y - fig.fontSize - 8),
        textStyle: { fontSize: fig.fontSize + 1, fontWeight: 600, color: t.fg },
      })
  }
  const frame = chartFrame(
    { ...fig, mode },
    {
      legendNames,
      ...(options.legend === undefined ? {} : { legend: options.legend }),
      ...(options.tooltip === undefined ? {} : { tooltip: options.tooltip }),
      ...(options.title === undefined ? {} : { title: options.title }),
    },
  )
  const linkGroups = (axis: 'x' | 'y'): number[][] =>
    linkedGroups(grid, axis)
      .map((g) => g.map((id) => axisIndex.get(id)?.[axis]).filter((i): i is number => i !== undefined))
      .filter((g) => g.length > 1)
  const lx = linkGroups('x')
  const ly = linkGroups('y')
  const frameTitle = frame['title']
  const legendStyle = frame['legend'] as Record<string, unknown>
  const option: Record<string, unknown> = {
    ...frame,
    // One legend per panel, at the panel's top right (above it where there is room), never over another panel.
    legend: legends.length > 0 ? legends.map((l) => ({ ...legendStyle, ...l })) : { ...legendStyle, show: false },
    ...(titles.length > 0 || frameTitle !== undefined
      ? { title: [...(frameTitle === undefined ? [] : [frameTitle]), ...titles] }
      : {}),
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series,
    ...(visualMap.length > 0 ? { visualMap } : {}),
    ...(graphics.length > 0 ? { graphic: graphics } : {}),
    ...((options.link ?? true) && (lx.length > 0 || ly.length > 0)
      ? {
          axisPointer: {
            link: [...lx.map((g) => ({ xAxisIndex: g })), ...ly.map((g) => ({ yAxisIndex: g }))],
          },
        }
      : {}),
    ...(options.zoom === true
      ? {
          dataZoom: [
            ...(lx.length > 0 ? lx : xAxes.map((_, i) => [i])).map((g) => ({
              type: 'inside',
              xAxisIndex: g,
              filterMode: 'none',
            })),
          ],
        }
      : {}),
  }
  return option as EChartsOption
}

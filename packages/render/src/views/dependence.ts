/**
 * Views of the dependence record: a feature's partial dependence with its ICE curves (raw, centred or derivative) or
 * its accumulated local effects, over the data's rug; small multiples of several features; and a pair's joint effect
 * as a heatmap.
 */
import type { EChartsOption } from 'echarts'
import { maxOf } from 'mdml-engine/base'
import type { Dependence, DependenceCurve, DependenceSurface } from 'mdml-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { rug } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { tokens } from '../theme/tokens.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link dependenceView}. */
export interface DependenceViewOptions extends ViewOptions {
  /** Which curve (default: the first; every curve as small multiples when `all`). */
  readonly curve?: number
  /** Show every curve in small multiples (default when the record has several and no `curve` is given). */
  readonly all?: boolean
  /** ICE curves drawn at most (default 60, spread evenly over the rows). */
  readonly maxIce?: number
}

function yLabel(record: Dependence): string {
  if (record.method === 'ale') return `ALE of ${record.output}`
  if (record.method === 'shape function') return `term of ${record.output}`
  if (record.transform === 'centred') return `centred ${record.output}`
  if (record.transform === 'derivative') return `∂ ${record.output}`
  return `partial dependence of ${record.output}`
}

/**
 * The rows an ICE plot draws: at most `max`, evenly spread.
 *
 * @param n - The number of rows there are.
 * @param max - The most to draw.
 * @returns The indices of the rows to draw.
 */
export function iceRows(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i)
  return Array.from({ length: max }, (_, i) => Math.round((i * (n - 1)) / Math.max(1, max - 1)))
}

function curveLayers(record: Dependence, c: DependenceCurve, mode: ColourMode, maxIce: number): Layer[] {
  const t = tokens(mode)
  const layers: Layer[] = []
  if (c.individual !== undefined && c.individual.length > 0) {
    const shown = iceRows(c.individual.length, maxIce)
    layers.push(
      lines(
        shown.map((i) => ({ x: c.grid, y: c.individual?.[i] ?? [], colour: t.muted, width: 1, opacity: 0.35 })),
        { mode },
      ),
    )
  }
  const ale = record.method === 'ale'
  const shape = record.method === 'shape function'
  // A shape function holds each bin's value from its left end: drawn as steps, the last bin carried to the data's end.
  const end = c.rug === undefined || c.rug.length === 0 ? undefined : maxOf(c.rug)
  const stepX = shape && end !== undefined && end > (c.grid[c.grid.length - 1] ?? end) ? [...c.grid, end] : c.grid
  const stepY = stepX.length > c.average.length ? [...c.average, c.average[c.average.length - 1] ?? 0] : c.average
  layers.push(
    lines(
      [
        {
          x: shape ? stepX : c.grid,
          y: shape ? stepY : c.average,
          name: ale ? 'ALE' : shape ? 'shape function' : 'average',
          colour: 'accent',
          width: 3,
          markers: c.categorical === true || ale,
          ...(shape ? { step: 'end' as const } : {}),
        },
      ],
      { mode },
    ),
  )
  if (ale || shape || record.transform === 'centred' || record.transform === 'derivative')
    layers.push(referenceLines({ y: [0] }, { mode, colour: t.muted, dash: 'dashed' }))
  if (c.rug !== undefined && c.rug.length > 0) layers.push(rug(c.rug, { mode, opacity: 0.45 }))
  return layers
}

function curvePanel(record: Dependence, index: number, mode: ColourMode, maxIce: number, title?: string): Panel {
  const c = record.curves[index] as DependenceCurve
  return {
    layers: curveLayers(record, c, mode, maxIce),
    options: {
      x: { label: record.features[c.feature] ?? `x${c.feature}` },
      y: { label: yLabel(record) },
      ...(title === undefined ? {} : { title }),
    },
  }
}

/**
 * A feature's dependence (sklearn's PartialDependenceDisplay, ALEPlot): the averaged effect as a thick line over the
 * feature's grid, every row's ICE curve as a thin grey line beneath it (centred or differentiated when the record
 * says), and the data's values as a rug, so the reader sees where the curve rests on data. ALE curves are drawn with
 * a marker at each bin edge and a zero line. With several curves and no `curve`, small multiples, one per feature.
 *
 * @param record - The record.
 * @param options - curve, all, maxIce, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { dependence } from 'mdml-engine/exposition'
 * import { dependenceView } from 'mdml-render/views'
 * const r = dependence({ method: 'partial dependence', features: ['a'], output: 'y', curves: [{ feature: 0, grid: [0, 1], average: [0, 1], individual: [[0, 2], [0, 0]] }] })
 * dependenceView(r).series.length // => 2
 * ```
 */
export function dependenceView(record: Dependence, options: DependenceViewOptions = {}): EChartsOption {
  const maxIce = options.maxIce ?? 60
  const n = record.curves.length
  if (n === 0) {
    const s = record.surfaces?.[0]
    if (s !== undefined) return dependenceSurfaceView(record, options)
  }
  const all = options.all ?? (options.curve === undefined && n > 1)
  if (!all || n <= 1) {
    const i = Math.min(Math.max(0, options.curve ?? 0), Math.max(0, n - 1))
    const fig = viewFigure(options, { aspect: 1.5 })
    const p = curvePanel(record, i, fig.mode, maxIce)
    return cartesian(p.layers, { ...p.options, figure: fig, mode: fig.mode, legend: false, tooltip: 'item' })
  }
  const k = Math.min(n, 9)
  const cols = Math.min(k, 3)
  const rows = Math.ceil(k / cols)
  const fig = viewFigure(options, { width: 720, height: Math.max(240, 40 + rows * 210) })
  const panels: Record<string, Panel> = {}
  for (let i = 0; i < k; i++) {
    const c = record.curves[i] as DependenceCurve
    const p = curvePanel(record, i, fig.mode, Math.min(maxIce, 30))
    panels[`c${i}`] = {
      layers: p.layers,
      options: {
        x: { label: record.features[c.feature] ?? `x${c.feature}` },
        ...(i % cols === 0 ? { y: { label: yLabel(record) } } : {}),
      },
    }
  }
  const grid = gridSpec(rows, cols, {
    wspace: 56,
    hspace: 64,
    cells: Array.from({ length: k }, (_, i) => ({ id: `c${i}`, row: Math.floor(i / cols), col: i % cols })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

/** Options of {@link dependenceSurfaceView}. */
export interface DependenceSurfaceViewOptions extends ViewOptions {
  /** Which surface (default 0). */
  readonly surface?: number
}

/**
 * A pair's joint effect as a heatmap over the two features' grids: the two-way partial dependence on a sequential
 * ramp, or the second-order ALE (a pure interaction, centred on zero) on a diverging one.
 *
 * @param record - The record (with surfaces).
 * @param options - surface, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { dependence } from 'mdml-engine/exposition'
 * import { dependenceSurfaceView } from 'mdml-render/views'
 * const r = dependence({ method: 'ale', features: ['a', 'b'], output: 'y', curves: [], surfaces: [{ features: [0, 1], gridX: [0, 1], gridY: [0, 1], values: [[0.5, -0.5], [-0.5, 0.5]] }] })
 * Array.isArray(dependenceSurfaceView(r).series) // => true
 * ```
 */
export function dependenceSurfaceView(record: Dependence, options: DependenceSurfaceViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.25, padding: { right: 96 } })
  const s = record.surfaces?.[options.surface ?? 0] as DependenceSurface | undefined
  if (s === undefined) return cartesian([], { figure: fig, mode: fig.mode })
  // heatmap rows run along y: z[j][i] = values[i][j].
  const z = s.gridY.map((_, j) => s.gridX.map((_, i) => s.values[i]?.[j] ?? 0))
  const ale = record.method === 'ale'
  const shape = record.method === 'shape function'
  const label = ale
    ? `interaction (ALE) in ${record.output}`
    : shape
      ? `pairwise term of ${record.output}`
      : `partial dependence of ${record.output}`
  return cartesian(
    [
      heatmap(z, {
        x: s.gridX,
        y: s.gridY,
        smooth: shape ? 'cells' : 'bilinear',
        mode: fig.mode,
        label,
        ...(ale || shape ? { centre: 0 } : {}),
      }),
    ],
    {
      figure: fig,
      mode: fig.mode,
      x: { label: record.features[s.features[0]] ?? `x${s.features[0]}` },
      y: { label: record.features[s.features[1]] ?? `x${s.features[1]}` },
      legend: false,
    },
  )
}

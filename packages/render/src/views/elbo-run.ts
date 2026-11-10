/**
 * The views of an `ElboRun` record, the charts of a variational fit: the bound over the steps (raw, smoothed, and
 * Pathfinder's paths, with its Pareto k-hat read in the title), the norm and variance of the gradient on log axes, the
 * guide's parameters over the run (locations, scales), the posterior beliefs as a mean with its sd band per quantity, and
 * the final posterior over the truth's marginal. Each reads what the record holds (`hasElboChart`), and draws a run in
 * progress as far as it has got.
 */

import type { EChartsOption } from 'echarts'
import { OmaiError, quantileOf } from 'omai-engine/base'
import { normal } from 'omai-engine/dists'
import type { ElboRun, GuideParameter, RunSeries } from 'omai-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { density } from '../charts/distributions.ts'
import { band, lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import { gridSpec } from '../figure/grid.ts'
import { ariaOf, panelColumns, panelGrid, short, type ViewOptions, viewFigure, wantsLog } from './_view.ts'

/** The charts of a variational run, by name. */
export type ElboChart = 'elbo' | 'gradient' | 'parameters' | 'beliefs' | 'posterior'

/**
 * Whether the record holds what a chart draws: the bound, a gradient series, parameters, beliefs, or the final posterior
 * over the truth.
 *
 * @param record - The run.
 * @param chart - The chart.
 * @returns Whether the chart can be drawn.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { hasElboChart } from 'omai-render/views'
 * const r = expositionSamples.elboRun()
 * [hasElboChart(r, 'elbo'), hasElboChart(r, 'gradient'), hasElboChart(r, 'beliefs')] // => [true, false, true]
 * ```
 */
export function hasElboChart(record: ElboRun, chart: ElboChart): boolean {
  switch (chart) {
    case 'elbo':
      return record.elbo.steps.length > 0
    case 'gradient':
      return (record.gradNorm?.steps.length ?? 0) > 0 || (record.gradVariance?.steps.length ?? 0) > 0
    case 'parameters':
      return record.parameters.length > 0
    case 'beliefs':
      return record.beliefs.some((b) => b.steps.length > 0)
    case 'posterior':
      return (record.final?.quantities.length ?? 0) > 0
  }
}

/** Options of {@link elboRunView}. */
export interface ElboRunViewOptions extends ViewOptions {
  /** The chart (default `elbo`, else the first the record can draw). */
  readonly chart?: ElboChart
  /** The most quantities drawn in the belief and posterior charts (default 6). */
  readonly max?: number
}

const pairs = (s: RunSeries): { x: number[]; y: number[] } => ({ x: [...s.steps], y: [...s.values] })

/** How the Pareto k-hat of Pathfinder's importance weights reads (Vehtari et al. 2024): under 0.7 usable, over it not. */
function khatReading(k: number): string {
  return k < 0.5 ? 'good' : k < 0.7 ? 'usable' : 'too high: the approximation is unreliable'
}

/** The limits of a bound's axis: the span of its values, less a start so poor it would flatten the rest. */
function boundLimits(values: readonly number[]): [number, number] | undefined {
  if (values.length < 10) return undefined
  const lo = quantileOf(values, 0.05)
  const hi = Math.max(...values)
  const median = quantileOf(values, 0.5)
  // Only when the start is far below the rest: more than the whole span of the rest again.
  if (Math.min(...values) >= lo - 2 * Math.max(hi - lo, 1e-9) || median <= lo) return undefined
  const pad = 0.12 * (hi - lo)
  return [lo - pad, hi + pad]
}

function elboPanel(record: ElboRun, mode: ReturnType<typeof viewFigure>['mode']): Panel {
  const layers: Layer[] = []
  const hasSmooth = record.smoothed !== undefined && record.smoothed.steps.length > 1
  layers.push(
    lines(
      [
        {
          ...pairs(record.elbo),
          name: record.method === 'pathfinder' ? 'best ELBO' : 'ELBO',
          colour: 'tab0',
          width: hasSmooth ? 1 : 2,
          opacity: hasSmooth ? 0.4 : 1,
        },
      ],
      { mode },
    ),
  )
  if (record.smoothed !== undefined && hasSmooth)
    layers.push(
      lines(
        [
          {
            ...pairs(record.smoothed),
            name: record.smoothing === undefined ? 'smoothed' : `mean of ${record.smoothing}`,
            colour: 'tab0',
            width: 2.4,
          },
        ],
        {
          mode,
        },
      ),
    )
  if (record.paths.length > 0)
    layers.push(
      lines(
        record.paths.map((p, i) => ({
          ...pairs(p),
          name: p.name,
          colour: `tab${(i + 1) % 10}`,
          width: 1,
          opacity: 0.8,
        })),
        { mode },
      ),
    )
  const limits = boundLimits(record.elbo.values)
  // The k-hat goes under the axis, not in the title: a long title would meet the legend.
  const k = record.paretoK === undefined ? '' : `   Pareto k̂ ${short(record.paretoK)} (${khatReading(record.paretoK)})`
  const last = record.elbo.values.at(-1)
  return {
    layers,
    options: {
      title: `${record.method === 'pathfinder' ? 'ELBO of the best normal' : 'ELBO'}${last === undefined ? '' : `   ${short(last)}`}${record.status === 'running' ? '   running' : ''}`,
      x: { label: `${record.method === 'pathfinder' ? 'L-BFGS iteration' : 'step'}${k}` },
      y: { label: 'evidence lower bound', ...(limits === undefined ? {} : { limits }) },
    },
  }
}

function gradientPanels(record: ElboRun, mode: ReturnType<typeof viewFigure>['mode']): Record<string, Panel> {
  const out: Record<string, Panel> = {}
  const add = (id: string, s: RunSeries, name: string, colour: string, label: string, last: boolean): void => {
    const log = wantsLog(s.values)
    out[id] = {
      layers: [lines([{ ...pairs(s), name, colour, width: 1.4 }], { mode })],
      options: {
        title: name,
        y: { label, ...(log ? { scale: 'log' as const } : {}) },
        ...(last ? { x: { label: 'step' } } : {}),
      },
    }
  }
  const both = (record.gradNorm?.steps.length ?? 0) > 0 && (record.gradVariance?.steps.length ?? 0) > 0
  if (record.gradNorm !== undefined && record.gradNorm.steps.length > 0)
    add('q0', record.gradNorm, 'gradient norm', 'tab1', 'norm', !both)
  if (record.gradVariance !== undefined && record.gradVariance.steps.length > 0)
    add(`q${Object.keys(out).length}`, record.gradVariance, 'gradient variance', 'tab2', 'total variance', true)
  return out
}

function parameterPanels(record: ElboRun, mode: ReturnType<typeof viewFigure>['mode']): Record<string, Panel> {
  const groups: [string, string, GuideParameter[]][] = [
    ['q', 'location of the guide', record.parameters.filter((p) => p.role === 'loc')],
    ['q', 'scale of the guide (unconstrained)', record.parameters.filter((p) => p.role === 'scale')],
    ['q', 'other parameters', record.parameters.filter((p) => p.role === 'other')],
  ]
  const out: Record<string, Panel> = {}
  const present = groups.filter(([, , ps]) => ps.length > 0)
  for (const [k, [, title, ps]] of present.entries()) {
    out[`q${k}`] = {
      layers: [
        lines(
          ps.map((p, i) => ({ ...pairs(p), name: p.name, colour: `tab${i % 10}`, width: 1.4 })),
          { mode },
        ),
      ],
      // Only the last panel is labelled: the label of a upper one would sit on the title of the next.
      options: { title, ...(k === present.length - 1 ? { x: { label: 'step' } } : {}) },
    }
  }
  return out
}

function beliefPanels(
  record: ElboRun,
  mode: ReturnType<typeof viewFigure>['mode'],
  max: number,
): Record<string, Panel> {
  const out: Record<string, Panel> = {}
  for (const [i, b] of record.beliefs
    .filter((x) => x.steps.length > 0)
    .slice(0, max)
    .entries()) {
    const lower = b.values.map((m, j) => m - (b.sd[j] ?? 0))
    const upper = b.values.map((m, j) => m + (b.sd[j] ?? 0))
    out[`q${i}`] = {
      layers: [
        band([{ x: [...b.steps], lower, upper, name: '__sd', colour: 'tab0', opacity: 0.22 }], { mode }),
        lines(
          [
            {
              x: [...b.steps],
              y: [...b.values],
              name: b.name,
              colour: 'tab0',
              width: 1.8,
              markers: b.steps.length < 12,
            },
          ],
          { mode },
        ),
      ],
      options: {
        title: `${b.name}   ${short(b.values.at(-1) ?? 0)} ± ${short(b.sd.at(-1) ?? 0)}`,
        x: { label: 'step' },
      },
    }
  }
  return out
}

function posteriorPanels(
  record: ElboRun,
  mode: ReturnType<typeof viewFigure>['mode'],
  max: number,
): Record<string, Panel> {
  const out: Record<string, Panel> = {}
  const final = record.final
  if (final === undefined) return out
  for (const [i, q] of final.quantities.slice(0, max).entries()) {
    const layers: Layer[] = []
    if (q.truth !== undefined) {
      const t = q.truth
      const slope = t.x.map((_, j) => {
        const a = Math.max(0, j - 1)
        const b = Math.min(t.x.length - 1, j + 1)
        return ((t.cdf[b] ?? 0) - (t.cdf[a] ?? 0)) / Math.max(1e-300, (t.x[b] ?? 0) - (t.x[a] ?? 0))
      })
      layers.push(
        lines([{ x: [...t.x], y: slope, name: final.oracle, colour: 'fg', width: 2.2, area: 0.12 }], { mode }),
      )
    }
    const range: [number, number] =
      q.truth === undefined ? [-3, 3] : [q.truth.x[0] as number, q.truth.x.at(-1) as number]
    for (const m of q.methods)
      if (m.normal !== undefined)
        layers.push(
          density(normal(m.normal.mean, Math.max(m.normal.sd, 1e-12)), {
            mode,
            name: final.method,
            colour: 'tab0',
            range,
            n: 160,
          }),
        )
    out[`q${i}`] = { layers, options: { title: q.name, x: { label: q.name } } }
  }
  return out
}

/**
 * The charts of a variational run. `elbo`: the bound (raw faint, smoothed bold; Pathfinder's paths and the Pareto k-hat
 * in the title); `gradient`: the norm and the variance of the gradient on log axes; `parameters`: the guide's locations
 * and scales over the steps; `beliefs`: the variational posterior's mean with a band of one sd per quantity at the steps a
 * snapshot was taken, the approximation sharpening; `posterior`: the final belief as a bell over the truth's marginal.
 * A run in progress draws what it has so far.
 *
 * @param record - The run.
 * @param options - The chart, mode, figure and the most quantities drawn.
 * @returns The ECharts option.
 * @throws OmaiError `value` when the record has no data for the chart (see {@link hasElboChart}).
 * @index function ELBO, gradient, guide parameters and beliefs of a variational run: `elboRunView(record)`.
 * @example
 * ```ts
 * import { expositionSamples } from 'omai-engine/exposition'
 * import { elboRunView } from 'omai-render/views'
 * elboRunView(expositionSamples.elboRun()).grid.length // => 1
 * elboRunView(expositionSamples.elboRun(), { chart: 'beliefs' }).grid.length // => 1
 * ```
 */
export function elboRunView(record: ElboRun, options: ElboRunViewOptions = {}): EChartsOption {
  const order: ElboChart[] = ['elbo', 'gradient', 'parameters', 'beliefs', 'posterior']
  const chart = options.chart ?? order.find((c) => hasElboChart(record, c)) ?? 'elbo'
  if (!hasElboChart(record, chart))
    throw new OmaiError('value', `elboRunView: the record has no data for the ${chart} chart`)
  const fig = viewFigure(options, { aspect: chart === 'elbo' ? 1.8 : 1.5 })
  const mode = fig.mode
  const max = options.max ?? 6
  let panels: Record<string, Panel>
  switch (chart) {
    case 'elbo':
      panels = { q0: elboPanel(record, mode) }
      break
    case 'gradient':
      panels = gradientPanels(record, mode)
      break
    case 'parameters':
      panels = parameterPanels(record, mode)
      break
    case 'beliefs':
      panels = beliefPanels(record, mode, max)
      break
    case 'posterior':
      panels = posteriorPanels(record, mode, max)
      break
  }
  const n = Object.keys(panels).length
  const cols = chart === 'elbo' || chart === 'gradient' || chart === 'parameters' ? 1 : panelColumns(n)
  const grid = n === 1 ? gridSpec(1, 1, { cells: [{ id: 'q0', row: 0, col: 0 }] }) : panelGrid(n, cols)
  const label = (c: ElboChart): string =>
    ({
      elbo: 'The evidence lower bound over the steps',
      gradient: 'The norm and variance of the gradient over the steps',
      parameters: 'The parameters of the guide over the steps',
      beliefs: 'The variational posterior mean and standard deviation over the steps',
      posterior: 'The final variational posterior against the truth',
    })[c]
  return {
    ...subplots(grid, panels, { figure: fig, legend: chart !== 'gradient', tooltip: 'item' }),
    ...ariaOf(label(chart)),
  }
}

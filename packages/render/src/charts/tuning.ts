/**
 * Figures of a hyperparameter study, after Optuna's visualisations: the optimisation history with the running best,
 * parameter importances, slice plots of the objective against each parameter, a contour plot of a Gaussian-process
 * surrogate over two parameters, and the intermediate values of every trial with pruned trials greyed.
 */

import type { EChartsOption } from 'echarts'
import { maxOf } from 'omai-engine/base'
import {
  contourData,
  intermediateValues,
  optimisationHistory,
  type ParamImportances,
  type Study,
  sliceData,
  parallelCoordinates as studyAxes,
} from 'omai-engine/tune'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { extentOf } from '../figure/limits.ts'
import { formatNumber, formatPercent } from '../theme/format.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { bars } from './bars.ts'
import { type Panel, subplots } from './compose.ts'
import { contours } from './fields.ts'
import { lines } from './lines.ts'
import { heatmap } from './matrices.ts'
import { type AxisSpec, cartesian, type Layer } from './plot.ts'
import { points } from './points.ts'

/** Options shared by the study charts. */
export interface StudyChartOptions {
  readonly mode?: ColourMode
  readonly figure?: Figure
  readonly title?: string
  /** Which objective of a multi-objective study (default 0). */
  readonly objective?: number
  /** The objective's name on the axes (default the study's metric name, else `objective`). */
  readonly label?: string
  /** A log-scaled objective axis (default false). */
  readonly log?: boolean
}

const objectiveLabel = (study: Study, options: StudyChartOptions): string =>
  options.label ?? study.metricNames?.[options.objective ?? 0] ?? 'objective'

/**
 * The optimisation history of a study as layers: each complete trial's value as a point, and the best value so far as a
 * step line (Optuna's `plot_optimization_history`).
 *
 * @param study - The study.
 * @param options - Objective and colour mode.
 * @returns The layers: the trials, then the running best.
 * @example
 * ```ts
 * import { createStudy } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * import { historyLayers } from 'omai-render/charts'
 * const trial = (number: number, value: number) => ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * historyLayers(createStudy(real(0, 9), { trials: [trial(0, 3), trial(1, 5), trial(2, 1)] })).length // => 2
 * ```
 */
export function historyLayers(study: Study, options: Pick<StudyChartOptions, 'objective' | 'mode'> = {}): Layer[] {
  const mode = options.mode ?? 'light'
  const h = optimisationHistory(study, options.objective)
  const done = h.number.flatMap((n, i) => (Number.isNaN(h.value[i] ?? Number.NaN) ? [] : [[n, h.value[i] as number]]))
  const best = h.number.flatMap((n, i) => (Number.isNaN(h.best[i] ?? Number.NaN) ? [] : [[n, h.best[i] as number]]))
  return [
    points([{ x: done.map((p) => p[0] ?? 0), y: done.map((p) => p[1] ?? 0), name: 'trial', size: 6, opacity: 0.75 }], {
      mode,
    }),
    lines([{ x: best.map((p) => p[0] ?? 0), y: best.map((p) => p[1] ?? 0), name: 'best so far', step: 'end' }], {
      mode,
      colourOffset: 3,
    }),
  ]
}

/**
 * A study's optimisation history: trial values against trial number with the running best (see
 * {@link historyLayers}).
 *
 * @param study - The study.
 * @param options - Figure, title, objective, label and log axis.
 * @returns The ECharts option.
 * @index function Objective per trial of a hyperparameter study: `historyChart(study)`.
 * @example
 * ```ts
 * import { createStudy } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * import { historyChart } from 'omai-render/charts'
 * const trial = (number: number, value: number) => ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * historyChart(createStudy(real(0, 9), { trials: [trial(0, 3), trial(1, 1)] })).series.length // => 2
 * ```
 */
export function historyChart(study: Study, options: StudyChartOptions = {}): EChartsOption {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  return cartesian(historyLayers(study, { ...options, mode }), {
    mode,
    ...(options.figure === undefined ? {} : { figure: options.figure }),
    ...(options.title === undefined ? {} : { title: options.title }),
    x: { label: 'trial' },
    y: { label: objectiveLabel(study, options), ...(options.log === true ? { scale: 'log' as const } : {}) },
    legend: true,
  })
}

/**
 * Parameter importances as horizontal bars, the most important at the top, each labelled with its share (Optuna's
 * `plot_param_importances`; the importances come from `correlationImportance` or `permutationImportance` in
 * omai-engine's `tune`).
 *
 * @param importances - Importance by parameter.
 * @param options - Figure, title, colour mode and how many parameters to show (default all).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { importanceChart } from 'omai-render/charts'
 * importanceChart({ lr: 0.7, depth: 0.2, dropout: 0.1 }).yAxis[0].data // => ['dropout', 'depth', 'lr']
 * ```
 */
export function importanceChart(
  importances: ParamImportances,
  options: Pick<StudyChartOptions, 'mode' | 'figure' | 'title'> & { readonly top?: number } = {},
): EChartsOption {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const ranked = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, options.top ?? Number.POSITIVE_INFINITY)
    .reverse()
  const layer = bars(
    ranked.map(([name]) => name),
    [{ name: 'importance', values: ranked.map(([, v]) => v) }],
    { horizontal: true, mode },
  )
  const option = cartesian([layer], {
    mode,
    ...(options.figure === undefined ? {} : { figure: options.figure }),
    ...(options.title === undefined ? {} : { title: options.title }),
    x: { label: 'importance', limits: [0, Math.max(0.05, maxOf(ranked.map(([, v]) => v))) * 1.18], format: 'percent' },
    legend: false,
  }) as Record<string, unknown>
  // Each bar's share written at its end.
  const t = tokens(mode)
  const series = (option['series'] as Record<string, unknown>[]).map((s) => ({
    ...s,
    label: {
      show: true,
      position: 'right',
      color: t.fgSecondary,
      formatter: (p: { value: unknown }) => {
        const v = Array.isArray(p.value) ? p.value[0] : p.value
        return typeof v === 'number' ? formatPercent(v, 0) : ''
      },
    },
  }))
  return { ...option, series } as EChartsOption
}

/** The axis of a parameter: log for log-scaled parameters, option names for choices. */
function parameterAxis(study: Study, path: string): AxisSpec {
  const axis = studyAxes(study).axes.find((a) => a.path === path)
  if (axis?.labels !== undefined) {
    const labels = axis.labels
    return {
      label: path,
      limits: [-0.5, labels.length - 0.5],
      format: (v) => (Number.isInteger(v) ? (labels[v] ?? '') : ''),
    }
  }
  return { label: path, ...(axis?.log === true ? { scale: 'log' as const, limits: axis.range } : {}) }
}

/**
 * Slice plots of a study (Optuna's `plot_slice`): for each parameter, the complete trials' objective values against the
 * parameter's value, coloured by trial number (later trials yellower), the objective axis shared across the panels.
 * Log-scaled parameters get log axes; choices show their option names.
 *
 * @param study - The study.
 * @param options - Figure, title, objective, label, log axis and which parameters (default all).
 * @returns The ECharts option, one panel per parameter.
 * @example
 * ```ts
 * import { createStudy } from 'omai-engine/tune'
 * import { dict, real } from 'omai-engine/space'
 * import { sliceChart } from 'omai-render/charts'
 * const t = (number: number, a: number, b: number) => ({ number, params: { a, b }, state: 'complete', values: [a + b], intermediate: [], source: 'x' }) as const
 * sliceChart(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials: [t(0, 0.1, 0.5), t(1, 0.7, 0.2)] })).grid.length // => 2
 * ```
 */
export function sliceChart(
  study: Study,
  options: StudyChartOptions & { readonly params?: readonly string[] } = {},
): EChartsOption {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const fig = options.figure ?? makeFigure({ mode, aspect: 2.6 })
  const slices = sliceData(study, options.objective).filter(
    (s) => options.params === undefined || options.params.includes(s.path),
  )
  const n = Math.max(1, slices.length)
  const ids = slices.map((_, i) => `r0c${i}`)
  const grid = gridSpec(1, n, {
    wspace: 28,
    cells: ids.map((id, i) => ({ id, row: 0, col: i, ...(i === 0 ? {} : { sharey: ids[0] as string }) })),
  })
  const allY = slices.flatMap((s) => s.y)
  const yLimits = extentOf(allY)
  const panels: Record<string, Panel> = {}
  slices.forEach((s, i) => {
    panels[ids[i] as string] = {
      layers: [points([{ x: s.x, y: s.y, values: s.numbers, name: '__trials', size: 6, opacity: 0.85 }], { mode })],
      options: {
        x: { ...parameterAxis(study, s.path), splitLine: false },
        y: {
          ...(i === 0 ? { label: objectiveLabel(study, options) } : {}),
          ...(options.log === true ? { scale: 'log' as const } : {}),
          ...(yLimits === undefined || options.log === true ? {} : { limits: { kind: 'auto' as const, pad: 0.06 } }),
        },
      },
    }
  })
  return subplots(grid, panels, {
    figure: { ...fig, mode },
    mode,
    legend: false,
    tooltip: false,
    ...(options.title === undefined ? {} : { title: options.title }),
  })
}

/**
 * A contour plot of a study over two parameters (Optuna's `plot_contour`): the objective predicted by a
 * Gaussian-process surrogate (`contourData` in omai-engine's `tune`) as a smooth field with contour lines, and the
 * trials on top.
 * Log-scaled parameters are drawn on log-10 axes (ticks show the parameter's own values).
 *
 * @param study - The study.
 * @param xPath - The parameter across.
 * @param yPath - The parameter up.
 * @param options - Figure, title, objective, label, the surrogate's grid size (default 24) and contour levels (default 8).
 * @returns The ECharts option.
 * @throws OmaiError with code `value` for unknown parameters or fewer than two trials with both.
 * @example
 * ```ts
 * import { createStudy } from 'omai-engine/tune'
 * import { dict, real } from 'omai-engine/space'
 * import { contourChart } from 'omai-render/charts'
 * const trials = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]].map(([a = 0, b = 0], number) => ({ number, params: { a, b }, state: 'complete', values: [a + b], intermediate: [], source: 'x' }) as const)
 * contourChart(createStudy(dict({ a: real(0, 1), b: real(0, 1) }), { trials }), 'a', 'b', { gridSize: 5 }).series.at(-1).type // => 'scatter'
 * ```
 */
export function contourChart(
  study: Study,
  xPath: string,
  yPath: string,
  options: StudyChartOptions & { readonly gridSize?: number; readonly levels?: number } = {},
): EChartsOption {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const c = contourData(study, xPath, yPath, {
    gridSize: options.gridSize ?? 24,
    ...(options.objective === undefined ? {} : { objective: options.objective }),
  })
  const tx = (v: number): number => (c.x.log ? Math.log10(v) : v)
  const ty = (v: number): number => (c.y.log ? Math.log10(v) : v)
  const power = (v: number): string => formatNumber(10 ** v, { digits: 2 })
  const xs = c.x.ticks.map(tx)
  const ys = c.y.ticks.map(ty)
  const t = tokens(mode)
  const grid = c.z.map((row) => [...row])
  const field = heatmap(grid, { x: xs, y: ys, mode, label: objectiveLabel(study, options) })
  const levels = contours(grid, {
    x: xs,
    y: ys,
    levels: options.levels ?? 8,
    colour: t.fg,
    opacity: 0.35,
    width: 0.8,
    labels: false,
    mode,
  })
  const trials = points(
    [
      {
        x: c.points.map((p) => tx(p.x)),
        y: c.points.map((p) => ty(p.y)),
        name: '__trials',
        colour: t.fg,
        size: 5,
        opacity: 0.85,
      },
    ],
    { mode },
  )
  const axis = (path: string, log: boolean, ticks: readonly number[], study: Study): AxisSpec => {
    const r = extentOf(ticks) ?? [0, 1]
    if (log) return { label: path, limits: r, format: power }
    const spec = parameterAxis(study, path)
    return { ...spec, limits: spec.limits ?? r }
  }
  return cartesian([field, levels, trials], {
    mode,
    ...(options.figure === undefined ? {} : { figure: options.figure }),
    ...(options.title === undefined ? {} : { title: options.title }),
    x: axis(xPath, c.x.log, xs, study),
    y: axis(yPath, c.y.log, ys, study),
    legend: false,
    tooltip: false,
  })
}

/**
 * The intermediate values every trial reported (Optuna's `plot_intermediate_values`): one curve per trial against its
 * steps, complete trials coloured and pruned trials grey, ending where the pruner stopped them.
 *
 * @param study - The study.
 * @param options - Figure, title, label, log axis and the step axis's name (default `step`).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { createStudy } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * import { intermediateChart } from 'omai-render/charts'
 * const t = { number: 0, params: 0, state: 'pruned', values: [], intermediate: [{ step: 0, value: 2 }, { step: 1, value: 1.5 }], source: 'x' } as const
 * intermediateChart(createStudy(real(0, 1), { trials: [t] })).series[0].name // => 'pruned'
 * ```
 */
export function intermediateChart(
  study: Study,
  options: StudyChartOptions & { readonly step?: string } = {},
): EChartsOption {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const t = tokens(mode)
  const curves = intermediateValues(study)
  const pruned = curves.filter((c) => c.state === 'pruned')
  const complete = curves.filter((c) => c.state !== 'pruned')
  const layers: Layer[] = []
  if (pruned.length > 0)
    layers.push(
      lines(
        pruned.map((c, i) => ({
          x: c.steps,
          y: c.values,
          name: i === 0 ? 'pruned' : '__pruned',
          colour: t.muted,
          width: 1,
          opacity: 0.55,
        })),
        { mode },
      ),
    )
  if (complete.length > 0)
    layers.push(
      lines(
        complete.map((c, i) => ({
          x: c.steps,
          y: c.values,
          name: i === 0 ? 'complete' : '__complete',
          colour: t.tab0,
          width: 1.4,
          opacity: 0.85,
        })),
        { mode },
      ),
    )
  return cartesian(layers, {
    mode,
    ...(options.figure === undefined ? {} : { figure: options.figure }),
    ...(options.title === undefined ? {} : { title: options.title }),
    x: { label: options.step ?? 'step' },
    y: { label: objectiveLabel(study, options), ...(options.log === true ? { scale: 'log' as const } : {}) },
    legend: true,
    tooltip: false,
  })
}

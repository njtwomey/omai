/**
 * Metric curves of traced runs, as TensorBoard draws them: each recorded metric against its steps, optionally smoothed
 * by a debiased exponential moving average over the faint raw values, on a log axis on request, and one metric from
 * several runs overlaid.
 */

import type { EChartsOption } from 'echarts'
import { metricSeries, type Run } from 'omai-engine/trace'
import { modeOf } from '../charts/_data.ts'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer, type PlotOptions } from '../charts/plot.ts'
import { categoricalColour } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'

/**
 * The names of the metrics a run recorded, in order of first appearance.
 *
 * @param run - The trace.
 * @returns The metric names.
 * @example
 * ```ts
 * import { metricNames } from 'omai-render/trace'
 * metricNames({ schema: 'omai.run/1', events: [{ type: 'metric', runId: 'r', step: 0, name: 'loss', value: 1 }] }) // => ['loss']
 * ```
 */
export function metricNames(run: Run): string[] {
  const out: string[] = []
  for (const e of run.events) if (e.type === 'metric' && !out.includes(e.name)) out.push(e.name)
  return out
}

/**
 * Exponential moving average with TensorBoard's debiasing: s_t = w s_{t-1} + (1 - w) x_t, divided by 1 - w^t, so the
 * first values are not dragged towards zero. Non-finite values pass through without updating the average.
 *
 * @param values - The values.
 * @param weight - The smoothing weight in [0, 1) (0 is no smoothing; default 0.6).
 * @returns The smoothed values.
 * @example
 * ```ts
 * import { smoothEma } from 'omai-render/trace'
 * smoothEma([1, 1, 1], 0.9) // => [1, 1, 1]
 * smoothEma([0, 10], 0.5) // => [0, 6.666666666666667]
 * ```
 */
export function smoothEma(values: readonly number[], weight = 0.6): number[] {
  const w = Math.min(Math.max(weight, 0), 0.999)
  let acc = 0
  let k = 0
  return values.map((v) => {
    if (!Number.isFinite(v)) return v
    acc = w * acc + (1 - w) * v
    k += 1
    return w === 0 ? v : acc / (1 - w ** k)
  })
}

/** Options of {@link metricCurves}. */
export interface MetricCurveOptions {
  readonly mode?: ColourMode
  /** EMA smoothing weight (default 0: none); the raw curve stays as a faint line under the smoothed one. */
  readonly smoothing?: number
  /** Restrict to one run of the trace. */
  readonly runId?: string
  /** A name for the curve (default the metric's). */
  readonly name?: string
  readonly colourOffset?: number
}

/**
 * Metric curves of a run as a layer: each metric against its steps, optionally smoothed with the raw values faint
 * underneath.
 *
 * @param run - The trace.
 * @param names - The metrics (default all of them).
 * @param options - Smoothing, run id, name and colour mode.
 * @returns The layer.
 * @example
 * ```ts
 * import { metricCurves } from 'omai-render/trace'
 * const run = { schema: 'omai.run/1' as const, events: [0, 1, 2].map((t) => ({ type: 'metric' as const, runId: 'r', step: t, name: 'loss', value: 3 - t })) }
 * metricCurves(run).extent // => { x: [0, 2], y: [1, 3] }
 * metricCurves(run, ['loss'], { smoothing: 0.5 }).series.length // => 2
 * ```
 */
export function metricCurves(run: Run, names?: readonly string[], options: MetricCurveOptions = {}): Layer {
  const mode = options.mode ?? 'light'
  const list = names ?? metricNames(run)
  const smoothing = options.smoothing ?? 0
  const curves = list.flatMap((name, i) => {
    const s = metricSeries(run, name, options.runId)
    const colour = categoricalColour((options.colourOffset ?? 0) + i)
    const label = options.name ?? name
    return smoothing > 0
      ? [
          { x: s.steps, y: s.values, colour, opacity: 0.25, width: 1, name: `__raw-${label}` },
          { x: s.steps, y: smoothEma(s.values, smoothing), colour, name: label },
        ]
      : [{ x: s.steps, y: s.values, colour, name: label }]
  })
  return lines(curves, { mode })
}

/**
 * A metric chart of a run: the chosen metrics against steps, optionally smoothed, on a log y axis on request.
 *
 * @param run - The trace.
 * @param names - The metrics (default all).
 * @param options - Plot options, smoothing and log scale.
 * @returns The ECharts option.
 * @index function Metric curves of a run against steps: `metricChart(run, ['loss'])`.
 * @example
 * ```ts
 * import { metricChart } from 'omai-render/trace'
 * const run = { schema: 'omai.run/1' as const, events: [{ type: 'metric' as const, runId: 'r', step: 0, name: 'loss', value: 1 }] }
 * metricChart(run, ['loss'], { log: true }).yAxis[0].type // => 'log'
 * ```
 */
export function metricChart(
  run: Run,
  names?: readonly string[],
  options: PlotOptions & { readonly smoothing?: number; readonly log?: boolean } = {},
): EChartsOption {
  const { smoothing, log, ...plot } = options
  return cartesian(
    [metricCurves(run, names, { mode: modeOf(options), ...(smoothing === undefined ? {} : { smoothing }) })],
    {
      tooltip: 'axis',
      x: { label: 'step' },
      ...plot,
      ...(log === true ? { y: { ...plot.y, scale: 'log' } } : {}),
    },
  )
}

/**
 * One metric from several runs overlaid (seeds, learning rates, models), each run a colour.
 *
 * @param runs - Named runs.
 * @param metric - The metric.
 * @param options - Plot options and smoothing.
 * @returns The ECharts option.
 * @index function One metric from several runs, overlaid: `runsChart([{ name, run }], 'loss')`.
 * @example
 * ```ts
 * import { runsChart } from 'omai-render/trace'
 * const run = (v: number) => ({ schema: 'omai.run/1' as const, events: [{ type: 'metric' as const, runId: 'r', step: 0, name: 'loss', value: v }] })
 * runsChart([{ name: 'a', run: run(1) }, { name: 'b', run: run(2) }], 'loss').legend.data // => ['a', 'b']
 * ```
 */
export function runsChart(
  runs: readonly { readonly name: string; readonly run: Run }[],
  metric: string,
  options: PlotOptions & { readonly smoothing?: number } = {},
): EChartsOption {
  const mode = modeOf(options)
  const { smoothing, ...plot } = options
  const layers = runs.map((r, i) =>
    metricCurves(r.run, [metric], {
      mode,
      name: r.name,
      colourOffset: i,
      ...(smoothing === undefined ? {} : { smoothing }),
    }),
  )
  return cartesian(layers, { tooltip: 'axis', x: { label: 'step' }, y: { label: metric }, ...plot })
}

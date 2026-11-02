/**
 * Figures of recorded tensors over training: chosen parameter elements against steps, histograms of a tensor over time
 * (TensorBoard's distributions view), and a heatmap of per-layer norms for spotting vanishing or exploding gradients.
 */

import { MdmlError } from 'mdml-engine/base'
import { metricSeries, type Run, tensorSeries } from 'mdml-engine/trace'
import { lines } from '../charts/lines.ts'
import { heatmap, type RampSpec } from '../charts/matrices.ts'
import type { Layer } from '../charts/plot.ts'
import type { ColourMode } from '../theme/tokens.ts'

/**
 * The recorded steps of a tensor channel and each step's position along x: the step itself, or with `x` a metric
 * channel's value at that step (the last one recorded at or before it), such as the time an equation has reached.
 *
 * @param run - The trace.
 * @param name - The tensor channel.
 * @param x - A metric channel giving the position of each step (default the step).
 * @returns The steps and their positions, in order.
 * @example
 * ```ts
 * import { stepPositions } from 'mdml-render/trace'
 * const ev = (step: number, values: number[]) => ({ type: 'tensor' as const, runId: 'r', step, name: 'y', role: 'state' as const, shape: [1], values })
 * const time = (step: number, value: number) => ({ type: 'metric' as const, runId: 'r', step, name: 't', value })
 * const run = { schema: 'mdml.run/1' as const, events: [time(0, 0), ev(0, [1]), time(1, 0.5), ev(1, [2])] }
 * stepPositions(run, 'y', 't') // => { steps: [0, 1], xs: [0, 0.5] }
 * ```
 */
export function stepPositions(run: Run, name: string, x?: string): { readonly steps: number[]; readonly xs: number[] } {
  const steps = tensorSeries(run, name)
    .filter((e) => e.values !== undefined)
    .map((e) => e.step)
  if (x === undefined) return { steps, xs: steps }
  const m = metricSeries(run, x)
  let i = 0
  const xs = steps.map((s) => {
    while (i + 1 < m.steps.length && (m.steps[i + 1] ?? Number.POSITIVE_INFINITY) <= s) i++
    return (m.steps[i] ?? Number.POSITIVE_INFINITY) <= s ? (m.values[i] ?? Number.NaN) : Number.NaN
  })
  return { steps, xs }
}

/**
 * Parameter trajectories: chosen elements of a recorded tensor channel against steps (from full values, or from the
 * recorded subsample's positions), or against a metric channel's value at each step (`x`).
 *
 * @param run - The trace.
 * @param name - The tensor channel.
 * @param options - Flat element indices to follow (default the first `max`, itself default eight), their names
 *   (default `name[index]`), a metric channel for the x axis, colour mode.
 * @returns The layer, one curve per element; many curves are drawn thin.
 * @index function Chosen parameter elements against steps: `parameterTrajectories(run, 'w')`.
 * @example
 * ```ts
 * import { parameterTrajectories } from 'mdml-render/trace'
 * const ev = (step: number, values: number[]) => ({ type: 'tensor' as const, runId: 'r', step, name: 'w', role: 'param' as const, shape: [2], values })
 * const run = { schema: 'mdml.run/1' as const, events: [ev(0, [0, 1]), ev(1, [1, 2])] }
 * parameterTrajectories(run, 'w').series.map((s) => s.name) // => ['w[0]', 'w[1]']
 * parameterTrajectories(run, 'w', { names: ['a', 'b'] }).series.map((s) => s.name) // => ['a', 'b']
 * ```
 */
export function parameterTrajectories(
  run: Run,
  name: string,
  options: {
    readonly indices?: readonly number[]
    readonly max?: number
    readonly names?: readonly string[]
    readonly x?: string
    readonly mode?: ColourMode
  } = {},
): Layer {
  const events = tensorSeries(run, name).filter((e) => e.values !== undefined)
  const first = events[0]
  const available = first?.indices ?? first?.values?.map((_, i) => i) ?? []
  const wanted = options.indices ?? available.slice(0, options.max ?? 8)
  const positions = options.x === undefined ? undefined : stepPositions(run, name, options.x).xs
  const many = wanted.length > 8
  const curves = wanted.map((flat, k) => {
    const steps: number[] = []
    const values: number[] = []
    events.forEach((e, i) => {
      const pos = e.indices === undefined ? flat : e.indices.indexOf(flat)
      const v = pos >= 0 ? e.values?.[pos] : undefined
      if (v !== undefined) {
        steps.push(positions === undefined ? e.step : (positions[i] ?? Number.NaN))
        values.push(v)
      }
    })
    return {
      x: steps,
      y: values,
      name: options.names?.[k] ?? `${name}[${flat}]`,
      ...(many ? { width: 1, opacity: 0.8 } : {}),
    }
  })
  return lines(curves, { mode: options.mode ?? 'light' })
}

/**
 * Histograms of a tensor channel over time (TensorBoard's distributions view): steps across, values up, the share of
 * elements in each bin as colour, from the recorded summaries (or rebinned from full values).
 *
 * @param run - The trace.
 * @param name - The tensor channel.
 * @param options - Bins for rebinning (default 30), ramp, smoothing of the image and colour mode.
 * @returns The layer.
 * @throws MdmlError with code `value` when the channel has neither summaries nor values.
 * @index function A tensor's histogram over training: `histogramsOverTime(run, 'grad')`.
 * @example
 * ```ts
 * import { histogramsOverTime } from 'mdml-render/trace'
 * const ev = (step: number, values: number[]) => ({ type: 'tensor' as const, runId: 'r', step, name: 'g', role: 'grad' as const, shape: [3], values })
 * const run = { schema: 'mdml.run/1' as const, events: [ev(0, [0, 1, 2]), ev(1, [1, 2, 3])] }
 * histogramsOverTime(run, 'g', { bins: 3 }).extent.x // => [0, 1]
 * ```
 */
export function histogramsOverTime(
  run: Run,
  name: string,
  options: { readonly bins?: number; readonly ramp?: RampSpec; readonly mode?: ColourMode } = {},
): Layer {
  const events = tensorSeries(run, name)
  if (events.length === 0 || events.every((e) => e.summary === undefined && e.values === undefined))
    throw new MdmlError('value', `histogramsOverTime: no summaries or values for '${name}'`)
  const bins = options.bins ?? 30
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const e of events) {
    const vs = e.values ?? []
    const s = e.summary
    lo = Math.min(lo, s?.min ?? Number.POSITIVE_INFINITY, ...vs)
    hi = Math.max(hi, s?.max ?? Number.NEGATIVE_INFINITY, ...vs)
  }
  if (!(hi > lo)) hi = lo + 1
  const width = (hi - lo) / bins
  // Columns are steps, rows are value bins (row 0 at the bottom).
  const grid = Array.from({ length: bins }, () => new Array<number>(events.length).fill(0))
  events.forEach((e, j) => {
    const add = (value: number, weight: number): void => {
      const b = Math.min(bins - 1, Math.max(0, Math.floor((value - lo) / width)))
      const row = grid[b]
      if (row !== undefined) row[j] = (row[j] ?? 0) + weight
    }
    if (e.values !== undefined) for (const v of e.values) add(v, 1 / e.values.length)
    else if (e.summary !== undefined) {
      const { edges, counts } = e.summary.histogram
      const total = counts.reduce((a, b) => a + b, 0) || 1
      counts.forEach((c, k) => {
        add(((edges[k] ?? 0) + (edges[k + 1] ?? 0)) / 2, c / total)
      })
    }
  })
  const steps = events.map((e) => e.step)
  return heatmap(grid, {
    x: steps.length > 1 ? steps : [steps[0] ?? 0, (steps[0] ?? 0) + 1],
    y: [lo + width / 2, hi - width / 2],
    ramp: options.ramp ?? 'blues',
    smooth: 'bilinear',
    label: 'share',
    mode: options.mode ?? 'light',
  })
}

/**
 * Norms of several tensor channels over time as a heatmap (one row per channel, steps across): per-layer gradient or
 * weight norms, to spot vanishing or exploding layers. Uses the recorded summaries' norms.
 *
 * @param run - The trace.
 * @param names - The channels, top row first.
 * @param options - Log colour scale (default true), colour mode.
 * @returns The layer, rows labelled by channel.
 * @example
 * ```ts
 * import { normHeatmap } from 'mdml-render/trace'
 * const s = { n: 1, mean: 0, std: 0, min: 0, max: 0, norm: 2, zeros: 0, histogram: { edges: [0, 1], counts: [1] } }
 * const run = { schema: 'mdml.run/1' as const, events: [{ type: 'tensor' as const, runId: 'r', step: 0, name: 'a', role: 'grad' as const, shape: [1], summary: s }] }
 * normHeatmap(run, ['a']).yCategories // => ['a']
 * ```
 */
export function normHeatmap(
  run: Run,
  names: readonly string[],
  options: { readonly log?: boolean; readonly mode?: ColourMode } = {},
): Layer {
  const series = names.map((n) => tensorSeries(run, n))
  const steps = [...new Set(series.flatMap((s) => s.map((e) => e.step)))].sort((a, b) => a - b)
  const log = options.log ?? true
  const z = series.map((s) =>
    steps.map((t) => {
      const e = s.find((x) => x.step === t)
      const norm = e?.summary?.norm ?? (e?.values === undefined ? Number.NaN : Math.hypot(...e.values))
      return log ? Math.log10(norm) : norm
    }),
  )
  return heatmapRows(z, names, steps, options.mode ?? 'light', log ? 'log10 norm' : 'norm')
}

function heatmapRows(z: number[][], names: readonly string[], steps: number[], mode: ColourMode, label: string): Layer {
  // Rows as categories (top row first) need the matrix view; steps are many, so cells are unannotated.
  const l = heatmap(z.slice().reverse(), {
    x: steps.length > 1 ? steps : [steps[0] ?? 0, (steps[0] ?? 0) + 1],
    smooth: 'cells',
    label,
    mode,
  })
  return { ...l, extent: { x: l.extent.x ?? [0, 1], tight: true }, yCategories: [...names].reverse() }
}

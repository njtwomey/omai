import type { EChartsOption, SeriesOption } from 'echarts'
import type { StateSequence } from 'whitebox-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { band } from '../charts/lines.ts'
import { type Layer, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatPercent } from '../theme/format.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link stateSequenceView}. */
export type StateSequenceViewOptions = ViewOptions

interface CellApi {
  coord(p: readonly number[]): number[]
  size(p: readonly number[]): number[]
  value(d: number): number
}

/** Runs of equal values in a path: `[start, end, value]`, end inclusive. */
export function runsOf(path: readonly number[]): [number, number, number][] {
  const out: [number, number, number][] = []
  for (const [t, s] of path.entries()) {
    const last = out[out.length - 1]
    if (last !== undefined && last[2] === s && last[1] === t - 1) last[1] = t
    else out.push([t, t, s])
  }
  return out
}

// Paths as strips of coloured runs, one category row each (row 0 at the top).
function stripLayer(paths: readonly { readonly name: string; readonly path: readonly number[] }[], n: number): Layer {
  const data: number[][] = []
  for (const [r, p] of paths.entries())
    for (const [a, b, s] of runsOf(p.path)) data.push([a - 0.5, b + 0.5, paths.length - 1 - r, s])
  const series: SeriesOption = {
    type: 'custom',
    name: '__strips',
    data,
    encode: { x: [0, 1], y: 2 },
    renderItem: (_: unknown, api: CellApi): Record<string, unknown> => {
      const [x0 = 0, y = 0] = api.coord([api.value(0), api.value(2)])
      const [x1 = 0] = api.coord([api.value(1), api.value(2)])
      const [, h = 0] = api.size([0, 1])
      return {
        type: 'rect',
        shape: { x: x0, y: y - h * 0.36, width: Math.max(0.5, x1 - x0), height: h * 0.72 },
        style: { fill: categoricalColour(api.value(3)) },
      }
    },
    tooltip: { show: false },
  } as unknown as SeriesOption
  return layer([series], { x: [-0.5, n - 0.5] }, { yCategories: paths.map((p) => p.name).reverse() })
}

/**
 * A hidden-state sequence as one figure, on a shared time axis: the posterior of every state at every step as stacked
 * bands (each state its colour, P summing to one), beneath it strips of the Viterbi path and the true states (where
 * they differ shows where decoding errs, with the agreement in the panel title), and above, when the record has them,
 * the observations.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { stateSequence } from 'whitebox-engine/exposition'
 * import { stateSequenceView } from 'whitebox-render/views'
 * const r = stateSequence({ method: 'hmm', states: ['fair', 'loaded'], posteriors: [[0.9, 0.1], [0.2, 0.8]], decoded: [0, 1], truth: [0, 1] })
 * stateSequenceView(r).grid.length // => 2
 * ```
 */
export function stateSequenceView(record: StateSequence, options: StateSequenceViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2 })
  const mode = fig.mode
  const n = record.posteriors.length
  const k = record.states.length
  const time = Array.from({ length: n }, (_, t) => t)
  const stepName = record.stepName ?? 'time'
  // Stacked posteriors: state j fills between the sum of the states before it and the sum including it.
  const below = new Array<number>(n).fill(0)
  const bands = record.states.map((name, j) => {
    const lower = [...below]
    for (let t = 0; t < n; t++) below[t] = (below[t] ?? 0) + (record.posteriors[t]?.[j] ?? 0)
    return { x: time, lower, upper: [...below], colour: categoricalColour(j), opacity: 0.75, name }
  })
  const panels: Record<string, Panel> = {
    posterior: {
      layers: [band(bands, { mode })],
      options: { x: { label: stepName, limits: [-0.5, n - 0.5] }, y: { label: 'P(state)', limits: [0, 1] } },
    },
  }
  const ids: string[] = []
  const obs = record.observations
  if (obs !== undefined) {
    const symbols = obs.symbols
    panels['observations'] = {
      layers: [
        points(
          [
            {
              x: time,
              y: obs.values,
              ...(record.truth === undefined
                ? { colour: 'fgSecondary' }
                : {
                    groups: record.truth,
                    groupNames: Object.fromEntries(record.states.map((s, j) => [String(j), s])),
                  }),
              size: n > 400 ? 3 : 4,
              outline: false,
              name: '__observations',
            },
          ],
          { mode },
        ),
      ],
      options: {
        x: { limits: [-0.5, n - 0.5] },
        y: {
          label: obs.name ?? 'observed',
          ...(symbols === undefined ? {} : { format: (v: number) => (Number.isInteger(v) ? (symbols[v] ?? '') : '') }),
        },
      },
    }
    ids.push('observations')
  }
  ids.push('posterior')
  const paths = [
    ...(record.decoded === undefined ? [] : [{ name: 'Viterbi', path: record.decoded }]),
    ...(record.truth === undefined ? [] : [{ name: 'truth', path: record.truth }]),
  ]
  if (paths.length > 0) {
    const agree =
      record.decoded !== undefined && record.truth !== undefined && n > 0
        ? record.decoded.filter((s, t) => s === record.truth?.[t]).length / n
        : undefined
    panels['paths'] = {
      layers: [stripLayer(paths, n)],
      options: {
        x: { label: stepName, limits: [-0.5, n - 0.5] },
        y: { splitLine: false },
        ...(agree === undefined ? {} : { title: `the Viterbi path matches the truth at ${formatPercent(agree, 1)}` }),
      },
    }
    ids.push('paths')
  }
  // Only the bottom panel names the time axis.
  if (ids.at(-1) !== 'posterior')
    panels['posterior'] = {
      ...(panels['posterior'] as Panel),
      options: { x: { limits: [-0.5, n - 0.5] }, y: { label: 'P(state)', limits: [0, 1] } },
    }
  const ratios: Record<string, number> = { observations: 1.1, posterior: 2.4, paths: 0.5 + 0.35 * paths.length }
  const grid = gridSpec(ids.length, 1, {
    hspace: 30,
    heightRatios: ids.map((id) => ratios[id] ?? 1),
    cells: ids.map((id, r) => ({ id, row: r, col: 0, ...(r === 0 ? {} : { sharex: ids[0] }) })),
  })
  return subplots(grid, panels, { figure: fig, legend: k > 1, tooltip: false })
}

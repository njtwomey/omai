/**
 * The view of a hidden-state model's reading of a sequence (an HMM, say): each state's posterior over time stacked to
 * one as a strip, with the decoded path and the true states as coloured strips beneath.
 */

import type { EChartsOption } from 'echarts'
import type { StateSequence } from 'whitebox-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { band, lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

const stateColour = (k: number): string => `tab${k % 10}`

/**
 * A hidden-state model's reading of a sequence as a state strip: each state's posterior over time stacked to one (the
 * colour of the likelier state fills more of the column), and beneath it the decoded (Viterbi) path and the true
 * states as strips of coloured cells, so where decoding misses the truth shows at a glance; the observations, when
 * the record has them, as dots above the strips.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { stateSequence } from 'whitebox-engine/exposition'
 * import { stateSequenceView } from 'whitebox-render/views'
 * const r = stateSequence({ states: ['a', 'b'], posteriors: [[0.9, 0.1], [0.3, 0.7]], decoded: [0, 1], truth: [0, 0] })
 * stateSequenceView(r).grid.length // => 2
 * ```
 */
export function stateSequenceView(record: StateSequence, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2.4 })
  const mode = fig.mode
  const T = record.posteriors.length
  const t = Array.from({ length: T }, (_, i) => i)
  // The posteriors stacked: state k's band runs from the sum of the states before it to the sum including it.
  const below = new Array<number>(T).fill(0)
  const stack: Layer[] = record.states.map((name, k) => {
    const lower = [...below]
    for (let i = 0; i < T; i++) below[i] = (below[i] ?? 0) + (record.posteriors[i]?.[k] ?? 0)
    return band([{ x: t, lower, upper: [...below], colour: stateColour(k), opacity: 0.75, name }], { mode })
  })
  const panels: Record<string, Panel> = {
    posteriors: {
      layers: stack,
      options: { x: { label: 'time', show: false }, y: { label: 'posterior', limits: [0, 1] } },
    },
  }
  // The paths as rows of cells: one category per path, a cell per time coloured by its state.
  const paths = [
    ...(record.truth === undefined ? [] : [{ name: 'truth', path: record.truth }]),
    ...(record.decoded === undefined ? [] : [{ name: 'decoded', path: record.decoded }]),
  ]
  if (paths.length > 0) {
    // Each path's runs of one state as thick horizontal segments, one series per state.
    const layers: Layer[] = [
      lines(
        record.states.map((name, k) => {
          const x: number[] = []
          const y: number[] = []
          for (const [r, p] of paths.entries()) {
            let i = 0
            while (i < p.path.length) {
              if (p.path[i] !== k) {
                i++
                continue
              }
              let j = i
              while (j + 1 < p.path.length && p.path[j + 1] === k) j++
              x.push(i - 0.5, j + 0.5, Number.NaN)
              y.push(r, r, Number.NaN)
              i = j + 1
            }
          }
          return { x, y, name, colour: stateColour(k), width: 12 }
        }),
        { mode },
      ),
    ]
    const obs = record.observations
    if (obs !== undefined) {
      // The observations scaled into the gap above the paths.
      const lo = Math.min(...obs.values)
      const hi = Math.max(...obs.values)
      const y = obs.values.map((v) => paths.length - 0.5 + (hi > lo ? (v - lo) / (hi - lo) : 0.5) * 0.9)
      layers.push(points([{ x: t, y, name: obs.name, colour: 'fgSecondary', size: 3, outline: false }], { mode }))
    }
    // Decoding errors: the times where the decoded state is not the true one.
    const wrong =
      record.truth !== undefined && record.decoded !== undefined
        ? record.decoded.flatMap((s, i) => (s !== record.truth?.[i] ? [i] : []))
        : []
    panels['paths'] = {
      layers: [
        ...layers,
        ...(wrong.length === 0
          ? []
          : [
              lines(
                [
                  {
                    x: wrong.flatMap((i) => [i, i, Number.NaN]),
                    y: wrong.flatMap(() => [-0.5, -0.38, Number.NaN]),
                    name: '__errors',
                    colour: 'danger',
                    width: 1.5,
                  },
                ],
                { mode },
              ),
            ]),
      ],
      options: {
        ...(record.truth === undefined || record.decoded === undefined
          ? {}
          : { title: `decoded ${Math.round((100 * (T - wrong.length)) / Math.max(1, T))}% of the states right` }),
        x: { label: 'time' },
        y: {
          label: '',
          limits: [-0.5, paths.length - 0.5 + (obs === undefined ? 0 : 1)],
          format: (v: number) =>
            paths[Math.round(v)]?.name ?? (obs !== undefined && v >= paths.length - 0.5 ? obs.name : ''),
        },
      },
    }
  }
  const ids = Object.keys(panels)
  const grid = gridSpec(ids.length, 1, {
    heightRatios: ids.map((id) => (id === 'posteriors' ? 2.2 : 1)),
    hspace: 36,
    cells: ids.map((id, r) => ({ id, row: r, col: 0, ...(r > 0 ? { sharex: 'posteriors' } : {}) })),
  })
  return subplots(grid, panels, { figure: fig, tooltip: 'item' })
}

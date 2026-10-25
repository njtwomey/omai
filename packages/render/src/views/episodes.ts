/**
 * The view of a reinforcement-learning run: every episode's return with its rolling mean and spread, the episode
 * lengths, and a snapshot of the value function and policy, as a map with arrows for grid worlds.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { EpisodeRecord } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { matrix } from '../charts/matrices.ts'
import { type Layer, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { rolling } from '../trace/episodes.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

const ARROWS: Readonly<Record<string, string>> = { up: '↑', down: '↓', left: '←', right: '→' }

/** Glyphs for named actions: arrows for directions, else the names themselves. */
function glyphsOf(names: readonly string[] | undefined): string[] | undefined {
  return names?.map((n) => ARROWS[n.toLowerCase()] ?? n)
}

/** Options of {@link episodeView}. */
export interface EpisodeViewOptions extends ViewOptions {
  /** The snapshot shown (an index into the record's snapshots; default the last). */
  readonly snapshot?: number
  /** The states as a grid `[rows, cols]` (a grid world), to draw values and policy as a map (default the record's). */
  readonly gridShape?: readonly [number, number]
  /**
   * Arrows (or names) of the actions, by action index, for the policy map (default from the record's action names,
   * else ←, ↓, →, ↑: gymnasium's frozen lake).
   */
  readonly actionGlyphs?: readonly string[]
  /** The rolling window of the mean return (default 20). */
  readonly window?: number
}

/**
 * A reinforcement-learning run as one figure: every episode's return with its rolling mean and spread (the snapshot's
 * episode marked), the episode lengths, and a value and policy snapshot: as a map with the greedy action in each cell
 * for grid worlds, else as bars per state coloured by the greedy action.
 *
 * @param record - The record.
 * @param options - Mode, figure, the snapshot, the grid's shape, action glyphs and the rolling window.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { episodeRecord } from 'whitebox-engine/exposition'
 * import { episodeView } from 'whitebox-render/views'
 * episodeView(episodeRecord({ environment: 'FrozenLake', agent: 'q', returns: [0, 1, 1], lengths: [5, 8, 6] })).grid.length // => 2
 * ```
 */
export function episodeView(record: EpisodeRecord, options: EpisodeViewOptions = {}): EChartsOption {
  const snaps = record.snapshots ?? []
  const snap = snaps[Math.min(Math.max(options.snapshot ?? snaps.length - 1, 0), snaps.length - 1)]
  const fig = viewFigure(options, { aspect: snap === undefined ? 1.6 : 1.9 })
  const mode = fig.mode
  const x = record.returns.map((_, i) => i)
  const { mean, std } = rolling(record.returns, options.window ?? 20)
  const returns: Layer[] = [
    points([{ x, y: record.returns, name: '__episodes', colour: 'tab0', size: 3, opacity: 0.35, outline: false }], {
      mode,
    }),
    band(
      [
        {
          x,
          lower: mean.map((m, k) => m - (std[k] ?? 0)),
          upper: mean.map((m, k) => m + (std[k] ?? 0)),
          colour: 'tab0',
          opacity: 0.15,
        },
      ],
      { mode },
    ),
    lines([{ x, y: mean, name: 'mean return', colour: 'tab0', width: 2 }], { mode }),
  ]
  if (snap !== undefined) returns.push(referenceLines({ x: [snap.episode] }, { mode, colour: 'accent', dash: 'solid' }))
  const panels: Record<string, Panel> = {
    returns: {
      layers: returns,
      options: { title: `${record.agent} on ${record.environment}`, x: { label: 'episode' }, y: { label: 'return' } },
    },
    lengths: {
      layers: [
        lines(
          [{ x, y: rolling(record.lengths, options.window ?? 20).mean, name: 'length', colour: 'tab1', width: 1.5 }],
          { mode },
        ),
      ],
      options: { x: { label: 'episode' }, y: { label: 'length' } },
    },
  }
  if (snap !== undefined) {
    const values = snap.values ?? snap.policy?.map(() => 0) ?? []
    // A policy over other indices than the values (a bandit's single best arm) is not drawn per cell.
    const policy = snap.policy !== undefined && snap.policy.length === values.length ? snap.policy : undefined
    const glyphs = options.actionGlyphs ?? glyphsOf(record.actionNames) ?? ['←', '↓', '→', '↑']
    const shape = options.gridShape ?? record.snapshotShape
    if (shape !== undefined && shape[0] * shape[1] === values.length) {
      const [r, c] = shape
      const grid = Array.from({ length: r }, (_, i) => values.slice(i * c, (i + 1) * c))
      panels['snapshot'] = {
        layers: [
          matrix(grid, {
            rowLabels: grid.map((_, i) => `${i}`),
            colLabels: grid[0]?.map((_, j) => `${j}`) ?? [],
            ramp: 'viridis',
            colourBar: false,
            annotate: policy === undefined,
            mode,
          }),
          ...(policy === undefined
            ? []
            : [
                // The greedy action written in each cell, over the values.
                layer(
                  [
                    {
                      type: 'scatter',
                      name: '__policy',
                      data: values.map((_, s) => ({
                        value: [`${s % c}`, `${Math.floor(s / c)}`],
                        label: { formatter: glyphs[policy[s] ?? 0] ?? String(policy[s]) },
                      })),
                      symbolSize: 0,
                      silent: true,
                      label: {
                        show: true,
                        color: '#ffffff',
                        fontSize: 14,
                        fontWeight: 'bold',
                        textBorderColor: '#000000',
                        textBorderWidth: 2,
                      },
                      tooltip: { show: false },
                      z: 5,
                    } as SeriesOption,
                  ],
                  {},
                ),
              ]),
        ],
        options: { title: `values and greedy policy, episode ${snap.episode}` },
      }
    } else {
      const k = Math.max(0, ...(policy ?? [0])) + 1
      panels['snapshot'] = {
        layers: [
          bars(
            values.map((_, s) => `${s}`),
            Array.from({ length: k }, (_, a) => ({
              name: record.actionNames?.[a] ?? glyphs[a] ?? `action ${a}`,
              colour: `tab${(a + 2) % 10}`,
              values: values.map((v, s) => ((policy?.[s] ?? 0) === a ? v : 0)),
            })),
            { mode, stacked: true },
          ),
        ],
        options: {
          title: `state values, episode ${snap.episode}`,
          x: { label: 'state', show: values.length <= 40 },
          y: { label: 'value' },
        },
      }
    }
  }
  const hasSnap = panels['snapshot'] !== undefined
  const grid = hasSnap
    ? gridSpec(2, 2, {
        widthRatios: [1.4, 1],
        heightRatios: [2, 1],
        wspace: 64,
        hspace: 44,
        cells: [
          { id: 'returns', row: 0, col: 0 },
          { id: 'lengths', row: 1, col: 0, sharex: 'returns' },
          { id: 'snapshot', row: 0, col: 1, rowSpan: 2 },
        ],
      })
    : gridSpec(2, 1, {
        heightRatios: [2, 1],
        hspace: 16,
        cells: [
          { id: 'returns', row: 0, col: 0 },
          { id: 'lengths', row: 1, col: 0, sharex: 'returns' },
        ],
      })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

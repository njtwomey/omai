import type { EChartsOption } from 'echarts'
import type { AnomalyScores } from 'whitebox-engine/exposition'
import { histogram } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { outlined, short, type ViewOptions, viewFigure } from './_view.ts'
import type { Rows } from './surfaces.ts'

/** Options of {@link anomalyView}. */
export interface AnomalyViewOptions extends ViewOptions {
  /** The rows scored (their first two columns, or `columns`), coloured by score with the flagged ones ringed. */
  readonly points?: Rows
  readonly columns?: readonly [number, number]
  /** A threshold to use instead of the record's (a control's value). */
  readonly threshold?: number
}

/**
 * Anomaly scores as one figure: the scores' histogram with the threshold and the number flagged, the per-row evidence
 * (path lengths, reachability densities, distances) against the score, and, given the rows, the rows coloured by score
 * with the flagged ones ringed.
 *
 * @param record - The record.
 * @param options - Mode, figure, the rows and columns, and a threshold.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { anomalyScores } from 'whitebox-engine/exposition'
 * import { anomalyView } from 'whitebox-render/views'
 * anomalyView(anomalyScores({ method: 'iforest', scores: [0.4, 0.5, 0.9], threshold: 0.6 })).grid.length // => 1
 * ```
 */
export function anomalyView(record: AnomalyScores, options: AnomalyViewOptions = {}): EChartsOption {
  const rows = options.points
  const fig = viewFigure(options, { aspect: rows === undefined && record.evidence === undefined ? 1.6 : 1.75 })
  const mode = fig.mode
  const threshold = options.threshold ?? record.threshold
  const flagged = record.scores.flatMap((s, i) => (s > threshold ? [i] : []))
  const panels: Record<string, Panel> = {
    scores: {
      layers: [
        histogram(record.scores, { bins: 30, name: '__scores', colour: 'tab0', mode }),
        referenceLines({ x: [threshold] }, { mode, colour: 'danger', dash: 'solid' }),
      ],
      options: {
        title: `${flagged.length} of ${record.scores.length} flagged`,
        x: { label: 'anomaly score' },
        y: { label: 'rows' },
      },
    },
  }
  if (record.evidence !== undefined)
    panels['evidence'] = {
      layers: [
        points(
          [
            {
              x: record.scores,
              y: record.evidence.values,
              name: '__evidence',
              colour: 'tab0',
              size: 5,
              opacity: 0.7,
            },
          ],
          { mode },
        ),
        referenceLines({ x: [threshold] }, { mode, colour: 'danger', dash: 'solid' }),
      ],
      options: { x: { label: 'anomaly score' }, y: { label: record.evidence.name } },
    }
  if (rows !== undefined) {
    const [a, b] = options.columns ?? [0, 1]
    const layers: Layer[] = [
      points(
        [
          {
            x: rows.map((r) => r[a] ?? 0),
            y: rows.map((r) => r[b] ?? 0),
            values: [...record.scores],
            name: '__rows',
            size: 6,
          },
        ],
        { mode },
      ),
      outlined(
        points(
          [
            {
              x: flagged.map((i) => rows[i]?.[a] ?? 0),
              y: flagged.map((i) => rows[i]?.[b] ?? 0),
              name: 'flagged',
              colour: 'none',
              size: 13,
              outline: false,
            },
          ],
          { mode },
        ),
        mode,
        1.6,
      ),
    ]
    panels['rows'] = {
      layers,
      options: {
        title: `${record.method}, threshold ${short(threshold)}`,
        x: { label: `x${a}` },
        y: { label: `x${b}` },
        aspect: 'equal',
      },
    }
  }
  const right = ['scores', 'evidence'].filter((id) => panels[id] !== undefined)
  if (rows === undefined && right.length === 1) {
    const p = panels['scores'] as Panel
    return cartesian(p.layers, { ...p.options, figure: fig, mode, legend: false, title: p.options?.title ?? '' })
  }
  const grid =
    rows !== undefined
      ? gridSpec(right.length, 2, {
          widthRatios: [1.3, 1],
          wspace: 64,
          hspace: 56,
          cells: [
            { id: 'rows', row: 0, col: 0, rowSpan: right.length },
            ...right.map((id, r) => ({ id, row: r, col: 1 })),
          ],
        })
      : gridSpec(1, right.length, { wspace: 64, cells: right.map((id, c) => ({ id, row: 0, col: c })) })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

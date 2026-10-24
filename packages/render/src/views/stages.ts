import type { EChartsOption } from 'echarts'
import type { EnsembleStages } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'
import { classIndices, type Rows } from './surfaces.ts'

/** Options of {@link ensembleFrames}. */
export interface EnsembleViewOptions extends ViewOptions {
  /** The training rows, to draw what each stage's member was fitted to (sample weights or residuals) over them. */
  readonly points?: Rows
  readonly labels?: readonly (number | string)[]
  readonly columns?: readonly [number, number]
  /** The y label of the stage scores (default 'score'; a boosting loss reads better as 'training loss'). */
  readonly scoreLabel?: string
}

/** Whether a stage's targets are sample weights (non-negative, summing to about one) rather than residuals. */
function areWeights(targets: readonly number[]): boolean {
  const sum = targets.reduce((s, v) => s + v, 0)
  return targets.every((v) => v >= 0) && Math.abs(sum - 1) < 1e-3
}

/**
 * An ensemble built stage by stage as frames for a player: the training (and validation or out-of-bag) score after each
 * stage with the current stage marked, the members' weights with the current one highlighted, and, given the training
 * rows, what the current member was fitted to: AdaBoost's sample weights as point sizes, gradient boosting's residuals
 * as colours.
 *
 * @param stages - The record.
 * @param options - Mode, figure, training rows, labels and columns, and the score label.
 * @returns The frames.
 */
export function ensembleFrames(stages: EnsembleStages, options: EnsembleViewOptions = {}): ViewFrames {
  const rows = options.points
  const withRows = rows !== undefined && stages.targets !== undefined
  const fig = viewFigure(options, { aspect: withRows ? 1.75 : 1.6 })
  const mode = fig.mode
  const count = stages.trainScores.length
  const x = stages.trainScores.map((_, i) => i + 1)
  const constant = stages.weights.every((w) => w === stages.weights[0])
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const scoreLayers: Layer[] = [
      lines(
        [
          { x, y: stages.trainScores, name: 'train', colour: 'tab0', width: 1.6 },
          ...(stages.validationScores === undefined
            ? []
            : [{ x, y: stages.validationScores, name: 'validation', colour: 'tab1', width: 1.6 }]),
        ],
        { mode },
      ),
      referenceLines({ x: [t + 1] }, { mode, colour: 'accent', dash: 'solid' }),
    ]
    const panels: Record<string, Panel> = {
      scores: {
        layers: scoreLayers,
        options: { x: { label: 'stage' }, y: { label: options.scoreLabel ?? 'score' } },
      },
    }
    if (!constant)
      panels['weights'] = {
        layers: [
          bars(
            x.map(String),
            [
              { name: '__weight', values: stages.weights.map((w, i) => (i === t ? 0 : w)), colour: 'muted' },
              { name: '__current', values: stages.weights.map((w, i) => (i === t ? w : 0)), colour: 'accent' },
            ],
            { mode, stacked: true },
          ),
        ],
        options: { x: { label: 'member', show: count <= 30 }, y: { label: 'member weight' } },
      }
    if (withRows) {
      const target = stages.targets?.[t] ?? []
      const columns = options.columns ?? [0, 1]
      const weights = areWeights(target)
      const labels = options.labels === undefined ? undefined : classIndices(options.labels, undefined)
      let sets: PointSet[]
      if (weights)
        // Sample weights as areas: the rows the next member must get right grow.
        sets = [
          {
            x: rows.map((r) => r[columns[0]] ?? 0),
            y: rows.map((r) => r[columns[1]] ?? 0),
            ...(labels === undefined ? { colour: 'tab0' } : { groups: labels }),
            sizes: rows.map((_, i) => target[i] ?? 0),
            sizeRange: [2, 24],
            opacity: 0.7,
            name: '__rows',
          },
        ]
      else
        sets = [
          {
            x: rows.map((r) => r[columns[0]] ?? 0),
            y: rows.map((r) => r[columns[1]] ?? 0),
            values: rows.map((_, i) => target[i] ?? 0),
            name: '__residual',
            size: 7,
          },
        ]
      panels['rows'] = {
        layers: [points(sets, { mode })],
        options: {
          title: weights ? `sample weights for member ${t + 1}` : `residuals fitted by member ${t + 1}`,
          x: { label: `x${columns[0]}` },
          y: { label: `x${columns[1]}` },
          aspect: 'equal',
        },
      }
    }
    const right = Object.keys(panels).filter((k) => k !== 'rows')
    const grid = withRows
      ? gridSpec(right.length, 2, {
          widthRatios: [1.4, 1],
          wspace: 64,
          hspace: 48,
          cells: [
            { id: 'rows', row: 0, col: 0, rowSpan: right.length },
            ...right.map((id, r) => ({ id, row: r, col: 1 })),
          ],
        })
      : gridSpec(right.length, 1, {
          hspace: 48,
          heightRatios: right.map((id) => (id === 'scores' ? 2 : 1)),
          cells: right.map((id, r) => ({ id, row: r, col: 0 })),
        })
    return subplots(grid, panels, { figure: fig, tooltip: 'item' })
  }
  return {
    count,
    label: (f) => `stage ${f + 1}/${count} · ${short(stages.trainScores[f] ?? 0)}`,
    option,
  }
}

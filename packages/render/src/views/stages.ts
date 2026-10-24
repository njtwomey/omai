import type { EChartsOption } from 'echarts'
import type { EnsembleStages, PipelineStages } from 'whitebox-engine/exposition'
import { arrow, group, line, type Paint, rect, type Scene, type SceneNode, scene, text } from 'whitebox-engine/scene'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import type { Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber } from '../theme/format.ts'
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

const STAGE_W = 196
const LINE = 15

function cell(v: number | string | readonly number[]): string {
  if (typeof v === 'string') return v.length > 22 ? `${v.slice(0, 21)}…` : v
  if (typeof v === 'number') return formatNumber(v, { digits: 3 })
  const shown = v.slice(0, 3).map((x) => formatNumber(x, { digits: 3 }))
  return `[${shown.join(', ')}${v.length > 3 ? ', …' : ''}]`
}

/**
 * A fitted pipeline as a scene: one card per stage, left to right, with its kind, the shape it takes and gives, what
 * it learned (a few name-value pairs), and the first rows of its output; arrows between the cards carry the shapes.
 *
 * @param record - The record.
 * @returns The scene.
 * @example
 * ```ts
 * import { pipelineStages } from 'whitebox-engine/exposition'
 * import { pipelineScene } from 'whitebox-render/views'
 * pipelineScene(pipelineStages({ stages: [] })).kind // => 'scene'
 * ```
 */
export function pipelineScene(record: PipelineStages): Scene {
  const stages = record.stages
  const learnedLines = Math.min(4, Math.max(0, ...stages.map((s) => s.learned.length)))
  const sampleRows = Math.min(4, Math.max(0, ...stages.map((s) => s.outputSample.length)))
  const height = 16 + 2 * LINE + 8 + learnedLines * LINE + 10 + (sampleRows + 1) * LINE + 12
  const gap = 56
  const children: SceneNode[] = []
  stages.forEach((s, k) => {
    const x0 = 8 + k * (STAGE_W + gap)
    const y0 = 8
    const parts: SceneNode[] = [
      rect(x0, y0, STAGE_W, height, { fill: 'surface', stroke: s.kind === 'estimator' ? 'accent' : 'grid', radius: 6 }),
      text(x0 + 10, y0 + 18, s.name, { size: 12, weight: 'bold', fill: 'fg' }),
      text(x0 + STAGE_W - 10, y0 + 18, s.kind, { size: 10, fill: 'muted', anchor: 'end' }),
      text(
        x0 + 10,
        y0 + 18 + LINE,
        `${s.inputShape[0]} × ${s.inputShape[1]} → ${s.outputShape[0]} × ${s.outputShape[1]}`,
        {
          size: 10,
          fill: 'fgSecondary' as Paint,
        },
      ),
    ]
    let y = y0 + 18 + 2 * LINE + 6
    for (const l of s.learned.slice(0, learnedLines)) {
      parts.push(text(x0 + 10, y, `${cell(l.name)}: ${cell(l.value)}`, { size: 10, fill: 'fg' }))
      y += LINE
    }
    y = y0 + 18 + 2 * LINE + 8 + learnedLines * LINE + 8
    // The first rows of the stage's output, first three columns, as a small table.
    const cols = s.outputColumns.slice(0, 3)
    const cw = (STAGE_W - 20) / Math.max(1, cols.length)
    parts.push(line(x0 + 8, y - 11, x0 + STAGE_W - 8, y - 11, { stroke: 'grid' }))
    for (const [j, c] of cols.entries())
      parts.push(text(x0 + 10 + j * cw, y, c.length > 9 ? `${c.slice(0, 8)}…` : c, { size: 9, fill: 'muted' }))
    for (const [i, r] of s.outputSample.slice(0, sampleRows).entries())
      for (let j = 0; j < cols.length; j++)
        parts.push(
          text(x0 + 10 + j * cw, y + (i + 1) * LINE, formatNumber(r[j] ?? 0, { digits: 3 }), { size: 9, fill: 'fg' }),
        )
    children.push(group(parts, { id: `stage:${k}`, title: `${s.name} (${s.kind})` }))
    if (k + 1 < stages.length) {
      const ax = x0 + STAGE_W + 6
      const ay = y0 + height / 2
      children.push(arrow(ax, ay, ax + gap - 12, ay, { stroke: 'fgSecondary' as Paint, strokeWidth: 1.4 }))
      children.push(
        text(ax + (gap - 12) / 2, ay - 6, `${s.outputShape[1]}`, { size: 10, fill: 'muted', anchor: 'middle' }),
      )
    }
  })
  const width = 16 + stages.length * STAGE_W + Math.max(0, stages.length - 1) * gap
  return scene(Math.max(width, 200), height + 16, children, { title: 'pipeline' })
}

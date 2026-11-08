/**
 * Views of models built in stages: an ensemble stage by stage (training and validation scores, member weights, and what
 * each member was fitted to) as player frames, and a fitted pipeline as a strip of stage cards with the panel each
 * step's output suits (a scatter, a projection with its explained variance, a table, or the final estimator's surface
 * over the raw inputs).
 */

import type { EChartsOption } from 'echarts'
import { maxOf } from 'omai-engine/base'
import {
  type DecisionSurface,
  type EnsembleStages,
  type PipelineStage,
  type PipelineStages,
  type StageData,
  surfaceAt,
} from 'omai-engine/exposition'
import { arrow, group, line, type Paint, rect, type Scene, type SceneNode, scene, text } from 'omai-engine/scene'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { contours } from '../charts/fields.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { type PointSet, points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { formatNumber, formatPercent } from '../theme/format.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { outlined, short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'
import {
  classIndices,
  classPointSets,
  decisionSurfaceLayers,
  type Rows,
  surfaceAxes,
  surfaceBoxAspect,
} from './surfaces.ts'

/** Options of {@link ensembleFrames}. */
export interface EnsembleViewOptions extends ViewOptions {
  /** Training rows to draw instead of the record's, to show what each stage's member was fitted to over them. */
  readonly points?: Rows
  readonly labels?: readonly (number | string)[]
  readonly columns?: readonly [number, number]
  /** The y label of the stage scores (default the record's score name, else 'score'). */
  readonly scoreLabel?: string
  /** A decision surface to draw instead of the record's. */
  readonly surface?: DecisionSurface
}

/** Whether a stage's targets are sample weights (non-negative, summing to about one) rather than residuals. */
function areWeights(targets: readonly number[]): boolean {
  const sum = targets.reduce((s, v) => s + v, 0)
  return targets.every((v) => v >= 0) && Math.abs(sum - 1) < 1e-3
}

/**
 * An ensemble built stage by stage as frames for a player: the training (and validation or out-of-bag) score after each
 * stage with the current stage marked, the members' weights with the current one highlighted, and the model so far:
 * when the record carries its decision surface through the stages, the regions after each recorded stage with the rows
 * sized by what the newest member was fitted to (AdaBoost's sample weights, boosting's residuals) and the misclassified
 * ones ringed; else, given the training rows, the sample weights as point sizes or the residuals as colours.
 *
 * @param stages - The record.
 * @param options - Mode, figure, training rows, labels and columns, the score label and a surface.
 * @returns The frames.
 */
export function ensembleFrames(stages: EnsembleStages, options: EnsembleViewOptions = {}): ViewFrames {
  const surface = options.surface ?? stages.surface
  const recorded = surface?.steps !== undefined && surface.steps.length > 1 ? surface.steps : undefined
  const rows = options.points ?? surface?.rows?.points
  const withSurface = surface !== undefined && surface.y.length >= 2 && rows !== undefined
  const withRows = withSurface || (rows !== undefined && stages.targets !== undefined)
  const fig = viewFigure(options, { aspect: withRows ? 1.75 : 1.6 })
  const mode = fig.mode
  const total = stages.trainScores.length
  const count = recorded?.length ?? total
  // The member index (0-based) a frame shows: the recorded step's stage count, or the frame itself.
  const stageOf = (frame: number): number => {
    const f = Math.min(Math.max(frame, 0), count - 1)
    return recorded === undefined ? f : Math.min(Math.max((recorded[f]?.iteration ?? 1) - 1, 0), total - 1)
  }
  const x = stages.trainScores.map((_, i) => i + 1)
  const constant = stages.weights.every((w) => w === stages.weights[0])
  const scoreLabel = options.scoreLabel ?? stages.scoreName ?? 'score'
  const option = (frame: number): EChartsOption => {
    const f = Math.min(Math.max(frame, 0), count - 1)
    const t = stageOf(f)
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
        options: { x: { label: 'stage' }, y: { label: scoreLabel } },
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
        options: { x: { label: 'member', show: total <= 30 }, y: { label: 'member weight' } },
      }
    if (withSurface && surface !== undefined) {
      const step = recorded?.[f]
      const at = surfaceAt(surface, f)
      const {
        layers,
        x: xa,
        y: ya,
      } = decisionSurfaceLayers(at, {
        mode,
        ...(options.points === undefined ? {} : { points: options.points }),
        ...(options.labels === undefined ? {} : { labels: options.labels }),
        ...(options.columns === undefined ? {} : { columns: options.columns }),
        ...(step?.highlight === undefined ? {} : { highlight: step.highlight }),
        ...(step?.rowWeights === undefined ? {} : { rowWeights: step.rowWeights }),
      })
      const wrong = step?.highlight?.length
      panels['rows'] = {
        layers,
        options: {
          ...surfaceAxes(surface, { x: xa, y: ya }),
          title: `after ${t + 1} member${t === 0 ? '' : 's'}${
            wrong === undefined ? '' : ` · ${wrong} ${surface.highlightName ?? 'highlighted'}`
          }`,
        },
      }
    } else if (withRows && rows !== undefined) {
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
    return subplots(grid, panels, { figure: fig, tooltip: 'item', ...(withSurface ? { legend: false } : {}) })
  }
  return {
    count,
    label: (f) => {
      const t = stageOf(f)
      return `stage ${t + 1}/${total} · ${short(stages.trainScores[t] ?? 0)}`
    },
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
 * import { pipelineStages } from 'omai-engine/exposition'
 * import { pipelineScene } from 'omai-render/views'
 * pipelineScene(pipelineStages({ stages: [] })).kind // => 'scene'
 * ```
 */
export function pipelineScene(record: PipelineStages): Scene {
  const stages = record.stages
  const learnedLines = Math.min(4, Math.max(0, maxOf(stages.map((s) => s.learned.length))))
  const sampleRows = Math.min(4, Math.max(0, maxOf(stages.map((s) => s.outputSample.length))))
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

/** One step of a pipeline as its view shows it: the data the pipeline was fitted to, or a stage. */
export interface PipelineStep {
  readonly name: string
  /** `input` for the data the pipeline took, else the stage's kind. */
  readonly kind: 'input' | PipelineStage['kind']
  /** The step's data: the input, or the stage's output. */
  readonly data?: StageData
  /** The data the step took (the previous step's), for comparing scales. */
  readonly before?: StageData
  readonly stage?: PipelineStage
}

/** How a pipeline step's output is best shown. */
export type PipelinePanel = 'scatter' | 'table' | 'projection' | 'surface'

/**
 * The steps of a pipeline view: the input (when the record carries it), then each stage with the data it took.
 *
 * @param record - The record.
 * @returns The steps, in order.
 * @example
 * ```ts
 * import { pipelineStages } from 'omai-engine/exposition'
 * import { pipelineSteps } from 'omai-render/views'
 * pipelineSteps(pipelineStages({ stages: [] })).length // => 0
 * ```
 */
export function pipelineSteps(record: PipelineStages): PipelineStep[] {
  const out: PipelineStep[] = []
  if (record.input !== undefined) out.push({ name: 'input', kind: 'input', data: record.input })
  let before = record.input
  for (const stage of record.stages) {
    out.push({
      name: stage.name,
      kind: stage.kind,
      stage,
      ...(stage.output === undefined ? {} : { data: stage.output }),
      ...(before === undefined ? {} : { before }),
    })
    before = stage.output
  }
  return out
}

/**
 * The panel a pipeline step's output suits: the final estimator's surface over the raw inputs when it has one, a
 * projection's first two components (with its explained variance), two columns as a scatter coloured by class, and
 * anything wider (or a single column, or an estimator's outputs) as a table with its column statistics.
 *
 * @param step - The step.
 * @returns The panel.
 * @example
 * ```ts
 * import { pipelinePanelOf } from 'omai-render/views'
 * pipelinePanelOf({ name: 'input', kind: 'input', data: { columns: ['a', 'b'], rows: [], n: 0 } }) // => 'scatter'
 * pipelinePanelOf({ name: 'poly', kind: 'transformer', data: { columns: ['a', 'b', 'a^2'], rows: [], n: 0 } }) // => 'table'
 * ```
 */
export function pipelinePanelOf(step: PipelineStep): PipelinePanel {
  const stage = step.stage
  if (stage?.surface !== undefined && stage.surface.y.length >= 2) return 'surface'
  if (step.kind === 'estimator') return 'table'
  const width = step.data?.columns.length ?? stage?.outputShape[1] ?? 0
  if (stage?.explainedVarianceRatio !== undefined && width >= 2) return 'projection'
  return width === 2 ? 'scatter' : 'table'
}

/** Options of the pipeline step views. */
export interface PipelineStepViewOptions extends ViewOptions {
  /** Class names, by class index (default the record's). */
  readonly classes?: readonly string[]
}

/** A step's first two columns as points, coloured by class (or by target). */
function stepPoints(data: StageData, classes: readonly string[] | undefined, mode: ColourMode): Layer {
  if (data.labels === undefined && data.targets !== undefined)
    return points(
      [
        {
          x: data.rows.map((r) => r[0] ?? 0),
          y: data.rows.map((r) => r[1] ?? 0),
          values: data.targets,
          name: '__target',
          size: 7,
        },
      ],
      { mode },
    )
  return outlined(points(classPointSets(data.rows, [0, 1], data.labels, classes), { mode }), mode, 0.6)
}

/** An axis label for a column, with its share of the variance when it is a projection's component. */
function columnLabel(data: StageData, j: number, ratio?: readonly number[]): string {
  const name = data.columns[j] ?? `x${j}`
  const share = ratio?.[j]
  return share === undefined ? name : `${name} (${formatPercent(share)} of variance)`
}

/**
 * A pipeline step's two output columns as a scatter coloured by class: the raw inputs, or any stage that gives two
 * columns.
 *
 * @param step - The step (with its data).
 * @param options - Mode, figure and class names.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { pipelineScatterView } from 'omai-render/views'
 * const step = { name: 'input', kind: 'input', data: { columns: ['a', 'b'], rows: [[0, 1], [1, 0]], n: 2, labels: [0, 1] } }
 * pipelineScatterView(step).series.length // => 2
 * ```
 */
export function pipelineScatterView(step: PipelineStep, options: PipelineStepViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.35 })
  const data = step.data ?? { columns: [], rows: [], n: 0 }
  return cartesian([stepPoints(data, options.classes, fig.mode)], {
    figure: fig,
    mode: fig.mode,
    x: { label: columnLabel(data, 0) },
    y: { label: columnLabel(data, 1) },
    title: step.kind === 'input' ? `the input: ${data.n} rows` : `${step.name}'s output: ${data.n} rows`,
    tooltip: 'item',
  })
}

/**
 * A projection stage (PCA, truncated SVD): the rows on its first two components, coloured by class, beside each
 * component's share of the variance with the cumulative share.
 *
 * @param step - The step (with its data and explained variance).
 * @param options - Mode, figure and class names.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { pipelineProjectionView } from 'omai-render/views'
 * const stage = { name: 'pca', kind: 'transformer', inputColumns: ['a', 'b', 'c'], outputColumns: ['p0', 'p1'], inputSample: [], outputSample: [], inputShape: [2, 3], outputShape: [2, 2], learned: [], explainedVarianceRatio: [0.8, 0.2] }
 * const data = { columns: ['p0', 'p1'], rows: [[0, 1], [1, 0]], n: 2 }
 * pipelineProjectionView({ name: 'pca', kind: 'transformer', stage, data }).grid.length // => 2
 * ```
 */
export function pipelineProjectionView(step: PipelineStep, options: PipelineStepViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2 })
  const mode = fig.mode
  const data = step.data ?? { columns: [], rows: [], n: 0 }
  const ratio = step.stage?.explainedVarianceRatio ?? []
  const names = ratio.map((_, i) => data.columns[i] ?? `${i + 1}`)
  let acc = 0
  const cumulative = ratio.map((r) => (acc += r))
  const panels: Record<string, Panel> = {
    scores: {
      layers: [stepPoints(data, options.classes, mode)],
      options: {
        x: { label: columnLabel(data, 0, ratio) },
        y: { label: columnLabel(data, 1, ratio) },
        title: `rows on the first two of ${data.columns.length} components`,
      },
    },
    scree: {
      layers: [
        bars(names, [{ name: 'share', values: [...ratio], colour: 'tab0' }], { mode }),
        lines(
          [{ x: names.map((_, i) => i), y: cumulative, name: 'cumulative', colour: 'fgSecondary', markers: true }],
          {
            mode,
          },
        ),
      ],
      options: {
        x: { label: 'component' },
        y: { label: 'variance explained', format: 'percent', limits: [0, 1] },
        title: `${formatPercent(cumulative[cumulative.length - 1] ?? 0)} of the variance kept`,
      },
    },
  }
  const grid = gridSpec(1, 2, {
    widthRatios: [1.5, 1],
    wspace: 72,
    cells: [
      { id: 'scores', row: 0, col: 0 },
      { id: 'scree', row: 0, col: 1 },
    ],
  })
  return subplots(grid, panels, { figure: fig, tooltip: 'item', legend: false })
}

/** What the final estimator's panel shades: its predicted probability, or the loss it expects (with each row's loss). */
export type PipelineSurfaceField = 'probability' | 'loss'

/** Options of {@link pipelineSurfaceView}. */
export interface PipelineSurfaceOptions extends PipelineStepViewOptions {
  /** What to shade (default the probability). */
  readonly field?: PipelineSurfaceField
}

/**
 * The final estimator's panel: the whole pipeline's predictions over the original two features with the training
 * rows. Shading the probability, the classes' regions (faded where the model is unsure) with the 0.5 boundary and the
 * 0.25 and 0.75 contours; shading the loss, the log loss the model expects at each point (its predicted
 * distribution's entropy, highest along the boundary) with the boundary drawn over it and each row sized by its own
 * log loss. Either way the misclassified rows are ringed.
 *
 * @param step - The step (a stage with a surface, and a loss for the loss field).
 * @param options - Mode, figure, class names and the field.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { decisionSurface } from 'omai-engine/exposition'
 * import { pipelineSurfaceView } from 'omai-render/views'
 * const surface = decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0, 1], probabilities: [[[0.9, 0.4], [0.6, 0.1]], [[0.1, 0.6], [0.4, 0.9]]] })
 * const stage = { name: 'model', kind: 'estimator', inputColumns: ['a', 'b'], outputColumns: ['p(0)', 'p(1)'], inputSample: [], outputSample: [], inputShape: [4, 2], outputShape: [4, 2], learned: [], surface }
 * pipelineSurfaceView({ name: 'model', kind: 'estimator', stage }).series.length > 0 // => true
 * ```
 */
export function pipelineSurfaceView(step: PipelineStep, options: PipelineSurfaceOptions = {}): EChartsOption {
  const surface = step.stage?.surface
  const fig = viewFigure(options, { aspect: surface === undefined ? 1.25 : surfaceBoxAspect(surface, 'equal', 1.25) })
  const mode = fig.mode
  if (surface === undefined) return cartesian([], { figure: fig, mode })
  const loss = step.stage?.loss
  const highlight = loss?.misclassified
  const field = options.field ?? 'probability'
  const p1 = surface.probabilities?.[1]
  let layers: Layer[]
  let title: string
  if (field === 'loss' && loss?.expected !== undefined) {
    // The loss the model expects from zero up (natural log: at most log 2 for two classes), the 0.5 boundary over it,
    // and each row sized by its own loss.
    const hi = Math.max(1e-12, maxOf(loss.expected.flat()))
    layers = [
      heatmap(loss.expected, {
        x: surface.x,
        y: surface.y,
        mode,
        range: [0, hi],
        colourBar: true,
        label: 'loss',
      }),
      contours(loss.expected, { x: surface.x, y: surface.y, levels: 5, colour: 'fg', width: 0.6, opacity: 0.45, mode }),
    ]
    if (p1 !== undefined)
      layers.push(
        contours(p1, { x: surface.x, y: surface.y, levels: [0.5], colour: 'fg', width: 1.6, labels: false, mode }),
      )
    const rows = surface.rows
    if (rows !== undefined)
      layers.push(
        outlined(
          points(classPointSets(rows.points, [0, 1], rows.labels, surface.classes ?? rows.classes, {}, loss.values), {
            mode,
          }),
          mode,
        ),
      )
    title = `expected ${loss.name}; rows sized by their own`
  } else {
    layers = [...decisionSurfaceLayers(surface, { mode, ...(highlight === undefined ? {} : { highlight }) }).layers]
    const positive = surface.classes?.[1]
    title = p1 === undefined ? 'predictions' : `p(${positive ?? 'class 1'}) over the raw inputs`
  }
  if (loss !== undefined)
    title += ` · mean ${loss.name} ${short(loss.mean)}${
      highlight === undefined ? '' : ` · ${highlight.length} of ${loss.values.length} misclassified`
    }`
  return cartesian(layers, {
    figure: fig,
    mode,
    ...surfaceAxes(surface, { x: { label: surface.features[0] }, y: { label: surface.features[1] } }),
    aspect: 'auto',
    title,
    tooltip: 'item',
    legend: false,
  })
}

/**
 * A step's card text: its shape change, and a line saying what it did (the rows and classes it took, the features it
 * made, the variance a projection kept, the training score and loss of the final estimator, else its first fitted
 * parameter).
 *
 * @param step - The step.
 * @param classes - Class names.
 * @returns The shape line and the summary line.
 * @example
 * ```ts
 * import { pipelineStepSummary } from 'omai-render/views'
 * pipelineStepSummary({ name: 'input', kind: 'input', data: { columns: ['a', 'b'], rows: [], n: 200 } }, ['0', '1'])
 * // => { shape: '200 × 2', detail: '2 classes' }
 * ```
 */
export function pipelineStepSummary(
  step: PipelineStep,
  classes?: readonly string[],
): { readonly shape: string; readonly detail: string } {
  const stage = step.stage
  if (stage === undefined) {
    const n = step.data?.n ?? 0
    const d = step.data?.columns.length ?? 0
    return { shape: `${n} × ${d}`, detail: classes === undefined ? `${d} features` : `${classes.length} classes` }
  }
  const [n0, d0] = stage.inputShape
  const [n1, d1] = stage.outputShape
  const shape = `${n0} × ${d0} → ${n1} × ${d1}`
  if (stage.loss !== undefined) {
    const score = stage.learned.find((l) => l.name === 'training score')?.value
    const scored = typeof score === 'number' ? `score ${short(score)} · ` : ''
    return { shape, detail: `${scored}${stage.loss.name} ${short(stage.loss.mean)}` }
  }
  const ratio = stage.explainedVarianceRatio
  if (ratio !== undefined)
    return { shape, detail: `keeps ${formatPercent(ratio.reduce((s, v) => s + v, 0))} of the variance` }
  if (stage.kind === 'resampler') return { shape, detail: `${n0} → ${n1} rows` }
  if (d1 > d0) return { shape, detail: `${d1 - d0} feature${d1 - d0 === 1 ? '' : 's'} made` }
  const stats = stage.output?.stats
  if (
    stats !== undefined &&
    stats.length > 0 &&
    stats.every((s) => Math.abs(s.mean) < 1e-6 && Math.abs(s.std - 1) < 1e-6)
  )
    return { shape, detail: 'every column: mean 0, std 1' }
  const first = stage.learned[0]
  return { shape, detail: first === undefined ? stage.kind : `${cell(first.name)}: ${cell(first.value)}` }
}

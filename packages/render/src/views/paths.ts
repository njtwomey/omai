/**
 * Views of paths and curves over a parameter: regularisation paths (each coefficient against the penalty, as for the
 * lasso, ridge or elastic net, with the active-set size beneath), cross-validation curves with standard-error bands and
 * the best and one-standard-error choices, learning curves, and cost-complexity pruning paths of trees.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { CrossValidationCurve, LearningCurve, PruningPath, RegularisationPath } from 'mdml-engine/exposition'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { type AxisSpec, cartesian, type Layer, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { limitsOf, nearestIndex, short, type ViewOptions, viewFigure, viewMode, wantsLog } from './_view.ts'

/** Options of {@link regularisationPathView}. */
export interface PathViewOptions extends ViewOptions {
  /** A penalty to mark (a control's value): a line across the panels and a dot on each coefficient there. */
  readonly selected?: number
}

function penaltyAxis(penalties: readonly number[], label: string): AxisSpec {
  const log = wantsLog(penalties)
  return { label, scale: log ? 'log' : 'linear', limits: limitsOf(penalties, log), inverse: true }
}

/** The coefficients at a penalty, interpolated between the path's penalties (in log penalty when the grid is). */
export function coefficientsAt(path: RegularisationPath, penalty: number): number[] {
  const p = path.penalties
  const log = wantsLog(p)
  const f = (v: number): number => (log ? Math.log(Math.max(v, 1e-300)) : v)
  const order = p.map((_, i) => i).sort((a, b) => (p[a] ?? 0) - (p[b] ?? 0))
  const at = f(penalty)
  for (let k = 0; k + 1 < order.length; k++) {
    const i = order[k] ?? 0
    const j = order[k + 1] ?? 0
    const a = f(p[i] ?? 0)
    const b = f(p[j] ?? 0)
    if (at >= a && at <= b) {
      const w = b > a ? (at - a) / (b - a) : 0
      return path.features.map(
        (_, c) => (1 - w) * (path.coefficients[i]?.[c] ?? 0) + w * (path.coefficients[j]?.[c] ?? 0),
      )
    }
  }
  return [...(path.coefficients[nearestIndex(p, penalty, log)] ?? path.features.map(() => 0))]
}

/** The number of non-zero coefficients at each penalty (the record's active sets when given). */
export function activeCounts(path: RegularisationPath): number[] {
  return path.penalties.map(
    (_, i) => path.active?.[i]?.length ?? (path.coefficients[i] ?? []).filter((c) => Math.abs(c) > 1e-12).length,
  )
}

/**
 * The panels of a regularisation path: every coefficient against the penalty (labelled at its unpenalised end), and
 * beneath it the size of the active set, the penalty axis decreasing to the right (log when the grid spans decades),
 * the chosen penalty dashed and a selected penalty marked with the coefficients there.
 *
 * @param path - The record.
 * @param options - Mode and the selected penalty.
 * @returns The coefficient and active-set panels.
 */
export function regularisationPathPanels(
  path: RegularisationPath,
  options: PathViewOptions = {},
): { readonly coefficients: Panel; readonly active: Panel } {
  const mode = viewMode(options)
  const x = penaltyAxis(path.penalties, 'penalty')
  const curves = lines(
    path.features.map((name, j) => ({
      x: path.penalties,
      y: path.coefficients.map((row) => row[j] ?? 0),
      name,
      width: 1.6,
    })),
    { mode },
  )
  // Each coefficient named at its least penalised end, nudged apart where names would overlap.
  const last = nearestIndex(path.penalties, Math.min(...path.penalties))
  const names = layer(
    [
      {
        type: 'scatter',
        name: '__names',
        data: path.features.map((name, j) => ({
          value: [path.penalties[last] ?? 0, path.coefficients[last]?.[j] ?? 0],
          label: { formatter: name, color: (curves.series[j] as { color?: string }).color },
        })),
        symbolSize: 0,
        silent: true,
        clip: false,
        label: { show: true, position: 'right', distance: 4, fontSize: 10 },
        labelLayout: { moveOverlap: 'shiftY' },
        tooltip: { show: false },
      } as SeriesOption,
    ],
    {},
  )
  const marks: Layer[] = []
  if (path.chosen !== undefined)
    marks.push(referenceLines({ x: [{ value: path.chosen, label: 'chosen' }] }, { mode, colour: 'fgSecondary' }))
  const selected: Layer[] = []
  if (options.selected !== undefined) {
    const at = coefficientsAt(path, options.selected)
    selected.push(referenceLines({ x: [options.selected] }, { mode, colour: 'accent', dash: 'solid' }))
    selected.push(
      points(
        [
          {
            x: at.map(() => options.selected ?? 0),
            y: at,
            groups: at.map((_, j) => j),
            groupNames: Object.fromEntries(path.features.map((f, j) => [String(j), f])),
            size: 8,
          },
        ],
        { mode },
      ),
    )
  }
  const zero = referenceLines({ y: [0] }, { mode, colour: 'grid', dash: 'solid' })
  const counts = activeCounts(path)
  const active = lines([{ x: path.penalties, y: counts, name: 'active', colour: 'fgSecondary', step: 'middle' }], {
    mode,
  })
  return {
    coefficients: {
      layers: [zero, ...marks, curves, names, ...selected],
      options: { x, y: { label: 'coefficient' } },
    },
    active: {
      layers: [
        ...(options.selected === undefined
          ? []
          : [referenceLines({ x: [options.selected] }, { mode, colour: 'accent', dash: 'solid' })]),
        ...(path.chosen === undefined ? [] : [referenceLines({ x: [path.chosen] }, { mode, colour: 'fgSecondary' })]),
        active,
      ],
      options: { x, y: { label: 'active', limits: [0, Math.max(1, ...counts) + 0.5] } },
    },
  }
}

/**
 * A regularisation path as one figure (lasso, ridge, elastic net, GAM smoothness sweeps): the coefficients against the
 * penalty above, the active-set size below, on a shared penalty axis.
 *
 * @param path - The record.
 * @param options - Mode, figure and the selected penalty.
 * @returns The ECharts option.
 * @index function Coefficients against the penalty, lasso style: `regularisationPathView(path)`.
 * @example
 * ```ts
 * import { regularisationPath } from 'mdml-engine/exposition'
 * import { regularisationPathView } from 'mdml-render/views'
 * const path = regularisationPath({ penalties: [1, 0.1, 0.01], features: ['a', 'b'], coefficients: [[0, 0], [0.4, 0], [0.9, -0.2]] })
 * regularisationPathView(path).grid.length // => 2
 * ```
 */
export function regularisationPathView(path: RegularisationPath, options: PathViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.45, padding: { right: 72 } })
  const panels = regularisationPathPanels(path, { ...options, mode: fig.mode })
  const grid = gridSpec(2, 1, {
    heightRatios: [3, 1],
    hspace: 14,
    cells: [
      { id: 'coefficients', row: 0, col: 0 },
      { id: 'active', row: 1, col: 0, sharex: 'coefficients' },
    ],
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

/** Options of {@link crossValidationView}. */
export interface CrossValidationViewOptions extends ViewOptions {
  /** A parameter value to mark (a control's value). */
  readonly selected?: number
  /**
   * Reverse the parameter axis (to match a regularisation path, whose penalty decreases to the right); default the
   * record's `penalty`.
   */
  readonly inverse?: boolean
  /** The y axis label (default 'validation score', or 'validation error' when lower is better). */
  readonly scoreLabel?: string
}

/**
 * The panel of a cross-validation curve: the mean score per parameter value with a band of one standard error, the
 * folds faintly behind it, the best value and the one-standard-error choice marked, and the one-standard-error
 * threshold as a horizontal line.
 *
 * @param curve - The record.
 * @param options - Mode, selected value, axis direction and label.
 * @returns The panel.
 */
export function crossValidationPanel(curve: CrossValidationCurve, options: CrossValidationViewOptions = {}): Panel {
  const mode = viewMode(options)
  const v = curve.values
  const log = wantsLog(v)
  const layers: Layer[] = []
  if (curve.foldScores !== undefined) {
    const folds = curve.foldScores[0]?.length ?? 0
    layers.push(
      lines(
        Array.from({ length: folds }, (_, f) => ({
          x: v,
          y: curve.foldScores?.map((row) => row[f] ?? Number.NaN) ?? [],
          name: `__fold${f}`,
          colour: 'muted',
          width: 0.8,
          opacity: 0.45,
        })),
        { mode },
      ),
    )
  }
  layers.push(
    band(
      [
        {
          x: v,
          lower: curve.meanScores.map((m, i) => m - (curve.standardErrors[i] ?? 0)),
          upper: curve.meanScores.map((m, i) => m + (curve.standardErrors[i] ?? 0)),
          colour: 'tab0',
          name: '__se',
          opacity: 0.28,
        },
      ],
      { mode },
    ),
  )
  layers.push(lines([{ x: v, y: curve.meanScores, name: 'mean', colour: 'tab0', markers: true }], { mode }))
  const marks: { value: number; label: string }[] = []
  if (curve.best !== undefined) marks.push({ value: curve.best, label: 'best' })
  if (curve.oneStandardError !== undefined && curve.oneStandardError !== curve.best)
    marks.push({ value: curve.oneStandardError, label: '1 SE' })
  if (marks.length > 0) layers.push(referenceLines({ x: marks }, { mode, colour: 'fgSecondary' }))
  if (curve.best !== undefined) {
    const b = nearestIndex(v, curve.best, log)
    const m = curve.meanScores[b] ?? 0
    const se = curve.standardErrors[b] ?? 0
    layers.push(
      referenceLines({ y: [curve.higherIsBetter ? m - se : m + se] }, { mode, colour: 'muted', dash: 'dotted' }),
      points([{ x: [v[b] ?? 0], y: [m], name: 'best', colour: 'tab0', size: 10 }], { mode }),
    )
  }
  if (options.selected !== undefined)
    layers.push(referenceLines({ x: [options.selected] }, { mode, colour: 'accent', dash: 'solid' }))
  return {
    layers,
    options: {
      x: {
        label: curve.parameter,
        scale: log ? 'log' : 'linear',
        limits: limitsOf(v, log),
        inverse: options.inverse ?? curve.penalty === true,
      },
      y: {
        label:
          options.scoreLabel ?? curve.scoreName ?? (curve.higherIsBetter ? 'validation score' : 'validation error'),
      },
    },
  }
}

/**
 * A cross-validation curve as one figure: mean scores with their standard-error band, folds, and the best and
 * one-standard-error choices.
 *
 * @param curve - The record.
 * @param options - Mode, figure, selected value, axis direction and label.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { crossValidationCurve } from 'mdml-engine/exposition'
 * import { crossValidationView } from 'mdml-render/views'
 * const curve = crossValidationCurve({ parameter: 'k', values: [1, 2, 3], meanScores: [0.7, 0.8, 0.75], standardErrors: [0.05, 0.04, 0.05], higherIsBetter: true, best: 2 })
 * crossValidationView(curve).xAxis[0].name // => 'k'
 * ```
 */
export function crossValidationView(
  curve: CrossValidationCurve,
  options: CrossValidationViewOptions = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const panel = crossValidationPanel(curve, { ...options, mode: fig.mode })
  return cartesian(panel.layers, { ...panel.options, figure: fig, mode: fig.mode, legend: false, tooltip: 'item' })
}

/**
 * The panel of a learning curve: training and validation scores against the training size (or another resource), each
 * with its spread as a band.
 *
 * @param curve - The record.
 * @param options - Mode.
 * @returns The panel.
 */
export function learningCurvePanel(curve: LearningCurve, options: ViewOptions = {}): Panel {
  const mode = viewMode(options)
  const layers: Layer[] = []
  const spread = (mean: readonly number[], s: readonly number[] | undefined, colour: string): void => {
    if (s === undefined) return
    layers.push(
      band(
        [
          {
            x: curve.sizes,
            lower: mean.map((m, i) => m - (s[i] ?? 0)),
            upper: mean.map((m, i) => m + (s[i] ?? 0)),
            colour,
          },
        ],
        { mode },
      ),
    )
  }
  // An empty validation column is a single series; a secondary axis takes the validation column's own scale.
  const single = curve.validation.length === 0
  const second = !single && curve.secondaryAxis === true
  const markers = curve.sizes.length <= 40
  spread(curve.train, curve.trainSpread, 'tab0')
  if (!single && !second) spread(curve.validation, curve.validationSpread, 'tab1')
  layers.push(
    lines([{ x: curve.sizes, y: curve.train, name: curve.labels?.train ?? 'train', colour: 'tab0', markers }], {
      mode,
    }),
  )
  if (!single) {
    const validation = lines(
      [
        {
          x: curve.sizes,
          y: curve.validation,
          name: curve.labels?.validation ?? 'validation',
          colour: 'tab1',
          markers,
        },
      ],
      { mode },
    )
    layers.push(second ? { ...validation, secondary: true } : validation)
  }
  return {
    layers,
    options: {
      x: { label: curve.resource ?? 'training size', scale: wantsLog(curve.sizes) ? 'log' : 'linear' },
      y: {
        label:
          curve.validation.length > 0 && curve.secondaryAxis === true
            ? (curve.labels?.train ?? 'train')
            : (curve.scoreName ?? 'score'),
      },
      ...(curve.validation.length > 0 && curve.secondaryAxis === true
        ? { y2: { label: curve.labels?.validation ?? 'validation' } }
        : {}),
    },
  }
}

/**
 * A learning curve as one figure.
 *
 * @param curve - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 */
export function learningCurveView(curve: LearningCurve, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const panel = learningCurvePanel(curve, { mode: fig.mode })
  return cartesian(panel.layers, { ...panel.options, figure: fig, mode: fig.mode, tooltip: 'axis' })
}

/** Options of {@link pruningPathView}. */
export interface PruningViewOptions extends ViewOptions {
  /** An alpha to mark (a control's value). */
  readonly selected?: number
}

/**
 * A cost-complexity pruning path: the total leaf impurity of the pruned tree against the effective alpha (a step
 * up at each prune), and the number of leaves on the right-hand axis.
 *
 * @param path - The record.
 * @param options - Mode, figure and the selected alpha.
 * @returns The ECharts option.
 */
export function pruningPathView(path: PruningPath, options: PruningViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const impurity = lines(
    [{ x: path.alphas, y: path.impurities, name: 'impurity', colour: 'tab0', step: 'end', markers: true }],
    {
      mode,
    },
  )
  const leaves = lines(
    [{ x: path.alphas, y: path.leaves, name: 'leaves', colour: 'tab1', step: 'end', dash: 'dashed' }],
    {
      mode,
    },
  )
  const layers: Layer[] = [impurity, { ...leaves, secondary: true }]
  if (options.selected !== undefined)
    layers.push(
      referenceLines(
        { x: [{ value: options.selected, label: `α = ${short(options.selected)}` }] },
        { mode, colour: 'accent', dash: 'solid' },
      ),
    )
  return cartesian(layers, {
    figure: fig,
    mode,
    x: { label: 'effective alpha' },
    y: { label: 'total leaf impurity' },
    y2: { label: 'leaves' },
    tooltip: 'axis',
  })
}

/**
 * Views of a threshold sweep (`thresholdSweep`): each class's score distribution with the threshold, the total cost
 * against the threshold with the cheapest, and the ROC and precision-recall curves with the operating point. Every
 * panel takes the same threshold, so a page can move one threshold through all of them.
 */

import type { EChartsOption } from 'echarts'
import { sweepAt, type ThresholdSweep } from 'whitebox-engine/exposition'
import { histogram } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { cartesian } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import type { Figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Options of the threshold panels and view. */
export interface ThresholdViewOptions extends ViewOptions {
  /** The threshold to show (default the cheapest when the sweep has costs, else the one with the best accuracy). */
  readonly threshold?: number
  /** The score axis (default `[0, 1]` for probabilities, else the scores' span, padded). */
  readonly range?: readonly [number, number]
  /** Histogram bins (default 30). */
  readonly bins?: number
}

/** The colour of the operating point and the threshold. */
const OPERATING = 'tab3'

/**
 * The threshold a sweep is shown at by default: the cheapest when it has costs, else the most accurate (the lowest of
 * any ties).
 *
 * @param record - The sweep.
 * @returns The threshold.
 */
export function defaultThreshold(record: ThresholdSweep): number {
  let i = record.cheapest
  if (i === undefined) {
    i = 0
    for (const [j, a] of record.accuracy.entries()) if (a > (record.accuracy[i] ?? 0)) i = j
  }
  return record.thresholds[i] ?? 0.5
}

/**
 * The score axis of a sweep: `[0, 1]` when every score is a probability, else the span of the scores padded by 5%.
 *
 * @param record - The sweep.
 * @returns The low and high of the score axis.
 */
export function scoreRange(record: ThresholdSweep): [number, number] {
  const finite = record.thresholds.filter(Number.isFinite)
  if (finite.length === 0) return [0, 1]
  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  if (lo >= 0 && hi <= 1) return [0, 1]
  const pad = Math.max((hi - lo) * 0.05, 1e-6)
  return [lo - pad, hi + pad]
}

/** The tallest bin of either class's histogram over a range, with headroom: a y limit that does not move. */
function histogramTop(record: ThresholdSweep, range: readonly [number, number], bins: number): number {
  const rows = record.scored
  if (rows === undefined) return 1
  const counts = [new Array<number>(bins).fill(0), new Array<number>(bins).fill(0)]
  const width = range[1] - range[0]
  for (const [i, s] of rows.scores.entries()) {
    if (s < range[0] || s > range[1]) continue
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((s - range[0]) / width) * bins)))
    const row = counts[rows.labels[i] === 1 ? 1 : 0]
    if (row !== undefined) row[b] = (row[b] ?? 0) + 1
  }
  const top = Math.max(1, ...counts.flat()) * 1.15
  return top > 10 ? Math.ceil(top / 5) * 5 : Math.ceil(top)
}

/**
 * Each class's score distribution (overlaid histograms) with the threshold as a line; the y axis is fixed by the
 * histograms so the panel does not move while a threshold is dragged over it.
 *
 * @param record - The sweep (its `scored` rows are drawn; without them only the threshold is).
 * @param mode - The colour mode.
 * @param options - Threshold, range and bins.
 * @returns The panel, and its y limit (the span of a dragged line).
 */
export function thresholdScoresPanel(
  record: ThresholdSweep,
  mode: ColourMode,
  options: Omit<ThresholdViewOptions, 'mode' | 'figure'> = {},
): Panel & { readonly top: number } {
  const range = options.range ?? scoreRange(record)
  const bins = options.bins ?? 30
  const threshold = options.threshold ?? defaultThreshold(record)
  const rows = record.scored
  const of = (c: number): number[] =>
    rows === undefined ? [] : rows.scores.filter((_, i) => (rows.labels[i] === 1 ? 1 : 0) === c)
  const top = histogramTop(record, range, bins)
  return {
    top,
    layers: [
      ...(rows === undefined
        ? []
        : [
            histogram(of(0), { bins, range: [range[0], range[1]], name: record.classes[0], opacity: 0.6, mode }),
            histogram(of(1), {
              bins,
              range: [range[0], range[1]],
              name: record.classes[1],
              opacity: 0.6,
              colourOffset: 1,
              mode,
            }),
          ]),
      referenceLines({ x: [threshold] }, { mode, colour: OPERATING, dash: 'solid' }),
    ],
    options: {
      title: 'scores by class',
      x: { label: 'score', limits: [range[0], range[1]] },
      y: { label: 'count', limits: [0, top] },
    },
  }
}

/**
 * The total cost against the threshold (a step: the cost of every threshold between two scores is the cost at the
 * upper one), the cheapest threshold marked, and the current threshold's cost as a point.
 *
 * @param record - A sweep with costs (without them the panel is empty).
 * @param mode - The colour mode.
 * @param options - Threshold and range.
 * @returns The panel, and its y limit.
 */
export function thresholdCostPanel(
  record: ThresholdSweep,
  mode: ColourMode,
  options: Omit<ThresholdViewOptions, 'mode' | 'figure' | 'bins'> = {},
): Panel & { readonly top: number } {
  const range = options.range ?? scoreRange(record)
  const cost = record.cost
  if (cost === undefined)
    return { top: 1, layers: [], options: { title: 'no costs given', x: { label: 'threshold', limits: [...range] } } }
  const threshold = options.threshold ?? defaultThreshold(record)
  const last = cost.length - 1
  const xs: number[] = [range[0]]
  const ys: number[] = [cost[0] ?? 0]
  for (const [i, t] of record.thresholds.entries())
    if (Number.isFinite(t) && t >= range[0] && t <= range[1]) {
      xs.push(t)
      ys.push(cost[i] ?? 0)
    }
  xs.push(range[1])
  ys.push(cost[last] ?? 0)
  const lo = Math.min(0, ...cost)
  const hi = Math.max(...cost)
  const top = hi + (hi - lo) * 0.08 || 1
  const at = sweepAt(record, threshold)
  const best = record.cheapest ?? 0
  const bestX = Math.min(range[1], record.thresholds[best] ?? range[1])
  return {
    top,
    layers: [
      lines([{ x: xs, y: ys, name: 'total cost', step: 'start' }], { mode }),
      referenceLines({ x: [{ value: bestX, label: 'cheapest' }] }, { mode, colour: 'success', dash: 'dotted' }),
      points(
        [
          {
            x: [Math.min(range[1], Math.max(range[0], threshold))],
            y: [at.cost ?? 0],
            name: 'at the threshold',
            colour: OPERATING,
            size: 11,
          },
        ],
        { mode },
      ),
    ],
    options: {
      title: 'total cost',
      x: { label: 'threshold', limits: [range[0], range[1]] },
      y: { label: 'cost', limits: [lo, top] },
    },
  }
}

/**
 * The ROC curve (true- against false-positive rate at every threshold) with the chance diagonal and the operating point.
 *
 * @param record - The sweep.
 * @param mode - The colour mode.
 * @param threshold - The threshold of the operating point (default the sweep's default).
 * @returns The panel.
 */
export function thresholdRocPanel(record: ThresholdSweep, mode: ColourMode, threshold?: number): Panel {
  const at = sweepAt(record, threshold ?? defaultThreshold(record))
  return {
    layers: [
      lines([{ x: [0, 1], y: [0, 1], name: '__chance', colour: 'muted', dash: 'dotted' }], { mode }),
      lines([{ x: record.fpr, y: record.tpr, name: 'ROC' }], { mode }),
      points([{ x: [at.fpr], y: [at.tpr], name: 'operating point', colour: OPERATING, size: 12 }], { mode }),
    ],
    options: {
      title: `ROC (AUC ${Number.isFinite(record.rocAuc) ? record.rocAuc.toFixed(3) : '–'})`,
      x: { label: 'false-positive rate', limits: [0, 1] },
      y: { label: 'true-positive rate', limits: [0, 1] },
    },
  }
}

/**
 * The precision-recall curve with the prevalence (the precision of predicting everyone positive) and the operating
 * point; thresholds that predict nobody positive have no precision and are left out.
 *
 * @param record - The sweep.
 * @param mode - The colour mode.
 * @param threshold - The threshold of the operating point (default the sweep's default).
 * @returns The panel.
 */
export function thresholdPrPanel(record: ThresholdSweep, mode: ColourMode, threshold?: number): Panel {
  const at = sweepAt(record, threshold ?? defaultThreshold(record))
  const keep = record.precision.flatMap((p, i) => (Number.isFinite(p) ? [i] : []))
  const prevalence = record.positives / (record.positives + record.negatives || 1)
  const ap = record.averagePrecision
  return {
    layers: [
      referenceLines({ y: [{ value: prevalence, label: 'prevalence' }] }, { mode, dash: 'dotted' }),
      lines([{ x: keep.map((i) => record.tpr[i] ?? 0), y: keep.map((i) => record.precision[i] ?? 0), name: 'PR' }], {
        mode,
      }),
      ...(Number.isFinite(at.precision)
        ? [points([{ x: [at.tpr], y: [at.precision], name: 'operating point', colour: OPERATING, size: 12 }], { mode })]
        : []),
    ],
    options: {
      title: `precision-recall (AP ${Number.isFinite(ap) ? ap.toFixed(3) : '–'})`,
      x: { label: 'recall (TPR)', limits: [0, 1] },
      y: { label: 'precision', limits: [0, 1] },
    },
  }
}

/**
 * One panel of a sweep as a whole chart, at a figure's size: for pages that lay the panels out themselves (one chart per
 * panel, each with its own dragged line).
 *
 * @param panel - A panel from the functions above.
 * @param figure - The figure.
 * @returns The ECharts option.
 */
export function thresholdPanelOption(panel: Panel, figure: Figure): EChartsOption {
  return cartesian(panel.layers, { ...panel.options, figure, mode: figure.mode, tooltip: 'item' })
}

/**
 * A threshold sweep as one figure: the score distributions with the threshold, the cost against the threshold (when
 * the sweep has costs), and the ROC and precision-recall curves with the operating point, all at one threshold.
 *
 * @param record - The sweep.
 * @param options - Mode, figure, threshold, range and bins.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { analyseThresholds } from 'whitebox-engine/metrics'
 * import { thresholdSweepView } from 'whitebox-render/views'
 * const r = analyseThresholds([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], { costs: [[0, 1], [5, 0]] })
 * thresholdSweepView(r, { threshold: 0.5 }).grid.length // => 4
 * ```
 */
export function thresholdSweepView(record: ThresholdSweep, options: ThresholdViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.5 })
  const mode = fig.mode
  const threshold = options.threshold ?? defaultThreshold(record)
  const panelOptions = { ...options, threshold }
  const costs = record.cost !== undefined
  const grid = costs
    ? gridSpec(2, 2, { wspace: 64, hspace: 72 })
    : gridSpec(2, 2, {
        wspace: 64,
        hspace: 72,
        cells: [
          { id: 'scores', row: 0, col: 0, colSpan: 2 },
          { id: 'roc', row: 1, col: 0 },
          { id: 'pr', row: 1, col: 1 },
        ],
      })
  const scores = thresholdScoresPanel(record, mode, panelOptions)
  const roc = thresholdRocPanel(record, mode, threshold)
  const pr = thresholdPrPanel(record, mode, threshold)
  const panels = costs
    ? { r0c0: scores, r0c1: thresholdCostPanel(record, mode, panelOptions), r1c0: roc, r1c1: pr }
    : { scores, roc, pr }
  return subplots(grid, panels, { figure: fig, tooltip: 'item' })
}

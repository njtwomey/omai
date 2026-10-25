/**
 * Views of a group fairness report: each group's rates (selection, true- and false-positive rates) side by side, each
 * group's gap to the reference group, the groups' ROC curves with their operating points, and the rates over a shared
 * threshold sweep.
 */

import type { EChartsOption } from 'echarts'
import { GROUP_RATES, type GroupRateName, type GroupReport, groupReportAt } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { cartesian } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** Short names of the group rates, for axes and tables. */
export const GROUP_RATE_LABELS: Readonly<Record<GroupRateName, string>> = {
  selectionRate: 'selection',
  truePositiveRate: 'TPR',
  falsePositiveRate: 'FPR',
  precision: 'precision',
  accuracy: 'accuracy',
  baseRate: 'base rate',
}

/** Options of the group report views. */
export interface GroupReportViewOptions extends ViewOptions {
  /** A threshold per group to show the report at instead of its own (recomputed from its rows). */
  readonly thresholds?: readonly number[]
  /** The rates to show (default all of them). */
  readonly rates?: readonly GroupRateName[]
}

/** The report at the asked thresholds, or as it is. */
export function reportAt(record: GroupReport, thresholds?: readonly number[]): GroupReport {
  if (thresholds === undefined || record.scored === undefined) return record
  return groupReportAt(record, thresholds)
}

/** The group's colour: the categorical slot of its index. */
export function groupColour(g: number): string {
  return `tab${g % 10}`
}

/**
 * The rates of every group side by side: one bar per group in each rate.
 *
 * @param record - The report.
 * @param mode - The colour mode.
 * @param shown - The rates to show, top to bottom.
 * @returns The panel.
 */
export function groupRatesPanel(
  record: GroupReport,
  mode: ColourMode,
  shown: readonly GroupRateName[] = GROUP_RATES,
): Panel {
  // Horizontal bars run bottom to top: reverse, so the first rate is on top.
  const rates = [...shown].reverse()
  return {
    layers: [
      bars(
        rates.map((r) => GROUP_RATE_LABELS[r]),
        record.groups.map((name, g) => ({
          name,
          values: rates.map((r) => record.rates[g]?.[r] ?? 0),
          colour: groupColour(g),
        })),
        { mode, horizontal: true },
      ),
    ],
    options: { x: { label: 'rate', limits: [0, 1], format: 'percent' } },
  }
}

/**
 * Each group's gap to the reference group in every rate (the group's rate minus the reference's): bars right of
 * zero where the group is higher.
 *
 * @param record - The report.
 * @param mode - The colour mode.
 * @param shown - The rates to show, top to bottom.
 * @returns The panel.
 */
export function groupGapsPanel(
  record: GroupReport,
  mode: ColourMode,
  shown: readonly GroupRateName[] = GROUP_RATES,
): Panel {
  const rates = [...shown].reverse()
  const ref = record.rates[record.reference]
  const others = record.groups.flatMap((name, g) => (g === record.reference ? [] : [{ name, g }]))
  const gaps = others.flatMap(({ g }) => rates.map((r) => (record.rates[g]?.[r] ?? 0) - (ref?.[r] ?? 0)))
  const top = Math.max(0.1, ...gaps.map(Math.abs))
  return {
    layers: [
      bars(
        rates.map((r) => GROUP_RATE_LABELS[r]),
        others.map(({ name, g }) => ({
          name: `${name} `,
          values: rates.map((r) => (record.rates[g]?.[r] ?? 0) - (ref?.[r] ?? 0)),
          colour: groupColour(g),
        })),
        { mode, horizontal: true },
      ),
      referenceLines({ x: [0] }, { mode, colour: 'fgSecondary', dash: 'solid' }),
    ],
    options: {
      title: `gap to ${record.groups[record.reference] ?? 'the reference'}`,
      x: { label: 'difference in rate', limits: [-top * 1.1, top * 1.1] },
    },
  }
}

/**
 * Each group's ROC curve with its operating point (its threshold's false- and true-positive rates), on the unit square;
 * one point per group, so a gap in either rate shows as distance between the points.
 *
 * @param record - The report (its `roc`, if any, is drawn).
 * @param mode - The colour mode.
 * @returns The panel.
 */
export function groupRocPanel(record: GroupReport, mode: ColourMode): Panel {
  const curves = (record.roc ?? []).map((c, g) => ({
    x: c.fpr,
    y: c.tpr,
    name: `__roc${g}`,
    colour: groupColour(g),
    width: 1.6,
  }))
  const operating = record.groups.map(
    (_, g) =>
      record.operatingPoints?.[g] ?? {
        fpr: record.rates[g]?.falsePositiveRate ?? 0,
        tpr: record.rates[g]?.truePositiveRate ?? 0,
      },
  )
  return {
    layers: [
      lines([{ x: [0, 1], y: [0, 1], name: '__chance', colour: 'muted', dash: 'dotted' }], { mode }),
      ...(curves.length === 0 ? [] : [lines(curves, { mode })]),
      points(
        operating.map((p, g) => ({
          x: [p.fpr],
          y: [p.tpr],
          name: `__point${g}`,
          colour: groupColour(g),
          size: 12,
        })),
        { mode },
      ),
    ],
    options: {
      title: 'ROC and operating points',
      x: { label: 'false-positive rate', limits: [0, 1] },
      y: { label: 'true-positive rate', limits: [0, 1] },
    },
  }
}

/**
 * A group report as one figure: the rates of every group side by side, each group's gap to the reference, and the
 * groups' ROC curves with their operating points. Given `thresholds`, the report is first re-thresholded from its rows
 * (a dragged threshold per group).
 *
 * @param record - The report.
 * @param options - Mode, figure, thresholds and rates.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { groupReport, groupReportFromTables } from 'whitebox-engine/exposition'
 * import { groupReportView } from 'whitebox-render/views'
 * const r = groupReport(groupReportFromTables({ attribute: 'a', groups: ['x', 'y'], tables: [[[5, 1], [1, 3]], [[4, 0], [3, 2]]] }))
 * groupReportView(r).grid.length // => 3
 * ```
 */
export function groupReportView(record: GroupReport, options: GroupReportViewOptions = {}): EChartsOption {
  const r = reportAt(record, options.thresholds)
  const fig = viewFigure(options, { aspect: 1.75 })
  const mode = fig.mode
  const grid = gridSpec(2, 2, {
    widthRatios: [1.25, 1],
    wspace: 64,
    hspace: 72,
    cells: [
      { id: 'rates', row: 0, col: 0 },
      { id: 'gaps', row: 1, col: 0 },
      { id: 'roc', row: 0, col: 1, rowSpan: 2 },
    ],
  })
  return subplots(
    grid,
    {
      rates: groupRatesPanel(r, mode, options.rates),
      gaps: groupGapsPanel(r, mode, options.rates),
      roc: groupRocPanel(r, mode),
    },
    { figure: fig, tooltip: 'item' },
  )
}

/**
 * A group report's rates over a shared threshold (its `sweep`): selection, true- and false-positive rate of every
 * group against the threshold, one colour per group and one dash per rate.
 *
 * @param record - A report with a sweep.
 * @param options - Mode and figure.
 * @returns The ECharts option, or an empty chart without a sweep.
 */
export function groupSweepView(record: GroupReport, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const s = record.sweep
  if (s === undefined) return cartesian([], { figure: fig, mode, title: 'no threshold sweep' })
  const dashes = { selectionRate: 'solid', truePositiveRate: 'dashed', falsePositiveRate: 'dotted' } as const
  const curves = record.groups.flatMap((name, g) =>
    (['selectionRate', 'truePositiveRate', 'falsePositiveRate'] as const).map((rate) => ({
      x: s.thresholds,
      y: s[rate][g] ?? [],
      name: `${name} ${GROUP_RATE_LABELS[rate]}`,
      colour: groupColour(g),
      dash: dashes[rate],
    })),
  )
  return cartesian([lines(curves, { mode })], {
    figure: fig,
    mode,
    x: { label: 'threshold' },
    y: { label: 'rate', limits: [0, 1], format: 'percent' },
  })
}

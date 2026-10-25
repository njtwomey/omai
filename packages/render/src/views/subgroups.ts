/**
 * Views of a subgroup report (`subgroupReport`): one rate as a heatmap over the attributes' values (rows the first
 * attribute, columns the combinations of the rest), and every subgroup's rate as sorted bars with its size, the overall
 * rate and the worst subgroup marked.
 */

import type { EChartsOption } from 'echarts'
import { type GroupRateName, type SubgroupReport, subgroupGrid } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { matrix } from '../charts/matrices.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { type ViewOptions, viewFigure } from './_view.ts'
import { GROUP_RATE_LABELS } from './fairness.ts'

/** Options of the subgroup views. */
export interface SubgroupViewOptions extends ViewOptions {
  /** The rate shown (default the selection rate). */
  readonly rate?: GroupRateName
}

/**
 * One rate of every subgroup as a heatmap over the attributes: rows are the first attribute's values, columns every
 * combination of the others; an empty subgroup is blank.
 *
 * @param record - The report.
 * @param mode - The colour mode.
 * @param rate - The rate.
 * @returns The panel.
 */
export function subgroupHeatmapPanel(record: SubgroupReport, mode: ColourMode, rate: GroupRateName): Panel {
  const g = subgroupGrid(record, rate)
  return {
    layers: [
      matrix(g.values, {
        rowLabels: g.rows,
        colLabels: g.columns,
        range: [0, 1],
        mode,
        colourBar: false,
      }),
    ],
    options: {
      title: `${GROUP_RATE_LABELS[rate]} by ${record.attributes.join(' × ')}`,
      x: { label: record.attributes.slice(1).join(' × ') || ' ' },
      y: { label: record.attributes[0] ?? '' },
    },
  }
}

/**
 * Every subgroup's rate as horizontal bars, highest at the top, each named with its size; the overall rate as a line.
 * Subgroups smaller than the report's minimum size are greyed (and left out of the gap in the title).
 *
 * @param record - The report.
 * @param mode - The colour mode.
 * @param rate - The rate.
 * @returns The panel.
 */
export function subgroupBarsPanel(record: SubgroupReport, mode: ColourMode, rate: GroupRateName): Panel {
  const order = record.subgroups
    .map((_, s) => s)
    .sort((a, b) => (record.rates[a]?.[rate] ?? 0) - (record.rates[b]?.[rate] ?? 0))
  const name = (s: number): string => `${(record.subgroups[s] ?? []).join(' × ')} (n=${record.counts[s] ?? 0})`
  const small = (s: number): boolean => (record.counts[s] ?? 0) < record.minSize
  const value = (s: number): number => record.rates[s]?.[rate] ?? 0
  return {
    layers: [
      bars(
        order.map(name),
        [
          { name: 'counted', values: order.map((s) => (small(s) ? 0 : value(s))), colour: 'tab0' },
          ...(order.some(small)
            ? [{ name: 'too small', values: order.map((s) => (small(s) ? value(s) : 0)), colour: 'muted' }]
            : []),
        ],
        { mode, horizontal: true, stacked: true },
      ),
      referenceLines({ x: [{ value: record.overall[rate], label: 'overall' }] }, { mode, colour: 'fgSecondary' }),
    ],
    options: {
      title: `gap ${(record.gaps[rate] ?? 0).toFixed(3)}, worst ${(record.subgroups[record.worst[rate]?.subgroup ?? 0] ?? []).join(' × ')}`,
      x: { label: GROUP_RATE_LABELS[rate], limits: [0, 1], format: 'percent' },
    },
  }
}

/**
 * A subgroup report as one figure: the chosen rate as a heatmap over the attributes beside every subgroup's rate as
 * sorted bars.
 *
 * @param record - The report.
 * @param options - Mode, figure and the rate.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { subgroupReport, tableRates } from 'whitebox-engine/exposition'
 * import { subgroupView } from 'whitebox-render/views'
 * const tables = [[[1, 1], [0, 2]], [[2, 0], [1, 1]]]
 * const rate = { subgroup: 0, gap: 0 }
 * const zero = { selectionRate: 0, truePositiveRate: 0, falsePositiveRate: 0, precision: 0, accuracy: 0, baseRate: 0 }
 * const r = subgroupReport({
 *   attributes: ['sex', 'age'], values: [['f'], ['old', 'young']], subgroups: [['f', 'old'], ['f', 'young']],
 *   counts: [4, 4], tables, rates: tables.map(tableRates), overall: zero, minSize: 1, gaps: zero,
 *   worst: { selectionRate: rate, truePositiveRate: rate, falsePositiveRate: rate, precision: rate, accuracy: rate, baseRate: rate },
 * })
 * subgroupView(r).grid.length // => 2
 * ```
 */
export function subgroupView(record: SubgroupReport, options: SubgroupViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { width: 900, aspect: 2.2 })
  const rate = options.rate ?? 'selectionRate'
  const grid = gridSpec(1, 2, { wspace: 150, widthRatios: [1, 1.1] })
  return subplots(
    grid,
    { r0c0: subgroupHeatmapPanel(record, fig.mode, rate), r0c1: subgroupBarsPanel(record, fig.mode, rate) },
    { figure: fig, tooltip: 'item', legend: false },
  )
}

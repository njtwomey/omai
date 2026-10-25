/**
 * Views of a data-bias record (`dataBias`): each group's share of the data (beside the population's), its base rate
 * (beside a model's selection rate), the features that act as proxies for the group, and label flips against clean
 * labels; panels without data are left out.
 */

import type { EChartsOption } from 'echarts'
import type { DataBias } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/**
 * The panels of a data-bias record that have data, by id: `shares`, `rates`, and when present `proxies` and `flips`.
 *
 * @param record - The record.
 * @param mode - The colour mode.
 * @returns The panels, in order.
 */
export function dataBiasPanels(record: DataBias, mode: ColourMode): Record<string, Panel> {
  const groups = [...record.groups]
  const panels: Record<string, Panel> = {
    shares: {
      layers: [
        bars(
          groups,
          [
            { name: 'in the data', values: record.shares, colour: 'tab0' },
            ...(record.populationShares === undefined
              ? []
              : [{ name: 'in the population', values: record.populationShares, colour: 'muted' }]),
          ],
          { mode },
        ),
      ],
      options: { title: 'representation', y: { label: 'share', limits: [0, 1], format: 'percent' } },
    },
    rates: {
      layers: [
        bars(
          groups,
          [
            { name: 'base rate (labels)', values: record.baseRates, colour: 'tab1' },
            ...(record.amplification === undefined
              ? []
              : [{ name: 'selection rate (model)', values: record.amplification.selectionRates, colour: 'tab2' }]),
          ],
          { mode },
        ),
        referenceLines({ y: [record.overallBaseRate] }, { mode, colour: 'fgSecondary' }),
      ],
      options: {
        title: 'rates',
        y: { label: 'rate', limits: [0, 1], format: 'percent' },
      },
    },
  }
  const p = record.proxies
  if (p !== undefined) {
    const order = p.features.map((_, j) => j).sort((a, b) => (p.auc[a] ?? 0) - (p.auc[b] ?? 0))
    panels['proxies'] = {
      layers: [
        bars(
          order.map((j) => p.features[j] ?? ''),
          [
            { name: 'AUC for the group', values: order.map((j) => p.auc[j] ?? 0.5), colour: 'tab3' },
            { name: 'mutual information', values: order.map((j) => p.dependence[j] ?? 0), colour: 'tab4' },
          ],
          { mode, horizontal: true },
        ),
        referenceLines({ x: [0.5] }, { mode, colour: 'fgSecondary', dash: 'dotted' }),
      ],
      options: {
        title: 'proxies',
        x: { label: 'score', limits: [0, 1] },
      },
    }
  }
  const f = record.labelFlips
  if (f !== undefined)
    panels['flips'] = {
      layers: [
        bars(
          groups,
          [
            { name: 'good recorded bad', values: f.positiveToNegative, colour: 'tab5' },
            { name: 'bad recorded good', values: f.negativeToPositive, colour: 'tab6' },
          ],
          { mode },
        ),
      ],
      options: { title: 'label flips', y: { label: 'share of the clean class', limits: [0, 1], format: 'percent' } },
    }
  return panels
}

/**
 * A data-bias record as one figure: representation, base rates (with a model's selection rates), proxies and label
 * flips, in a grid of the panels that have data.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { dataBias } from 'whitebox-engine/exposition'
 * import { dataBiasView } from 'whitebox-render/views'
 * const r = dataBias({ attribute: 'sex', groups: ['f', 'm'], reference: 1, counts: [40, 60], shares: [0.4, 0.6],
 *   baseRates: [0.3, 0.5], overallBaseRate: 0.42, disparities: [] })
 * dataBiasView(r).grid.length // => 2
 * ```
 */
export function dataBiasView(record: DataBias, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { width: 900, aspect: 1.6 })
  const panels = dataBiasPanels(record, fig.mode)
  const ids = Object.keys(panels)
  const cols = 2
  const rows = Math.ceil(ids.length / cols)
  const grid = gridSpec(rows, cols, {
    wspace: 110,
    hspace: 90,
    cells: ids.map((id, i) => ({ id, row: Math.floor(i / cols), col: i % cols })),
  })
  return subplots(grid, panels, { figure: fig, tooltip: 'item' })
}

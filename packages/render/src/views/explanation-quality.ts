/**
 * The view of explanation-quality records: one small panel per metric (a bar per explainer, the metric's direction
 * in its title, the best explainer marked) and the explainers' agreement as a matrix.
 */

import type { EChartsOption } from 'echarts'
import type { ExplanationQuality } from 'mdml-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { matrix } from '../charts/matrices.ts'
import { gridSpec } from '../figure/grid.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/** The rows and columns of a quality view's panels (metrics, then the agreement matrix). */
export function qualityLayout(record: ExplanationQuality): { readonly rows: number; readonly cols: number } {
  const panels = record.metrics.length + (record.agreement === undefined ? 0 : 1)
  const cols = Math.min(3, Math.max(1, panels))
  return { rows: Math.max(1, Math.ceil(panels / cols)), cols }
}

/**
 * Explainers compared on quality metrics (Quantus's evaluation tables, as panels): per metric a bar per explainer
 * with its spread, the title saying whether higher or lower is better and which explainer wins; then the mean
 * pairwise agreement of their explanations as an annotated matrix.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { explanationQuality } from 'mdml-engine/exposition'
 * import { explanationQualityView } from 'mdml-render/views'
 * const r = explanationQuality({ explainers: ['a', 'b'], metrics: [{ name: 'm', category: 'faithfulness', values: [1, 2], better: 'higher' }], agreement: { metric: 'spearman', matrix: [[1, 0.5], [0.5, 1]] } })
 * explanationQualityView(r).grid.length // => 2
 * ```
 */
export function explanationQualityView(record: ExplanationQuality, options: ViewOptions = {}): EChartsOption {
  const { rows, cols } = qualityLayout(record)
  const fig = viewFigure(options, { width: 720, height: Math.max(240, 40 + rows * 230) })
  const mode = fig.mode
  const panels: Record<string, Panel> = {}
  record.metrics.forEach((m, i) => {
    const best = m.values.reduce(
      (b, v, j) => ((m.better === 'higher' ? v > (m.values[b] ?? v) : v < (m.values[b] ?? v)) ? j : b),
      0,
    )
    panels[`m${i}`] = {
      layers: [
        bars(
          record.explainers.map((e) => e),
          [
            {
              name: `__${m.name}`,
              values: m.values,
              colour: 'accent',
              ...(m.spread === undefined ? {} : { errors: m.spread }),
            },
          ],
          { mode },
        ),
      ],
      options: {
        title: `${m.name} (${m.better === 'higher' ? '↑' : '↓'}) · ${record.explainers[best] ?? ''}`,
        x: { label: m.category },
      },
    }
  })
  if (record.agreement !== undefined) {
    panels['agreement'] = {
      layers: [
        matrix(
          record.agreement.matrix.map((r) => [...r]),
          {
            rowLabels: record.explainers,
            colLabels: record.explainers,
            range: [-1, 1],
            centre: 0,
            colourBar: false,
            format: (v) => v.toFixed(2),
            mode,
          },
        ),
      ],
      options: { title: `agreement (${record.agreement.metric})` },
    }
  }
  const ids = Object.keys(panels)
  const grid = gridSpec(rows, cols, {
    wspace: 56,
    hspace: 70,
    cells: ids.map((id, i) => ({ id, row: Math.floor(i / cols), col: i % cols })),
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}

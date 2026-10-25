/**
 * The view of a Pareto-front record (`paretoFront`): every candidate as a faint point, each family's front as a step
 * line through its points, reference marks, and the chosen candidate ringed.
 */

import type { EChartsOption } from 'echarts'
import type { ParetoFront } from 'mdml-engine/exposition'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/**
 * A Pareto-front record as one figure: accuracy against violation, each family's candidates faint and its front a
 * line, reference marks as labelled points, and the chosen candidate ringed.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { paretoFront } from 'mdml-engine/exposition'
 * import { paretoFrontView } from 'mdml-render/views'
 * const r = paretoFront({ violationName: 'gap', families: [{ name: 'grid', violation: [0.1, 0.2], accuracy: [0.7, 0.8], front: [0, 1] }] })
 * paretoFrontView(r).series.length > 0 // => true
 * ```
 */
export function paretoFrontView(record: ParetoFront, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.5 })
  const mode = fig.mode
  const layers: Layer[] = []
  record.families.forEach((f, k) => {
    const colour = `tab${k % 10}`
    layers.push(
      points([{ x: f.violation, y: f.accuracy, name: `__${f.name} candidates`, colour, size: 6, opacity: 0.35 }], {
        mode,
      }),
      lines(
        [
          {
            x: f.front.map((i) => f.violation[i] ?? 0),
            y: f.front.map((i) => f.accuracy[i] ?? 0),
            name: f.name,
            colour,
            step: 'end',
            markers: true,
          },
        ],
        { mode },
      ),
    )
  })
  for (const mark of record.marks ?? [])
    layers.push(
      points(
        [{ x: [mark.violation], y: [mark.accuracy], name: mark.label, colour: 'fg', size: 10, symbol: 'diamond' }],
        {
          mode,
        },
      ),
    )
  const c = record.chosen
  if (c !== undefined) {
    const f = record.families[c.family]
    layers.push(
      points(
        [
          {
            x: [f?.violation[c.index] ?? 0],
            y: [f?.accuracy[c.index] ?? 0],
            name: `chosen: ${c.rule}`,
            colour: 'tab3',
            size: 16,
            symbol: 'emptyCircle',
          },
        ],
        { mode },
      ),
    )
  }
  return cartesian(layers, {
    figure: fig,
    mode,
    x: { label: record.violationName },
    y: { label: 'accuracy' },
  })
}

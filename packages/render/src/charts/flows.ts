/**
 * Flows between stages: a Sankey diagram, whose links have widths proportional to the quantity flowing (energy,
 * money, people moving between groups), its nodes stacked in columns by stage.
 */

import type { EChartsOption } from 'echarts'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { chartFrame } from './plot.ts'

/** A flow from one node to another. */
export interface FlowLink {
  readonly source: string
  readonly target: string
  readonly value: number
}

/**
 * A Sankey diagram: nodes in columns (placed by the links), links as bands as wide as their value, coloured by
 * their source.
 *
 * @param links - The flows.
 * @param options - Mode, figure, title, and the nodes' order or colours when given.
 * @returns The ECharts option.
 * @index function A Sankey diagram of flows: `sankeyChart([{ source, target, value }])`.
 * @example
 * ```ts
 * import { sankeyChart } from 'omai-render/charts'
 * sankeyChart([{ source: 'a', target: 'b', value: 3 }]).series[0].data.length // => 2
 * ```
 */
export function sankeyChart(
  links: readonly FlowLink[],
  options: {
    readonly mode?: ColourMode
    readonly figure?: Figure
    readonly title?: string
    readonly nodes?: readonly string[]
    readonly orient?: 'horizontal' | 'vertical'
  } = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const names = options.nodes ?? [...new Set(links.flatMap((l) => [l.source, l.target]))]
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'sankey',
        orient: options.orient ?? 'horizontal',
        top: options.title === undefined ? 12 : 40,
        bottom: 12,
        left: 12,
        right: 96,
        nodeGap: 10,
        emphasis: { focus: 'adjacency' },
        data: names.map((name, i) => ({ name, itemStyle: { color: categoricalColour(i), borderColor: t.bg } })),
        links: links.map((l) => ({ ...l })),
        lineStyle: { color: 'source', opacity: 0.35, curveness: 0.5 },
        label: { color: t.fg, fontFamily: t.font, fontSize: fig.fontSize },
      },
    ],
  } as EChartsOption
}

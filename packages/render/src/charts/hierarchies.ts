/**
 * Hierarchies drawn by area: a treemap (nested rectangles) and a sunburst (nested rings), from a tree of named nodes
 * whose leaves carry values, coloured by top-level branch.
 */

import type { EChartsOption } from 'echarts'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { chartFrame } from './plot.ts'

/** A node of a hierarchy: a name, and either a value (a leaf) or children (a branch's value is its leaves' sum). */
export interface TreeNode {
  readonly name: string
  readonly value?: number
  readonly children?: readonly TreeNode[]
}

/** Options of the hierarchy charts. */
export interface HierarchyOptions {
  readonly mode?: ColourMode
  readonly figure?: Figure
  readonly title?: string
  /** How many levels to show at once (default all). */
  readonly depth?: number
}

// The tree with every top-level branch coloured, for ECharts.
function coloured(nodes: readonly TreeNode[]): unknown[] {
  return nodes.map((n, i) => ({ ...n, itemStyle: { color: categoricalColour(i) } }))
}

/**
 * A treemap: each leaf a rectangle with area proportional to its value, nested inside its branches, coloured by
 * top-level branch. Click a branch to zoom into it.
 *
 * @param nodes - The top-level nodes.
 * @param options - Mode, figure, title and levels shown.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { treemapChart } from 'mdml-render/charts'
 * treemapChart([{ name: 'a', value: 2 }, { name: 'b', children: [{ name: 'c', value: 1 }] }]).series[0].type // => 'treemap'
 * ```
 */
export function treemapChart(nodes: readonly TreeNode[], options: HierarchyOptions = {}): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'treemap',
        data: coloured(nodes),
        top: options.title === undefined ? 8 : 36,
        left: 8,
        right: 8,
        bottom: 32,
        roam: false,
        ...(options.depth === undefined ? {} : { leafDepth: options.depth }),
        breadcrumb: { show: true, bottom: 4, itemStyle: { color: t.surface, textStyle: { color: t.fg } } },
        label: { show: true, color: '#fff', fontFamily: t.font, fontSize: fig.fontSize },
        upperLabel: { show: true, height: 18, color: '#fff', fontFamily: t.font },
        itemStyle: { borderColor: t.bg, borderWidth: 1, gapWidth: 1 },
        levels: [
          { itemStyle: { borderWidth: 0, gapWidth: 3 } },
          { itemStyle: { gapWidth: 1 } },
          { colorSaturation: [0.35, 0.6] },
        ],
      },
    ],
  } as EChartsOption
}

/**
 * A sunburst: the hierarchy as rings, the root at the centre, each node's arc proportional to its value. Click a
 * node to make it the centre.
 *
 * @param nodes - The top-level nodes.
 * @param options - Mode, figure, title and levels shown.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { sunburstChart } from 'mdml-render/charts'
 * sunburstChart([{ name: 'a', value: 2 }]).series[0].type // => 'sunburst'
 * ```
 */
export function sunburstChart(nodes: readonly TreeNode[], options: HierarchyOptions = {}): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'sunburst',
        data: coloured(nodes),
        radius: [0, '92%'],
        center: ['50%', options.title === undefined ? '50%' : '54%'],
        sort: undefined,
        label: { color: '#fff', fontFamily: t.font, fontSize: fig.fontSize - 1, minAngle: 8 },
        itemStyle: { borderColor: t.bg, borderWidth: 1 },
        ...(options.depth === undefined ? {} : { levels: Array.from({ length: options.depth + 1 }, () => ({})) }),
      },
    ],
  } as EChartsOption
}

/**
 * Regular graph layouts: nodes on a grid row by row, two-column bipartite layouts (factor graphs, bipartite networks),
 * and lattices whose row-major node order matches grid graphs and Ising-style models.
 */

import { maxOf } from 'mdml-engine/base'
import { type GraphLayout, type LayoutBox, type LayoutGraph, normaliseLayout, straightRoutes } from './graph.ts'

/**
 * Nodes on a grid, row by row, in order (`columns` per row, default the square root of the count).
 *
 * @param graph - The graph.
 * @param options - Columns, cell size (default the largest node plus 24) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { gridLayout } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c', 'd'].map((id) => ({ id, width: 10, height: 10 })), edges: [] }
 * const l = gridLayout(g, { columns: 2, cell: 50 })
 * [l.nodes['b'].x - l.nodes['a'].x, l.nodes['c'].y - l.nodes['a'].y] // => [50, 50]
 * ```
 */
export function gridLayout(
  graph: LayoutGraph,
  options: { readonly columns?: number; readonly cell?: number; readonly margin?: number } = {},
): GraphLayout {
  const n = graph.nodes.length
  const columns = Math.max(1, options.columns ?? Math.ceil(Math.sqrt(n)))
  const cell = options.cell ?? Math.max(10, maxOf(graph.nodes.map((v) => Math.max(v.width, v.height)))) + 24
  const placed: Record<string, LayoutBox> = {}
  graph.nodes.forEach((v, i) => {
    placed[v.id] = { x: (i % columns) * cell, y: Math.floor(i / columns) * cell, width: v.width, height: v.height }
  })
  return normaliseLayout(placed, straightRoutes(placed, graph.edges), {}, options.margin ?? 16)
}

/**
 * A bipartite layout: the nodes of one side in a column (or row), the rest in a second, each in order.
 *
 * @param graph - The graph.
 * @param left - The ids on the first side.
 * @param options - Orientation (`columns`, the default, or `rows`), the gap between the sides (default 200), the
 * spacing along a side (default the largest node plus 16) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { bipartiteLayout } from 'mdml-render/layout'
 * const g = { nodes: ['u', 'v', 'a'].map((id) => ({ id, width: 10, height: 10 })), edges: [{ from: 'u', to: 'a' }] }
 * const l = bipartiteLayout(g, ['u', 'v'])
 * l.nodes['a'].x - l.nodes['u'].x // => 200
 * ```
 */
export function bipartiteLayout(
  graph: LayoutGraph,
  left: readonly string[],
  options: {
    readonly orientation?: 'columns' | 'rows'
    readonly gap?: number
    readonly spacing?: number
    readonly margin?: number
  } = {},
): GraphLayout {
  const { orientation = 'columns', gap = 200, margin = 16 } = options
  const spacing = options.spacing ?? Math.max(10, maxOf(graph.nodes.map((v) => Math.max(v.width, v.height)))) + 16
  const isLeft = new Set(left)
  const sides = [graph.nodes.filter((v) => isLeft.has(v.id)), graph.nodes.filter((v) => !isLeft.has(v.id))]
  const longest = Math.max(sides[0]?.length ?? 0, sides[1]?.length ?? 0)
  const placed: Record<string, LayoutBox> = {}
  sides.forEach((side, s) => {
    const offset = ((longest - side.length) * spacing) / 2
    side.forEach((v, i) => {
      const along = offset + i * spacing
      const across = s * gap
      placed[v.id] =
        orientation === 'columns'
          ? { x: across, y: along, width: v.width, height: v.height }
          : { x: along, y: across, width: v.width, height: v.height }
    })
  })
  return normaliseLayout(placed, straightRoutes(placed, graph.edges), {}, margin)
}

/**
 * A lattice: node `i` at row `floor(i / columns)`, column `i % columns` of a `rows` by `columns` lattice with the given
 * spacing, for grid graphs and Ising-style models whose node order is row-major.
 *
 * @param graph - The graph, nodes in row-major order.
 * @param columns - Nodes per row.
 * @param options - Spacing (default 48) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { latticeLayout } from 'mdml-render/layout'
 * const g = { nodes: ['a', 'b', 'c', 'd'].map((id) => ({ id, width: 8, height: 8 })), edges: [] }
 * latticeLayout(g, 2, { spacing: 30 }).nodes['d'].x - latticeLayout(g, 2, { spacing: 30 }).nodes['c'].x // => 30
 * ```
 */
export function latticeLayout(
  graph: LayoutGraph,
  columns: number,
  options: { readonly spacing?: number; readonly margin?: number } = {},
): GraphLayout {
  return gridLayout(graph, {
    columns,
    cell: options.spacing ?? 48,
    ...(options.margin === undefined ? {} : { margin: options.margin }),
  })
}

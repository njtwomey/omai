/**
 * Tree layouts: tidy trees (leaves side by side, each parent centred over its children, depth in rows) as decision and
 * search trees are drawn, radial trees with depths on rings, and dendrograms with merges at their heights, as
 * hierarchical clustering is drawn.
 */

import { type Box, type GraphLayout, type LayoutGraph, normaliseLayout, straightRoutes } from './graph.ts'

/** Options of the tree layouts. */
export interface TreeOptions {
  /** The root (default the first node with no parent). */
  readonly root?: string
  /** Grow downwards (`down`, the default) or rightwards (`right`). */
  readonly direction?: 'down' | 'right'
  /** Space between depths (default 56). */
  readonly levelSep?: number
  /** Space between neighbouring leaves (default 16). */
  readonly siblingSep?: number
  /** Space around the drawing (default 16). */
  readonly margin?: number
}

interface Shape {
  readonly root: string
  readonly children: ReadonlyMap<string, readonly string[]>
  readonly size: ReadonlyMap<string, { readonly width: number; readonly height: number }>
}

function shape(graph: LayoutGraph, root?: string): Shape {
  const children = new Map<string, string[]>()
  const hasParent = new Set<string>()
  const ids = new Set(graph.nodes.map((n) => n.id))
  for (const e of graph.edges) {
    if (!ids.has(e.from) || !ids.has(e.to) || hasParent.has(e.to) || e.to === root) continue
    children.set(e.from, [...(children.get(e.from) ?? []), e.to])
    hasParent.add(e.to)
  }
  const top = root ?? graph.nodes.find((n) => !hasParent.has(n.id))?.id ?? graph.nodes[0]?.id ?? ''
  return { root: top, children, size: new Map(graph.nodes.map((n) => [n.id, n])) }
}

/** Leaves placed in order, each parent centred over its children; returns the breadth position and depth of each node. */
function tidyPositions(s: Shape, siblingSep: number, horizontal: boolean): Map<string, [number, number]> {
  const out = new Map<string, [number, number]>()
  let next = 0
  const extent = (v: string): number => {
    const z = s.size.get(v)
    return (horizontal ? z?.height : z?.width) ?? 0
  }
  const visit = (v: string, depth: number, seen: Set<string>): number => {
    seen.add(v)
    const kids = (s.children.get(v) ?? []).filter((c) => !seen.has(c))
    let at: number
    if (kids.length === 0) {
      at = next + extent(v) / 2
      next += extent(v) + siblingSep
    } else {
      const xs = kids.map((c) => visit(c, depth + 1, seen))
      at = ((xs[0] as number) + (xs[xs.length - 1] as number)) / 2
      // A parent wider than its children's span pushes the next subtree along.
      next = Math.max(next, at + extent(v) / 2 + siblingSep)
    }
    out.set(v, [at, depth])
    return at
  }
  if (s.root !== '') visit(s.root, 0, new Set())
  return out
}

function finish(graph: LayoutGraph, boxes: Record<string, Box>, margin: number): GraphLayout {
  return normaliseLayout(boxes, straightRoutes(boxes, graph.edges), {}, margin)
}

/**
 * A tidy tree: leaves side by side in order, every parent centred over its children, depths in rows (or columns), as
 * decision trees and search trees are drawn. Edges are those of the graph (the first parent of each node counts; nodes
 * not reachable from the root are left out).
 *
 * @param graph - The tree as nodes with sizes and parent-to-child edges.
 * @param options - Root, direction and spacing.
 * @returns The layout.
 * @index function Lay out a tree with parents over their children: `tidyTree(graph)`.
 * @example
 * ```ts
 * import { tidyTree } from 'mdml-render/layout'
 * const g = { nodes: ['r', 'a', 'b'].map((id) => ({ id, width: 20, height: 20 })), edges: [{ from: 'r', to: 'a' }, { from: 'r', to: 'b' }] }
 * const l = tidyTree(g)
 * l.nodes['r'].x === (l.nodes['a'].x + l.nodes['b'].x) / 2 // => true
 * ```
 */
export function tidyTree(graph: LayoutGraph, options: TreeOptions = {}): GraphLayout {
  const { direction = 'down', levelSep = 56, siblingSep = 16, margin = 16 } = options
  const s = shape(graph, options.root)
  const horizontal = direction === 'right'
  const pos = tidyPositions(s, siblingSep, horizontal)
  const depthSize = new Map<number, number>()
  for (const [v, [, d]] of pos) {
    const z = s.size.get(v)
    depthSize.set(d, Math.max(depthSize.get(d) ?? 0, (horizontal ? z?.width : z?.height) ?? 0))
  }
  const offsets: number[] = []
  let acc = 0
  for (let d = 0; d <= Math.max(0, ...depthSize.keys()); d++) {
    const h = depthSize.get(d) ?? 0
    offsets.push(acc + h / 2)
    acc += h + levelSep
  }
  const boxes: Record<string, Box> = {}
  for (const [v, [at, d]] of pos) {
    const z = s.size.get(v) ?? { width: 0, height: 0 }
    const depthAt = offsets[d] ?? 0
    boxes[v] = horizontal
      ? { x: depthAt, y: at, width: z.width, height: z.height }
      : { x: at, y: depthAt, width: z.width, height: z.height }
  }
  return finish(graph, boxes, margin)
}

/**
 * A radial tree: the tidy tree's order around a circle, the root at the centre and each depth on a ring.
 *
 * @param graph - The tree.
 * @param options - Root, the ring spacing (`levelSep`, default 70) and margin.
 * @returns The layout.
 * @example
 * ```ts
 * import { radialTree } from 'mdml-render/layout'
 * const g = { nodes: ['r', 'a', 'b', 'c'].map((id) => ({ id, width: 10, height: 10 })), edges: ['a', 'b', 'c'].map((to) => ({ from: 'r', to })) }
 * const l = radialTree(g)
 * Math.round(Math.hypot(l.nodes['a'].x - l.nodes['r'].x, l.nodes['a'].y - l.nodes['r'].y)) // => 70
 * ```
 */
export function radialTree(graph: LayoutGraph, options: Omit<TreeOptions, 'direction'> = {}): GraphLayout {
  const { levelSep = 70, margin = 16 } = options
  const s = shape(graph, options.root)
  const pos = tidyPositions(s, 1, false)
  // Breadth positions of the leaves span [min, max]; spread them over the full circle.
  const ats = [...pos.values()].map(([a]) => a)
  const lo = Math.min(...ats)
  const span = Math.max(...ats) - lo
  const leaves = [...pos.keys()].filter((v) => (s.children.get(v) ?? []).length === 0).length
  const full = span === 0 ? 1 : (span * leaves) / Math.max(leaves - 1, 1)
  const boxes: Record<string, Box> = {}
  for (const [v, [at, d]] of pos) {
    const z = s.size.get(v) ?? { width: 0, height: 0 }
    const angle = (2 * Math.PI * (at - lo)) / full - Math.PI / 2
    boxes[v] = {
      x: d * levelSep * Math.cos(angle),
      y: d * levelSep * Math.sin(angle),
      width: z.width,
      height: z.height,
    }
  }
  return finish(graph, boxes, margin)
}

/**
 * A dendrogram: leaves in a row in tree order, each merge drawn at its height (a merge distance), with elbow edges, as
 * hierarchical clustering is drawn.
 *
 * @param graph - The tree (merges to their two or more children).
 * @param heights - The height of each merge by id (leaves default to 0).
 * @param options - Root, the drawing's height for the largest merge (`levelSep` is ignored; `scale`, default 240), leaf
 * spacing and margin.
 * @returns The layout, with elbow routes.
 * @index function Lay out merges at their heights: `dendrogram(graph, heights)`.
 * @example
 * ```ts
 * import { dendrogram } from 'mdml-render/layout'
 * const g = { nodes: ['m', 'a', 'b'].map((id) => ({ id, width: 0, height: 0 })), edges: [{ from: 'm', to: 'a' }, { from: 'm', to: 'b' }] }
 * const l = dendrogram(g, { m: 2 })
 * l.nodes['m'].y < l.nodes['a'].y // => true
 * ```
 */
export function dendrogram(
  graph: LayoutGraph,
  heights: Readonly<Record<string, number>>,
  options: Omit<TreeOptions, 'direction' | 'levelSep'> & { readonly scale?: number } = {},
): GraphLayout {
  const { siblingSep = 16, margin = 16, scale = 240 } = options
  const s = shape(graph, options.root)
  const pos = tidyPositions(s, siblingSep, false)
  const top = Math.max(1e-12, ...[...pos.keys()].map((v) => heights[v] ?? 0))
  const boxes: Record<string, Box> = {}
  for (const [v, [at]] of pos) {
    const z = s.size.get(v) ?? { width: 0, height: 0 }
    boxes[v] = { x: at, y: scale * (1 - (heights[v] ?? 0) / top), width: z.width, height: z.height }
  }
  const routes = graph.edges.flatMap((e) => {
    const a = boxes[e.from]
    const b = boxes[e.to]
    if (a === undefined || b === undefined) return []
    return [
      {
        from: e.from,
        to: e.to,
        points: [
          [a.x, a.y],
          [b.x, a.y],
          [b.x, b.y],
        ] as [number, number][],
      },
    ]
  })
  return normaliseLayout(boxes, routes, {}, margin)
}

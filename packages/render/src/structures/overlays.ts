/**
 * Overlays for animating a traversal of a structure: `traversalFrames` makes each visited node active in turn with the
 * edge it was reached by and numbers the nodes in visit order, and `searchOrder` gives the breadth-first and
 * depth-first expansion orders from a root.
 */

import type { Structure, StructureOverlay } from 'whitebox-engine/scene'

/**
 * Frames of a traversal (a search's expansion order, a topological order): frame `t` makes node `order[t]` active
 * and the edge it was reached by, and numbers every node visited so far by when it was visited. (For message passing
 * recorded in a trace, see `messageFrames` in `whitebox-render/trace`.)
 *
 * @param s - The structure.
 * @param order - Node ids in visiting order.
 * @param options - The badge's name (default `order`).
 * @returns One overlay per visit.
 * @example
 * ```ts
 * import { structure } from 'whitebox-engine/scene'
 * import { traversalFrames } from 'whitebox-render/structures'
 * const s = structure('tree', [{ id: 'a', label: 'a', role: 'internal' }, { id: 'b', label: 'b', role: 'leaf' }], [{ from: 'a', to: 'b' }])
 * traversalFrames(s, ['a', 'b'])[1] // => { active: ['b'], edges: [{ from: 'a', to: 'b' }], badges: { a: { order: 1 }, b: { order: 2 } } }
 * ```
 */
export function traversalFrames(
  s: Structure,
  order: readonly string[],
  options: { readonly badge?: string } = {},
): StructureOverlay[] {
  const name = options.badge ?? 'order'
  const seen = new Set<string>()
  const badges: Record<string, Record<string, number>> = {}
  return order.map((id, t) => {
    // The edge it was reached by: from a node already visited.
    const via = s.edges.find((e) => e.to === id && seen.has(e.from))
    seen.add(id)
    badges[id] = { [name]: t + 1 }
    return {
      active: [id],
      edges: via === undefined ? [] : [{ from: via.from, to: via.to }],
      badges: Object.fromEntries(Object.entries(badges).map(([k, v]) => [k, { ...v }])),
    }
  })
}

/**
 * The orders in which breadth-first and depth-first search expand a structure from a root (children in edge order).
 *
 * @param s - The structure (edges point from parent to child).
 * @param root - The start.
 * @param kind - `bfs` or `dfs`.
 * @returns The node ids in expansion order.
 * @example
 * ```ts
 * import { structure } from 'whitebox-engine/scene'
 * import { searchOrder } from 'whitebox-render/structures'
 * const s = structure('tree', ['r', 'a', 'b', 'c'].map((id) => ({ id, label: id, role: 'internal' as const })), [{ from: 'r', to: 'a' }, { from: 'r', to: 'b' }, { from: 'a', to: 'c' }])
 * [searchOrder(s, 'r', 'bfs'), searchOrder(s, 'r', 'dfs')] // => [['r', 'a', 'b', 'c'], ['r', 'a', 'c', 'b']]
 * ```
 */
export function searchOrder(s: Structure, root: string, kind: 'bfs' | 'dfs'): string[] {
  const children = new Map<string, string[]>()
  for (const e of s.edges) children.set(e.from, [...(children.get(e.from) ?? []), e.to])
  const out: string[] = []
  const seen = new Set<string>([root])
  const pending = [root]
  while (pending.length > 0) {
    const v = (kind === 'bfs' ? pending.shift() : pending.pop()) as string
    out.push(v)
    const next = (children.get(v) ?? []).filter((c) => !seen.has(c))
    for (const c of next) seen.add(c)
    pending.push(...(kind === 'bfs' ? next : [...next].reverse()))
  }
  return out
}

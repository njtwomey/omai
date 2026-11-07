/**
 * The spanning-tree algorithms as steppable algorithms: Kruskal's (one edge considered per step, in weight order),
 * Prim's (one node joined to the tree per step) and Boruvka's (one component's cheapest edge considered per step, round
 * by round). Each state serialises and resumes and logs the paths of {@link spanningTreeContract}; the slot is the
 * option table {@link spanningTreeMethods}. `minimumSpanningTree` in `trees.ts` is these run to the end.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { dict, knobOption, type OptionTable, optionTable } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import type { Edge, Graph } from './graph.ts'
import { IndexedQueue, UnionFind } from './structures.ts'

/** The paths the spanning-tree algorithms log, as a type. */
export type SpanningTreePath = 'mst/edge' | 'mst/accepted' | 'mst/weight' | 'mst/components' | 'mst/chosen'

/**
 * What the spanning-tree algorithms log, by path: the edge considered at this step (its index in `graph.edges`, -1
 * when Prim starts a new tree at a root), whether it joined the forest, the weight of the forest so far, its number of
 * components and the edges chosen so far.
 */
export const spanningTreeContract: Contract<unknown, SpanningTreePath> = defineContract({
  'mst/edge': scalar({ description: 'the edge considered at this step, by index in the graph (-1 for a new root)' }),
  'mst/accepted': scalar({ description: '1 when the edge joined the forest, 0 when it was rejected as a cycle' }),
  'mst/weight': scalar({ description: 'the total weight of the forest so far' }),
  'mst/components': scalar({ description: 'the number of components of the forest so far (nodes minus edges chosen)' }),
  'mst/chosen': tensor({ role: 'state', shape: ['chosen'], description: 'the indices of the edges chosen so far' }),
})

/** The algorithm behind a spanning tree. */
export type SpanningTreeAlgorithm = 'kruskal' | 'prim' | 'boruvka'

/** Options of the spanning-tree algorithms. */
export interface SpanningTreeAlgorithmOptions {
  /** Find a maximum spanning tree instead of a minimum one (default false). */
  readonly maximum?: boolean
}

/** The state of a spanning-tree algorithm: the edges chosen so far and where the sweep stands. */
export interface SpanningTreeState {
  /** The indices (in `graph.edges`) of the edges chosen, in the order they were chosen. */
  readonly chosen: number[]
  /** The edge considered at the last step (-1 before the first and for a new Prim root). */
  readonly edge: number
  /** Whether the last edge was chosen. */
  readonly accepted: boolean
  /** The total weight of the chosen edges. */
  readonly weight: number
  /** Kruskal and Boruvka: the next position in the sorted edges; Prim: the next root to try. */
  readonly cursor: number
  /** Boruvka: the candidate edges of the current round still to consider. */
  readonly pending: number[]
  /** Prim: 1 for nodes in the tree. */
  readonly inTree: Uint8Array
  /** Prim: the edge by which each queued node would join (-1 for none). */
  readonly via: Int32Array
  /** Prim: the frontier, by rank of the joining edge. */
  readonly queue: IndexedQueue
  /** Kruskal and Boruvka: the components of the forest. */
  readonly components: UnionFind
}

/** Edge order: by weight (negated for a maximum tree), then by edge index, so every algorithm finds the same tree. */
function edgeOrder(g: Graph, maximum: boolean): (p: number, q: number) => number {
  const sign = maximum ? -1 : 1
  return (p, q) => sign * ((g.edges[p]?.weight ?? 0) - (g.edges[q]?.weight ?? 0)) || p - q
}

/** The components of the whole graph, to know when a forest is complete. */
function componentCount(g: Graph): number {
  const uf = new UnionFind(g.n)
  for (const e of g.edges) uf.union(e.source, e.target)
  return uf.count
}

function serialiseState(s: SpanningTreeState, n: number): Json {
  const queued: [number, number][] = []
  for (let v = 0; v < n; v++) if (s.queue.has(v)) queued.push([v, s.queue.priority(v) ?? 0])
  return {
    chosen: [...s.chosen],
    edge: s.edge,
    accepted: s.accepted,
    weight: s.weight,
    cursor: s.cursor,
    pending: [...s.pending],
    inTree: stateToJson(s.inTree),
    via: stateToJson(s.via),
    queue: queued,
    components: s.components.toJSON() as unknown as Json,
  }
}

function deserialiseState(j: Json, n: number): SpanningTreeState {
  const o = j as { readonly [key: string]: Json }
  const queue = new IndexedQueue(n)
  for (const [v, p] of o['queue'] as unknown as readonly (readonly [number, number])[]) queue.push(v, p)
  return {
    chosen: [...(o['chosen'] as unknown as readonly number[])],
    edge: o['edge'] as number,
    accepted: o['accepted'] as boolean,
    weight: o['weight'] as number,
    cursor: o['cursor'] as number,
    pending: [...(o['pending'] as unknown as readonly number[])],
    inTree: stateFromJson(o['inTree'] as Json) as Uint8Array,
    via: stateFromJson(o['via'] as Json) as Int32Array,
    queue,
    components: UnionFind.fromJSON(o['components'] as unknown as Parameters<typeof UnionFind.fromJSON>[0]),
  }
}

function logStep(
  s: SpanningTreeState,
  n: number,
  ctx: Parameters<Algorithm<SpanningTreeState, SpanningTreePath>['step']>[1],
): void {
  if (ctx.wants('mst/edge')) ctx.log('mst/edge', s.edge)
  if (ctx.wants('mst/accepted')) ctx.log('mst/accepted', s.accepted ? 1 : 0)
  if (ctx.wants('mst/weight')) ctx.log('mst/weight', s.weight)
  if (ctx.wants('mst/components')) ctx.log('mst/components', n - s.chosen.length)
  if (ctx.wants('mst/chosen')) ctx.log('mst/chosen', Int32Array.from(s.chosen))
}

function emptyState(g: Graph): SpanningTreeState {
  return {
    chosen: [],
    edge: -1,
    accepted: false,
    weight: 0,
    cursor: 0,
    pending: [],
    inTree: new Uint8Array(g.n),
    via: new Int32Array(g.n).fill(-1),
    queue: new IndexedQueue(g.n),
    components: new UnionFind(g.n),
  }
}

function checkUndirected(g: Graph): void {
  if (g.directed) throw new MdmlError('value', 'spanningTreeAlgorithm: the graph must be undirected')
}

/** Shared parts of the three algorithms: the name, the contract of the run and the serialisation. */
function base(
  name: string,
  g: Graph,
  maximum: boolean,
  parts: Pick<Algorithm<SpanningTreeState, SpanningTreePath>, 'init' | 'step' | 'done'>,
): Algorithm<SpanningTreeState, SpanningTreePath> {
  return {
    name,
    config: { nodes: g.n, edges: g.edges.length, maximum },
    ...parts,
    serialise: (s) => serialiseState(s, g.n),
    deserialise: (j) => deserialiseState(j, g.n),
  }
}

/** Accept edge `k` into the forest when it joins two components; shared by Kruskal and Boruvka. */
function consider(g: Graph, s: SpanningTreeState, k: number): SpanningTreeState {
  const e = g.edges[k] as Edge
  const accepted = s.components.union(e.source, e.target)
  if (accepted) s.chosen.push(k)
  return { ...s, edge: k, accepted, weight: accepted ? s.weight + e.weight : s.weight }
}

/**
 * Kruskal's algorithm as a steppable algorithm: each step considers the next edge in weight order and adds it when it
 * joins two components, rejects it when it would close a cycle. Done when the forest spans every component.
 *
 * @param g - An undirected graph.
 * @param options - `maximum` for a maximum spanning tree.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept spanning-trees
 * @example
 * ```ts
 * import { graph, kruskalAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = graph([[0, 1, 4], [0, 2, 1], [1, 2, 2], [1, 3, 5], [2, 3, 8]])
 * const run = runAlgorithm(kruskalAlgorithm(g), { maxSteps: 10 })
 * [run.steps, run.state.chosen, run.state.weight] // => [4, [1, 2, 3], 8]
 * ```
 */
export function kruskalAlgorithm(
  g: Graph,
  options: SpanningTreeAlgorithmOptions = {},
): Algorithm<SpanningTreeState, SpanningTreePath> {
  checkUndirected(g)
  const maximum = options.maximum ?? false
  const order = g.edges.map((_, i) => i).sort(edgeOrder(g, maximum))
  const target = g.n - componentCount(g)
  return base('graph/kruskal', g, maximum, {
    init: () => emptyState(g),
    step: (s, ctx) => {
      const k = order[s.cursor]
      if (k === undefined) return s
      const next = { ...consider(g, s, k), cursor: s.cursor + 1 }
      logStep(next, g.n, ctx)
      return next
    },
    done: (s) => (s.chosen.length >= target || s.cursor >= order.length ? 'converged' : undefined),
  })
}

/**
 * Prim's algorithm as a steppable algorithm: each step joins the cheapest frontier node to the tree through its
 * cheapest edge (a node of a new component starts a new tree with edge -1). Done when every node is in the forest.
 *
 * @param g - An undirected graph.
 * @param options - `maximum` for a maximum spanning tree.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept spanning-trees
 * @example
 * ```ts
 * import { graph, primAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = graph([[0, 1, 4], [0, 2, 1], [1, 2, 2], [1, 3, 5], [2, 3, 8]])
 * const run = runAlgorithm(primAlgorithm(g), { maxSteps: 10 })
 * [run.steps, run.state.chosen, run.state.weight] // => [4, [1, 2, 3], 8]
 * ```
 */
export function primAlgorithm(
  g: Graph,
  options: SpanningTreeAlgorithmOptions = {},
): Algorithm<SpanningTreeState, SpanningTreePath> {
  checkUndirected(g)
  const maximum = options.maximum ?? false
  // Priorities are ranks in the total edge order, so Prim picks the same tree as Kruskal.
  const rank = new Int32Array(g.edges.length)
  for (const [r, k] of g.edges
    .map((_, i) => i)
    .sort(edgeOrder(g, maximum))
    .entries())
    rank[k] = r
  const a = g.out
  /** The first node at or after `from` that is not in the tree yet (`g.n` when none). */
  const firstFree = (s: SpanningTreeState, from: number): number => {
    let v = from
    while (v < g.n && s.inTree[v] === 1) v++
    return v
  }
  return base('graph/prim', g, maximum, {
    init: () => emptyState(g),
    step: (s, ctx) => {
      let cursor = s.cursor
      if (s.queue.size === 0) {
        cursor = firstFree(s, cursor)
        if (cursor >= g.n) return s
        s.queue.push(cursor, -1)
        s.via[cursor] = -1
      }
      const u = s.queue.pop() as number
      s.inTree[u] = 1
      const e = s.via[u] ?? -1
      let weight = s.weight
      if (e >= 0) {
        s.chosen.push(e)
        weight += g.edges[e]?.weight ?? 0
      }
      for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
        const v = a.indices[k] ?? 0
        if (s.inTree[v] === 1) continue
        const id = a.edges[k] ?? 0
        if (s.queue.decrease(v, rank[id] ?? 0)) s.via[v] = id
      }
      const next = { ...s, edge: e, accepted: e >= 0, weight, cursor }
      logStep(next, g.n, ctx)
      return next
    },
    done: (s) => (s.queue.size === 0 && firstFree(s, s.cursor) >= g.n ? 'converged' : undefined),
  })
}

/**
 * Boruvka's algorithm as a steppable algorithm. A round picks the cheapest edge leaving each component; each step then
 * considers one of those edges (in edge order) and adds it unless the round has already joined its two ends. Done when
 * no edge leaves any component.
 *
 * @param g - An undirected graph.
 * @param options - `maximum` for a maximum spanning tree.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept spanning-trees
 * @example
 * ```ts
 * import { boruvkaAlgorithm, graph } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = graph([[0, 1, 4], [0, 2, 1], [1, 2, 2], [1, 3, 5], [2, 3, 8]])
 * const run = runAlgorithm(boruvkaAlgorithm(g), { maxSteps: 10 })
 * run.state.weight // => 8
 * ```
 */
export function boruvkaAlgorithm(
  g: Graph,
  options: SpanningTreeAlgorithmOptions = {},
): Algorithm<SpanningTreeState, SpanningTreePath> {
  checkUndirected(g)
  const maximum = options.maximum ?? false
  const before = edgeOrder(g, maximum)
  /** The cheapest edge leaving each component, each once, in edge order. */
  const round = (uf: UnionFind): number[] => {
    const cheapest = new Map<number, number>()
    for (const [k, e] of g.edges.entries()) {
      const a = uf.find(e.source)
      const b = uf.find(e.target)
      if (a === b) continue
      for (const c of [a, b]) {
        const old = cheapest.get(c)
        if (old === undefined || before(k, old) < 0) cheapest.set(c, k)
      }
    }
    return [...new Set(cheapest.values())].sort((p, q) => p - q)
  }
  return base('graph/boruvka', g, maximum, {
    init: () => {
      const s = emptyState(g)
      return { ...s, pending: round(s.components) }
    },
    step: (s, ctx) => {
      const k = s.pending.shift()
      if (k === undefined) return s
      const next = consider(g, s, k)
      if (next.pending.length === 0) next.pending.push(...round(next.components))
      logStep(next, g.n, ctx)
      return next
    },
    done: (s) => (s.pending.length === 0 ? 'converged' : undefined),
  })
}

/** What a spanning-tree method is built from: a graph and the `maximum` flag. */
export type SpanningTreeBuilder = (
  g: Graph,
  options?: SpanningTreeAlgorithmOptions,
) => Algorithm<SpanningTreeState, SpanningTreePath>

/** What a spanning-tree method can do, for validity rules and selectors. */
export interface SpanningTreeProperties {
  /** What one step is: an edge considered in weight order, a node joined, or a component's cheapest edge. */
  readonly step: 'edge' | 'node' | 'component-edge'
  /** Whether the tree grows from one root at a time (Prim) or as a forest of fragments. */
  readonly grows: 'from-root' | 'fragments'
}

/**
 * The spanning-tree methods as one slot: `kruskal` (the edges in weight order, a union-find for cycles), `prim` (a tree
 * grown from a root by its cheapest frontier edge) and `boruvka` (every component takes its cheapest edge, in rounds).
 *
 * @concept spanning-trees
 * @example
 * ```ts
 * import { graph, spanningTreeMethods } from 'mdml-engine/graph'
 * spanningTreeMethods.make({ kind: 'prim' })(graph([[0, 1]])).name // => 'graph/prim'
 * ```
 */
export const spanningTreeMethods: OptionTable<SpanningTreeBuilder, SpanningTreeProperties> = optionTable(
  {
    kruskal: knobOption<SpanningTreeBuilder, SpanningTreeProperties>(
      'consider the edges in weight order and keep each that joins two components',
      { step: 'edge', grows: 'fragments' },
      dict({}),
      () => (g, options) => kruskalAlgorithm(g, options),
    ),
    prim: knobOption<SpanningTreeBuilder, SpanningTreeProperties>(
      'grow one tree from a root by its cheapest edge to a new node',
      { step: 'node', grows: 'from-root' },
      dict({}),
      () => (g, options) => primAlgorithm(g, options),
    ),
    boruvka: knobOption<SpanningTreeBuilder, SpanningTreeProperties>(
      'every component takes its cheapest outgoing edge, round by round',
      { step: 'component-edge', grows: 'fragments' },
      dict({}),
      () => (g, options) => boruvkaAlgorithm(g, options),
    ),
  },
  {
    default: 'kruskal',
    label: 'spanning-tree method',
    description:
      'How a minimum (or maximum) spanning forest is built: by sorted edges, by growing a tree, or by rounds.',
  },
)

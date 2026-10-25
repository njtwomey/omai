/**
 * The graph data type, modelled on NetworkX's Graph and DiGraph: nodes with labels and data, weighted edges and sorted
 * compressed-sparse-row adjacency in both directions. Construction from edge lists, neighbour and degree queries, edge
 * lookup, induced subgraphs, reversal and conversion between directed and undirected forms, and a description of a
 * graph as a drawable structure.
 */

import { type Json, WhiteboxError } from '../base/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure } from '../scene/index.ts'
import type { RunOptions } from '../trace/index.ts'
import { assemble } from './_build.ts'

/**
 * How a traced graph algorithm is run: the tracer and policy (or a parent run), a run id, and a seed or key for the
 * randomised ones. Untraced calls need none of these.
 */
export type GraphRunOptions = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId' | 'seed' | 'key'>>

/** A node reference: an index `0 .. n - 1`, or a label. */
export type NodeId = number | string

/** An edge of a {@link Graph}: end indices and a weight (1 when none was given). */
export interface Edge {
  readonly source: number
  readonly target: number
  readonly weight: number
}

/**
 * One direction of adjacency in compressed sparse row form (as scipy.sparse.csgraph): the neighbours of node `v` are
 * `indices[indptr[v] .. indptr[v + 1]]`, sorted by index, with their edge weights and positions in `graph.edges`.
 */
export interface Adjacency {
  readonly indptr: Int32Array
  readonly indices: Int32Array
  readonly weights: Float64Array
  /** The position in `graph.edges` of each entry. */
  readonly edges: Int32Array
}

/**
 * A graph as data (NetworkX Graph and DiGraph): `n` nodes with labels and data, weighted edges, directedness and
 * attributes, plus sorted adjacency in both directions. Nodes are the indices `0 .. n - 1`; every function that takes a
 * node also accepts its label. There is at most one edge per ordered pair (per unordered pair when undirected); an
 * undirected self-loop appears once in its node's adjacency and counts twice towards its degree, as in NetworkX.
 *
 * @concept graphs
 */
export interface Graph {
  readonly kind: 'graph'
  readonly directed: boolean
  readonly n: number
  readonly labels: readonly string[]
  /** The edges; for undirected graphs `source <= target`. */
  readonly edges: readonly Edge[]
  /** Successors (neighbours when undirected). */
  readonly out: Adjacency
  /** Predecessors (the same object as `out` when undirected). */
  readonly in: Adjacency
  readonly attributes: Readonly<Record<string, Json>>
  readonly nodeData: readonly Readonly<Record<string, Json>>[]
  readonly edgeData: readonly Readonly<Record<string, Json>>[]
  /** Label to index. */
  readonly index: ReadonlyMap<string, number>
}

/** An edge as given to {@link graph}: `[u, v]`, `[u, v, weight]` or an object with optional weight and data. */
export type EdgeInput =
  | readonly [NodeId, NodeId]
  | readonly [NodeId, NodeId, number]
  | {
      readonly source: NodeId
      readonly target: NodeId
      readonly weight?: number
      readonly data?: Readonly<Record<string, Json>>
    }

/** Options of {@link graph}. */
export interface GraphOptions {
  readonly directed?: boolean
  /** A node count (labels `'0'`, `'1'`, ...), or the node labels in order (isolated nodes included). */
  readonly nodes?: number | readonly NodeId[]
  /** What a repeated edge does: keep the last weight (default, as NetworkX), sum, take the min or max, or throw. */
  readonly duplicates?: 'last' | 'sum' | 'min' | 'max' | 'error'
  readonly attributes?: Readonly<Record<string, Json>>
  /** Data per node, in node order. */
  readonly nodeData?: readonly Readonly<Record<string, Json>>[]
}

/**
 * Build a graph from edges (NetworkX `Graph(edges)` / `DiGraph(edges)`). Numeric node ids are indices (the node count
 * grows to cover them); string ids are labels, added in order of first mention.
 *
 * @param edges - `[u, v]`, `[u, v, weight]` or `{ source, target, weight, data }`.
 * @param options - Directedness, the nodes (a count or labels), what repeated edges do, attributes and node data.
 * @returns The graph.
 * @throws WhiteboxError with code `value` for a negative or non-integer node index, a non-finite weight, or a repeated
 *   edge when `duplicates` is `'error'`.
 * @concept graphs
 * @example
 * ```ts
 * import { graph } from 'whitebox-engine/graph'
 * const g = graph([['a', 'b'], ['b', 'c', 2]])
 * g.n // => 3
 * g.edges.map((e) => e.weight) // => [1, 2]
 * graph([[0, 1]], { nodes: 4 }).n // => 4
 * ```
 */
export function graph(edges: readonly EdgeInput[], options: GraphOptions = {}): Graph {
  const directed = options.directed ?? false
  const duplicates = options.duplicates ?? 'last'
  const labels: string[] =
    typeof options.nodes === 'number'
      ? Array.from({ length: options.nodes }, (_, i) => String(i))
      : (options.nodes ?? []).map((v) => String(v))
  const index = new Map<string, number>()
  for (const [i, l] of labels.entries()) if (!index.has(l)) index.set(l, i)
  const resolve = (v: NodeId): number => {
    if (typeof v === 'number') {
      if (!Number.isInteger(v) || v < 0) throw new WhiteboxError('value', `graph: node index ${v} is not valid`)
      while (labels.length <= v) {
        index.set(String(labels.length), labels.length)
        labels.push(String(labels.length))
      }
      return v
    }
    const i = index.get(v)
    if (i !== undefined) return i
    index.set(v, labels.length)
    labels.push(v)
    return labels.length - 1
  }
  const out: Edge[] = []
  const data: Readonly<Record<string, Json>>[] = []
  const seen = new Map<string, number>()
  for (const e of edges) {
    const [s, t, w, d] = Array.isArray(e)
      ? [e[0] as NodeId, e[1] as NodeId, (e[2] as number | undefined) ?? 1, undefined]
      : [
          (e as { source: NodeId }).source,
          (e as { target: NodeId }).target,
          (e as { weight?: number }).weight ?? 1,
          (e as { data?: Readonly<Record<string, Json>> }).data,
        ]
    if (!Number.isFinite(w)) throw new WhiteboxError('value', `graph: edge weight ${w} is not finite`)
    let u = resolve(s)
    let v = resolve(t)
    if (!directed && u > v) [u, v] = [v, u]
    const key = `${u},${v}`
    const at = seen.get(key)
    if (at === undefined) {
      seen.set(key, out.length)
      out.push({ source: u, target: v, weight: w })
      data.push(d ?? {})
      continue
    }
    if (duplicates === 'error') throw new WhiteboxError('value', `graph: repeated edge (${s}, ${t})`)
    const old = out[at]?.weight ?? 0
    const weight =
      duplicates === 'sum'
        ? old + w
        : duplicates === 'min'
          ? Math.min(old, w)
          : duplicates === 'max'
            ? Math.max(old, w)
            : w
    out[at] = { source: u, target: v, weight }
    data[at] = { ...data[at], ...d }
  }
  return assemble(directed, labels, out, data, options.attributes ?? {}, options.nodeData ?? [])
}

/**
 * Build a directed graph (NetworkX `DiGraph(edges)`): {@link graph} with `directed: true`.
 *
 * @param edges - `[u, v]`, `[u, v, weight]` or edge objects.
 * @param options - As for {@link graph}.
 * @returns The directed graph.
 * @example
 * ```ts
 * import { digraph, successors } from 'whitebox-engine/graph'
 * successors(digraph([[0, 1], [0, 2], [2, 0]]), 0) // => [1, 2]
 * ```
 */
export function digraph(edges: readonly EdgeInput[], options: Omit<GraphOptions, 'directed'> = {}): Graph {
  return graph(edges, { ...options, directed: true })
}

/**
 * Whether a value is a {@link Graph}.
 *
 * @param x - Any value.
 * @returns True for a graph.
 * @example
 * ```ts
 * import { graph, isGraph } from 'whitebox-engine/graph'
 * [isGraph(graph([[0, 1]])), isGraph([[0, 1], [1, 0]])] // => [true, false]
 * ```
 */
export function isGraph(x: unknown): x is Graph {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'graph'
}

/**
 * The index of a node given by index or label.
 *
 * @param g - The graph.
 * @param v - An index or a label.
 * @returns The index.
 * @throws WhiteboxError with code `value` when the node is not in the graph.
 * @example
 * ```ts
 * import { graph, nodeIndex } from 'whitebox-engine/graph'
 * nodeIndex(graph([['x', 'y']]), 'y') // => 1
 * ```
 */
export function nodeIndex(g: Graph, v: NodeId): number {
  const i = typeof v === 'number' ? v : g.index.get(v)
  if (i === undefined || !Number.isInteger(i) || i < 0 || i >= g.n) {
    throw new WhiteboxError('value', `node ${String(v)} is not in the graph`)
  }
  return i
}

function row(a: Adjacency, v: number): number[] {
  return Array.from(a.indices.subarray(a.indptr[v] ?? 0, a.indptr[v + 1] ?? 0))
}

/**
 * The neighbours of a node, sorted: successors and predecessors together for a directed graph.
 *
 * @param g - The graph.
 * @param v - The node.
 * @returns Neighbour indices, ascending and without repeats.
 * @example
 * ```ts
 * import { graph, neighbours } from 'whitebox-engine/graph'
 * neighbours(graph([[0, 1], [1, 2], [1, 3]]), 1) // => [0, 2, 3]
 * ```
 */
export function neighbours(g: Graph, v: NodeId): number[] {
  const i = nodeIndex(g, v)
  if (!g.directed) return row(g.out, i)
  return [...new Set([...row(g.out, i), ...row(g.in, i)])].sort((a, b) => a - b)
}

/**
 * The successors of a node (its neighbours when undirected).
 *
 * @param g - The graph.
 * @param v - The node.
 * @returns Successor indices, ascending.
 * @example
 * ```ts
 * import { digraph, successors } from 'whitebox-engine/graph'
 * successors(digraph([[1, 0], [1, 2]]), 1) // => [0, 2]
 * ```
 */
export function successors(g: Graph, v: NodeId): number[] {
  return row(g.out, nodeIndex(g, v))
}

/**
 * The predecessors of a node (its neighbours when undirected).
 *
 * @param g - The graph.
 * @param v - The node.
 * @returns Predecessor indices, ascending.
 * @example
 * ```ts
 * import { digraph, predecessors } from 'whitebox-engine/graph'
 * predecessors(digraph([[1, 0], [2, 0]]), 0) // => [1, 2]
 * ```
 */
export function predecessors(g: Graph, v: NodeId): number[] {
  return row(g.in, nodeIndex(g, v))
}

function sideDegree(g: Graph, a: Adjacency, weighted: boolean, selfTwice: boolean): number[] {
  return Array.from({ length: g.n }, (_, v) => {
    let d = 0
    for (let k = a.indptr[v] ?? 0; k < (a.indptr[v + 1] ?? 0); k++) {
      const w = weighted ? (a.weights[k] ?? 0) : 1
      d += selfTwice && a.indices[k] === v ? 2 * w : w
    }
    return d
  })
}

/**
 * Node degrees (NetworkX `G.degree`): for a directed graph in-degree plus out-degree; an undirected self-loop counts
 * twice. With `weighted`, the sums of incident edge weights (the strengths).
 *
 * @param g - The graph.
 * @param options - `weighted` (default false).
 * @returns The degree of each node.
 * @example
 * ```ts
 * import { degree, graph } from 'whitebox-engine/graph'
 * degree(graph([[0, 1], [1, 2, 5]])) // => [1, 2, 1]
 * degree(graph([[0, 1], [1, 2, 5]]), { weighted: true }) // => [1, 6, 5]
 * ```
 */
export function degree(g: Graph, options: { readonly weighted?: boolean } = {}): number[] {
  const weighted = options.weighted ?? false
  if (!g.directed) return sideDegree(g, g.out, weighted, true)
  const o = sideDegree(g, g.out, weighted, false)
  const i = sideDegree(g, g.in, weighted, false)
  return o.map((d, v) => d + (i[v] ?? 0))
}

/**
 * In-degrees (the degrees when undirected).
 *
 * @param g - The graph.
 * @param options - `weighted` (default false).
 * @returns The in-degree of each node.
 * @example
 * ```ts
 * import { digraph, inDegree } from 'whitebox-engine/graph'
 * inDegree(digraph([[0, 1], [2, 1]])) // => [0, 2, 0]
 * ```
 */
export function inDegree(g: Graph, options: { readonly weighted?: boolean } = {}): number[] {
  return g.directed ? sideDegree(g, g.in, options.weighted ?? false, false) : degree(g, options)
}

/**
 * Out-degrees (the degrees when undirected).
 *
 * @param g - The graph.
 * @param options - `weighted` (default false).
 * @returns The out-degree of each node.
 * @example
 * ```ts
 * import { digraph, outDegree } from 'whitebox-engine/graph'
 * outDegree(digraph([[0, 1], [0, 2]])) // => [2, 0, 0]
 * ```
 */
export function outDegree(g: Graph, options: { readonly weighted?: boolean } = {}): number[] {
  return g.directed ? sideDegree(g, g.out, options.weighted ?? false, false) : degree(g, options)
}

/**
 * The weight of the edge from `u` to `v` (either way when undirected), found by binary search in the sorted
 * adjacency.
 *
 * @param g - The graph.
 * @param u - The source.
 * @param v - The target.
 * @returns The weight, or `undefined` when there is no such edge.
 * @example
 * ```ts
 * import { edgeWeight, graph } from 'whitebox-engine/graph'
 * edgeWeight(graph([[0, 1, 2.5]]), 1, 0) // => 2.5
 * edgeWeight(graph([[0, 1, 2.5]]), 0, 0) // => undefined
 * ```
 */
export function edgeWeight(g: Graph, u: NodeId, v: NodeId): number | undefined {
  const a = nodeIndex(g, u)
  const b = nodeIndex(g, v)
  let lo = g.out.indptr[a] ?? 0
  let hi = (g.out.indptr[a + 1] ?? 0) - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const x = g.out.indices[mid] ?? 0
    if (x === b) return g.out.weights[mid]
    if (x < b) lo = mid + 1
    else hi = mid - 1
  }
  return undefined
}

/**
 * Whether there is an edge from `u` to `v` (either way when undirected).
 *
 * @param g - The graph.
 * @param u - The source.
 * @param v - The target.
 * @returns True when the edge exists.
 * @example
 * ```ts
 * import { digraph, hasEdge } from 'whitebox-engine/graph'
 * hasEdge(digraph([[0, 1]]), 0, 1) // => true
 * hasEdge(digraph([[0, 1]]), 1, 0) // => false
 * ```
 */
export function hasEdge(g: Graph, u: NodeId, v: NodeId): boolean {
  return edgeWeight(g, u, v) !== undefined
}

/**
 * The subgraph induced by some nodes (NetworkX `G.subgraph(nodes)`), renumbered `0 .. k - 1` in the given order, with
 * labels, node data and edge data kept.
 *
 * @param g - The graph.
 * @param nodes - The nodes to keep.
 * @returns The induced subgraph.
 * @example
 * ```ts
 * import { graph, subgraph } from 'whitebox-engine/graph'
 * const h = subgraph(graph([['a', 'b'], ['b', 'c'], ['c', 'a'], ['c', 'd']]), ['a', 'c', 'd'])
 * h.labels // => ['a', 'c', 'd']
 * h.edges.length // => 2
 * ```
 */
export function subgraph(g: Graph, nodes: readonly NodeId[]): Graph {
  const keep = [...new Set(nodes.map((v) => nodeIndex(g, v)))]
  const map = new Int32Array(g.n).fill(-1)
  for (const [i, v] of keep.entries()) map[v] = i
  const edges: Edge[] = []
  const data: Readonly<Record<string, Json>>[] = []
  for (const [k, e] of g.edges.entries()) {
    let s = map[e.source] ?? -1
    let t = map[e.target] ?? -1
    if (s < 0 || t < 0) continue
    if (!g.directed && s > t) [s, t] = [t, s]
    edges.push({ source: s, target: t, weight: e.weight })
    data.push(g.edgeData[k] ?? {})
  }
  return assemble(
    g.directed,
    keep.map((v) => g.labels[v] ?? String(v)),
    edges,
    data,
    g.attributes,
    keep.map((v) => g.nodeData[v] ?? {}),
  )
}

/**
 * The graph with every edge reversed (NetworkX `G.reverse()`); an undirected graph is returned unchanged.
 *
 * @param g - The graph.
 * @returns The reversed graph.
 * @example
 * ```ts
 * import { digraph, reverse, successors } from 'whitebox-engine/graph'
 * successors(reverse(digraph([[0, 1], [0, 2]])), 2) // => [0]
 * ```
 */
export function reverse(g: Graph): Graph {
  if (!g.directed) return g
  const edges = g.edges.map((e) => ({ source: e.target, target: e.source, weight: e.weight }))
  return assemble(true, g.labels, edges, g.edgeData, g.attributes, g.nodeData)
}

/**
 * The undirected graph with an edge wherever there is an edge either way (NetworkX `G.to_undirected()`). Reciprocal
 * edges become one edge whose weight is combined by `combine` (default `'max'`, so unweighted graphs stay unweighted).
 *
 * @param g - The graph.
 * @param options - How to combine the weights of reciprocal edges: `'max'`, `'min'`, `'sum'` or `'mean'`.
 * @returns The undirected graph.
 * @example
 * ```ts
 * import { digraph, toUndirected } from 'whitebox-engine/graph'
 * toUndirected(digraph([[0, 1, 2], [1, 0, 3]]), { combine: 'sum' }).edges // => [{ source: 0, target: 1, weight: 5 }]
 * ```
 */
export function toUndirected(g: Graph, options: { readonly combine?: 'max' | 'min' | 'sum' | 'mean' } = {}): Graph {
  if (!g.directed) return g
  const combine = options.combine ?? 'max'
  const at = new Map<string, number>()
  const edges: Edge[] = []
  const data: Readonly<Record<string, Json>>[] = []
  const counts: number[] = []
  for (const [k, e] of g.edges.entries()) {
    const s = Math.min(e.source, e.target)
    const t = Math.max(e.source, e.target)
    const key = `${s},${t}`
    const i = at.get(key)
    if (i === undefined) {
      at.set(key, edges.length)
      edges.push({ source: s, target: t, weight: e.weight })
      data.push(g.edgeData[k] ?? {})
      counts.push(1)
      continue
    }
    const old = edges[i]?.weight ?? 0
    const w = combine === 'max' ? Math.max(old, e.weight) : combine === 'min' ? Math.min(old, e.weight) : old + e.weight
    edges[i] = { source: s, target: t, weight: w }
    data[i] = { ...data[i], ...g.edgeData[k] }
    counts[i] = (counts[i] ?? 1) + 1
  }
  const final = combine === 'mean' ? edges.map((e, i) => ({ ...e, weight: e.weight / (counts[i] ?? 1) })) : edges
  return assemble(false, g.labels, final, data, g.attributes, g.nodeData)
}

/**
 * The directed graph with both orientations of every undirected edge (NetworkX `G.to_directed()`); a directed graph is
 * returned unchanged.
 *
 * @param g - The graph.
 * @returns The directed graph.
 * @example
 * ```ts
 * import { graph, toDirected } from 'whitebox-engine/graph'
 * toDirected(graph([[0, 1], [1, 2]])).edges.length // => 4
 * ```
 */
export function toDirected(g: Graph): Graph {
  if (g.directed) return g
  const edges: Edge[] = []
  const data: Readonly<Record<string, Json>>[] = []
  for (const [k, e] of g.edges.entries()) {
    edges.push(e)
    data.push(g.edgeData[k] ?? {})
    if (e.source !== e.target) {
      edges.push({ source: e.target, target: e.source, weight: e.weight })
      data.push(g.edgeData[k] ?? {})
    }
  }
  return assemble(true, g.labels, edges, data, g.attributes, g.nodeData)
}

/**
 * The edges as `[source, target, weight]` triples, in edge order.
 *
 * @param g - The graph.
 * @returns The triples.
 * @example
 * ```ts
 * import { edgeList, graph } from 'whitebox-engine/graph'
 * edgeList(graph([[1, 0, 3]])) // => [[0, 1, 3]]
 * ```
 */
export function edgeList(g: Graph): [number, number, number][] {
  return g.edges.map((e) => [e.source, e.target, e.weight])
}

/**
 * Describe a graph as a {@link Structure} for drawing: one node per graph node (labelled, with its data and any
 * highlight), one edge per graph edge (labelled with its weight when the graph is weighted).
 *
 * @param g - The graph.
 * @param options - A title, node groups (a community per node, drawn as clusters), highlighted nodes and edges, and
 *   whether to label edges with weights (default: when some weight is not 1).
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { graph, toStructure } from 'whitebox-engine/graph'
 * const s = toStructure(graph([['a', 'b', 2]]))
 * s.nodes.map((n) => n.label) // => ['a', 'b']
 * s.edges[0]?.label // => '2'
 * ```
 */
export function toStructure(
  g: Graph,
  options: {
    readonly title?: string
    readonly groups?: ArrayLike<number>
    readonly highlight?: readonly NodeId[]
    readonly edgeLabels?: boolean
  } = {},
): Structure {
  const highlighted = new Set((options.highlight ?? []).map((v) => nodeIndex(g, v)))
  const labelled = options.edgeLabels ?? g.edges.some((e) => e.weight !== 1)
  const nodes: StructureNode[] = g.labels.map((label, i) => {
    const group = options.groups?.[i]
    const data = g.nodeData[i] ?? {}
    return {
      id: `n${i}`,
      label,
      role: 'other',
      ...(group === undefined ? {} : { group: `c${group}` }),
      ...(Object.keys(data).length > 0 ? { data } : {}),
    }
  })
  const edges: StructureEdge[] = g.edges.map((e) => ({
    from: `n${e.source}`,
    to: `n${e.target}`,
    directed: g.directed,
    ...(labelled ? { label: String(e.weight) } : {}),
  }))
  const groups =
    options.groups === undefined
      ? []
      : [...new Set(Array.from(options.groups))]
          .sort((a, b) => a - b)
          .map((c) => ({ id: `c${c}`, label: `community ${c}`, kind: 'cluster' as const }))
  return structure(g.directed ? 'digraph' : 'graph', nodes, edges, {
    groups,
    ...(options.title === undefined ? {} : { title: options.title }),
    ...(highlighted.size > 0
      ? {
          annotations: [...highlighted].map((v) => ({
            target: `n${v}`,
            text: 'highlighted',
            kind: 'highlight' as const,
          })),
        }
      : {}),
  })
}

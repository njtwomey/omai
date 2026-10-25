/**
 * Graph traversal: breadth-first and depth-first search as traced algorithms (BFS by layers with distances and parents,
 * DFS with pre- and postorder and tree, back, forward and cross edge classification), and what follows from them:
 * topological sorting, acyclicity and cycle finding, bipartite colouring and k-hop neighbourhoods.
 */

import { type Json, WhiteboxError } from '../base/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import type { Adjacency, Graph, GraphRunOptions, NodeId } from './graph.ts'
import { nodeIndex } from './graph.ts'
import { IndexedQueue } from './structures.ts'

/** Which edges a traversal follows in a directed graph: out-edges (default), in-edges, or both. */
export type Direction = 'out' | 'in' | 'both'

function sides(g: Graph, direction: Direction): readonly Adjacency[] {
  if (!g.directed || direction === 'out') return [g.out]
  return direction === 'in' ? [g.in] : [g.out, g.in]
}

/** The state of a breadth-first search: one layer per step. */
export interface BfsState {
  /** Hops from the nearest source (-1 when not reached). */
  readonly distance: Int32Array
  /** The node each was discovered from (-1 for sources and unreached nodes). */
  readonly parent: Int32Array
  /** Nodes in discovery order. */
  readonly order: readonly number[]
  /** The nodes discovered last (the next layer to expand). */
  readonly frontier: readonly number[]
  readonly layers: readonly (readonly number[])[]
}

/** The result of {@link bfs}. */
export interface BfsResult {
  readonly order: readonly number[]
  readonly distance: Int32Array
  readonly parent: Int32Array
  readonly layers: readonly (readonly number[])[]
}

/**
 * Breadth-first search as a traced algorithm: each step expands one layer, discovering neighbours in adjacency order
 * (ascending index), so the order matches scipy's `breadth_first_order`.
 *
 * @param g - The graph.
 * @param sources - The start node or nodes.
 * @param options - Which edges to follow in a directed graph, and a depth limit.
 * @returns The algorithm.
 * @concept graph-traversal
 * @example
 * ```ts
 * import { bfsAlgorithm, pathGraph } from 'whitebox-engine/graph'
 * bfsAlgorithm(pathGraph(3), 0).name // => 'bfs'
 * ```
 */
export function bfsAlgorithm(
  g: Graph,
  sources: NodeId | readonly NodeId[],
  options: { readonly direction?: Direction; readonly maxDepth?: number } = {},
): Algorithm<BfsState> {
  const starts = [...new Set((Array.isArray(sources) ? sources : [sources]).map((v: NodeId) => nodeIndex(g, v)))]
  const adj = sides(g, options.direction ?? 'out')
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  return {
    name: 'bfs',
    config: { sources: starts, direction: options.direction ?? 'out' } as Json,
    schema: channels({
      metrics: {
        visited: { description: 'nodes discovered so far' },
        frontier: { description: 'size of the newest layer' },
      },
      messages: { layer: { description: 'the newest layer', payload: { depth: 'hops', nodes: 'node indices' } } },
    }),
    init: () => {
      const distance = new Int32Array(g.n).fill(-1)
      const parent = new Int32Array(g.n).fill(-1)
      for (const s of starts) distance[s] = 0
      return { distance, parent, order: starts, frontier: starts, layers: [starts] }
    },
    step: (s) => {
      const distance = Int32Array.from(s.distance)
      const parent = Int32Array.from(s.parent)
      const next: number[] = []
      for (const u of s.frontier) {
        for (const a of adj) {
          for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
            const v = a.indices[k] ?? 0
            if (distance[v] !== -1) continue
            distance[v] = (distance[u] ?? 0) + 1
            parent[v] = u
            next.push(v)
          }
        }
      }
      return { distance, parent, order: [...s.order, ...next], frontier: next, layers: [...s.layers, next] }
    },
    done: (s) => (s.frontier.length === 0 || s.layers.length - 1 >= maxDepth ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { visited: s.order.length, frontier: s.frontier.length },
      messages: [{ kind: 'layer', payload: { depth: s.layers.length - 1, nodes: [...s.frontier] } }],
    }),
  }
}

/**
 * Breadth-first search (scipy.sparse.csgraph.breadth_first_order, NetworkX bfs_layers): discovery order, hop
 * distances, the BFS tree as parents, and the layers.
 *
 * @param g - The graph.
 * @param sources - The start node or nodes.
 * @param options - Edge direction, a depth limit, and tracing.
 * @returns Order, distances (-1 when unreached), parents (-1 for roots and unreached nodes) and layers.
 * @concept graph-traversal
 * @example
 * ```ts
 * import { bfs, graph } from 'whitebox-engine/graph'
 * const r = bfs(graph([[0, 1], [0, 2], [1, 3], [2, 3]]), 0)
 * r.order // => [0, 1, 2, 3]
 * Array.from(r.distance) // => [0, 1, 1, 2]
 * r.layers // => [[0], [1, 2], [3]]
 * ```
 */
export function bfs(
  g: Graph,
  sources: NodeId | readonly NodeId[],
  options: { readonly direction?: Direction; readonly maxDepth?: number } & GraphRunOptions = {},
): BfsResult {
  const { direction, maxDepth, ...run } = options
  const algorithm = bfsAlgorithm(g, sources, {
    ...(direction === undefined ? {} : { direction }),
    ...(maxDepth === undefined ? {} : { maxDepth }),
  })
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: g.n + 1 })
  const layers = state.layers.filter((l) => l.length > 0)
  return { order: state.order, distance: state.distance, parent: state.parent, layers }
}

/** The kind of an edge in a depth-first search: tree, back (to an ancestor), forward (to a descendant) or cross. */
export type DfsEdgeKind = 'tree' | 'back' | 'forward' | 'cross'

/** An edge examined by a depth-first search, with its classification. */
export interface DfsEdge {
  readonly source: number
  readonly target: number
  readonly kind: DfsEdgeKind
}

/**
 * The state of a depth-first search: the explicit stack and the times so far. Steps update the arrays in place (the
 * search only ever adds to them), so keep a copy of a state you want to look at later.
 */
export interface DfsState {
  /** Stack frames: a node and the position of its next adjacency entry. */
  readonly stack: [number, number][]
  /** 0 white (unseen), 1 grey (on the stack), 2 black (finished). */
  readonly colour: Uint8Array
  readonly discovery: Int32Array
  readonly finish: Int32Array
  readonly parent: Int32Array
  readonly preorder: number[]
  readonly postorder: number[]
  readonly edges: DfsEdge[]
  /** Undirected graphs: whether a node has skipped the edge back to its parent. */
  readonly skipped: Uint8Array
  /** The next candidate root. */
  readonly nextRoot: number
  readonly time: number
  /** What the last step did. */
  readonly event?: {
    readonly kind: DfsEdgeKind | 'discover' | 'finish'
    readonly source: number
    readonly target: number
  }
}

/** The result of {@link dfs}. */
export interface DfsResult {
  readonly preorder: readonly number[]
  readonly postorder: readonly number[]
  readonly parent: Int32Array
  /** Discovery and finish times (-1 when unreached), counting both events on one clock. */
  readonly discovery: Int32Array
  readonly finish: Int32Array
  /** Every examined edge, classified, in the order examined (each undirected edge once). */
  readonly edges: readonly DfsEdge[]
}

/**
 * Depth-first search as a traced algorithm: each step discovers a node, examines one edge or finishes a node.
 * Neighbours are visited in ascending index order. Edges are classified as tree, back, forward or cross (undirected
 * graphs have only tree and back edges, and each edge is reported once, from the side that examines it first).
 *
 * @param g - The graph.
 * @param source - The start node; omitted, every node is a root in index order (a depth-first forest).
 * @returns The algorithm.
 * @concept graph-traversal
 * @example
 * ```ts
 * import { dfsAlgorithm, pathGraph } from 'whitebox-engine/graph'
 * dfsAlgorithm(pathGraph(3)).name // => 'dfs'
 * ```
 */
export function dfsAlgorithm(g: Graph, source?: NodeId): Algorithm<DfsState> {
  const only = source === undefined ? undefined : nodeIndex(g, source)
  const a = g.out
  const pushNode = (s: DfsState, v: number, from: number, time: number): number => {
    s.colour[v] = 1
    s.discovery[v] = time
    s.parent[v] = from
    s.preorder.push(v)
    s.stack.push([v, a.indptr[v] ?? 0])
    return time + 1
  }
  return {
    name: 'dfs',
    config: { source: only ?? null } as Json,
    schema: channels({
      metrics: { visited: { description: 'nodes discovered so far' }, depth: { description: 'stack depth' } },
      messages: {
        event: { description: 'what the step did', payload: { kind: 'event', source: 'node', target: 'node' } },
      },
    }),
    init: () => ({
      stack: [],
      colour: new Uint8Array(g.n),
      discovery: new Int32Array(g.n).fill(-1),
      finish: new Int32Array(g.n).fill(-1),
      parent: new Int32Array(g.n).fill(-1),
      preorder: [],
      postorder: [],
      edges: [],
      skipped: new Uint8Array(g.n),
      nextRoot: only ?? 0,
      time: 0,
    }),
    step: (s) => {
      for (;;) {
        const top = s.stack[s.stack.length - 1]
        if (top === undefined) {
          let r = s.nextRoot
          while (r < g.n && s.colour[r] !== 0) r++
          const time = pushNode(s, r, -1, s.time)
          return { ...s, nextRoot: r + 1, time, event: { kind: 'discover', source: r, target: r } }
        }
        const [u, k] = top
        if (k >= (a.indptr[u + 1] ?? 0)) {
          s.stack.pop()
          s.colour[u] = 2
          s.finish[u] = s.time
          s.postorder.push(u)
          return { ...s, time: s.time + 1, event: { kind: 'finish', source: u, target: u } }
        }
        top[1] = k + 1
        const v = a.indices[k] ?? 0
        const c = s.colour[v] ?? 0
        // An undirected graph skips the reverse of the tree edge it came in on (once: there are no parallel edges).
        if (!g.directed && v === s.parent[u] && s.skipped[u] === 0) {
          s.skipped[u] = 1
          continue
        }
        if (c === 0) {
          s.edges.push({ source: u, target: v, kind: 'tree' })
          const time = pushNode(s, v, u, s.time)
          return { ...s, time, event: { kind: 'tree', source: u, target: v } }
        }
        // A finished neighbour in an undirected graph: the edge was reported from its other end.
        if (c === 2 && !g.directed) continue
        const kind: DfsEdgeKind = c === 1 ? 'back' : (s.discovery[u] ?? 0) < (s.discovery[v] ?? 0) ? 'forward' : 'cross'
        s.edges.push({ source: u, target: v, kind })
        return { ...s, event: { kind, source: u, target: v } }
      }
    },
    done: (s) => {
      if (s.stack.length > 0) return undefined
      if (only !== undefined) return s.preorder.length > 0 ? 'converged' : undefined
      let r = s.nextRoot
      while (r < g.n && s.colour[r] !== 0) r++
      return r >= g.n ? 'converged' : undefined
    },
    observe: (s) => ({
      metrics: { visited: s.preorder.length, depth: s.stack.length },
      ...(s.event === undefined ? {} : { messages: [{ kind: 'event', payload: { ...s.event } }] }),
    }),
  }
}

/**
 * Depth-first search with edge classification (NetworkX dfs_preorder_nodes, dfs_postorder_nodes and
 * dfs_labeled_edges; scipy's depth_first_order): pre- and postorder, the DFS forest as parents, discovery and finish
 * times, and every examined edge classified.
 *
 * @param g - The graph.
 * @param source - The start node; omitted, a depth-first forest over all nodes in index order.
 * @param options - Tracing.
 * @returns Orders, parents, times and classified edges.
 * @concept graph-traversal
 * @example
 * ```ts
 * import { dfs, digraph } from 'whitebox-engine/graph'
 * const r = dfs(digraph([[0, 1], [1, 2], [2, 0], [0, 2]]), 0)
 * r.preorder // => [0, 1, 2]
 * r.postorder // => [2, 1, 0]
 * r.edges.map((e) => e.kind) // => ['tree', 'tree', 'back', 'forward']
 * ```
 */
export function dfs(g: Graph, source?: NodeId, options: GraphRunOptions = {}): DfsResult {
  const { state } = runAlgorithm(dfsAlgorithm(g, source), { ...options, maxSteps: 3 * (g.n + g.edges.length) + 3 })
  return {
    preorder: state.preorder,
    postorder: state.postorder,
    parent: state.parent,
    discovery: state.discovery,
    finish: state.finish,
    edges: state.edges,
  }
}

/**
 * A topological order of a directed acyclic graph (NetworkX lexicographical_topological_sort): Kahn's algorithm,
 * always taking the smallest available node, so the order is unique.
 *
 * @param g - A directed graph.
 * @returns The nodes in topological order.
 * @throws WhiteboxError with code `value` when the graph is undirected or has a cycle.
 * @concept topological-sort
 * @example
 * ```ts
 * import { digraph, topologicalSort } from 'whitebox-engine/graph'
 * topologicalSort(digraph([[2, 0], [1, 0], [2, 1]])) // => [2, 1, 0]
 * ```
 */
export function topologicalSort(g: Graph): number[] {
  if (!g.directed) throw new WhiteboxError('value', 'topologicalSort: the graph is undirected')
  const indegree = Int32Array.from({ length: g.n }, (_, v) => (g.in.indptr[v + 1] ?? 0) - (g.in.indptr[v] ?? 0))
  const ready = new IndexedQueue(g.n)
  for (let v = 0; v < g.n; v++) if (indegree[v] === 0) ready.push(v, v)
  const order: number[] = []
  for (let u = ready.pop(); u !== undefined; u = ready.pop()) {
    order.push(u)
    for (let k = g.out.indptr[u] ?? 0; k < (g.out.indptr[u + 1] ?? 0); k++) {
      const v = g.out.indices[k] ?? 0
      indegree[v] = (indegree[v] ?? 0) - 1
      if (indegree[v] === 0) ready.push(v, v)
    }
  }
  if (order.length < g.n) throw new WhiteboxError('value', 'topologicalSort: the graph has a cycle')
  return order
}

/**
 * Whether a graph is a directed acyclic graph (NetworkX is_directed_acyclic_graph).
 *
 * @param g - The graph.
 * @returns True for a directed graph without cycles (false for undirected graphs).
 * @example
 * ```ts
 * import { digraph, isDag } from 'whitebox-engine/graph'
 * [isDag(digraph([[0, 1], [1, 2]])), isDag(digraph([[0, 1], [1, 0]]))] // => [true, false]
 * ```
 */
export function isDag(g: Graph): boolean {
  return g.directed && findCycle(g) === null
}

/**
 * A cycle, as its edges in order (NetworkX find_cycle, which raises where this returns null). Directed graphs follow
 * edge directions; in undirected graphs a cycle has at least three nodes, or is a self-loop.
 *
 * @param g - The graph.
 * @param source - Search only from this node; omitted, from every node in index order.
 * @returns `[u, v]` edges forming a cycle, or null when there is none.
 * @example
 * ```ts
 * import { digraph, findCycle, graph } from 'whitebox-engine/graph'
 * findCycle(digraph([[0, 1], [1, 2], [2, 1]])) // => [[1, 2], [2, 1]]
 * findCycle(graph([[0, 1], [1, 2]])) // => null
 * ```
 */
export function findCycle(g: Graph, source?: NodeId): [number, number][] | null {
  const { parent, edges } = dfs(g, source)
  const back = edges.find((e) => e.kind === 'back')
  if (back === undefined) return null
  // Walk the tree from the back edge's source up to its target (an ancestor).
  const path: number[] = [back.source]
  for (let v = back.source; v !== back.target; ) {
    v = parent[v] ?? -1
    if (v < 0) break
    path.push(v)
  }
  path.reverse()
  const cycle: [number, number][] = []
  for (let i = 0; i + 1 < path.length; i++) cycle.push([path[i] ?? 0, path[i + 1] ?? 0])
  cycle.push([back.source, back.target])
  return cycle
}

/**
 * A two-colouring of a graph whose edges all join different colours, if there is one (NetworkX bipartite.color):
 * each component is coloured by breadth-first search from its smallest node, which gets colour 0. Edge directions are
 * ignored.
 *
 * @param g - The graph.
 * @returns Colour 0 or 1 per node, or null when the graph is not bipartite.
 * @concept bipartite-graphs
 * @example
 * ```ts
 * import { bipartiteColouring, cycleGraph } from 'whitebox-engine/graph'
 * bipartiteColouring(cycleGraph(4)) // => [0, 1, 0, 1]
 * bipartiteColouring(cycleGraph(3)) // => null
 * ```
 */
export function bipartiteColouring(g: Graph): number[] | null {
  const colour = new Int8Array(g.n).fill(-1)
  const adj = sides(g, 'both')
  for (let s = 0; s < g.n; s++) {
    if (colour[s] !== -1) continue
    colour[s] = 0
    const queue = [s]
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      for (const a of adj) {
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
          const v = a.indices[k] ?? 0
          if (colour[v] === -1) {
            colour[v] = 1 - (colour[u] ?? 0)
            queue.push(v)
          } else if (colour[v] === colour[u]) return null
        }
      }
    }
  }
  return Array.from(colour)
}

/**
 * Whether a graph is bipartite (NetworkX is_bipartite): no odd cycle, ignoring edge directions.
 *
 * @param g - The graph.
 * @returns True when the nodes split into two sets with every edge between them.
 * @concept bipartite-graphs
 * @example
 * ```ts
 * import { gridGraph, isBipartite, completeGraph } from 'whitebox-engine/graph'
 * [isBipartite(gridGraph(3, 3)), isBipartite(completeGraph(3))] // => [true, false]
 * ```
 */
export function isBipartite(g: Graph): boolean {
  return bipartiteColouring(g) !== null
}

/**
 * The nodes within `k` hops of a node, itself included (the receptive field of `k` rounds of message passing; NetworkX
 * single_source_shortest_path_length with a cutoff, or ego_graph).
 *
 * @param g - The graph.
 * @param v - The centre.
 * @param k - The number of hops.
 * @param options - Which edges to follow in a directed graph (default out-edges).
 * @returns The nodes, ascending.
 * @example
 * ```ts
 * import { kHopNeighbourhood, pathGraph } from 'whitebox-engine/graph'
 * kHopNeighbourhood(pathGraph(6), 2, 2) // => [0, 1, 2, 3, 4]
 * ```
 */
export function kHopNeighbourhood(
  g: Graph,
  v: NodeId,
  k: number,
  options: { readonly direction?: Direction } = {},
): number[] {
  return [...bfs(g, v, { ...options, maxDepth: k }).order].sort((a, b) => a - b)
}

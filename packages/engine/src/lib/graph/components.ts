import { WhiteboxError } from '../base/index.ts'
import { partitionFromLabels } from './_build.ts'
import { type Graph, graph } from './graph.ts'

/**
 * A partition of the nodes into components: a label per node and the members of each component. Components are
 * numbered in order of their smallest node, and members are sorted, so the result is canonical.
 */
export interface Components {
  readonly count: number
  readonly labels: Int32Array
  readonly components: readonly (readonly number[])[]
}

/**
 * Connected components (scipy.sparse.csgraph.connected_components, NetworkX connected_components): for a directed
 * graph, weakly connected by default, or strongly connected with `connection: 'strong'`.
 *
 * @param g - The graph.
 * @param options - `connection`: `'weak'` (default) or `'strong'` (directed graphs only).
 * @returns The canonical partition.
 * @concept connected-components
 * @example
 * ```ts
 * import { connectedComponents, graph } from 'whitebox-engine/graph'
 * const c = connectedComponents(graph([[0, 1], [2, 3]], { nodes: 5 }))
 * c.count // => 3
 * c.components // => [[0, 1], [2, 3], [4]]
 * ```
 */
export function connectedComponents(g: Graph, options: { readonly connection?: 'weak' | 'strong' } = {}): Components {
  if (options.connection === 'strong' && g.directed) return stronglyConnectedComponents(g)
  const label = new Int32Array(g.n).fill(-1)
  const sides = g.directed ? [g.out, g.in] : [g.out]
  let c = 0
  for (let s = 0; s < g.n; s++) {
    if (label[s] !== -1) continue
    label[s] = c
    const queue = [s]
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      for (const a of sides) {
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
          const v = a.indices[k] ?? 0
          if (label[v] === -1) {
            label[v] = c
            queue.push(v)
          }
        }
      }
    }
    c++
  }
  return partitionFromLabels(label)
}

/**
 * Whether a graph is connected (weakly, for a directed graph); the empty graph is not, as in NetworkX.
 *
 * @param g - The graph.
 * @returns True when there is one component.
 * @example
 * ```ts
 * import { graph, isConnected } from 'whitebox-engine/graph'
 * [isConnected(graph([[0, 1], [1, 2]])), isConnected(graph([[0, 1]], { nodes: 3 }))] // => [true, false]
 * ```
 */
export function isConnected(g: Graph): boolean {
  return g.n > 0 && connectedComponents(g).count === 1
}

function tarjan(g: Graph): Int32Array {
  const index = new Int32Array(g.n).fill(-1)
  const low = new Int32Array(g.n)
  const onStack = new Uint8Array(g.n)
  const stack: number[] = []
  const comp = new Int32Array(g.n).fill(-1)
  let counter = 0
  let found = 0
  const a = g.out
  for (let root = 0; root < g.n; root++) {
    if (index[root] !== -1) continue
    const calls: [number, number][] = []
    const enter = (v: number): void => {
      index[v] = counter
      low[v] = counter
      counter++
      stack.push(v)
      onStack[v] = 1
      calls.push([v, a.indptr[v] ?? 0])
    }
    enter(root)
    while (calls.length > 0) {
      const frame = calls[calls.length - 1] as [number, number]
      const [v, k] = frame
      if (k < (a.indptr[v + 1] ?? 0)) {
        frame[1] = k + 1
        const w = a.indices[k] ?? 0
        if (index[w] === -1) enter(w)
        else if (onStack[w] === 1) low[v] = Math.min(low[v] ?? 0, index[w] ?? 0)
        continue
      }
      calls.pop()
      const parent = calls[calls.length - 1]
      if (parent !== undefined) low[parent[0]] = Math.min(low[parent[0]] ?? 0, low[v] ?? 0)
      if (low[v] === index[v]) {
        for (;;) {
          const w = stack.pop() ?? v
          onStack[w] = 0
          comp[w] = found
          if (w === v) break
        }
        found++
      }
    }
  }
  return comp
}

function kosaraju(g: Graph): Int32Array {
  // First pass: finishing order on the graph; second: search the reversed graph in reverse finishing order.
  const seen = new Uint8Array(g.n)
  const finished: number[] = []
  for (let root = 0; root < g.n; root++) {
    if (seen[root] === 1) continue
    seen[root] = 1
    const calls: [number, number][] = [[root, g.out.indptr[root] ?? 0]]
    while (calls.length > 0) {
      const frame = calls[calls.length - 1] as [number, number]
      const [v, k] = frame
      if (k < (g.out.indptr[v + 1] ?? 0)) {
        frame[1] = k + 1
        const w = g.out.indices[k] ?? 0
        if (seen[w] === 0) {
          seen[w] = 1
          calls.push([w, g.out.indptr[w] ?? 0])
        }
        continue
      }
      calls.pop()
      finished.push(v)
    }
  }
  const comp = new Int32Array(g.n).fill(-1)
  let c = 0
  for (let i = finished.length - 1; i >= 0; i--) {
    const s = finished[i] ?? 0
    if (comp[s] !== -1) continue
    comp[s] = c
    const stack = [s]
    while (stack.length > 0) {
      const u = stack.pop() ?? 0
      for (let k = g.in.indptr[u] ?? 0; k < (g.in.indptr[u + 1] ?? 0); k++) {
        const w = g.in.indices[k] ?? 0
        if (comp[w] === -1) {
          comp[w] = c
          stack.push(w)
        }
      }
    }
    c++
  }
  return comp
}

/**
 * Strongly connected components (NetworkX strongly_connected_components, scipy's connected_components with
 * `connection='strong'`) by Tarjan's single-pass lowlink algorithm (default) or Kosaraju's two passes; both give the
 * same canonical partition. An undirected graph's strong components are its connected components.
 *
 * @param g - The graph.
 * @param options - `method`: `'tarjan'` (default) or `'kosaraju'`.
 * @returns The canonical partition.
 * @concept connected-components
 * @reference R. Tarjan, Depth-first search and linear graph algorithms, SIAM J. Computing 1 (1972).
 * @example
 * ```ts
 * import { digraph, stronglyConnectedComponents } from 'whitebox-engine/graph'
 * const g = digraph([[0, 1], [1, 0], [1, 2], [2, 3], [3, 2]])
 * stronglyConnectedComponents(g).components // => [[0, 1], [2, 3]]
 * stronglyConnectedComponents(g, { method: 'kosaraju' }).count // => 2
 * ```
 */
export function stronglyConnectedComponents(
  g: Graph,
  options: { readonly method?: 'tarjan' | 'kosaraju' } = {},
): Components {
  if (!g.directed) return connectedComponents(g)
  return partitionFromLabels(options.method === 'kosaraju' ? kosaraju(g) : tarjan(g))
}

/**
 * Whether a directed graph is strongly connected (every node reaches every other).
 *
 * @param g - The graph.
 * @returns True when there is one strong component.
 * @example
 * ```ts
 * import { cycleGraph, isStronglyConnected, pathGraph } from 'whitebox-engine/graph'
 * [isStronglyConnected(cycleGraph(4, { directed: true })), isStronglyConnected(pathGraph(3, { directed: true }))] // => [true, false]
 * ```
 */
export function isStronglyConnected(g: Graph): boolean {
  return g.n > 0 && stronglyConnectedComponents(g).count === 1
}

/** The result of {@link condensation}. */
export interface Condensation {
  /** The directed acyclic graph of strong components; node data records each component's `members`. */
  readonly graph: Graph
  /** The component of each original node. */
  readonly mapping: Int32Array
  readonly components: readonly (readonly number[])[]
}

/**
 * The condensation of a directed graph (NetworkX condensation): each strongly connected component contracted to a
 * node, giving a directed acyclic graph with an edge between components whenever some edge joins their members.
 *
 * @param g - A directed graph.
 * @returns The component DAG, the mapping from nodes to components, and the members.
 * @throws WhiteboxError with code `value` for an undirected graph.
 * @example
 * ```ts
 * import { condensation, digraph } from 'whitebox-engine/graph'
 * const c = condensation(digraph([[0, 1], [1, 0], [1, 2], [2, 3], [3, 2], [0, 3]]))
 * c.graph.edges.map((e) => [e.source, e.target]) // => [[0, 1]]
 * ```
 */
export function condensation(g: Graph): Condensation {
  if (!g.directed) throw new WhiteboxError('value', 'condensation: the graph is undirected')
  const scc = stronglyConnectedComponents(g)
  const edges: [number, number][] = []
  const seen = new Set<string>()
  for (const e of g.edges) {
    const a = scc.labels[e.source] ?? 0
    const b = scc.labels[e.target] ?? 0
    const key = `${a},${b}`
    if (a !== b && !seen.has(key)) {
      seen.add(key)
      edges.push([a, b])
    }
  }
  edges.sort((p, q) => p[0] - q[0] || p[1] - q[1])
  const nodeData = scc.components.map((members) => ({ members: [...members] }))
  return {
    graph: graph(edges, { nodes: scc.count, directed: true, nodeData }),
    mapping: scc.labels,
    components: scc.components,
  }
}

/** Articulation points, bridges and biconnected components from one Hopcroft-Tarjan search. */
interface Biconnectivity {
  readonly cut: Uint8Array
  readonly bridges: [number, number][]
  readonly blocks: [number, number][][]
}

function biconnectivity(g: Graph, where: string): Biconnectivity {
  if (g.directed) throw new WhiteboxError('value', `${where}: the graph must be undirected`)
  const a = g.out
  const disc = new Int32Array(g.n).fill(-1)
  const low = new Int32Array(g.n)
  const cut = new Uint8Array(g.n)
  const bridges: [number, number][] = []
  const blocks: [number, number][][] = []
  const edgeStack: [number, number][] = []
  let time = 0
  for (let root = 0; root < g.n; root++) {
    if (disc[root] !== -1) continue
    disc[root] = time
    low[root] = time
    time++
    let rootChildren = 0
    // Frames: node, the edge it was entered by, the next adjacency position.
    const calls: [number, number, number][] = [[root, -1, a.indptr[root] ?? 0]]
    while (calls.length > 0) {
      const frame = calls[calls.length - 1] as [number, number, number]
      const [v, via, k] = frame
      if (k < (a.indptr[v + 1] ?? 0)) {
        frame[2] = k + 1
        const w = a.indices[k] ?? 0
        const e = a.edges[k] ?? -1
        if (w === v || e === via) continue
        if (disc[w] === -1) {
          edgeStack.push([v, w])
          disc[w] = time
          low[w] = time
          time++
          calls.push([w, e, a.indptr[w] ?? 0])
        } else if ((disc[w] ?? 0) < (disc[v] ?? 0)) {
          low[v] = Math.min(low[v] ?? 0, disc[w] ?? 0)
          edgeStack.push([v, w])
        }
        continue
      }
      calls.pop()
      const parent = calls[calls.length - 1]
      if (parent === undefined) continue
      const p = parent[0]
      low[p] = Math.min(low[p] ?? 0, low[v] ?? 0)
      if (p === root) rootChildren++
      if ((low[v] ?? 0) >= (disc[p] ?? 0)) {
        if (p !== root) cut[p] = 1
        const block: [number, number][] = []
        for (;;) {
          const top = edgeStack.pop()
          if (top === undefined) break
          block.push(top)
          if (top[0] === p && top[1] === v) break
        }
        blocks.push(block)
      }
      if ((low[v] ?? 0) > (disc[p] ?? 0)) bridges.push([Math.min(p, v), Math.max(p, v)])
    }
    if (rootChildren >= 2) cut[root] = 1
  }
  return { cut, bridges, blocks }
}

/**
 * The articulation points (cut vertices) of an undirected graph: the nodes whose removal disconnects their component
 * (NetworkX articulation_points), by the Hopcroft-Tarjan lowpoint search.
 *
 * @param g - An undirected graph.
 * @returns The articulation points, ascending.
 * @throws WhiteboxError with code `value` for a directed graph.
 * @concept biconnectivity
 * @example
 * ```ts
 * import { articulationPoints, barbellGraph } from 'whitebox-engine/graph'
 * articulationPoints(barbellGraph(3, 1)) // => [2, 3, 4]
 * ```
 */
export function articulationPoints(g: Graph): number[] {
  const { cut } = biconnectivity(g, 'articulationPoints')
  return Array.from({ length: g.n }, (_, v) => v).filter((v) => cut[v] === 1)
}

/**
 * The bridges of an undirected graph: the edges whose removal disconnects their component (NetworkX bridges).
 *
 * @param g - An undirected graph.
 * @returns The bridges as `[u, v]` with `u < v`, sorted.
 * @throws WhiteboxError with code `value` for a directed graph.
 * @concept biconnectivity
 * @example
 * ```ts
 * import { barbellGraph, bridges } from 'whitebox-engine/graph'
 * bridges(barbellGraph(3, 1)) // => [[2, 3], [3, 4]]
 * ```
 */
export function bridges(g: Graph): [number, number][] {
  return biconnectivity(g, 'bridges').bridges.sort((p, q) => p[0] - q[0] || p[1] - q[1])
}

/** A biconnected component: its nodes and its edges. */
export interface BiconnectedComponent {
  readonly nodes: readonly number[]
  readonly edges: readonly (readonly [number, number])[]
}

/**
 * The biconnected components (blocks) of an undirected graph: the maximal subgraphs that stay connected after removing
 * any one node (NetworkX biconnected_components and biconnected_component_edges). Every edge lies in exactly one block;
 * a bridge is a block of its own; isolated nodes and self-loops belong to none.
 *
 * @param g - An undirected graph.
 * @returns The blocks, each with sorted nodes and sorted `[u, v]` (`u < v`) edges, ordered by their node lists.
 * @throws WhiteboxError with code `value` for a directed graph.
 * @concept biconnectivity
 * @example
 * ```ts
 * import { barbellGraph, biconnectedComponents } from 'whitebox-engine/graph'
 * biconnectedComponents(barbellGraph(3, 1)).map((b) => b.nodes) // => [[0, 1, 2], [2, 3], [3, 4], [4, 5, 6]]
 * ```
 */
export function biconnectedComponents(g: Graph): BiconnectedComponent[] {
  const { blocks } = biconnectivity(g, 'biconnectedComponents')
  const out = blocks.map((block) => {
    const edges = block
      .map(([u, v]): [number, number] => [Math.min(u, v), Math.max(u, v)])
      .sort((p, q) => p[0] - q[0] || p[1] - q[1])
    const nodes = [...new Set(block.flat())].sort((p, q) => p - q)
    return { nodes, edges }
  })
  const lexical = (p: readonly number[], q: readonly number[]): number => {
    for (let i = 0; i < Math.min(p.length, q.length); i++) if (p[i] !== q[i]) return (p[i] ?? 0) - (q[i] ?? 0)
    return p.length - q.length
  }
  return out.sort((p, q) => lexical(p.nodes, q.nodes))
}

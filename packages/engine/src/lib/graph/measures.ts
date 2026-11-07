/**
 * Whole-graph and node-level measures, NetworkX style: density, eccentricity, diameter, radius, centre and periphery;
 * triangles, clustering and transitivity; degree, numeric and attribute assortativity and label homophily; conductance,
 * edge expansion and the Cheeger constant; and the spectral quantities algebraic connectivity and spectral gap.
 */

import { MdmlError, maxOf, meanOf, minOf } from '../base/index.ts'
import { ddot, eigvalsh } from '../numerics/linalg/index.ts'
import { concreteOrThrow, type Tensor, toFlat } from '../tensor/index.ts'
import { neighbourSets } from './_neighbours.ts'
import { degree, type Graph, inDegree, type NodeId, nodeIndex, outDegree } from './graph.ts'
import { laplacian, normalisedLaplacian } from './matrices.ts'
import { dijkstra } from './paths.ts'
import { bfs } from './traversal.ts'

/**
 * The density (NetworkX density): m / (n (n - 1) / 2) for an undirected graph, m / (n (n - 1)) for a directed one.
 *
 * @param g - The graph.
 * @returns The fraction of possible edges present (0 for fewer than two nodes).
 * @concept graph-measures
 * @example
 * ```ts
 * import { completeGraph, density, pathGraph } from 'mdml-engine/graph'
 * [density(completeGraph(5)), density(pathGraph(4))] // => [1, 0.5]
 * ```
 */
export function density(g: Graph): number {
  if (g.n <= 1) return 0
  const pairs = g.n * (g.n - 1)
  return (g.directed ? 1 : 2) * (g.edges.length / pairs)
}

/**
 * The eccentricity of every node (NetworkX eccentricity): its greatest distance to another node (hops, or weighted
 * lengths with `weighted`).
 *
 * @param g - A connected (strongly connected, if directed) graph.
 * @param options - `weighted` (default false).
 * @returns The eccentricity of each node.
 * @throws MdmlError with code `value` when some node cannot reach another.
 * @concept graph-measures
 * @example
 * ```ts
 * import { eccentricity, pathGraph } from 'mdml-engine/graph'
 * Array.from(eccentricity(pathGraph(5))) // => [4, 3, 2, 3, 4]
 * ```
 */
export function eccentricity(g: Graph, options: { readonly weighted?: boolean } = {}): Float64Array {
  return Float64Array.from({ length: g.n }, (_, v) => {
    const d =
      options.weighted === true
        ? dijkstra(g, v).distance
        : Float64Array.from(bfs(g, v).distance, (x) => (x < 0 ? Number.POSITIVE_INFINITY : x))
    const e = Math.max(0, maxOf(d))
    if (e === Number.POSITIVE_INFINITY) {
      throw new MdmlError('value', 'eccentricity: the graph is not (strongly) connected')
    }
    return e
  })
}

/**
 * The diameter (NetworkX diameter): the largest eccentricity.
 *
 * @param g - A connected graph.
 * @param options - `weighted` (default false).
 * @returns The diameter.
 * @concept graph-measures
 * @example
 * ```ts
 * import { diameter, petersenGraph } from 'mdml-engine/graph'
 * diameter(petersenGraph()) // => 2
 * ```
 */
export function diameter(g: Graph, options: { readonly weighted?: boolean } = {}): number {
  return Math.max(0, maxOf(eccentricity(g, options)))
}

/**
 * The radius (NetworkX radius): the smallest eccentricity.
 *
 * @param g - A connected graph.
 * @param options - `weighted` (default false).
 * @returns The radius.
 * @concept graph-measures
 * @example
 * ```ts
 * import { pathGraph, radius } from 'mdml-engine/graph'
 * radius(pathGraph(5)) // => 2
 * ```
 */
export function radius(g: Graph, options: { readonly weighted?: boolean } = {}): number {
  return g.n === 0 ? 0 : minOf(eccentricity(g, options))
}

/**
 * The centre (NetworkX center): the nodes of smallest eccentricity.
 *
 * @param g - A connected graph.
 * @param options - `weighted` (default false).
 * @returns The central nodes, ascending.
 * @concept graph-measures
 * @example
 * ```ts
 * import { center, pathGraph } from 'mdml-engine/graph'
 * center(pathGraph(4)) // => [1, 2]
 * ```
 */
export function center(g: Graph, options: { readonly weighted?: boolean } = {}): number[] {
  const e = eccentricity(g, options)
  const r = radius(g, options)
  return Array.from({ length: g.n }, (_, v) => v).filter((v) => e[v] === r)
}

/**
 * The periphery (NetworkX periphery): the nodes whose eccentricity equals the diameter.
 *
 * @param g - A connected graph.
 * @param options - `weighted` (default false).
 * @returns The peripheral nodes, ascending.
 * @concept graph-measures
 * @example
 * ```ts
 * import { pathGraph, periphery } from 'mdml-engine/graph'
 * periphery(pathGraph(4)) // => [0, 3]
 * ```
 */
export function periphery(g: Graph, options: { readonly weighted?: boolean } = {}): number[] {
  const e = eccentricity(g, options)
  const d = diameter(g, options)
  return Array.from({ length: g.n }, (_, v) => v).filter((v) => e[v] === d)
}

/**
 * The number of triangles through each node of an undirected graph (NetworkX triangles).
 *
 * @param g - An undirected graph.
 * @returns The triangle count of each node.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept clustering
 * @example
 * ```ts
 * import { completeGraph, triangles } from 'mdml-engine/graph'
 * Array.from(triangles(completeGraph(4))) // => [3, 3, 3, 3]
 * ```
 */
export function triangles(g: Graph): Int32Array {
  if (g.directed) throw new MdmlError('value', 'triangles: the graph must be undirected')
  const nb = neighbourSets(g)
  return Int32Array.from({ length: g.n }, (_, u) => {
    const list = [...(nb[u] ?? [])]
    let t = 0
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) if (nb[list[i] ?? 0]?.has(list[j] ?? 0) === true) t++
    }
    return t
  })
}

/**
 * Local clustering coefficients (NetworkX clustering): for an undirected graph the fraction of pairs of neighbours that
 * are linked, or with `weighted` the geometric mean of the normalised triangle weights (Onnela et al., 2005); for a
 * directed graph Fagiolo's (2007) generalisation counting all orientations. Self-loops are ignored.
 *
 * @param g - The graph.
 * @param options - `weighted` (default false; weights are divided by the largest).
 * @returns The coefficient of each node (0 for degree below 2).
 * @concept clustering
 * @reference G. Fagiolo, Clustering in complex directed networks, Phys. Rev. E 76 (2007).
 * @example
 * ```ts
 * import { clustering, graph } from 'mdml-engine/graph'
 * Array.from(clustering(graph([[0, 1], [1, 2], [2, 0], [2, 3]]))) // => [1, 1, 0.3333333333333333, 0]
 * ```
 */
export function clustering(g: Graph, options: { readonly weighted?: boolean } = {}): Float64Array {
  const weighted = options.weighted ?? false
  const top = Math.max(0, maxOf(g.edges.map((e) => e.weight)))
  const w = (u: number, v: number): number => {
    if (!weighted) return 1
    // Binary search in the sorted out-adjacency.
    const a = g.out
    let lo = a.indptr[u] ?? 0
    let hi = (a.indptr[u + 1] ?? 0) - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const x = a.indices[mid] ?? 0
      if (x === v) return Math.cbrt((a.weights[mid] ?? 0) / (top || 1))
      if (x < v) lo = mid + 1
      else hi = mid - 1
    }
    return 0
  }
  const nb = neighbourSets(g)
  if (!g.directed) {
    return Float64Array.from({ length: g.n }, (_, u) => {
      const list = [...(nb[u] ?? [])]
      const d = list.length
      if (d < 2) return 0
      let t = 0
      for (let i = 0; i < d; i++) {
        const v = list[i] ?? 0
        for (let j = i + 1; j < d; j++) {
          const x = list[j] ?? 0
          if (nb[v]?.has(x) === true) t += weighted ? w(u, v) * w(v, x) * w(x, u) : 1
        }
      }
      return (2 * t) / (d * (d - 1))
    })
  }
  const a = (u: number, v: number): number => (u === v ? 0 : w(u, v) * (hasArc(g, u, v) ? 1 : 0))
  const dIn = inDegree(g)
  const dOut = outDegree(g)
  return Float64Array.from({ length: g.n }, (_, u) => {
    const list = [...(nb[u] ?? [])]
    let t = 0
    for (const v of list) {
      for (const x of list) {
        if (v === x) continue
        t += (a(u, v) + a(v, u)) * (a(v, x) + a(x, v)) * (a(x, u) + a(u, x))
      }
    }
    let self = 0
    if (hasArc(g, u, u)) self = 1
    const total = (dIn[u] ?? 0) + (dOut[u] ?? 0) - 2 * self
    let reciprocal = 0
    for (const v of list) if (hasArc(g, u, v) && hasArc(g, v, u)) reciprocal++
    const denominator = 2 * (total * (total - 1) - 2 * reciprocal)
    return t === 0 || denominator === 0 ? 0 : t / denominator
  })
}

function hasArc(g: Graph, u: number, v: number): boolean {
  const a = g.out
  let lo = a.indptr[u] ?? 0
  let hi = (a.indptr[u + 1] ?? 0) - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const x = a.indices[mid] ?? 0
    if (x === v) return true
    if (x < v) lo = mid + 1
    else hi = mid - 1
  }
  return false
}

/**
 * The average clustering coefficient (NetworkX average_clustering).
 *
 * @param g - The graph.
 * @param options - `weighted` (default false) and `countZeros` (default true: include nodes with coefficient 0).
 * @returns The mean local coefficient.
 * @concept clustering
 * @example
 * ```ts
 * import { averageClustering, completeGraph } from 'mdml-engine/graph'
 * averageClustering(completeGraph(5)) // => 1
 * ```
 */
export function averageClustering(
  g: Graph,
  options: { readonly weighted?: boolean; readonly countZeros?: boolean } = {},
): number {
  const c = Array.from(clustering(g, options)).filter((x) => options.countZeros !== false || x > 0)
  return c.length === 0 ? 0 : meanOf(c)
}

/**
 * The average shortest path length (NetworkX average_shortest_path_length): the mean number of edges on a shortest
 * path, over all ordered pairs of distinct nodes joined by a path. For a connected graph this is NetworkX's value;
 * where NetworkX raises on a disconnected graph, pairs with no path are left out here.
 *
 * @param g - The graph (edge directions followed when directed).
 * @returns The mean path length in hops (0 when no two nodes are joined).
 * @concept graph-measures
 * @example
 * ```ts
 * import { averageShortestPathLength, pathGraph } from 'mdml-engine/graph'
 * averageShortestPathLength(pathGraph(4)) // => 1.6666666666666667
 * ```
 */
export function averageShortestPathLength(g: Graph): number {
  const dist = new Int32Array(g.n)
  const queue = new Int32Array(g.n)
  let total = 0
  let pairs = 0
  for (let source = 0; source < g.n; source++) {
    dist.fill(-1)
    dist[source] = 0
    let head = 0
    let tail = 0
    queue[tail++] = source
    while (head < tail) {
      const u = queue[head++] as number
      for (let e = g.out.indptr[u] ?? 0; e < (g.out.indptr[u + 1] ?? 0); e++) {
        const v = g.out.indices[e] ?? 0
        if (dist[v] === -1) {
          dist[v] = (dist[u] ?? 0) + 1
          total += dist[v] ?? 0
          pairs += 1
          queue[tail++] = v
        }
      }
    }
  }
  return pairs === 0 ? 0 : total / pairs
}

/**
 * The transitivity (NetworkX transitivity): three times the number of triangles over the number of connected triples,
 * the global clustering coefficient.
 *
 * @param g - An undirected graph.
 * @returns The transitivity (0 when there are no triples).
 * @concept clustering
 * @example
 * ```ts
 * import { graph, transitivity } from 'mdml-engine/graph'
 * transitivity(graph([[0, 1], [1, 2], [2, 0], [2, 3]])) // => 0.6
 * ```
 */
export function transitivity(g: Graph): number {
  const t = triangles(g)
  const nb = neighbourSets(g)
  let tri = 0
  let triples = 0
  for (let u = 0; u < g.n; u++) {
    tri += t[u] ?? 0
    const d = nb[u]?.size ?? 0
    triples += (d * (d - 1)) / 2
  }
  return triples === 0 ? 0 : tri / triples
}

/** The Pearson correlation of paired values (centred, then cosine similarity by BLAS dot products). */
function pearson(xs: readonly number[], ys: readonly number[]): number {
  const mx = meanOf(xs)
  const my = meanOf(ys)
  const dx = xs.map((x) => x - mx)
  const dy = ys.map((y) => y - my)
  return ddot(dx, dy) / Math.sqrt(ddot(dx, dx) * ddot(dy, dy))
}

/** The values at the two ends of every edge (both orientations of an undirected edge). */
function edgeEnds(g: Graph, x: ArrayLike<number>, y: ArrayLike<number>): [number[], number[]] {
  const xs: number[] = []
  const ys: number[] = []
  for (const e of g.edges) {
    xs.push(x[e.source] ?? 0)
    ys.push(y[e.target] ?? 0)
    if (!g.directed && e.source !== e.target) {
      xs.push(x[e.target] ?? 0)
      ys.push(y[e.source] ?? 0)
    }
  }
  return [xs, ys]
}

/**
 * The degree assortativity coefficient (NetworkX degree_assortativity_coefficient): the Pearson correlation of the
 * degrees at the two ends of the edges (for a directed graph, the source's out-degree against the target's
 * in-degree by default).
 *
 * @param g - The graph.
 * @param options - For directed graphs, which degrees: `x` (default `'out'`) at the source and `y` (default `'in'`) at
 *   the target.
 * @returns The coefficient, in [-1, 1] (NaN when every edge joins equal degrees).
 * @concept assortativity
 * @reference M. E. J. Newman, Mixing patterns in networks, Phys. Rev. E 67 (2003).
 * @example
 * ```ts
 * import { degreeAssortativity, starGraph } from 'mdml-engine/graph'
 * degreeAssortativity(starGraph(4)) // => -1
 * ```
 */
export function degreeAssortativity(
  g: Graph,
  options: { readonly x?: 'in' | 'out'; readonly y?: 'in' | 'out' } = {},
): number {
  const pick = (side: 'in' | 'out'): number[] => (side === 'in' ? inDegree(g) : outDegree(g))
  const x = g.directed ? pick(options.x ?? 'out') : degree(g)
  const y = g.directed ? pick(options.y ?? 'in') : x
  const [xs, ys] = edgeEnds(g, x, y)
  return pearson(xs, ys)
}

/**
 * The numeric assortativity coefficient (NetworkX numeric_assortativity_coefficient): the Pearson correlation of a
 * node value at the two ends of the edges.
 *
 * @param g - The graph.
 * @param values - A number per node.
 * @returns The coefficient, in [-1, 1].
 * @concept assortativity
 * @example
 * ```ts
 * import { numericAssortativity, pathGraph } from 'mdml-engine/graph'
 * numericAssortativity(pathGraph(4), [1, 1, 2, 2]).toFixed(4) // => '0.3333'
 * ```
 */
export function numericAssortativity(g: Graph, values: ArrayLike<number>): number {
  const [xs, ys] = edgeEnds(g, values, values)
  return pearson(xs, ys)
}

/**
 * The attribute (categorical) assortativity coefficient (NetworkX attribute_assortativity_coefficient): Newman's
 * r = (sum_i e_ii - sum_i a_i b_i) / (1 - sum_i a_i b_i) over the mixing matrix e of the categories at edge ends.
 *
 * @param g - The graph.
 * @param categories - A category per node.
 * @returns The coefficient, in [-1, 1] (1 when every edge joins equal categories).
 * @concept assortativity
 * @example
 * ```ts
 * import { attributeAssortativity, barbellGraph } from 'mdml-engine/graph'
 * attributeAssortativity(barbellGraph(3, 0), [0, 0, 0, 1, 1, 1]).toFixed(4) // => '0.7143'
 * ```
 */
export function attributeAssortativity(g: Graph, categories: ArrayLike<number | string>): number {
  const index = new Map<number | string, number>()
  for (let v = 0; v < g.n; v++) {
    const c = categories[v] ?? 0
    if (!index.has(c)) index.set(c, index.size)
  }
  const k = index.size
  const e = new Float64Array(k * k)
  let total = 0
  const add = (u: number, v: number): void => {
    const i = index.get(categories[u] ?? 0) ?? 0
    const j = index.get(categories[v] ?? 0) ?? 0
    e[i * k + j] = (e[i * k + j] ?? 0) + 1
    total++
  }
  for (const edge of g.edges) {
    add(edge.source, edge.target)
    if (!g.directed && edge.source !== edge.target) add(edge.target, edge.source)
  }
  let trace = 0
  let ab = 0
  for (let i = 0; i < k; i++) {
    trace += (e[i * k + i] ?? 0) / total
    let a = 0
    let b = 0
    for (let j = 0; j < k; j++) {
      a += (e[i * k + j] ?? 0) / total
      b += (e[j * k + i] ?? 0) / total
    }
    ab += a * b
  }
  return (trace - ab) / (1 - ab)
}

/**
 * Homophily of node labels on a graph, as used to describe graph-learning benchmarks: the edge homophily ratio (Zhu et
 * al., 2020: the fraction of edges joining equal labels), node homophily (Pei et al., 2020: the mean over nodes of the
 * fraction of neighbours sharing the label), or class-insensitive homophily (Lim et al., 2021: the average excess of
 * each class's edge homophily over its share of nodes, clipped at 0).
 *
 * @param g - The graph (edge directions are ignored).
 * @param labels - A class label per node (integers 0 .. C - 1).
 * @param options - `kind`: `'edge'` (default), `'node'` or `'class'`.
 * @returns The homophily, in [0, 1].
 * @concept homophily
 * @reference J. Zhu et al., Beyond homophily in graph neural networks (NeurIPS 2020); H. Pei et al., Geom-GCN (ICLR
 *   2020); D. Lim et al., New benchmarks for learning on non-homophilous graphs (2021).
 * @example
 * ```ts
 * import { barbellGraph, homophily } from 'mdml-engine/graph'
 * homophily(barbellGraph(3, 0), [0, 0, 0, 1, 1, 1]) // => 0.8571428571428571
 * ```
 */
export function homophily(
  g: Graph,
  labels: ArrayLike<number>,
  options: { readonly kind?: 'edge' | 'node' | 'class' } = {},
): number {
  const kind = options.kind ?? 'edge'
  const nb = neighbourSets(g)
  if (kind === 'edge') {
    let same = 0
    let total = 0
    for (const e of g.edges) {
      if (e.source === e.target) continue
      total++
      if (labels[e.source] === labels[e.target]) same++
    }
    return total === 0 ? 0 : same / total
  }
  if (kind === 'node') {
    let s = 0
    let counted = 0
    for (let u = 0; u < g.n; u++) {
      const set = nb[u] ?? new Set<number>()
      if (set.size === 0) continue
      let same = 0
      for (const v of set) if (labels[v] === labels[u]) same++
      s += same / set.size
      counted++
    }
    return counted === 0 ? 0 : s / counted
  }
  let classes = 0
  for (let u = 0; u < g.n; u++) classes = Math.max(classes, (labels[u] ?? 0) + 1)
  const same = new Float64Array(classes)
  const all = new Float64Array(classes)
  const sizes = new Float64Array(classes)
  for (let u = 0; u < g.n; u++) {
    const c = labels[u] ?? 0
    sizes[c] = (sizes[c] ?? 0) + 1
    for (const v of nb[u] ?? []) {
      all[c] = (all[c] ?? 0) + 1
      if (labels[v] === c) same[c] = (same[c] ?? 0) + 1
    }
  }
  let h = 0
  for (let c = 0; c < classes; c++) {
    const hc = (all[c] ?? 0) === 0 ? 0 : (same[c] ?? 0) / (all[c] ?? 1)
    h += Math.max(0, hc - (sizes[c] ?? 0) / g.n)
  }
  return classes <= 1 ? 1 : h / (classes - 1)
}

function cutAndVolumes(g: Graph, inside: Uint8Array, weighted: boolean): { cut: number; volS: number; volT: number } {
  let cut = 0
  let volS = 0
  let volT = 0
  for (const e of g.edges) {
    const w = weighted ? e.weight : 1
    const a = inside[e.source] === 1
    const b = inside[e.target] === 1
    if (a !== b) cut += w
    for (const end of g.directed ? [a] : [a, b]) {
      if (end) volS += w
      else volT += w
    }
  }
  return { cut, volS, volT }
}

/**
 * The conductance of a node set (NetworkX conductance): the weight of the edges leaving it over the smaller of its
 * volume and its complement's.
 *
 * @param g - The graph.
 * @param nodes - The set S.
 * @param options - `weighted` (default false).
 * @returns cut(S) / min(vol S, vol S^c).
 * @concept expansion
 * @example
 * ```ts
 * import { barbellGraph, conductance } from 'mdml-engine/graph'
 * conductance(barbellGraph(3, 0), [0, 1, 2]) // => 0.14285714285714285
 * ```
 */
export function conductance(g: Graph, nodes: readonly NodeId[], options: { readonly weighted?: boolean } = {}): number {
  const inside = new Uint8Array(g.n)
  for (const v of nodes) inside[nodeIndex(g, v)] = 1
  const { cut, volS, volT } = cutAndVolumes(g, inside, options.weighted ?? false)
  return cut / Math.min(volS, volT)
}

/**
 * The edge expansion of a node set (NetworkX edge_expansion): the weight of the edges leaving it over the smaller of
 * its size and its complement's.
 *
 * @param g - The graph.
 * @param nodes - The set S.
 * @param options - `weighted` (default false).
 * @returns cut(S) / min(|S|, |S^c|).
 * @concept expansion
 * @example
 * ```ts
 * import { cycleGraph, edgeExpansion } from 'mdml-engine/graph'
 * edgeExpansion(cycleGraph(6), [0, 1, 2]) // => 0.6666666666666666
 * ```
 */
export function edgeExpansion(
  g: Graph,
  nodes: readonly NodeId[],
  options: { readonly weighted?: boolean } = {},
): number {
  const inside = new Uint8Array(g.n)
  for (const v of nodes) inside[nodeIndex(g, v)] = 1
  let size = 0
  for (const x of inside) size += x
  const { cut } = cutAndVolumes(g, inside, options.weighted ?? false)
  return cut / Math.min(size, g.n - size)
}

/** The result of {@link cheegerConstant}. */
export interface Cheeger {
  readonly value: number
  /** A set attaining it. */
  readonly set: readonly number[]
}

/**
 * The Cheeger constant of an undirected graph by exhaustive search over node sets (2^(n - 1) of them, so at most 24
 * nodes): the conductance form h = min_S cut(S) / min(vol S, vol S^c), or with `kind: 'expansion'` the edge-expansion
 * (isoperimetric) number min_S cut(S) / min(|S|, |S^c|). Gray-code order updates the cut in O(degree) per set. The
 * spectral gap bounds the conductance form: lambda_2 / 2 <= h <= sqrt(2 lambda_2) for the normalised Laplacian.
 *
 * @param g - An undirected graph with 2 to 24 nodes.
 * @param options - `kind` (default `'conductance'`) and `weighted` (default false).
 * @returns The constant and a minimising set.
 * @throws MdmlError with code `value` for a directed graph or a size outside 2 .. 24.
 * @concept expansion
 * @example
 * ```ts
 * import { barbellGraph, cheegerConstant } from 'mdml-engine/graph'
 * cheegerConstant(barbellGraph(3, 0)).set // => [0, 1, 2]
 * ```
 */
export function cheegerConstant(
  g: Graph,
  options: { readonly kind?: 'conductance' | 'expansion'; readonly weighted?: boolean } = {},
): Cheeger {
  if (g.directed) throw new MdmlError('value', 'cheegerConstant: the graph must be undirected')
  if (g.n < 2 || g.n > 24) throw new MdmlError('value', 'cheegerConstant: exhaustive search needs 2 to 24 nodes')
  const weighted = options.weighted ?? false
  const expansion = options.kind === 'expansion'
  const a = g.out
  const strength = Float64Array.from(degree(g, { weighted }))
  let total = 0
  for (const s of strength) total += s
  const inside = new Uint8Array(g.n)
  let cut = 0
  let vol = 0
  let size = 0
  let best = Number.POSITIVE_INFINITY
  let bestCode = 0
  let code = 0
  // Node n - 1 always stays outside; the Gray code walks the 2^(n - 1) - 1 non-empty subsets of the others.
  for (let i = 1; i < 2 ** (g.n - 1); i++) {
    let bit = 0
    while (((i >> bit) & 1) === 0) bit++
    const v = bit
    const entering = inside[v] === 0
    for (let k = a.indptr[v] ?? 0; k < (a.indptr[v + 1] ?? 0); k++) {
      const u = a.indices[k] ?? 0
      if (u === v) continue
      const w = weighted ? (a.weights[k] ?? 0) : 1
      cut += (inside[u] === 1) === entering ? -w : w
    }
    inside[v] = entering ? 1 : 0
    vol += (entering ? 1 : -1) * (strength[v] ?? 0)
    size += entering ? 1 : -1
    code ^= 1 << v
    const denominator = expansion ? Math.min(size, g.n - size) : Math.min(vol, total - vol)
    const value = denominator > 0 ? cut / denominator : Number.POSITIVE_INFINITY
    if (value < best - 1e-15) {
      best = value
      bestCode = code
    }
  }
  const set = Array.from({ length: g.n }, (_, v) => v).filter((v) => ((bestCode >> v) & 1) === 1)
  // Report the side with the smaller denominator (the side the minimum is attained on).
  const other = Array.from({ length: g.n }, (_, v) => v).filter((v) => ((bestCode >> v) & 1) === 0)
  return { value: best, set: set.length <= other.length ? set : other }
}

function sortedEigenvalues(g: Graph, normalised: boolean, weighted: boolean): Float64Array {
  const l = normalised ? normalisedLaplacian(g, { weighted }) : laplacian(g, { weighted })
  return Float64Array.from(toFlat(concreteOrThrow(eigvalsh(l) as Tensor, 'spectrum')))
}

/**
 * The algebraic connectivity (NetworkX algebraic_connectivity): the second-smallest Laplacian eigenvalue, positive
 * exactly when the graph is connected (Fiedler, 1973).
 *
 * @param g - The graph (at least two nodes).
 * @param options - `normalised` (default false) and `weighted` (default true).
 * @returns lambda_2.
 * @concept spectral-graph-theory
 * @example
 * ```ts
 * import { algebraicConnectivity, completeGraph } from 'mdml-engine/graph'
 * Math.round(algebraicConnectivity(completeGraph(5)) * 1e9) / 1e9 // => 5
 * ```
 */
export function algebraicConnectivity(
  g: Graph,
  options: { readonly normalised?: boolean; readonly weighted?: boolean } = {},
): number {
  if (g.n < 2) throw new MdmlError('value', 'algebraicConnectivity: the graph needs at least two nodes')
  return sortedEigenvalues(g, options.normalised ?? false, options.weighted ?? true)[1] ?? 0
}

/**
 * The spectral gap of the random walk: 1 - lambda_2(P) = mu_2, the second-smallest eigenvalue of the normalised
 * Laplacian, which sets the mixing rate and bounds the Cheeger constant. With `absolute`, the absolute gap
 * 1 - max(|lambda_2(P)|, |lambda_n(P)|), which also accounts for near-bipartite oscillation.
 *
 * @param g - An undirected graph (at least two nodes).
 * @param options - `absolute` (default false) and `weighted` (default true).
 * @returns The gap.
 * @concept spectral-graph-theory
 * @example
 * ```ts
 * import { completeGraph, cycleGraph, spectralGap } from 'mdml-engine/graph'
 * Math.round(spectralGap(completeGraph(4)) * 1e9) / 1e9 // => 1.333333333
 * Math.round(spectralGap(cycleGraph(4), { absolute: true }) * 1e9) / 1e9 // => 0
 * ```
 */
export function spectralGap(
  g: Graph,
  options: { readonly absolute?: boolean; readonly weighted?: boolean } = {},
): number {
  if (g.n < 2) throw new MdmlError('value', 'spectralGap: the graph needs at least two nodes')
  const mu = sortedEigenvalues(g, true, options.weighted ?? true)
  const second = mu[1] ?? 0
  if (options.absolute !== true) return second
  const last = mu[g.n - 1] ?? 0
  return 1 - Math.max(Math.abs(1 - second), Math.abs(1 - last))
}

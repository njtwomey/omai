/**
 * Graph generators, NetworkX style. Deterministic families (complete, path, cycle, star, wheel, grid, hypercube,
 * complete bipartite, balanced tree, barbell), named graphs (Petersen, Zachary's karate club) and random models
 * (Erdos-Renyi, G(n, m), Watts-Strogatz small world, Barabasi-Albert preferential attachment, stochastic block model,
 * random DAGs, random geometric and k-nearest-neighbour graphs) driven by explicit keys.
 */

import { MdmlError } from '../base/index.ts'
import type { Metric } from '../numerics/geometry/index.ts'
import { kNeighboursGraph } from '../numerics/spatial/index.ts'
import { foldIn, type Key, permutation, subsample, uniform } from '../random/index.ts'
import { concreteOrThrow, promoteFloat, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type EdgeInput, type Graph, graph } from './graph.ts'

function count(n: number, name: string): number {
  if (!Number.isInteger(n) || n < 0) throw new MdmlError('value', `${name}: ${n} is not a valid size`)
  return n
}

function probability(p: number, name: string): number {
  if (!(p >= 0 && p <= 1)) throw new MdmlError('value', `${name}: probability ${p} is not in [0, 1]`)
  return p
}

/**
 * The complete graph K_n (NetworkX complete_graph); directed, every ordered pair.
 *
 * @param n - The number of nodes.
 * @param options - `directed` (default false).
 * @returns The graph.
 * @example
 * ```ts
 * import { completeGraph } from 'mdml-engine/graph'
 * completeGraph(4).edges.length // => 6
 * ```
 */
export function completeGraph(n: number, options: { readonly directed?: boolean } = {}): Graph {
  const directed = options.directed ?? false
  const edges: [number, number][] = []
  for (let i = 0; i < count(n, 'completeGraph'); i++) {
    for (let j = directed ? 0 : i + 1; j < n; j++) if (i !== j) edges.push([i, j])
  }
  return graph(edges, { nodes: n, directed })
}

/**
 * The path P_n: 0 - 1 - ... - (n - 1) (NetworkX path_graph).
 *
 * @param n - The number of nodes.
 * @param options - `directed` (default false).
 * @returns The graph.
 * @example
 * ```ts
 * import { pathGraph } from 'mdml-engine/graph'
 * pathGraph(4).edges.length // => 3
 * ```
 */
export function pathGraph(n: number, options: { readonly directed?: boolean } = {}): Graph {
  const edges = Array.from({ length: Math.max(0, count(n, 'pathGraph') - 1) }, (_, i): [number, number] => [i, i + 1])
  return graph(edges, { nodes: n, directed: options.directed ?? false })
}

/**
 * The cycle C_n (NetworkX cycle_graph).
 *
 * @param n - The number of nodes.
 * @param options - `directed` (default false).
 * @returns The graph.
 * @example
 * ```ts
 * import { cycleGraph } from 'mdml-engine/graph'
 * cycleGraph(5).edges.length // => 5
 * ```
 */
export function cycleGraph(n: number, options: { readonly directed?: boolean } = {}): Graph {
  const edges = Array.from({ length: count(n, 'cycleGraph') }, (_, i): [number, number] => [i, (i + 1) % n])
  return graph(n === 1 ? [] : n === 2 ? [[0, 1]] : edges, { nodes: n, directed: options.directed ?? false })
}

/**
 * The star with a centre 0 and `n` leaves, `n + 1` nodes in all (NetworkX star_graph).
 *
 * @param n - The number of leaves.
 * @returns The graph.
 * @example
 * ```ts
 * import { starGraph } from 'mdml-engine/graph'
 * starGraph(3).n // => 4
 * ```
 */
export function starGraph(n: number): Graph {
  return graph(
    Array.from({ length: count(n, 'starGraph') }, (_, i): [number, number] => [0, i + 1]),
    { nodes: n + 1 },
  )
}

/**
 * The wheel on `n` nodes: a hub 0 joined to a cycle 1 .. n - 1 (NetworkX wheel_graph).
 *
 * @param n - The number of nodes.
 * @returns The graph.
 * @example
 * ```ts
 * import { wheelGraph } from 'mdml-engine/graph'
 * wheelGraph(5).edges.length // => 8
 * ```
 */
export function wheelGraph(n: number): Graph {
  const edges: [number, number][] = []
  for (let i = 1; i < count(n, 'wheelGraph'); i++) edges.push([0, i])
  for (let i = 1; i < n && n > 3; i++) edges.push([i, i === n - 1 ? 1 : i + 1])
  if (n === 3) edges.push([1, 2])
  return graph(edges, { nodes: n })
}

/**
 * The m-by-n grid (NetworkX grid_2d_graph): node `i * n + j`, labelled `'i,j'`, joined to its horizontal and vertical
 * neighbours; `periodic` wraps both directions into a torus.
 *
 * @param m - Rows.
 * @param n - Columns.
 * @param options - `periodic` (default false).
 * @returns The graph.
 * @example
 * ```ts
 * import { gridGraph } from 'mdml-engine/graph'
 * gridGraph(2, 3).edges.length // => 7
 * gridGraph(3, 3).labels[4] // => '1,1'
 * ```
 */
export function gridGraph(m: number, n: number, options: { readonly periodic?: boolean } = {}): Graph {
  const periodic = options.periodic ?? false
  const edges: [number, number][] = []
  count(m, 'gridGraph')
  count(n, 'gridGraph')
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const v = i * n + j
      if (j + 1 < n) edges.push([v, v + 1])
      else if (periodic && n > 2) edges.push([v, i * n])
      if (i + 1 < m) edges.push([v, v + n])
      else if (periodic && m > 2) edges.push([v, j])
    }
  }
  const labels = Array.from({ length: m * n }, (_, v) => `${Math.floor(v / n)},${v % n}`)
  return graph(edges, { nodes: labels })
}

/**
 * The d-dimensional hypercube Q_d: nodes `0 .. 2^d - 1` joined when their binary codes differ in one bit (NetworkX
 * hypercube_graph).
 *
 * @param d - The dimension.
 * @returns The graph.
 * @example
 * ```ts
 * import { hypercubeGraph } from 'mdml-engine/graph'
 * hypercubeGraph(3).edges.length // => 12
 * ```
 */
export function hypercubeGraph(d: number): Graph {
  const n = 2 ** count(d, 'hypercubeGraph')
  const edges: [number, number][] = []
  for (let v = 0; v < n; v++) for (let b = 0; b < d; b++) if (((v >> b) & 1) === 0) edges.push([v, v | (1 << b)])
  return graph(edges, { nodes: n })
}

/**
 * The complete bipartite graph K_{m,n}: nodes `0 .. m - 1` on one side, `m .. m + n - 1` on the other (NetworkX
 * complete_bipartite_graph); node data records the side as `bipartite: 0 | 1`.
 *
 * @param m - Size of the first side.
 * @param n - Size of the second side.
 * @returns The graph.
 * @example
 * ```ts
 * import { completeBipartiteGraph } from 'mdml-engine/graph'
 * completeBipartiteGraph(2, 3).edges.length // => 6
 * ```
 */
export function completeBipartiteGraph(m: number, n: number): Graph {
  const edges: [number, number][] = []
  for (let i = 0; i < count(m, 'completeBipartiteGraph'); i++) {
    for (let j = 0; j < count(n, 'completeBipartiteGraph'); j++) edges.push([i, m + j])
  }
  const nodeData = Array.from({ length: m + n }, (_, v) => ({ bipartite: v < m ? 0 : 1 }))
  return graph(edges, { nodes: m + n, nodeData })
}

/**
 * The perfectly balanced r-ary tree of height h (NetworkX balanced_tree): node 0 is the root and node `v` has children
 * `r v + 1 .. r v + r`.
 *
 * @param r - The branching factor.
 * @param h - The height.
 * @returns The tree.
 * @example
 * ```ts
 * import { balancedTree } from 'mdml-engine/graph'
 * balancedTree(2, 2).n // => 7
 * ```
 */
export function balancedTree(r: number, h: number): Graph {
  count(r, 'balancedTree')
  const n = r === 1 ? h + 1 : (r ** (count(h, 'balancedTree') + 1) - 1) / (r - 1)
  const edges = Array.from({ length: Math.max(0, n - 1) }, (_, i): [number, number] => [
    Math.floor(i / Math.max(r, 1)),
    i + 1,
  ])
  return graph(edges, { nodes: n })
}

/**
 * The barbell: two complete graphs K_m joined by a path through `bridge` extra nodes (NetworkX barbell_graph). Nodes
 * `0 .. m - 1` form the left bell, then the path, then the right bell.
 *
 * @param m - The size of each bell (at least 2).
 * @param bridge - The number of nodes on the joining path.
 * @returns The graph.
 * @example
 * ```ts
 * import { barbellGraph } from 'mdml-engine/graph'
 * barbellGraph(3, 2).edges.length // => 9
 * ```
 */
export function barbellGraph(m: number, bridge: number): Graph {
  if (count(m, 'barbellGraph') < 2) throw new MdmlError('value', 'barbellGraph: each bell needs at least 2 nodes')
  count(bridge, 'barbellGraph')
  const edges: [number, number][] = []
  const right = m + bridge
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      edges.push([i, j])
      edges.push([right + i, right + j])
    }
  }
  for (let v = m - 1; v < right; v++) edges.push([v, v + 1])
  return graph(edges, { nodes: 2 * m + bridge })
}

/**
 * The Petersen graph: 10 nodes, 15 edges, 3-regular, girth 5 (NetworkX petersen_graph).
 *
 * @returns The graph.
 * @example
 * ```ts
 * import { petersenGraph } from 'mdml-engine/graph'
 * petersenGraph().edges.length // => 15
 * ```
 */
export function petersenGraph(): Graph {
  const edges: [number, number][] = []
  for (let i = 0; i < 5; i++) {
    edges.push([i, (i + 1) % 5])
    edges.push([i, i + 5])
    edges.push([5 + i, 5 + ((i + 2) % 5)])
  }
  return graph(edges, { nodes: 10 })
}

const KARATE: readonly (readonly [number, number])[] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [0, 6],
  [0, 7],
  [0, 8],
  [0, 10],
  [0, 11],
  [0, 12],
  [0, 13],
  [0, 17],
  [0, 19],
  [0, 21],
  [0, 31],
  [1, 2],
  [1, 3],
  [1, 7],
  [1, 13],
  [1, 17],
  [1, 19],
  [1, 21],
  [1, 30],
  [2, 3],
  [2, 7],
  [2, 8],
  [2, 9],
  [2, 13],
  [2, 27],
  [2, 28],
  [2, 32],
  [3, 7],
  [3, 12],
  [3, 13],
  [4, 6],
  [4, 10],
  [5, 6],
  [5, 10],
  [5, 16],
  [6, 16],
  [8, 30],
  [8, 32],
  [8, 33],
  [9, 33],
  [13, 33],
  [14, 32],
  [14, 33],
  [15, 32],
  [15, 33],
  [18, 32],
  [18, 33],
  [19, 33],
  [20, 32],
  [20, 33],
  [22, 32],
  [22, 33],
  [23, 25],
  [23, 27],
  [23, 29],
  [23, 32],
  [23, 33],
  [24, 25],
  [24, 27],
  [24, 31],
  [25, 31],
  [26, 29],
  [26, 33],
  [27, 33],
  [28, 31],
  [28, 33],
  [29, 32],
  [29, 33],
  [30, 32],
  [30, 33],
  [31, 32],
  [31, 33],
  [32, 33],
]
const OFFICER = new Set([9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

/**
 * Zachary's karate club (1977): 34 members, 78 friendships, and the club each member joined after the split (node data
 * `club`: `'Mr. Hi'` or `'Officer'`), the standard community-detection benchmark (NetworkX karate_club_graph,
 * unweighted).
 *
 * @returns The graph.
 * @reference W. W. Zachary, An information flow model for conflict and fission in small groups, J. Anthropological
 *   Research 33 (1977).
 * @example
 * ```ts
 * import { karateClubGraph } from 'mdml-engine/graph'
 * const g = karateClubGraph()
 * [g.n, g.edges.length, g.nodeData[33]?.['club']] // => [34, 78, 'Officer']
 * ```
 */
export function karateClubGraph(): Graph {
  const nodeData = Array.from({ length: 34 }, (_, v) => ({ club: OFFICER.has(v) ? 'Officer' : 'Mr. Hi' }))
  return graph(KARATE, { nodes: 34, nodeData, attributes: { name: "Zachary's Karate Club" } })
}

/**
 * The Erdos-Renyi random graph G(n, p) (NetworkX gnp_random_graph): every pair (every ordered pair when directed) is
 * an edge independently with probability p.
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param p - The edge probability.
 * @param options - `directed` (default false).
 * @returns The graph.
 * @concept random-graphs
 * @example
 * ```ts
 * import { erdosRenyiGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * erdosRenyiGraph(key(0), 10, 1).edges.length // => 45
 * ```
 */
export function erdosRenyiGraph(key: Key, n: number, p: number, options: { readonly directed?: boolean } = {}): Graph {
  const directed = options.directed ?? false
  probability(p, 'erdosRenyiGraph')
  const pairs: [number, number][] = []
  for (let i = 0; i < count(n, 'erdosRenyiGraph'); i++) {
    for (let j = directed ? 0 : i + 1; j < n; j++) if (i !== j) pairs.push([i, j])
  }
  const u = uniform(key, [pairs.length]).data
  return graph(
    pairs.filter((_, k) => (u[k] ?? 1) < p),
    { nodes: n, directed },
  )
}

/**
 * The uniform random graph G(n, m) with exactly m edges (NetworkX gnm_random_graph).
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param m - The number of edges (at most the number of pairs).
 * @param options - `directed` (default false).
 * @returns The graph.
 * @throws MdmlError with code `value` when m exceeds the number of possible edges.
 * @concept random-graphs
 * @example
 * ```ts
 * import { gnmRandomGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * gnmRandomGraph(key(1), 8, 5).edges.length // => 5
 * ```
 */
export function gnmRandomGraph(key: Key, n: number, m: number, options: { readonly directed?: boolean } = {}): Graph {
  const directed = options.directed ?? false
  const total = directed ? n * (n - 1) : (n * (n - 1)) / 2
  if (!Number.isInteger(m) || m < 0 || m > total) {
    throw new MdmlError('value', `gnmRandomGraph: ${m} edges do not fit in ${total} pairs`)
  }
  const pairs: [number, number][] = []
  for (let i = 0; i < count(n, 'gnmRandomGraph'); i++) {
    for (let j = directed ? 0 : i + 1; j < n; j++) if (i !== j) pairs.push([i, j])
  }
  const chosen = Array.from(subsample(key, total, m).data).sort((a, b) => a - b)
  return graph(
    chosen.map((k) => pairs[k] ?? [0, 0]),
    { nodes: n, directed },
  )
}

/**
 * The Watts-Strogatz small-world graph (NetworkX watts_strogatz_graph): a ring where each node joins its k nearest
 * neighbours (k / 2 each side), then each lattice edge (u, u + j) is rewired with probability p to (u, w) for a
 * uniform w that creates neither a self-loop nor a repeated edge.
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param k - Each node's lattice degree (even; odd values round down, as NetworkX).
 * @param p - The rewiring probability.
 * @returns The graph.
 * @concept random-graphs
 * @example
 * ```ts
 * import { wattsStrogatzGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * wattsStrogatzGraph(key(0), 10, 4, 0.2).edges.length // => 20
 * ```
 */
export function wattsStrogatzGraph(key: Key, n: number, k: number, p: number): Graph {
  probability(p, 'wattsStrogatzGraph')
  if (count(k, 'wattsStrogatzGraph') >= count(n, 'wattsStrogatzGraph') && n > 0) {
    throw new MdmlError('value', 'wattsStrogatzGraph: k must be smaller than n')
  }
  const half = Math.floor(k / 2)
  const adj = Array.from({ length: n }, () => new Set<number>())
  const link = (a: number, b: number): void => {
    adj[a]?.add(b)
    adj[b]?.add(a)
  }
  for (let u = 0; u < n; u++) for (let j = 1; j <= half; j++) link(u, (u + j) % n)
  let draw = 0
  const next = (): number => uniform(foldIn(key, draw++), [1]).data[0] ?? 0
  for (let j = 1; j <= half; j++) {
    for (let u = 0; u < n; u++) {
      const v = (u + j) % n
      if (next() >= p) continue
      const nu = adj[u] as Set<number>
      if (nu.size >= n - 1) continue
      let w = Math.floor(next() * n)
      while (w === u || nu.has(w)) w = Math.floor(next() * n)
      adj[u]?.delete(v)
      adj[v]?.delete(u)
      link(u, w)
    }
  }
  const edges: [number, number][] = []
  for (let u = 0; u < n; u++) for (const v of adj[u] ?? []) if (u < v) edges.push([u, v])
  return graph(
    edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]),
    { nodes: n },
  )
}

/**
 * The Barabasi-Albert preferential-attachment graph (NetworkX barabasi_albert_graph): start from a star on m + 1
 * nodes; each new node joins m distinct existing nodes chosen with probability proportional to their degree.
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param m - Edges added per new node (1 <= m < n).
 * @returns The graph, with `(n - m - 1) m + m` edges.
 * @concept random-graphs
 * @example
 * ```ts
 * import { barabasiAlbertGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * barabasiAlbertGraph(key(0), 20, 2).edges.length // => 36
 * ```
 */
export function barabasiAlbertGraph(key: Key, n: number, m: number): Graph {
  if (!(Number.isInteger(m) && m >= 1 && m < count(n, 'barabasiAlbertGraph'))) {
    throw new MdmlError('value', 'barabasiAlbertGraph: need 1 <= m < n')
  }
  const edges: [number, number][] = []
  const repeated: number[] = []
  for (let i = 1; i <= m; i++) {
    edges.push([0, i])
    repeated.push(0, i)
  }
  let draw = 0
  for (let source = m + 1; source < n; source++) {
    const targets = new Set<number>()
    while (targets.size < m) {
      const u = uniform(foldIn(key, draw++), [1]).data[0] ?? 0
      targets.add(repeated[Math.floor(u * repeated.length)] ?? 0)
    }
    for (const t of [...targets].sort((a, b) => a - b)) {
      edges.push([t, source])
      repeated.push(t, source)
    }
  }
  return graph(edges, { nodes: n })
}

/**
 * The stochastic block model (NetworkX stochastic_block_model): blocks of the given sizes, nodes numbered block by
 * block, and an edge between nodes of blocks r and s with probability `probs[r][s]`. Node data records each node's
 * `block`.
 *
 * @param key - The random key.
 * @param sizes - Block sizes.
 * @param probs - The symmetric (for undirected graphs) block probability matrix.
 * @param options - `directed` (default false) and `selfloops` (default false).
 * @returns The graph.
 * @throws MdmlError with code `value` for a matrix of the wrong shape, an asymmetric one when undirected, or a
 *   probability outside [0, 1].
 * @concept random-graphs
 * @example
 * ```ts
 * import { stochasticBlockModel } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * stochasticBlockModel(key(0), [3, 3], [[1, 0], [0, 1]]).edges.length // => 6
 * ```
 */
export function stochasticBlockModel(
  key: Key,
  sizes: readonly number[],
  probs: readonly (readonly number[])[],
  options: { readonly directed?: boolean; readonly selfloops?: boolean } = {},
): Graph {
  const directed = options.directed ?? false
  const selfloops = options.selfloops ?? false
  const k = sizes.length
  if (probs.length !== k || probs.some((r) => r.length !== k)) {
    throw new MdmlError('value', 'stochasticBlockModel: probs must be a square matrix with one row per block')
  }
  for (let r = 0; r < k; r++) {
    for (let s = 0; s < k; s++) {
      probability(probs[r]?.[s] ?? 0, 'stochasticBlockModel')
      if (!directed && probs[r]?.[s] !== probs[s]?.[r]) {
        throw new MdmlError('value', 'stochasticBlockModel: probs must be symmetric for an undirected graph')
      }
    }
  }
  const block = sizes.flatMap((size, b) => Array.from({ length: count(size, 'stochasticBlockModel') }, () => b))
  const n = block.length
  const pairs: [number, number][] = []
  for (let i = 0; i < n; i++) {
    for (let j = directed ? 0 : i; j < n; j++) if (i !== j || selfloops) pairs.push([i, j])
  }
  const u = uniform(key, [pairs.length]).data
  const edges = pairs.filter(([i, j], t) => (u[t] ?? 1) < (probs[block[i] ?? 0]?.[block[j] ?? 0] ?? 0))
  return graph(edges, { nodes: n, directed, nodeData: block.map((b) => ({ block: b })) })
}

/**
 * A random directed acyclic graph: a uniformly random ordering of the nodes, then each pair joined from the earlier to
 * the later node with probability p (G(n, p) restricted to one orientation).
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param p - The edge probability.
 * @returns A directed acyclic graph.
 * @concept random-graphs
 * @example
 * ```ts
 * import { isDag, randomDag } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * isDag(randomDag(key(0), 8, 0.5)) // => true
 * ```
 */
export function randomDag(key: Key, n: number, p: number): Graph {
  probability(p, 'randomDag')
  const order = Array.from(permutation(foldIn(key, 0), count(n, 'randomDag')).data)
  const u = uniform(foldIn(key, 1), [(n * (n - 1)) / 2]).data
  const edges: [number, number][] = []
  let t = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) if ((u[t++] ?? 1) < p) edges.push([order[i] ?? 0, order[j] ?? 0])
  }
  return graph(edges, { nodes: n, directed: true })
}

/**
 * The random geometric graph (NetworkX random_geometric_graph): n uniform points in the unit cube, joined when their
 * Euclidean distance is at most `radius`. Node data records each point as `pos`.
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param radius - The connection radius.
 * @param options - The dimension (default 2).
 * @returns The graph.
 * @concept random-graphs
 * @example
 * ```ts
 * import { randomGeometricGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * randomGeometricGraph(key(0), 6, 2).edges.length // => 15
 * ```
 */
export function randomGeometricGraph(
  key: Key,
  n: number,
  radius: number,
  options: { readonly dim?: number } = {},
): Graph {
  const dim = options.dim ?? 2
  const x = uniform(key, [count(n, 'randomGeometricGraph'), dim]).data
  const edges: [number, number][] = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d2 = 0
      for (let c = 0; c < dim; c++) d2 += ((x[i * dim + c] ?? 0) - (x[j * dim + c] ?? 0)) ** 2
      if (d2 <= radius * radius) edges.push([i, j])
    }
  }
  const nodeData = Array.from({ length: n }, (_, i) => ({ pos: Array.from(x.subarray(i * dim, i * dim + dim)) }))
  return graph(edges, { nodes: n, nodeData })
}

/**
 * The k-nearest-neighbour graph of points (sklearn kneighbors_graph, made undirected): an edge joins each point to
 * each of its k nearest neighbours, kept when either end chose it (or, with `mutual`, only when both did). With
 * `weighted`, edge weights are the distances.
 *
 * @param points - Points `[n, d]`.
 * @param k - Neighbours per point.
 * @param options - `mutual` (default false), `weighted` (default false) and the metric (default Euclidean).
 * @returns The undirected graph; node data records each point as `pos`.
 * @concept nearest-neighbours
 * @example
 * ```ts
 * import { kNearestNeighbourGraph } from 'mdml-engine/graph'
 * kNearestNeighbourGraph([[0], [1], [3], [7]], 1).edges.length // => 3
 * ```
 */
export function kNearestNeighbourGraph(
  points: TensorLike,
  k: number,
  options: { readonly mutual?: boolean; readonly weighted?: boolean; readonly metric?: Metric } = {},
): Graph {
  const x = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'kNearestNeighbourGraph')
  const n = x.shape[0] ?? 0
  const d = x.shape[1] ?? 1
  const a = kNeighboursGraph(x, k, {
    mode: 'distance',
    ...(options.metric === undefined ? {} : { metric: options.metric }),
  })
  const w = toFlat(a.values)
  const chosen = new Map<string, { count: number; weight: number }>()
  for (let t = 0; t < a.indices.length; t++) {
    const i = a.rows[t] ?? 0
    const j = a.indices[t] ?? 0
    const key = `${Math.min(i, j)},${Math.max(i, j)}`
    const old = chosen.get(key)
    chosen.set(key, { count: (old?.count ?? 0) + 1, weight: w[t] ?? 0 })
  }
  const edges: EdgeInput[] = []
  for (const [key, { count: c, weight }] of chosen) {
    if (options.mutual === true && c < 2) continue
    const [i, j] = key.split(',').map(Number) as [number, number]
    edges.push(options.weighted === true ? [i, j, weight] : [i, j])
  }
  const nodeData = Array.from({ length: n }, (_, i) => ({ pos: Array.from(x.data.subarray(i * d, i * d + d)) }))
  return graph(edges, { nodes: n, nodeData })
}

/**
 * The k-nearest-neighbour graph of n uniform random points in the unit cube (a random geometric graph with a degree
 * floor instead of a radius).
 *
 * @param key - The random key.
 * @param n - The number of nodes.
 * @param k - Neighbours per point.
 * @param options - Dimension (default 2), `mutual` and `weighted` as for {@link kNearestNeighbourGraph}.
 * @returns The graph.
 * @concept random-graphs
 * @example
 * ```ts
 * import { degree, randomKnnGraph } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * degree(randomKnnGraph(key(0), 12, 3)).every((d) => d >= 3) // => true
 * ```
 */
export function randomKnnGraph(
  key: Key,
  n: number,
  k: number,
  options: { readonly dim?: number; readonly mutual?: boolean; readonly weighted?: boolean } = {},
): Graph {
  const { dim = 2, ...rest } = options
  return kNearestNeighbourGraph(uniform(key, [count(n, 'randomKnnGraph'), dim]), k, rest)
}

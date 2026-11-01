/**
 * Spanning trees and rooted trees: minimum (or maximum) spanning forests by Kruskal's, Prim's or Boruvka's algorithm,
 * tree and forest tests, rooting a tree with parents, depths and distances, subtree sizes, traversal orders and Euler
 * tours, and lowest common ancestors by binary lifting.
 */

import { kahanSum, MdmlError } from '../base/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { connectedComponents } from './components.ts'
import { type Edge, type Graph, graph, type NodeId, nodeIndex } from './graph.ts'
import { type SpanningTreeAlgorithm, spanningTreeMethods } from './spanning-trees.ts'

/** A minimum (or maximum) spanning forest: its edges, their total weight and the forest as a graph. */
export interface SpanningTree {
  /** The edges, sorted by `(source, target)`. */
  readonly edges: readonly Edge[]
  readonly weight: number
  readonly graph: Graph
}

/**
 * A minimum spanning tree, or forest when the graph is disconnected (NetworkX minimum_spanning_tree,
 * scipy.sparse.csgraph.minimum_spanning_tree), by Kruskal's (default), Prim's or Boruvka's algorithm. Ties are broken by
 * edge index, so all three give the same tree. With `maximum`, a maximum spanning tree.
 *
 * @param g - An undirected graph.
 * @param options - `algorithm` and `maximum` (default false).
 * @returns The tree's edges, total weight and graph.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept spanning-trees
 * @example
 * ```ts
 * import { graph, minimumSpanningTree } from 'mdml-engine/graph'
 * const g = graph([[0, 1, 4], [0, 2, 1], [1, 2, 2], [1, 3, 5], [2, 3, 8]])
 * minimumSpanningTree(g).weight // => 8
 * minimumSpanningTree(g, { algorithm: 'prim' }).edges.map((e) => [e.source, e.target]) // => [[0, 2], [1, 2], [1, 3]]
 * minimumSpanningTree(g, { maximum: true }).weight // => 17
 * ```
 */
export function minimumSpanningTree(
  g: Graph,
  options: { readonly algorithm?: SpanningTreeAlgorithm; readonly maximum?: boolean } = {},
): SpanningTree {
  if (g.directed) throw new MdmlError('value', 'minimumSpanningTree: the graph must be undirected')
  const maximum = options.maximum ?? false
  const run = runAlgorithm(spanningTreeMethods.make({ kind: options.algorithm ?? 'kruskal' })(g, { maximum }), {
    maxSteps: g.edges.length + 2 * g.n + 1,
  })
  const chosen = run.state.chosen
  const edges = chosen.map((k) => g.edges[k] as Edge).sort((p, q) => p.source - q.source || p.target - q.target)
  const weight = kahanSum(edges.map((e) => e.weight))
  return {
    edges,
    weight,
    graph: graph(
      edges.map((e) => [e.source, e.target, e.weight] as const),
      { nodes: g.labels, nodeData: g.nodeData },
    ),
  }
}

/**
 * Whether an undirected graph is a tree: connected with n - 1 edges (NetworkX is_tree).
 *
 * @param g - The graph.
 * @returns True for a tree (false for the empty graph and for directed graphs).
 * @example
 * ```ts
 * import { balancedTree, cycleGraph, isTree } from 'mdml-engine/graph'
 * [isTree(balancedTree(2, 3)), isTree(cycleGraph(4))] // => [true, false]
 * ```
 */
export function isTree(g: Graph): boolean {
  return !g.directed && g.n > 0 && g.edges.length === g.n - 1 && connectedComponents(g).count === 1
}

/**
 * Whether an undirected graph is a forest: every component is a tree (NetworkX is_forest).
 *
 * @param g - The graph.
 * @returns True for a forest.
 * @example
 * ```ts
 * import { graph, isForest } from 'mdml-engine/graph'
 * isForest(graph([[0, 1], [2, 3]])) // => true
 * ```
 */
export function isForest(g: Graph): boolean {
  return !g.directed && g.n > 0 && g.edges.length === g.n - connectedComponents(g).count
}

/**
 * A tree hung from a root: parents, children (ascending), depths in edges, weighted distances from the root, and the
 * nodes in breadth-first order. Nodes outside the root's component have parent -1 and depth -1.
 */
export interface RootedTree {
  readonly root: number
  readonly n: number
  readonly parent: Int32Array
  readonly children: readonly (readonly number[])[]
  readonly depth: Int32Array
  /** The sum of edge weights from the root. */
  readonly distance: Float64Array
  /** The tree's nodes in breadth-first order from the root. */
  readonly order: readonly number[]
}

/**
 * Root the tree containing a node (as NetworkX bfs_tree on a tree): parents, children, depths and distances.
 *
 * @param g - An undirected graph whose component of `root` is a tree.
 * @param root - The root.
 * @returns The rooted tree.
 * @throws MdmlError with code `value` when the graph is directed or the root's component has a cycle.
 * @concept trees
 * @example
 * ```ts
 * import { balancedTree, rootTree } from 'mdml-engine/graph'
 * const t = rootTree(balancedTree(2, 2), 0)
 * Array.from(t.depth) // => [0, 1, 1, 2, 2, 2, 2]
 * t.children[1] // => [3, 4]
 * ```
 */
export function rootTree(g: Graph, root: NodeId): RootedTree {
  if (g.directed) throw new MdmlError('value', 'rootTree: the graph must be undirected')
  const r = nodeIndex(g, root)
  const a = g.out
  const parent = new Int32Array(g.n).fill(-1)
  const depth = new Int32Array(g.n).fill(-1)
  const distance = new Float64Array(g.n).fill(Number.NaN)
  const children: number[][] = Array.from({ length: g.n }, () => [])
  depth[r] = 0
  distance[r] = 0
  const order = [r]
  for (let head = 0; head < order.length; head++) {
    const u = order[head] ?? 0
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      if (v === parent[u]) continue
      if (depth[v] !== -1 || v === u) throw new MdmlError('value', 'rootTree: the component has a cycle')
      parent[v] = u
      depth[v] = (depth[u] ?? 0) + 1
      distance[v] = (distance[u] ?? 0) + (a.weights[k] ?? 0)
      children[u]?.push(v)
      order.push(v)
    }
  }
  return { root: r, n: g.n, parent, children, depth, distance, order }
}

/**
 * The size of every subtree of a rooted tree (the node and its descendants; 0 outside the tree).
 *
 * @param tree - The rooted tree.
 * @returns Subtree sizes.
 * @example
 * ```ts
 * import { balancedTree, rootTree, subtreeSizes } from 'mdml-engine/graph'
 * Array.from(subtreeSizes(rootTree(balancedTree(2, 2), 0))) // => [7, 3, 3, 1, 1, 1, 1]
 * ```
 */
export function subtreeSizes(tree: RootedTree): Int32Array {
  const size = new Int32Array(tree.n)
  for (let i = tree.order.length - 1; i >= 0; i--) {
    const v = tree.order[i] ?? 0
    size[v] = (size[v] ?? 0) + 1
    const p = tree.parent[v] ?? -1
    if (p >= 0) size[p] = (size[p] ?? 0) + (size[v] ?? 0)
  }
  return size
}

/**
 * The nodes of a rooted tree in preorder, postorder or level order (children visited in ascending order).
 *
 * @param tree - The rooted tree.
 * @param kind - `'pre'` (default), `'post'` or `'level'`.
 * @returns The order.
 * @example
 * ```ts
 * import { balancedTree, rootTree, treeOrder } from 'mdml-engine/graph'
 * const t = rootTree(balancedTree(2, 2), 0)
 * treeOrder(t) // => [0, 1, 3, 4, 2, 5, 6]
 * treeOrder(t, 'post') // => [3, 4, 1, 5, 6, 2, 0]
 * ```
 */
export function treeOrder(tree: RootedTree, kind: 'pre' | 'post' | 'level' = 'pre'): number[] {
  if (kind === 'level') return [...tree.order]
  const out: number[] = []
  const stack: [number, number][] = [[tree.root, 0]]
  if (kind === 'pre') out.push(tree.root)
  while (stack.length > 0) {
    const frame = stack[stack.length - 1] as [number, number]
    const [v, i] = frame
    const c = tree.children[v]?.[i]
    if (c === undefined) {
      stack.pop()
      if (kind === 'post') out.push(v)
      continue
    }
    frame[1] = i + 1
    if (kind === 'pre') out.push(c)
    stack.push([c, 0])
  }
  return out
}

/**
 * The Euler tour of a rooted tree: the nodes in the order a depth-first walk visits them, each node listed on entry and
 * again after each child returns (2n - 1 entries). With the depths, the range-minimum view of lowest common
 * ancestors.
 *
 * @param tree - The rooted tree.
 * @returns The tour.
 * @example
 * ```ts
 * import { eulerTour, pathGraph, rootTree } from 'mdml-engine/graph'
 * eulerTour(rootTree(pathGraph(3), 1)) // => [1, 0, 1, 2, 1]
 * ```
 */
export function eulerTour(tree: RootedTree): number[] {
  const tour = [tree.root]
  const stack: [number, number][] = [[tree.root, 0]]
  while (stack.length > 0) {
    const frame = stack[stack.length - 1] as [number, number]
    const [v, i] = frame
    const c = tree.children[v]?.[i]
    if (c === undefined) {
      stack.pop()
      const p = stack[stack.length - 1]
      if (p !== undefined) tour.push(p[0])
      continue
    }
    frame[1] = i + 1
    tour.push(c)
    stack.push([c, 0])
  }
  return tour
}

/** A preprocessed lowest-common-ancestor index: answers each query in O(log n). */
export interface LcaIndex {
  readonly lca: (u: number, v: number) => number
}

/**
 * Binary-lifting tables for lowest common ancestors (O(n log n) preprocessing, O(log n) per query).
 *
 * @param tree - The rooted tree.
 * @returns The index.
 * @concept trees
 * @example
 * ```ts
 * import { balancedTree, lcaIndex, rootTree } from 'mdml-engine/graph'
 * const index = lcaIndex(rootTree(balancedTree(2, 3), 0))
 * [index.lca(7, 10), index.lca(7, 8), index.lca(3, 14)] // => [1, 3, 0]
 * ```
 */
export function lcaIndex(tree: RootedTree): LcaIndex {
  const levels = Math.max(1, Math.ceil(Math.log2(Math.max(2, tree.n))))
  const up: Int32Array[] = [Int32Array.from(tree.parent, (p, v) => (p < 0 ? v : p))]
  for (let j = 1; j < levels; j++) {
    const prev = up[j - 1] as Int32Array
    up.push(Int32Array.from(prev, (p) => prev[p] ?? p))
  }
  const lca = (u: number, v: number): number => {
    if ((tree.depth[u] ?? -1) < 0 || (tree.depth[v] ?? -1) < 0) {
      throw new MdmlError('value', `lca: nodes ${u} and ${v} are not both in the tree`)
    }
    let a = u
    let b = v
    if ((tree.depth[a] ?? 0) < (tree.depth[b] ?? 0)) [a, b] = [b, a]
    let diff = (tree.depth[a] ?? 0) - (tree.depth[b] ?? 0)
    for (let j = 0; diff > 0; j++, diff >>= 1) if ((diff & 1) === 1) a = up[j]?.[a] ?? a
    if (a === b) return a
    for (let j = levels - 1; j >= 0; j--) {
      const ua = up[j]?.[a] ?? a
      const ub = up[j]?.[b] ?? b
      if (ua !== ub) {
        a = ua
        b = ub
      }
    }
    return tree.parent[a] ?? a
  }
  return { lca }
}

/**
 * The lowest common ancestor of two nodes of a rooted tree (NetworkX lowest_common_ancestor): the deepest node that
 * is an ancestor of both. For many queries, build an {@link lcaIndex} once.
 *
 * @param tree - The rooted tree.
 * @param u - A node.
 * @param v - Another node.
 * @returns Their lowest common ancestor.
 * @throws MdmlError with code `value` when a node is outside the tree.
 * @concept trees
 * @example
 * ```ts
 * import { balancedTree, lowestCommonAncestor, rootTree } from 'mdml-engine/graph'
 * lowestCommonAncestor(rootTree(balancedTree(2, 2), 0), 3, 4) // => 1
 * ```
 */
export function lowestCommonAncestor(tree: RootedTree, u: number, v: number): number {
  if ((tree.depth[u] ?? -1) < 0 || (tree.depth[v] ?? -1) < 0) {
    throw new MdmlError('value', `lowestCommonAncestor: nodes ${u} and ${v} are not both in the tree`)
  }
  let a = u
  let b = v
  while ((tree.depth[a] ?? 0) > (tree.depth[b] ?? 0)) a = tree.parent[a] ?? a
  while ((tree.depth[b] ?? 0) > (tree.depth[a] ?? 0)) b = tree.parent[b] ?? b
  while (a !== b) {
    a = tree.parent[a] ?? a
    b = tree.parent[b] ?? b
  }
  return a
}

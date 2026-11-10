/**
 * Builders of the structures graph CRFs are used on: trees and forests (from a parent array or an edge list), grids
 * with the 4- or 8-neighbourhood and undirected or directional edge types, skip chains (a chain with extra edges
 * between repeated tokens, Sutton and McCallum 2004), higher-order chains (label n-grams as cliques) and fully connected
 * graphs. Each structure has an `...Input` function that turns a list of node attribute records into a `GraphInput`,
 * and a `...Crf` function that makes the model with the edge types and symmetry that go with it; the same oracle
 * (`graphOracle`) trains and decodes all of them, with weights tied over every instance of the structure.
 */

import { OmaiError } from '../../base/index.ts'
import { completeGraph, graph, gridGraph, isForest, pathGraph } from '../../graph/index.ts'
import type { ItemFeatures } from './crf.ts'
import { type GraphCrf, type GraphCrfOptions, type GraphEdge, type GraphInput, graphCrf } from './graph-crf.ts'

/** Computes the attributes of an edge from the attribute records of its two ends (a contrast, a distance, a match). */
export type EdgeAttributes = (from: ItemFeatures, to: ItemFeatures) => ItemFeatures

/** The option every input builder shares. */
export interface InputOptions {
  /** Attributes of each edge from the records of its ends, which switch on the edge-attribute weights. */
  readonly edgeAttributes?: EdgeAttributes
}

function edgeOf(
  nodes: readonly ItemFeatures[],
  from: number,
  to: number,
  type: string,
  attributes: EdgeAttributes | undefined,
): GraphEdge {
  return attributes === undefined
    ? { from, to, type }
    : { from, to, type, attributes: attributes(nodes[from] as ItemFeatures, nodes[to] as ItemFeatures) }
}

/** The structure of a tree or forest: a parent array (-1 for a root) or an edge list. */
export type TreeStructure =
  | { readonly parents: readonly number[] }
  | { readonly edges: readonly (readonly [number, number])[] }

/**
 * The input of a tree or forest CRF: the nodes joined by parent-to-child edges (from a parent array, where `parents[i]`
 * is the parent of node i and -1 marks a root) or by the given edges, in their direction.
 *
 * @param nodes - The attribute record of each node.
 * @param tree - A parent array or an edge list.
 * @param options - The edge type (default `'parent'`) and edge attributes.
 * @returns The input.
 * @throws OmaiError with code `shape` for a parent array of the wrong length, `value` for a structure with a cycle, a repeated edge or an index out of range.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { treeInput } from 'omai-engine/pgm'
 * treeInput([{}, {}, {}], { parents: [-1, 0, 0] }).edges.map((e) => [e.from, e.to, e.type]) // => [[0, 1, 'parent'], [0, 2, 'parent']]
 * ```
 */
export function treeInput(
  nodes: readonly ItemFeatures[],
  tree: TreeStructure,
  options: InputOptions & { readonly type?: string } = {},
): GraphInput {
  const type = options.type ?? 'parent'
  let pairs: readonly (readonly [number, number])[]
  if ('parents' in tree) {
    if (tree.parents.length !== nodes.length) throw new OmaiError('shape', 'treeInput: one parent per node')
    pairs = tree.parents.flatMap((p, i): [number, number][] => (p < 0 ? [] : [[p, i]]))
  } else pairs = tree.edges
  for (const [a, b] of pairs)
    if (!(Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a < nodes.length && b < nodes.length))
      throw new OmaiError('value', `treeInput: edge ${a}-${b} is out of range`)
  if (nodes.length > 0 && !isForest(graph(pairs, { nodes: nodes.length, duplicates: 'error' })))
    throw new OmaiError('value', 'treeInput: the structure has a cycle')
  return { nodes, edges: pairs.map(([a, b]) => edgeOf(nodes, a, b, type, options.edgeAttributes)) }
}

/**
 * A CRF for trees and forests: one directional edge type (`'parent'` by default, or your own `edgeTypes`).
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attribute vocabulary.
 * @param options - The options of {@link graphCrf}.
 * @returns The model.
 * @concept conditional-random-fields
 * @index structure Tree and forest CRF, exact by sum-product on the tree.
 * @example
 * ```ts
 * import { treeCrf } from 'omai-engine/pgm'
 * treeCrf(['a', 'b'], ['f']).edgeTypes // => ['parent']
 * ```
 */
export function treeCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  options: GraphCrfOptions = {},
): GraphCrf {
  return graphCrf(labels, nodeAttributes, { edgeTypes: ['parent'], ...options })
}

/** Options of {@link gridInput} and {@link gridCrf}. */
export interface GridOptions {
  /** 4 (left, right, up, down) or 8 (with the diagonals; default 4). */
  readonly neighbourhood?: 4 | 8
  /**
   * Distinct edge types per direction (`right`, `down`, and for 8 also `down-right`, `down-left`), so the pairwise
   * weights depend on the direction, as pystruct's `DirectionalGridCRF`. Without it the types are `grid` (axis-aligned)
   * and `diagonal`, symmetric in the label pair.
   */
  readonly directional?: boolean
}

function gridTypes(options: GridOptions): { axis: [string, string]; diagonal: [string, string] } {
  return options.directional === true
    ? { axis: ['right', 'down'], diagonal: ['down-right', 'down-left'] }
    : { axis: ['grid', 'grid'], diagonal: ['diagonal', 'diagonal'] }
}

/**
 * The input of an image-like grid: `rows` x `cols` nodes in row-major order, joined to their right and lower
 * neighbours (and, for the 8-neighbourhood, their lower-right and lower-left ones). Each edge runs from the earlier
 * node to the later one.
 *
 * @param nodes - The attribute record of each cell, row-major.
 * @param rows - The number of rows.
 * @param cols - The number of columns.
 * @param options - The neighbourhood, directional types and edge attributes.
 * @returns The input.
 * @throws OmaiError with code `shape` when the cells are not rows x cols.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { gridInput } from 'omai-engine/pgm'
 * [gridInput(Array.from({ length: 6 }, () => ({})), 2, 3).edges.length, gridInput(Array.from({ length: 6 }, () => ({})), 2, 3, { neighbourhood: 8 }).edges.length] // => [7, 11]
 * ```
 */
export function gridInput(
  nodes: readonly ItemFeatures[],
  rows: number,
  cols: number,
  options: GridOptions & InputOptions = {},
): GraphInput {
  if (nodes.length !== rows * cols)
    throw new OmaiError('shape', `gridInput: ${nodes.length} cells for ${rows} x ${cols}`)
  const types = gridTypes(options)
  const edges: GraphEdge[] = gridGraph(rows, cols).edges.map((e) =>
    edgeOf(
      nodes,
      e.source,
      e.target,
      e.target - e.source === cols ? types.axis[1] : types.axis[0],
      options.edgeAttributes,
    ),
  )
  if (options.neighbourhood === 8)
    for (let r = 0; r + 1 < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = r * cols + c
        if (c + 1 < cols) edges.push(edgeOf(nodes, v, v + cols + 1, types.diagonal[0], options.edgeAttributes))
        if (c > 0) edges.push(edgeOf(nodes, v, v + cols - 1, types.diagonal[1], options.edgeAttributes))
      }
  return { nodes, edges }
}

/**
 * A CRF for grids: the edge types that {@link gridInput} produces for the same options, symmetric when they are
 * undirected.
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attribute vocabulary.
 * @param options - The grid options and those of {@link graphCrf}.
 * @returns The model.
 * @concept conditional-random-fields
 * @index structure Grid CRF: 4- or 8-neighbourhood, undirected or directional edge types.
 * @example
 * ```ts
 * import { gridCrf } from 'omai-engine/pgm'
 * [gridCrf(['a', 'b'], []).edgeTypes, gridCrf(['a', 'b'], [], { directional: true, neighbourhood: 8 }).edgeTypes] // => [['grid'], ['right', 'down', 'down-right', 'down-left']]
 * ```
 */
export function gridCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  options: GridOptions & GraphCrfOptions = {},
): GraphCrf {
  const types = gridTypes(options)
  const used = [...new Set([...types.axis, ...(options.neighbourhood === 8 ? types.diagonal : [])])]
  return graphCrf(labels, nodeAttributes, { edgeTypes: used, symmetric: options.directional !== true, ...options })
}

/** Options of {@link skipChainInput}. */
export interface SkipChainOptions extends InputOptions {
  /**
   * A key per node: nodes with the same key (and not adjacent) are joined by a skip edge, as Sutton and McCallum join
   * repeated capitalised words, which should take the same label. `undefined` keys join nothing.
   */
  readonly keys?: readonly (string | undefined)[]
  /** Explicit skip edges as node pairs. */
  readonly skips?: readonly (readonly [number, number])[]
  /** Skip edges only between nodes at most this far apart (default unbounded). */
  readonly maxDistance?: number
}

/**
 * The input of a skip-chain CRF: the chain edges (type `chain`, from each node to the next) and the skip edges (type
 * `skip`, from the earlier node to the later), which carry long-range dependencies a plain chain cannot.
 *
 * @param nodes - The attribute record of each position.
 * @param options - Keys or explicit pairs for the skip edges, the longest skip and edge attributes.
 * @returns The input.
 * @throws OmaiError with code `shape` for keys of the wrong length, `value` for a skip outside the sequence.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { skipChainInput } from 'omai-engine/pgm'
 * skipChainInput([{}, {}, {}, {}], { keys: ['x', undefined, undefined, 'x'] }).edges.map((e) => [e.from, e.to, e.type]) // => [[0, 1, 'chain'], [1, 2, 'chain'], [2, 3, 'chain'], [0, 3, 'skip']]
 * ```
 */
export function skipChainInput(nodes: readonly ItemFeatures[], options: SkipChainOptions = {}): GraphInput {
  const n = nodes.length
  if (options.keys !== undefined && options.keys.length !== n)
    throw new OmaiError('shape', 'skipChainInput: one key per node')
  const edges: GraphEdge[] = pathGraph(n).edges.map((e) =>
    edgeOf(nodes, e.source, e.target, 'chain', options.edgeAttributes),
  )
  const seen = new Set<number>()
  const skip = (a: number, b: number): void => {
    const i = Math.min(a, b)
    const j = Math.max(a, b)
    if (!(i >= 0 && j < n && i !== j && Number.isInteger(i) && Number.isInteger(j)))
      throw new OmaiError('value', `skipChainInput: skip ${a}-${b} is outside the sequence`)
    if (j === i + 1 || seen.has(i * n + j) || j - i > (options.maxDistance ?? Number.POSITIVE_INFINITY)) return
    seen.add(i * n + j)
    edges.push(edgeOf(nodes, i, j, 'skip', options.edgeAttributes))
  }
  const keys = options.keys ?? []
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) if (keys[i] !== undefined && keys[i] === keys[j]) skip(i, j)
  for (const [a, b] of options.skips ?? []) skip(a, b)
  return { nodes, edges }
}

/**
 * A skip-chain CRF: a directional `chain` type and an undirected (symmetric) `skip` type.
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attribute vocabulary.
 * @param options - The options of {@link graphCrf}.
 * @returns The model.
 * @concept conditional-random-fields
 * @index structure Skip-chain CRF: a chain plus edges between repeated tokens.
 * @example
 * ```ts
 * import { skipChainCrf } from 'omai-engine/pgm'
 * skipChainCrf(['a', 'b'], ['f']).symmetric // => ['skip']
 * ```
 */
export function skipChainCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  options: GraphCrfOptions = {},
): GraphCrf {
  return graphCrf(labels, nodeAttributes, { edgeTypes: ['chain', 'skip'], symmetric: ['skip'], ...options })
}

/**
 * The input of an order-`order` chain CRF: transition edges between consecutive positions (type `transition`) and, for
 * each longer window of 3 to order + 1 positions, a clique (type `ngram3`, `ngram4`, ...) whose weights are tied across
 * every window of the sequence. The structure is exact to infer: a junction tree over the cliques is the chain of
 * label tuples.
 *
 * @param nodes - The attribute record of each position.
 * @param order - The number of previous labels each label depends on (1 is the ordinary linear chain).
 * @returns The input.
 * @throws OmaiError with code `value` for an order below 1.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { higherOrderChainInput } from 'omai-engine/pgm'
 * higherOrderChainInput([{}, {}, {}, {}], 2).cliques?.map((c) => c.nodes) // => [[0, 1, 2], [1, 2, 3]]
 * ```
 */
export function higherOrderChainInput(nodes: readonly ItemFeatures[], order: number): GraphInput {
  if (!Number.isInteger(order) || order < 1)
    throw new OmaiError('value', 'higherOrderChainInput: the order must be an integer >= 1')
  const n = nodes.length
  const edges: GraphEdge[] = pathGraph(n).edges.map((e) => ({ from: e.source, to: e.target, type: 'transition' }))
  const cliques = []
  for (let size = 3; size <= order + 1; size++)
    for (let end = size - 1; end < n; end++)
      cliques.push({ type: `ngram${size}`, nodes: Array.from({ length: size }, (_, k) => end - size + 1 + k) })
  return { nodes, edges, cliques }
}

/**
 * An order-`order` chain CRF: per-label node weights, bigram transition weights and a weight per label n-gram of
 * every size up to order + 1, all tied along the sequence. Order 1 has the parameters of `linearChainCrf`; order 2 adds a
 * weight per label trigram. With the higher-order weights at zero it is the lower-order model.
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attribute vocabulary.
 * @param order - The order, at least 1.
 * @param options - The options of {@link graphCrf}.
 * @returns The model.
 * @throws OmaiError with code `value` for an order below 1.
 * @concept conditional-random-fields
 * @index structure Order-k chain CRF with tied label n-gram weights, exact by junction tree.
 * @example
 * ```ts
 * import { higherOrderChainCrf } from 'omai-engine/pgm'
 * higherOrderChainCrf(['a', 'b'], [], 2, { bias: false }).layout.size // => 4 + 8
 * ```
 */
export function higherOrderChainCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  order: number,
  options: GraphCrfOptions = {},
): GraphCrf {
  if (!Number.isInteger(order) || order < 1)
    throw new OmaiError('value', 'higherOrderChainCrf: the order must be an integer >= 1')
  const cliqueTypes = Object.fromEntries(Array.from({ length: order - 1 }, (_, k) => [`ngram${k + 3}`, k + 3]))
  return graphCrf(labels, nodeAttributes, { edgeTypes: ['transition'], cliqueTypes, ...options })
}

/**
 * The input of a fully connected CRF: an edge between every pair of nodes (from the earlier to the later), as in dense
 * CRFs and relational models where every item interacts with every other.
 *
 * @param nodes - The attribute record of each node.
 * @param options - The edge type (default `'pair'`) and edge attributes.
 * @returns The input.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { fullyConnectedInput } from 'omai-engine/pgm'
 * fullyConnectedInput([{}, {}, {}, {}]).edges.length // => 6
 * ```
 */
export function fullyConnectedInput(
  nodes: readonly ItemFeatures[],
  options: InputOptions & { readonly type?: string } = {},
): GraphInput {
  const type = options.type ?? 'pair'
  return {
    nodes,
    edges: completeGraph(nodes.length).edges.map((e) =>
      edgeOf(nodes, e.source, e.target, type, options.edgeAttributes),
    ),
  }
}

/**
 * A fully connected CRF: one undirected (symmetric) `pair` type.
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attribute vocabulary.
 * @param options - The options of {@link graphCrf}.
 * @returns The model.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { fullyConnectedCrf } from 'omai-engine/pgm'
 * fullyConnectedCrf(['a', 'b'], []).edgeTypes // => ['pair']
 * ```
 */
export function fullyConnectedCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  options: GraphCrfOptions = {},
): GraphCrf {
  return graphCrf(labels, nodeAttributes, { edgeTypes: ['pair'], symmetric: true, ...options })
}

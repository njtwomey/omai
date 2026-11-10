/**
 * Conditional random fields on any graph, with parameters tied across instances: the model of trees, grids, skip
 * chains, fully connected graphs and higher-order chains, and the arbitrary structure of a relational problem. An input
 * is a graph of its own (node attribute records, typed edges with optional attribute records, and optionally cliques of
 * three or more nodes), so instances differ in size and shape, but the weights belong to the model: one per (attribute,
 * label) for nodes, one per (edge type, label pair) for edges, one per (edge attribute, label pair) for edge features,
 * and one per (clique type, label tuple) for cliques, as in Sutton and McCallum's templated CRFs and pystruct's
 * `GraphCRF`, `EdgeFeatureGraphCRF` and `DirectionalGridCRF`. The oracle that trains it is in `graph-oracle.ts`, the
 * builders of the common structures in `graph-structures.ts`.
 */

import { OmaiError } from '../../base/index.ts'
import type { ItemFeatures } from './crf.ts'

/** What a graph CRF labels with: one label per node, as names or indices into the model's labels. */
export type GraphLabelling = readonly (string | number)[]

/**
 * An edge of an input graph. The potential of a labelling is `w(type, y_from, y_to)`, so the direction matters unless
 * the type is declared symmetric in the model. `attributes` (such as the contrast between two pixels) switch on the
 * edge-attribute weights `w(attribute, y_from, y_to)`.
 */
export interface GraphEdge {
  readonly from: number
  readonly to: number
  /** The edge type (default `'edge'`); every type has its own label-pair weights. */
  readonly type?: string
  readonly attributes?: ItemFeatures
}

/**
 * A clique of three or more nodes (or any number) with its own weights per label tuple, for higher-order structure:
 * `w(type, y_nodes[0], y_nodes[1], ...)`. The arity of a type is fixed by the model.
 */
export interface GraphClique {
  readonly nodes: readonly number[]
  readonly type: string
}

/**
 * One input of a graph CRF: the attributes of each node (the observation, as a name-to-value record), the edges and the
 * higher-order cliques. Nodes are numbered by their position in `nodes`.
 *
 * @concept conditional-random-fields
 */
export interface GraphInput {
  readonly nodes: readonly ItemFeatures[]
  readonly edges: readonly GraphEdge[]
  readonly cliques?: readonly GraphClique[]
}

/** A clique type of a model: its name and the number of nodes of its cliques. */
export interface GraphCliqueType {
  readonly name: string
  readonly arity: number
}

/** Where each block of weights starts in the parameter vector (all blocks are row-major as documented on {@link GraphCrf}). */
export interface GraphLayout {
  /** The number of labels. */
  readonly labels: number
  /** The size of the node block: (the bias, if any, and each attribute) times the labels. */
  readonly nodeSize: number
  /** The start of each edge type's label-pair block. */
  readonly pairOffsets: readonly number[]
  /** The start of the edge-attribute blocks (one label-pair block per attribute). */
  readonly attributeOffset: number
  /** The start of each clique type's block (labels to the power of the arity). */
  readonly cliqueOffsets: readonly number[]
  /** The number of parameters. */
  readonly size: number
}

/**
 * A conditional random field on arbitrary graphs: log p(y | x) = score(x, y) - log Z(x), with
 * score = sum_i sum_a x_i(a) w(a, y_i) + sum_edges [w(type, y_f, y_t) + sum_b e(b) w(b, y_f, y_t)] + sum_cliques
 * w(type, y_nodes). The parameter vector is the node weights (row-major [attribute][label], the bias first when there
 * is one), then the label-pair weights of each edge type (row-major [from][to], or the upper triangle row by row for a
 * symmetric type), then the edge-attribute weights (one label-pair block per attribute), then one block per clique
 * type (row-major over the label tuple). The model holds structure and weights; the oracle takes weights as an
 * argument, so a solver can evaluate any point.
 *
 * @concept conditional-random-fields
 */
export interface GraphCrf {
  readonly kind: 'graphCrf'
  readonly labels: readonly string[]
  /** The node attribute vocabulary (attributes outside it are ignored, as at test time in CRFsuite). */
  readonly nodeAttributes: readonly string[]
  /** Whether every node has a label bias, an attribute that is always 1. */
  readonly bias: boolean
  readonly edgeTypes: readonly string[]
  /** The edge types whose label-pair weights are symmetric, w(a, b) = w(b, a): undirected relations. */
  readonly symmetric: readonly string[]
  /** The edge attribute vocabulary. */
  readonly edgeAttributes: readonly string[]
  /** Whether the edge-attribute weights are symmetric in the label pair. */
  readonly symmetricAttributes: boolean
  readonly cliqueTypes: readonly GraphCliqueType[]
  readonly layout: GraphLayout
  readonly weights: Float64Array
}

/** Options of {@link graphCrf}. */
export interface GraphCrfOptions {
  /** Edge types with their own label-pair weights (default `['edge']`). */
  readonly edgeTypes?: readonly string[]
  /** The edge types that are undirected (`true` for all of them): their label-pair weights are symmetric. */
  readonly symmetric?: readonly string[] | boolean
  /** The edge attribute vocabulary: the attributes of `GraphEdge.attributes` that carry weights. */
  readonly edgeAttributes?: readonly string[]
  /** Tie the edge-attribute weights for (a, b) and (b, a). */
  readonly symmetricAttributes?: boolean
  /** Clique types by name with their number of nodes (at least 2). */
  readonly cliqueTypes?: Readonly<Record<string, number>>
  /** A label bias on every node (default true). */
  readonly bias?: boolean
  /** The weights (default zero). */
  readonly weights?: ArrayLike<number>
}

const triangle = (k: number): number => (k * (k + 1)) / 2

/**
 * Build a graph CRF from its labels, node attribute vocabulary and the kinds of edges and cliques it has.
 *
 * @param labels - The labels.
 * @param nodeAttributes - The node attributes that carry weights.
 * @param options - Edge types, symmetry, edge attributes, clique types, the bias and the weights.
 * @returns The model.
 * @throws OmaiError with code `value` for repeated labels or names and cliques of fewer than two nodes, `shape` for weights of the wrong length.
 * @concept conditional-random-fields
 * @index structure CRF model for trees, grids and any graph, with weights tied by type.
 * @example
 * ```ts
 * import { graphCrf, graphParameterNames } from 'omai-engine/pgm'
 * graphParameterNames(graphCrf(['a', 'b'], ['f'], { bias: false, symmetric: true })) // => ['node:f|a', 'node:f|b', 'pair:edge|a|a', 'pair:edge|a|b', 'pair:edge|b|b']
 * ```
 */
export function graphCrf(
  labels: readonly string[],
  nodeAttributes: readonly string[],
  options: GraphCrfOptions = {},
): GraphCrf {
  const K = labels.length
  const unique = (xs: readonly string[], what: string): void => {
    if (new Set(xs).size !== xs.length) throw new OmaiError('value', `graphCrf: ${what} must be distinct`)
  }
  unique(labels, 'labels')
  unique(nodeAttributes, 'node attributes')
  const edgeTypes = [...(options.edgeTypes ?? ['edge'])]
  unique(edgeTypes, 'edge types')
  const edgeAttributes = [...(options.edgeAttributes ?? [])]
  unique(edgeAttributes, 'edge attributes')
  const symmetric =
    options.symmetric === true ? [...edgeTypes] : options.symmetric === false ? [] : [...(options.symmetric ?? [])]
  for (const s of symmetric)
    if (!edgeTypes.includes(s)) throw new OmaiError('value', `graphCrf: symmetric type ${s} is not an edge type`)
  const symmetricAttributes = options.symmetricAttributes ?? false
  const cliqueTypes = Object.entries(options.cliqueTypes ?? {}).map(([name, arity]): GraphCliqueType => {
    if (!Number.isInteger(arity) || arity < 2)
      throw new OmaiError('value', `graphCrf: clique type ${name} needs arity >= 2`)
    return { name, arity }
  })
  const bias = options.bias ?? true
  const nodeSize = (nodeAttributes.length + (bias ? 1 : 0)) * K
  let at = nodeSize
  const pairOffsets = edgeTypes.map((t) => {
    const o = at
    at += symmetric.includes(t) ? triangle(K) : K * K
    return o
  })
  const attributeOffset = at
  at += edgeAttributes.length * (symmetricAttributes ? triangle(K) : K * K)
  const cliqueOffsets = cliqueTypes.map((c) => {
    const o = at
    at += K ** c.arity
    return o
  })
  const layout: GraphLayout = { labels: K, nodeSize, pairOffsets, attributeOffset, cliqueOffsets, size: at }
  if (options.weights !== undefined && options.weights.length !== at)
    throw new OmaiError('shape', `graphCrf: the weights need ${at} entries`)
  return {
    kind: 'graphCrf',
    labels: [...labels],
    nodeAttributes: [...nodeAttributes],
    bias,
    edgeTypes,
    symmetric,
    edgeAttributes,
    symmetricAttributes,
    cliqueTypes,
    layout,
    weights: options.weights === undefined ? new Float64Array(at) : Float64Array.from(options.weights),
  }
}

/**
 * The names of the parameters, in weight order: `node:<attribute>|<label>` (`<bias>` for the bias),
 * `pair:<type>|<from>|<to>` (the upper triangle for a symmetric type), `edge:<attribute>|<from>|<to>` and
 * `clique:<type>|<label>|<label>...`.
 *
 * @param model - The model.
 * @returns One name per parameter.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphParameterNames } from 'omai-engine/pgm'
 * graphParameterNames(graphCrf(['a', 'b'], [], { cliqueTypes: { triple: 3 } })).slice(0, 2) // => ['node:<bias>|a', 'node:<bias>|b']
 * ```
 */
export function graphParameterNames(model: GraphCrf): string[] {
  const { labels } = model
  const K = labels.length
  const out: string[] = []
  for (const a of model.bias ? ['<bias>', ...model.nodeAttributes] : model.nodeAttributes)
    for (const l of labels) out.push(`node:${a}|${l}`)
  const block = (prefix: string, sym: boolean): void => {
    for (let i = 0; i < K; i++) for (let j = sym ? i : 0; j < K; j++) out.push(`${prefix}|${labels[i]}|${labels[j]}`)
  }
  for (const t of model.edgeTypes) block(`pair:${t}`, model.symmetric.includes(t))
  for (const b of model.edgeAttributes) block(`edge:${b}`, model.symmetricAttributes)
  for (const c of model.cliqueTypes) {
    const tuple = (prefix: string, depth: number): void => {
      if (depth === c.arity) {
        out.push(prefix)
        return
      }
      for (const l of labels) tuple(`${prefix}|${l}`, depth + 1)
    }
    tuple(`clique:${c.name}`, 0)
  }
  return out
}

/**
 * The weights of a model as a flat vector (a copy), the point a solver starts from.
 *
 * @param model - The model.
 * @returns The weights in {@link graphParameterNames} order.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphParameters } from 'omai-engine/pgm'
 * graphParameters(graphCrf(['a', 'b'], [], { bias: false })).length // => 4
 * ```
 */
export function graphParameters(model: GraphCrf): Float64Array {
  return Float64Array.from(model.weights)
}

/**
 * The same model with other weights.
 *
 * @param model - The model.
 * @param weights - A weight vector of the model's size.
 * @returns The model with those weights.
 * @throws OmaiError with code `shape` for a vector of the wrong length.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, withGraphParameters } from 'omai-engine/pgm'
 * Array.from(withGraphParameters(graphCrf(['a', 'b'], [], { bias: false }), [1, 2, 3, 4]).weights) // => [1, 2, 3, 4]
 * ```
 */
export function withGraphParameters(model: GraphCrf, weights: ArrayLike<number>): GraphCrf {
  if (weights.length !== model.layout.size)
    throw new OmaiError('shape', `withGraphParameters: the weights need ${model.layout.size} entries`)
  return { ...model, weights: Float64Array.from(weights) }
}

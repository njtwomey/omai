/**
 * Schemas for graph datasets: single graphs as edge lists with optional weights, node features and labels, collections
 * of graphs with graph labels or node-level train, validation and test masks, and knowledge-graph triples of head,
 * relation and tail. The constructors check node indices and field lengths.
 */

import { OmaiError } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { cardinality, checkRows, concrete, concreteFloat, defaultNames, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/** One graph: an edge list over `nNodes` nodes, with optional weights, features and labels. */
export interface GraphRecord {
  readonly nNodes: number
  /** Edges `[m, 2]` (int32) as `[source, target]`; an undirected edge is listed once. */
  readonly edges: Tensor<'int32'>
  readonly directed: boolean
  readonly edgeWeights?: Tensor
  /** Node features `[nNodes, f]`. */
  readonly nodeFeatures?: Tensor
  /** Edge features `[m, f]`. */
  readonly edgeFeatures?: Tensor
  /** A class per node `[nNodes]` (int32). */
  readonly nodeLabels?: Tensor<'int32'>
  /** A class per edge `[m]` (int32), such as a bond type. */
  readonly edgeLabels?: Tensor<'int32'>
  readonly nodeNames?: readonly string[]
}

/** Node-level train, validation and test masks for transductive tasks on one graph. */
export interface GraphMasks {
  readonly train?: Tensor<'bool'>
  readonly val?: Tensor<'bool'>
  readonly test?: Tensor<'bool'>
}

/**
 * One or many graphs. Graph-level tasks label each graph (`graphLabels`); node-level tasks label the nodes of each
 * graph and, for a single graph, split them with `masks`. The rows of the dataset are its graphs.
 */
export interface GraphData {
  readonly kind: 'graph'
  readonly graphs: readonly GraphRecord[]
  /** A label (int32 class or float target) per graph `[g]`. */
  readonly graphLabels?: Tensor
  readonly masks?: GraphMasks
  /** Names of the node classes (or of the graph classes for graph-level tasks). */
  readonly classes?: readonly string[]
  readonly meta: DatasetMeta
}

/** Knowledge-graph triples `(head, relation, tail)`, one per row. */
export interface TripleData {
  readonly kind: 'triples'
  readonly heads: Tensor<'int32'>
  readonly relations: Tensor<'int32'>
  readonly tails: Tensor<'int32'>
  readonly entityNames: readonly string[]
  readonly relationNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Options of {@link graphRecord}. */
export interface GraphRecordOptions {
  /** Default: one more than the largest node index. */
  readonly nNodes?: number
  readonly directed?: boolean
  readonly edgeWeights?: TensorLike
  readonly nodeFeatures?: TensorLike
  readonly edgeFeatures?: TensorLike
  readonly nodeLabels?: TensorLike
  readonly edgeLabels?: TensorLike
  readonly nodeNames?: readonly string[]
}

/**
 * One graph from an edge list `[[u, v], ...]` (or an `[m, 2]` tensor), checking node indices and field lengths.
 *
 * @param edges - The edges.
 * @param options - Node count, directedness, weights, features, labels and names.
 * @returns The graph.
 * @throws OmaiError with code `value` for a node index outside `[0, nNodes)`.
 * @example
 * ```ts
 * import { graphRecord } from 'omai-engine/data'
 * graphRecord([[0, 1], [1, 2]]).nNodes // => 3
 * ```
 */
export function graphRecord(edges: TensorLike, options: GraphRecordOptions = {}): GraphRecord {
  let e = concrete(edges, 'int32', 'graphRecord')
  if (e.shape.length === 1 && e.shape[0] === 0) e = fromStorage(new Int32Array(0), [0, 2], 'int32')
  if (e.shape.length !== 2 || e.shape[1] !== 2) throw new OmaiError('shape', `graphRecord: edges must be [m, 2]`)
  const nNodes = options.nNodes ?? cardinality(e.data)
  for (const v of e.data) {
    if (v < 0 || v >= nNodes) throw new OmaiError('value', `graphRecord: node ${v} outside [0, ${nNodes})`)
  }
  const m = e.shape[0] ?? 0
  const fields: Record<string, unknown> = { nNodes, edges: e, directed: options.directed ?? false }
  for (const k of ['edgeWeights', 'nodeFeatures', 'edgeFeatures'] as const) {
    const v = options[k]
    if (v !== undefined) fields[k] = concreteFloat(v, 'graphRecord')
  }
  for (const k of ['nodeLabels', 'edgeLabels'] as const) {
    const v = options[k]
    if (v !== undefined) fields[k] = concrete(v, 'int32', 'graphRecord')
  }
  if (options.nodeNames !== undefined) fields['nodeNames'] = options.nodeNames
  const out = fields as unknown as GraphRecord
  checkRows('graphRecord edges', m, {
    edgeWeights: out.edgeWeights,
    edgeFeatures: out.edgeFeatures,
    edgeLabels: out.edgeLabels,
  })
  checkRows('graphRecord nodes', nNodes, {
    nodeFeatures: out.nodeFeatures,
    nodeLabels: out.nodeLabels,
    nodeNames: out.nodeNames,
  })
  return out
}

/**
 * A graph dataset from graphs, with optional graph labels and node masks.
 *
 * @param graphs - The graphs (one for node-level tasks).
 * @param options - Graph labels, masks, class names and metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { graphData, graphRecord } from 'omai-engine/data'
 * graphData([graphRecord([[0, 1]]), graphRecord([[0, 1], [1, 2]])], { graphLabels: [0, 1] }).graphs.length // => 2
 * ```
 */
export function graphData(
  graphs: readonly GraphRecord[],
  options: {
    readonly graphLabels?: TensorLike
    readonly masks?: { readonly train?: TensorLike; readonly val?: TensorLike; readonly test?: TensorLike }
    readonly classes?: readonly string[]
    readonly meta?: Partial<DatasetMeta>
  } = {},
): GraphData {
  const labels = options.graphLabels === undefined ? undefined : concreteFloat(options.graphLabels, 'graphData')
  const intLabels =
    labels !== undefined && Array.from(labels.data).every((v) => Number.isInteger(v))
      ? concrete(labels, 'int32', 'graphData')
      : labels
  checkRows('graphData', graphs.length, { graphLabels: intLabels })
  let masks: GraphMasks | undefined
  if (options.masks !== undefined) {
    const nodes = graphs[0]?.nNodes ?? 0
    const m: Record<string, Tensor<'bool'>> = {}
    for (const [k, v] of Object.entries(options.masks)) {
      if (v === undefined) continue
      const t = concrete(v as TensorLike, 'bool', 'graphData')
      if ((t.shape[0] ?? 0) !== nodes) throw new OmaiError('shape', `graphData: mask ${k} must have ${nodes} entries`)
      m[k] = t
    }
    masks = m as GraphMasks
  }
  return {
    kind: 'graph',
    graphs,
    ...(intLabels === undefined ? {} : { graphLabels: intLabels }),
    ...(masks === undefined ? {} : { masks }),
    ...(options.classes === undefined ? {} : { classes: options.classes }),
    meta: metaFor('graph', options.meta),
  }
}

/**
 * Knowledge-graph triples from head, relation and tail indices.
 *
 * @param heads - Head entity indices `[n]`.
 * @param relations - Relation indices `[n]`.
 * @param tails - Tail entity indices `[n]`.
 * @param options - Entity and relation names, and metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { triples } from 'omai-engine/data'
 * triples([0, 1], [0, 0], [1, 2], { relationNames: ['parentOf'] }).entityNames // => ['e0', 'e1', 'e2']
 * ```
 */
export function triples(
  heads: TensorLike,
  relations: TensorLike,
  tails: TensorLike,
  options: {
    readonly entityNames?: readonly string[]
    readonly relationNames?: readonly string[]
    readonly meta?: Partial<DatasetMeta>
  } = {},
): TripleData {
  const h = concrete(heads, 'int32', 'triples')
  const r = concrete(relations, 'int32', 'triples')
  const t = concrete(tails, 'int32', 'triples')
  checkRows('triples', h.shape[0] ?? 0, { relations: r, tails: t })
  const nEntities = Math.max(cardinality(h.data), cardinality(t.data))
  return {
    kind: 'triples',
    heads: h,
    relations: r,
    tails: t,
    entityNames: options.entityNames ?? defaultNames('e', nEntities),
    relationNames: options.relationNames ?? defaultNames('r', cardinality(r.data)),
    meta: metaFor('triples', options.meta),
  }
}

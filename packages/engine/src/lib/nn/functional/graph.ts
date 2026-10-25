/**
 * Graph neural network layers as functions, PyTorch Geometric style: edge lists and message passing (each node
 * aggregates messages from its neighbours), edge softmax, self loops and GCN normalisation, the GCN, graph attention
 * (GAT), GraphSAGE and GIN convolutions, and global and top-k graph pooling.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  exp,
  expandDims,
  fromStorage,
  matmul,
  max,
  mean,
  mul,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  segmentSum,
  sqrt,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  tanh,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { leakyRelu } from './activations.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** An edge list: sources (row 0 of an edge index) and targets (row 1); messages flow from source to target. */
export interface Edges {
  readonly source: readonly number[]
  readonly target: readonly number[]
}

/**
 * Read an edge index `[2, E]` (PyTorch Geometric's layout: row 0 the source j, row 1 the target i of each edge j -> i).
 *
 * @param edgeIndex - `[2, E]` integers.
 * @returns The sources and targets.
 * @throws MdmlError with code `shape` for another shape.
 * @example
 * ```ts
 * import { edgeList } from 'mdml-engine/nn'
 * edgeList([[0, 1], [1, 2]]).target // => [1, 2]
 * ```
 */
export function edgeList(edgeIndex: TensorLike): Edges {
  const e = asTensor(edgeIndex, 'int32')
  if (e.shape.length !== 2 || e.shape[0] !== 2) throw new MdmlError('shape', 'edgeList: edge index must be [2, E]')
  const flat = toFlat(primalOf(e))
  const m = e.shape[1] ?? 0
  return { source: flat.slice(0, m), target: flat.slice(m) }
}

const indexTensor = (xs: readonly number[]): Tensor<'int32'> => fromStorage(Int32Array.from(xs), [xs.length], 'int32')

/** How messages arriving at a node are combined. */
export type Aggregation = 'sum' | 'mean' | 'max'

/**
 * Aggregate per-edge messages at their target nodes: sum, mean or max over the incoming edges (zeros for a node with
 * none). Differentiable in the messages.
 *
 * @param messages - `[E, ...]`.
 * @param target - The target node of each edge.
 * @param numNodes - The number of nodes.
 * @param reduce - `sum` (default), `mean` or `max`.
 * @returns `[numNodes, ...]`.
 * @example
 * ```ts
 * import { aggregate } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(aggregate([[1], [3], [5]], [0, 0, 2], 3, 'mean')) // => [[2], [0], [5]]
 * ```
 */
export function aggregate(
  messages: TensorLike,
  target: readonly number[],
  numNodes: number,
  reduce: Aggregation = 'sum',
): Value {
  const m = fl(messages)
  const rest = m.shape.slice(1)
  if (target.length === 0) return zeros([numNodes, ...rest])
  if (reduce === 'max') {
    const rows: Value[] = []
    for (let i = 0; i < numNodes; i++) {
      const incoming = target.flatMap((t, e) => (t === i ? [e] : []))
      rows.push(incoming.length === 0 ? zeros([1, ...rest]) : max(take(m, indexTensor(incoming), 0), 0, true))
    }
    return concat(rows, 0)
  }
  const total = segmentSum(m, indexTensor(target), numNodes)
  if (reduce === 'sum') return total
  const counts = new Float64Array(numNodes)
  for (const t of target) counts[t] = (counts[t] ?? 0) + 1
  const shape = [numNodes, ...rest.map(() => 1)]
  return div(
    total,
    fromStorage(
      Float64Array.from(counts, (c) => Math.max(c, 1)),
      shape,
      'float64',
    ),
  )
}

/** Options of {@link messagePassing}. */
export interface MessagePassingOptions {
  readonly aggregate?: Aggregation
  /**
   * The message of each edge from the source and target features (default: the source features), for example an edge
   * network or an attention-weighted value.
   */
  readonly message?: (source: Value, target: Value, edges: Edges) => Value
  /** Multiply each message by a weight `[E]`. */
  readonly edgeWeight?: TensorLike
  /** The number of nodes (default: the rows of x). */
  readonly numNodes?: number
}

/**
 * General message passing (Gilmer et al. 2017; PyTorch Geometric's MessagePassing): gather the features of each edge's
 * source and target, form a message, weight it, and aggregate the messages at the targets.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]` (source, target).
 * @param options - Aggregation, message function, edge weights and node count.
 * @returns The aggregated messages `[N, F']`.
 * @concept graph-neural-networks
 * @reference Gilmer, Schoenholz, Riley, Vinyals and Dahl (2017), Neural message passing for quantum chemistry.
 * @example
 * ```ts
 * import { messagePassing } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(messagePassing([[1], [2], [4]], [[0, 1, 2], [1, 2, 0]])) // => [[4], [1], [2]]
 * ```
 */
export function messagePassing(x: TensorLike, edgeIndex: TensorLike, options: MessagePassingOptions = {}): Value {
  const xv = fl(x)
  const edges = edgeList(edgeIndex)
  const n = options.numNodes ?? xv.shape[0] ?? 0
  const src = take(xv, indexTensor(edges.source), 0)
  let msg = options.message === undefined ? src : options.message(src, take(xv, indexTensor(edges.target), 0), edges)
  if (options.edgeWeight !== undefined) {
    const w = fl(options.edgeWeight)
    msg = mul(msg, reshape(w, [-1, ...msg.shape.slice(1).map(() => 1)]))
  }
  return aggregate(msg, edges.target, n, options.aggregate ?? 'sum')
}

/**
 * Softmax of edge scores over the incoming edges of each target node (the attention normalisation of GAT and graph
 * transformers). Shifted by each group's maximum, so stable; differentiable in the scores.
 *
 * @param scores - `[E, ...]` (for example `[E, heads]`).
 * @param target - The target node of each edge.
 * @param numNodes - The number of nodes.
 * @returns Weights `[E, ...]` summing to one over each node's incoming edges.
 * @example
 * ```ts
 * import { edgeSoftmax } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(edgeSoftmax([0, 0, 3], [1, 1, 0], 2)) // => [0.5, 0.5, 1]
 * ```
 */
export function edgeSoftmax(scores: TensorLike, target: readonly number[], numNodes: number): Value {
  const s = fl(scores)
  const rest = s.shape.slice(1)
  const width = rest.reduce((a, b) => a * b, 1)
  // The per-node maximum is a constant shift (softmax is invariant to it), taken from the concrete scores.
  const values = toFlat(primalOf(s))
  const top = new Float64Array(numNodes * width).fill(Number.NEGATIVE_INFINITY)
  for (const [e, t] of target.entries()) {
    for (let k = 0; k < width; k++) top[t * width + k] = Math.max(top[t * width + k] ?? 0, values[e * width + k] ?? 0)
  }
  const shift = take(
    fromStorage(
      top.map((v) => (Number.isFinite(v) ? v : 0)),
      [numNodes, ...rest],
      'float64',
    ),
    indexTensor(target),
    0,
  )
  const e = exp(sub(s, shift))
  const total = take(segmentSum(e, indexTensor(target), numNodes), indexTensor(target), 0)
  return div(e, total)
}

/**
 * Add a self loop i -> i for every node that has none (PyTorch Geometric's add_remaining_self_loops), with a fill
 * weight when edges are weighted.
 *
 * @param edgeIndex - `[2, E]`.
 * @param numNodes - The number of nodes.
 * @param edgeWeight - Weights `[E]` (optional).
 * @param fill - The new loops' weight (default 1).
 * @returns The extended edge index and weights.
 * @example
 * ```ts
 * import { addSelfLoops } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(addSelfLoops([[0], [1]], 2).edgeIndex) // => [[0, 0, 1], [1, 0, 1]]
 * ```
 */
export function addSelfLoops(
  edgeIndex: TensorLike,
  numNodes: number,
  edgeWeight?: TensorLike,
  fill = 1,
): { readonly edgeIndex: Tensor<'int32'>; readonly edgeWeight: Value } {
  const { source, target } = edgeList(edgeIndex)
  const has = new Set(source.flatMap((s, e) => (s === target[e] ? [s] : [])))
  const loops = Array.from({ length: numNodes }, (_, i) => i).filter((i) => !has.has(i))
  const src = [...source, ...loops]
  const dst = [...target, ...loops]
  const w =
    edgeWeight === undefined
      ? fromStorage(new Float64Array(source.length).fill(1), [source.length], 'float64')
      : fl(edgeWeight)
  return {
    edgeIndex: fromStorage(Int32Array.from([...src, ...dst]), [2, src.length], 'int32'),
    edgeWeight: concat([w, fromStorage(new Float64Array(loops.length).fill(fill), [loops.length], 'float64')], 0),
  }
}

/**
 * The symmetric GCN normalisation of edge weights (PyTorch Geometric's gcn_norm): with degrees counted at the targets
 * (self loops included), w_ji / sqrt(deg_j deg_i).
 *
 * @param edgeIndex - `[2, E]`.
 * @param numNodes - The number of nodes.
 * @param options - Edge weights and whether to add self loops (default true).
 * @returns The (extended) edge index and normalised weights.
 * @example
 * ```ts
 * import { gcnNorm } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gcnNorm([[0, 1], [1, 0]], 2).weight) // => [0.5, 0.5, 0.5, 0.5]
 * ```
 */
export function gcnNorm(
  edgeIndex: TensorLike,
  numNodes: number,
  options: { readonly edgeWeight?: TensorLike; readonly addSelfLoops?: boolean } = {},
): { readonly edgeIndex: Tensor<'int32'>; readonly weight: Value } {
  const looped =
    options.addSelfLoops === false
      ? {
          edgeIndex: asTensor(edgeIndex, 'int32') as Tensor<'int32'>,
          edgeWeight:
            options.edgeWeight === undefined
              ? fromStorage(
                  new Float64Array(edgeList(edgeIndex).source.length).fill(1),
                  [edgeList(edgeIndex).source.length],
                  'float64',
                )
              : fl(options.edgeWeight),
        }
      : addSelfLoops(edgeIndex, numNodes, options.edgeWeight)
  const { source, target } = edgeList(looped.edgeIndex)
  const deg = segmentSum(looped.edgeWeight, indexTensor(target), numNodes)
  const inv = pow(
    add(
      deg,
      fromStorage(
        Float64Array.from(toFlat(primalOf(deg)), (d) => (d === 0 ? 1 : 0)),
        [numNodes],
        'float64',
      ),
    ),
    -0.5,
  )
  const weight = mul(mul(take(inv, indexTensor(source), 0), looped.edgeWeight), take(inv, indexTensor(target), 0))
  return { edgeIndex: looped.edgeIndex, weight }
}

/** Options of {@link gcnConv}. */
export interface GcnOptions {
  readonly bias?: TensorLike
  readonly edgeWeight?: TensorLike
  readonly addSelfLoops?: boolean
  /** Apply the symmetric normalisation (default true). */
  readonly normalize?: boolean
}

/**
 * Graph convolution (Kipf and Welling 2017; PyTorch Geometric's GCNConv): X' = D^{-1/2} (A + I) D^{-1/2} X W^T + b,
 * in message-passing form.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]`.
 * @param weight - `[F', F]`.
 * @param options - Bias, edge weights, self loops and normalisation.
 * @returns `[N, F']`.
 * @concept graph-neural-networks
 * @reference Kipf and Welling (2017), Semi-supervised classification with graph convolutional networks.
 * @example
 * ```ts
 * import { gcnConv } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gcnConv([[1], [3]], [[0, 1], [1, 0]], [[1]])) // => [[2], [2]]
 * ```
 */
export function gcnConv(x: TensorLike, edgeIndex: TensorLike, weight: TensorLike, options: GcnOptions = {}): Value {
  const xv = fl(x)
  const n = xv.shape[0] ?? 0
  const h = matmul(xv, swapaxes(fl(weight), 0, 1))
  let out: Value
  if (options.normalize === false) {
    const looped =
      options.addSelfLoops === false
        ? { edgeIndex, edgeWeight: options.edgeWeight }
        : addSelfLoops(edgeIndex, n, options.edgeWeight)
    out = messagePassing(h, looped.edgeIndex, looped.edgeWeight === undefined ? {} : { edgeWeight: looped.edgeWeight })
  } else {
    const norm = gcnNorm(edgeIndex, n, {
      ...(options.edgeWeight === undefined ? {} : { edgeWeight: options.edgeWeight }),
      ...(options.addSelfLoops === undefined ? {} : { addSelfLoops: options.addSelfLoops }),
    })
    out = messagePassing(h, norm.edgeIndex, { edgeWeight: norm.weight })
  }
  return options.bias === undefined ? out : add(out, fl(options.bias))
}

/** The weights of {@link gatConv} (PyTorch Geometric's GATConv names). */
export interface GatParams {
  /** The shared projection `[heads C, F]`. */
  readonly weight: TensorLike
  /** Source and target attention vectors `[heads, C]`. */
  readonly attSrc: TensorLike
  readonly attDst: TensorLike
  /** `[heads C]` when concatenating, `[C]` when averaging. */
  readonly bias?: TensorLike
}

/** Options of {@link gatConv}. */
export interface GatOptions {
  readonly heads?: number
  /** Concatenate the heads (default true) or average them. */
  readonly concat?: boolean
  readonly negativeSlope?: number
  readonly addSelfLoops?: boolean
}

/**
 * Graph attention (Velickovic et al. 2018; PyTorch Geometric's GATConv): per head, alpha_ij = softmax over i's
 * incoming edges of LeakyReLU(a_src . W x_j + a_dst . W x_i), and x'_i = sum_j alpha_ij W x_j; heads concatenated or
 * averaged. Existing self loops are replaced by one per node, as PyTorch Geometric does.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]`.
 * @param params - Projection, attention vectors and bias.
 * @param options - Heads (1), concat (true), negative slope (0.2) and self loops (true).
 * @returns The output `[N, heads C]` (or `[N, C]`), the attention weights `[E', heads]` and the edge index used.
 * @concept graph-neural-networks
 * @reference Velickovic et al. (2018), Graph attention networks.
 * @example
 * ```ts
 * import { gatConv } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(gatConv([[1], [3]], [[0, 1], [1, 0]], { weight: [[1]], attSrc: [[0]], attDst: [[0]] }).output) // => [[2], [2]]
 * ```
 */
export function gatConv(
  x: TensorLike,
  edgeIndex: TensorLike,
  params: GatParams,
  options: GatOptions = {},
): { readonly output: Value; readonly attention: Value; readonly edgeIndex: Tensor<'int32'> } {
  const xv = fl(x)
  const n = xv.shape[0] ?? 0
  const heads = options.heads ?? 1
  const w = fl(params.weight)
  const c = (w.shape[0] ?? 0) / heads
  let edges = edgeList(edgeIndex)
  if (options.addSelfLoops !== false) {
    const keep = edges.source.flatMap((s, e) => (s === edges.target[e] ? [] : [e]))
    const loops = Array.from({ length: n }, (_, i) => i)
    edges = {
      source: [...keep.map((e) => edges.source[e] ?? 0), ...loops],
      target: [...keep.map((e) => edges.target[e] ?? 0), ...loops],
    }
  }
  const h = reshape(matmul(xv, swapaxes(w, 0, 1)), [n, heads, c])
  const aSrc = sum(mul(h, fl(params.attSrc)), 2) // [N, H]
  const aDst = sum(mul(h, fl(params.attDst)), 2)
  const src = indexTensor(edges.source)
  const scores = leakyRelu(
    add(take(aSrc, src, 0), take(aDst, indexTensor(edges.target), 0)),
    options.negativeSlope ?? 0.2,
  )
  const alpha = edgeSoftmax(scores, edges.target, n) // [E, H]
  const msg = mul(take(h, src, 0), expandDims(alpha, 2)) // [E, H, C]
  const agg = aggregate(msg, edges.target, n, 'sum')
  let out = options.concat === false ? mean(agg, 1) : reshape(agg, [n, heads * c])
  if (params.bias !== undefined) out = add(out, fl(params.bias))
  return {
    output: out,
    attention: alpha,
    edgeIndex: fromStorage(Int32Array.from([...edges.source, ...edges.target]), [2, edges.source.length], 'int32'),
  }
}

/** The weights of {@link sageConv} (PyTorch Geometric's SAGEConv: `lin_l` on neighbours with a bias, `lin_r` on the root). */
export interface SageParams {
  /** `[F', F]` applied to the aggregated neighbours. */
  readonly weightNeighbour: TensorLike
  readonly biasNeighbour?: TensorLike
  /** `[F', F]` applied to the node itself (omit for no root weight). */
  readonly weightRoot?: TensorLike
}

/**
 * GraphSAGE convolution (Hamilton et al. 2017): x'_i = W_l agg_{j -> i} x_j + b + W_r x_i, with mean, max or sum
 * aggregation and optional L2 normalisation of the output.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]`.
 * @param params - Neighbour and root weights.
 * @param options - Aggregation (default mean) and normalize (default false).
 * @returns `[N, F']`.
 * @concept graph-neural-networks
 * @reference Hamilton, Ying and Leskovec (2017), Inductive representation learning on large graphs.
 * @example
 * ```ts
 * import { sageConv } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sageConv([[1], [3]], [[0], [1]], { weightNeighbour: [[1]], weightRoot: [[1]] })) // => [[1], [4]]
 * ```
 */
export function sageConv(
  x: TensorLike,
  edgeIndex: TensorLike,
  params: SageParams,
  options: { readonly aggregate?: Aggregation; readonly normalize?: boolean } = {},
): Value {
  const xv = fl(x)
  const agg = messagePassing(xv, edgeIndex, { aggregate: options.aggregate ?? 'mean' })
  let out = matmul(agg, swapaxes(fl(params.weightNeighbour), 0, 1))
  if (params.biasNeighbour !== undefined) out = add(out, fl(params.biasNeighbour))
  if (params.weightRoot !== undefined) out = add(out, matmul(xv, swapaxes(fl(params.weightRoot), 0, 1)))
  if (options.normalize === true) {
    const len = pow(add(sum(mul(out, out), 1, true), 1e-24), 0.5)
    out = div(out, len)
  }
  return out
}

/**
 * Graph isomorphism network convolution (Xu et al. 2019): x'_i = MLP((1 + eps) x_i + sum_{j -> i} x_j), as expressive
 * as the Weisfeiler-Lehman test with an injective MLP.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]`.
 * @param mlp - The update network.
 * @param eps - The self weight (default 0; may be a trainable value).
 * @returns `mlp` of the combined features.
 * @concept graph-neural-networks
 * @reference Xu, Hu, Leskovec and Jegelka (2019), How powerful are graph neural networks?
 * @example
 * ```ts
 * import { ginConv } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ginConv([[1], [3]], [[0], [1]], (h) => h)) // => [[1], [4]]
 * ```
 */
export function ginConv(x: TensorLike, edgeIndex: TensorLike, mlp: (h: Value) => Value, eps: TensorLike = 0): Value {
  const xv = fl(x)
  return mlp(add(mul(xv, add(fl(eps), 1)), messagePassing(xv, edgeIndex)))
}

/**
 * Pool node features into one vector per graph of a batch (PyTorch Geometric's global_add/mean/max_pool): sum, mean
 * or max over the nodes with the same batch index.
 *
 * @param x - Node features `[N, F]`.
 * @param batch - The graph of each node `[N]` (default: one graph).
 * @param reduce - `sum`, `mean` (default) or `max`.
 * @param numGraphs - The number of graphs (default: the largest index plus one).
 * @returns `[numGraphs, F]`.
 * @example
 * ```ts
 * import { globalPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(globalPool([[1], [3], [5]], [0, 0, 1], 'sum')) // => [[4], [5]]
 * ```
 */
export function globalPool(x: TensorLike, batch?: TensorLike, reduce: Aggregation = 'mean', numGraphs?: number): Value {
  const xv = fl(x)
  const n = xv.shape[0] ?? 0
  const b = batch === undefined ? new Array<number>(n).fill(0) : toFlat(asTensor(batch, 'int32'))
  let g = numGraphs ?? 0
  if (numGraphs === undefined) for (const v of b) g = Math.max(g, v + 1)
  return aggregate(xv, b, g, reduce)
}

/** Result of {@link topKPool}. */
export interface TopKPoolResult {
  /** The kept nodes' features, gated by their scores, `[N', F]`. */
  readonly x: Value
  /** The edges among the kept nodes, renumbered, `[2, E']`. */
  readonly edgeIndex: Tensor<'int32'>
  /** The graph of each kept node. */
  readonly batch: readonly number[]
  /** The original index of each kept node. */
  readonly perm: readonly number[]
  /** All nodes' scores tanh(x p / |p|) `[N]`. */
  readonly score: Value
}

/**
 * Top-k graph pooling (Gao and Ji 2019; PyTorch Geometric's TopKPooling): score each node by projecting its features
 * on a learnt vector, keep the ceil(ratio n) best nodes of each graph, gate their features by tanh of the score (so
 * the projection gets gradients) and keep the edges between kept nodes.
 *
 * @param x - Node features `[N, F]`.
 * @param edgeIndex - `[2, E]`.
 * @param weight - The projection `[F]`.
 * @param options - `ratio` (default 0.5) and the `batch` vector (default one graph).
 * @returns The pooled graph.
 * @reference Gao and Ji (2019), Graph U-Nets.
 * @example
 * ```ts
 * import { topKPool } from 'mdml-engine/nn'
 * topKPool([[1], [3], [2], [0]], [[0, 1, 2], [1, 2, 3]], [1]).perm // => [1, 2]
 * ```
 */
export function topKPool(
  x: TensorLike,
  edgeIndex: TensorLike,
  weight: TensorLike,
  options: { readonly ratio?: number; readonly batch?: readonly number[] } = {},
): TopKPoolResult {
  const xv = fl(x)
  const w = reshape(fl(weight), [-1, 1])
  const n = xv.shape[0] ?? 0
  const len = sqrt(sum(mul(w, w)))
  const score = reshape(tanh(div(matmul(xv, w), len)), [n])
  const s = toFlat(primalOf(score))
  const batch = options.batch ?? new Array<number>(n).fill(0)
  const ratio = options.ratio ?? 0.5
  const graphs = new Map<number, number[]>()
  for (let i = 0; i < n; i++) graphs.set(batch[i] ?? 0, [...(graphs.get(batch[i] ?? 0) ?? []), i])
  const perm: number[] = []
  for (const g of [...graphs.keys()].sort((a, b) => a - b)) {
    const nodes = [...(graphs.get(g) ?? [])].sort((a, b) => (s[b] ?? 0) - (s[a] ?? 0) || a - b)
    perm.push(...nodes.slice(0, Math.ceil(ratio * nodes.length)))
  }
  const index = new Map(perm.map((v, i) => [v, i]))
  const { source, target } = edgeList(edgeIndex)
  const kept: [number, number][] = []
  for (const [e, a] of source.entries()) {
    const ia = index.get(a)
    const ib = index.get(target[e] ?? -1)
    if (ia !== undefined && ib !== undefined) kept.push([ia, ib])
  }
  const idx = indexTensor(perm)
  return {
    x: mul(take(xv, idx, 0), expandDims(take(score, idx, 0), 1)),
    edgeIndex: fromStorage(
      Int32Array.from([...kept.map((k) => k[0]), ...kept.map((k) => k[1])]),
      [2, kept.length],
      'int32',
    ),
    batch: perm.map((v) => batch[v] ?? 0),
    perm,
    score,
  }
}

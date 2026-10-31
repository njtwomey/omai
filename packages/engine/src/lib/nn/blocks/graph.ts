/**
 * Graph blocks beyond the basic convolutions: the relational graph convolution (`rgcnConv`, `RGCNConv`, one weight per
 * edge type, optionally by basis decomposition), the E(n)-equivariant graph layer and network of Satorras et al.
 * (`EGNNLayer`, `EGNN`: features updated from distances, coordinates from relative vectors, so rotations and
 * translations of the coordinates carry through), and the graph transformer (`GraphTransformerLayer`,
 * `GraphTransformer`: multi-head attention over the edges with Laplacian or random-walk positional encodings from
 * `graph`).
 */

import { MdmlError } from '../../base/index.ts'
import { bernoulli, foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  einsum,
  fromStorage,
  matmul,
  mul,
  reshape,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  tanh,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { gelu } from '../functional/activations.ts'
import { addSelfLoops, aggregate, edgeList, edgeSoftmax } from '../functional/graph.ts'
import { xavierUniform } from '../init.ts'
import { SiLU } from '../module/activations.ts'
import { Composite, Sequential } from '../module/containers.ts'
import { Linear } from '../module/layers.ts'
import { type ApplyContext, initKey, Module, type Params } from '../module/module.ts'
import { LayerNorm } from '../module/normalisation.ts'

const index = (xs: readonly number[]): Tensor<'int32'> => fromStorage(Int32Array.from(xs), [xs.length], 'int32')

/** Options of {@link rgcnConv}. */
export interface RgcnOptions {
  /**
   * The relation weights `[R, in, out]`, or with `comp` the shared bases `[B, in, out]` from which relation `r` takes
   * `sum_b comp[r, b] basis[b]`.
   */
  readonly weight: TensorLike
  /** Basis coefficients `[R, B]`. */
  readonly comp?: TensorLike
  /** The self-connection weight `[in, out]` (PyTorch Geometric's `root`). */
  readonly root?: TensorLike
  readonly bias?: TensorLike
  /** How messages of one relation are normalised at a node: `mean` (default, `1 / c_{i,r}`) or `sum`. */
  readonly aggregate?: 'mean' | 'sum'
}

/**
 * The relational graph convolution (Schlichtkrull et al. 2018): `h_i' = root h_i + sum_r sum_{j in N_r(i)} W_r h_j /
 * c_{i,r} + bias`, one weight per edge type, `c_{i,r}` the number of type-`r` neighbours of `i` (mean) or 1 (sum).
 * Differentiable in the features and every weight.
 *
 * @param x - Node features `[N, in]`.
 * @param edgeIndex - `[2, E]` (source, target).
 * @param edgeType - The relation of each edge, integers in `[0, R)`.
 * @param options - The weights and the normalisation.
 * @returns `[N, out]`.
 * @reference Schlichtkrull, Kipf, Bloem, van den Berg, Titov and Welling (2018), Modeling relational data with graph convolutional networks.
 * @example
 * ```ts
 * import { rgcnConv } from 'mdml-engine/nn/blocks'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(rgcnConv([[1], [2]], [[0], [1]], [1], { weight: [[[3]], [[5]]] })) // => [[0], [5]]
 * ```
 */
export function rgcnConv(
  x: TensorLike,
  edgeIndex: TensorLike,
  edgeType: readonly number[],
  options: RgcnOptions,
): Value {
  const h = asTensor(x)
  const n = h.shape[0] ?? 0
  const { source, target } = edgeList(edgeIndex)
  if (edgeType.length !== source.length) throw new MdmlError('shape', 'rgcnConv: one edge type per edge')
  const raw = asTensor(options.weight)
  const w = options.comp === undefined ? raw : einsum('rb,bio->rio', options.comp, raw)
  const [r, , out] = w.shape as [number, number, number]
  if (edgeType.some((t) => !Number.isInteger(t) || t < 0 || t >= r))
    throw new MdmlError('value', `rgcnConv: edge types must be in [0, ${r})`)
  // Every node through every relation's weight, then one gather: row (type * N + source).
  const transformed = reshape(einsum('ni,rio->rno', h, w), [r * n, out])
  const messages = take(transformed, index(source.map((s, e) => (edgeType[e] as number) * n + s)), 0)
  let weights: number[] = source.map(() => 1)
  if ((options.aggregate ?? 'mean') === 'mean') {
    const counts = new Map<number, number>()
    for (const [e, t] of target.entries()) {
      const key = t * r + (edgeType[e] as number)
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    weights = target.map((t, e) => 1 / (counts.get(t * r + (edgeType[e] as number)) as number))
  }
  const column = reshape(fromStorage(Float64Array.from(weights), [weights.length], 'float64'), [-1, 1])
  let y = aggregate(mul(messages, column), target, n, 'sum')
  if (options.root !== undefined) y = add(y, matmul(h, options.root))
  if (options.bias !== undefined) y = add(y, options.bias)
  return y
}

/** The input of {@link RGCNConv}: features, the edge index and the relation of each edge. */
export interface RelationalInput {
  readonly x: TensorLike
  readonly edgeIndex: TensorLike
  readonly edgeType: readonly number[]
}

/** Options of {@link RGCNConv}. */
export interface RgcnConvOptions {
  /** Decompose the relation weights over this many shared bases (default: one weight per relation). */
  readonly numBases?: number
  readonly aggregate?: 'mean' | 'sum'
  readonly rootWeight?: boolean
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * The relational graph convolution layer (PyTorch Geometric's RGCNConv parameter names): `weight` `[R, in, out]` (or
 * `[B, in, out]` bases with `comp` `[R, B]`), `root` `[in, out]` and `bias`. The basis decomposition cuts the
 * parameters from `R in out` to `B in out + R B`.
 *
 * @example
 * ```ts
 * import { RGCNConv } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new RGCNConv(3, 2, 4, { key: key(0) }).forward({ x: zeros([5, 3]), edgeIndex: [[0, 1], [1, 2]], edgeType: [0, 3] }).shape // => [5, 2]
 * ```
 */
export class RGCNConv extends Module<RelationalInput, Value> {
  readonly kind = 'RGCNConv'
  readonly inChannels: number
  readonly outChannels: number
  readonly numRelations: number
  readonly numBases: number | undefined
  readonly aggregate: 'mean' | 'sum'
  readonly rootWeight: boolean
  readonly bias: boolean

  constructor(inChannels: number, outChannels: number, numRelations: number, options: RgcnConvOptions = {}) {
    super()
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.numRelations = numRelations
    this.numBases = options.numBases
    this.aggregate = options.aggregate ?? 'mean'
    this.rootWeight = options.rootWeight ?? true
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('RGCNConv', options.key))
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  override initParams(key: Key): Params {
    const count = this.numBases ?? this.numRelations
    const glorot = (k: number, shape: number[]): Tensor => xavierUniform(foldIn(key, k), shape)
    return {
      weight: glorot(0, [count, this.inChannels, this.outChannels]),
      ...(this.numBases === undefined ? {} : { comp: glorot(1, [this.numRelations, this.numBases]) }),
      ...(this.rootWeight ? { root: glorot(2, [this.inChannels, this.outChannels]) } : {}),
      ...(this.bias ? { bias: zeros([this.outChannels]) } : {}),
    }
  }

  apply(params: Params, input: RelationalInput): Value {
    return rgcnConv(input.x, input.edgeIndex, input.edgeType, {
      weight: params['weight'] as Value,
      aggregate: this.aggregate,
      ...(params['comp'] === undefined ? {} : { comp: params['comp'] as Value }),
      ...(params['root'] === undefined ? {} : { root: params['root'] as Value }),
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
    })
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, num_relations=${this.numRelations}`
  }
}

/** The input of {@link EGNNLayer}: features, coordinates, the edge index and optional edge attributes. */
export interface EgnnInput {
  readonly h: TensorLike
  readonly x: TensorLike
  readonly edgeIndex: TensorLike
  /** Edge attributes `[E, A]`. */
  readonly edgeAttr?: TensorLike
}

/** The output of {@link EGNNLayer}: the updated features and coordinates. */
export interface EgnnOutput {
  readonly h: Value
  readonly x: Value
}

/** Options of {@link EGNNLayer}. */
export interface EgnnOptions {
  /** The width of the edge messages (default: the feature width). */
  readonly hidden?: number
  /** The width of the edge attributes (default 0). */
  readonly edgeDim?: number
  /** Combine the coordinate updates of a node's edges by `mean` (default) or `sum`. */
  readonly coordsAggregate?: 'mean' | 'sum'
  /** Squash the coordinate weight with tanh (more stable, default false). */
  readonly tanh?: boolean
  /** Add the input features to the update (default true). */
  readonly residual?: boolean
  /** Update the coordinates (default true); without it the layer is invariant in `h` only. */
  readonly updateCoords?: boolean
  readonly key?: Key
}

const silu = (): Module => new SiLU()

/**
 * The E(n)-equivariant graph layer (Satorras, Hoogeboom and Welling 2021). For each edge `j -> i` a message
 * `m_ij = phi_e(h_i, h_j, |x_i - x_j|^2, a_ij)`; the coordinates move along the relative vectors,
 * `x_i' = x_i + mean_j (x_i - x_j) phi_x(m_ij)`, and the features update from the summed messages,
 * `h_i' = h_i + phi_h(h_i, sum_j m_ij)`. Features depend on the coordinates only through distances, so under any
 * rotation, reflection or translation of `x` the output `h` is unchanged and the output `x` is transformed the same
 * way.
 *
 * @reference Satorras, Hoogeboom and Welling (2021), E(n) equivariant graph neural networks.
 * @example
 * ```ts
 * import { EGNNLayer } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * const layer = new EGNNLayer(4, { key: key(0) })
 * layer.forward({ h: zeros([3, 4]), x: zeros([3, 2]), edgeIndex: [[0, 1, 2], [1, 2, 0]] }).x.shape // => [3, 2]
 * ```
 */
export class EGNNLayer extends Composite<EgnnInput, EgnnOutput> {
  readonly kind = 'EGNNLayer'
  readonly features: number
  readonly coordsAggregate: 'mean' | 'sum'
  readonly squash: boolean
  readonly residual: boolean
  readonly updateCoords: boolean

  constructor(features: number, options: EgnnOptions = {}) {
    const key = initKey('EGNNLayer', options.key)
    const hidden = options.hidden ?? features
    const edgeIn = 2 * features + 1 + (options.edgeDim ?? 0)
    super({
      edge: new Sequential(
        new Linear(edgeIn, hidden, { key: foldIn(key, 0) }),
        silu(),
        new Linear(hidden, hidden, { key: foldIn(key, 1) }),
        silu(),
      ),
      node: new Sequential(
        new Linear(features + hidden, hidden, { key: foldIn(key, 2) }),
        silu(),
        new Linear(hidden, features, { key: foldIn(key, 3) }),
      ),
      ...(options.updateCoords === false
        ? {}
        : {
            coord: new Sequential(
              new Linear(hidden, hidden, { key: foldIn(key, 4) }),
              silu(),
              new Linear(hidden, 1, { bias: false, key: foldIn(key, 5) }),
            ),
          }),
    })
    this.features = features
    this.coordsAggregate = options.coordsAggregate ?? 'mean'
    this.squash = options.tanh ?? false
    this.residual = options.residual ?? true
    this.updateCoords = options.updateCoords ?? true
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: EgnnInput, ctx: ApplyContext = {}): EgnnOutput {
    const h = asTensor(input.h)
    const x = asTensor(input.x)
    const n = h.shape[0] ?? 0
    const { source, target } = edgeList(input.edgeIndex)
    const p = this.parts(params, ctx)
    const hi = take(h, index(target), 0)
    const hj = take(h, index(source), 0)
    const relative = sub(take(x, index(target), 0), take(x, index(source), 0))
    const distance = sum(mul(relative, relative), -1, true)
    const parts: Value[] = [hi, hj, distance]
    if (input.edgeAttr !== undefined) parts.push(asTensor(input.edgeAttr))
    const messages = p.run<Value, Value>('edge', concat(parts, -1))
    let coords: Value = x
    if (this.updateCoords) {
      let weight = p.run<Value, Value>('coord', messages)
      if (this.squash) weight = tanh(weight)
      coords = add(x, aggregate(mul(relative, weight), target, n, this.coordsAggregate))
    }
    const pooled = aggregate(messages, target, n, 'sum')
    const update = p.run<Value, Value>('node', concat([h, pooled], -1))
    p.done()
    return { h: this.residual ? add(h, update) : update, x: coords }
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of {@link EGNN}. */
export interface EgnnNetworkOptions extends Omit<EgnnOptions, 'key'> {
  readonly layers?: number
  readonly key?: Key
}

/**
 * A stack of E(n)-equivariant layers: an input linear embedding of the features (invariant), then `layers`
 * {@link EGNNLayer}s over the same edges, returning the final features and coordinates.
 *
 * @example
 * ```ts
 * import { EGNN } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new EGNN(2, 8, { layers: 2, key: key(0) }).forward({ h: zeros([3, 2]), x: zeros([3, 3]), edgeIndex: [[0, 1], [1, 0]] }).h.shape // => [3, 8]
 * ```
 */
export class EGNN extends Composite<EgnnInput, EgnnOutput> {
  readonly kind = 'EGNN'
  readonly layerCount: number
  readonly width: number

  constructor(inFeatures: number, width: number, options: EgnnNetworkOptions = {}) {
    const key = initKey('EGNN', options.key)
    const layers = options.layers ?? 2
    const { layers: _l, key: _k, ...rest } = options
    const parts: Record<string, Module<never, unknown>> = {
      embed: new Linear(inFeatures, width, { key: foldIn(key, 0) }),
    }
    for (let i = 0; i < layers; i++)
      parts[`layer${i}`] = new EGNNLayer(width, { ...rest, key: foldIn(key, i + 1) }) as never
    super(parts as never)
    this.layerCount = layers
    this.width = width
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: EgnnInput, ctx: ApplyContext = {}): EgnnOutput {
    const p = this.parts(params, ctx)
    let state: EgnnOutput = { h: p.run<Value, Value>('embed', asTensor(input.h)), x: asTensor(input.x) }
    for (let i = 0; i < this.layerCount; i++) {
      state = p.run<EgnnInput, EgnnOutput>(`layer${i}`, {
        h: state.h,
        x: state.x,
        edgeIndex: input.edgeIndex,
        ...(input.edgeAttr === undefined ? {} : { edgeAttr: input.edgeAttr }),
      })
    }
    p.done()
    return state
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.width]
  }
}

/** The input of the graph transformer: features, the edge index (absent: every pair of nodes attends) and encodings. */
export interface GraphTransformerInput {
  readonly x: TensorLike
  readonly edgeIndex?: TensorLike
  /** Positional encodings `[N, p]` (for example `laplacianPositionalEncoding` or `randomWalkPositionalEncoding`). */
  readonly pe?: TensorLike
}

/** Options of {@link GraphTransformerLayer}. */
export interface GraphTransformerLayerOptions {
  readonly heads?: number
  /** The feed-forward width (default twice the model width). */
  readonly feedForward?: number
  /** Add a self loop to every node (default true). */
  readonly selfLoops?: boolean
  readonly key?: Key
}

const allPairs = (n: number): Tensor<'int32'> => {
  const src: number[] = []
  const dst: number[] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      src.push(j)
      dst.push(i)
    }
  return fromStorage(Int32Array.from([...src, ...dst]), [2, src.length], 'int32')
}

/**
 * A graph transformer layer (Dwivedi and Bresson 2021): multi-head attention in which each node attends to its
 * neighbours (the edge softmax over incoming edges), a residual connection and layer norm, then a feed-forward
 * network with its own residual and norm. With no edge index every pair of nodes attends, the dense transformer on
 * the graph's nodes.
 *
 * @reference Dwivedi and Bresson (2021), A generalization of transformer networks to graphs.
 * @example
 * ```ts
 * import { GraphTransformerLayer } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new GraphTransformerLayer(8, { heads: 2, key: key(0) }).forward({ x: zeros([4, 8]), edgeIndex: [[0, 1], [1, 2]] }).shape // => [4, 8]
 * ```
 */
export class GraphTransformerLayer extends Composite<GraphTransformerInput, Value> {
  readonly kind = 'GraphTransformerLayer'
  readonly width: number
  readonly heads: number
  readonly selfLoops: boolean

  constructor(width: number, options: GraphTransformerLayerOptions = {}) {
    const heads = options.heads ?? 4
    if (width % heads !== 0)
      throw new MdmlError('value', `GraphTransformerLayer: ${width} is not divisible by ${heads} heads`)
    const key = initKey('GraphTransformerLayer', options.key)
    const ff = options.feedForward ?? 2 * width
    super({
      query: new Linear(width, width, { key: foldIn(key, 0) }),
      key: new Linear(width, width, { key: foldIn(key, 1) }),
      value: new Linear(width, width, { key: foldIn(key, 2) }),
      out: new Linear(width, width, { key: foldIn(key, 3) }),
      norm1: new LayerNorm(width),
      ff1: new Linear(width, ff, { key: foldIn(key, 4) }),
      ff2: new Linear(ff, width, { key: foldIn(key, 5) }),
      norm2: new LayerNorm(width),
    })
    this.width = width
    this.heads = heads
    this.selfLoops = options.selfLoops ?? true
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: GraphTransformerInput, ctx: ApplyContext = {}): Value {
    const h = asTensor(input.x)
    const n = h.shape[0] ?? 0
    const d = this.width / this.heads
    let edgeIndex: TensorLike = input.edgeIndex === undefined ? allPairs(n) : input.edgeIndex
    if (this.selfLoops && input.edgeIndex !== undefined) edgeIndex = addSelfLoops(edgeIndex, n).edgeIndex
    const { source, target } = edgeList(edgeIndex)
    const p = this.parts(params, ctx)
    const heads = (v: Value): Value => reshape(v, [n, this.heads, d])
    const q = heads(p.run<Value, Value>('query', h))
    const k = heads(p.run<Value, Value>('key', h))
    const v = heads(p.run<Value, Value>('value', h))
    const scores = div(sum(mul(take(q, index(target), 0), take(k, index(source), 0)), -1), Math.sqrt(d))
    const alpha = edgeSoftmax(scores, target, n)
    const messages = mul(take(v, index(source), 0), reshape(alpha, [-1, this.heads, 1]))
    const attended = reshape(aggregate(messages, target, n, 'sum'), [n, this.width])
    const a = p.run<Value, Value>('norm1', add(h, p.run<Value, Value>('out', attended)))
    const f = p.run<Value, Value>('ff2', gelu(p.run<Value, Value>('ff1', a)))
    const y = p.run<Value, Value>('norm2', add(a, f))
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of {@link GraphTransformer}. */
export interface GraphTransformerOptions {
  readonly layers?: number
  readonly heads?: number
  /** The width of the positional encodings, to embed them (default 0: none). */
  readonly peDim?: number
  /** Flip the sign of each encoding column at random in training, as Laplacian eigenvectors are only defined up to sign. */
  readonly randomSignFlip?: boolean
  readonly selfLoops?: boolean
  readonly key?: Key
}

/**
 * A graph transformer: linear embeddings of the node features and of the positional encodings (added), then a stack
 * of {@link GraphTransformerLayer}s. The encodings come from `graph` (`laplacianPositionalEncoding`, the first
 * eigenvectors of the normalised Laplacian, or `randomWalkPositionalEncoding`). Laplacian eigenvectors are defined up
 * to sign, so in training mode with a key the sign of each column is flipped at random.
 *
 * @example
 * ```ts
 * import { GraphTransformer } from 'mdml-engine/nn/blocks'
 * import { cycleGraph, laplacianPositionalEncoding } from 'mdml-engine/graph'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * const model = new GraphTransformer(3, 8, { peDim: 2, heads: 2, key: key(0) })
 * model.forward({ x: zeros([6, 3]), edgeIndex: [[0, 1], [1, 2]], pe: laplacianPositionalEncoding(cycleGraph(6), 2) }).shape // => [6, 8]
 * ```
 */
export class GraphTransformer extends Composite<GraphTransformerInput, Value> {
  readonly kind = 'GraphTransformer'
  readonly width: number
  readonly layerCount: number
  readonly randomSignFlip: boolean

  constructor(inFeatures: number, width: number, options: GraphTransformerOptions = {}) {
    const key = initKey('GraphTransformer', options.key)
    const layers = options.layers ?? 2
    const parts: Record<string, Module<never, unknown>> = {
      embed: new Linear(inFeatures, width, { key: foldIn(key, 0) }),
    }
    if ((options.peDim ?? 0) > 0) parts['peEmbed'] = new Linear(options.peDim as number, width, { key: foldIn(key, 1) })
    for (let i = 0; i < layers; i++)
      parts[`layer${i}`] = new GraphTransformerLayer(width, {
        ...(options.heads === undefined ? {} : { heads: options.heads }),
        ...(options.selfLoops === undefined ? {} : { selfLoops: options.selfLoops }),
        key: foldIn(key, i + 2),
      }) as never
    super(parts as never)
    this.width = width
    this.layerCount = layers
    this.randomSignFlip = options.randomSignFlip ?? true
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: GraphTransformerInput, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    let h = p.run<Value, Value>('embed', asTensor(input.x))
    if (input.pe !== undefined && this.has('peEmbed')) {
      let pe = asTensor(input.pe)
      if (this.randomSignFlip && (ctx.train ?? true) && ctx.key !== undefined) {
        const flips = toFlat(bernoulli(foldIn(ctx.key, 97), 0.5, [pe.shape[1] ?? 1])).map((b) => (b ? -1 : 1))
        pe = mul(pe, fromStorage(Float64Array.from(flips), [1, flips.length], 'float64'))
      }
      h = add(h, p.run<Value, Value>('peEmbed', pe))
    }
    for (let i = 0; i < this.layerCount; i++) {
      h = p.run<GraphTransformerInput, Value>(`layer${i}`, {
        x: h,
        ...(input.edgeIndex === undefined ? {} : { edgeIndex: input.edgeIndex }),
      })
    }
    p.done()
    return h
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.width]
  }
}

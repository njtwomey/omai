/**
 * Graph convolution layers as modules, mirroring PyTorch Geometric's parameter names: `GCNConv`, the graph attention
 * layer `GATConv`, GraphSAGE's `SAGEConv` and the graph isomorphism layer `GINConv`, each taking node features with an
 * edge index.
 */

import { foldIn, type Key } from '../../random/index.ts'
import { type TensorLike, tensor, type Value, zeros } from '../../tensor/index.ts'
import { type Aggregation, gatConv, gcnConv, ginConv, sageConv } from '../functional/graph.ts'
import { torchDefaultUniform, xavierUniform } from '../init.ts'
import { type ApplyContext, childScope, initKey, Module, type Params, tap } from './module.ts'

/** The input of a graph convolution: node features, the edge index `[2, E]` (source, target) and edge weights. */
export interface GraphInput {
  readonly x: TensorLike
  readonly edgeIndex: TensorLike
  readonly edgeWeight?: TensorLike
}

/** A graph convolution module: structured input, `[N, out]` output. */
abstract class GraphConv extends Module<GraphInput, Value> {
  readonly inChannels: number
  readonly outChannels: number

  protected constructor(inChannels: number, outChannels: number, params: Params = {}) {
    super(params)
    this.inChannels = inChannels
    this.outChannels = outChannels
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[0] ?? 1, this.outChannels]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}`
  }
}

/** Options of {@link GCNConv}. */
export interface GCNConvOptions {
  readonly bias?: boolean
  readonly addSelfLoops?: boolean
  readonly normalize?: boolean
  readonly key?: Key
}

/**
 * Graph convolution (PyTorch Geometric's GCNConv): `lin.weight` Glorot-initialised, `bias` zeros; symmetric
 * normalisation with self loops, optional edge weights.
 *
 * @concept graph-neural-networks
 * @example
 * ```ts
 * import { GCNConv } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import { zeros } from 'whitebox-engine/tensor'
 * new GCNConv(3, 2, { key: key(0) }).forward({ x: zeros([4, 3]), edgeIndex: [[0, 1], [1, 2]] }).shape // => [4, 2]
 * ```
 */
export class GCNConv extends GraphConv {
  readonly kind = 'GCNConv'
  readonly bias: boolean
  readonly addSelfLoops: boolean
  readonly normalize: boolean

  constructor(inChannels: number, outChannels: number, options: GCNConvOptions = {}) {
    super(inChannels, outChannels)
    this.bias = options.bias ?? true
    this.addSelfLoops = options.addSelfLoops ?? true
    this.normalize = options.normalize ?? true
    this.params = this.initParams(initKey('GCNConv', options.key))
  }

  override initParams(key: Key): Params {
    return {
      lin: { weight: xavierUniform(foldIn(key, 0), [this.outChannels, this.inChannels]) },
      ...(this.bias ? { bias: zeros([this.outChannels]) } : {}),
    }
  }

  apply(params: Params, input: GraphInput): Value {
    return gcnConv(input.x, input.edgeIndex, (params['lin'] as Params)['weight'] as Value, {
      addSelfLoops: this.addSelfLoops,
      normalize: this.normalize,
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      ...(input.edgeWeight === undefined ? {} : { edgeWeight: input.edgeWeight }),
    })
  }
}

/** Options of {@link GATConv}. */
export interface GATConvOptions {
  readonly heads?: number
  readonly concat?: boolean
  readonly negativeSlope?: number
  readonly addSelfLoops?: boolean
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * Graph attention (PyTorch Geometric's GATConv): `lin.weight` `[heads out, in]`, `att_src` and `att_dst`
 * `[1, heads, out]` (Glorot), `bias`; heads concatenated or averaged. The attention weights are tapped as `attention`.
 *
 * @concept graph-neural-networks
 * @example
 * ```ts
 * import { GATConv } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * new GATConv(3, 2, { heads: 4, key: key(0) }).outputShape([5, 3]) // => [5, 8]
 * ```
 */
export class GATConv extends GraphConv {
  readonly kind = 'GATConv'
  readonly heads: number
  readonly concat: boolean
  readonly negativeSlope: number
  readonly addSelfLoops: boolean
  readonly bias: boolean

  constructor(inChannels: number, outChannels: number, options: GATConvOptions = {}) {
    super(inChannels, outChannels)
    this.heads = options.heads ?? 1
    this.concat = options.concat ?? true
    this.negativeSlope = options.negativeSlope ?? 0.2
    this.addSelfLoops = options.addSelfLoops ?? true
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('GATConv', options.key))
  }

  override initParams(key: Key): Params {
    const h = this.heads
    const c = this.outChannels
    return {
      lin: { weight: xavierUniform(foldIn(key, 0), [h * c, this.inChannels]) },
      att_src: xavierUniform(foldIn(key, 1), [1, h, c]),
      att_dst: xavierUniform(foldIn(key, 2), [1, h, c]),
      ...(this.bias ? { bias: zeros([this.concat ? h * c : c]) } : {}),
    }
  }

  apply(params: Params, input: GraphInput, ctx: ApplyContext = {}): Value {
    const r = gatConv(
      input.x,
      input.edgeIndex,
      {
        weight: (params['lin'] as Params)['weight'] as Value,
        attSrc: params['att_src'] as Value,
        attDst: params['att_dst'] as Value,
        ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      },
      { heads: this.heads, concat: this.concat, negativeSlope: this.negativeSlope, addSelfLoops: this.addSelfLoops },
    )
    tap(ctx, 'attention', r.attention)
    return r.output
  }

  override outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[0] ?? 1, this.concat ? this.heads * this.outChannels : this.outChannels]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, heads=${this.heads}`
  }
}

/** Options of {@link SAGEConv}. */
export interface SAGEConvOptions {
  readonly aggregate?: Aggregation
  readonly normalize?: boolean
  /** Add the root weight `lin_r` (default true). */
  readonly rootWeight?: boolean
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * GraphSAGE convolution (PyTorch Geometric's SAGEConv): `lin_l` (with bias) on the aggregated neighbours and `lin_r`
 * (no bias) on the node itself; mean, max or sum aggregation, optional L2 normalisation.
 *
 * @concept graph-neural-networks
 * @example
 * ```ts
 * import { SAGEConv } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * Object.keys(new SAGEConv(3, 2, { key: key(0) }).stateDict()) // => ['lin_l.bias', 'lin_l.weight', 'lin_r.weight']
 * ```
 */
export class SAGEConv extends GraphConv {
  readonly kind = 'SAGEConv'
  readonly aggregate: Aggregation
  readonly normalize: boolean
  readonly rootWeight: boolean
  readonly bias: boolean

  constructor(inChannels: number, outChannels: number, options: SAGEConvOptions = {}) {
    super(inChannels, outChannels)
    this.aggregate = options.aggregate ?? 'mean'
    this.normalize = options.normalize ?? false
    this.rootWeight = options.rootWeight ?? true
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('SAGEConv', options.key))
  }

  override initParams(key: Key): Params {
    const shape = [this.outChannels, this.inChannels]
    return {
      lin_l: {
        weight: torchDefaultUniform(foldIn(key, 0), shape, this.inChannels),
        ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], this.inChannels) } : {}),
      },
      ...(this.rootWeight ? { lin_r: { weight: torchDefaultUniform(foldIn(key, 2), shape, this.inChannels) } } : {}),
    }
  }

  apply(params: Params, input: GraphInput): Value {
    const l = params['lin_l'] as Params
    const r = params['lin_r'] as Params | undefined
    return sageConv(
      input.x,
      input.edgeIndex,
      {
        weightNeighbour: l['weight'] as Value,
        ...(l['bias'] === undefined ? {} : { biasNeighbour: l['bias'] as Value }),
        ...(r === undefined ? {} : { weightRoot: r['weight'] as Value }),
      },
      { aggregate: this.aggregate, normalize: this.normalize },
    )
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, aggr='${this.aggregate}'`
  }
}

/**
 * Graph isomorphism convolution (PyTorch Geometric's GINConv): an update network `nn` applied to
 * (1 + eps) x_i + sum_j x_j, with eps fixed or trainable (`trainEps`).
 *
 * @concept graph-neural-networks
 * @example
 * ```ts
 * import { GINConv, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * Object.keys(new GINConv(new Linear(3, 2, { key: key(0) }), { trainEps: true }).stateDict()) // => ['eps', 'nn.bias', 'nn.weight']
 * ```
 */
export class GINConv extends Module<GraphInput, Value> {
  readonly kind = 'GINConv'
  readonly mlp: Module
  readonly eps: number
  readonly trainEps: boolean

  constructor(mlp: Module, options: { readonly eps?: number; readonly trainEps?: boolean } = {}) {
    super()
    this.mlp = mlp
    this.eps = options.eps ?? 0
    this.trainEps = options.trainEps ?? false
    this.params = { nn: mlp.params, ...(this.trainEps ? { eps: tensor(this.eps) } : {}) }
    this.buffers = { nn: mlp.buffers }
  }

  override children(): readonly (readonly [string, Module])[] {
    return [['nn', this.mlp]]
  }

  override initParams(key: Key): Params {
    return { nn: this.mlp.initParams(foldIn(key, 0)), ...(this.trainEps ? { eps: tensor(this.eps) } : {}) }
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: GraphInput, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const inner = scope.child('nn', 0)
    const y = ginConv(
      input.x,
      input.edgeIndex,
      (h) => this.mlp.apply((params['nn'] ?? {}) as Params, h, inner),
      (params['eps'] as Value | undefined) ?? this.eps,
    )
    scope.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.mlp.outputShape(inputShape)
  }

  override extraRepr(): string {
    return `eps=${this.eps}, train_eps=${this.trainEps}`
  }
}

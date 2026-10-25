/**
 * The transformer layers of torch.nn: encoder and decoder layers (self-attention, cross-attention and a feed-forward
 * block, each with a residual connection and layer normalisation, post-norm or pre-norm), encoder and decoder stacks,
 * the full encoder-decoder `Transformer` with PyTorch's initialisation, and `squareSubsequentMask` for causal decoding.
 */

import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  isValue,
  type Tensor,
  type TensorLike,
  treeMap,
  type Value,
} from '../../tensor/index.ts'
import { gelu, relu } from '../functional/activations.ts'
import { xavierUniform } from '../init.ts'
import { MultiheadAttention } from './attention.ts'
import { ModuleList } from './containers.ts'
import { Linear } from './layers.ts'
import { type ApplyContext, assemble, type ChildScope, childScope, initKey, Module, type Params } from './module.ts'
import { LayerNorm } from './normalisation.ts'
import { Dropout } from './regularisation.ts'

/** The feed-forward activation of a transformer layer. */
export type TransformerActivation = 'relu' | 'gelu' | ((x: Value) => Value)

/** Options shared by the transformer layers. */
export interface TransformerLayerOptions {
  /** Hidden width of the feed-forward block (default 2048). */
  readonly dimFeedforward?: number
  /** Dropout after attention, inside and after the feed-forward block (default 0.1). */
  readonly dropout?: number
  readonly activation?: TransformerActivation
  readonly layerNormEps?: number
  /** `[B, L, E]` inputs (default false: `[L, B, E]`). */
  readonly batchFirst?: boolean
  /** Pre-norm (normalise before each block) instead of post-norm (default false). */
  readonly normFirst?: boolean
  readonly bias?: boolean
  readonly key?: Key
}

/** The input of {@link TransformerEncoderLayer} and {@link TransformerEncoder}. */
export interface EncoderInput {
  readonly src: TensorLike
  /** `[L, L]` mask: boolean true = not allowed, or additive (PyTorch's src_mask). */
  readonly mask?: TensorLike
  /** `[B, L]`, true for padding. */
  readonly keyPaddingMask?: TensorLike
  readonly isCausal?: boolean
}

/** The input of {@link TransformerDecoderLayer} and {@link TransformerDecoder}. */
export interface DecoderInput {
  readonly tgt: TensorLike
  readonly memory: TensorLike
  readonly tgtMask?: TensorLike
  readonly memoryMask?: TensorLike
  readonly tgtKeyPaddingMask?: TensorLike
  readonly memoryKeyPaddingMask?: TensorLike
  readonly tgtIsCausal?: boolean
}

const activationOf = (a: TransformerActivation | undefined): ((x: Value) => Value) =>
  a === undefined || a === 'relu' ? (x) => relu(x) : a === 'gelu' ? (x) => gelu(x) : a

const sub = (p: Params, name: string): Params => (p[name] ?? {}) as Params

/**
 * A float causal mask with zeros on and below the diagonal and -infinity above it
 * (torch.nn.Transformer.generate_square_subsequent_mask), added to attention scores.
 *
 * @param n - The sequence length.
 * @returns `[n, n]`.
 * @example
 * ```ts
 * import { squareSubsequentMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(squareSubsequentMask(2)) // => [[0, -Infinity], [0, 0]]
 * ```
 */
export function squareSubsequentMask(n: number): Tensor {
  const data = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) data[i * n + j] = Number.NEGATIVE_INFINITY
  return fromStorage(data, [n, n], 'float64')
}

/** The feed-forward and attention blocks every transformer layer shares. */
abstract class TransformerBlockBase<I> extends Module<I, Value> {
  readonly dModel: number
  readonly nhead: number
  readonly dimFeedforward: number
  readonly normFirst: boolean
  readonly batchFirst: boolean
  protected readonly act: (x: Value) => Value
  readonly linear1: Linear
  readonly linear2: Linear
  readonly dropout: Dropout
  readonly dropout1: Dropout
  readonly dropout2: Dropout

  protected constructor(dModel: number, nhead: number, options: TransformerLayerOptions, k: Key) {
    super()
    this.dModel = dModel
    this.nhead = nhead
    this.dimFeedforward = options.dimFeedforward ?? 2048
    this.normFirst = options.normFirst ?? false
    this.batchFirst = options.batchFirst ?? false
    this.act = activationOf(options.activation)
    const bias = options.bias ?? true
    const p = options.dropout ?? 0.1
    this.linear1 = new Linear(dModel, this.dimFeedforward, { bias, key: foldIn(k, 10) })
    this.linear2 = new Linear(this.dimFeedforward, dModel, { bias, key: foldIn(k, 11) })
    this.dropout = new Dropout(p)
    this.dropout1 = new Dropout(p)
    this.dropout2 = new Dropout(p)
  }

  protected feedForward(params: Params, x: Value, scope: ChildScope): Value {
    const h = this.act(this.linear1.apply(sub(params, 'linear1'), x))
    const y = this.linear2.apply(sub(params, 'linear2'), this.dropout.apply({}, h, scope.child('dropout', 20)))
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `d_model=${this.dModel}, nhead=${this.nhead}, dim_feedforward=${this.dimFeedforward}, norm_first=${this.normFirst}`
  }
}

/**
 * A transformer encoder layer (torch.nn.TransformerEncoderLayer): self-attention and a feed-forward block, each with
 * dropout, a residual connection and layer normalisation, post-norm by default or pre-norm with `normFirst`. Its
 * parameters follow PyTorch's state-dict layout (`self_attn.*`, `linear1`, `linear2`, `norm1`, `norm2`).
 *
 * @concept transformer
 * @example
 * ```ts
 * import { TransformerEncoderLayer } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new TransformerEncoderLayer(8, 2, { dimFeedforward: 16, batchFirst: true, key: key(0) }).eval().forward(zeros([2, 5, 8])).shape // => [2, 5, 8]
 * ```
 */
export class TransformerEncoderLayer extends TransformerBlockBase<Value | EncoderInput> {
  readonly kind = 'TransformerEncoderLayer'
  readonly selfAttn: MultiheadAttention
  readonly norm1: LayerNorm
  readonly norm2: LayerNorm

  constructor(dModel: number, nhead: number, options: TransformerLayerOptions = {}) {
    const k = initKey('TransformerEncoderLayer', options.key)
    super(dModel, nhead, options, k)
    const eps = options.layerNormEps ?? 1e-5
    const bias = options.bias ?? true
    this.selfAttn = new MultiheadAttention(dModel, nhead, {
      dropout: options.dropout ?? 0.1,
      bias,
      batchFirst: this.batchFirst,
      key: foldIn(k, 0),
    })
    this.norm1 = new LayerNorm(dModel, { eps, bias })
    this.norm2 = new LayerNorm(dModel, { eps, bias })
    this.params = this.initParams(k)
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return [
      ['self_attn', this.selfAttn],
      ['linear1', this.linear1],
      ['dropout', this.dropout],
      ['linear2', this.linear2],
      ['norm1', this.norm1],
      ['norm2', this.norm2],
      ['dropout1', this.dropout1],
      ['dropout2', this.dropout2],
    ]
  }

  override initParams(key: Key): Params {
    return {
      self_attn: this.selfAttn.initParams(foldIn(key, 0)),
      linear1: this.linear1.initParams(foldIn(key, 10)),
      linear2: this.linear2.initParams(foldIn(key, 11)),
      norm1: this.norm1.initParams(),
      norm2: this.norm2.initParams(),
    }
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  apply(params: Params, x: Value | EncoderInput, ctx: ApplyContext = {}): Value {
    const input: EncoderInput = isValue(x) ? { src: x } : x
    const scope = childScope(ctx, this.buffers)
    const sa = (h: Value): Value =>
      this.dropout1.apply(
        {},
        this.selfAttn.apply(
          sub(params, 'self_attn'),
          {
            query: h,
            needWeights: false,
            ...(input.mask === undefined ? {} : { attnMask: input.mask }),
            ...(input.keyPaddingMask === undefined ? {} : { keyPaddingMask: input.keyPaddingMask }),
            ...(input.isCausal === true ? { isCausal: true } : {}),
          },
          scope.child('self_attn', 0),
        ).output,
        scope.child('dropout1', 1),
      )
    const ff = (h: Value): Value =>
      this.dropout2.apply({}, this.feedForward(params, h, scope), scope.child('dropout2', 2))
    const n1 = (h: Value): Value => this.norm1.apply(sub(params, 'norm1'), h)
    const n2 = (h: Value): Value => this.norm2.apply(sub(params, 'norm2'), h)
    let h = asTensor(input.src)
    if (this.normFirst) {
      h = add(h, sa(n1(h)))
      h = add(h, ff(n2(h)))
    } else {
      h = n1(add(h, sa(h)))
      h = n2(add(h, ff(h)))
    }
    scope.done()
    return h
  }
}

/**
 * A transformer decoder layer (torch.nn.TransformerDecoderLayer): masked self-attention, cross-attention to the
 * encoder's memory and a feed-forward block, each with dropout, a residual connection and layer normalisation
 * (`self_attn`, `multihead_attn`, `linear1`, `linear2`, `norm1` to `norm3`).
 *
 * @concept transformer
 * @example
 * ```ts
 * import { TransformerDecoderLayer } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new TransformerDecoderLayer(8, 2, { dimFeedforward: 16, batchFirst: true, key: key(0) }).eval().forward({ tgt: zeros([1, 3, 8]), memory: zeros([1, 5, 8]) }).shape // => [1, 3, 8]
 * ```
 */
export class TransformerDecoderLayer extends TransformerBlockBase<DecoderInput> {
  readonly kind = 'TransformerDecoderLayer'
  readonly selfAttn: MultiheadAttention
  readonly multiheadAttn: MultiheadAttention
  readonly norm1: LayerNorm
  readonly norm2: LayerNorm
  readonly norm3: LayerNorm
  readonly dropout3: Dropout

  constructor(dModel: number, nhead: number, options: TransformerLayerOptions = {}) {
    const k = initKey('TransformerDecoderLayer', options.key)
    super(dModel, nhead, options, k)
    const eps = options.layerNormEps ?? 1e-5
    const bias = options.bias ?? true
    const attn = (i: number): MultiheadAttention =>
      new MultiheadAttention(dModel, nhead, {
        dropout: options.dropout ?? 0.1,
        bias,
        batchFirst: this.batchFirst,
        key: foldIn(k, i),
      })
    this.selfAttn = attn(0)
    this.multiheadAttn = attn(1)
    this.norm1 = new LayerNorm(dModel, { eps, bias })
    this.norm2 = new LayerNorm(dModel, { eps, bias })
    this.norm3 = new LayerNorm(dModel, { eps, bias })
    this.dropout3 = new Dropout(options.dropout ?? 0.1)
    this.params = this.initParams(k)
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return [
      ['self_attn', this.selfAttn],
      ['multihead_attn', this.multiheadAttn],
      ['linear1', this.linear1],
      ['dropout', this.dropout],
      ['linear2', this.linear2],
      ['norm1', this.norm1],
      ['norm2', this.norm2],
      ['norm3', this.norm3],
      ['dropout1', this.dropout1],
      ['dropout2', this.dropout2],
      ['dropout3', this.dropout3],
    ]
  }

  override initParams(key: Key): Params {
    return {
      self_attn: this.selfAttn.initParams(foldIn(key, 0)),
      multihead_attn: this.multiheadAttn.initParams(foldIn(key, 1)),
      linear1: this.linear1.initParams(foldIn(key, 10)),
      linear2: this.linear2.initParams(foldIn(key, 11)),
      norm1: this.norm1.initParams(),
      norm2: this.norm2.initParams(),
      norm3: this.norm3.initParams(),
    }
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: DecoderInput, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const memory = asTensor(input.memory)
    const sa = (h: Value): Value =>
      this.dropout1.apply(
        {},
        this.selfAttn.apply(
          sub(params, 'self_attn'),
          {
            query: h,
            needWeights: false,
            ...(input.tgtMask === undefined ? {} : { attnMask: input.tgtMask }),
            ...(input.tgtKeyPaddingMask === undefined ? {} : { keyPaddingMask: input.tgtKeyPaddingMask }),
            ...(input.tgtIsCausal === true ? { isCausal: true } : {}),
          },
          scope.child('self_attn', 0),
        ).output,
        scope.child('dropout1', 1),
      )
    const ca = (h: Value): Value =>
      this.dropout2.apply(
        {},
        this.multiheadAttn.apply(
          sub(params, 'multihead_attn'),
          {
            query: h,
            key: memory,
            value: memory,
            needWeights: false,
            ...(input.memoryMask === undefined ? {} : { attnMask: input.memoryMask }),
            ...(input.memoryKeyPaddingMask === undefined ? {} : { keyPaddingMask: input.memoryKeyPaddingMask }),
          },
          scope.child('multihead_attn', 3),
        ).output,
        scope.child('dropout2', 4),
      )
    const ff = (h: Value): Value =>
      this.dropout3.apply({}, this.feedForward(params, h, scope), scope.child('dropout3', 5))
    const norm = (name: 'norm1' | 'norm2' | 'norm3', h: Value): Value => this[name].apply(sub(params, name), h)
    let h = asTensor(input.tgt)
    if (this.normFirst) {
      h = add(h, sa(norm('norm1', h)))
      h = add(h, ca(norm('norm2', h)))
      h = add(h, ff(norm('norm3', h)))
    } else {
      h = norm('norm1', add(h, sa(h)))
      h = norm('norm2', add(h, ca(h)))
      h = norm('norm3', add(h, ff(h)))
    }
    scope.done()
    return h
  }
}

/** Options of {@link TransformerEncoder} and {@link TransformerDecoder}. */
export interface StackOptions {
  /** A final normalisation (pre-norm stacks need one). */
  readonly norm?: LayerNorm
}

/** Layers held in a `layers` list and an optional final `norm` (PyTorch's stack layout). */
abstract class LayerStack<L extends Module<never, unknown>, I> extends Module<I, Value> {
  readonly layers: readonly L[]
  readonly layerList: ModuleList<L>
  readonly norm: LayerNorm | undefined

  protected constructor(layer: (i: number) => L, numLayers: number, options: StackOptions) {
    super()
    this.layers = Array.from({ length: numLayers }, (_, i) => layer(i))
    this.layerList = new ModuleList(this.layers)
    this.norm = options.norm
    this.params = assemble(this.children().map(([n, c]) => [n, c.params] as const))
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return this.norm === undefined
      ? [['layers', this.layerList]]
      : [
          ['layers', this.layerList],
          ['norm', this.norm],
        ]
  }

  /** Run the layers in turn, each with its share of the parameters and context, then the final norm. */
  protected run(
    params: Params,
    h0: Value,
    ctx: ApplyContext,
    step: (layer: L, p: Params, h: Value, c: ApplyContext) => Value,
  ): Value {
    const scope = childScope(ctx, this.buffers)
    const listCtx = scope.child('layers', 0)
    const inner = childScope(listCtx, this.layerList.buffers)
    const layers = sub(params, 'layers')
    let h = h0
    for (const [i, layer] of this.layers.entries())
      h = step(layer, sub(layers, String(i)), h, inner.child(String(i), i))
    inner.done()
    scope.done()
    return this.norm === undefined ? h : this.norm.apply(sub(params, 'norm'), h)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

/**
 * A stack of encoder layers (torch.nn.TransformerEncoder): `layers.0` to `layers.{n-1}` and an optional final `norm`.
 * Layers come from a factory (PyTorch deep-copies one layer, so its copies start with equal weights; a factory gives
 * each its own).
 *
 * @concept transformer
 * @example
 * ```ts
 * import { TransformerEncoder, TransformerEncoderLayer } from 'mdml-engine/nn'
 * import { key, foldIn } from 'mdml-engine/random'
 * const enc = new TransformerEncoder((i) => new TransformerEncoderLayer(8, 2, { dimFeedforward: 16, key: foldIn(key(0), i) }), 2)
 * Object.keys(enc.stateDict()).filter((k) => k.startsWith('layers.1.norm2')) // => ['layers.1.norm2.bias', 'layers.1.norm2.weight']
 * ```
 */
export class TransformerEncoder extends LayerStack<TransformerEncoderLayer, Value | EncoderInput> {
  readonly kind = 'TransformerEncoder'

  constructor(layer: (i: number) => TransformerEncoderLayer, numLayers: number, options: StackOptions = {}) {
    super(layer, numLayers, options)
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  apply(params: Params, x: Value | EncoderInput, ctx: ApplyContext = {}): Value {
    const input: EncoderInput = isValue(x) ? { src: x } : x
    return this.run(params, asTensor(input.src), ctx, (layer, p, h, c) => layer.apply(p, { ...input, src: h }, c))
  }
}

/**
 * A stack of decoder layers (torch.nn.TransformerDecoder): `layers.*` and an optional final `norm`.
 *
 * @concept transformer
 * @example
 * ```ts
 * import { TransformerDecoder, TransformerDecoderLayer } from 'mdml-engine/nn'
 * import { key, foldIn } from 'mdml-engine/random'
 * new TransformerDecoder((i) => new TransformerDecoderLayer(8, 2, { dimFeedforward: 16, key: foldIn(key(0), i) }), 3).layers.length // => 3
 * ```
 */
export class TransformerDecoder extends LayerStack<TransformerDecoderLayer, DecoderInput> {
  readonly kind = 'TransformerDecoder'

  constructor(layer: (i: number) => TransformerDecoderLayer, numLayers: number, options: StackOptions = {}) {
    super(layer, numLayers, options)
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: DecoderInput, ctx: ApplyContext = {}): Value {
    return this.run(params, asTensor(input.tgt), ctx, (layer, p, h, c) => layer.apply(p, { ...input, tgt: h }, c))
  }
}

/** The input of {@link Transformer}. */
export interface TransformerInput {
  readonly src: TensorLike
  readonly tgt: TensorLike
  readonly srcMask?: TensorLike
  readonly tgtMask?: TensorLike
  readonly memoryMask?: TensorLike
  readonly srcKeyPaddingMask?: TensorLike
  readonly tgtKeyPaddingMask?: TensorLike
  readonly memoryKeyPaddingMask?: TensorLike
}

/** Options of {@link Transformer}. */
export interface TransformerOptions extends TransformerLayerOptions {
  readonly numEncoderLayers?: number
  readonly numDecoderLayers?: number
}

/**
 * The encoder-decoder transformer (torch.nn.Transformer): an encoder stack and a decoder stack, each ending in a
 * layer normalisation, every matrix initialised Xavier-uniform as PyTorch does.
 *
 * @concept transformer
 * @example
 * ```ts
 * import { Transformer } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * const t = new Transformer(8, 2, { numEncoderLayers: 1, numDecoderLayers: 1, dimFeedforward: 16, batchFirst: true, key: key(0) })
 * t.eval().forward({ src: zeros([1, 4, 8]), tgt: zeros([1, 3, 8]) }).shape // => [1, 3, 8]
 * ```
 */
export class Transformer extends Module<TransformerInput, Value> {
  readonly kind = 'Transformer'
  readonly dModel: number
  readonly nhead: number
  readonly encoder: TransformerEncoder
  readonly decoder: TransformerDecoder

  constructor(dModel = 512, nhead = 8, options: TransformerOptions = {}) {
    super()
    const k = initKey('Transformer', options.key)
    this.dModel = dModel
    this.nhead = nhead
    const eps = options.layerNormEps ?? 1e-5
    const bias = options.bias ?? true
    this.encoder = new TransformerEncoder(
      (i) => new TransformerEncoderLayer(dModel, nhead, { ...options, key: foldIn(foldIn(k, 0), i) }),
      options.numEncoderLayers ?? 6,
      { norm: new LayerNorm(dModel, { eps, bias }) },
    )
    this.decoder = new TransformerDecoder(
      (i) => new TransformerDecoderLayer(dModel, nhead, { ...options, key: foldIn(foldIn(k, 1), i) }),
      options.numDecoderLayers ?? 6,
      { norm: new LayerNorm(dModel, { eps, bias }) },
    )
    this.setParams(this.initParams(k))
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return [
      ['encoder', this.encoder],
      ['decoder', this.decoder],
    ]
  }

  /** PyTorch's _reset_parameters: Xavier-uniform for every parameter with more than one axis. */
  override initParams(key: Key): Params {
    const base = { encoder: this.encoder.initParams(foldIn(key, 0)), decoder: this.decoder.initParams(foldIn(key, 1)) }
    let i = 0
    return treeMap((v) => {
      i += 1
      return v.shape.length > 1 ? xavierUniform(foldIn(foldIn(key, 2), i), v.shape) : v
    }, base)
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  apply(params: Params, input: TransformerInput, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const memory = this.encoder.apply(
      sub(params, 'encoder'),
      {
        src: input.src,
        ...(input.srcMask === undefined ? {} : { mask: input.srcMask }),
        ...(input.srcKeyPaddingMask === undefined ? {} : { keyPaddingMask: input.srcKeyPaddingMask }),
      },
      scope.child('encoder', 0),
    )
    const out = this.decoder.apply(
      sub(params, 'decoder'),
      {
        tgt: input.tgt,
        memory,
        ...(input.tgtMask === undefined ? {} : { tgtMask: input.tgtMask }),
        ...(input.memoryMask === undefined ? {} : { memoryMask: input.memoryMask }),
        ...(input.tgtKeyPaddingMask === undefined ? {} : { tgtKeyPaddingMask: input.tgtKeyPaddingMask }),
        ...(input.memoryKeyPaddingMask === undefined ? {} : { memoryKeyPaddingMask: input.memoryKeyPaddingMask }),
      },
      scope.child('decoder', 1),
    )
    scope.done()
    return out
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `d_model=${this.dModel}, nhead=${this.nhead}`
  }
}

/**
 * Transformer building blocks in the modern pre-norm form: plain and gated (SwiGLU-style) feed-forward blocks, a
 * `TransformerBlock` with grouped-query self-attention, rotary positions, a KV cache and optional cross-attention,
 * stacks of such blocks, an `EncoderDecoder` (T5 style) and a GPT- and LLaMA-shaped decoder-only `TransformerLM`.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { add, asTensor, isValue, mul, type TensorLike, type Value } from '../../tensor/index.ts'
import { type ActivationName, activation } from '../functional/activations.ts'
import { linear } from '../functional/linear.ts'
import { emptyKVCache, GroupedQueryAttention, type GroupedQueryOutput, type KVCache } from '../module/attention.ts'
import { Composite, ModuleList } from '../module/containers.ts'
import { Embedding, Linear } from '../module/layers.ts'
import { type ApplyContext, childScope, initKey, type Module, type Params } from '../module/module.ts'
import { LayerNorm, RMSNorm } from '../module/normalisation.ts'
import { Dropout } from '../module/regularisation.ts'

const keyAt =
  (k: Key) =>
  (i: number): Key =>
    foldIn(k, i)

/** Options of {@link FeedForward}. */
export interface FeedForwardOptions {
  readonly activation?: ActivationName
  readonly dropout?: number
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * The position-wise feed-forward block of a transformer: `fc2(dropout(act(fc1(x))))`, expanding to a hidden width
 * (four times the model width by default).
 *
 * @concept transformer
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.FeedForward(8).eval().forward(zeros([2, 3, 8])).shape // => [2, 3, 8]
 * ```
 */
export class FeedForward extends Composite {
  readonly kind = 'FeedForward'
  readonly dim: number
  readonly hidden: number
  private readonly act: (x: Value) => Value

  constructor(dim: number, hidden: number = 4 * dim, options: FeedForwardOptions = {}) {
    const k = keyAt(initKey('FeedForward', options.key))
    const bias = options.bias ?? true
    super({
      fc1: new Linear(dim, hidden, { bias, key: k(0) }),
      dropout: new Dropout(options.dropout ?? 0),
      fc2: new Linear(hidden, dim, { bias, key: k(2) }),
    })
    this.dim = dim
    this.hidden = hidden
    const f = activation(options.activation ?? 'gelu')
    this.act = (x) => f(x)
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const y = p.run<Value, Value>('fc2', p.run<Value, Value>('dropout', this.act(p.run<Value, Value>('fc1', x))))
    p.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, hidden=${this.hidden}`
  }
}

/**
 * The gated feed-forward block of LLaMA and PaLM (SwiGLU, GeGLU, ReGLU): `down_proj(act(gate_proj(x)) * up_proj(x))`,
 * without biases by default.
 *
 * @concept transformer
 * @reference Shazeer (2020), GLU variants improve transformer.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.GatedFeedForward(8, 16).forward(zeros([1, 2, 8])).shape // => [1, 2, 8]
 * ```
 */
export class GatedFeedForward extends Composite {
  readonly kind = 'GatedFeedForward'
  readonly dim: number
  readonly hidden: number
  private readonly act: (x: Value) => Value

  constructor(dim: number, hidden: number, options: FeedForwardOptions = {}) {
    const k = keyAt(initKey('GatedFeedForward', options.key))
    const bias = options.bias ?? false
    super({
      gate_proj: new Linear(dim, hidden, { bias, key: k(0) }),
      up_proj: new Linear(dim, hidden, { bias, key: k(1) }),
      dropout: new Dropout(options.dropout ?? 0),
      down_proj: new Linear(hidden, dim, { bias, key: k(3) }),
    })
    this.dim = dim
    this.hidden = hidden
    const f = activation(options.activation ?? 'silu')
    this.act = (x) => f(x)
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const h = mul(this.act(p.run<Value, Value>('gate_proj', x)), p.run<Value, Value>('up_proj', x))
    const y = p.run<Value, Value>('down_proj', p.run<Value, Value>('dropout', h))
    p.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, hidden=${this.hidden}`
  }
}

/** Options of {@link TransformerBlock}. */
export interface TransformerBlockOptions {
  /** Key and value heads (grouped-query attention; default numHeads). */
  readonly numKvHeads?: number
  /** Feed-forward hidden width (default 4 dim, or 8/3 dim rounded for a gated block). */
  readonly hidden?: number
  /** Gated feed-forward (SwiGLU) instead of the plain one (default true). */
  readonly gated?: boolean
  readonly activation?: ActivationName
  /** RMS or layer normalisation (default rms). */
  readonly norm?: 'rms' | 'layer'
  /** Normalise before each sub-block (default true) or after the residual sum. */
  readonly preNorm?: boolean
  /** Causal self-attention (default true). */
  readonly causal?: boolean
  /** Sliding-window self-attention. */
  readonly window?: number
  /** Rotary position base (default 10000; false for none). */
  readonly rope?: number | false
  /** A cross-attention sub-block to a context (an encoder-decoder's decoder). */
  readonly crossAttention?: boolean
  readonly dropout?: number
  readonly bias?: boolean
  readonly softCap?: number
  readonly key?: Key
}

/** The input of a transformer block or stack. */
export interface BlockInput {
  /** `[B, L, D]`. */
  readonly x: TensorLike
  /** The self-attention cache of earlier positions. */
  readonly cache?: KVCache
  /** The cross-attention source `[B, S, D]`. */
  readonly context?: TensorLike
  /** A boolean self-attention mask (true = attend), combined with the causal mask. */
  readonly mask?: TensorLike
}

/** The output of a transformer block: its output and the self-attention cache including the new positions. */
export interface BlockOutput {
  readonly output: Value
  readonly cache: KVCache
}

const normOf = (kind: 'rms' | 'layer', dim: number, bias: boolean): Module =>
  kind === 'rms' ? new RMSNorm(dim, { eps: 1e-6 }) : new LayerNorm(dim, { bias })

/**
 * A transformer block: (grouped-query) self-attention with rotary positions and a KV cache, optional cross-attention
 * to a context, and a (gated) feed-forward block, each with a residual connection and RMS or layer normalisation
 * before (pre-norm, LLaMA and GPT-2) or after (post-norm, the original Transformer) the residual.
 *
 * @concept transformer
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.TransformerBlock(8, 2, { numKvHeads: 1 }).forward({ x: zeros([1, 4, 8]) }).output.shape // => [1, 4, 8]
 * ```
 */
export class TransformerBlock extends Composite<BlockInput | Value, BlockOutput> {
  readonly kind = 'TransformerBlock'
  readonly dim: number
  readonly numHeads: number
  readonly preNorm: boolean
  readonly crossAttention: boolean

  constructor(dim: number, numHeads: number, options: TransformerBlockOptions = {}) {
    const k = keyAt(initKey('TransformerBlock', options.key))
    const gated = options.gated ?? true
    const hidden = options.hidden ?? (gated ? Math.round((8 * dim) / 3) : 4 * dim)
    const norm = options.norm ?? 'rms'
    const bias = options.bias ?? false
    const attn = (i: number, cross: boolean): GroupedQueryAttention =>
      new GroupedQueryAttention(dim, numHeads, {
        numKvHeads: options.numKvHeads ?? numHeads,
        causal: cross ? false : (options.causal ?? true),
        rope: cross ? false : options.rope === undefined ? 10000 : options.rope,
        bias,
        ...(cross || options.window === undefined ? {} : { window: options.window }),
        ...(options.softCap === undefined ? {} : { softCap: options.softCap }),
        key: k(i),
      })
    const cross = options.crossAttention ?? false
    super({
      attn_norm: normOf(norm, dim, bias),
      attn: attn(1, false),
      ...(cross ? { cross_norm: normOf(norm, dim, bias), cross_attn: attn(3, true) } : {}),
      ffn_norm: normOf(norm, dim, bias),
      ffn: gated
        ? new GatedFeedForward(dim, hidden, { activation: options.activation ?? 'silu', bias, key: k(5) })
        : new FeedForward(dim, hidden, { activation: options.activation ?? 'gelu', bias, key: k(5) }),
      dropout: new Dropout(options.dropout ?? 0),
    })
    this.dim = dim
    this.numHeads = numHeads
    this.preNorm = options.preNorm ?? true
    this.crossAttention = cross
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? { x: asTensor(x as TensorLike) } : x
  }

  apply(params: Params, input: BlockInput | Value, ctx: ApplyContext = {}): BlockOutput {
    const inp: BlockInput = isValue(input) ? { x: input } : input
    const p = this.parts(params, ctx)
    const drop = (v: Value): Value => p.run<Value, Value>('dropout', v)
    const sub = (h: Value, normName: string, f: (v: Value) => Value): Value => {
      if (this.preNorm) return add(h, drop(f(p.run<Value, Value>(normName, h))))
      return p.run<Value, Value>(normName, add(h, drop(f(h))))
    }
    let cache: KVCache | undefined
    let h = asTensor(inp.x)
    h = sub(h, 'attn_norm', (v) => {
      const r = p.run<object, GroupedQueryOutput>('attn', {
        x: v,
        ...(inp.cache === undefined ? {} : { cache: inp.cache }),
        ...(inp.mask === undefined ? {} : { mask: inp.mask }),
      })
      cache = r.cache
      return r.output
    })
    if (this.crossAttention) {
      if (inp.context === undefined) throw new OmaiError('value', 'TransformerBlock: cross-attention needs a context')
      const context = asTensor(inp.context)
      h = sub(h, 'cross_norm', (v) => p.run<object, GroupedQueryOutput>('cross_attn', { x: v, context }).output)
    }
    h = sub(h, 'ffn_norm', (v) => p.run<Value, Value>('ffn', v))
    p.done()
    return { output: h, cache: cache as KVCache }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.numHeads}, ${this.preNorm ? 'pre' : 'post'}-norm${this.crossAttention ? ', cross-attention' : ''}`
  }
}

/** The input of a {@link TransformerStack}. */
export interface StackInput {
  readonly x: TensorLike
  /** One cache per block, from an earlier step. */
  readonly caches?: readonly KVCache[]
  readonly context?: TensorLike
  readonly mask?: TensorLike
}

/** The output of a {@link TransformerStack}: the output and one cache per block. */
export interface StackOutput {
  readonly output: Value
  readonly caches: readonly KVCache[]
}

/**
 * A stack of transformer blocks with an optional final normalisation (`blocks.*`, `norm`): a decoder-only stack, an
 * encoder (non-causal blocks) or a decoder with cross-attention, threading one KV cache per block for incremental
 * decoding.
 *
 * @concept transformer
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * new blocks.TransformerStack(2, (i) => new blocks.TransformerBlock(8, 2)).forward({ x: zeros([1, 3, 8]) }).caches.length // => 2
 * ```
 */
export class TransformerStack extends Composite<StackInput | Value, StackOutput> {
  readonly kind = 'TransformerStack'
  readonly numLayers: number

  constructor(
    numLayers: number,
    block: (i: number) => TransformerBlock,
    options: { readonly finalNorm?: Module } = {},
  ) {
    super({
      blocks: new ModuleList(Array.from({ length: numLayers }, (_, i) => block(i))),
      ...(options.finalNorm === undefined ? {} : { norm: options.finalNorm }),
    })
    this.numLayers = numLayers
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? { x: asTensor(x as TensorLike) } : x
  }

  /** The blocks, in order. */
  get blockList(): readonly TransformerBlock[] {
    return this.part<ModuleList<TransformerBlock>>('blocks').modules
  }

  /** Empty caches for a batch (one per block). */
  emptyCaches(batch: number): KVCache[] {
    return this.blockList.map((b) => {
      const attn = b.part<GroupedQueryAttention>('attn')
      return emptyKVCache(batch, attn.numKvHeads, attn.headDim)
    })
  }

  apply(params: Params, input: StackInput | Value, ctx: ApplyContext = {}): StackOutput {
    const inp: StackInput = isValue(input) ? { x: input } : input
    const p = this.parts(params, ctx)
    const list = this.part<ModuleList<TransformerBlock>>('blocks')
    const inner = childScope(p.context('blocks'), list.buffers)
    const listParams = (params['blocks'] ?? {}) as Params
    let h = asTensor(inp.x)
    const caches: KVCache[] = []
    for (const [i, block] of this.blockList.entries()) {
      const cache = inp.caches?.[i]
      const r = block.apply(
        (listParams[String(i)] ?? {}) as Params,
        {
          x: h,
          ...(cache === undefined ? {} : { cache }),
          ...(inp.context === undefined ? {} : { context: inp.context }),
          ...(inp.mask === undefined ? {} : { mask: inp.mask }),
        },
        inner.child(String(i), i),
      )
      h = r.output
      caches.push(r.cache)
    }
    inner.done()
    const output = this.has('norm') ? p.run<Value, Value>('norm', h) : h
    p.done()
    return { output, caches }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

/**
 * An encoder-decoder transformer from two stacks: the encoder's output is the decoder's cross-attention context
 * (T5 and the original Transformer, in the pre-norm form).
 *
 * @concept transformer
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * const m = blocks.encoderDecoder(8, 2, { encoderLayers: 1, decoderLayers: 1 })
 * m.forward({ source: zeros([1, 5, 8]), target: zeros([1, 3, 8]) }).output.shape // => [1, 3, 8]
 * ```
 */
export class EncoderDecoder extends Composite<
  { readonly source: TensorLike; readonly target: TensorLike; readonly caches?: readonly KVCache[] },
  StackOutput
> {
  readonly kind = 'EncoderDecoder'

  constructor(encoder: TransformerStack, decoder: TransformerStack) {
    super({ encoder, decoder })
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  /** Encode a source once (to reuse across decoding steps). */
  encode(params: Params, source: TensorLike, ctx: ApplyContext = {}): Value {
    return this.part<TransformerStack>('encoder').apply((params['encoder'] ?? {}) as Params, { x: source }, ctx).output
  }

  apply(
    params: Params,
    input: { readonly source: TensorLike; readonly target: TensorLike; readonly caches?: readonly KVCache[] },
    ctx: ApplyContext = {},
  ): StackOutput {
    const p = this.parts(params, ctx)
    const memory = p.run<StackInput, StackOutput>('encoder', { x: input.source }).output
    const out = p.run<StackInput, StackOutput>('decoder', {
      x: input.target,
      context: memory,
      ...(input.caches === undefined ? {} : { caches: input.caches }),
    })
    p.done()
    return out
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

/**
 * Build an encoder-decoder transformer: a non-causal encoder stack and a causal decoder stack with cross-attention,
 * both pre-norm with a final RMS normalisation.
 *
 * @param dim - Model width.
 * @param numHeads - Attention heads.
 * @param options - Layer counts and the block options.
 * @returns The model.
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * blocks.encoderDecoder(8, 2, { encoderLayers: 2, decoderLayers: 1 }).kind // => 'EncoderDecoder'
 * ```
 */
export function encoderDecoder(
  dim: number,
  numHeads: number,
  options: TransformerBlockOptions & { readonly encoderLayers?: number; readonly decoderLayers?: number } = {},
): EncoderDecoder {
  const k = initKey('EncoderDecoder', options.key)
  const { encoderLayers = 6, decoderLayers = 6, ...block } = options
  const encoder = new TransformerStack(
    encoderLayers,
    (i) => new TransformerBlock(dim, numHeads, { ...block, causal: false, key: foldIn(foldIn(k, 0), i) }),
    { finalNorm: new RMSNorm(dim, { eps: 1e-6 }) },
  )
  const decoder = new TransformerStack(
    decoderLayers,
    (i) => new TransformerBlock(dim, numHeads, { ...block, crossAttention: true, key: foldIn(foldIn(k, 1), i) }),
    { finalNorm: new RMSNorm(dim, { eps: 1e-6 }) },
  )
  return new EncoderDecoder(encoder, decoder)
}

/** Options of {@link TransformerLM}. */
export interface TransformerLMOptions extends TransformerBlockOptions {
  /** Share the embedding with the output projection (default true). */
  readonly tieEmbeddings?: boolean
}

/** The input of {@link TransformerLM}: token ids `[B, T]`, or with caches from earlier steps. */
export type LMInput = TensorLike | { readonly tokens: TensorLike; readonly caches?: readonly KVCache[] }

/**
 * A decoder-only transformer language model (GPT and LLaMA shaped): token embedding, a stack of pre-norm blocks with
 * rotary positions, a final normalisation and an output projection (tied to the embedding by default); returns logits
 * `[B, T, V]` and the KV caches for incremental decoding.
 *
 * @concept language-models
 * @example
 * ```ts
 * import { blocks } from 'omai-engine/nn'
 * new blocks.TransformerLM(50, 16, 2, 2).forward([[1, 2, 3]]).output.shape // => [1, 3, 50]
 * ```
 */
export class TransformerLM extends Composite<LMInput, StackOutput> {
  readonly kind = 'TransformerLM'
  readonly vocabSize: number
  readonly dim: number
  readonly tieEmbeddings: boolean

  constructor(vocabSize: number, dim: number, numLayers: number, numHeads: number, options: TransformerLMOptions = {}) {
    const k = initKey('TransformerLM', options.key)
    const tie = options.tieEmbeddings ?? true
    super({
      embed: new Embedding(vocabSize, dim, { key: foldIn(k, 0) }),
      stack: new TransformerStack(
        numLayers,
        (i) => new TransformerBlock(dim, numHeads, { ...options, key: foldIn(foldIn(k, 1), i) }),
        { finalNorm: options.norm === 'layer' ? new LayerNorm(dim) : new RMSNorm(dim, { eps: 1e-6 }) },
      ),
      ...(tie ? {} : { lm_head: new Linear(dim, vocabSize, { bias: false, key: foldIn(k, 2) }) }),
    })
    this.vocabSize = vocabSize
    this.dim = dim
    this.tieEmbeddings = tie
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  /** Empty caches for a batch. */
  emptyCaches(batch: number): KVCache[] {
    return this.part<TransformerStack>('stack').emptyCaches(batch)
  }

  apply(params: Params, input: LMInput, ctx: ApplyContext = {}): StackOutput {
    const record =
      typeof input === 'object' && input !== null && 'tokens' in input
        ? (input as { readonly tokens: TensorLike; readonly caches?: readonly KVCache[] })
        : { tokens: input as TensorLike }
    const p = this.parts(params, ctx)
    const h = p.run<Value, Value>('embed', asTensor(record.tokens, 'int32'))
    const r = p.run<StackInput, StackOutput>('stack', {
      x: h,
      ...(record.caches === undefined ? {} : { caches: record.caches }),
    })
    const embed = (params['embed'] ?? {}) as Params
    const logits = this.tieEmbeddings
      ? linear(r.output, embed['weight'] as Value)
      : p.run<Value, Value>('lm_head', r.output)
    p.done()
    return { output: logits, caches: r.caches }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape, this.vocabSize]
  }

  override extraRepr(): string {
    return `vocab=${this.vocabSize}, dim=${this.dim}${this.tieEmbeddings ? ', tied' : ''}`
  }
}

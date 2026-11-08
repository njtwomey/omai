/**
 * Attention layers as modules: `MultiheadAttention` with torch.nn.MultiheadAttention's parameter layout (so PyTorch
 * weights load directly), and `GroupedQueryAttention` (Ainslie et al. 2023, where several query heads share each key
 * and value head) with rotary positions and a KV cache of earlier positions for incremental decoding.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  asTensor,
  concat,
  expandDims,
  fromStorage,
  isValue,
  logicalAnd,
  squeeze,
  swapaxes,
  type TensorLike,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import {
  attention,
  type MultiHeadAttentionParams,
  mergeHeads,
  multiHeadAttention,
  splitHeads,
} from '../functional/attention.ts'
import { linear } from '../functional/linear.ts'
import { applyRotary, rotaryAngles } from '../functional/positional.ts'
import { torchDefaultUniform, xavierNormal, xavierUniform } from '../init.ts'
import { type ApplyContext, initKey, isTraining, Module, type Params, tap } from './module.ts'

/** The input of {@link MultiheadAttention}: a query (self-attention) or a record with key, value and masks. */
export interface AttentionInput {
  readonly query: TensorLike
  /** Keys (default: the query). */
  readonly key?: TensorLike
  /** Values (default: the keys). */
  readonly value?: TensorLike
  /** `[B, S]`, true for padding positions to ignore. */
  readonly keyPaddingMask?: TensorLike
  /** `[L, S]` or `[B H, L, S]`: boolean true = not allowed (PyTorch's convention), or additive. */
  readonly attnMask?: TensorLike
  readonly isCausal?: boolean
  /** Return the attention weights (default true, as PyTorch). */
  readonly needWeights?: boolean
  /** Average the weights over heads (default true). */
  readonly averageAttnWeights?: boolean
}

/** The output of {@link MultiheadAttention}. */
export interface AttentionOutput {
  readonly output: Value
  /** `[B, L, S]` (or `[B, H, L, S]`), when asked for. */
  readonly weights?: Value
}

/** Options of {@link MultiheadAttention}. */
export interface MultiheadAttentionOptions {
  /** Dropout on the attention weights in training (default 0). */
  readonly dropout?: number
  readonly bias?: boolean
  /** Learnt key and value rows appended to the sequence. */
  readonly addBiasKv?: boolean
  /** Append a zero key and value. */
  readonly addZeroAttn?: boolean
  /** Key and value widths (default the embedding width). */
  readonly kdim?: number
  readonly vdim?: number
  /** Inputs are `[B, L, E]` (default false: `[L, B, E]`, as PyTorch). */
  readonly batchFirst?: boolean
  readonly key?: Key
}

/**
 * Multi-head attention (torch.nn.MultiheadAttention) with PyTorch's parameter layout (`in_proj_weight`,
 * `in_proj_bias`, `out_proj.weight`, `out_proj.bias`, or `q/k/v_proj_weight` for other key and value widths;
 * `bias_k`, `bias_v`) and initialisation, sequence-first by default. A single input is self-attention; the weights
 * (averaged over heads unless asked otherwise) are returned and tapped as `weights`.
 *
 * @concept attention
 * @index module Multi-head attention with torch.nn's parameter layout.
 * @example
 * ```ts
 * import { MultiheadAttention } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { zeros } from 'omai-engine/tensor'
 * new MultiheadAttention(8, 2, { batchFirst: true, key: key(0) }).forward(zeros([3, 5, 8])).weights?.shape // => [3, 5, 5]
 * ```
 */
export class MultiheadAttention extends Module<Value | AttentionInput, AttentionOutput> {
  readonly kind = 'MultiheadAttention'
  readonly embedDim: number
  readonly numHeads: number
  readonly kdim: number
  readonly vdim: number
  readonly dropout: number
  readonly bias: boolean
  readonly addBiasKv: boolean
  readonly addZeroAttn: boolean
  readonly batchFirst: boolean

  constructor(embedDim: number, numHeads: number, options: MultiheadAttentionOptions = {}) {
    super()
    if (embedDim % numHeads !== 0)
      throw new OmaiError('value', `MultiheadAttention: ${embedDim} does not divide into ${numHeads} heads`)
    this.embedDim = embedDim
    this.numHeads = numHeads
    this.kdim = options.kdim ?? embedDim
    this.vdim = options.vdim ?? embedDim
    this.dropout = options.dropout ?? 0
    this.bias = options.bias ?? true
    this.addBiasKv = options.addBiasKv ?? false
    this.addZeroAttn = options.addZeroAttn ?? false
    this.batchFirst = options.batchFirst ?? false
    this.params = this.initParams(initKey('MultiheadAttention', options.key))
  }

  private get packed(): boolean {
    return this.kdim === this.embedDim && this.vdim === this.embedDim
  }

  override initParams(key: Key): Params {
    const e = this.embedDim
    const proj = this.packed
      ? { in_proj_weight: xavierUniform(foldIn(key, 0), [3 * e, e]) }
      : {
          q_proj_weight: xavierUniform(foldIn(key, 0), [e, e]),
          k_proj_weight: xavierUniform(foldIn(key, 1), [e, this.kdim]),
          v_proj_weight: xavierUniform(foldIn(key, 2), [e, this.vdim]),
        }
    return {
      ...proj,
      ...(this.bias ? { in_proj_bias: zeros([3 * e]) } : {}),
      out_proj: {
        weight: torchDefaultUniform(foldIn(key, 3), [e, e], e),
        ...(this.bias ? { bias: zeros([e]) } : {}),
      },
      ...(this.addBiasKv
        ? { bias_k: xavierNormal(foldIn(key, 4), [1, 1, e]), bias_v: xavierNormal(foldIn(key, 5), [1, 1, e]) }
        : {}),
    }
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  /** The projections in the functional form's layout. */
  projections(params: Params): MultiHeadAttentionParams {
    const out = params['out_proj'] as Params
    const pick = (name: string): Value | undefined => params[name] as Value | undefined
    const optional = (name: string, field: keyof MultiHeadAttentionParams): Partial<MultiHeadAttentionParams> => {
      const v = pick(name)
      return v === undefined ? {} : { [field]: v }
    }
    return {
      outProjWeight: out['weight'] as Value,
      ...(out['bias'] === undefined ? {} : { outProjBias: out['bias'] as Value }),
      ...optional('in_proj_weight', 'inProjWeight'),
      ...optional('q_proj_weight', 'qProjWeight'),
      ...optional('k_proj_weight', 'kProjWeight'),
      ...optional('v_proj_weight', 'vProjWeight'),
      ...optional('in_proj_bias', 'inProjBias'),
      ...optional('bias_k', 'biasK'),
      ...optional('bias_v', 'biasV'),
    }
  }

  apply(params: Params, x: Value | AttentionInput, ctx: ApplyContext = {}): AttentionOutput {
    const input: AttentionInput = isValue(x) ? { query: x } : x
    const q0 = asTensor(input.query)
    const k0 = input.key === undefined ? q0 : asTensor(input.key)
    const v0 = input.value === undefined ? k0 : asTensor(input.value)
    const unbatched = q0.shape.length === 2
    // To batch-first [B, L, E].
    const toBatch = (t: Value): Value => (unbatched ? expandDims(t, 0) : this.batchFirst ? t : swapaxes(t, 0, 1))
    const training = isTraining(ctx)
    const r = multiHeadAttention(toBatch(q0), toBatch(k0), toBatch(v0), this.projections(params), {
      numHeads: this.numHeads,
      ...(input.keyPaddingMask === undefined
        ? {}
        : { keyPaddingMask: unbatched ? expandDims(asTensor(input.keyPaddingMask), 0) : input.keyPaddingMask }),
      ...(input.attnMask === undefined ? {} : { attnMask: input.attnMask }),
      ...(input.isCausal === true ? { causal: true } : {}),
      addZeroAttn: this.addZeroAttn,
      averageAttnWeights: input.averageAttnWeights ?? true,
      ...(this.dropout > 0 && training ? { dropout: this.dropout, training } : {}),
      ...(ctx.key === undefined ? {} : { key: ctx.key }),
    })
    tap(ctx, 'weights', r.weights)
    const output = unbatched ? squeeze(r.output, 0) : this.batchFirst ? r.output : swapaxes(r.output, 0, 1)
    if (input.needWeights === false) return { output }
    return { output, weights: unbatched ? squeeze(r.weights, 0) : r.weights }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.embedDim]
  }

  override extraRepr(): string {
    return `embed_dim=${this.embedDim}, num_heads=${this.numHeads}, batch_first=${this.batchFirst}`
  }
}

/** Cached keys and values of earlier positions for incremental decoding, `[B, H_kv, T, D]` each. */
export interface KVCache {
  readonly keys: Value
  readonly values: Value
}

/**
 * An empty KV cache for a batch.
 *
 * @param batch - Batch size.
 * @param numKvHeads - Key and value heads.
 * @param headDim - Head width.
 * @returns A cache with no positions.
 * @example
 * ```ts
 * import { emptyKVCache } from 'omai-engine/nn'
 * emptyKVCache(2, 1, 4).keys.shape // => [2, 1, 0, 4]
 * ```
 */
export function emptyKVCache(batch: number, numKvHeads: number, headDim: number): KVCache {
  return { keys: zeros([batch, numKvHeads, 0, headDim]), values: zeros([batch, numKvHeads, 0, headDim]) }
}

/** The input of {@link GroupedQueryAttention}. */
export interface GroupedQueryInput {
  /** `[B, L, E]`. */
  readonly x: TensorLike
  /** Keys and values from earlier steps; the new positions come after them. */
  readonly cache?: KVCache
  /** Cross-attention source `[B, S, E]` (no cache, no causal mask, no rotation of the source). */
  readonly context?: TensorLike
  /** A boolean `[L, S]` or `[B, 1, L, S]` mask, true = attend. */
  readonly mask?: TensorLike
}

/** The output of {@link GroupedQueryAttention}: the output and the cache including the new positions. */
export interface GroupedQueryOutput {
  readonly output: Value
  readonly cache: KVCache
}

/** Options of {@link GroupedQueryAttention}. */
export interface GroupedQueryOptions {
  /** Key and value heads (default numHeads: multi-head; 1: multi-query attention). */
  readonly numKvHeads?: number
  /** Head width (default embedDim / numHeads). */
  readonly headDim?: number
  readonly bias?: boolean
  /** Causal self-attention (default true). */
  readonly causal?: boolean
  /** Sliding-window size. */
  readonly window?: number
  /** Rotary position embeddings with this base (default 10000; `false` for none). */
  readonly rope?: number | false
  /** Soft-cap the scores. */
  readonly softCap?: number
  readonly dropout?: number
  readonly key?: Key
}

/**
 * Grouped-query attention (Ainslie et al. 2023; LLaMA 2 and 3, Mistral) with separate `q_proj`, `k_proj`, `v_proj`
 * and `o_proj` linear maps: numKvHeads key and value heads shared by groups of query heads (one shared head is
 * multi-query attention), rotary positions, causal or sliding-window masks, soft-capping and a KV cache for
 * incremental decoding.
 *
 * @concept attention
 * @reference Ainslie et al. (2023), GQA: training generalized multi-query transformer models from multi-head
 *   checkpoints.
 * @example
 * ```ts
 * import { GroupedQueryAttention } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { zeros } from 'omai-engine/tensor'
 * new GroupedQueryAttention(8, 4, { numKvHeads: 2, key: key(0) }).forward({ x: zeros([1, 3, 8]) }).cache.keys.shape // => [1, 2, 3, 2]
 * ```
 */
export class GroupedQueryAttention extends Module<GroupedQueryInput, GroupedQueryOutput> {
  readonly kind = 'GroupedQueryAttention'
  readonly embedDim: number
  readonly numHeads: number
  readonly numKvHeads: number
  readonly headDim: number
  readonly bias: boolean
  readonly causal: boolean
  readonly window: number | undefined
  readonly rope: number | false
  readonly softCap: number | undefined
  readonly dropout: number

  constructor(embedDim: number, numHeads: number, options: GroupedQueryOptions = {}) {
    super()
    this.embedDim = embedDim
    this.numHeads = numHeads
    this.numKvHeads = options.numKvHeads ?? numHeads
    if (numHeads % this.numKvHeads !== 0)
      throw new OmaiError('value', `GroupedQueryAttention: ${numHeads} heads do not divide into ${this.numKvHeads}`)
    this.headDim = options.headDim ?? embedDim / numHeads
    this.bias = options.bias ?? false
    this.causal = options.causal ?? true
    this.window = options.window
    this.rope = options.rope === undefined ? 10000 : options.rope
    this.softCap = options.softCap
    this.dropout = options.dropout ?? 0
    this.params = this.initParams(initKey('GroupedQueryAttention', options.key))
  }

  override initParams(key: Key): Params {
    const e = this.embedDim
    const q = this.numHeads * this.headDim
    const kv = this.numKvHeads * this.headDim
    const lin = (k: Key, out: number, fanIn: number): Params => ({
      weight: torchDefaultUniform(foldIn(k, 0), [out, fanIn], fanIn),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(k, 1), [out], fanIn) } : {}),
    })
    return {
      q_proj: lin(foldIn(key, 0), q, e),
      k_proj: lin(foldIn(key, 1), kv, e),
      v_proj: lin(foldIn(key, 2), kv, e),
      o_proj: lin(foldIn(key, 3), e, q),
    }
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? { x: asTensor(x as TensorLike) } : x
  }

  apply(params: Params, input: GroupedQueryInput, ctx: ApplyContext = {}): GroupedQueryOutput {
    const x = asTensor(input.x)
    const lin = (name: string, v: Value): Value => {
      const p = params[name] as Params
      return linear(v, p['weight'] as Value, p['bias'] as Value | undefined)
    }
    const source = input.context === undefined ? x : asTensor(input.context)
    let q = splitHeads(lin('q_proj', x), this.numHeads)
    let k = splitHeads(lin('k_proj', source), this.numKvHeads)
    const v = splitHeads(lin('v_proj', source), this.numKvHeads)
    const past = input.context === undefined ? (input.cache?.keys.shape[2] ?? 0) : 0
    if (this.rope !== false && input.context === undefined) {
      const l = x.shape[1] ?? 0
      const angles = rotaryAngles(l, this.headDim, { base: this.rope, offset: past })
      q = applyRotary(q, angles)
      k = applyRotary(k, angles)
    }
    const cache: KVCache =
      input.context === undefined && input.cache !== undefined
        ? { keys: concat([input.cache.keys, k], 2), values: concat([input.cache.values, v], 2) }
        : { keys: k, values: v }
    const l = q.shape[2] ?? 0
    const s = cache.keys.shape[2] ?? 0
    // With a cache, query i sits at position past + i: allow keys j <= past + i (and within the window).
    let mask: Value | undefined
    if (this.causal && input.context === undefined) {
      const data = new Uint8Array(l * s)
      for (let i = 0; i < l; i++)
        for (let j = 0; j < s; j++)
          data[i * s + j] = j <= past + i && (this.window === undefined || past + i - j < this.window) ? 1 : 0
      mask = fromStorage(data, [l, s], 'bool')
    }
    if (input.mask !== undefined) {
      const given = asTensor(input.mask, 'bool')
      mask = mask === undefined ? given : logicalAnd(mask, given)
    }
    const training = isTraining(ctx)
    const r = attention(q, cache.keys, cache.values, {
      enableGqa: true,
      ...(mask === undefined ? {} : { mask }),
      ...(this.softCap === undefined ? {} : { softCap: this.softCap }),
      ...(this.dropout > 0 && training
        ? { dropout: this.dropout, training, ...(ctx.key === undefined ? {} : { key: ctx.key }) }
        : {}),
    })
    tap(ctx, 'weights', r.weights)
    return { output: lin('o_proj', mergeHeads(r.output)), cache }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `embed_dim=${this.embedDim}, num_heads=${this.numHeads}, num_kv_heads=${this.numKvHeads}, head_dim=${this.headDim}`
  }
}

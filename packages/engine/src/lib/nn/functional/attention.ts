/**
 * Attention as functions, torch.nn.functional style: scaled dot-product attention softmax(q k^T / sqrt(d)) v with
 * boolean or additive masks, causal, sliding-window and local masks, soft-capping and dropout, returning the attention
 * weights when asked. Also head splitting and merging, multi-head attention with PyTorch's parameter layout, and linear
 * attention (Katharopoulos et al. 2020).
 *
 * @index function Scaled dot-product attention with masks, causal and sliding-window forms.
 */

import { MdmlError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  cast,
  concat,
  cumsum,
  div,
  equal,
  exp,
  expandDims,
  fromStorage,
  isfinite,
  logicalAnd,
  matmul,
  max,
  mean,
  mul,
  promoteFloat,
  repeat,
  reshape,
  sliceAxis,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  tanh,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { elu } from './activations.ts'
import { dropout } from './regularisation.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/**
 * Softmax along the last axis over the positions a mask allows (true = take part), safe when a row allows nothing:
 * such a row gives zeros rather than NaN.
 *
 * @param logits - Scores `[..., S]`.
 * @param mask - Boolean mask broadcastable to the scores (true where a position takes part).
 * @returns Weights that sum to one over each row's allowed positions.
 * @example
 * ```ts
 * import { maskedSoftmax } from 'mdml-engine/nn'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(maskedSoftmax([[0, 0, 5]], tensor([[true, true, false]], { dtype: 'bool' }))) // => [[0.5, 0.5, 0]]
 * ```
 */
export function maskedSoftmax(logits: TensorLike, mask: TensorLike): Value {
  const z = fl(logits)
  const m = asTensor(mask, 'bool')
  const masked = where(m, z, Number.NEGATIVE_INFINITY)
  const top = max(masked, -1, true)
  const shift = where(isfinite(top), top, 0)
  const e = where(m, exp(sub(z, shift)), 0)
  const total = sum(e, -1, true)
  return div(e, where(equal(total, 0), 1, total))
}

/**
 * A causal mask `[L, S]`: position i attends to positions j <= i (aligned at the top left, as PyTorch's `is_causal`),
 * or, with `window`, to the `window` most recent positions i - window < j <= i (sliding-window attention, Mistral).
 *
 * @param length - Query length L.
 * @param sourceLength - Key length S (default L).
 * @param window - Window size (default: unlimited).
 * @returns A boolean mask, true where attention is allowed.
 * @example
 * ```ts
 * import { causalMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(causalMask(3, 3, 2)) // => [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
 * ```
 */
export function causalMask(length: number, sourceLength: number = length, window?: number): Value {
  const data = new Uint8Array(length * sourceLength)
  for (let i = 0; i < length; i++) {
    for (let j = 0; j < sourceLength; j++) {
      data[i * sourceLength + j] = j <= i && (window === undefined || i - j < window) ? 1 : 0
    }
  }
  return fromStorage(data, [length, sourceLength], 'bool')
}

/**
 * A symmetric local mask `[L, S]`: position i attends to j with |i - j| <= radius (Longformer's local attention).
 *
 * @param length - Query length.
 * @param sourceLength - Key length (default L).
 * @param radius - How far each position sees on either side.
 * @returns A boolean mask.
 * @example
 * ```ts
 * import { localMask } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(localMask(3, 3, 1)) // => [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
 * ```
 */
export function localMask(length: number, sourceLength: number = length, radius = 1): Value {
  const data = new Uint8Array(length * sourceLength)
  for (let i = 0; i < length; i++)
    for (let j = 0; j < sourceLength; j++) data[i * sourceLength + j] = Math.abs(i - j) <= radius ? 1 : 0
  return fromStorage(data, [length, sourceLength], 'bool')
}

/** Options of {@link scaledDotProductAttention}. */
export interface AttentionOptions {
  /**
   * A boolean mask broadcastable to `[..., L, S]`, true where attention is allowed (torch's scaled_dot_product_attention
   * convention), or a float mask added to the scores.
   */
  readonly mask?: TensorLike
  /** An additive bias on the scores `[..., L, S]` (relative positions, ALiBi). */
  readonly bias?: TensorLike
  /** Causal attention (a query sees keys up to its own position). */
  readonly causal?: boolean
  /** Restrict a causal attention to this many most recent positions (sliding-window attention). */
  readonly window?: number
  /** Score scale (default 1 / sqrt(E)). */
  readonly scale?: number
  /** Soft-cap the scores: cap tanh(scores / cap) (Gemma 2). */
  readonly softCap?: number
  /** Dropout on the attention weights (needs `key` when training). */
  readonly dropout?: number
  readonly key?: Key
  readonly training?: boolean
  /** Query heads share key and value heads (grouped-query attention): repeat the key/value heads (axis -3). */
  readonly enableGqa?: boolean
}

/** Result of {@link attention}: the output and the attention weights. */
export interface AttentionResult {
  /** `[..., L, Ev]`. */
  readonly output: Value
  /** `[..., L, S]`, each row summing to one over the attended positions. */
  readonly weights: Value
}

const repeatKv = (x: Value, groups: number): Value => (groups === 1 ? x : repeat(x, groups, x.shape.length - 3))

/**
 * Scaled dot-product attention with its weights: softmax(q k^T scale + bias + mask) v, with boolean or additive masks,
 * causal and sliding-window masking, score soft-capping, dropout on the weights and grouped-query heads. Rows with no
 * allowed position give zeros.
 *
 * @param query - `[..., L, E]`.
 * @param key - `[..., S, E]`.
 * @param value - `[..., S, Ev]`.
 * @param options - Mask, bias, causal, window, scale, softCap, dropout (with key and training) and enableGqa.
 * @returns The output and the weights.
 * @concept attention
 * @reference Vaswani et al. (2017), Attention is all you need.
 * @example
 * ```ts
 * import { attention } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(attention([[1, 0]], [[1, 0], [0, 1]], [[1, 2], [3, 4]], { causal: true }).output) // => [[1, 2]]
 * ```
 */
export function attention(
  query: TensorLike,
  key: TensorLike,
  value: TensorLike,
  options: AttentionOptions = {},
): AttentionResult {
  const [q, k0, v0] = promoteFloat(query, key, value) as [Value, Value, Value]
  let k = k0
  let v = v0
  if (options.enableGqa === true) {
    const hq = q.shape[q.shape.length - 3] ?? 1
    const hk = k.shape[k.shape.length - 3] ?? 1
    if (hq % hk !== 0) throw new MdmlError('shape', `attention: ${hq} query heads do not divide into ${hk} groups`)
    k = repeatKv(k, hq / hk)
    v = repeatKv(v, hq / hk)
  }
  const e = q.shape[q.shape.length - 1] ?? 1
  const l = q.shape[q.shape.length - 2] ?? 1
  const s = k.shape[k.shape.length - 2] ?? 1
  let scores = mul(matmul(q, swapaxes(k, -1, -2)), options.scale ?? 1 / Math.sqrt(e))
  if (options.softCap !== undefined) scores = mul(tanh(div(scores, options.softCap)), options.softCap)
  if (options.bias !== undefined) scores = add(scores, fl(options.bias))
  let allowed: Value | undefined
  if (options.causal === true || options.window !== undefined) allowed = causalMask(l, s, options.window)
  if (options.mask !== undefined) {
    const m = asTensor(options.mask)
    if (m.dtype === 'bool') allowed = allowed === undefined ? m : logicalAnd(allowed, m)
    else scores = add(scores, m)
  }
  let weights = allowed === undefined ? softmaxLast(scores) : maskedSoftmax(scores, allowed)
  if (options.dropout !== undefined && options.dropout > 0 && options.training !== false) {
    weights = dropout(weights, options.dropout, {
      training: true,
      ...(options.key === undefined ? {} : { key: options.key }),
    })
  }
  return { output: matmul(weights, v), weights }
}

const softmaxLast = (z: Value): Value => {
  const top = max(z, -1, true)
  const shift = where(isfinite(top), top, 0)
  const e = exp(sub(z, shift))
  return div(e, sum(e, -1, true))
}

/**
 * Scaled dot-product attention (torch.nn.functional.scaled_dot_product_attention): the output of {@link attention}.
 *
 * @param query - `[..., L, E]`.
 * @param key - `[..., S, E]`.
 * @param value - `[..., S, Ev]`.
 * @param options - Mask (true = attend, or additive), bias, causal, window, scale, softCap, dropout and enableGqa.
 * @returns `[..., L, Ev]`.
 * @concept attention
 * @example
 * ```ts
 * import { scaledDotProductAttention } from 'mdml-engine/nn'
 * scaledDotProductAttention([[[1, 0], [0, 1]]], [[[1, 0], [0, 1]]], [[[1], [2]]]).shape // => [1, 2, 1]
 * ```
 */
export function scaledDotProductAttention(
  query: TensorLike,
  key: TensorLike,
  value: TensorLike,
  options: AttentionOptions = {},
): Value {
  return attention(query, key, value, options).output
}

/**
 * Split the feature axis into heads: `[B, L, H D]` to `[B, H, L, D]`.
 *
 * @param x - `[B, L, H D]`.
 * @param numHeads - H.
 * @returns `[B, H, L, D]`.
 * @example
 * ```ts
 * import { splitHeads } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * splitHeads(zeros([2, 5, 12]), 3).shape // => [2, 3, 5, 4]
 * ```
 */
export function splitHeads(x: TensorLike, numHeads: number): Value {
  const v = fl(x)
  const [b = 1, l = 1, e = 1] = v.shape
  if (e % numHeads !== 0) throw new MdmlError('shape', `splitHeads: ${e} does not divide into ${numHeads} heads`)
  return swapaxes(reshape(v, [b, l, numHeads, e / numHeads]), 1, 2)
}

/**
 * Merge heads back: `[B, H, L, D]` to `[B, L, H D]` (the inverse of {@link splitHeads}).
 *
 * @param x - `[B, H, L, D]`.
 * @returns `[B, L, H D]`.
 * @example
 * ```ts
 * import { mergeHeads } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * mergeHeads(zeros([2, 3, 5, 4])).shape // => [2, 5, 12]
 * ```
 */
export function mergeHeads(x: TensorLike): Value {
  const v = fl(x)
  const [b = 1, h = 1, l = 1, d = 1] = v.shape
  return reshape(swapaxes(v, 1, 2), [b, l, h * d])
}

/** The projection parameters of {@link multiHeadAttention} (PyTorch's nn.MultiheadAttention layout). */
export interface MultiHeadAttentionParams {
  /** Packed query, key and value projections `[3E, E]` (when the key and value widths equal E). */
  readonly inProjWeight?: Value
  /** Separate projections (when the key or value width differs from E). */
  readonly qProjWeight?: Value
  readonly kProjWeight?: Value
  readonly vProjWeight?: Value
  /** Packed biases `[3E]`. */
  readonly inProjBias?: Value
  readonly outProjWeight: Value
  readonly outProjBias?: Value
  /** Learnt key and value rows appended to the sequence (`add_bias_kv`), `[1, 1, E]`. */
  readonly biasK?: Value
  readonly biasV?: Value
}

/** Options of {@link multiHeadAttention}. */
export interface MultiHeadAttentionOptions {
  readonly numHeads: number
  /** `[B, S]`, true for padding positions to ignore (PyTorch's key_padding_mask). */
  readonly keyPaddingMask?: TensorLike
  /** `[L, S]` or `[B H, L, S]`: boolean true = not allowed (PyTorch's nn.MultiheadAttention convention), or additive. */
  readonly attnMask?: TensorLike
  readonly causal?: boolean
  /** Append a zero key and value (`add_zero_attn`). */
  readonly addZeroAttn?: boolean
  /** Average the returned weights over heads (default true). */
  readonly averageAttnWeights?: boolean
  readonly dropout?: number
  readonly key?: Key
  readonly training?: boolean
}

/**
 * Multi-head attention (torch.nn.functional.multi_head_attention_forward, batch first): project the query, key and
 * value, attend in each head, merge and project out. Takes PyTorch's packed `in_proj` layout, key padding masks,
 * attention masks (boolean true = masked, as nn.MultiheadAttention), causal masking, `add_bias_kv` and
 * `add_zero_attn`, and returns the weights averaged over heads (or per head).
 *
 * @param query - `[B, L, E]`.
 * @param key - `[B, S, kdim]`.
 * @param value - `[B, S, vdim]`.
 * @param params - The projections.
 * @param options - Heads, masks, causal, addZeroAttn, averageAttnWeights, dropout.
 * @returns The output `[B, L, E]` and weights `[B, L, S]` (or `[B, H, L, S]`).
 * @concept attention
 * @example
 * ```ts
 * import { multiHeadAttention } from 'mdml-engine/nn'
 * import { eye, zeros } from 'mdml-engine/tensor'
 * const x = [[[1, 0], [0, 1]]]
 * const p = { inProjWeight: zeros([6, 2]), outProjWeight: eye(2) }
 * multiHeadAttention(x, x, x, p, { numHeads: 2 }).output.shape // => [1, 2, 2]
 * ```
 */
export function multiHeadAttention(
  query: TensorLike,
  key: TensorLike,
  value: TensorLike,
  params: MultiHeadAttentionParams,
  options: MultiHeadAttentionOptions,
): AttentionResult {
  const [q0, k0, v0] = promoteFloat(query, key, value) as [Value, Value, Value]
  const e = params.outProjWeight.shape[0] ?? 1
  const h = options.numHeads
  const [b = 1, l = 1] = q0.shape
  const proj = (x: Value, i: number, separate: Value | undefined): Value => {
    if (separate === undefined && params.inProjWeight === undefined)
      throw new MdmlError('value', 'multiHeadAttention: need inProjWeight or separate projection weights')
    const w = separate ?? sliceAxis(params.inProjWeight as Value, 0, i * e, (i + 1) * e)
    const y = matmul(x, swapaxes(w, 0, 1))
    return params.inProjBias === undefined ? y : add(y, sliceAxis(params.inProjBias, 0, i * e, (i + 1) * e))
  }
  const q = proj(q0, 0, params.qProjWeight)
  let k = proj(k0, 1, params.kProjWeight)
  let v = proj(v0, 2, params.vProjWeight)
  let padding = options.keyPaddingMask === undefined ? undefined : asTensor(options.keyPaddingMask, 'bool')
  let attnMask = options.attnMask === undefined ? undefined : asTensor(options.attnMask)
  const extend = (n: number): void => {
    // Appended key positions are never padding and never masked.
    if (padding !== undefined) padding = cast(concatLast(cast(padding, 'float64'), n), 'bool')
    if (attnMask !== undefined)
      attnMask =
        attnMask.dtype === 'bool' ? cast(concatLast(cast(attnMask, 'float64'), n), 'bool') : concatLast(attnMask, n)
  }
  if (params.biasK !== undefined && params.biasV !== undefined) {
    k = concatRows(k, reshape(params.biasK, [1, 1, e]), b)
    v = concatRows(v, reshape(params.biasV, [1, 1, e]), b)
    extend(1)
  }
  if (options.addZeroAttn === true) {
    k = concat([k, zeros([b, 1, e])], 1)
    v = concat([v, zeros([b, 1, e])], 1)
    extend(1)
  }
  const s = k.shape[1] ?? 1
  // Allowed positions: not padding, not masked.
  let allowed: Value | undefined
  let additive: Value | undefined
  if (attnMask !== undefined) {
    const m = attnMask.shape.length === 3 ? reshape(attnMask, [b, h, l, s]) : attnMask
    if (attnMask.dtype === 'bool') allowed = notMask(m)
    else additive = m
  }
  if (padding !== undefined) {
    const keep = reshape(notMask(padding), [b, 1, 1, s])
    allowed = allowed === undefined ? keep : logicalAnd(allowed, keep)
  }
  const r = attention(splitHeads(q, h), splitHeads(k, h), splitHeads(v, h), {
    ...(allowed === undefined ? {} : { mask: allowed }),
    ...(additive === undefined ? {} : { bias: additive }),
    ...(options.causal === true ? { causal: true } : {}),
    ...(options.dropout === undefined ? {} : { dropout: options.dropout }),
    ...(options.key === undefined ? {} : { key: options.key }),
    ...(options.training === undefined ? {} : { training: options.training }),
  })
  let out = matmul(mergeHeads(r.output), swapaxes(params.outProjWeight, 0, 1))
  if (params.outProjBias !== undefined) out = add(out, params.outProjBias)
  return { output: out, weights: (options.averageAttnWeights ?? true) ? mean(r.weights, 1) : r.weights }
}

const notMask = (m: Value): Value => equal(cast(m, 'float64'), 0)

/** Append n zero columns to the last axis of a mask. */
const concatLast = (m: Value, n: number): Value => concat([m, zeros([...m.shape.slice(0, -1), n])], m.shape.length - 1)

/** Append one row (shared by the batch) along the sequence axis. */
const concatRows = (x: Value, row: Value, b: number): Value => concat([x, b === 1 ? row : repeat(row, b, 0)], 1)

/** Options of {@link linearAttention}. */
export interface LinearAttentionOptions {
  readonly causal?: boolean
  /** The positive feature map (default elu(x) + 1). */
  readonly featureMap?: (x: Value) => Value
  /** Added to the normaliser (default 1e-6). */
  readonly eps?: number
}

/**
 * Linear attention (Katharopoulos et al. 2020): softmax(q k^T) replaced by phi(q) phi(k)^T with a positive feature map,
 * so attention costs O(L) by associativity: out_i = phi(q_i)^T sum_j phi(k_j) v_j^T / (phi(q_i)^T sum_j phi(k_j)). The
 * causal form uses prefix sums (a recurrent network in disguise).
 *
 * @param query - `[..., L, E]`.
 * @param key - `[..., S, E]` (S = L when causal).
 * @param value - `[..., S, Ev]`.
 * @param options - Causal, feature map and eps.
 * @returns `[..., L, Ev]`.
 * @reference Katharopoulos, Vyas, Pappas and Fleuret (2020), Transformers are RNNs.
 * @example
 * ```ts
 * import { linearAttention } from 'mdml-engine/nn'
 * linearAttention([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1], [2]], { causal: true }).shape // => [2, 1]
 * ```
 */
export function linearAttention(
  query: TensorLike,
  key: TensorLike,
  value: TensorLike,
  options: LinearAttentionOptions = {},
): Value {
  const [q, k, v] = promoteFloat(query, key, value) as [Value, Value, Value]
  const phi = options.featureMap ?? ((x: Value) => add(elu(x), 1))
  const eps = options.eps ?? 1e-6
  const fq = phi(q)
  const fk = phi(k)
  if (options.causal !== true) {
    const kv = matmul(swapaxes(fk, -1, -2), v)
    const z = sum(fk, -2, true)
    return div(matmul(fq, kv), add(sum(mul(fq, z), -1, true), eps))
  }
  // Prefix sums of the outer products phi(k_j) v_j^T: [..., L, E, Ev].
  const outer = mul(expandDims(fk, -1), expandDims(v, -2))
  const kv = cumsum(outer, outer.shape.length - 3)
  const z = cumsum(fk, fk.shape.length - 2)
  const num = sum(mul(expandDims(fq, -1), kv), -2)
  return div(num, add(sum(mul(fq, z), -1, true), eps))
}

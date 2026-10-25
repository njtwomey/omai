/**
 * Layers for sets, where the output must not depend on the order of the elements: masked pooling over padded sets (Deep
 * Sets) and the Set Transformer's attention blocks, `mab`, `sab`, the induced `isab` that attends through a few learnt
 * points, and `pma`, which pools a set into k vectors by attention.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  cast,
  div,
  equal,
  expandDims,
  logicalNot,
  matmul,
  max,
  maximum,
  mul,
  promoteFloat,
  relu,
  sum,
  swapaxes,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { type MultiHeadAttentionParams, multiHeadAttention } from './attention.ts'
import { layerNorm } from './normalisation.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** How {@link setPool} reduces a set. */
export type SetReduction = 'sum' | 'mean' | 'max'

/**
 * Pool a batch of padded sets over the set axis with a validity mask: sum, mean or max of the valid elements, zeros for
 * an empty set (Deep Sets' permutation-invariant pooling).
 *
 * @param x - `[B, N, F]`.
 * @param mask - `[B, N]`, true for valid elements (default all valid).
 * @param reduce - `sum`, `mean` (default) or `max`.
 * @returns `[B, F]`, differentiable in x.
 * @concept set-functions
 * @reference Zaheer et al. (2017), Deep sets.
 * @example
 * ```ts
 * import { setPool } from 'whitebox-engine/nn'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(setPool([[[1], [3], [9]]], tensor([[true, true, false]], { dtype: 'bool' }), 'mean')) // => [[2]]
 * ```
 */
export function setPool(x: TensorLike, mask?: TensorLike, reduce: SetReduction = 'mean'): Value {
  const xv = fl(x)
  if (xv.shape.length !== 3) throw new WhiteboxError('shape', 'setPool: input must be [B, N, F]')
  if (mask === undefined) {
    if (reduce === 'max') return max(xv, 1)
    return reduce === 'sum' ? sum(xv, 1) : div(sum(xv, 1), xv.shape[1] ?? 1)
  }
  const m = expandDims(asTensor(mask, 'bool'), 2) // [B, N, 1]
  const count = sum(cast(m, 'float64'), 1) // [B, 1]
  const empty = equal(count, 0)
  if (reduce === 'max') {
    const top = max(where(m, xv, Number.NEGATIVE_INFINITY), 1)
    return where(empty, 0, top)
  }
  const total = sum(where(m, xv, 0), 1)
  return reduce === 'sum' ? total : div(total, maximum(count, 1))
}

/** The parameters of a multihead attention block (Set Transformer's MAB). */
export interface MabParams {
  /** The attention's projections. */
  readonly attention: MultiHeadAttentionParams
  /** The row-wise feed-forward layer `[D, D]` and bias. */
  readonly ffWeight: TensorLike
  readonly ffBias?: TensorLike
  /** Layer normalisations after the attention and after the feed-forward (absent: none). */
  readonly norm0?: { readonly weight: TensorLike; readonly bias: TensorLike }
  readonly norm1?: { readonly weight: TensorLike; readonly bias: TensorLike }
}

/** Options of the set attention blocks. */
export interface SetAttentionOptions {
  readonly numHeads: number
  /** `[B, M]`, true for the valid elements of the attended set. */
  readonly mask?: TensorLike
}

const norm = (h: Value, p: { readonly weight: TensorLike; readonly bias: TensorLike } | undefined): Value =>
  p === undefined ? h : layerNorm(h, h.shape[h.shape.length - 1] ?? 1, { weight: p.weight, bias: p.bias })

/**
 * The multihead attention block of the Set Transformer: H = LN(X + Multihead(X, Y, Y)), MAB(X, Y) = LN(H + rFF(H)) with
 * rFF a row-wise ReLU layer; Y's padding is masked.
 *
 * @param x - Queries `[B, N, D]`.
 * @param y - The attended set `[B, M, D]`.
 * @param params - Attention, feed-forward and normalisation parameters.
 * @param options - Heads and Y's validity mask.
 * @returns `[B, N, D]`.
 * @concept set-functions
 * @reference Lee et al. (2019), Set transformer: a framework for attention-based permutation-invariant neural networks.
 * @example
 * ```ts
 * import { mab } from 'whitebox-engine/nn'
 * import { eye, zeros } from 'whitebox-engine/tensor'
 * mab(zeros([1, 3, 2]), zeros([1, 5, 2]), { attention: { inProjWeight: zeros([6, 2]), outProjWeight: eye(2) }, ffWeight: zeros([2, 2]) }, { numHeads: 1 }).shape // => [1, 3, 2]
 * ```
 */
export function mab(x: TensorLike, y: TensorLike, params: MabParams, options: SetAttentionOptions): Value {
  const [xv, yv] = promoteFloat(x, y) as [Value, Value]
  const padding = options.mask === undefined ? undefined : logicalNot(asTensor(options.mask, 'bool'))
  const a = multiHeadAttention(xv, yv, yv, params.attention, {
    numHeads: options.numHeads,
    ...(padding === undefined ? {} : { keyPaddingMask: padding }),
  }).output
  const h = norm(add(xv, a), params.norm0)
  let ff = matmul(h, swapaxes(fl(params.ffWeight), 0, 1))
  if (params.ffBias !== undefined) ff = add(ff, fl(params.ffBias))
  return norm(add(h, relu(ff)), params.norm1)
}

/**
 * Set attention block, SAB(X) = MAB(X, X): self-attention among the elements of a set (permutation equivariant).
 *
 * @param x - `[B, N, D]`.
 * @param params - The block's parameters.
 * @param options - Heads and the validity mask.
 * @returns `[B, N, D]`.
 * @example
 * ```ts
 * import { sab } from 'whitebox-engine/nn'
 * import { eye, zeros } from 'whitebox-engine/tensor'
 * sab(zeros([2, 3, 2]), { attention: { inProjWeight: zeros([6, 2]), outProjWeight: eye(2) }, ffWeight: zeros([2, 2]) }, { numHeads: 2 }).shape // => [2, 3, 2]
 * ```
 */
export function sab(x: TensorLike, params: MabParams, options: SetAttentionOptions): Value {
  return mab(x, x, params, options)
}

const tile = (seeds: TensorLike, b: number): Value => {
  const s = fl(seeds)
  return broadcastTo(expandDims(s, 0), [b, ...s.shape])
}

/**
 * Induced set attention block, ISAB_m(X) = MAB(X, MAB(I, X)) with m learnt inducing points I: attention in O(n m)
 * instead of O(n^2), still permutation equivariant.
 *
 * @param x - `[B, N, D]`.
 * @param inducing - The inducing points `[m, D]`.
 * @param first - The block attending from the inducing points to X.
 * @param second - The block attending from X to the induced summary.
 * @param options - Heads and X's validity mask.
 * @returns `[B, N, D]`.
 * @example
 * ```ts
 * import { isab } from 'whitebox-engine/nn'
 * import { eye, zeros } from 'whitebox-engine/tensor'
 * const p = { attention: { inProjWeight: zeros([6, 2]), outProjWeight: eye(2) }, ffWeight: zeros([2, 2]) }
 * isab(zeros([1, 5, 2]), zeros([2, 2]), p, p, { numHeads: 1 }).shape // => [1, 5, 2]
 * ```
 */
export function isab(
  x: TensorLike,
  inducing: TensorLike,
  first: MabParams,
  second: MabParams,
  options: SetAttentionOptions,
): Value {
  const xv = fl(x)
  const h = mab(tile(inducing, xv.shape[0] ?? 1), xv, first, options)
  return mab(xv, h, second, { numHeads: options.numHeads })
}

/**
 * Pooling by multihead attention, PMA_k(Z) = MAB(S, rFF(Z)) with k learnt seed vectors S: a permutation-invariant
 * pooling of a set into k vectors (rFF optional, a row-wise ReLU layer).
 *
 * @param x - `[B, N, D]`.
 * @param seeds - `[k, D]`.
 * @param params - The block's parameters.
 * @param options - Heads, the validity mask, and the optional row-wise layer before pooling.
 * @returns `[B, k, D]`.
 * @example
 * ```ts
 * import { pma } from 'whitebox-engine/nn'
 * import { eye, zeros } from 'whitebox-engine/tensor'
 * pma(zeros([2, 5, 2]), zeros([1, 2]), { attention: { inProjWeight: zeros([6, 2]), outProjWeight: eye(2) }, ffWeight: zeros([2, 2]) }, { numHeads: 1 }).shape // => [2, 1, 2]
 * ```
 */
export function pma(
  x: TensorLike,
  seeds: TensorLike,
  params: MabParams,
  options: SetAttentionOptions & { readonly ffWeight?: TensorLike; readonly ffBias?: TensorLike },
): Value {
  let z = fl(x)
  if (options.ffWeight !== undefined) {
    z = matmul(z, swapaxes(fl(options.ffWeight), 0, 1))
    if (options.ffBias !== undefined) z = add(z, fl(options.ffBias))
    z = relu(z)
  }
  return mab(tile(seeds, z.shape[0] ?? 1), z, params, {
    numHeads: options.numHeads,
    ...(options.mask === undefined ? {} : { mask: options.mask }),
  })
}

/**
 * Zero the invalid elements of padded sets (the rows of padding that equivariant blocks still compute).
 *
 * @param x - `[B, N, F]`.
 * @param mask - `[B, N]`, true for valid elements.
 * @returns x with padded rows zeroed.
 * @example
 * ```ts
 * import { maskSet } from 'whitebox-engine/nn'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(maskSet([[[1], [2]]], tensor([[true, false]], { dtype: 'bool' }))) // => [[[1], [0]]]
 * ```
 */
export function maskSet(x: TensorLike, mask: TensorLike): Value {
  return mul(fl(x), expandDims(cast(asTensor(mask, 'bool'), 'float64'), 2))
}

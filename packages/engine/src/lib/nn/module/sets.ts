/**
 * The Set Transformer's blocks as modules, for inputs that are sets of vectors with optional padding masks:
 * `SetAttentionBlock` (self-attention among elements), `InducedSetAttentionBlock` (attention through a few learnt
 * inducing points) and `PoolingByMultiheadAttention`, which pools a set into k vectors.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { asTensor, full, isValue, type TensorLike, type Value, zeros } from '../../tensor/index.ts'
import { isab, type MabParams, mab, pma } from '../functional/sets.ts'
import { torchDefaultUniform, xavierUniform } from '../init.ts'
import { initKey, Module, type Params } from './module.ts'

/** The input of a set block: `[B, N, D]` sets, optionally with a validity mask `[B, N]` (true = valid). */
export type SetInput = Value | { readonly x: TensorLike; readonly mask?: TensorLike }

/** Fresh parameters of one multihead attention block of width d (PyTorch's attention layout plus `ff`, `norm0/1`). */
function mabInit(key: Key, d: number, layerNorm: boolean): Params {
  return {
    attn: {
      in_proj_weight: xavierUniform(foldIn(key, 0), [3 * d, d]),
      in_proj_bias: zeros([3 * d]),
      out_proj: { weight: torchDefaultUniform(foldIn(key, 1), [d, d], d), bias: zeros([d]) },
    },
    ff: { weight: torchDefaultUniform(foldIn(key, 2), [d, d], d), bias: torchDefaultUniform(foldIn(key, 3), [d], d) },
    ...(layerNorm
      ? { norm0: { weight: full([d], 1), bias: zeros([d]) }, norm1: { weight: full([d], 1), bias: zeros([d]) } }
      : {}),
  }
}

/** A block's parameter tree in the functional form's layout. */
function mabParams(p: Params): MabParams {
  const attn = p['attn'] as Params
  const out = attn['out_proj'] as Params
  const ff = p['ff'] as Params
  const norm = (name: string): { weight: Value; bias: Value } | undefined => {
    const q = p[name] as Params | undefined
    return q === undefined ? undefined : { weight: q['weight'] as Value, bias: q['bias'] as Value }
  }
  const norm0 = norm('norm0')
  const norm1 = norm('norm1')
  return {
    attention: {
      inProjWeight: attn['in_proj_weight'] as Value,
      inProjBias: attn['in_proj_bias'] as Value,
      outProjWeight: out['weight'] as Value,
      outProjBias: out['bias'] as Value,
    },
    ffWeight: ff['weight'] as Value,
    ffBias: ff['bias'] as Value,
    ...(norm0 === undefined ? {} : { norm0 }),
    ...(norm1 === undefined ? {} : { norm1 }),
  }
}

/** Options of the set blocks. */
export interface SetBlockOptions {
  /** Layer normalisation inside the blocks (default true). */
  readonly layerNorm?: boolean
  readonly key?: Key
}

/** The shared input handling of the set blocks. */
abstract class SetBlock extends Module<SetInput, Value> {
  readonly dim: number
  readonly numHeads: number
  readonly layerNorm: boolean

  protected constructor(dim: number, numHeads: number, options: SetBlockOptions) {
    super()
    if (dim % numHeads !== 0) throw new MdmlError('value', `set block: ${dim} does not divide into ${numHeads} heads`)
    this.dim = dim
    this.numHeads = numHeads
    this.layerNorm = options.layerNorm ?? true
  }

  protected override prepare(x: unknown): unknown {
    return isValue(x) || Array.isArray(x) ? asTensor(x as TensorLike) : x
  }

  protected unpack(x: SetInput): { x: Value; mask?: TensorLike } {
    if (isValue(x)) return { x }
    return { x: asTensor(x.x), ...(x.mask === undefined ? {} : { mask: x.mask }) }
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, heads=${this.numHeads}, ln=${this.layerNorm}`
  }
}

/**
 * Set attention block (Set Transformer's SAB): self-attention among the elements of each set, permutation
 * equivariant.
 *
 * @concept set-functions
 * @example
 * ```ts
 * import { SetAttentionBlock } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new SetAttentionBlock(4, 2, { key: key(0) }).forward(zeros([2, 5, 4])).shape // => [2, 5, 4]
 * ```
 */
export class SetAttentionBlock extends SetBlock {
  readonly kind = 'SetAttentionBlock'

  constructor(dim: number, numHeads: number, options: SetBlockOptions = {}) {
    super(dim, numHeads, options)
    this.params = this.initParams(initKey('SetAttentionBlock', options.key))
  }

  override initParams(key: Key): Params {
    return { mab: mabInit(key, this.dim, this.layerNorm) }
  }

  apply(params: Params, input: SetInput): Value {
    const { x, mask } = this.unpack(input)
    return mab(x, x, mabParams(params['mab'] as Params), {
      numHeads: this.numHeads,
      ...(mask === undefined ? {} : { mask }),
    })
  }
}

/**
 * Induced set attention block (Set Transformer's ISAB): attention through m learnt inducing points, O(n m).
 *
 * @concept set-functions
 * @example
 * ```ts
 * import { InducedSetAttentionBlock } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new InducedSetAttentionBlock(4, 2, 3, { key: key(0) }).forward(zeros([2, 6, 4])).shape // => [2, 6, 4]
 * ```
 */
export class InducedSetAttentionBlock extends SetBlock {
  readonly kind = 'InducedSetAttentionBlock'
  readonly numInducing: number

  constructor(dim: number, numHeads: number, numInducing: number, options: SetBlockOptions = {}) {
    super(dim, numHeads, options)
    this.numInducing = numInducing
    this.params = this.initParams(initKey('InducedSetAttentionBlock', options.key))
  }

  override initParams(key: Key): Params {
    return {
      inducing: xavierUniform(foldIn(key, 0), [this.numInducing, this.dim]),
      mab0: mabInit(foldIn(key, 1), this.dim, this.layerNorm),
      mab1: mabInit(foldIn(key, 2), this.dim, this.layerNorm),
    }
  }

  apply(params: Params, input: SetInput): Value {
    const { x, mask } = this.unpack(input)
    return isab(
      x,
      params['inducing'] as Value,
      mabParams(params['mab0'] as Params),
      mabParams(params['mab1'] as Params),
      {
        numHeads: this.numHeads,
        ...(mask === undefined ? {} : { mask }),
      },
    )
  }

  override extraRepr(): string {
    return `${super.extraRepr()}, inducing=${this.numInducing}`
  }
}

/**
 * Pooling by multihead attention (Set Transformer's PMA): k learnt seeds attend to the set (after a row-wise ReLU
 * layer `ff`), a permutation-invariant pooling to `[B, k, D]`.
 *
 * @concept set-functions
 * @example
 * ```ts
 * import { PoolingByMultiheadAttention } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * new PoolingByMultiheadAttention(4, 2, 1, { key: key(0) }).forward(zeros([2, 6, 4])).shape // => [2, 1, 4]
 * ```
 */
export class PoolingByMultiheadAttention extends SetBlock {
  readonly kind = 'PoolingByMultiheadAttention'
  readonly numSeeds: number

  constructor(dim: number, numHeads: number, numSeeds: number, options: SetBlockOptions = {}) {
    super(dim, numHeads, options)
    this.numSeeds = numSeeds
    this.params = this.initParams(initKey('PoolingByMultiheadAttention', options.key))
  }

  override initParams(key: Key): Params {
    const d = this.dim
    return {
      seeds: xavierUniform(foldIn(key, 0), [this.numSeeds, d]),
      ff: { weight: torchDefaultUniform(foldIn(key, 1), [d, d], d), bias: torchDefaultUniform(foldIn(key, 2), [d], d) },
      mab: mabInit(foldIn(key, 3), d, this.layerNorm),
    }
  }

  apply(params: Params, input: SetInput): Value {
    const { x, mask } = this.unpack(input)
    const ff = params['ff'] as Params
    return pma(x, params['seeds'] as Value, mabParams(params['mab'] as Params), {
      numHeads: this.numHeads,
      ffWeight: ff['weight'] as Value,
      ffBias: ff['bias'] as Value,
      ...(mask === undefined ? {} : { mask }),
    })
  }

  override outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[0] ?? 1, this.numSeeds, this.dim]
  }

  override extraRepr(): string {
    return `${super.extraRepr()}, seeds=${this.numSeeds}`
  }
}

import { foldIn, type Key } from '../../random/index.ts'
import { add, type Value } from '../../tensor/index.ts'
import { applyRotary, learnedPositions, rotaryAngles, sinusoidalPositions } from '../functional/positional.ts'
import { normalInit } from '../init.ts'
import { type ApplyContext, initKey, Module, type Params } from './module.ts'
import { Dropout } from './regularisation.ts'

/** Options of the additive position encodings. */
export interface PositionOptions {
  /** Dropout after adding the positions (default 0). */
  readonly dropout?: number
  /** The first position (default 0; the cache length when decoding step by step). */
  readonly offset?: number
}

/**
 * Add sinusoidal position encodings to `[B, L, D]` embeddings (the original Transformer's), then dropout.
 *
 * @concept positional-encoding
 * @example
 * ```ts
 * import { SinusoidalPositionalEncoding } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new SinusoidalPositionalEncoding(4).eval().forward(zeros([2, 3, 4])).shape // => [2, 3, 4]
 * ```
 */
export class SinusoidalPositionalEncoding extends Module {
  readonly kind = 'SinusoidalPositionalEncoding'
  readonly dim: number
  readonly base: number
  readonly offset: number
  private readonly drop: Dropout

  constructor(dim: number, options: PositionOptions & { readonly base?: number } = {}) {
    super()
    this.dim = dim
    this.base = options.base ?? 10000
    this.offset = options.offset ?? 0
    this.drop = new Dropout(options.dropout ?? 0)
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const l = x.shape[x.shape.length - 2] ?? 0
    return this.drop.apply({}, add(x, sinusoidalPositions(l, this.dim, { base: this.base, offset: this.offset })), ctx)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, base=${this.base}`
  }
}

/**
 * Learned absolute position embeddings added to `[B, L, D]` embeddings (BERT's and GPT-2's), a `weight` table
 * `[numPositions, dim]` initialised N(0, 0.02^2).
 *
 * @example
 * ```ts
 * import { LearnedPositionalEmbedding } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * new LearnedPositionalEmbedding(16, 4, { key: key(0) }).parameterCount() // => 64
 * ```
 */
export class LearnedPositionalEmbedding extends Module {
  readonly kind = 'LearnedPositionalEmbedding'
  readonly numPositions: number
  readonly dim: number
  readonly offset: number
  private readonly drop: Dropout

  constructor(numPositions: number, dim: number, options: PositionOptions & { readonly key?: Key } = {}) {
    super()
    this.numPositions = numPositions
    this.dim = dim
    this.offset = options.offset ?? 0
    this.drop = new Dropout(options.dropout ?? 0)
    this.params = this.initParams(initKey('LearnedPositionalEmbedding', options.key))
  }

  override initParams(key: Key): Params {
    return { weight: normalInit(foldIn(key, 0), [this.numPositions, this.dim], 0, 0.02) }
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const l = x.shape[x.shape.length - 2] ?? 0
    return this.drop.apply({}, add(x, learnedPositions(params['weight'] as Value, l, this.offset)), ctx)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `num_positions=${this.numPositions}, dim=${this.dim}`
  }
}

/**
 * Rotary position embeddings as a module: rotates the last axis of `[..., L, D]` queries or keys by position (half
 * pairs by default, adjacent pairs with `interleaved`), optionally only the first `rotaryDim` features (GPT-J,
 * partial rotary).
 *
 * @concept positional-encoding
 * @example
 * ```ts
 * import { RotaryEmbedding } from 'whitebox-engine/nn'
 * import { zeros } from 'whitebox-engine/tensor'
 * new RotaryEmbedding(8).forward(zeros([1, 2, 5, 8])).shape // => [1, 2, 5, 8]
 * ```
 */
export class RotaryEmbedding extends Module {
  readonly kind = 'RotaryEmbedding'
  readonly rotaryDim: number
  readonly base: number
  readonly interleaved: boolean
  readonly scale: number

  constructor(
    rotaryDim: number,
    options: { readonly base?: number; readonly interleaved?: boolean; readonly scale?: number } = {},
  ) {
    super()
    this.rotaryDim = rotaryDim
    this.base = options.base ?? 10000
    this.interleaved = options.interleaved ?? false
    this.scale = options.scale ?? 1
  }

  /** Rotate with positions starting at `offset`. */
  rotate(x: Value, offset = 0): Value {
    const l = x.shape[x.shape.length - 2] ?? 0
    const angles = rotaryAngles(l, this.rotaryDim, { base: this.base, offset, scale: this.scale })
    return applyRotary(x, angles, this.interleaved)
  }

  apply(_params: Params, x: Value): Value {
    return this.rotate(x)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `dim=${this.rotaryDim}, base=${this.base}${this.interleaved ? ', interleaved' : ''}`
  }
}

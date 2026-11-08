/**
 * The basic layers as modules, torch.nn style: `Linear` and `Bilinear` with PyTorch's initialisation, `Embedding` and
 * `EmbeddingBag` lookup tables, and the shape helpers `Flatten`, `Unflatten` and `Identity`.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import {
  asTensor,
  fromStorage,
  isValue,
  normaliseAxis,
  reshape,
  sizeOf,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { bilinear, embedding, embeddingBag, linear } from '../functional/linear.ts'
import { torchDefaultUniform } from '../init.ts'
import { initKey, Module, type Params } from './module.ts'

/** Options of {@link Linear}. */
export interface LinearOptions {
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * A fully connected layer y = x W^T + b (torch.nn.Linear), initialised as PyTorch does: weight and bias uniform on
 * +-1/sqrt(inFeatures).
 *
 * @concept modules
 * @index module Fully connected layer with PyTorch's initialisation.
 * @example
 * ```ts
 * import { Linear } from 'omai-engine/nn'
 * new Linear(3, 2).parameterCount() // => 8
 * ```
 */
export class Linear extends Module {
  readonly kind = 'Linear'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly bias: boolean

  constructor(inFeatures: number, outFeatures: number, options: LinearOptions = {}) {
    super()
    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('Linear', options.key))
  }

  override initParams(key: Key): Params {
    return {
      weight: torchDefaultUniform(foldIn(key, 0), [this.outFeatures, this.inFeatures], this.inFeatures),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outFeatures], this.inFeatures) } : {}),
    }
  }

  apply(params: Params, x: Value): Value {
    return linear(x, params['weight'] as Value, params['bias'] as Value | undefined)
  }

  outputShape(inputShape: readonly number[]): number[] {
    if (inputShape[inputShape.length - 1] !== this.inFeatures) {
      throw new OmaiError('shape', `Linear: expected ${this.inFeatures} input features, got [${inputShape.join(', ')}]`)
    }
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias}`
  }
}

/**
 * A bilinear layer y_k = x1^T W_k x2 + b_k (torch.nn.Bilinear), weight `[out, in1, in2]` and bias uniform on
 * +-1/sqrt(in1). Its input is the pair `[x1, x2]`.
 *
 * @example
 * ```ts
 * import { Bilinear } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * new Bilinear(2, 3, 4, { key: key(0) }).forward([[[1, 2]], [[1, 2, 3]]]).shape // => [1, 4]
 * ```
 */
export class Bilinear extends Module<readonly [Value, Value]> {
  readonly kind = 'Bilinear'
  readonly in1Features: number
  readonly in2Features: number
  readonly outFeatures: number
  readonly bias: boolean

  constructor(in1Features: number, in2Features: number, outFeatures: number, options: LinearOptions = {}) {
    super()
    this.in1Features = in1Features
    this.in2Features = in2Features
    this.outFeatures = outFeatures
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('Bilinear', options.key))
  }

  override initParams(key: Key): Params {
    const shape = [this.outFeatures, this.in1Features, this.in2Features]
    return {
      weight: torchDefaultUniform(foldIn(key, 0), shape, this.in1Features),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outFeatures], this.in1Features) } : {}),
    }
  }

  protected override prepare(x: unknown): readonly [Value, Value] {
    const [a, b] = x as readonly [TensorLike, TensorLike]
    return [asTensor(a), asTensor(b)]
  }

  apply(params: Params, x: readonly [Value, Value]): Value {
    return bilinear(x[0], x[1], params['weight'] as Value, params['bias'] as Value | undefined)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in1_features=${this.in1Features}, in2_features=${this.in2Features}, out_features=${this.outFeatures}, bias=${this.bias}`
  }
}

/** Options of {@link Embedding}. */
export interface EmbeddingModuleOptions {
  /** An index whose row starts at zero and gets no gradient. */
  readonly paddingIdx?: number
  readonly maxNorm?: number
  readonly normType?: number
  readonly key?: Key
}

/** N(0, 1) rows with the padding row zeroed (PyTorch's embedding initialisation). */
function embeddingInit(key: Key, n: number, dim: number, paddingIdx: number | undefined): Tensor {
  const w = Float64Array.from(normal(key, [n, dim]).data)
  if (paddingIdx !== undefined) w.fill(0, paddingIdx * dim, (paddingIdx + 1) * dim)
  return fromStorage(w, [n, dim], 'float64')
}

const normaliseIndex = (i: number | undefined, n: number): number | undefined =>
  i === undefined ? undefined : i < 0 ? i + n : i

/**
 * A lookup table of embeddings (torch.nn.Embedding): integer indices of any shape to rows of a `[n, dim]` table,
 * initialised N(0, 1) with the padding row at zero.
 *
 * @concept embeddings
 * @index module Lookup table of embeddings from integer indices.
 * @example
 * ```ts
 * import { Embedding } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * new Embedding(10, 3, { key: key(0) }).forward([[1, 2], [3, 4]]).shape // => [2, 2, 3]
 * ```
 */
export class Embedding extends Module {
  readonly kind = 'Embedding'
  readonly numEmbeddings: number
  readonly embeddingDim: number
  readonly paddingIdx: number | undefined
  readonly maxNorm: number | undefined
  readonly normType: number

  constructor(numEmbeddings: number, embeddingDim: number, options: EmbeddingModuleOptions = {}) {
    super()
    this.numEmbeddings = numEmbeddings
    this.embeddingDim = embeddingDim
    this.paddingIdx = normaliseIndex(options.paddingIdx, numEmbeddings)
    this.maxNorm = options.maxNorm
    this.normType = options.normType ?? 2
    this.params = this.initParams(initKey('Embedding', options.key))
  }

  /**
   * An embedding with given weights (torch.nn.Embedding.from_pretrained), frozen unless `freeze` is false.
   *
   * @param weight - The table `[n, dim]`.
   * @param options - Freeze (default true) and the other options.
   * @returns The module.
   */
  static fromPretrained(
    weight: TensorLike,
    options: EmbeddingModuleOptions & { readonly freeze?: boolean } = {},
  ): Embedding {
    const w = asTensor(weight)
    const e = new Embedding(w.shape[0] ?? 0, w.shape[1] ?? 0, options)
    e.setParams({ weight: w })
    return e.requiresGrad(!(options.freeze ?? true))
  }

  override initParams(key: Key): Params {
    return { weight: embeddingInit(key, this.numEmbeddings, this.embeddingDim, this.paddingIdx) }
  }

  protected override prepare(x: unknown): Value {
    return asTensor(x as TensorLike, 'int32')
  }

  apply(params: Params, x: Value): Value {
    return embedding(x, params['weight'] as Value, {
      ...(this.paddingIdx === undefined ? {} : { paddingIdx: this.paddingIdx }),
      ...(this.maxNorm === undefined ? {} : { maxNorm: this.maxNorm, normType: this.normType }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape, this.embeddingDim]
  }

  override extraRepr(): string {
    return `${this.numEmbeddings}, ${this.embeddingDim}${this.paddingIdx === undefined ? '' : `, padding_idx=${this.paddingIdx}`}`
  }
}

/** The input of {@link EmbeddingBag}: 2-D indices, or 1-D indices with offsets and optional per-sample weights. */
export type EmbeddingBagInput =
  | Value
  | { readonly input: TensorLike; readonly offsets?: TensorLike; readonly perSampleWeights?: TensorLike }

/**
 * Bags of embeddings reduced by sum, mean or max (torch.nn.EmbeddingBag), for bag-of-words and sparse-feature models.
 *
 * @example
 * ```ts
 * import { EmbeddingBag } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * new EmbeddingBag(10, 3, { key: key(0) }).forward([[1, 2], [3, 4]]).shape // => [2, 3]
 * ```
 */
export class EmbeddingBag extends Module<EmbeddingBagInput> {
  readonly kind = 'EmbeddingBag'
  readonly numEmbeddings: number
  readonly embeddingDim: number
  readonly mode: 'sum' | 'mean' | 'max'
  readonly paddingIdx: number | undefined
  readonly includeLastOffset: boolean
  readonly maxNorm: number | undefined

  constructor(
    numEmbeddings: number,
    embeddingDim: number,
    options: EmbeddingModuleOptions & {
      readonly mode?: 'sum' | 'mean' | 'max'
      readonly includeLastOffset?: boolean
    } = {},
  ) {
    super()
    this.numEmbeddings = numEmbeddings
    this.embeddingDim = embeddingDim
    this.mode = options.mode ?? 'mean'
    this.paddingIdx = normaliseIndex(options.paddingIdx, numEmbeddings)
    this.includeLastOffset = options.includeLastOffset ?? false
    this.maxNorm = options.maxNorm
    this.params = this.initParams(initKey('EmbeddingBag', options.key))
  }

  override initParams(key: Key): Params {
    return { weight: embeddingInit(key, this.numEmbeddings, this.embeddingDim, this.paddingIdx) }
  }

  protected override prepare(x: unknown): EmbeddingBagInput {
    if (isValue(x) || Array.isArray(x)) return asTensor(x as TensorLike, 'int32')
    const r = x as { input: TensorLike; offsets?: TensorLike; perSampleWeights?: TensorLike }
    return {
      input: asTensor(r.input, 'int32'),
      ...(r.offsets === undefined ? {} : { offsets: asTensor(r.offsets, 'int32') }),
      ...(r.perSampleWeights === undefined ? {} : { perSampleWeights: asTensor(r.perSampleWeights) }),
    }
  }

  apply(params: Params, x: EmbeddingBagInput): Value {
    const r = isValue(x) ? { input: x } : x
    return embeddingBag(r.input, params['weight'] as Value, {
      mode: this.mode,
      includeLastOffset: this.includeLastOffset,
      ...('offsets' in r && r.offsets !== undefined ? { offsets: r.offsets } : {}),
      ...('perSampleWeights' in r && r.perSampleWeights !== undefined ? { perSampleWeights: r.perSampleWeights } : {}),
      ...(this.paddingIdx === undefined ? {} : { paddingIdx: this.paddingIdx }),
      ...(this.maxNorm === undefined ? {} : { maxNorm: this.maxNorm }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[0] ?? 1, this.embeddingDim]
  }

  override extraRepr(): string {
    return `${this.numEmbeddings}, ${this.embeddingDim}, mode='${this.mode}'`
  }
}

/**
 * Flatten a range of axes into one (torch.nn.Flatten, default axes 1 to the last).
 *
 * @example
 * ```ts
 * import { Flatten } from 'omai-engine/nn'
 * new Flatten().outputShape([2, 3, 4]) // => [2, 12]
 * ```
 */
export class Flatten extends Module {
  readonly kind = 'Flatten'
  readonly startDim: number
  readonly endDim: number

  constructor(startDim = 1, endDim = -1) {
    super()
    this.startDim = startDim
    this.endDim = endDim
  }

  apply(_params: Params, x: Value): Value {
    return x.shape.length === 0 ? reshape(x, [1]) : reshape(x, this.outputShape(x.shape))
  }

  outputShape(inputShape: readonly number[]): number[] {
    if (inputShape.length === 0) return [1]
    const s = normaliseAxis(this.startDim, inputShape.length)
    const e = normaliseAxis(this.endDim, inputShape.length)
    return [...inputShape.slice(0, s), sizeOf(inputShape.slice(s, e + 1)), ...inputShape.slice(e + 1)]
  }

  override extraRepr(): string {
    return `start_dim=${this.startDim}, end_dim=${this.endDim}`
  }
}

/**
 * Expand one axis into a shape (torch.nn.Unflatten).
 *
 * @example
 * ```ts
 * import { Unflatten } from 'omai-engine/nn'
 * new Unflatten(1, [3, 4]).outputShape([2, 12]) // => [2, 3, 4]
 * ```
 */
export class Unflatten extends Module {
  readonly kind = 'Unflatten'
  readonly dim: number
  readonly sizes: readonly number[]

  constructor(dim: number, sizes: readonly number[]) {
    super()
    this.dim = dim
    this.sizes = sizes
  }

  apply(_params: Params, x: Value): Value {
    return reshape(x, this.outputShape(x.shape))
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = normaliseAxis(this.dim, inputShape.length)
    const n = inputShape[d] ?? 0
    const known = this.sizes.filter((s) => s !== -1).reduce((a, b) => a * b, 1)
    const sizes = this.sizes.map((s) => (s === -1 ? n / known : s))
    if (sizeOf(sizes) !== n) throw new OmaiError('shape', `Unflatten: [${this.sizes.join(', ')}] does not fill ${n}`)
    return [...inputShape.slice(0, d), ...sizes, ...inputShape.slice(d + 1)]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, unflattened_size=(${this.sizes.join(', ')})`
  }
}

/**
 * The identity (torch.nn.Identity).
 *
 * @example
 * ```ts
 * import { Identity } from 'omai-engine/nn'
 * new Identity().outputShape([2, 3]) // => [2, 3]
 * ```
 */
export class Identity extends Module {
  readonly kind = 'Identity'

  apply(_params: Params, x: Value): Value {
    return x
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

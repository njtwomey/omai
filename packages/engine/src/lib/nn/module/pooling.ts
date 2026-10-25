/**
 * Pooling layers as modules, torch.nn style: max, average and Lp pooling in one, two and three dimensions, adaptive
 * average and max pooling to a fixed output size, Keras-style global pooling and `MaxUnpool`.
 */

import { asTensor, type TensorLike, type Value } from '../../tensor/index.ts'
import type { SpatialSize } from '../functional/conv.ts'
import {
  type AvgPoolOptions,
  adaptiveAvgPool,
  adaptiveMaxPool,
  avgPool,
  globalAvgPool,
  globalMaxPool,
  lpPool,
  type MaxPoolOptions,
  maxPool,
  maxUnpool,
} from '../functional/pooling.ts'
import { Module, type Params } from './module.ts'

const spread = (v: SpatialSize | undefined, d: number, fallback: readonly number[] | number): number[] =>
  v === undefined
    ? typeof fallback === 'number'
      ? new Array<number>(d).fill(fallback)
      : [...fallback]
    : typeof v === 'number'
      ? new Array<number>(d).fill(v)
      : [...v]

const tuple = (xs: readonly number[]): string => (xs.length === 1 ? `${xs[0]}` : `(${xs.join(', ')})`)

/** The output length of a pooling window along one axis, with ceil mode. */
function pooledSize(n: number, k: number, s: number, p: number, dil: number, ceil: boolean): number {
  const num = n + 2 * p - dil * (k - 1) - 1
  let o = (ceil ? Math.ceil(num / s) : Math.floor(num / s)) + 1
  if (ceil && (o - 1) * s >= n + p) o -= 1
  return o
}

/** A sliding-window pooling over d spatial axes. */
abstract class Pool extends Module {
  readonly kind: string
  readonly kernelSize: readonly number[]
  readonly stride: readonly number[]
  readonly padding: readonly number[]
  readonly dilation: readonly number[]
  readonly ceilMode: boolean

  protected constructor(
    kind: string,
    d: number,
    kernelSize: SpatialSize,
    options: { stride?: SpatialSize; padding?: SpatialSize; dilation?: SpatialSize; ceilMode?: boolean },
  ) {
    super()
    this.kind = kind
    this.kernelSize = spread(kernelSize, d, 1)
    this.stride = spread(options.stride, d, this.kernelSize)
    this.padding = spread(options.padding, d, 0)
    this.dilation = spread(options.dilation, d, 1)
    this.ceilMode = options.ceilMode ?? false
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.kernelSize.length
    const out = inputShape
      .slice(-d)
      .map((n, i) =>
        pooledSize(
          n,
          this.kernelSize[i] ?? 1,
          this.stride[i] ?? 1,
          this.padding[i] ?? 0,
          this.dilation[i] ?? 1,
          this.ceilMode,
        ),
      )
    return [...inputShape.slice(0, -d), ...out]
  }

  override extraRepr(): string {
    return `kernel_size=${tuple(this.kernelSize)}, stride=${tuple(this.stride)}, padding=${tuple(this.padding)}${this.ceilMode ? ', ceil_mode=True' : ''}`
  }
}

/**
 * Max pooling over d spatial axes (shared by {@link MaxPool1d}, {@link MaxPool2d} and {@link MaxPool3d}).
 *
 * @example
 * ```ts
 * import { MaxPoolNd } from 'whitebox-engine/nn'
 * new MaxPoolNd(2, 2).kind // => 'MaxPool2d'
 * ```
 */
export class MaxPoolNd extends Pool {
  constructor(d: number, kernelSize: SpatialSize, options: MaxPoolOptions = {}) {
    super(`MaxPool${d}d`, d, kernelSize, options)
  }

  apply(_params: Params, x: Value): Value {
    return maxPool(x, this.kernelSize, {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      ceilMode: this.ceilMode,
    })
  }
}

/** 1-D max pooling (torch.nn.MaxPool1d). @example new MaxPool1d(2).outputShape([1, 3, 8]) // => [1, 3, 4] */
export class MaxPool1d extends MaxPoolNd {
  constructor(kernelSize: SpatialSize, options: MaxPoolOptions = {}) {
    super(1, kernelSize, options)
  }
}

/**
 * 2-D max pooling (torch.nn.MaxPool2d).
 *
 * @concept pooling
 * @example
 * ```ts
 * import { MaxPool2d } from 'whitebox-engine/nn'
 * new MaxPool2d(3, { stride: 2, padding: 1 }).outputShape([1, 8, 32, 32]) // => [1, 8, 16, 16]
 * ```
 */
export class MaxPool2d extends MaxPoolNd {
  constructor(kernelSize: SpatialSize, options: MaxPoolOptions = {}) {
    super(2, kernelSize, options)
  }
}

/** 3-D max pooling (torch.nn.MaxPool3d). @example new MaxPool3d(2).outputShape([1, 1, 4, 4, 4]) // => [1, 1, 2, 2, 2] */
export class MaxPool3d extends MaxPoolNd {
  constructor(kernelSize: SpatialSize, options: MaxPoolOptions = {}) {
    super(3, kernelSize, options)
  }
}

/**
 * Average pooling over d spatial axes (shared by {@link AvgPool1d}, {@link AvgPool2d} and {@link AvgPool3d}).
 *
 * @example
 * ```ts
 * import { AvgPoolNd } from 'whitebox-engine/nn'
 * new AvgPoolNd(1, 3, { stride: 1 }).outputShape([1, 2, 5]) // => [1, 2, 3]
 * ```
 */
export class AvgPoolNd extends Pool {
  readonly countIncludePad: boolean
  readonly divisorOverride: number | undefined

  constructor(d: number, kernelSize: SpatialSize, options: AvgPoolOptions = {}) {
    super(`AvgPool${d}d`, d, kernelSize, options)
    this.countIncludePad = options.countIncludePad ?? true
    this.divisorOverride = options.divisorOverride
  }

  apply(_params: Params, x: Value): Value {
    return avgPool(x, this.kernelSize, {
      stride: this.stride,
      padding: this.padding,
      ceilMode: this.ceilMode,
      countIncludePad: this.countIncludePad,
      ...(this.divisorOverride === undefined ? {} : { divisorOverride: this.divisorOverride }),
    })
  }
}

/** 1-D average pooling (torch.nn.AvgPool1d). @example new AvgPool1d(2).outputShape([1, 3, 8]) // => [1, 3, 4] */
export class AvgPool1d extends AvgPoolNd {
  constructor(kernelSize: SpatialSize, options: AvgPoolOptions = {}) {
    super(1, kernelSize, options)
  }
}

/** 2-D average pooling (torch.nn.AvgPool2d). @example new AvgPool2d(2).outputShape([1, 3, 8, 8]) // => [1, 3, 4, 4] */
export class AvgPool2d extends AvgPoolNd {
  constructor(kernelSize: SpatialSize, options: AvgPoolOptions = {}) {
    super(2, kernelSize, options)
  }
}

/** 3-D average pooling (torch.nn.AvgPool3d). @example new AvgPool3d(2).outputShape([1, 1, 4, 4, 4]) // => [1, 1, 2, 2, 2] */
export class AvgPool3d extends AvgPoolNd {
  constructor(kernelSize: SpatialSize, options: AvgPoolOptions = {}) {
    super(3, kernelSize, options)
  }
}

/**
 * Lp (power-average) pooling over d spatial axes (shared by {@link LPPool1d}, {@link LPPool2d}, {@link LPPool3d}).
 *
 * @example
 * ```ts
 * import { LPPoolNd } from 'whitebox-engine/nn'
 * new LPPoolNd(2, 2, 2).kind // => 'LPPool2d'
 * ```
 */
export class LPPoolNd extends Pool {
  readonly normType: number

  constructor(
    d: number,
    normType: number,
    kernelSize: SpatialSize,
    options: { readonly stride?: SpatialSize; readonly ceilMode?: boolean } = {},
  ) {
    super(`LPPool${d}d`, d, kernelSize, options)
    this.normType = normType
  }

  apply(_params: Params, x: Value): Value {
    return lpPool(x, this.normType, this.kernelSize, { stride: this.stride, ceilMode: this.ceilMode })
  }

  override extraRepr(): string {
    return `norm_type=${this.normType}, ${super.extraRepr()}`
  }
}

/** 1-D Lp pooling (torch.nn.LPPool1d). @example new LPPool1d(2, 2).outputShape([1, 3, 8]) // => [1, 3, 4] */
export class LPPool1d extends LPPoolNd {
  constructor(normType: number, kernelSize: SpatialSize, options: { stride?: SpatialSize; ceilMode?: boolean } = {}) {
    super(1, normType, kernelSize, options)
  }
}

/** 2-D Lp pooling (torch.nn.LPPool2d). @example new LPPool2d(2, 2).outputShape([1, 3, 8, 8]) // => [1, 3, 4, 4] */
export class LPPool2d extends LPPoolNd {
  constructor(normType: number, kernelSize: SpatialSize, options: { stride?: SpatialSize; ceilMode?: boolean } = {}) {
    super(2, normType, kernelSize, options)
  }
}

/** 3-D Lp pooling (torch.nn.LPPool3d). @example new LPPool3d(2, 2).outputShape([1, 1, 4, 4, 4]) // => [1, 1, 2, 2, 2] */
export class LPPool3d extends LPPoolNd {
  constructor(normType: number, kernelSize: SpatialSize, options: { stride?: SpatialSize; ceilMode?: boolean } = {}) {
    super(3, normType, kernelSize, options)
  }
}

/**
 * Adaptive pooling to a fixed output size over d spatial axes (shared by the adaptive pooling modules).
 *
 * @example
 * ```ts
 * import { AdaptivePoolNd } from 'whitebox-engine/nn'
 * new AdaptivePoolNd('max', 1, 3).outputShape([1, 2, 10]) // => [1, 2, 3]
 * ```
 */
export class AdaptivePoolNd extends Module {
  readonly kind: string
  readonly outputSize: readonly (number | null)[]
  readonly reduce: 'avg' | 'max'

  constructor(reduce: 'avg' | 'max', d: number, outputSize: number | readonly (number | null)[]) {
    super()
    this.kind = `Adaptive${reduce === 'avg' ? 'Avg' : 'Max'}Pool${d}d`
    this.reduce = reduce
    this.outputSize = typeof outputSize === 'number' ? new Array<number>(d).fill(outputSize) : outputSize
  }

  apply(_params: Params, x: Value): Value {
    return this.reduce === 'avg' ? adaptiveAvgPool(x, this.outputSize) : adaptiveMaxPool(x, this.outputSize)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.outputSize.length
    const spatial = inputShape.slice(-d)
    return [...inputShape.slice(0, -d), ...this.outputSize.map((o, i) => o ?? spatial[i] ?? 1)]
  }

  override extraRepr(): string {
    return `output_size=${this.outputSize.length === 1 ? this.outputSize[0] : `(${this.outputSize.join(', ')})`}`
  }
}

/** 1-D adaptive average pooling (torch.nn.AdaptiveAvgPool1d). @example new AdaptiveAvgPool1d(4).outputShape([1, 3, 10]) // => [1, 3, 4] */
export class AdaptiveAvgPool1d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('avg', 1, outputSize)
  }
}

/**
 * 2-D adaptive average pooling (torch.nn.AdaptiveAvgPool2d; `1` gives global average pooling).
 *
 * @example
 * ```ts
 * import { AdaptiveAvgPool2d } from 'whitebox-engine/nn'
 * new AdaptiveAvgPool2d(1).outputShape([2, 64, 7, 7]) // => [2, 64, 1, 1]
 * ```
 */
export class AdaptiveAvgPool2d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('avg', 2, outputSize)
  }
}

/** 3-D adaptive average pooling (torch.nn.AdaptiveAvgPool3d). @example new AdaptiveAvgPool3d(1).outputShape([1, 2, 4, 4, 4]) // => [1, 2, 1, 1, 1] */
export class AdaptiveAvgPool3d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('avg', 3, outputSize)
  }
}

/** 1-D adaptive max pooling (torch.nn.AdaptiveMaxPool1d). @example new AdaptiveMaxPool1d(4).outputShape([1, 3, 10]) // => [1, 3, 4] */
export class AdaptiveMaxPool1d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('max', 1, outputSize)
  }
}

/** 2-D adaptive max pooling (torch.nn.AdaptiveMaxPool2d). @example new AdaptiveMaxPool2d([2, 3]).outputShape([1, 3, 7, 9]) // => [1, 3, 2, 3] */
export class AdaptiveMaxPool2d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('max', 2, outputSize)
  }
}

/** 3-D adaptive max pooling (torch.nn.AdaptiveMaxPool3d). @example new AdaptiveMaxPool3d(2).outputShape([1, 1, 4, 4, 4]) // => [1, 1, 2, 2, 2] */
export class AdaptiveMaxPool3d extends AdaptivePoolNd {
  constructor(outputSize: number | readonly (number | null)[]) {
    super('max', 3, outputSize)
  }
}

/**
 * Global pooling over every spatial axis to `[B, C]` (Keras's GlobalAveragePooling and GlobalMaxPooling).
 *
 * @example
 * ```ts
 * import { GlobalPool } from 'whitebox-engine/nn'
 * new GlobalPool('avg').outputShape([2, 16, 7, 7]) // => [2, 16]
 * ```
 */
export class GlobalPool extends Module {
  readonly kind: string
  readonly reduce: 'avg' | 'max'

  constructor(reduce: 'avg' | 'max' = 'avg') {
    super()
    this.reduce = reduce
    this.kind = reduce === 'avg' ? 'GlobalAvgPool' : 'GlobalMaxPool'
  }

  apply(_params: Params, x: Value): Value {
    return this.reduce === 'avg' ? globalAvgPool(x) : globalMaxPool(x)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return inputShape.slice(0, 2)
  }
}

/**
 * The partial inverse of max pooling over d spatial axes (torch.nn.MaxUnpool1d/2d/3d); its input is the pair
 * `[values, indices]` from `maxPoolWithIndices`.
 *
 * @example
 * ```ts
 * import { MaxUnpool } from 'whitebox-engine/nn'
 * new MaxUnpool(1, 2).outputShape([1, 1, 3]) // => [1, 1, 6]
 * ```
 */
export class MaxUnpool extends Module<readonly [Value, Value]> {
  readonly kind: string
  readonly kernelSize: readonly number[]
  readonly stride: readonly number[]
  readonly padding: readonly number[]

  constructor(
    d: number,
    kernelSize: SpatialSize,
    options: { readonly stride?: SpatialSize; readonly padding?: SpatialSize } = {},
  ) {
    super()
    this.kind = `MaxUnpool${d}d`
    this.kernelSize = spread(kernelSize, d, 1)
    this.stride = spread(options.stride, d, this.kernelSize)
    this.padding = spread(options.padding, d, 0)
  }

  protected override prepare(x: unknown): readonly [Value, Value] {
    const [v, i] = x as readonly [TensorLike, TensorLike]
    return [asTensor(v), asTensor(i, 'int32')]
  }

  apply(_params: Params, x: readonly [Value, Value]): Value {
    return maxUnpool(x[0], x[1], this.kernelSize, { stride: this.stride, padding: this.padding })
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.kernelSize.length
    const out = inputShape
      .slice(-d)
      .map((n, i) => (n - 1) * (this.stride[i] ?? 1) - 2 * (this.padding[i] ?? 0) + (this.kernelSize[i] ?? 1))
    return [...inputShape.slice(0, -d), ...out]
  }
}

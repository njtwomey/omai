/**
 * Group-equivariant convolutions for the rotation group p4 (Cohen and Welling 2016): layers whose outputs rotate
 * predictably when the input image is rotated by quarter turns. The lifting and group convolutions, orientation
 * pooling, the group action on feature maps, module wrappers, and `equivarianceError` and `invarianceError` for
 * checking the property numerically.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  abs,
  add,
  flip,
  item,
  max,
  mean,
  normaliseAxis,
  promoteFloat,
  reshape,
  roll,
  sliceAxis,
  stack,
  sub,
  swapaxes,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { conv2d, type SpatialSize } from './functional/conv.ts'
import { torchDefaultUniform } from './init.ts'
import { initKey, Module, type Params } from './module/module.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/**
 * Rotate the last two axes by k quarter turns counterclockwise (numpy.rot90 on the trailing axes).
 *
 * @param x - `[..., H, W]`.
 * @param k - Quarter turns (any integer; default 1).
 * @returns The rotated value, differentiable.
 * @concept equivariance
 * @example
 * ```ts
 * import { rotate90 } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rotate90([[1, 2], [3, 4]])) // => [[2, 4], [1, 3]]
 * ```
 */
export function rotate90(x: TensorLike, k = 1): Value {
  let v = fl(x)
  const r = ((k % 4) + 4) % 4
  const a = v.shape.length - 2
  if (a < 0) throw new OmaiError('shape', 'rotate90: need at least two axes')
  for (let i = 0; i < r; i++) v = flip(swapaxes(v, a, a + 1), a)
  return v
}

/**
 * The action of a p4 element (k quarter turns) on p4 feature maps `[B, C, 4, H, W]`: rotate each map spatially and
 * shift the orientation axis cyclically, (T f)_s = R f_{s - k}.
 *
 * @param x - `[..., 4, H, W]`.
 * @param k - Quarter turns.
 * @returns The transformed features.
 * @example
 * ```ts
 * import { p4Transform } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * p4Transform(zeros([1, 2, 4, 3, 3]), 1).shape // => [1, 2, 4, 3, 3]
 * ```
 */
export function p4Transform(x: TensorLike, k = 1): Value {
  const v = fl(x)
  return rotate90(roll(v, k, v.shape.length - 3), k)
}

/** Options of the p4 convolutions. */
export interface P4ConvOptions {
  readonly bias?: TensorLike
  readonly stride?: SpatialSize
  readonly padding?: SpatialSize | 'same' | 'valid'
}

const convOptions = (o: P4ConvOptions): Parameters<typeof conv2d>[2] => ({
  ...(o.stride === undefined ? {} : { stride: o.stride }),
  ...(o.padding === undefined ? {} : { padding: o.padding }),
})

const addBias = (y: Value, bias: TensorLike | undefined): Value => {
  if (bias === undefined) return y
  const b = reshape(fl(bias), [1, -1, 1, 1, 1])
  return add(y, b)
}

/**
 * The lifting convolution from images to the rotation group p4 (Cohen and Welling 2016): correlate with the filter
 * rotated by 0, 90, 180 and 270 degrees, giving one map per orientation. Rotating the input rotates every map and
 * shifts the orientation axis: L(R x) = T L(x).
 *
 * @param input - `[B, C_in, H, W]`.
 * @param weight - `[C_out, C_in, k, k]`.
 * @param options - Bias `[C_out]` (shared across orientations), stride and padding (symmetric, for exact
 *   equivariance on square inputs).
 * @returns `[B, C_out, 4, H', W']`.
 * @concept group-convolution
 * @reference Cohen and Welling (2016), Group equivariant convolutional networks.
 * @example
 * ```ts
 * import { p4LiftingConv } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * p4LiftingConv(zeros([1, 1, 5, 5]), zeros([2, 1, 3, 3]), { padding: 1 }).shape // => [1, 2, 4, 5, 5]
 * ```
 */
export function p4LiftingConv(input: TensorLike, weight: TensorLike, options: P4ConvOptions = {}): Value {
  const w = fl(weight)
  if (w.shape.length !== 4 || w.shape[2] !== w.shape[3])
    throw new OmaiError('shape', 'p4LiftingConv: weight must be [C_out, C_in, k, k]')
  const outs = [0, 1, 2, 3].map((r) => conv2d(input, rotate90(w, r), convOptions(options)))
  return addBias(stack(outs, 2), options.bias)
}

/**
 * The group convolution on p4 (Cohen and Welling 2016): for output orientation r the filter is rotated by r quarter
 * turns and its input-orientation axis shifted by r, so G(T f) = T G(f).
 *
 * @param input - `[B, C_in, 4, H, W]`.
 * @param weight - `[C_out, C_in, 4, k, k]`.
 * @param options - Bias `[C_out]`, stride and padding.
 * @returns `[B, C_out, 4, H', W']`.
 * @concept group-convolution
 * @example
 * ```ts
 * import { p4GroupConv } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * p4GroupConv(zeros([1, 2, 4, 5, 5]), zeros([3, 2, 4, 3, 3])).shape // => [1, 3, 4, 3, 3]
 * ```
 */
export function p4GroupConv(input: TensorLike, weight: TensorLike, options: P4ConvOptions = {}): Value {
  const x = fl(input)
  const w = fl(weight)
  if (x.shape.length !== 5 || x.shape[2] !== 4)
    throw new OmaiError('shape', 'p4GroupConv: input must be [B, C, 4, H, W]')
  if (w.shape.length !== 5 || w.shape[2] !== 4)
    throw new OmaiError('shape', 'p4GroupConv: weight must be [C_out, C_in, 4, k, k]')
  const [b = 1, cin = 1, , h = 1, wd = 1] = x.shape
  const [cout = 1, , , k = 1] = w.shape
  const flat = reshape(x, [b, cin * 4, h, wd])
  const outs = [0, 1, 2, 3].map((r) => {
    // w_r[:, :, s] = R^r w[:, :, s - r]
    const wr = rotate90(roll(w, r, 2), r)
    return conv2d(flat, reshape(wr, [cout, cin * 4, k, k]), convOptions(options))
  })
  return addBias(stack(outs, 2), options.bias)
}

/**
 * Pool over the orientation axis of p4 features (max or mean), leaving rotation-equivariant planar maps:
 * pool(T f) = R pool(f).
 *
 * @param x - `[B, C, 4, H, W]`.
 * @param reduce - `max` (default) or `mean`.
 * @returns `[B, C, H, W]`.
 * @example
 * ```ts
 * import { orientationPool } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * orientationPool(zeros([1, 2, 4, 3, 3])).shape // => [1, 2, 3, 3]
 * ```
 */
export function orientationPool(x: TensorLike, reduce: 'max' | 'mean' = 'max'): Value {
  const v = fl(x)
  const axis = v.shape.length - 3
  return reduce === 'max' ? max(v, axis) : mean(v, axis)
}

/**
 * The largest deviation from equivariance: max |f(T_in x) - T_out f(x)|.
 *
 * @param f - The map.
 * @param x - An input.
 * @param transformIn - The input transformation.
 * @param transformOut - The matching output transformation.
 * @returns The error (zero for an exactly equivariant map, up to rounding).
 * @example
 * ```ts
 * import { equivarianceError, rotate90 } from 'omai-engine/nn'
 * import { mul } from 'omai-engine/tensor'
 * equivarianceError((v) => mul(v, 2), [[1, 2], [3, 4]], rotate90, rotate90) // => 0
 * ```
 */
export function equivarianceError(
  f: (x: Value) => Value,
  x: TensorLike,
  transformIn: (x: Value) => Value,
  transformOut: (y: Value) => Value,
): number {
  const v = fl(x)
  return item(max(abs(sub(f(transformIn(v)), transformOut(f(v))))))
}

/**
 * The largest deviation from invariance: max |f(T x) - f(x)|.
 *
 * @param f - The map.
 * @param x - An input.
 * @param transform - The transformation.
 * @returns The error.
 * @example
 * ```ts
 * import { invarianceError, rotate90 } from 'omai-engine/nn'
 * import { sum } from 'omai-engine/tensor'
 * invarianceError((v) => sum(v), [[1, 2], [3, 4]], rotate90) // => 0
 * ```
 */
export function invarianceError(f: (x: Value) => Value, x: TensorLike, transform: (x: Value) => Value): number {
  const v = fl(x)
  return item(max(abs(sub(f(transform(v)), f(v)))))
}

/** Options of the p4 convolution modules. */
export interface P4ModuleOptions {
  readonly bias?: boolean
  readonly stride?: SpatialSize
  readonly padding?: SpatialSize | 'same' | 'valid'
  readonly key?: Key
}

const outSize = (n: number, k: number, stride: number, padding: number): number =>
  Math.floor((n + 2 * padding - k) / stride) + 1

/** Shared shape logic of the p4 convolution modules. */
abstract class P4ConvBase extends Module {
  readonly inChannels: number
  readonly outChannels: number
  readonly kernelSize: number
  readonly bias: boolean
  readonly stride: SpatialSize | undefined
  readonly padding: SpatialSize | 'same' | 'valid' | undefined
  protected abstract readonly orientationsIn: number

  protected constructor(inChannels: number, outChannels: number, kernelSize: number, options: P4ModuleOptions) {
    super()
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = kernelSize
    this.bias = options.bias ?? true
    this.stride = options.stride
    this.padding = options.padding
  }

  override initParams(key: Key): Params {
    const k = this.kernelSize
    const fanIn = this.inChannels * this.orientationsIn * k * k
    const shape =
      this.orientationsIn === 1
        ? [this.outChannels, this.inChannels, k, k]
        : [this.outChannels, this.inChannels, 4, k, k]
    return {
      weight: torchDefaultUniform(foldIn(key, 0), shape, fanIn),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], fanIn) } : {}),
    }
  }

  protected options(params: Params): P4ConvOptions {
    return {
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      ...(this.stride === undefined ? {} : { stride: this.stride }),
      ...(this.padding === undefined ? {} : { padding: this.padding }),
    }
  }

  outputShape(inputShape: readonly number[]): number[] {
    const [b = 1] = inputShape
    const h = inputShape[inputShape.length - 2] ?? 0
    const w = inputShape[inputShape.length - 1] ?? 0
    const s = typeof this.stride === 'number' ? this.stride : (this.stride?.[0] ?? 1)
    const p =
      this.padding === 'same'
        ? (this.kernelSize - 1) / 2
        : typeof this.padding === 'number'
          ? this.padding
          : Array.isArray(this.padding)
            ? ((this.padding as readonly number[])[0] ?? 0)
            : 0
    return [b, this.outChannels, 4, outSize(h, this.kernelSize, s, p), outSize(w, this.kernelSize, s, p)]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}`
  }
}

/**
 * The p4 lifting convolution as a module: `[B, C_in, H, W]` to `[B, C_out, 4, H', W']`.
 *
 * @concept group-convolution
 * @example
 * ```ts
 * import { P4LiftingConv } from 'omai-engine/nn'
 * new P4LiftingConv(1, 4, 3, { padding: 1 }).outputShape([2, 1, 8, 8]) // => [2, 4, 4, 8, 8]
 * ```
 */
export class P4LiftingConv extends P4ConvBase {
  readonly kind = 'P4LiftingConv'
  protected readonly orientationsIn = 1

  constructor(inChannels: number, outChannels: number, kernelSize: number, options: P4ModuleOptions = {}) {
    super(inChannels, outChannels, kernelSize, options)
    this.params = this.initParams(initKey('P4LiftingConv', options.key))
  }

  apply(params: Params, x: Value): Value {
    return p4LiftingConv(x, params['weight'] as Value, this.options(params))
  }
}

/**
 * The p4 group convolution as a module: `[B, C_in, 4, H, W]` to `[B, C_out, 4, H', W']`.
 *
 * @concept group-convolution
 * @example
 * ```ts
 * import { P4GroupConv } from 'omai-engine/nn'
 * new P4GroupConv(4, 4, 3).outputShape([2, 4, 4, 8, 8]) // => [2, 4, 4, 6, 6]
 * ```
 */
export class P4GroupConv extends P4ConvBase {
  readonly kind = 'P4GroupConv'
  protected readonly orientationsIn = 4

  constructor(inChannels: number, outChannels: number, kernelSize: number, options: P4ModuleOptions = {}) {
    super(inChannels, outChannels, kernelSize, options)
    this.params = this.initParams(initKey('P4GroupConv', options.key))
  }

  apply(params: Params, x: Value): Value {
    return p4GroupConv(x, params['weight'] as Value, this.options(params))
  }
}

/**
 * Orientation pooling as a module: `[B, C, 4, H, W]` to `[B, C, H, W]`.
 *
 * @example
 * ```ts
 * import { OrientationPool } from 'omai-engine/nn'
 * new OrientationPool().outputShape([2, 3, 4, 5, 5]) // => [2, 3, 5, 5]
 * ```
 */
export class OrientationPool extends Module {
  readonly kind = 'OrientationPool'
  readonly reduce: 'max' | 'mean'

  constructor(reduce: 'max' | 'mean' = 'max') {
    super()
    this.reduce = reduce
  }

  apply(_params: Params, x: Value): Value {
    return orientationPool(x, this.reduce)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const a = normaliseAxis(-3, inputShape.length)
    return inputShape.filter((_, i) => i !== a)
  }

  override extraRepr(): string {
    return `reduce='${this.reduce}'`
  }
}

/**
 * One orientation channel of p4 features (`[B, C, H, W]` at orientation r), for inspecting feature maps.
 *
 * @param x - `[B, C, 4, H, W]`.
 * @param r - The orientation (0 to 3).
 * @returns `[B, C, H, W]`.
 * @example
 * ```ts
 * import { orientation } from 'omai-engine/nn'
 * import { zeros } from 'omai-engine/tensor'
 * orientation(zeros([1, 2, 4, 3, 3]), 2).shape // => [1, 2, 3, 3]
 * ```
 */
export function orientation(x: TensorLike, r: number): Value {
  const v = fl(x)
  const axis = v.shape.length - 3
  return reshape(
    sliceAxis(v, axis, r, r + 1),
    v.shape.filter((_, i) => i !== axis),
  )
}

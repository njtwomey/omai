/**
 * Convolutions that only look at the past: causal 1-D convolutions with dilation (left padding, so the output at time
 * t sees inputs up to t), the masked 2-D convolutions of PixelCNN (mask A for the first layer, which excludes the pixel
 * itself, mask B for the later ones, optionally with the red-green-blue ordering of the channels), and the gated
 * layer of Gated PixelCNN with separate vertical and horizontal stacks, which removes the blind spot a stack of
 * masked convolutions leaves to the upper right.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  mul,
  sigmoid,
  sliceAxis,
  type Tensor,
  type TensorLike,
  tanh,
  tensor,
  type Value,
} from '../../tensor/index.ts'
import { conv, type SpatialSize } from '../functional/conv.ts'
import { pad } from '../functional/vision.ts'
import { torchDefaultUniform } from '../init.ts'
import { Composite } from '../module/containers.ts'
import { Conv2d } from '../module/conv.ts'
import { type ApplyContext, initKey, Module, type Params } from '../module/module.ts'

/** Options of {@link CausalConv1d}. */
export interface CausalConv1dOptions {
  readonly dilation?: number
  readonly groups?: number
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * A causal 1-D convolution with dilation: the input is padded on the left by `dilation (kernel - 1)` so the output at
 * time `t` depends on inputs `t, t - d, ..., t - d (k - 1)` only, and the length is kept. Weight `[out, in / groups, k]`
 * with PyTorch's layout and initialisation.
 *
 * @example
 * ```ts
 * import { CausalConv1d } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new CausalConv1d(2, 3, 2, { dilation: 4 }).forward(zeros([1, 2, 10])).shape // => [1, 3, 10]
 * ```
 */
export class CausalConv1d extends Module {
  readonly kind = 'CausalConv1d'
  readonly inChannels: number
  readonly outChannels: number
  readonly kernelSize: number
  readonly dilation: number
  readonly groups: number
  readonly bias: boolean

  constructor(inChannels: number, outChannels: number, kernelSize: number, options: CausalConv1dOptions = {}) {
    super()
    this.groups = options.groups ?? 1
    if (inChannels % this.groups !== 0 || outChannels % this.groups !== 0)
      throw new OmaiError('value', 'CausalConv1d: channels must be divisible by groups')
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = kernelSize
    this.dilation = options.dilation ?? 1
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey('CausalConv1d', options.key))
  }

  override initParams(key: Key): Params {
    const fanIn = (this.inChannels / this.groups) * this.kernelSize
    return {
      weight: torchDefaultUniform(
        foldIn(key, 0),
        [this.outChannels, this.inChannels / this.groups, this.kernelSize],
        fanIn,
      ),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], fanIn) } : {}),
    }
  }

  apply(params: Params, x: Value): Value {
    return conv(pad(x, [this.dilation * (this.kernelSize - 1), 0]), params['weight'] as Value, {
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      dilation: this.dilation,
      groups: this.groups,
    })
  }

  outputShape(s: readonly number[]): number[] {
    return [s[0] ?? 1, this.outChannels, s[2] ?? 0]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, dilation=${this.dilation}`
  }
}

/** Mask A excludes the pixel itself (the first layer); mask B includes it (the later layers). */
export type PixelCnnMaskType = 'A' | 'B'

/**
 * The mask of a PixelCNN convolution, `[out, in, kh, kw]`: taps above the centre row, and in the centre row those left
 * of the centre, are 1; the centre tap is 1 for mask B and 0 for mask A; everything after the centre in raster order
 * is 0. With `colours > 1` (the red-green-blue ordering) the channels are split into that many equal groups and, at
 * the centre tap, an output group connects to input groups before it (mask A) or up to it (mask B).
 *
 * @param outChannels - Output channels.
 * @param inChannels - Input channels.
 * @param kernelSize - `[kh, kw]` (odd), or one number.
 * @param type - `A` or `B`.
 * @param colours - The number of channel groups (default 1).
 * @returns The 0/1 mask.
 * @reference van den Oord, Kalchbrenner and Kavukcuoglu (2016), Pixel recurrent neural networks.
 * @example
 * ```ts
 * import { pixelCnnMask } from 'omai-engine/nn/blocks'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(pixelCnnMask(1, 1, 3, 'A')) // => [[[[1, 1, 1], [1, 0, 0], [0, 0, 0]]]]
 * ```
 */
export function pixelCnnMask(
  outChannels: number,
  inChannels: number,
  kernelSize: SpatialSize,
  type: PixelCnnMaskType,
  colours = 1,
): Tensor {
  const [kh, kw] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : [kernelSize[0] ?? 1, kernelSize[1] ?? 1]
  if (kh % 2 === 0 || kw % 2 === 0) throw new OmaiError('value', 'pixelCnnMask: the kernel must have odd sides')
  if (outChannels % colours !== 0 || inChannels % colours !== 0)
    throw new OmaiError('value', `pixelCnnMask: channels must be divisible by ${colours} colour groups`)
  const ch = (kh - 1) / 2
  const cw = (kw - 1) / 2
  const outGroup = outChannels / colours
  const inGroup = inChannels / colours
  const m: number[][][][] = []
  for (let o = 0; o < outChannels; o++) {
    const rows: number[][][] = []
    for (let i = 0; i < inChannels; i++) {
      const plane: number[][] = []
      for (let r = 0; r < kh; r++) {
        const row: number[] = []
        for (let c = 0; c < kw; c++) {
          const before = r < ch || (r === ch && c < cw)
          const centre = r === ch && c === cw
          const go = Math.floor(o / outGroup)
          const gi = Math.floor(i / inGroup)
          row.push(before || (centre && (type === 'B' ? gi <= go : gi < go)) ? 1 : 0)
        }
        plane.push(row)
      }
      rows.push(plane)
    }
    m.push(rows)
  }
  return tensor(m)
}

/** Options of {@link MaskedConv2d}. */
export interface MaskedConv2dOptions {
  readonly dilation?: number
  readonly bias?: boolean
  /** Channel groups for the colour ordering (default 1). */
  readonly colours?: number
  readonly key?: Key
}

/**
 * The masked convolution of PixelCNN: an ordinary 2-D convolution whose weight is multiplied by {@link pixelCnnMask}
 * (so the masked taps get no gradient), with "same" padding. Stacking mask A then mask B layers gives outputs that
 * depend only on pixels earlier in raster order.
 *
 * @example
 * ```ts
 * import { MaskedConv2d } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new MaskedConv2d(1, 4, 5, 'A').forward(zeros([1, 1, 8, 8])).shape // => [1, 4, 8, 8]
 * ```
 */
export class MaskedConv2d extends Module {
  readonly kind = 'MaskedConv2d'
  readonly inChannels: number
  readonly outChannels: number
  readonly kernelSize: readonly [number, number]
  readonly type: PixelCnnMaskType
  readonly dilation: number
  readonly bias: boolean
  /** The 0/1 mask applied to the weight. */
  readonly mask: Tensor

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    type: PixelCnnMaskType,
    options: MaskedConv2dOptions = {},
  ) {
    super()
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize =
      typeof kernelSize === 'number' ? [kernelSize, kernelSize] : [kernelSize[0] ?? 1, kernelSize[1] ?? 1]
    this.type = type
    this.dilation = options.dilation ?? 1
    this.bias = options.bias ?? true
    this.mask = pixelCnnMask(outChannels, inChannels, this.kernelSize, type, options.colours ?? 1)
    this.params = this.initParams(initKey('MaskedConv2d', options.key))
  }

  override initParams(key: Key): Params {
    const fanIn = this.inChannels * this.kernelSize[0] * this.kernelSize[1]
    return {
      weight: torchDefaultUniform(foldIn(key, 0), [this.outChannels, this.inChannels, ...this.kernelSize], fanIn),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], fanIn) } : {}),
    }
  }

  apply(params: Params, x: Value): Value {
    const [kh, kw] = this.kernelSize
    return conv(x, mul(params['weight'] as Value, this.mask), {
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      padding: [(this.dilation * (kh - 1)) / 2, (this.dilation * (kw - 1)) / 2],
      dilation: this.dilation,
    })
  }

  outputShape(s: readonly number[]): number[] {
    return [...s.slice(0, -3), this.outChannels, ...s.slice(-2)]
  }

  override extraRepr(): string {
    return `${this.inChannels}, ${this.outChannels}, kernel_size=(${this.kernelSize.join(', ')}), mask=${this.type}`
  }
}

/** The two stacks of Gated PixelCNN: vertical features (rows above, including the current) and horizontal ones. */
export interface GatedStacks {
  readonly v: TensorLike
  readonly h: TensorLike
}

/** The output of a {@link GatedMaskedConv} layer. */
export interface GatedStacksOutput {
  readonly v: Value
  readonly h: Value
}

/** Options of {@link GatedMaskedConv}. */
export interface GatedMaskedConvOptions {
  /** The first layer: the horizontal stack must not see the pixel itself (mask A) and there is no residual. */
  readonly first?: boolean
  readonly key?: Key
}

/**
 * A layer of Gated PixelCNN (van den Oord et al. 2016): a vertical stack (a `(k/2 + 1) x k` convolution over the rows
 * above and the current one) and a horizontal stack (a `1 x (k/2 + 1)` convolution over the pixels to the left),
 * each gated as `tanh(a) sigmoid(b)`. The vertical features, shifted down one row, feed the horizontal stack, so the
 * horizontal output at a pixel sees every earlier pixel (no blind spot) and none after it. The horizontal stack adds a
 * residual connection except in the first layer, where the centre tap is masked.
 *
 * @reference van den Oord et al. (2016), Conditional image generation with PixelCNN decoders.
 * @example
 * ```ts
 * import { GatedMaskedConv } from 'omai-engine/nn/blocks'
 * import { zeros } from 'omai-engine/tensor'
 * new GatedMaskedConv(4, 3).forward({ v: zeros([1, 4, 6, 6]), h: zeros([1, 4, 6, 6]) }).h.shape // => [1, 4, 6, 6]
 * ```
 */
export class GatedMaskedConv extends Composite<GatedStacks, GatedStacksOutput> {
  readonly kind = 'GatedMaskedConv'
  readonly channels: number
  readonly kernelSize: number
  readonly first: boolean

  constructor(channels: number, kernelSize: number, options: GatedMaskedConvOptions = {}) {
    if (kernelSize % 2 === 0) throw new OmaiError('value', 'GatedMaskedConv: the kernel size must be odd')
    const k = initKey('GatedMaskedConv', options.key)
    const half = (kernelSize - 1) / 2
    super({
      vertical: new Conv2d(channels, 2 * channels, [half + 1, kernelSize], { key: foldIn(k, 0) }),
      horizontal: new Conv2d(channels, 2 * channels, [1, half + 1], { key: foldIn(k, 1) }),
      verticalToHorizontal: new Conv2d(2 * channels, 2 * channels, 1, { key: foldIn(k, 2) }),
      output: new Conv2d(channels, channels, 1, { key: foldIn(k, 3) }),
    })
    this.channels = channels
    this.kernelSize = kernelSize
    this.first = options.first ?? false
  }

  protected override prepare(x: unknown): unknown {
    const s = x as GatedStacks
    return { v: asTensor(s.v), h: asTensor(s.h) }
  }

  apply(params: Params, input: GatedStacks, ctx: ApplyContext = {}): GatedStacksOutput {
    const half = (this.kernelSize - 1) / 2
    const c = this.channels
    const p = this.parts(params, ctx)
    const v = asTensor(input.v)
    const h = asTensor(input.h)
    const height = v.shape[v.shape.length - 2] ?? 0
    const gate = (z: Value): Value => mul(tanh(sliceAxis(z, 1, 0, c)), sigmoid(sliceAxis(z, 1, c, 2 * c)))
    // Vertical stack: pad above and to both sides, so row i sees rows i - half .. i.
    const vFeatures = p.run<Value, Value>('vertical', pad(v, [half, half, half, 0]))
    // The horizontal stack sees the vertical features of the row above (shift down by one row).
    const shifted = sliceAxis(pad(vFeatures, [0, 0, 1, 0]), 2, 0, height)
    const toH = p.run<Value, Value>('verticalToHorizontal', shifted)
    const hp = (params['horizontal'] ?? {}) as Params
    const tapMask = tensor([[[Array.from({ length: half + 1 }, (_, i) => (this.first && i === half ? 0 : 1))]]])
    const hFeatures = this.part<Conv2d>('horizontal').apply(
      { ...hp, weight: mul(hp['weight'] as Value, tapMask) },
      pad(h, [half, 0]),
    )
    const hOut = p.run<Value, Value>('output', gate(add(hFeatures, toH)))
    const out = { v: gate(vFeatures), h: this.first ? hOut : add(h, hOut) }
    p.done()
    return out
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

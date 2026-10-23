import { foldIn, type Key } from '../../random/index.ts'
import { reshape, type Value } from '../../tensor/index.ts'
import { affineGrid, type GridSampleOptions, gridSample } from '../functional/sampling.ts'
import {
  type InterpolateOptions,
  interpolate,
  type PadModeTorch,
  pad,
  patchEmbedding,
  pixelShuffle,
  pixelUnshuffle,
} from '../functional/vision.ts'
import { torchDefaultUniform } from '../init.ts'
import { type ApplyContext, childScope, initKey, Module, type Params } from './module.ts'

/**
 * Upsampling by interpolation (torch.nn.Upsample): a size or scale factor and a mode.
 *
 * @concept upsampling
 * @example
 * ```ts
 * import { Upsample } from 'whitebox-engine/nn'
 * new Upsample({ scaleFactor: 2, mode: 'bilinear' }).outputShape([1, 3, 4, 5]) // => [1, 3, 8, 10]
 * ```
 */
export class Upsample extends Module {
  readonly kind: string = 'Upsample'
  readonly options: InterpolateOptions

  constructor(options: InterpolateOptions) {
    super()
    this.options = options
  }

  apply(_params: Params, x: Value): Value {
    return interpolate(x, this.options)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const spatial = inputShape.slice(2)
    const { size, scaleFactor } = this.options
    const out = spatial.map((n, i) =>
      size !== undefined
        ? typeof size === 'number'
          ? size
          : (size[i] ?? n)
        : Math.floor(n * (typeof scaleFactor === 'number' ? scaleFactor : (scaleFactor?.[i] ?? 1))),
    )
    return [...inputShape.slice(0, 2), ...out]
  }

  override extraRepr(): string {
    return `mode='${this.options.mode ?? 'nearest'}'`
  }
}

/** Nearest-neighbour 2-D upsampling (torch.nn.UpsamplingNearest2d). @example new UpsamplingNearest2d(2).kind // => 'UpsamplingNearest2d' */
export class UpsamplingNearest2d extends Upsample {
  override readonly kind = 'UpsamplingNearest2d'
  constructor(scaleFactor: number) {
    super({ scaleFactor, mode: 'nearest' })
  }
}

/** Bilinear 2-D upsampling with aligned corners (torch.nn.UpsamplingBilinear2d). @example new UpsamplingBilinear2d(2).kind // => 'UpsamplingBilinear2d' */
export class UpsamplingBilinear2d extends Upsample {
  override readonly kind = 'UpsamplingBilinear2d'
  constructor(scaleFactor: number) {
    super({ scaleFactor, mode: 'bilinear', alignCorners: true })
  }
}

/** Sub-pixel rearrangement (torch.nn.PixelShuffle). @example new PixelShuffle(2).outputShape([1, 8, 3, 3]) // => [1, 2, 6, 6] */
export class PixelShuffle extends Module {
  readonly kind = 'PixelShuffle'
  readonly upscale: number

  constructor(upscale: number) {
    super()
    this.upscale = upscale
  }

  apply(_params: Params, x: Value): Value {
    return pixelShuffle(x, this.upscale)
  }

  outputShape(s: readonly number[]): number[] {
    const r = this.upscale
    return [s[0] ?? 1, (s[1] ?? 1) / (r * r), (s[2] ?? 1) * r, (s[3] ?? 1) * r]
  }
}

/** The inverse rearrangement (torch.nn.PixelUnshuffle). @example new PixelUnshuffle(2).outputShape([1, 2, 6, 6]) // => [1, 8, 3, 3] */
export class PixelUnshuffle extends Module {
  readonly kind = 'PixelUnshuffle'
  readonly downscale: number

  constructor(downscale: number) {
    super()
    this.downscale = downscale
  }

  apply(_params: Params, x: Value): Value {
    return pixelUnshuffle(x, this.downscale)
  }

  outputShape(s: readonly number[]): number[] {
    const r = this.downscale
    return [s[0] ?? 1, (s[1] ?? 1) * r * r, (s[2] ?? 1) / r, (s[3] ?? 1) / r]
  }
}

/**
 * Padding of the trailing d axes (the shared implementation of the padding modules).
 *
 * @example
 * ```ts
 * import { PadNd } from 'whitebox-engine/nn'
 * new PadNd('ReflectionPad1d', 1, 2, 'reflect').outputShape([1, 1, 5]) // => [1, 1, 9]
 * ```
 */
export class PadNd extends Module {
  readonly kind: string
  readonly padding: readonly number[]
  readonly mode: PadModeTorch
  readonly value: number

  constructor(kind: string, d: number, padding: number | readonly number[], mode: PadModeTorch, value = 0) {
    super()
    this.kind = kind
    this.padding = typeof padding === 'number' ? new Array<number>(2 * d).fill(padding) : [...padding]
    this.mode = mode
    this.value = value
  }

  apply(_params: Params, x: Value): Value {
    return pad(x, this.padding, this.mode, this.value)
  }

  outputShape(s: readonly number[]): number[] {
    const out = [...s]
    for (let k = 0; k < this.padding.length / 2; k++) {
      const i = out.length - 1 - k
      out[i] = (out[i] ?? 0) + (this.padding[2 * k] ?? 0) + (this.padding[2 * k + 1] ?? 0)
    }
    return out
  }

  override extraRepr(): string {
    return `(${this.padding.join(', ')})`
  }
}

type Padding = number | readonly number[]

/** Zero padding of the last axis (torch.nn.ZeroPad1d). @example new ZeroPad1d(2).outputShape([1, 3]) // => [1, 7] */
export class ZeroPad1d extends PadNd {
  constructor(padding: Padding) {
    super('ZeroPad1d', 1, padding, 'constant')
  }
}

/** Zero padding of the last two axes (torch.nn.ZeroPad2d). @example new ZeroPad2d([1, 1, 2, 0]).outputShape([1, 1, 3, 3]) // => [1, 1, 5, 5] */
export class ZeroPad2d extends PadNd {
  constructor(padding: Padding) {
    super('ZeroPad2d', 2, padding, 'constant')
  }
}

/** Zero padding of the last three axes (torch.nn.ZeroPad3d). @example new ZeroPad3d(1).outputShape([1, 1, 2, 2, 2]) // => [1, 1, 4, 4, 4] */
export class ZeroPad3d extends PadNd {
  constructor(padding: Padding) {
    super('ZeroPad3d', 3, padding, 'constant')
  }
}

/** Constant padding of the last axis (torch.nn.ConstantPad1d). @example new ConstantPad1d(1, 3).kind // => 'ConstantPad1d' */
export class ConstantPad1d extends PadNd {
  constructor(padding: Padding, value: number) {
    super('ConstantPad1d', 1, padding, 'constant', value)
  }
}

/** Constant padding of the last two axes (torch.nn.ConstantPad2d). @example new ConstantPad2d(1, 3).kind // => 'ConstantPad2d' */
export class ConstantPad2d extends PadNd {
  constructor(padding: Padding, value: number) {
    super('ConstantPad2d', 2, padding, 'constant', value)
  }
}

/** Constant padding of the last three axes (torch.nn.ConstantPad3d). @example new ConstantPad3d(1, 3).kind // => 'ConstantPad3d' */
export class ConstantPad3d extends PadNd {
  constructor(padding: Padding, value: number) {
    super('ConstantPad3d', 3, padding, 'constant', value)
  }
}

/** Reflection padding of the last axis (torch.nn.ReflectionPad1d). @example new ReflectionPad1d(1).kind // => 'ReflectionPad1d' */
export class ReflectionPad1d extends PadNd {
  constructor(padding: Padding) {
    super('ReflectionPad1d', 1, padding, 'reflect')
  }
}

/** Reflection padding of the last two axes (torch.nn.ReflectionPad2d). @example new ReflectionPad2d(1).kind // => 'ReflectionPad2d' */
export class ReflectionPad2d extends PadNd {
  constructor(padding: Padding) {
    super('ReflectionPad2d', 2, padding, 'reflect')
  }
}

/** Reflection padding of the last three axes (torch.nn.ReflectionPad3d). @example new ReflectionPad3d(1).kind // => 'ReflectionPad3d' */
export class ReflectionPad3d extends PadNd {
  constructor(padding: Padding) {
    super('ReflectionPad3d', 3, padding, 'reflect')
  }
}

/** Edge replication padding of the last axis (torch.nn.ReplicationPad1d). @example new ReplicationPad1d(1).kind // => 'ReplicationPad1d' */
export class ReplicationPad1d extends PadNd {
  constructor(padding: Padding) {
    super('ReplicationPad1d', 1, padding, 'replicate')
  }
}

/** Edge replication padding of the last two axes (torch.nn.ReplicationPad2d). @example new ReplicationPad2d(1).kind // => 'ReplicationPad2d' */
export class ReplicationPad2d extends PadNd {
  constructor(padding: Padding) {
    super('ReplicationPad2d', 2, padding, 'replicate')
  }
}

/** Edge replication padding of the last three axes (torch.nn.ReplicationPad3d). @example new ReplicationPad3d(1).kind // => 'ReplicationPad3d' */
export class ReplicationPad3d extends PadNd {
  constructor(padding: Padding) {
    super('ReplicationPad3d', 3, padding, 'replicate')
  }
}

/** Circular padding of the last axis (torch.nn.CircularPad1d). @example new CircularPad1d(1).kind // => 'CircularPad1d' */
export class CircularPad1d extends PadNd {
  constructor(padding: Padding) {
    super('CircularPad1d', 1, padding, 'circular')
  }
}

/** Circular padding of the last two axes (torch.nn.CircularPad2d). @example new CircularPad2d(1).kind // => 'CircularPad2d' */
export class CircularPad2d extends PadNd {
  constructor(padding: Padding) {
    super('CircularPad2d', 2, padding, 'circular')
  }
}

/** Circular padding of the last three axes (torch.nn.CircularPad3d). @example new CircularPad3d(1).kind // => 'CircularPad3d' */
export class CircularPad3d extends PadNd {
  constructor(padding: Padding) {
    super('CircularPad3d', 3, padding, 'circular')
  }
}

/**
 * ViT's patch embedding: `proj` is a convolution with kernel and stride equal to the patch size, its output flattened
 * to tokens `[B, N, D]` (timm's PatchEmbed layout, `proj.weight` and `proj.bias`).
 *
 * @concept vision-transformer
 * @example
 * ```ts
 * import { PatchEmbedding } from 'whitebox-engine/nn'
 * new PatchEmbedding(4, 3, 16).outputShape([2, 3, 32, 32]) // => [2, 64, 16]
 * ```
 */
export class PatchEmbedding extends Module {
  readonly kind = 'PatchEmbedding'
  readonly patchSize: number
  readonly inChannels: number
  readonly embedDim: number

  constructor(patchSize: number, inChannels: number, embedDim: number, options: { readonly key?: Key } = {}) {
    super()
    this.patchSize = patchSize
    this.inChannels = inChannels
    this.embedDim = embedDim
    this.params = this.initParams(initKey('PatchEmbedding', options.key))
  }

  override initParams(key: Key): Params {
    const p = this.patchSize
    const fanIn = this.inChannels * p * p
    return {
      proj: {
        weight: torchDefaultUniform(foldIn(key, 0), [this.embedDim, this.inChannels, p, p], fanIn),
        bias: torchDefaultUniform(foldIn(key, 1), [this.embedDim], fanIn),
      },
    }
  }

  apply(params: Params, x: Value): Value {
    const proj = params['proj'] as Params
    return patchEmbedding(x, proj['weight'] as Value, proj['bias'] as Value)
  }

  outputShape(s: readonly number[]): number[] {
    const p = this.patchSize
    return [s[0] ?? 1, Math.floor((s[2] ?? 0) / p) * Math.floor((s[3] ?? 0) / p), this.embedDim]
  }

  override extraRepr(): string {
    return `patch_size=${this.patchSize}, in_chans=${this.inChannels}, embed_dim=${this.embedDim}`
  }
}

/**
 * A spatial transformer (Jaderberg et al. 2015): a localisation network predicts an affine transform `[N, 6]` from
 * the input, which is resampled through the transformed grid (differentiable end to end).
 *
 * @concept spatial-transformer
 * @example
 * ```ts
 * import { SpatialTransformer, Sequential, Flatten, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * const loc = new Sequential(new Flatten(), new Linear(16, 6, { key: key(0) }))
 * new SpatialTransformer(loc).outputShape([2, 1, 4, 4]) // => [2, 1, 4, 4]
 * ```
 */
export class SpatialTransformer extends Module {
  readonly kind = 'SpatialTransformer'
  readonly localisation: Module
  readonly sampling: GridSampleOptions

  constructor(localisation: Module, sampling: GridSampleOptions = {}) {
    super({ localisation: localisation.params }, { localisation: localisation.buffers })
    this.localisation = localisation
    this.sampling = sampling
  }

  override children(): readonly (readonly [string, Module])[] {
    return [['localisation', this.localisation]]
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const theta = reshape(
      this.localisation.apply((params['localisation'] ?? {}) as Params, x, scope.child('localisation', 0)),
      [-1, 2, 3],
    )
    scope.done()
    return gridSample(x, affineGrid(theta, x.shape, this.sampling.alignCorners ?? false), this.sampling)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

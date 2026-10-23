import { WhiteboxError } from '../../base/index.ts'
import { convOutputSize } from '../../numerics/convolution/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { sizeOf, type Value } from '../../tensor/index.ts'
import {
  conv,
  convTransposed,
  convTransposeOutputSize,
  fold,
  type PaddingMode,
  type SpatialSize,
  type UnfoldOptions,
  unfold,
} from '../functional/conv.ts'
import { torchDefaultUniform } from '../init.ts'
import { initKey, Module, type Params } from './module.ts'

/** Options of the convolution modules. */
export interface ConvModuleOptions {
  readonly stride?: SpatialSize
  /** Padding per axis, or `same` / `valid`. */
  readonly padding?: SpatialSize | 'same' | 'valid'
  readonly dilation?: SpatialSize
  readonly groups?: number
  readonly bias?: boolean
  readonly paddingMode?: PaddingMode
  readonly key?: Key
}

const spread = (v: SpatialSize | undefined, d: number, fallback: number): number[] =>
  v === undefined ? new Array<number>(d).fill(fallback) : typeof v === 'number' ? new Array<number>(d).fill(v) : [...v]

const tuple = (xs: readonly number[]): string => (xs.length === 1 ? `(${xs[0]},)` : `(${xs.join(', ')})`)

/**
 * A convolution over d spatial axes (the shared implementation of {@link Conv1d}, {@link Conv2d} and {@link Conv3d}):
 * weight `[out, in / groups, *kernel]` and bias, initialised as PyTorch does (uniform on +-1/sqrt(fanIn)).
 */
export class ConvNd extends Module {
  readonly kind: string
  readonly inChannels: number
  readonly outChannels: number
  readonly kernelSize: readonly number[]
  readonly stride: readonly number[]
  readonly padding: readonly number[] | 'same' | 'valid'
  readonly dilation: readonly number[]
  readonly groups: number
  readonly bias: boolean
  readonly paddingMode: PaddingMode

  constructor(
    d: number,
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    options: ConvModuleOptions = {},
  ) {
    super()
    this.kind = `Conv${d}d`
    this.groups = options.groups ?? 1
    if (inChannels % this.groups !== 0 || outChannels % this.groups !== 0)
      throw new WhiteboxError('value', `${this.kind}: channels must be divisible by groups`)
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = spread(kernelSize, d, 1)
    this.stride = spread(options.stride, d, 1)
    this.padding =
      options.padding === 'same' || options.padding === 'valid' ? options.padding : spread(options.padding, d, 0)
    this.dilation = spread(options.dilation, d, 1)
    this.bias = options.bias ?? true
    this.paddingMode = options.paddingMode ?? 'zeros'
    this.params = this.initParams(initKey(this.kind, options.key))
  }

  override initParams(key: Key): Params {
    const fanIn = (this.inChannels / this.groups) * sizeOf(this.kernelSize)
    return {
      weight: torchDefaultUniform(
        foldIn(key, 0),
        [this.outChannels, this.inChannels / this.groups, ...this.kernelSize],
        fanIn,
      ),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], fanIn) } : {}),
    }
  }

  apply(params: Params, x: Value): Value {
    return conv(x, params['weight'] as Value, {
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
      paddingMode: this.paddingMode,
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.kernelSize.length
    const spatial = inputShape.slice(-d)
    const lead = inputShape.slice(0, -d - 1)
    if (inputShape[inputShape.length - d - 1] !== this.inChannels)
      throw new WhiteboxError(
        'shape',
        `${this.kind}: expected ${this.inChannels} channels, got [${inputShape.join(', ')}]`,
      )
    const out = spatial.map((n, i) =>
      this.padding === 'same'
        ? n
        : convOutputSize(
            n,
            this.kernelSize[i] ?? 1,
            this.stride[i] ?? 1,
            this.padding === 'valid' ? 0 : (this.padding[i] ?? 0),
            this.dilation[i] ?? 1,
          ),
    )
    return [...lead, this.outChannels, ...out]
  }

  override extraRepr(): string {
    const parts = [`${this.inChannels}, ${this.outChannels}, kernel_size=${tuple(this.kernelSize)}`]
    parts.push(`stride=${tuple(this.stride)}`)
    if (typeof this.padding === 'string') parts.push(`padding=${this.padding}`)
    else if (this.padding.some((p) => p !== 0)) parts.push(`padding=${tuple(this.padding)}`)
    if (this.dilation.some((v) => v !== 1)) parts.push(`dilation=${tuple(this.dilation)}`)
    if (this.groups !== 1) parts.push(`groups=${this.groups}`)
    if (!this.bias) parts.push('bias=False')
    if (this.paddingMode !== 'zeros') parts.push(`padding_mode=${this.paddingMode}`)
    return parts.join(', ')
  }
}

/**
 * 1-D convolution (torch.nn.Conv1d): `[B, C_in, L]` to `[B, C_out, L_out]`.
 *
 * @concept convolution
 * @example
 * ```ts
 * import { Conv1d } from 'whitebox-engine/nn'
 * new Conv1d(3, 8, 5, { padding: 2 }).outputShape([4, 3, 32]) // => [4, 8, 32]
 * ```
 */
export class Conv1d extends ConvNd {
  constructor(inChannels: number, outChannels: number, kernelSize: SpatialSize, options: ConvModuleOptions = {}) {
    super(1, inChannels, outChannels, kernelSize, options)
  }
}

/**
 * 2-D convolution (torch.nn.Conv2d): `[B, C_in, H, W]` to `[B, C_out, H_out, W_out]`.
 *
 * @concept convolution
 * @example
 * ```ts
 * import { Conv2d } from 'whitebox-engine/nn'
 * new Conv2d(3, 16, 3, { stride: 2, padding: 1 }).outputShape([1, 3, 32, 32]) // => [1, 16, 16, 16]
 * ```
 */
export class Conv2d extends ConvNd {
  constructor(inChannels: number, outChannels: number, kernelSize: SpatialSize, options: ConvModuleOptions = {}) {
    super(2, inChannels, outChannels, kernelSize, options)
  }
}

/**
 * 3-D convolution (torch.nn.Conv3d): `[B, C_in, D, H, W]` to `[B, C_out, D_out, H_out, W_out]`.
 *
 * @example
 * ```ts
 * import { Conv3d } from 'whitebox-engine/nn'
 * new Conv3d(1, 4, 3).outputShape([1, 1, 8, 8, 8]) // => [1, 4, 6, 6, 6]
 * ```
 */
export class Conv3d extends ConvNd {
  constructor(inChannels: number, outChannels: number, kernelSize: SpatialSize, options: ConvModuleOptions = {}) {
    super(3, inChannels, outChannels, kernelSize, options)
  }
}

/** Options of the transposed convolution modules. */
export interface ConvTransposeModuleOptions {
  readonly stride?: SpatialSize
  readonly padding?: SpatialSize
  readonly outputPadding?: SpatialSize
  readonly dilation?: SpatialSize
  readonly groups?: number
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * A transposed convolution over d spatial axes (the shared implementation of {@link ConvTranspose1d},
 * {@link ConvTranspose2d} and {@link ConvTranspose3d}): weight `[in, out / groups, *kernel]`.
 */
export class ConvTransposeNd extends Module {
  readonly kind: string
  readonly inChannels: number
  readonly outChannels: number
  readonly kernelSize: readonly number[]
  readonly stride: readonly number[]
  readonly padding: readonly number[]
  readonly outputPadding: readonly number[]
  readonly dilation: readonly number[]
  readonly groups: number
  readonly bias: boolean

  constructor(
    d: number,
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    options: ConvTransposeModuleOptions = {},
  ) {
    super()
    this.kind = `ConvTranspose${d}d`
    this.groups = options.groups ?? 1
    if (inChannels % this.groups !== 0 || outChannels % this.groups !== 0)
      throw new WhiteboxError('value', `${this.kind}: channels must be divisible by groups`)
    this.inChannels = inChannels
    this.outChannels = outChannels
    this.kernelSize = spread(kernelSize, d, 1)
    this.stride = spread(options.stride, d, 1)
    this.padding = spread(options.padding, d, 0)
    this.outputPadding = spread(options.outputPadding, d, 0)
    this.dilation = spread(options.dilation, d, 1)
    this.bias = options.bias ?? true
    this.params = this.initParams(initKey(this.kind, options.key))
  }

  override initParams(key: Key): Params {
    // PyTorch computes the fan-in from the weight's second axis: out / groups times the kernel size.
    const fanIn = (this.outChannels / this.groups) * sizeOf(this.kernelSize)
    return {
      weight: torchDefaultUniform(
        foldIn(key, 0),
        [this.inChannels, this.outChannels / this.groups, ...this.kernelSize],
        fanIn,
      ),
      ...(this.bias ? { bias: torchDefaultUniform(foldIn(key, 1), [this.outChannels], fanIn) } : {}),
    }
  }

  apply(params: Params, x: Value): Value {
    return convTransposed(x, params['weight'] as Value, {
      ...(params['bias'] === undefined ? {} : { bias: params['bias'] as Value }),
      stride: this.stride,
      padding: this.padding,
      outputPadding: this.outputPadding,
      dilation: this.dilation,
      groups: this.groups,
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.kernelSize.length
    const out = inputShape
      .slice(-d)
      .map((n, i) =>
        convTransposeOutputSize(
          n,
          this.kernelSize[i] ?? 1,
          this.stride[i] ?? 1,
          this.padding[i] ?? 0,
          this.outputPadding[i] ?? 0,
          this.dilation[i] ?? 1,
        ),
      )
    return [...inputShape.slice(0, -d - 1), this.outChannels, ...out]
  }

  override extraRepr(): string {
    const parts = [`${this.inChannels}, ${this.outChannels}, kernel_size=${tuple(this.kernelSize)}`]
    parts.push(`stride=${tuple(this.stride)}`)
    if (this.padding.some((p) => p !== 0)) parts.push(`padding=${tuple(this.padding)}`)
    if (this.outputPadding.some((p) => p !== 0)) parts.push(`output_padding=${tuple(this.outputPadding)}`)
    if (this.dilation.some((v) => v !== 1)) parts.push(`dilation=${tuple(this.dilation)}`)
    if (this.groups !== 1) parts.push(`groups=${this.groups}`)
    if (!this.bias) parts.push('bias=False')
    return parts.join(', ')
  }
}

/**
 * 1-D transposed convolution (torch.nn.ConvTranspose1d).
 *
 * @example
 * ```ts
 * import { ConvTranspose1d } from 'whitebox-engine/nn'
 * new ConvTranspose1d(4, 2, 4, { stride: 2, padding: 1 }).outputShape([1, 4, 8]) // => [1, 2, 16]
 * ```
 */
export class ConvTranspose1d extends ConvTransposeNd {
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    options: ConvTransposeModuleOptions = {},
  ) {
    super(1, inChannels, outChannels, kernelSize, options)
  }
}

/**
 * 2-D transposed convolution (torch.nn.ConvTranspose2d; the upsampling layer of DCGAN generators and U-Nets).
 *
 * @concept transposed-convolution
 * @example
 * ```ts
 * import { ConvTranspose2d } from 'whitebox-engine/nn'
 * new ConvTranspose2d(8, 4, 4, { stride: 2, padding: 1 }).outputShape([1, 8, 7, 7]) // => [1, 4, 14, 14]
 * ```
 */
export class ConvTranspose2d extends ConvTransposeNd {
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    options: ConvTransposeModuleOptions = {},
  ) {
    super(2, inChannels, outChannels, kernelSize, options)
  }
}

/**
 * 3-D transposed convolution (torch.nn.ConvTranspose3d).
 *
 * @example
 * ```ts
 * import { ConvTranspose3d } from 'whitebox-engine/nn'
 * new ConvTranspose3d(2, 1, 2, { stride: 2 }).outputShape([1, 2, 3, 3, 3]) // => [1, 1, 6, 6, 6]
 * ```
 */
export class ConvTranspose3d extends ConvTransposeNd {
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: SpatialSize,
    options: ConvTransposeModuleOptions = {},
  ) {
    super(3, inChannels, outChannels, kernelSize, options)
  }
}

/**
 * Sliding local blocks as a module (torch.nn.Unfold).
 *
 * @example
 * ```ts
 * import { Unfold } from 'whitebox-engine/nn'
 * new Unfold([2, 2]).outputShape([1, 3, 4, 4]) // => [1, 12, 9]
 * ```
 */
export class Unfold extends Module {
  readonly kind = 'Unfold'
  readonly kernelSize: readonly number[]
  readonly options: UnfoldOptions

  constructor(kernelSize: readonly number[], options: UnfoldOptions = {}) {
    super()
    this.kernelSize = kernelSize
    this.options = options
  }

  apply(_params: Params, x: Value): Value {
    return unfold(x, this.kernelSize, this.options)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const d = this.kernelSize.length
    const [b = 1, c = 1, ...spatial] = inputShape
    const stride = spread(this.options.stride, d, 1)
    const padding = spread(this.options.padding, d, 0)
    const dilation = spread(this.options.dilation, d, 1)
    const blocks = spatial.map((n, i) =>
      convOutputSize(n, this.kernelSize[i] ?? 1, stride[i] ?? 1, padding[i] ?? 0, dilation[i] ?? 1),
    )
    return [b, c * sizeOf(this.kernelSize), sizeOf(blocks)]
  }

  override extraRepr(): string {
    return `kernel_size=${tuple(this.kernelSize)}`
  }
}

/**
 * Combining sliding local blocks as a module (torch.nn.Fold).
 *
 * @example
 * ```ts
 * import { Fold } from 'whitebox-engine/nn'
 * new Fold([3, 3], [2, 2]).outputShape([1, 12, 4]) // => [1, 3, 3, 3]
 * ```
 */
export class Fold extends Module {
  readonly kind = 'Fold'
  readonly outputSize: readonly number[]
  readonly kernelSize: readonly number[]
  readonly options: UnfoldOptions

  constructor(outputSize: readonly number[], kernelSize: readonly number[], options: UnfoldOptions = {}) {
    super()
    this.outputSize = outputSize
    this.kernelSize = kernelSize
    this.options = options
  }

  apply(_params: Params, x: Value): Value {
    return fold(x, this.outputSize, this.kernelSize, this.options)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const [b = 1, ck = 1] = inputShape
    return [b, ck / sizeOf(this.kernelSize), ...this.outputSize]
  }

  override extraRepr(): string {
    return `output_size=${tuple(this.outputSize)}, kernel_size=${tuple(this.kernelSize)}`
  }
}

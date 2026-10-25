/**
 * Convolutions as functions over any number of spatial axes (torch.nn.functional style, cross-correlation as in deep
 * learning): `conv1d`, `conv2d` and `conv3d` with stride, padding modes, dilation and groups, depthwise and separable
 * convolutions, transposed convolutions for upsampling, and `unfold` and `fold` for sliding local blocks.
 */

import { WhiteboxError } from '../../base/index.ts'
import { col2im, convNd, convTranspose, im2col } from '../../numerics/convolution/index.ts'
import { expandDims, promoteFloat, squeeze, type TensorLike, type Value } from '../../tensor/index.ts'

/** How a convolution pads its input (torch's padding_mode): zeros, reflection, replication of the edge, or wrapping. */
export type PaddingMode = 'zeros' | 'reflect' | 'replicate' | 'circular'

/** A size per spatial axis, or one size for every axis. */
export type SpatialSize = number | readonly number[]

/** Options of the convolutions ({@link conv1d}, {@link conv2d}, {@link conv3d}). */
export interface ConvOptions {
  readonly bias?: TensorLike
  readonly stride?: SpatialSize
  /** Padding per axis, or `same` (output as long as the input; stride 1) or `valid` (none, the default). */
  readonly padding?: SpatialSize | 'same' | 'valid'
  readonly dilation?: SpatialSize
  /** Split the channels into groups convolved separately (`groups = C_in` is a depthwise convolution). */
  readonly groups?: number
  readonly paddingMode?: PaddingMode
}

/** Options of the transposed convolutions. */
export interface ConvTransposeOptions {
  readonly bias?: TensorLike
  readonly stride?: SpatialSize
  readonly padding?: SpatialSize
  /** Extra size added to one side of the output (resolves the ambiguity of strided outputs). */
  readonly outputPadding?: SpatialSize
  /** The wanted spatial output size (instead of `outputPadding`). */
  readonly outputSize?: readonly number[]
  readonly groups?: number
  readonly dilation?: SpatialSize
}

const spread = (v: SpatialSize | undefined, d: number, fallback: number): number[] =>
  v === undefined ? new Array<number>(d).fill(fallback) : typeof v === 'number' ? new Array<number>(d).fill(v) : [...v]

const numpyMode = (mode: PaddingMode | undefined): 'constant' | 'reflect' | 'edge' | 'wrap' =>
  mode === 'reflect' ? 'reflect' : mode === 'replicate' ? 'edge' : mode === 'circular' ? 'wrap' : 'constant'

/** Run `f` on a batched input, adding and removing the batch axis for an unbatched `[C, *S]` input. */
function batched(x: Value, d: number, name: string, f: (v: Value) => Value): Value {
  if (x.shape.length === d + 2) return f(x)
  if (x.shape.length === d + 1) return squeeze(f(expandDims(x, 0)), 0)
  throw new WhiteboxError('shape', `${name}: input must be [B, C, ...${d} spatial] or [C, ...${d} spatial]`)
}

/**
 * Convolution over any number of spatial axes (torch.nn.functional.conv1d/2d/3d, a cross-correlation as in deep
 * learning): input `[B, C_in, *S]` (or unbatched `[C_in, *S]`), weight `[C_out, C_in / groups, *K]`, with bias,
 * stride, padding (or 'same'), padding mode, dilation and groups. Differentiable in input, weight and bias.
 *
 * @param input - Input `[B, C_in, *S]` or `[C_in, *S]`.
 * @param weight - Kernels `[C_out, C_in / groups, *K]`.
 * @param options - Bias, stride, padding, dilation, groups and padding mode.
 * @returns `[B, C_out, *O]` with O = floor((S + 2 padding - dilation (K - 1) - 1) / stride) + 1.
 * @concept convolution
 * @example
 * ```ts
 * import { conv } from 'whitebox-engine/nn'
 * conv(new Array(2).fill(new Array(3).fill(new Array(5).fill(1))), new Array(4).fill(new Array(3).fill([1, 1, 1])), { padding: 1 }).shape // => [2, 4, 5]
 * ```
 */
export function conv(input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value {
  const [x, w] = promoteFloat(input, weight) as [Value, Value]
  const d = w.shape.length - 2
  if (d < 1) throw new WhiteboxError('shape', 'conv: weight must be [C_out, C_in / groups, ...kernel]')
  return batched(x, d, 'conv', (v) =>
    convNd(v, w, {
      ...(options.bias === undefined ? {} : { bias: options.bias }),
      stride: spread(options.stride, d, 1),
      dilation: spread(options.dilation, d, 1),
      padding:
        options.padding === 'same' || options.padding === 'valid' ? options.padding : spread(options.padding, d, 0),
      groups: options.groups ?? 1,
      paddingMode: numpyMode(options.paddingMode),
    }),
  )
}

const rankChecked =
  (d: number, name: string) =>
  (input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value => {
    const [w] = promoteFloat(weight) as [Value]
    if (w.shape.length !== d + 2) throw new WhiteboxError('shape', `${name}: weight must have ${d + 2} axes`)
    return conv(input, w, options)
  }

/**
 * 1-D convolution (torch.nn.functional.conv1d): input `[B, C_in, L]`, weight `[C_out, C_in / groups, K]`.
 *
 * @param input - Input `[B, C_in, L]` or `[C_in, L]`.
 * @param weight - Kernels `[C_out, C_in / groups, K]`.
 * @param options - Bias, stride, padding ('same', 'valid' or sizes), dilation, groups and padding mode.
 * @returns `[B, C_out, L_out]`.
 * @example
 * ```ts
 * import { conv1d } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(conv1d([[[1, 2, 3, 4]]], [[[1, 0, -1]]])) // => [[[-2, -2]]]
 * ```
 */
export function conv1d(input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value {
  return rankChecked(1, 'conv1d')(input, weight, options)
}

/**
 * 2-D convolution (torch.nn.functional.conv2d): input `[B, C_in, H, W]`, weight `[C_out, C_in / groups, kH, kW]`.
 *
 * @param input - Input `[B, C_in, H, W]` or `[C_in, H, W]`.
 * @param weight - Kernels `[C_out, C_in / groups, kH, kW]`.
 * @param options - Bias, stride, padding, dilation, groups and padding mode.
 * @returns `[B, C_out, H_out, W_out]`.
 * @example
 * ```ts
 * import { conv2d } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(conv2d([[[[1, 2], [3, 4]]]], [[[[1, 1], [1, 1]]]])) // => [[[[10]]]]
 * ```
 */
export function conv2d(input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value {
  return rankChecked(2, 'conv2d')(input, weight, options)
}

/**
 * 3-D convolution (torch.nn.functional.conv3d): input `[B, C_in, D, H, W]`, weight `[C_out, C_in / groups, kD, kH, kW]`.
 *
 * @param input - Input `[B, C_in, D, H, W]` or `[C_in, D, H, W]`.
 * @param weight - Kernels `[C_out, C_in / groups, kD, kH, kW]`.
 * @param options - Bias, stride, padding, dilation, groups and padding mode.
 * @returns `[B, C_out, D_out, H_out, W_out]`.
 * @example
 * ```ts
 * import { conv3d } from 'whitebox-engine/nn'
 * conv3d([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]], [[[[[1]]]]]).shape // => [1, 1, 2, 2, 2]
 * ```
 */
export function conv3d(input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value {
  return rankChecked(3, 'conv3d')(input, weight, options)
}

/**
 * Depthwise convolution: each input channel convolved with its own `multiplier` kernels (groups = C_in; MobileNet's
 * depthwise stage). Weight `[C_in multiplier, 1, *K]`.
 *
 * @param input - Input `[B, C_in, *S]`.
 * @param weight - Kernels `[C_in multiplier, 1, *K]`.
 * @param options - Bias, stride, padding, dilation and padding mode (groups is set).
 * @returns `[B, C_in multiplier, *O]`.
 * @example
 * ```ts
 * import { depthwiseConv } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(depthwiseConv([[[1, 2, 3], [1, 1, 1]]], [[[1, 1]], [[2, 0]]])) // => [[[3, 5], [2, 2]]]
 * ```
 */
export function depthwiseConv(input: TensorLike, weight: TensorLike, options: Omit<ConvOptions, 'groups'> = {}): Value {
  const [x, w] = promoteFloat(input, weight) as [Value, Value]
  const d = w.shape.length - 2
  const c = x.shape[x.shape.length - d - 1] ?? 0
  if (w.shape[1] !== 1 || (w.shape[0] ?? 0) % c !== 0)
    throw new WhiteboxError('shape', 'depthwiseConv: weight must be [C_in multiplier, 1, ...kernel]')
  return conv(x, w, { ...options, groups: c })
}

/**
 * Depthwise-separable convolution (Xception, MobileNet): a depthwise convolution followed by a pointwise (1x1)
 * convolution that mixes channels.
 *
 * @param input - Input `[B, C_in, *S]`.
 * @param depthwiseWeight - `[C_in multiplier, 1, *K]`.
 * @param pointwiseWeight - `[C_out, C_in multiplier, 1, ...]`.
 * @param options - Stride, padding, dilation and padding mode of the depthwise stage, and the pointwise bias.
 * @returns `[B, C_out, *O]`.
 * @example
 * ```ts
 * import { separableConv } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(separableConv([[[1, 2, 3]]], [[[1, 1]]], [[[2]]])) // => [[[6, 10]]]
 * ```
 */
export function separableConv(
  input: TensorLike,
  depthwiseWeight: TensorLike,
  pointwiseWeight: TensorLike,
  options: Omit<ConvOptions, 'groups'> = {},
): Value {
  const { bias, ...rest } = options
  return conv(depthwiseConv(input, depthwiseWeight, rest), pointwiseWeight, bias === undefined ? {} : { bias })
}

/**
 * The output length of a transposed convolution along one axis: (n - 1) stride - 2 padding + dilation (k - 1) +
 * outputPadding + 1.
 *
 * @param n - Input length.
 * @param k - Kernel length.
 * @param stride - Stride (default 1).
 * @param padding - Padding (default 0).
 * @param outputPadding - Output padding (default 0).
 * @param dilation - Dilation (default 1).
 * @returns The output length.
 * @example
 * ```ts
 * import { convTransposeOutputSize } from 'whitebox-engine/nn'
 * convTransposeOutputSize(16, 4, 2, 1) // => 32
 * ```
 */
export function convTransposeOutputSize(
  n: number,
  k: number,
  stride = 1,
  padding = 0,
  outputPadding = 0,
  dilation = 1,
): number {
  return (n - 1) * stride - 2 * padding + dilation * (k - 1) + outputPadding + 1
}

/**
 * Transposed convolution over any number of spatial axes (torch.nn.functional.conv_transpose1d/2d/3d; the gradient
 * of a convolution with respect to its input, used to upsample): weight `[C_in, C_out / groups, *K]`.
 *
 * @param input - Input `[B, C_in, *S]` or `[C_in, *S]`.
 * @param weight - Kernels `[C_in, C_out / groups, *K]`.
 * @param options - Bias, stride, padding, output padding (or the wanted output size), groups and dilation.
 * @returns `[B, C_out, *O]`.
 * @concept transposed-convolution
 * @example
 * ```ts
 * import { convTransposed } from 'whitebox-engine/nn'
 * convTransposed([[[[1, 2], [3, 4]]]], [[[[1, 1], [1, 1]]]], { stride: 2 }).shape // => [1, 1, 4, 4]
 * ```
 */
export function convTransposed(input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value {
  const [x, w] = promoteFloat(input, weight) as [Value, Value]
  const d = w.shape.length - 2
  if (d < 1) throw new WhiteboxError('shape', 'convTransposed: weight must be [C_in, C_out / groups, ...kernel]')
  const stride = spread(options.stride, d, 1)
  const padding = spread(options.padding, d, 0)
  const dilation = spread(options.dilation, d, 1)
  let outputPadding = spread(options.outputPadding, d, 0)
  if (options.outputSize !== undefined) {
    const size = options.outputSize.slice(-d)
    const spatial = x.shape.slice(-d)
    outputPadding = size.map((s, i) => {
      const base = convTransposeOutputSize(spatial[i] ?? 0, w.shape[i + 2] ?? 1, stride[i], padding[i], 0, dilation[i])
      const extra = s - base
      if (extra < 0 || extra >= Math.max(stride[i] ?? 1, dilation[i] ?? 1))
        throw new WhiteboxError('value', `convTransposed: output size ${s} is not reachable along axis ${i}`)
      return extra
    })
  }
  return batched(x, d, 'convTransposed', (v) =>
    convTranspose(v, w, {
      ...(options.bias === undefined ? {} : { bias: options.bias }),
      stride,
      padding,
      dilation,
      outputPadding,
      groups: options.groups ?? 1,
    }),
  )
}

const transposeRank =
  (d: number, name: string) =>
  (input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value => {
    const [w] = promoteFloat(weight) as [Value]
    if (w.shape.length !== d + 2) throw new WhiteboxError('shape', `${name}: weight must have ${d + 2} axes`)
    return convTransposed(input, w, options)
  }

/**
 * 1-D transposed convolution (torch.nn.functional.conv_transpose1d).
 *
 * @param input - Input `[B, C_in, L]`.
 * @param weight - Kernels `[C_in, C_out / groups, K]`.
 * @param options - Bias, stride, padding, output padding or size, groups and dilation.
 * @returns `[B, C_out, L_out]`.
 * @example
 * ```ts
 * import { convTranspose1d } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(convTranspose1d([[[1, 2]]], [[[1, 1, 1]]], { stride: 2 })) // => [[[1, 1, 3, 2, 2]]]
 * ```
 */
export function convTranspose1d(input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value {
  return transposeRank(1, 'convTranspose1d')(input, weight, options)
}

/**
 * 2-D transposed convolution (torch.nn.functional.conv_transpose2d).
 *
 * @param input - Input `[B, C_in, H, W]`.
 * @param weight - Kernels `[C_in, C_out / groups, kH, kW]`.
 * @param options - Bias, stride, padding, output padding or size, groups and dilation.
 * @returns `[B, C_out, H_out, W_out]`.
 * @example
 * ```ts
 * import { convTranspose2d } from 'whitebox-engine/nn'
 * convTranspose2d([[[[1, 2], [3, 4]]]], [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], { stride: 2, padding: 1, outputPadding: 1 }).shape // => [1, 1, 4, 4]
 * ```
 */
export function convTranspose2d(input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value {
  return transposeRank(2, 'convTranspose2d')(input, weight, options)
}

/**
 * 3-D transposed convolution (torch.nn.functional.conv_transpose3d).
 *
 * @param input - Input `[B, C_in, D, H, W]`.
 * @param weight - Kernels `[C_in, C_out / groups, kD, kH, kW]`.
 * @param options - Bias, stride, padding, output padding or size, groups and dilation.
 * @returns `[B, C_out, D_out, H_out, W_out]`.
 * @example
 * ```ts
 * import { convTranspose3d } from 'whitebox-engine/nn'
 * convTranspose3d([[[[[1]]]]], [[[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]]).shape // => [1, 1, 2, 2, 2]
 * ```
 */
export function convTranspose3d(input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value {
  return transposeRank(3, 'convTranspose3d')(input, weight, options)
}

/** Options of {@link unfold} and {@link fold}. */
export interface UnfoldOptions {
  readonly dilation?: SpatialSize
  readonly padding?: SpatialSize
  readonly stride?: SpatialSize
}

/**
 * Extract sliding local blocks (torch.nn.functional.unfold, any number of spatial axes): `[B, C, *S]` to
 * `[B, C prod(K), L]`, channels major then kernel offsets, L block positions.
 *
 * @param input - Input `[B, C, *S]`.
 * @param kernelSize - Block size per spatial axis (or one size).
 * @param options - Dilation, zero padding and stride.
 * @returns The blocks.
 * @example
 * ```ts
 * import { unfold } from 'whitebox-engine/nn'
 * unfold([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], 2).shape // => [1, 4, 4]
 * ```
 */
export function unfold(input: TensorLike, kernelSize: SpatialSize, options: UnfoldOptions = {}): Value {
  const [x] = promoteFloat(input) as [Value]
  const d = x.shape.length - 2
  return im2col(x, spread(kernelSize, d, 1), {
    stride: spread(options.stride, d, 1),
    dilation: spread(options.dilation, d, 1),
    padding: spread(options.padding, d, 0),
  })
}

/**
 * Combine sliding local blocks into an array, summing overlaps (torch.nn.functional.fold, the adjoint of
 * {@link unfold}).
 *
 * @param input - Blocks `[B, C prod(K), L]`.
 * @param outputSize - Spatial size of the result.
 * @param kernelSize - Block size.
 * @param options - Dilation, padding and stride (as given to `unfold`).
 * @returns `[B, C, *outputSize]`.
 * @example
 * ```ts
 * import { fold, unfold } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fold(unfold([[[[1, 1], [1, 1]]]], 1), [2, 2], 1)) // => [[[[1, 1], [1, 1]]]]
 * ```
 */
export function fold(
  input: TensorLike,
  outputSize: readonly number[],
  kernelSize: SpatialSize,
  options: UnfoldOptions = {},
): Value {
  const d = outputSize.length
  return col2im(input, outputSize, spread(kernelSize, d, 1), {
    stride: spread(options.stride, d, 1),
    dilation: spread(options.dilation, d, 1),
    padding: spread(options.padding, d, 0),
  })
}

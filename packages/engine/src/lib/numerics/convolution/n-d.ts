/**
 * Convolution as in deep learning (a cross-correlation, `torch.nn.functional.conv1d/2d/3d`) over any number of spatial
 * axes, with stride, padding, dilation and groups, its transpose, and the `im2col` and `col2im` patch extraction it is
 * built on.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  fromStorage,
  matmul,
  promoteFloat,
  reshape,
  scatterAdd,
  sliceAxis,
  swapaxes,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'
import { type PadMode, padArray } from './padding.ts'

const expand = (v: number | readonly number[] | undefined, d: number, fallback: number): number[] =>
  v === undefined ? new Array<number>(d).fill(fallback) : typeof v === 'number' ? new Array<number>(d).fill(v) : [...v]

const product = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1)

/**
 * The output length of a convolution along one axis: floor((n + 2 padding - dilation (k - 1) - 1) / stride) + 1.
 *
 * @param n - Input length.
 * @param k - Kernel length.
 * @param stride - Stride (default 1).
 * @param padding - Padding on each side (default 0).
 * @param dilation - Dilation (default 1).
 * @returns The output length.
 * @example
 * ```ts
 * import { convOutputSize } from 'omai-engine/numerics/convolution'
 * convOutputSize(32, 3, 2, 1) // => 16
 * ```
 */
export function convOutputSize(n: number, k: number, stride = 1, padding = 0, dilation = 1): number {
  return Math.floor((n + 2 * padding - dilation * (k - 1) - 1) / stride) + 1
}

/** For each (kernel offset, output position) pair, the flat input index, kernel offsets major. */
function patchIndices(
  spatial: readonly number[],
  kernel: readonly number[],
  stride: readonly number[],
  dilation: readonly number[],
): { idx: Int32Array; out: number[] } {
  const d = spatial.length
  const out = spatial.map((n, i) => convOutputSize(n, kernel[i] ?? 1, stride[i] ?? 1, 0, dilation[i] ?? 1))
  if (out.some((o) => o < 1)) throw new OmaiError('shape', 'convolution: the kernel is larger than the (padded) input')
  const strides = spatial.map((_, i) => product(spatial.slice(i + 1)))
  const nk = product(kernel)
  const nl = product(out)
  const idx = new Int32Array(nk * nl)
  const kpos = new Array<number>(d).fill(0)
  for (let a = 0; a < nk; a++) {
    let rem = a
    for (let i = d - 1; i >= 0; i--) {
      kpos[i] = rem % (kernel[i] ?? 1)
      rem = Math.floor(rem / (kernel[i] ?? 1))
    }
    for (let b = 0; b < nl; b++) {
      let r = b
      let flat = 0
      for (let i = d - 1; i >= 0; i--) {
        const o = r % (out[i] ?? 1)
        r = Math.floor(r / (out[i] ?? 1))
        flat += (o * (stride[i] ?? 1) + (kpos[i] ?? 0) * (dilation[i] ?? 1)) * (strides[i] ?? 1)
      }
      idx[a * nl + b] = flat
    }
  }
  return { idx, out }
}

/** Options of {@link im2col} and {@link col2im}. */
export interface PatchOptions {
  readonly stride?: number | readonly number[]
  readonly dilation?: number | readonly number[]
  readonly padding?: number | readonly number[]
}

/**
 * Extract sliding patches (torch.nn.functional.unfold, any number of spatial axes): `[B, C, *S]` to
 * `[B, C prod(K), L]` with channels major, then kernel offsets, and L output positions. A gather, so differentiable.
 *
 * @param x - Input `[B, C, *S]`.
 * @param kernel - Kernel size per spatial axis.
 * @param options - Stride, dilation and zero padding.
 * @returns The patches.
 * @concept im2col
 * @example
 * ```ts
 * import { im2col } from 'omai-engine/numerics/convolution'
 * im2col([[[1, 2, 3, 4]]], [2]).shape // => [1, 2, 3]
 * ```
 */
export function im2col(x: TensorLike, kernel: readonly number[], options: PatchOptions = {}): Value {
  const [v0] = promoteFloat(x) as [Value]
  const d = kernel.length
  if (v0.shape.length !== d + 2) throw new OmaiError('shape', `im2col: input must be [B, C, ...${d} spatial axes]`)
  const padding = expand(options.padding, d, 0)
  const v = padding.some((p) => p > 0)
    ? padArray(v0, [[0, 0], [0, 0], ...padding.map((p): [number, number] => [p, p])])
    : v0
  const [b = 0, c = 0, ...spatial] = v.shape
  const { idx, out } = patchIndices(spatial, kernel, expand(options.stride, d, 1), expand(options.dilation, d, 1))
  const flat = reshape(v, [b, c, product(spatial)])
  const gathered = take(flat, fromStorage(idx, [idx.length], 'int32'), 2)
  return reshape(gathered, [b, c * product(kernel), product(out)])
}

/**
 * Sum patches back into an array (torch.nn.functional.fold): the adjoint of {@link im2col}; overlapping patches add.
 *
 * @param cols - Patches `[B, C prod(K), L]`.
 * @param outputSize - Spatial size of the result (before removing padding).
 * @param kernel - Kernel size.
 * @param options - Stride, dilation and padding (removed from the result).
 * @returns `[B, C, *outputSize - 2 padding]`.
 * @example
 * ```ts
 * import { col2im, im2col } from 'omai-engine/numerics/convolution'
 * import { toArray } from 'omai-engine/tensor'
 * // Each interior sample appears in two windows.
 * toArray(col2im(im2col([[[1, 1, 1, 1]]], [2]), [4], [2])) // => [[[1, 2, 2, 1]]]
 * ```
 */
export function col2im(
  cols: TensorLike,
  outputSize: readonly number[],
  kernel: readonly number[],
  options: PatchOptions = {},
): Value {
  const [v] = promoteFloat(cols) as [Value]
  const d = kernel.length
  const padding = expand(options.padding, d, 0)
  const spatial = outputSize.map((n, i) => n + 2 * (padding[i] ?? 0))
  const [b = 0, ck = 0] = v.shape
  const nk = product(kernel)
  const c = ck / nk
  if (!Number.isInteger(c)) throw new OmaiError('shape', 'col2im: channels must be a multiple of the kernel size')
  const { idx } = patchIndices(spatial, kernel, expand(options.stride, d, 1), expand(options.dilation, d, 1))
  const updates = reshape(v, [b, c, idx.length])
  const full = new Int32Array(b * c * idx.length)
  for (let r = 0; r < b * c; r++) full.set(idx, r * idx.length)
  let out = reshape(scatterAdd([b, c, product(spatial)], fromStorage(full, [b, c, idx.length], 'int32'), updates, 2), [
    b,
    c,
    ...spatial,
  ])
  padding.forEach((p, i) => {
    if (p > 0) out = sliceAxis(out, i + 2, p, p + (outputSize[i] ?? 0))
  })
  return out
}

/** Options of {@link convNd}. */
export interface ConvOptions {
  readonly stride?: number | readonly number[]
  /** Zero padding per axis, or `same` (stride 1) or `valid` (none). */
  readonly padding?: number | readonly number[] | 'same' | 'valid'
  readonly dilation?: number | readonly number[]
  readonly groups?: number
  readonly bias?: TensorLike
  /** How to pad (default zeros, as torch's padding_mode). */
  readonly paddingMode?: Extract<PadMode, 'constant' | 'reflect' | 'edge' | 'wrap'>
}

/**
 * N-dimensional convolution as in deep learning (torch.nn.functional.conv1d/2d/3d; a cross-correlation): input
 * `[B, C_in, *S]`, weight `[C_out, C_in / groups, *K]`, with stride, padding (or 'same'), dilation, groups and bias.
 * Built from im2col and a batched matrix product, so differentiable in input, weight and bias.
 *
 * @param input - Input `[B, C_in, *S]`.
 * @param weight - Kernels `[C_out, C_in / groups, *K]`.
 * @param options - Stride, padding, dilation, groups, bias and padding mode.
 * @returns `[B, C_out, *O]`.
 * @concept convolution
 * @index function Deep-learning convolution over any number of spatial axes.
 * @example
 * ```ts
 * import { convNd } from 'omai-engine/numerics/convolution'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(convNd([[[1, 2, 3, 4]]], [[[1, -1]]])) // => [[[-1, -1, -1]]]
 * ```
 */
export function convNd(input: TensorLike, weight: TensorLike, options: ConvOptions = {}): Value {
  const [x, w] = promoteFloat(input, weight) as [Value, Value]
  const d = x.shape.length - 2
  if (d < 1 || w.shape.length !== d + 2)
    throw new OmaiError('shape', 'convNd: input [B, C, *S] and weight [C_out, C_in / groups, *K] of matching rank')
  const groups = options.groups ?? 1
  const [b = 0, cin = 0] = x.shape
  const [cout = 0, cinPerGroup = 0, ...kernel] = w.shape
  if (cin !== cinPerGroup * groups || cout % groups !== 0)
    throw new OmaiError('shape', 'convNd: channels do not match the groups')
  const stride = expand(options.stride, d, 1)
  const dilation = expand(options.dilation, d, 1)
  let widths: [number, number][]
  if (options.padding === 'same') {
    if (stride.some((s) => s !== 1)) throw new OmaiError('value', "convNd: padding 'same' needs stride 1")
    widths = kernel.map((k, i) => {
      const total = (dilation[i] ?? 1) * (k - 1)
      return [Math.floor(total / 2), total - Math.floor(total / 2)]
    })
  } else {
    const p = options.padding === 'valid' ? 0 : options.padding
    widths = expand(p, d, 0).map((q): [number, number] => [q, q])
  }
  const padded = widths.some(([l, r]) => l > 0 || r > 0)
    ? padArray(x, [[0, 0], [0, 0], ...widths], options.paddingMode ?? 'constant')
    : x
  const cols = im2col(padded, kernel, { stride, dilation })
  const l = cols.shape[2] ?? 0
  const kc = cinPerGroup * product(kernel)
  const out = matmul(reshape(w, [1, groups, cout / groups, kc]), reshape(cols, [b, groups, kc, l]))
  const spatial = padded.shape
    .slice(2)
    .map((n, i) => convOutputSize(n, kernel[i] ?? 1, stride[i] ?? 1, 0, dilation[i] ?? 1))
  let result = reshape(out, [b, cout, ...spatial])
  if (options.bias !== undefined)
    result = add(result, reshape(promoteFloat(options.bias)[0] as Value, [1, cout, ...new Array<number>(d).fill(1)]))
  return result
}

/** Options of {@link convTranspose}. */
export interface ConvTransposeOptions {
  readonly stride?: number | readonly number[]
  readonly padding?: number | readonly number[]
  readonly outputPadding?: number | readonly number[]
  readonly dilation?: number | readonly number[]
  readonly groups?: number
  readonly bias?: TensorLike
}

/**
 * Transposed convolution (torch.nn.functional.conv_transpose1d/2d/3d), the gradient of convNd with respect to its
 * input: weight `[C_in, C_out / groups, *K]`; output length (n - 1) stride - 2 padding + dilation (k - 1) +
 * outputPadding + 1. Built from a matrix product and col2im.
 *
 * @param input - Input `[B, C_in, *S]`.
 * @param weight - Kernels `[C_in, C_out / groups, *K]`.
 * @param options - Stride, padding, output padding, dilation, groups and bias.
 * @returns `[B, C_out, *O]`.
 * @example
 * ```ts
 * import { convTranspose } from 'omai-engine/numerics/convolution'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(convTranspose([[[1, 2]]], [[[1, 1, 1]]], { stride: 2 })) // => [[[1, 1, 3, 2, 2]]]
 * ```
 */
export function convTranspose(input: TensorLike, weight: TensorLike, options: ConvTransposeOptions = {}): Value {
  const [x, w] = promoteFloat(input, weight) as [Value, Value]
  const d = x.shape.length - 2
  const groups = options.groups ?? 1
  const [b = 0, cin = 0, ...inSpatial] = x.shape
  const [wcin = 0, coutPerGroup = 0, ...kernel] = w.shape
  if (wcin !== cin || cin % groups !== 0)
    throw new OmaiError('shape', 'convTranspose: weight must be [C_in, C_out / groups, *K]')
  const stride = expand(options.stride, d, 1)
  const dilation = expand(options.dilation, d, 1)
  const padding = expand(options.padding, d, 0)
  const outputPadding = expand(options.outputPadding, d, 0)
  const full = inSpatial.map(
    (n, i) => (n - 1) * (stride[i] ?? 1) + (dilation[i] ?? 1) * ((kernel[i] ?? 1) - 1) + 1 + (outputPadding[i] ?? 0),
  )
  const cinPerGroup = cin / groups
  const kc = coutPerGroup * product(kernel)
  const l = product(inSpatial)
  // cols[g] = W[g]^T x[g]: [B, g, C_out/g * prod(K), L].
  const wt = swapaxes(reshape(w, [1, groups, cinPerGroup, kc]), 2, 3)
  const cols = matmul(wt, reshape(x, [b, groups, cinPerGroup, l]))
  let out = col2im(reshape(cols, [b, groups * kc, l]), full, kernel, { stride, dilation })
  padding.forEach((p, i) => {
    if (p > 0) out = sliceAxis(out, i + 2, p, (full[i] ?? 0) - p)
  })
  if (options.bias !== undefined) {
    out = add(
      out,
      reshape(promoteFloat(options.bias)[0] as Value, [1, groups * coutPerGroup, ...new Array<number>(d).fill(1)]),
    )
  }
  return out
}

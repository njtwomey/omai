/**
 * Image-shaped tensor operations, torch.nn.functional style: `interpolate` for resizing (nearest, linear, bicubic and
 * area), pixel shuffle and unshuffle for sub-pixel upsampling, `pad` with PyTorch's width order and modes, and ViT's
 * `patchEmbedding`, which cuts images into patches and embeds each one as a token.
 */

import { MdmlError } from '../../base/index.ts'
import { padArray } from '../../numerics/convolution/index.ts'
import {
  fromStorage,
  matmul,
  moveaxis,
  promoteFloat,
  reshape,
  swapaxes,
  type TensorLike,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { conv } from './conv.ts'
import { adaptiveAvgPool } from './pooling.ts'

/** Interpolation modes of {@link interpolate} (torch.nn.functional.interpolate). */
export type InterpolationMode = 'nearest' | 'nearest-exact' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'

/** Options of {@link interpolate}. */
export interface InterpolateOptions {
  /** The output spatial size (one number for every axis). */
  readonly size?: number | readonly number[]
  /** Or a scale factor per axis. */
  readonly scaleFactor?: number | readonly number[]
  readonly mode?: InterpolationMode
  /** Align the corner pixels' centres (linear and cubic modes). */
  readonly alignCorners?: boolean
  /** Compute the source positions from the output size rather than the given scale factor. */
  readonly recomputeScaleFactor?: boolean
}

// Keys's cubic convolution kernel with a = -0.75, as PyTorch's bicubic.
const A = -0.75
const cubic1 = (t: number): number => ((A + 2) * t - (A + 3)) * t * t + 1
const cubic2 = (t: number): number => ((A * t - 5 * A) * t + 8 * A) * t - 4 * A

/** The interpolation matrix `[out, in]` of one axis. */
function axisMatrix(
  n: number,
  out: number,
  mode: InterpolationMode,
  alignCorners: boolean,
  scale: number,
): Float64Array {
  const m = new Float64Array(out * n)
  const set = (o: number, i: number, w: number): void => {
    const j = Math.min(Math.max(i, 0), n - 1)
    m[o * n + j] = (m[o * n + j] ?? 0) + w
  }
  for (let o = 0; o < out; o++) {
    if (mode === 'nearest' || mode === 'nearest-exact') {
      const src = mode === 'nearest' ? Math.floor(o * scale) : Math.floor((o + 0.5) * scale)
      set(o, Math.min(src, n - 1), 1)
      continue
    }
    const linearScale = alignCorners ? (out > 1 ? (n - 1) / (out - 1) : 0) : scale
    let src = alignCorners ? o * linearScale : (o + 0.5) * linearScale - 0.5
    if (mode === 'bicubic') {
      const i = Math.floor(src)
      const t = src - i
      set(o, i - 1, cubic2(t + 1))
      set(o, i, cubic1(t))
      set(o, i + 1, cubic1(1 - t))
      set(o, i + 2, cubic2(2 - t))
      continue
    }
    if (src < 0) src = 0
    const i0 = Math.floor(src)
    const i1 = Math.min(i0 + 1, n - 1)
    const w = src - i0
    set(o, i0, 1 - w)
    set(o, i1, w)
  }
  return m
}

/**
 * Resize by interpolation (torch.nn.functional.interpolate) over any number of spatial axes: nearest, nearest-exact,
 * (bi/tri)linear, bicubic (Keys's kernel, a = -0.75) and area. Separable, so applied one axis at a time as a matrix
 * product, hence differentiable.
 *
 * @param input - `[B, C, *S]`.
 * @param options - Output size or scale factor, mode (default nearest), alignCorners, recomputeScaleFactor.
 * @returns `[B, C, *size]`.
 * @concept upsampling
 * @example
 * ```ts
 * import { interpolate } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(interpolate([[[1, 2]]], { scaleFactor: 2, mode: 'linear', alignCorners: true })) // => [[[1, 1.3333333333333333, 1.6666666666666665, 2]]]
 * ```
 */
export function interpolate(input: TensorLike, options: InterpolateOptions): Value {
  const [x] = promoteFloat(input) as [Value]
  const spatial = x.shape.slice(2)
  const d = spatial.length
  if (d < 1) throw new MdmlError('shape', 'interpolate: input must be [B, C, ...spatial]')
  const mode = options.mode ?? 'nearest'
  const factors =
    options.scaleFactor === undefined
      ? undefined
      : typeof options.scaleFactor === 'number'
        ? new Array<number>(d).fill(options.scaleFactor)
        : [...options.scaleFactor]
  const sizes =
    options.size !== undefined
      ? typeof options.size === 'number'
        ? new Array<number>(d).fill(options.size)
        : [...options.size]
      : factors !== undefined
        ? spatial.map((n, i) => Math.floor(n * (factors[i] ?? 1)))
        : undefined
  if (sizes === undefined) throw new MdmlError('value', 'interpolate: give a size or a scale factor')
  if (mode === 'area') return adaptiveAvgPool(x, sizes)
  const alignCorners = options.alignCorners ?? false
  let y = x
  for (let i = 0; i < d; i++) {
    const n = spatial[i] ?? 1
    const out = sizes[i] ?? 1
    const scale = factors !== undefined && options.recomputeScaleFactor !== true ? 1 / (factors[i] ?? 1) : n / out
    const m = axisMatrix(n, out, mode, alignCorners, scale)
    const moved = moveaxis(y, i + 2, -1)
    y = moveaxis(matmul(moved, transpose(fromStorage(m, [out, n], 'float64'))), -1, i + 2)
  }
  return y
}

/**
 * Rearrange channels into space (torch.nn.functional.pixel_shuffle; sub-pixel convolution upsampling):
 * `[B, C r^2, H, W]` to `[B, C, H r, W r]`.
 *
 * @param input - `[B, C r^2, H, W]`.
 * @param upscale - r.
 * @returns `[B, C, H r, W r]`.
 * @example
 * ```ts
 * import { pixelShuffle } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(pixelShuffle([[[[1]], [[2]], [[3]], [[4]]]], 2)) // => [[[[1, 2], [3, 4]]]]
 * ```
 */
export function pixelShuffle(input: TensorLike, upscale: number): Value {
  const [x] = promoteFloat(input) as [Value]
  const [b = 1, c = 1, h = 1, w = 1] = x.shape
  const r = upscale
  if (c % (r * r) !== 0) throw new MdmlError('shape', `pixelShuffle: ${c} channels do not divide by ${r * r}`)
  const oc = c / (r * r)
  const y = transpose(reshape(x, [b, oc, r, r, h, w]), [0, 1, 4, 2, 5, 3])
  return reshape(y, [b, oc, h * r, w * r])
}

/**
 * Rearrange space into channels (torch.nn.functional.pixel_unshuffle), the inverse of {@link pixelShuffle}.
 *
 * @param input - `[B, C, H r, W r]`.
 * @param downscale - r.
 * @returns `[B, C r^2, H, W]`.
 * @example
 * ```ts
 * import { pixelUnshuffle } from 'mdml-engine/nn'
 * pixelUnshuffle([[[[1, 2], [3, 4]]]], 2).shape // => [1, 4, 1, 1]
 * ```
 */
export function pixelUnshuffle(input: TensorLike, downscale: number): Value {
  const [x] = promoteFloat(input) as [Value]
  const [b = 1, c = 1, hr = 1, wr = 1] = x.shape
  const r = downscale
  if (hr % r !== 0 || wr % r !== 0) throw new MdmlError('shape', 'pixelUnshuffle: sizes must divide by the factor')
  const y = transpose(reshape(x, [b, c, hr / r, r, wr / r, r]), [0, 1, 3, 5, 2, 4])
  return reshape(y, [b, c * r * r, hr / r, wr / r])
}

/** Padding modes of {@link pad}. */
export type PadModeTorch = 'constant' | 'reflect' | 'replicate' | 'circular'

/**
 * Pad the trailing axes (torch.nn.functional.pad): widths in PyTorch's order, the last axis first
 * (`[left, right, top, bottom, front, back]`), with a constant, reflection, replication of the edge or wrapping.
 *
 * @param input - Input.
 * @param widths - Pairs for the trailing axes, last axis first.
 * @param mode - The mode (default constant).
 * @param value - The constant (default 0).
 * @returns The padded input.
 * @example
 * ```ts
 * import { pad } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(pad([[1, 2, 3]], [1, 2], 'replicate')) // => [[1, 1, 2, 3, 3, 3]]
 * ```
 */
export function pad(input: TensorLike, widths: readonly number[], mode: PadModeTorch = 'constant', value = 0): Value {
  const [x] = promoteFloat(input) as [Value]
  if (widths.length % 2 !== 0 || widths.length / 2 > x.shape.length)
    throw new MdmlError('value', 'pad: widths come in pairs, at most one pair per axis')
  const pairs: [number, number][] = x.shape.map((): [number, number] => [0, 0])
  for (let k = 0; k < widths.length / 2; k++) {
    pairs[x.shape.length - 1 - k] = [widths[2 * k] ?? 0, widths[2 * k + 1] ?? 0]
  }
  const numpy = mode === 'replicate' ? 'edge' : mode === 'circular' ? 'wrap' : mode
  return padArray(x, pairs, numpy, value)
}

/**
 * Cut images into non-overlapping patches and embed each linearly (ViT's patch embedding): a convolution with
 * kernel and stride equal to the patch size, flattened to a sequence of tokens.
 *
 * @param input - Images `[B, C, H, W]`.
 * @param weight - `[D, C, p, p]`.
 * @param bias - `[D]` (optional).
 * @returns Tokens `[B, (H / p)(W / p), D]`, in row-major patch order.
 * @concept vision-transformer
 * @reference Dosovitskiy et al. (2021), An image is worth 16x16 words.
 * @example
 * ```ts
 * import { patchEmbedding } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * patchEmbedding(zeros([2, 3, 8, 8]), zeros([16, 3, 4, 4])).shape // => [2, 4, 16]
 * ```
 */
export function patchEmbedding(input: TensorLike, weight: TensorLike, bias?: TensorLike): Value {
  const [w] = promoteFloat(weight) as [Value]
  const p = w.shape.slice(2)
  const y = conv(input, w, { stride: p, ...(bias === undefined ? {} : { bias }) })
  const [b = 1, dim = 1] = y.shape
  return swapaxes(reshape(y, [b, dim, -1]), 1, 2)
}

/**
 * Pooling as functions over any number of spatial axes (torch.nn.functional style): max pooling (optionally with
 * indices, and its partial inverse `maxUnpool`), average and power-average (Lp) pooling, adaptive pooling to a fixed
 * output size, and global average and max pooling.
 */

import { MdmlError } from '../../base/index.ts'
import { im2col, padArray } from '../../numerics/convolution/index.ts'
import {
  abs,
  argmax,
  concat,
  div,
  expandDims,
  fromStorage,
  matmul,
  max,
  mean,
  moveaxis,
  mul,
  pow,
  primalOf,
  promoteFloat,
  relu,
  reshape,
  scatterAdd,
  sign,
  sizeOf,
  sliceAxis,
  squeeze,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { SpatialSize } from './conv.ts'

/** Options of {@link maxPool}. */
export interface MaxPoolOptions {
  /** Stride (default: the kernel size). */
  readonly stride?: SpatialSize
  /** Implicit negative-infinity padding on both sides (at most half the kernel). */
  readonly padding?: SpatialSize
  readonly dilation?: SpatialSize
  /** Use ceil instead of floor for the output size (windows may overhang the end). */
  readonly ceilMode?: boolean
}

/** Options of {@link avgPool}. */
export interface AvgPoolOptions {
  readonly stride?: SpatialSize
  /** Implicit zero padding on both sides. */
  readonly padding?: SpatialSize
  readonly ceilMode?: boolean
  /** Count the zero padding in the divisor (default true, as PyTorch). */
  readonly countIncludePad?: boolean
  /** Divide by this instead of the window size. */
  readonly divisorOverride?: number
}

const spread = (v: SpatialSize | undefined, d: number, fallback: readonly number[] | number): number[] =>
  v === undefined
    ? typeof fallback === 'number'
      ? new Array<number>(d).fill(fallback)
      : [...fallback]
    : typeof v === 'number'
      ? new Array<number>(d).fill(v)
      : [...v]

interface Geometry {
  readonly d: number
  readonly kernel: number[]
  readonly stride: number[]
  readonly padding: number[]
  readonly dilation: number[]
  readonly spatial: number[]
  readonly out: number[]
  /** Extra padding at the end of each axis for ceil mode. */
  readonly extra: number[]
}

function geometry(
  x: Value,
  kernelSize: SpatialSize,
  options: { stride?: SpatialSize; padding?: SpatialSize; dilation?: SpatialSize; ceilMode?: boolean },
  name: string,
): Geometry {
  const d = typeof kernelSize === 'number' ? x.shape.length - 2 : kernelSize.length
  if (d < 1 || x.shape.length !== d + 2) throw new MdmlError('shape', `${name}: input must be [B, C, ...${d} spatial]`)
  const kernel = spread(kernelSize, d, 1)
  const stride = spread(options.stride, d, kernel)
  const padding = spread(options.padding, d, 0)
  const dilation = spread(options.dilation, d, 1)
  if (padding.some((p, i) => p > (kernel[i] ?? 1) / 2))
    throw new MdmlError('value', `${name}: padding must be at most half the kernel size`)
  const spatial = x.shape.slice(2)
  const out: number[] = []
  const extra: number[] = []
  for (let i = 0; i < d; i++) {
    const n = spatial[i] ?? 0
    const span = (dilation[i] ?? 1) * ((kernel[i] ?? 1) - 1) + 1
    const s = stride[i] ?? 1
    const p = padding[i] ?? 0
    const num = n + 2 * p - span
    let o = (options.ceilMode === true ? Math.ceil(num / s) : Math.floor(num / s)) + 1
    // A window must start inside the input or the left padding (PyTorch's rule).
    if (options.ceilMode === true && (o - 1) * s >= n + p) o -= 1
    if (o < 1) throw new MdmlError('shape', `${name}: the kernel is larger than the padded input`)
    out.push(o)
    extra.push(Math.max(0, (o - 1) * s + span - (n + 2 * p)))
  }
  return { d, kernel, stride, padding, dilation, spatial, out, extra }
}

/** Windows `[B, C, prod(K), prod(O)]` of the input padded with `fill` (padding and ceil-mode overhang). */
function windows(x: Value, g: Geometry, fill: number): Value {
  const widths: [number, number][] = g.padding.map((p, i) => [p, p + (g.extra[i] ?? 0)])
  const padded = widths.some(([a, b]) => a > 0 || b > 0)
    ? padArray(x, [[0, 0], [0, 0], ...widths], 'constant', fill)
    : x
  const cols = im2col(padded, g.kernel, { stride: g.stride, dilation: g.dilation })
  const [b = 0, c = 0] = x.shape
  return reshape(cols, [b, c, sizeOf(g.kernel), sizeOf(g.out)])
}

function unbatched(x: Value, kernelSize: SpatialSize, f: (v: Value) => Value): Value {
  if (typeof kernelSize !== 'number' && x.shape.length === kernelSize.length + 1) return squeeze(f(expandDims(x, 0)), 0)
  return f(x)
}

/**
 * Max pooling over any number of spatial axes (torch.nn.functional.max_pool1d/2d/3d): the largest value of each
 * window, with stride, negative-infinity padding, dilation and ceil mode. The number of spatial axes is the length of
 * `kernelSize`, or the input rank minus two.
 *
 * @param input - Input `[B, C, *S]` (or `[C, *S]` with an array kernel size).
 * @param kernelSize - Window size per axis.
 * @param options - Stride (default the kernel), padding, dilation and ceil mode.
 * @returns `[B, C, *O]`, differentiable (the gradient goes to each window's maximum).
 * @concept pooling
 * @example
 * ```ts
 * import { maxPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(maxPool([[[1, 3, 2, 5, 4]]], 2)) // => [[[3, 5]]]
 * ```
 */
export function maxPool(input: TensorLike, kernelSize: SpatialSize, options: MaxPoolOptions = {}): Value {
  const [x0] = promoteFloat(input) as [Value]
  return unbatched(x0, kernelSize, (x) => {
    const g = geometry(x, kernelSize, options, 'maxPool')
    const [b = 0, c = 0] = x.shape
    return reshape(max(windows(x, g, Number.NEGATIVE_INFINITY), 2), [b, c, ...g.out])
  })
}

/**
 * Max pooling with the position of each maximum (torch's `return_indices=True`): indices are flat positions within
 * each channel's unpadded spatial array, as {@link maxUnpool} expects.
 *
 * @param input - Input `[B, C, *S]`.
 * @param kernelSize - Window size per axis.
 * @param options - Stride, padding, dilation and ceil mode.
 * @returns The pooled values and int32 indices, both `[B, C, *O]`.
 * @example
 * ```ts
 * import { maxPoolWithIndices } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(maxPoolWithIndices([[[1, 3, 2, 5, 4]]], 2).indices) // => [[[1, 3]]]
 * ```
 */
export function maxPoolWithIndices(
  input: TensorLike,
  kernelSize: SpatialSize,
  options: MaxPoolOptions = {},
): { readonly values: Value; readonly indices: Tensor<'int32'> } {
  const [x] = promoteFloat(input) as [Value]
  const g = geometry(x, kernelSize, options, 'maxPoolWithIndices')
  const [b = 0, c = 0] = x.shape
  const w = windows(x, g, Number.NEGATIVE_INFINITY)
  const values = reshape(max(w, 2), [b, c, ...g.out])
  const arg = toFlat(argmax(primalOf(w), 2))
  const nk = sizeOf(g.kernel)
  const no = sizeOf(g.out)
  const idx = new Int32Array(arg.length)
  const kpos = new Array<number>(g.d).fill(0)
  const opos = new Array<number>(g.d).fill(0)
  for (let r = 0; r < arg.length; r++) {
    let a = (arg[r] ?? 0) % nk
    let o = r % no
    for (let i = g.d - 1; i >= 0; i--) {
      kpos[i] = a % (g.kernel[i] ?? 1)
      a = Math.floor(a / (g.kernel[i] ?? 1))
      opos[i] = o % (g.out[i] ?? 1)
      o = Math.floor(o / (g.out[i] ?? 1))
    }
    let flat = 0
    for (let i = 0; i < g.d; i++) {
      const coord = (opos[i] ?? 0) * (g.stride[i] ?? 1) + (kpos[i] ?? 0) * (g.dilation[i] ?? 1) - (g.padding[i] ?? 0)
      flat = flat * (g.spatial[i] ?? 1) + coord
    }
    idx[r] = flat
  }
  return { values, indices: fromStorage(idx, [b, c, ...g.out], 'int32') }
}

/** Options of {@link maxUnpool}. */
export interface MaxUnpoolOptions {
  readonly stride?: SpatialSize
  readonly padding?: SpatialSize
  /** The spatial output size (default (S - 1) stride - 2 padding + kernel). */
  readonly outputSize?: readonly number[]
}

/**
 * The partial inverse of max pooling (torch.nn.functional.max_unpool1d/2d/3d): each value goes back to the position
 * its index names, every other position is zero.
 *
 * @param input - Pooled values `[B, C, *O]`.
 * @param indices - Flat positions from {@link maxPoolWithIndices}.
 * @param kernelSize - The pooling window.
 * @param options - Stride, padding and output size of the pooling.
 * @returns `[B, C, *S]`, differentiable in `input`.
 * @example
 * ```ts
 * import { maxUnpool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(maxUnpool([[[3, 5]]], [[[1, 3]]], [2])) // => [[[0, 3, 0, 5]]]
 * ```
 */
export function maxUnpool(
  input: TensorLike,
  indices: TensorLike,
  kernelSize: SpatialSize,
  options: MaxUnpoolOptions = {},
): Value {
  const [x] = promoteFloat(input) as [Value]
  const d = typeof kernelSize === 'number' ? x.shape.length - 2 : kernelSize.length
  const kernel = spread(kernelSize, d, 1)
  const stride = spread(options.stride, d, kernel)
  const padding = spread(options.padding, d, 0)
  const [b = 0, c = 0, ...pooled] = x.shape
  const size =
    options.outputSize?.slice(-d) ??
    pooled.map((n, i) => (n - 1) * (stride[i] ?? 1) - 2 * (padding[i] ?? 0) + (kernel[i] ?? 1))
  const l = sizeOf(pooled)
  const out = scatterAdd([b, c, sizeOf(size)], reshape(indices, [b, c, l]), reshape(x, [b, c, l]), 2)
  return reshape(out, [b, c, ...size])
}

/**
 * Average pooling over any number of spatial axes (torch.nn.functional.avg_pool1d/2d/3d), with stride, zero padding
 * (counted in the divisor or not), ceil mode and a divisor override.
 *
 * @param input - Input `[B, C, *S]` (or `[C, *S]` with an array kernel size).
 * @param kernelSize - Window size per axis.
 * @param options - Stride (default the kernel), padding, ceil mode, countIncludePad and divisorOverride.
 * @returns `[B, C, *O]`.
 * @concept pooling
 * @example
 * ```ts
 * import { avgPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(avgPool([[[1, 3, 2, 6]]], 2)) // => [[[2, 4]]]
 * ```
 */
export function avgPool(input: TensorLike, kernelSize: SpatialSize, options: AvgPoolOptions = {}): Value {
  const [x0] = promoteFloat(input) as [Value]
  return unbatched(x0, kernelSize, (x) => {
    const g = geometry(x, kernelSize, options, 'avgPool')
    const [b = 0, c = 0] = x.shape
    const total = reshape(sum(windows(x, g, 0), 2), [b, c, ...g.out])
    if (options.divisorOverride !== undefined) return div(total, options.divisorOverride)
    const include = options.countIncludePad ?? true
    // The divisor of each output position: the window clipped to the padded input (or to the input itself).
    const counts = new Float64Array(sizeOf(g.out))
    for (let r = 0; r < counts.length; r++) {
      let rem = r
      let count = 1
      for (let i = g.d - 1; i >= 0; i--) {
        const o = rem % (g.out[i] ?? 1)
        rem = Math.floor(rem / (g.out[i] ?? 1))
        const n = g.spatial[i] ?? 0
        const p = g.padding[i] ?? 0
        const start = o * (g.stride[i] ?? 1) - p
        const end = Math.min(start + (g.kernel[i] ?? 1), n + p)
        count *= include ? end - start : Math.min(end, n) - Math.max(start, 0)
      }
      counts[r] = count
    }
    return div(total, fromStorage(counts, g.out, 'float64'))
  })
}

/** Options of {@link lpPool}. */
export interface LpPoolOptions {
  readonly stride?: SpatialSize
  readonly ceilMode?: boolean
}

/**
 * Power-average (Lp) pooling (torch.nn.functional.lp_pool1d/2d/3d): (sum of x^p over each window)^(1/p); p = 1 is sum
 * pooling and p = infinity max pooling.
 *
 * @param input - Input `[B, C, *S]`.
 * @param normType - The power p.
 * @param kernelSize - Window size per axis.
 * @param options - Stride (default the kernel) and ceil mode.
 * @returns `[B, C, *O]`.
 * @example
 * ```ts
 * import { lpPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lpPool([[[3, 4]]], 2, 2)) // => [[[5]]]
 * ```
 */
export function lpPool(
  input: TensorLike,
  normType: number,
  kernelSize: SpatialSize,
  options: LpPoolOptions = {},
): Value {
  const [x] = promoteFloat(input) as [Value]
  if (normType === Number.POSITIVE_INFINITY) return maxPool(x, kernelSize, options)
  const d = typeof kernelSize === 'number' ? x.shape.length - 2 : kernelSize.length
  const k = sizeOf(spread(kernelSize, d, 1))
  const avg = avgPool(pow(x, normType), kernelSize, options)
  // As PyTorch: sign(out) relu(|out|) times the window size, to the power 1 / p.
  return pow(mul(mul(sign(avg), relu(abs(avg))), k), 1 / normType)
}

/** Start and end of each adaptive window along an axis (PyTorch's floor and ceil rule). */
function adaptiveBounds(n: number, o: number): [number, number][] {
  return Array.from({ length: o }, (_, i) => [Math.floor((i * n) / o), Math.ceil(((i + 1) * n) / o)])
}

const outputSizes = (x: Value, outputSize: number | readonly (number | null)[]): number[] => {
  const spatial = x.shape.slice(2)
  if (typeof outputSize === 'number') return spatial.map(() => outputSize)
  return outputSize.map((o, i) => o ?? spatial[spatial.length - outputSize.length + i] ?? 1)
}

function adaptive(
  input: TensorLike,
  outputSize: number | readonly (number | null)[],
  step: (y: Value, axis: number, n: number, o: number) => Value,
): Value {
  const [x0] = promoteFloat(input) as [Value]
  const added = typeof outputSize !== 'number' && x0.shape.length === outputSize.length + 1
  const x = added ? expandDims(x0, 0) : x0
  let y = x
  for (const [i, o] of outputSizes(x, outputSize).entries()) y = step(y, i + 2, y.shape[i + 2] ?? 0, o)
  return added ? squeeze(y, 0) : y
}

/**
 * Adaptive average pooling (torch.nn.functional.adaptive_avg_pool1d/2d/3d): windows chosen so the output has the
 * given size, window i spanning [floor(i n / o), ceil((i + 1) n / o)). Separable, so computed one axis at a time as a
 * matrix product.
 *
 * @param input - Input `[B, C, *S]` (or `[C, *S]` with an array size).
 * @param outputSize - The output size per spatial axis (one number for all; `null` keeps an axis).
 * @returns `[B, C, *outputSize]`.
 * @concept pooling
 * @example
 * ```ts
 * import { adaptiveAvgPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(adaptiveAvgPool([[[1, 2, 3, 4, 5]]], 2)) // => [[[2, 4]]]
 * ```
 */
export function adaptiveAvgPool(input: TensorLike, outputSize: number | readonly (number | null)[]): Value {
  return adaptive(input, outputSize, (y, axis, n, o) => {
    // The averaging matrix [n, o]: move the axis last, multiply, move it back.
    const m = new Float64Array(n * o)
    for (const [r, [s, e]] of adaptiveBounds(n, o).entries()) for (let j = s; j < e; j++) m[j * o + r] = 1 / (e - s)
    return moveaxis(matmul(moveaxis(y, axis, -1), fromStorage(m, [n, o], 'float64')), -1, axis)
  })
}

/**
 * Adaptive max pooling (torch.nn.functional.adaptive_max_pool1d/2d/3d): the maximum of each adaptive window. The max
 * over a box is the max over each axis in turn, so it is computed axis by axis.
 *
 * @param input - Input `[B, C, *S]`.
 * @param outputSize - The output size per spatial axis (one number for all; `null` keeps an axis).
 * @returns `[B, C, *outputSize]`.
 * @example
 * ```ts
 * import { adaptiveMaxPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(adaptiveMaxPool([[[1, 5, 3, 2, 4]]], 2)) // => [[[5, 4]]]
 * ```
 */
export function adaptiveMaxPool(input: TensorLike, outputSize: number | readonly (number | null)[]): Value {
  return adaptive(input, outputSize, (y, axis, n, o) =>
    concat(
      adaptiveBounds(n, o).map(([s, e]) => max(sliceAxis(y, axis, s, e), axis, true)),
      axis,
    ),
  )
}

/**
 * Global average pooling: the mean over every spatial axis (the head of most convolutional classifiers).
 *
 * @param input - Input `[B, C, *S]`.
 * @param keepDims - Keep the spatial axes with size 1 (default false: `[B, C]`).
 * @returns `[B, C]` or `[B, C, 1, ...]`.
 * @example
 * ```ts
 * import { globalAvgPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(globalAvgPool([[[[1, 2], [3, 6]]]])) // => [[3]]
 * ```
 */
export function globalAvgPool(input: TensorLike, keepDims = false): Value {
  const [x] = promoteFloat(input) as [Value]
  return mean(
    x,
    x.shape.slice(2).map((_, i) => i + 2),
    keepDims,
  )
}

/**
 * Global max pooling: the maximum over every spatial axis.
 *
 * @param input - Input `[B, C, *S]`.
 * @param keepDims - Keep the spatial axes with size 1 (default false).
 * @returns `[B, C]` or `[B, C, 1, ...]`.
 * @example
 * ```ts
 * import { globalMaxPool } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(globalMaxPool([[[[1, 2], [3, 6]]]])) // => [[6]]
 * ```
 */
export function globalMaxPool(input: TensorLike, keepDims = false): Value {
  const [x] = promoteFloat(input) as [Value]
  return max(
    x,
    x.shape.slice(2).map((_, i) => i + 2),
    keepDims,
  )
}

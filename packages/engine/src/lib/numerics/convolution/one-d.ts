import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  flip,
  fromStorage,
  matmul,
  pad,
  promoteFloat,
  reshape,
  sliceAxis,
  type TensorLike,
  take,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { cMul } from '../fourier/complex.ts'
import { fftn, ifftn } from '../fourier/fft.ts'
import { nextFastLen } from '../fourier/helpers.ts'

/** Output extent of a convolution: the full overlap, the size of the first input, or only complete overlaps. */
export type ConvolutionMode = 'full' | 'same' | 'valid'

function vector(x: TensorLike, name: string): Value {
  const [v] = promoteFloat(x) as [Value]
  if (v.shape.length !== 1 || (v.shape[0] ?? 0) === 0)
    throw new WhiteboxError('shape', `${name}: inputs must be non-empty 1-D arrays`)
  return v
}

/** Crop a full-length result of length nFull to the mode (numpy's convention, centred like scipy's _centered). */
function cropRange(n1: number, n2: number, mode: ConvolutionMode): [number, number] {
  const full = n1 + n2 - 1
  if (mode === 'full') return [0, full]
  if (mode === 'same') {
    const size = Math.max(n1, n2)
    const start = Math.floor((full - size) / 2)
    return [start, start + size]
  }
  const size = Math.max(n1, n2) - Math.min(n1, n2) + 1
  const start = Math.min(n1, n2) - 1
  return [start, start + size]
}

/**
 * Discrete linear convolution of two 1-D arrays (numpy.convolve): (a * v)_k = sum_j a_j v_(k - j), by sliding windows
 * and a matrix product, so differentiable in both inputs.
 *
 * @param a - First input.
 * @param v - Second input.
 * @param mode - full (default), same or valid.
 * @returns The convolution.
 * @concept convolution
 * @example
 * ```ts
 * import { convolve } from 'whitebox-engine/numerics/convolution'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(convolve([1, 2, 3], [0, 1, 0.5])) // => [0, 1, 2.5, 4, 1.5]
 * ```
 */
export function convolve(a: TensorLike, v: TensorLike, mode: ConvolutionMode = 'full'): Value {
  const x = vector(a, 'convolve')
  const k = vector(v, 'convolve')
  const n = x.shape[0] ?? 0
  const m = k.shape[0] ?? 0
  const [start, stop] = cropRange(n, m, mode)
  // Windows of the zero-padded signal times the reversed kernel.
  const padded = pad(x, [[m - 1, m - 1]])
  const count = stop - start
  const idx = new Int32Array(count * m)
  for (let r = 0; r < count; r++) for (let j = 0; j < m; j++) idx[r * m + j] = start + r + j
  const windows = reshape(take(padded, fromStorage(idx, [count * m], 'int32'), 0), [count, m])
  return matmul(windows, flip(k, 0))
}

/**
 * Cross-correlation of two 1-D arrays (numpy.correlate): c_k = sum_n a_(n + k) v_n, which is the convolution with v
 * reversed.
 *
 * @param a - First input.
 * @param v - Second input.
 * @param mode - valid (default, as numpy), same or full.
 * @returns The correlation.
 * @example
 * ```ts
 * import { correlate } from 'whitebox-engine/numerics/convolution'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(correlate([1, 2, 3], [0, 1, 0.5], 'full')) // => [0.5, 2, 3.5, 3, 0]
 * ```
 */
export function correlate(a: TensorLike, v: TensorLike, mode: ConvolutionMode = 'valid'): Value {
  return convolve(a, flip(vector(v, 'correlate'), 0), mode)
}

/**
 * The lags of each output of a cross-correlation (scipy.signal.correlation_lags).
 *
 * @param n1 - Length of the first input.
 * @param n2 - Length of the second input.
 * @param mode - The correlation mode (default full).
 * @returns The lags.
 * @example
 * ```ts
 * import { correlationLags } from 'whitebox-engine/numerics/convolution'
 * correlationLags(3, 2) // => [-1, 0, 1, 2]
 * ```
 */
export function correlationLags(n1: number, n2: number, mode: ConvolutionMode = 'full'): number[] {
  const full = Array.from({ length: n1 + n2 - 1 }, (_, i) => i - (n2 - 1))
  if (mode === 'full') return full
  if (mode === 'same') {
    const mid = Math.floor(full.length / 2)
    const bound = Math.floor(n1 / 2)
    return full.slice(mid - bound, mid + bound + (n1 % 2))
  }
  const bound = n1 - n2
  return bound >= 0
    ? Array.from({ length: bound + 1 }, (_, i) => i)
    : Array.from({ length: 1 - bound }, (_, i) => bound + i)
}

/** Crop each axis of a full n-d result to the mode, as scipy.signal (same: the shape of in1, centred). */
function cropNd(
  full: Value,
  s1: readonly number[],
  s2: readonly number[],
  axes: readonly number[],
  mode: ConvolutionMode,
): Value {
  let out = full
  for (const a of axes) {
    const n1 = s1[a] ?? 0
    const n2 = s2[a] ?? 0
    const total = n1 + n2 - 1
    let start = 0
    let size = total
    if (mode === 'same') {
      size = n1
      start = Math.floor((total - size) / 2)
    } else if (mode === 'valid') {
      size = Math.max(n1 - n2, n2 - n1) + 1
      start = Math.floor((total - size) / 2)
    }
    out = sliceAxis(out, a, start, start + size)
  }
  return out
}

/**
 * Convolution of two n-dimensional arrays by FFT (scipy.signal.fftconvolve): zero-pad to a fast length, multiply the
 * spectra, invert. O(N log N) instead of O(N M); differentiable in both inputs.
 *
 * @param in1 - First input.
 * @param in2 - Second input (same rank).
 * @param mode - full (default), same (the shape of in1) or valid.
 * @param axes - Axes to convolve over (default all; the others broadcast).
 * @returns The convolution.
 * @example
 * ```ts
 * import { fftConvolve } from 'whitebox-engine/numerics/convolution'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fftConvolve([1, 2, 3], [0, 1, 0.5])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 2.5, 4, 1.5]
 * ```
 */
export function fftConvolve(
  in1: TensorLike,
  in2: TensorLike,
  mode: ConvolutionMode = 'full',
  axes?: readonly number[],
): Value {
  const [a, b] = promoteFloat(in1, in2) as [Value, Value]
  if (a.shape.length !== b.shape.length) throw new WhiteboxError('shape', 'fftConvolve: inputs must have the same rank')
  const list = (axes ?? a.shape.map((_, i) => i)).map((x) => (x < 0 ? x + a.shape.length : x))
  const sizes = list.map((ax) => nextFastLen((a.shape[ax] ?? 0) + (b.shape[ax] ?? 0) - 1))
  const product = cMul(fftn(a, sizes, list), fftn(b, sizes, list))
  const full = ifftn(product, undefined, list).re
  let trimmed = full
  list.forEach((ax) => {
    trimmed = sliceAxis(trimmed, ax, 0, (a.shape[ax] ?? 0) + (b.shape[ax] ?? 0) - 1)
  })
  return cropNd(trimmed, a.shape, b.shape, list, mode)
}

/**
 * Convolution by the overlap-add method (scipy.signal.oaconvolve), for a long signal and a short kernel in one
 * dimension: FFT-convolve blocks of the signal and add the overlapping tails. Same result as fftConvolve; memory is
 * per block.
 *
 * @param in1 - Signal (1-D).
 * @param in2 - Kernel (1-D).
 * @param mode - full (default), same or valid.
 * @returns The convolution.
 * @example
 * ```ts
 * import { oaConvolve } from 'whitebox-engine/numerics/convolution'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(oaConvolve([1, 2, 3, 4, 5, 6], [1, -1])).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1, 1, 1, 1, 1, -6]
 * ```
 */
export function oaConvolve(in1: TensorLike, in2: TensorLike, mode: ConvolutionMode = 'full'): Value {
  const x = vector(in1, 'oaConvolve')
  const k = vector(in2, 'oaConvolve')
  const n1 = x.shape[0] ?? 0
  const n2 = k.shape[0] ?? 0
  const full = n2 > n1 ? overlapAdd(k, x) : overlapAdd(x, k)
  return cropNd(full, [n1], [n2], [0], mode)
}

/** The full convolution of a long signal with a shorter kernel, block by block. */
function overlapAdd(x: Value, k: Value): Value {
  const n = x.shape[0] ?? 0
  const m = k.shape[0] ?? 0
  // Block length chosen so that each FFT has a fast size of about 8 m.
  const fftLen = nextFastLen(Math.max(8 * m, 2 * m))
  const block = fftLen - m + 1
  let out: Value = zeros([n + m - 1])
  for (let start = 0; start < n; start += block) {
    const stop = Math.min(n, start + block)
    const piece = fftConvolve(sliceAxis(x, 0, start, stop), k)
    const len = piece.shape[0] ?? 0
    out = add(out, pad(piece, [[start, n + m - 1 - start - len]]))
  }
  return out
}

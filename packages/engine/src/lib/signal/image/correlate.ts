import { WhiteboxError } from '../../base/index.ts'
import { type BorderMode, padArray, padModeOf } from '../../numerics/convolution/index.ts'
import {
  add,
  flip,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  type TensorLike,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { axisOf, float } from '../_shared.ts'

// Correlation and convolution of n-dimensional arrays with scipy.ndimage's border modes and origins: the shared
// machinery of the image filters, the Savitzky-Golay filter and the separable smoothing filters. Results are sums of
// shifted copies of the border-extended input, so they are differentiable in the input and in the weights.

/** Options shared by the ndimage-style filters. */
export interface BorderOptions {
  /** How the input is extended beyond its edges (default 'reflect', which repeats the edge sample). */
  readonly mode?: BorderMode
  /** The value outside the edges for mode 'constant' (default 0). */
  readonly cval?: number
}

function checkOrigin(origin: number, size: number, name: string): void {
  if (!Number.isInteger(origin) || origin < -Math.floor(size / 2) || origin > Math.floor((size - 1) / 2))
    throw new WhiteboxError('value', `${name}: the origin must satisfy -(size // 2) <= origin <= (size - 1) // 2`)
}

/** Extend one axis by `before` and `after` samples under an ndimage border mode. */
function extend(x: Value, axis: number, before: number, after: number, options: BorderOptions): Value {
  if (before === 0 && after === 0) return x
  const widths = x.shape.map((_, i): [number, number] => (i === axis ? [before, after] : [0, 0]))
  return padArray(x, widths, padModeOf(options.mode ?? 'reflect'), options.cval ?? 0)
}

/**
 * One-dimensional correlation along an axis (scipy.ndimage.correlate1d): out[i] = sum_j w[j] x[i + j - k // 2 -
 * origin] over the border-extended input. Differentiable in x and the weights.
 *
 * @param x - Input.
 * @param weights - Weights `[k]`.
 * @param axis - Axis (default -1).
 * @param options - Border mode, constant and origin (default 0, centred).
 * @returns The correlation, the shape of x.
 * @throws WhiteboxError with code `value` for an invalid origin.
 * @concept correlation
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(image.correlate1d([1, 2, 3, 4], [1, 0, -1])) // => [-1, -2, -2, -1]
 * ```
 */
export function correlate1d(
  x: TensorLike,
  weights: TensorLike,
  axis = -1,
  options: BorderOptions & { readonly origin?: number } = {},
): Value {
  const v = float(x)
  const w = float(weights)
  if (w.shape.length !== 1 || (w.shape[0] ?? 0) < 1) throw new WhiteboxError('shape', 'correlate1d: weights are 1-D')
  const a = axisOf(axis, v.shape.length, 'correlate1d')
  const k = w.shape[0] ?? 1
  const origin = options.origin ?? 0
  checkOrigin(origin, k, 'correlate1d')
  const size1 = Math.floor(k / 2)
  const size2 = k - size1 - 1
  const n = v.shape[a] ?? 0
  const ext = extend(v, a, size1 + origin, size2 - origin, options)
  let out: Value | null = null
  for (let j = 0; j < k; j++) {
    const term = mul(sliceAxis(ext, a, j, j + n), reshape(sliceAxis(w, 0, j, j + 1), []))
    out = out === null ? term : add(out, term)
  }
  return out ?? zeros(v.shape)
}

/**
 * One-dimensional convolution along an axis (scipy.ndimage.convolve1d): correlation with the reversed weights, the
 * origin mirrored (and shifted by one for even lengths).
 *
 * @param x - Input.
 * @param weights - Weights `[k]`.
 * @param axis - Axis (default -1).
 * @param options - Border mode, constant and origin.
 * @returns The convolution, the shape of x.
 * @concept convolution
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(image.convolve1d([1, 2, 3, 4], [1, 0, -1])) // => [1, 2, 2, 1]
 * ```
 */
export function convolve1d(
  x: TensorLike,
  weights: TensorLike,
  axis = -1,
  options: BorderOptions & { readonly origin?: number } = {},
): Value {
  const w = float(weights)
  const k = w.shape[0] ?? 1
  let origin = -(options.origin ?? 0)
  if (k % 2 === 0) origin -= 1
  return correlate1d(x, flip(w, 0), axis, { ...options, origin })
}

function origins(origin: number | readonly number[] | undefined, rank: number): number[] {
  if (origin === undefined) return new Array<number>(rank).fill(0)
  if (typeof origin === 'number') return new Array<number>(rank).fill(origin)
  if (origin.length !== rank) throw new WhiteboxError('shape', `expected ${rank} origins`)
  return [...origin]
}

/**
 * Multidimensional correlation (scipy.ndimage.correlate): out[i] = sum_j w[j] x[i + j - k // 2 - origin] over every
 * axis, with the input extended at its borders. Differentiable in x and the weights.
 *
 * @param x - Input.
 * @param weights - Weights of the same rank as x.
 * @param options - Border mode, constant and origin (one per axis or shared).
 * @returns The correlation, the shape of x.
 * @concept correlation
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(image.correlateNd([[1, 2], [3, 4]], [[0, 1], [0, 0]], { mode: 'constant' })) // => [[0, 0], [1, 2]]
 * ```
 */
export function correlateNd(
  x: TensorLike,
  weights: TensorLike,
  options: BorderOptions & { readonly origin?: number | readonly number[] } = {},
): Value {
  const v = float(x)
  const w = float(weights)
  const rank = v.shape.length
  if (w.shape.length !== rank) throw new WhiteboxError('shape', 'correlateNd: weights must have the rank of the input')
  const o = origins(options.origin, rank)
  let ext = v
  for (let d = 0; d < rank; d++) {
    const k = w.shape[d] ?? 1
    checkOrigin(o[d] ?? 0, k, 'correlateNd')
    const size1 = Math.floor(k / 2)
    ext = extend(ext, d, size1 + (o[d] ?? 0), k - size1 - 1 - (o[d] ?? 0), options)
  }
  const wflat = reshape(w, [-1])
  const wvals = toFlat(primalOf(w))
  const total = wvals.length
  let out: Value | null = null
  const idx = new Array<number>(rank).fill(0)
  for (let flat = 0; flat < total; flat++) {
    let rem = flat
    for (let d = rank - 1; d >= 0; d--) {
      const s = w.shape[d] ?? 1
      idx[d] = rem % s
      rem = Math.floor(rem / s)
    }
    let piece = ext
    for (let d = 0; d < rank; d++) piece = sliceAxis(piece, d, idx[d] ?? 0, (idx[d] ?? 0) + (v.shape[d] ?? 0))
    const term = mul(piece, reshape(sliceAxis(wflat, 0, flat, flat + 1), []))
    out = out === null ? term : add(out, term)
  }
  return out ?? zeros(v.shape)
}

/**
 * Multidimensional convolution (scipy.ndimage.convolve): correlation with the weights reversed on every axis and the
 * origins mirrored (shifted by one on even-length axes).
 *
 * @param x - Input.
 * @param weights - Weights of the same rank as x.
 * @param options - Border mode, constant and origin.
 * @returns The convolution, the shape of x.
 * @concept convolution
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(image.convolveNd([[1, 2], [3, 4]], [[0, 1], [0, 0]], { mode: 'constant' })) // => [[3, 4], [0, 0]]
 * ```
 */
export function convolveNd(
  x: TensorLike,
  weights: TensorLike,
  options: BorderOptions & { readonly origin?: number | readonly number[] } = {},
): Value {
  const w = float(weights)
  const rank = w.shape.length
  const o = origins(options.origin, rank).map((v, d) => -v - ((w.shape[d] ?? 1) % 2 === 0 ? 1 : 0))
  return correlateNd(
    x,
    flip(
      w,
      w.shape.map((_, i) => i),
    ),
    { ...options, origin: o },
  )
}

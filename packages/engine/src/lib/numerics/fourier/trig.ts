/**
 * Real trigonometric transforms, scipy.fft style: the discrete cosine and sine transforms of types 1 to 4 and their
 * inverses, computed through FFTs and differentiable, and the modified discrete cosine transform (MDCT) with a sine
 * window and overlap-add inverse, the lapped transform behind audio codecs.
 */

import { MdmlError } from '../../base/index.ts'
import {
  add,
  concat,
  flip,
  fromStorage,
  matmul,
  moveaxis,
  mul,
  neg,
  pad,
  promoteFloat,
  reshape,
  sliceAxis,
  type Tensor,
  type TensorLike,
  type Value,
  zerosLike,
} from '../../tensor/index.ts'
import type { FftNorm } from './fft.ts'
import { fft, ifft } from './fft.ts'

/** The four DCT and DST types of scipy.fft. */
export type TrigType = 1 | 2 | 3 | 4

const vector = (values: readonly number[]): Tensor => fromStorage(Float64Array.from(values), [values.length], 'float64')
const last = (x: Value, start: number, stop: number): Value => sliceAxis(x, -1, start, stop)
const alternating = (n: number): Tensor => vector(Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1 : -1)))

/** Unnormalised DCT along the last axis, built from FFTs (so it is differentiable). */
function rawDct(x: Value, type: TrigType): Value {
  const n = x.shape[x.shape.length - 1] ?? 0
  switch (type) {
    case 1: {
      if (n < 2) throw new MdmlError('value', 'dct type 1 needs at least two points')
      const v = n === 2 ? x : concat([x, flip(last(x, 1, n - 1), -1)], -1)
      return last(fft(v).re, 0, n)
    }
    case 2: {
      // y_k = Re(exp(-i pi k / 2N) FFT([x, reverse(x)])_k).
      const spectrum = fft(concat([x, flip(x, -1)], -1))
      const angle = Array.from({ length: n }, (_, k) => (Math.PI * k) / (2 * n))
      return add(
        mul(last(spectrum.re, 0, n), vector(angle.map(Math.cos))),
        mul(last(spectrum.im, 0, n), vector(angle.map(Math.sin))),
      )
    }
    case 3: {
      // The transpose of type 2 with weight 1/2 on x_0: y_k = 2 Re(sum_n w_n x_n exp(i pi n (2k + 1) / 2N)).
      const weights = Array.from({ length: n }, (_, i) => (i === 0 ? 0.5 : 1))
      const angle = Array.from({ length: n }, (_, i) => (Math.PI * i) / (2 * n))
      const wx = mul(x, vector(weights))
      const z = { re: mul(wx, vector(angle.map(Math.cos))), im: mul(wx, vector(angle.map(Math.sin))) }
      return mul(last(ifft(z, 2 * n, -1, 'forward').re, 0, n), 2)
    }
    case 4: {
      // y_k = 2 Re(exp(-i pi (2k + 1) / 4N) FFT_2N(x_n exp(-i pi n / 2N))_k).
      const angle = Array.from({ length: n }, (_, i) => (Math.PI * i) / (2 * n))
      const z = { re: mul(x, vector(angle.map(Math.cos))), im: neg(mul(x, vector(angle.map(Math.sin)))) }
      const spectrum = fft(z, 2 * n)
      const post = Array.from({ length: n }, (_, k) => (Math.PI * (2 * k + 1)) / (4 * n))
      return mul(
        add(
          mul(last(spectrum.re, 0, n), vector(post.map(Math.cos))),
          mul(last(spectrum.im, 0, n), vector(post.map(Math.sin))),
        ),
        2,
      )
    }
  }
}

/** Unnormalised DST along the last axis, through the DCT identities. */
function rawDst(x: Value, type: TrigType): Value {
  const n = x.shape[x.shape.length - 1] ?? 0
  switch (type) {
    case 1: {
      // -Im FFT([0, x, 0, -reverse(x)])_{k + 1}.
      const zero = mul(last(x, 0, 1), 0)
      const v = concat([zero, x, zero, neg(flip(x, -1))], -1)
      return neg(last(fft(v).im, 1, n + 1))
    }
    case 2:
      return flip(rawDct(mul(x, alternating(n)), 2), -1)
    case 3:
      return mul(rawDct(flip(x, -1), 3), alternating(n))
    case 4:
      return mul(rawDct(flip(x, -1), 4), alternating(n))
  }
}

function factor(kind: 'dct' | 'dst', type: TrigType, n: number): number {
  if (type === 1) return kind === 'dct' ? 2 * (n - 1) : 2 * (n + 1)
  return 2 * n
}

/** The orthonormal transform (scipy's norm='ortho', orthogonalised). */
function orthoTransform(kind: 'dct' | 'dst', x: Value, type: TrigType): Value {
  const n = x.shape[x.shape.length - 1] ?? 0
  const raw = kind === 'dct' ? rawDct : rawDst
  const ones = (): number[] => new Array<number>(n).fill(1)
  switch (type) {
    case 1: {
      if (kind === 'dst') return mul(raw(x, 1), 1 / Math.sqrt(2 * (n + 1)))
      const pre = ones()
      pre[0] = Math.SQRT2
      pre[n - 1] = Math.SQRT2
      const post = ones().map(() => 1 / Math.sqrt(2 * (n - 1)))
      post[0] = (post[0] ?? 1) / Math.SQRT2
      post[n - 1] = (post[n - 1] ?? 1) / Math.SQRT2
      return mul(raw(mul(x, vector(pre)), 1), vector(post))
    }
    case 2: {
      const post = ones().map(() => 1 / Math.sqrt(2 * n))
      const edge = kind === 'dct' ? 0 : n - 1
      post[edge] = (post[edge] ?? 1) / Math.SQRT2
      return mul(raw(x, 2), vector(post))
    }
    case 3: {
      const pre = ones().map(() => 1 / Math.sqrt(2 * n))
      pre[kind === 'dct' ? 0 : n - 1] = 1 / Math.sqrt(n)
      return raw(mul(x, vector(pre)), 3)
    }
    case 4:
      return mul(raw(x, 4), 1 / Math.sqrt(2 * n))
  }
}

const INVERSE: Readonly<Record<TrigType, TrigType>> = { 1: 1, 2: 3, 3: 2, 4: 4 }

function prepare(x: TensorLike, n: number | undefined, axis: number): { v: Value; restore: (y: Value) => Value } {
  const [xv] = promoteFloat(x) as [Value]
  const rank = xv.shape.length
  const a = axis < 0 ? axis + rank : axis
  let v = moveaxis(xv, a, -1)
  const len = v.shape[rank - 1] ?? 0
  if (n !== undefined && n !== len) {
    v =
      n < len
        ? last(v, 0, n)
        : pad(
            v,
            v.shape.map((_, i): [number, number] => (i === rank - 1 ? [0, n - len] : [0, 0])),
          )
  }
  return { v, restore: (y) => moveaxis(y, -1, a) }
}

function transform(
  kind: 'dct' | 'dst',
  x: TensorLike,
  type: TrigType,
  n: number | undefined,
  axis: number,
  norm: FftNorm,
  inverse: boolean,
): Value {
  const { v, restore } = prepare(x, n, axis)
  const len = v.shape[v.shape.length - 1] ?? 0
  const t = inverse ? INVERSE[type] : type
  if (norm === 'ortho') return restore(orthoTransform(kind, v, t))
  const raw = (kind === 'dct' ? rawDct : rawDst)(v, t)
  const f = factor(kind, type, len)
  const divide = inverse ? norm === 'backward' : norm === 'forward'
  return restore(divide ? mul(raw, 1 / f) : raw)
}

/**
 * The discrete cosine transform of type 1, 2 (the DCT), 3 or 4 along an axis (scipy.fft.dct), computed through FFTs and
 * differentiable. Type 2: y_k = 2 sum_n x_n cos(pi k (2n + 1) / 2N).
 *
 * @param x - Input.
 * @param type - DCT type (default 2).
 * @param options - Length n (crop or pad), axis (default -1) and normalisation (backward, ortho or forward).
 * @returns The coefficients.
 * @concept discrete-cosine-transform
 * @example
 * ```ts
 * import { dct } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(dct([1, 1, 1, 1])).map((v) => Math.round(v * 1e12) / 1e12) // => [8, 0, 0, 0]
 * ```
 */
export function dct(
  x: TensorLike,
  type: TrigType = 2,
  options: { readonly n?: number; readonly axis?: number; readonly norm?: FftNorm } = {},
): Value {
  return transform('dct', x, type, options.n, options.axis ?? -1, options.norm ?? 'backward', false)
}

/**
 * The inverse DCT (scipy.fft.idct): idct(dct(x, t), t) = x for every type and normalisation.
 *
 * @param x - Coefficients.
 * @param type - DCT type of the forward transform (default 2).
 * @param options - Length, axis and normalisation.
 * @returns The signal.
 * @example
 * ```ts
 * import { dct, idct } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(idct(dct([1, 2, 3]))).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2, 3]
 * ```
 */
export function idct(
  x: TensorLike,
  type: TrigType = 2,
  options: { readonly n?: number; readonly axis?: number; readonly norm?: FftNorm } = {},
): Value {
  return transform('dct', x, type, options.n, options.axis ?? -1, options.norm ?? 'backward', true)
}

/**
 * The discrete sine transform of type 1 to 4 along an axis (scipy.fft.dst). Type 2:
 * y_k = 2 sum_n x_n sin(pi (k + 1)(2n + 1) / 2N).
 *
 * @param x - Input.
 * @param type - DST type (default 2).
 * @param options - Length, axis and normalisation.
 * @returns The coefficients.
 * @example
 * ```ts
 * import { dst } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(dst([1, 0, 0, 0], 1)).map((v) => Math.round(v * 1e12) / 1e12) // => [1.175570504585, 1.902113032590, 1.902113032590, 1.175570504585]
 * ```
 */
export function dst(
  x: TensorLike,
  type: TrigType = 2,
  options: { readonly n?: number; readonly axis?: number; readonly norm?: FftNorm } = {},
): Value {
  return transform('dst', x, type, options.n, options.axis ?? -1, options.norm ?? 'backward', false)
}

/**
 * The inverse DST (scipy.fft.idst).
 *
 * @param x - Coefficients.
 * @param type - DST type of the forward transform (default 2).
 * @param options - Length, axis and normalisation.
 * @returns The signal.
 * @example
 * ```ts
 * import { dst, idst } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(idst(dst([1, 2, 3], 4), 4)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2, 3]
 * ```
 */
export function idst(
  x: TensorLike,
  type: TrigType = 2,
  options: { readonly n?: number; readonly axis?: number; readonly norm?: FftNorm } = {},
): Value {
  return transform('dst', x, type, options.n, options.axis ?? -1, options.norm ?? 'backward', true)
}

/** The MDCT basis M[n, k] = cos(pi / N (n + 1/2 + N/2)(k + 1/2)), `[2N, N]`. */
function mdctBasis(n: number): Tensor {
  const m = new Float64Array(2 * n * n)
  for (let i = 0; i < 2 * n; i++)
    for (let k = 0; k < n; k++) m[i * n + k] = Math.cos((Math.PI / n) * (i + 0.5 + n / 2) * (k + 0.5))
  return fromStorage(m, [2 * n, n], 'float64')
}

/** The sine window sin(pi (n + 1/2) / 2N), which satisfies the Princen-Bradley condition w_n^2 + w_{n+N}^2 = 1. */
export function sineWindow(length: number): Tensor {
  return vector(Array.from({ length }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / length)))
}

/**
 * The modified discrete cosine transform of a signal: frames of 2N samples with hop N (the signal padded with N zeros
 * at each end), windowed and projected onto the MDCT basis. With a Princen-Bradley window, {@link imdct} reconstructs
 * the signal exactly by time-domain aliasing cancellation, although each frame is critically sampled (N coefficients
 * for N new samples). Differentiable.
 *
 * @param x - Signal `[L]`, L a multiple of N.
 * @param n - Coefficients per frame N.
 * @param window - Window of length 2N (default the sine window).
 * @returns Coefficients `[frames, N]`.
 * @concept mdct
 * @example
 * ```ts
 * import { mdct } from 'mdml-engine/numerics/fourier'
 * mdct(new Array(16).fill(1), 4).shape // => [5, 4]
 * ```
 */
export function mdct(x: TensorLike, n: number, window?: TensorLike): Value {
  const [xv] = promoteFloat(x) as [Value]
  const len = xv.shape[0] ?? 0
  if (len % n !== 0) throw new MdmlError('value', 'mdct: the signal length must be a multiple of N')
  const w = window === undefined ? sineWindow(2 * n) : (promoteFloat(window)[0] as Value)
  const padded = pad(xv, [[n, n]])
  const frames = len / n + 1
  const rows = Array.from({ length: frames }, (_, f) =>
    reshape(mul(sliceAxis(padded, 0, f * n, f * n + 2 * n), w), [1, 2 * n]),
  )
  return matmul(concat(rows, 0), mdctBasis(n))
}

/**
 * The inverse MDCT with overlap-add (time-domain aliasing cancellation): frames (2 / N) M X, windowed, overlapped by N
 * and stripped of the padding.
 *
 * @param coefficients - `[frames, N]` from {@link mdct}.
 * @param window - The window used by {@link mdct} (default sine).
 * @returns The signal `[(frames - 1) N]`.
 * @example
 * ```ts
 * import { imdct, mdct } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(imdct(mdct([1, 2, 3, 4, 5, 6, 7, 8], 4))).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2, 3, 4, 5, 6, 7, 8]
 * ```
 */
export function imdct(coefficients: TensorLike, window?: TensorLike): Value {
  const [c] = promoteFloat(coefficients) as [Value]
  const frames = c.shape[0] ?? 0
  const n = c.shape[1] ?? 0
  const w = window === undefined ? sineWindow(2 * n) : (promoteFloat(window)[0] as Value)
  const basis = mdctBasis(n)
  const blocks = mul(mul(matmul(c, transpose2(basis)), 2 / n), reshape(w, [1, 2 * n]))
  // Overlap-add: the second half of frame f plus the first half of frame f + 1.
  const pieces: Value[] = []
  for (let f = 0; f + 1 < frames; f++) {
    const tail = reshape(sliceAxis(sliceAxis(blocks, 0, f, f + 1), 1, n, 2 * n), [n])
    const head = reshape(sliceAxis(sliceAxis(blocks, 0, f + 1, f + 2), 1, 0, n), [n])
    pieces.push(add(tail, head))
  }
  return pieces.length === 0 ? zerosLike(fromStorage(new Float64Array(0), [0], 'float64')) : concat(pieces, 0)
}

function transpose2(m: Tensor): Tensor {
  const [r = 0, c = 0] = m.shape
  const out = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = m.data[i * c + j] ?? 0
  return fromStorage(out, [c, r], 'float64')
}

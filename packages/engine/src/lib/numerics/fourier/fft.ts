import { WhiteboxError } from '../../base/index.ts'
import {
  batchAtFront,
  bind,
  type ComplexLike,
  type ComplexValue,
  complex,
  concat,
  definePrimitive,
  flip,
  fromStorage,
  neg,
  type Primitive,
  pad,
  sliceAxis,
  type Tensor,
  type Value,
  zerosLike,
} from '../../tensor/index.ts'
import { transformAxis } from './_kernel.ts'

/** Normalisation of a transform pair, as numpy.fft: which direction carries the 1 / n. */
export type FftNorm = 'backward' | 'ortho' | 'forward'

interface FftParams {
  readonly axis: number
  readonly sign: 1 | -1
  readonly scale: number
}

/**
 * The DFT along one axis of a complex pair: X_k = scale sum_n x_n exp(sign 2 pi i k n / N). Linear, so its tangent is
 * the same transform and its cotangent the conjugate transform (opposite sign, same scale).
 */
const fftPrimitive: Primitive<FftParams> = definePrimitive<FftParams>({
  name: 'fft',
  abstractEval: ([re, im]) => [re as NonNullable<typeof re>, im as NonNullable<typeof im>],
  impl: ([re, im], { axis, sign, scale }) => {
    const r = re as Tensor
    const out = transformAxis(
      Float64Array.from(r.data),
      Float64Array.from((im as Tensor).data),
      r.shape,
      axis,
      sign,
      scale,
    )
    return [fromStorage(out.re, r.shape, 'float64'), fromStorage(out.im, r.shape, 'float64')]
  },
  jvp: ([re, im], [dre, dim], params) => {
    if (dre === null && dim === null) return [null, null]
    const [tr, ti] = bind(fftPrimitive, [dre ?? zerosLike(re as Value), dim ?? zerosLike(im as Value)], params)
    return [tr as Value, ti as Value]
  },
  vjp: ([gre, gim], [re, im], params) => {
    if (gre === null && gim === null) return [null, null]
    const [cr, ci] = bind(fftPrimitive, [gre ?? zerosLike(re as Value), gim ?? zerosLike(im as Value)], {
      ...params,
      sign: params.sign === 1 ? -1 : 1,
    })
    return [cr as Value, ci as Value]
  },
  batch: (xs, ds, params, n) => {
    const outputs = bind(fftPrimitive, batchAtFront(xs, ds, n), { ...params, axis: params.axis + 1 })
    return { outputs, dims: [0, 0] }
  },
  describe: ({ axis, sign }) => `fft(axis=${axis}, sign=${sign})`,
})

function normaliseAxis(axis: number, rank: number): number {
  const a = axis < 0 ? axis + rank : axis
  if (a < 0 || a >= rank) throw new WhiteboxError('value', `fft: axis ${axis} out of range for rank ${rank}`)
  return a
}

/** Crop or zero-pad along an axis to length n. */
function resize(x: Value, axis: number, n: number): Value {
  const len = x.shape[axis] ?? 0
  if (n === len) return x
  if (n < len) return sliceAxis(x, axis, 0, n)
  const widths = x.shape.map((_, i): [number, number] => (i === axis ? [0, n - len] : [0, 0]))
  return pad(x, widths)
}

function scaleFor(norm: FftNorm, n: number, inverse: boolean): number {
  if (norm === 'ortho') return 1 / Math.sqrt(n)
  if (norm === 'backward') return inverse ? 1 / n : 1
  return inverse ? 1 : 1 / n
}

function run(x: ComplexLike, n: number | undefined, axis: number, norm: FftNorm, inverse: boolean): ComplexValue {
  const z = complex(x)
  const a = normaliseAxis(axis, z.re.shape.length)
  const len = n ?? z.re.shape[a] ?? 0
  if (len < 1) throw new WhiteboxError('value', 'fft: the transform length must be positive')
  const re = resize(z.re, a, len)
  const im = resize(z.im, a, len)
  const [or, oi] = bind(fftPrimitive, [re, im], {
    axis: a,
    sign: inverse ? 1 : -1,
    scale: scaleFor(norm, len, inverse),
  })
  return { re: or as Value, im: oi as Value }
}

/**
 * The discrete Fourier transform along an axis (numpy.fft.fft): X_k = sum_n x_n exp(-2 pi i k n / N), by radix-2 for
 * powers of two and Bluestein's algorithm otherwise (O(N log N) for every length). Differentiable.
 *
 * @param x - Real or complex input.
 * @param n - Transform length (crop or zero-pad; default the axis length).
 * @param axis - Axis (default -1).
 * @param norm - Normalisation (default backward: no scaling forward).
 * @returns The spectrum as a complex pair.
 * @concept fourier-transform
 * @example
 * ```ts
 * import { fft } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fft([1, 2, 3, 4]).re) // => [10, -2, -2, -2]
 * ```
 */
export function fft(x: ComplexLike, n?: number, axis = -1, norm: FftNorm = 'backward'): ComplexValue {
  return run(x, n, axis, norm, false)
}

/**
 * The inverse DFT (numpy.fft.ifft): x_n = (1 / N) sum_k X_k exp(2 pi i k n / N).
 *
 * @param x - Spectrum.
 * @param n - Transform length.
 * @param axis - Axis (default -1).
 * @param norm - Normalisation (default backward: 1 / N here).
 * @returns The signal as a complex pair.
 * @example
 * ```ts
 * import { ifft } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(ifft({ re: [10, -2, -2, -2], im: [0, 2, 0, -2] }).re) // => [1, 2, 3, 4]
 * ```
 */
export function ifft(x: ComplexLike, n?: number, axis = -1, norm: FftNorm = 'backward'): ComplexValue {
  return run(x, n, axis, norm, true)
}

/**
 * The DFT of real input, non-negative frequencies only (numpy.fft.rfft): n / 2 + 1 terms.
 *
 * @param x - Real input.
 * @param n - Transform length.
 * @param axis - Axis (default -1).
 * @param norm - Normalisation.
 * @returns The half spectrum.
 * @example
 * ```ts
 * import { rfft } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(rfft([1, 2, 3, 4]).im) // => [0, 2, 0]
 * ```
 */
export function rfft(x: ComplexLike, n?: number, axis = -1, norm: FftNorm = 'backward'): ComplexValue {
  const z = complex(x)
  const a = normaliseAxis(axis, z.re.shape.length)
  const full = run({ re: z.re, im: zerosLike(z.re) }, n, a, norm, false)
  const len = n ?? z.re.shape[a] ?? 0
  const half = Math.floor(len / 2) + 1
  return { re: sliceAxis(full.re, a, 0, half), im: sliceAxis(full.im, a, 0, half) }
}

/**
 * The inverse of {@link rfft} (numpy.fft.irfft): the real signal of length n (default 2 (m - 1)) whose spectrum has
 * the given non-negative half, extended by Hermitian symmetry.
 *
 * @param x - Half spectrum.
 * @param n - Output length.
 * @param axis - Axis (default -1).
 * @param norm - Normalisation.
 * @returns The real signal.
 * @example
 * ```ts
 * import { irfft } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(irfft({ re: [10, -2, -2], im: [0, 2, 0] })) // => [1, 2, 3, 4]
 * ```
 */
export function irfft(x: ComplexLike, n?: number, axis = -1, norm: FftNorm = 'backward'): Value {
  const z = complex(x)
  const a = normaliseAxis(axis, z.re.shape.length)
  const m = z.re.shape[a] ?? 0
  const len = n ?? 2 * (m - 1)
  if (len < 1) throw new WhiteboxError('value', 'irfft: the output length must be positive')
  const half = Math.floor(len / 2) + 1
  const re = resize(z.re, a, half)
  const im = resize(z.im, a, half)
  // X_{len - k} = conj(X_k) for k = 1 .. len - half. The imaginary parts at DC and Nyquist only add purely imaginary
  // terms, so taking the real part ignores them, as numpy does.
  const upper = len - half
  const fullRe = upper > 0 ? concat([re, flip(sliceAxis(re, a, 1, 1 + upper), a)], a) : re
  const fullIm = upper > 0 ? concat([im, neg(flip(sliceAxis(im, a, 1, 1 + upper), a))], a) : im
  return run({ re: fullRe, im: fullIm }, len, a, norm, true).re
}

/**
 * The DFT of a Hermitian-symmetric signal given by its first half (numpy.fft.hfft): a real spectrum of length n.
 *
 * @param x - Half of the signal.
 * @param n - Output length (default 2 (m - 1)).
 * @param axis - Axis (default -1).
 * @param norm - Normalisation.
 * @returns The real spectrum.
 * @example
 * ```ts
 * import { hfft } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(hfft([1, 2, 3])) // => [8, -2, 0, -2]
 * ```
 */
export function hfft(x: ComplexLike, n?: number, axis = -1, norm: FftNorm = 'backward'): Value {
  const z = complex(x)
  const a = normaliseAxis(axis, z.re.shape.length)
  const len = n ?? 2 * ((z.re.shape[a] ?? 0) - 1)
  // hfft(x, n) = irfft(conj(x), n) with the opposite normalisation.
  const swapped: FftNorm = norm === 'backward' ? 'forward' : norm === 'forward' ? 'backward' : 'ortho'
  return irfft({ re: z.re, im: neg(z.im) }, len, a, swapped)
}

/**
 * The N-dimensional DFT over the given axes (numpy.fft.fftn; all axes by default).
 *
 * @param x - Input.
 * @param shape - Transform lengths per axis (default the input's).
 * @param axes - Axes (default all, or the last len(shape)).
 * @param norm - Normalisation.
 * @returns The spectrum.
 * @example
 * ```ts
 * import { fftn } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fftn([[1, 2], [3, 4]]).re) // => [[10, -2], [-4, 0]]
 * ```
 */
export function fftn(
  x: ComplexLike,
  shape?: readonly number[],
  axes?: readonly number[],
  norm: FftNorm = 'backward',
): ComplexValue {
  return multi(x, shape, axes, norm, false)
}

/**
 * The N-dimensional inverse DFT (numpy.fft.ifftn).
 *
 * @param x - Spectrum.
 * @param shape - Transform lengths per axis.
 * @param axes - Axes.
 * @param norm - Normalisation.
 * @returns The signal.
 * @example
 * ```ts
 * import { ifftn } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(ifftn({ re: [[10, -2], [-4, 0]], im: [[0, 0], [0, 0]] }).re) // => [[1, 2], [3, 4]]
 * ```
 */
export function ifftn(
  x: ComplexLike,
  shape?: readonly number[],
  axes?: readonly number[],
  norm: FftNorm = 'backward',
): ComplexValue {
  return multi(x, shape, axes, norm, true)
}

function multi(
  x: ComplexLike,
  shape: readonly number[] | undefined,
  axes: readonly number[] | undefined,
  norm: FftNorm,
  inverse: boolean,
): ComplexValue {
  let z = complex(x)
  const rank = z.re.shape.length
  const list =
    axes ??
    (shape === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : Array.from({ length: shape.length }, (_, i) => rank - shape.length + i))
  list.forEach((axis, i) => {
    z = run(z, shape?.[i], axis, norm, inverse)
  })
  return z
}

/**
 * The two-dimensional DFT over the last two axes (numpy.fft.fft2).
 *
 * @param x - Input.
 * @param norm - Normalisation.
 * @returns The spectrum.
 * @example
 * ```ts
 * import { fft2 } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(fft2([[1, 0], [0, 0]]).re) // => [[1, 1], [1, 1]]
 * ```
 */
export function fft2(x: ComplexLike, norm: FftNorm = 'backward'): ComplexValue {
  return multi(x, undefined, [-2, -1], norm, false)
}

/**
 * The two-dimensional inverse DFT over the last two axes (numpy.fft.ifft2).
 *
 * @param x - Spectrum.
 * @param norm - Normalisation.
 * @returns The signal.
 * @example
 * ```ts
 * import { ifft2 } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(ifft2({ re: [[1, 1], [1, 1]], im: [[0, 0], [0, 0]] }).re) // => [[1, 0], [0, 0]]
 * ```
 */
export function ifft2(x: ComplexLike, norm: FftNorm = 'backward'): ComplexValue {
  return multi(x, undefined, [-2, -1], norm, true)
}

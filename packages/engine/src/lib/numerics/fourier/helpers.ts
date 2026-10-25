/**
 * Companions to the FFT: sample frequencies (`fftfreq`, `rfftfreq`), shifting the zero frequency to the centre, fast
 * transform lengths (`nextFastLen`), the explicit DFT matrix, and the O(n^2) DFT by its definition as a reference for
 * teaching and checking.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type ComplexValue,
  fromStorage,
  promoteFloat,
  roll,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/**
 * The sample frequencies of a length-n DFT with sample spacing d (numpy.fft.fftfreq): [0, 1, ..., ceil(n/2) - 1,
 * -floor(n/2), ..., -1] / (d n).
 *
 * @param n - Window length.
 * @param d - Sample spacing (default 1).
 * @returns Frequencies `[n]`.
 * @example
 * ```ts
 * import { fftfreq } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fftfreq(4)) // => [0, 0.25, -0.5, -0.25]
 * ```
 */
export function fftfreq(n: number, d = 1): Tensor {
  return fromStorage(
    Float64Array.from({ length: n }, (_, k) => (k < Math.ceil(n / 2) ? k : k - n) / (d * n)),
    [n],
    'float64',
  )
}

/**
 * The non-negative sample frequencies of a real DFT (numpy.fft.rfftfreq): [0, 1, ..., floor(n/2)] / (d n).
 *
 * @param n - Window length.
 * @param d - Sample spacing.
 * @returns Frequencies `[floor(n/2) + 1]`.
 * @example
 * ```ts
 * import { rfftfreq } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(rfftfreq(4, 0.5)) // => [0, 0.5, 1]
 * ```
 */
export function rfftfreq(n: number, d = 1): Tensor {
  return fromStorage(
    Float64Array.from({ length: Math.floor(n / 2) + 1 }, (_, k) => k / (d * n)),
    [Math.floor(n / 2) + 1],
    'float64',
  )
}

function shiftAxes(x: Value, axes: number | readonly number[] | undefined, inverse: boolean): Value {
  const list = axes === undefined ? x.shape.map((_, i) => i) : typeof axes === 'number' ? [axes] : axes
  let out = x
  for (const a of list) {
    const n = x.shape[a < 0 ? a + x.shape.length : a] ?? 0
    out = roll(out, inverse ? -Math.floor(n / 2) : Math.floor(n / 2), a)
  }
  return out
}

/**
 * Move the zero frequency to the centre (numpy.fft.fftshift).
 *
 * @param x - Values (or a complex pair).
 * @param axes - Axes (default all).
 * @returns The shifted values.
 * @example
 * ```ts
 * import { fftshift, fftfreq } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(fftshift(fftfreq(5))) // => [-0.4, -0.2, 0, 0.2, 0.4]
 * ```
 */
export function fftshift<T extends TensorLike | ComplexValue>(
  x: T,
  axes?: number | readonly number[],
): T extends ComplexValue ? ComplexValue : Value
export function fftshift(x: TensorLike | ComplexValue, axes?: number | readonly number[]): Value | ComplexValue {
  if (typeof x === 'object' && x !== null && 're' in x && 'im' in x && !Array.isArray(x)) {
    const z = x as ComplexValue
    return { re: shiftAxes(z.re, axes, false), im: shiftAxes(z.im, axes, false) }
  }
  return shiftAxes(promoteFloat(x as TensorLike)[0] as Value, axes, false)
}

/**
 * The inverse of {@link fftshift} (numpy.fft.ifftshift).
 *
 * @param x - Values (or a complex pair).
 * @param axes - Axes (default all).
 * @returns The unshifted values.
 * @example
 * ```ts
 * import { ifftshift } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(ifftshift([-2, -1, 0, 1, 2])) // => [0, 1, 2, -2, -1]
 * ```
 */
export function ifftshift<T extends TensorLike | ComplexValue>(
  x: T,
  axes?: number | readonly number[],
): T extends ComplexValue ? ComplexValue : Value
export function ifftshift(x: TensorLike | ComplexValue, axes?: number | readonly number[]): Value | ComplexValue {
  if (typeof x === 'object' && x !== null && 're' in x && 'im' in x && !Array.isArray(x)) {
    const z = x as ComplexValue
    return { re: shiftAxes(z.re, axes, true), im: shiftAxes(z.im, axes, true) }
  }
  return shiftAxes(promoteFloat(x as TensorLike)[0] as Value, axes, true)
}

/**
 * The smallest fast transform length at least target (scipy.fft.next_fast_len): a product of 2, 3, 5, 7 and 11 for
 * complex transforms, of 2, 3 and 5 for real ones.
 *
 * @param target - Minimum length.
 * @param real - Whether the transform is real (default false).
 * @returns The length.
 * @example
 * ```ts
 * import { nextFastLen } from 'mdml-engine/numerics/fourier'
 * nextFastLen(97) // => 98
 * ```
 */
export function nextFastLen(target: number, real = false): number {
  if (!(target >= 1)) throw new MdmlError('value', 'nextFastLen: target must be at least 1')
  const primes = real ? [2, 3, 5] : [2, 3, 5, 7, 11]
  for (let n = Math.ceil(target); ; n++) {
    let m = n
    for (const p of primes) while (m % p === 0) m /= p
    if (m === 1) return n
  }
}

/** exp(-2 pi i m / n), exact at multiples of pi / 2. */
function unitRoot(m: number, n: number): [number, number] {
  if ((4 * m) % n === 0) {
    const quarter = ((4 * m) / n) % 4
    return (
      [
        [1, 0],
        [0, -1],
        [-1, 0],
        [0, 1],
      ] as const
    )[quarter]?.slice() as [number, number]
  }
  const angle = (-2 * Math.PI * m) / n
  return [Math.cos(angle), Math.sin(angle)]
}

/**
 * The DFT matrix (scipy.linalg.dft): F[j, k] = exp(-2 pi i j k / n), optionally scaled by 1/n or 1/sqrt(n).
 *
 * @param n - Size.
 * @param scale - `none` (default), `n` or `sqrtn`.
 * @returns The matrix as a complex pair.
 * @example
 * ```ts
 * import { dftMatrix } from 'mdml-engine/numerics/fourier'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(dftMatrix(2).re) // => [[1, 1], [1, -1]]
 * ```
 */
export function dftMatrix(n: number, scale: 'none' | 'n' | 'sqrtn' = 'none'): { re: Tensor; im: Tensor } {
  const s = scale === 'n' ? 1 / n : scale === 'sqrtn' ? 1 / Math.sqrt(n) : 1
  const re = new Float64Array(n * n)
  const im = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      const [c, si] = unitRoot((j * k) % n, n)
      re[j * n + k] = s * c
      im[j * n + k] = s * si
    }
  }
  return { re: fromStorage(re, [n, n], 'float64'), im: fromStorage(im, [n, n], 'float64') }
}

/**
 * The DFT by its definition, O(n^2), along the last axis of a real or complex 1-D input: a reference for teaching and
 * checking.
 *
 * @param re - Real parts.
 * @param im - Imaginary parts (default zeros).
 * @returns The spectrum.
 * @example
 * ```ts
 * import { dft } from 'mdml-engine/numerics/fourier'
 * dft([1, 2, 3, 4]).re // => [10, -2, -2, -2]
 * ```
 */
export function dft(re: readonly number[], im: readonly number[] = []): { re: number[]; im: number[] } {
  const n = re.length
  const outRe = new Array<number>(n).fill(0)
  const outIm = new Array<number>(n).fill(0)
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const [c, s] = unitRoot((k * t) % n, n)
      const xr = re[t] ?? 0
      const xi = im[t] ?? 0
      outRe[k] = (outRe[k] ?? 0) + xr * c - xi * s
      outIm[k] = (outIm[k] ?? 0) + xr * s + xi * c
    }
  }
  return { re: outRe, im: outIm }
}

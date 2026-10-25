/**
 * The stationary (undecimated) wavelet transform and its inverse (pywt.swt, iswt): the DWT without downsampling,
 * upsampling the filters instead, so every level keeps the signal's length and the transform is shift-invariant.
 */

import { WhiteboxError } from '../../base/index.ts'
import { add, matmul, mul, reshape, sliceAxis, type TensorLike, take, type Value } from '../../tensor/index.ts'
import { constant, float, indexTensor, rowsOf } from '../_shared.ts'
import { type DiscreteWavelet, type WaveletSpec, wavelet } from './families.ts'

// The stationary (undecimated, a trous) wavelet transform as pywt.swt and iswt: circular convolution with the filters
// upsampled by 2^(j - 1) at level j, no downsampling, so every level has the signal's length.

/** Circular correlation-convolution of rows with a filter upsampled by `step`, shifted as pywt: out[i] = sum_k
 * f[k] x[(i + shift - k step) mod n]. */
function circular(rows: Value, filter: Float64Array, step: number, shift: number, transpose: boolean): Value {
  const n = rows.shape[1] ?? 0
  const f = filter.length
  const idx: number[] = []
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < f; k++) {
      const j = transpose ? i - shift + k * step : i + shift - k * step
      idx.push(((j % n) + n) % n)
    }
  }
  const batch = rows.shape[0] ?? 1
  const windows = reshape(take(rows, indexTensor(idx), 1), [batch, n, f])
  return reshape(matmul(windows, constant(filter, [f, 1])), [batch, n])
}

function levelShift(f: number, j: number): number {
  return 2 ** (j - 1) * (f / 2)
}

/** Options for {@link swt}. */
export interface SwtOptions {
  /** Number of levels (default the largest with n divisible by 2^level). */
  readonly level?: number
  /** First level (default 0). */
  readonly startLevel?: number
  readonly axis?: number
}

/**
 * The stationary wavelet transform (pywt.swt): at each level the approximation is filtered (circularly) by the
 * decomposition filters upsampled by 2^(j - 1), without downsampling. The signal length must be a multiple of
 * 2^level. Differentiable.
 *
 * @param x - Data.
 * @param w - The wavelet.
 * @param options - Levels, start level and axis.
 * @returns [(cA_n, cD_n), ..., (cA_1, cD_1)], coarsest first, each the signal's length.
 * @throws WhiteboxError with code `value` when the length is not a multiple of 2^level.
 * @concept stationary-wavelet-transform
 * @example
 * ```ts
 * import { swt } from 'whitebox-engine/signal'
 * swt([1, 2, 3, 4, 5, 6, 7, 8], 'haar', { level: 2 }).map((c) => c.cA.shape[0]) // => [8, 8]
 * ```
 */
export function swt(
  x: TensorLike,
  w: WaveletSpec,
  options: SwtOptions = {},
): { readonly cA: Value; readonly cD: Value }[] {
  const wv = wavelet(w)
  const r = rowsOf(float(x), options.axis ?? -1, 'swt')
  const n = r.n
  let maxLevel = 0
  while (n % 2 ** (maxLevel + 1) === 0 && 2 ** (maxLevel + 1) <= n) maxLevel++
  const start = options.startLevel ?? 0
  const level = options.level ?? maxLevel - start
  if (n % 2 ** (start + level) !== 0)
    throw new WhiteboxError('value', `swt: the length must be a multiple of 2^${start + level}`)
  const f = wv.decLo.length
  const out: { cA: Value; cD: Value }[] = []
  let a = r.rows
  for (let j = start + 1; j <= start + level; j++) {
    const step = 2 ** (j - 1)
    const shift = levelShift(f, j)
    const cA = circular(a, wv.decLo, step, shift, false)
    const cD = circular(a, wv.decHi, step, shift, false)
    out.push({ cA: r.restore(cA), cD: r.restore(cD) })
    a = cA
  }
  return out.reverse()
}

/**
 * The inverse stationary wavelet transform (pywt.iswt): at each level, the average of the reconstructions from the
 * two cosets, which for the undecimated filter bank is half the sum of the reconstruction-filtered approximation and
 * detail.
 *
 * @param coeffs - [(cA_n, cD_n), ..., (cA_1, cD_1)] (only the coarsest cA is used).
 * @param w - The wavelet.
 * @param options - Axis and start level (as given to swt).
 * @returns The reconstruction.
 * @concept stationary-wavelet-transform
 * @example
 * ```ts
 * import { iswt, swt } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(iswt(swt([3, 1, 4, 1, 5, 9, 2, 6], 'db2', { level: 2 }), 'db2')).map((v) => Math.round(v * 1e12) / 1e12) // => [3, 1, 4, 1, 5, 9, 2, 6]
 * ```
 */
export function iswt(
  coeffs: readonly { readonly cA: TensorLike; readonly cD: TensorLike }[],
  w: WaveletSpec,
  options: { readonly axis?: number; readonly startLevel?: number } = {},
): Value {
  const first = coeffs[0]
  if (first === undefined) throw new WhiteboxError('value', 'iswt: no coefficients')
  const wv: DiscreteWavelet = wavelet(w)
  const axis = options.axis ?? -1
  const r = rowsOf(float(first.cA), axis, 'iswt')
  const f = wv.recLo.length
  let a = r.rows
  const start = options.startLevel ?? 0
  const levels = coeffs.length
  for (const [i, c] of coeffs.entries()) {
    const j = start + levels - i
    const step = 2 ** (j - 1)
    // The adjoint of out[i] = sum_k f[k] x[i + s - k step] with the reconstruction filters.
    const shift = levelShift(f, j)
    const d = rowsOf(float(c.cD), axis, 'iswt').rows
    const lo = circular(a, Float64Array.from(wv.recLo).reverse(), step, shift, true)
    const hi = circular(d, Float64Array.from(wv.recHi).reverse(), step, shift, true)
    a = mul(add(lo, hi), 0.5)
  }
  return r.restore(sliceAxis(a, 1, 0, r.n))
}

/**
 * The cosine-sum family of window functions, scipy.signal.windows style: the general cosine and general Hamming forms
 * and the named members (rectangular, Hann, Hamming, Blackman, Blackman-Harris, Nuttall and flat top), which taper a
 * segment to reduce spectral leakage.
 */

import type { Tensor } from '../../tensor/index.ts'
import { build } from './_build.ts'

/** Options shared by the windows. */
export interface WindowOptions {
  /** Symmetric (default true) or periodic. */
  readonly sym?: boolean
}

/**
 * A generalised cosine window (scipy.signal.windows.general_cosine): w(n) = sum_k a_k cos(k x) for x in [-pi, pi].
 *
 * @param m - Length.
 * @param a - Coefficients a_k.
 * @param options - Symmetric or periodic.
 * @returns The window `[m]`.
 * @concept window-functions
 * @example
 * ```ts
 * import { generalCosine } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(generalCosine(3, [0.5, 0.5])) // => [0, 1, 0]
 * ```
 */
export function generalCosine(m: number, a: readonly number[], options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) =>
    Float64Array.from({ length: len }, (_, n) => {
      const x = len === 1 ? -Math.PI : -Math.PI + (2 * Math.PI * n) / (len - 1)
      let s = 0
      for (const [k, c] of a.entries()) s += c * Math.cos(k * x)
      return s
    }),
  )
}

/**
 * A generalised Hamming window (scipy.signal.windows.general_hamming): alpha - (1 - alpha) cos(2 pi n / (M - 1)).
 *
 * @param m - Length.
 * @param alpha - The coefficient (0.5 is Hann, 0.54 Hamming).
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { generalHamming } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(generalHamming(3, 0.5)) // => [0, 1, 0]
 * ```
 */
export function generalHamming(m: number, alpha: number, options: WindowOptions = {}): Tensor {
  return generalCosine(m, [alpha, 1 - alpha], options)
}

/**
 * The rectangular window (scipy.signal.windows.boxcar): all ones.
 *
 * @param m - Length.
 * @param options - Symmetric or periodic (the same here).
 * @returns The window.
 * @example
 * ```ts
 * import { boxcar } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(boxcar(3)) // => [1, 1, 1]
 * ```
 */
export function boxcar(m: number, options: WindowOptions = {}): Tensor {
  return build(m, options.sym ?? true, (len) => new Float64Array(len).fill(1))
}

/**
 * The Hann window (scipy.signal.windows.hann): 0.5 - 0.5 cos(2 pi n / (M - 1)).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @concept window-functions
 * @example
 * ```ts
 * import { hann } from 'omai-engine/signal'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(hann(4, { sym: false })) // => [0, 0.5, 1, 0.5]
 * ```
 */
export function hann(m: number, options: WindowOptions = {}): Tensor {
  return generalHamming(m, 0.5, options)
}

/**
 * The Hamming window (scipy.signal.windows.hamming): 0.54 - 0.46 cos(2 pi n / (M - 1)).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { hamming } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(hamming(3)).map((v) => Math.round(v * 1e12) / 1e12) // => [0.08, 1, 0.08]
 * ```
 */
export function hamming(m: number, options: WindowOptions = {}): Tensor {
  return generalHamming(m, 0.54, options)
}

/**
 * The Blackman window (scipy.signal.windows.blackman): 0.42 - 0.5 cos + 0.08 cos(2x).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { blackman } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(blackman(3)).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 0]
 * ```
 */
export function blackman(m: number, options: WindowOptions = {}): Tensor {
  return generalCosine(m, [0.42, 0.5, 0.08], options)
}

/**
 * The minimum four-term Blackman-Harris window (scipy.signal.windows.blackmanharris).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { blackmanHarris } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(blackmanHarris(3)).map((v) => Math.round(v * 1e8) / 1e8) // => [0.00006, 1, 0.00006]
 * ```
 */
export function blackmanHarris(m: number, options: WindowOptions = {}): Tensor {
  return generalCosine(m, [0.35875, 0.48829, 0.14128, 0.01168], options)
}

/**
 * The minimum four-term Blackman-Harris window of Nuttall (scipy.signal.windows.nuttall).
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { nuttall } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(nuttall(3)).map((v) => Math.round(v * 1e8) / 1e8) // => [0.0003628, 1, 0.0003628]
 * ```
 */
export function nuttall(m: number, options: WindowOptions = {}): Tensor {
  return generalCosine(m, [0.3635819, 0.4891775, 0.1365995, 0.0106411], options)
}

/**
 * The flat top window (scipy.signal.windows.flattop), for amplitude-accurate spectra.
 *
 * @param m - Length.
 * @param options - Symmetric or periodic.
 * @returns The window.
 * @example
 * ```ts
 * import { flattop } from 'omai-engine/signal'
 * import { toFlat } from 'omai-engine/tensor'
 * Math.round((toFlat(flattop(3))[1] ?? 0) * 1e9) / 1e9 // => 1.000000003
 * ```
 */
export function flattop(m: number, options: WindowOptions = {}): Tensor {
  return generalCosine(m, [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368], options)
}

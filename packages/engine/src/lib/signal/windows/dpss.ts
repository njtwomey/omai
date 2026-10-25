/**
 * Discrete prolate spheroidal (Slepian) sequences, the windows of a given length whose energy is most concentrated in a
 * frequency band, as scipy.signal.windows.dpss, computed from a tridiagonal eigenproblem, with their concentration
 * ratios. They are the tapers of the multitaper spectrum.
 */

import { WhiteboxError } from '../../base/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { constant } from '../_shared.ts'
import { dpssTapers } from './_tridiagonal.ts'
import type { WindowOptions } from './cosine.ts'

/** Options for {@link dpss}. */
export interface DpssOptions extends WindowOptions {
  /** Number of tapers (orders 0 .. kmax - 1); omitted for a single window `[M]`. */
  readonly kmax?: number
  /** Normalisation: 2 (unit energy), `'approximate'` or `'subsample'` (unit maximum, corrected for even lengths);
   * default `'approximate'` for one window and 2 for several. */
  readonly norm?: 2 | 'approximate' | 'subsample'
}

/**
 * Discrete prolate spheroidal (Slepian) sequences (scipy.signal.windows.dpss): the sequences of length M most
 * concentrated in the band |f| < NW / M, from the symmetric tridiagonal eigenproblem of Percival and Walden (solved by
 * Sturm bisection and inverse iteration), with scipy's sign conventions.
 *
 * @param m - Length.
 * @param nw - Time-halfbandwidth product (less than M / 2).
 * @param options - Number of tapers, symmetry and normalisation.
 * @returns One window `[M]`, or the tapers `[kmax, M]`.
 * @throws WhiteboxError with code `value` for an invalid NW or taper count.
 * @concept window-functions
 * @example
 * ```ts
 * import { dpss } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const w = toFlat(dpss(8, 2, { kmax: 2 }))
 * Math.round(w.slice(0, 8).reduce((a, v) => a + v * v, 0) * 1e12) / 1e12 // => 1
 * ```
 */
export function dpss(m: number, nw: number, options: DpssOptions = {}): Tensor {
  const single = options.kmax === undefined
  const kmax = options.kmax ?? 1
  const norm = options.norm ?? (single ? 'approximate' : 2)
  const sym = options.sym ?? true
  if (m <= 1) return constant(new Float64Array(single ? m : kmax * m).fill(1), single ? [m] : [kmax, m])
  if (!Number.isInteger(kmax) || kmax < 1 || kmax > m) throw new WhiteboxError('value', 'dpss: kmax must be in 1 .. M')
  if (!(nw > 0) || nw >= m / 2) throw new WhiteboxError('value', 'dpss: NW must be positive and less than M / 2')
  const len = sym ? m : m + 1
  const tapers = dpssTapers(len, nw, kmax)
  if (norm !== 2) {
    for (const w of tapers) {
      let peak = Number.NEGATIVE_INFINITY
      for (const v of w) if (v > peak) peak = v
      for (let i = 0; i < len; i++) w[i] = (w[i] ?? 0) / peak
    }
    if (len % 2 === 0) {
      let correction: number
      if (norm === 'approximate') correction = (len * len) / (len * len + nw)
      else {
        // The real part of 2 sum_k S_k exp(-i pi shift_k) over the positive frequencies of the first taper.
        const w0 = tapers[0] ?? new Float64Array(len)
        let total = 0
        for (let k = 0; k <= len / 2; k++) {
          let re = 0
          let im = 0
          for (let n = 0; n < len; n++) {
            const t = (-2 * Math.PI * k * n) / len
            re += (w0[n] ?? 0) * Math.cos(t)
            im += (w0[n] ?? 0) * Math.sin(t)
          }
          if (k === 0) total += re
          else {
            const shift = -(1 - 1 / len) * k
            const c = Math.cos(-Math.PI * shift)
            const s = Math.sin(-Math.PI * shift)
            total += 2 * (re * c - im * s)
          }
        }
        correction = len / total
      }
      for (const w of tapers) for (let i = 0; i < len; i++) w[i] = (w[i] ?? 0) * correction
    }
  }
  const out = new Float64Array(kmax * m)
  for (const [k, w] of tapers.entries()) out.set(w.subarray(0, m), k * m)
  return constant(out, single ? [m] : [kmax, m])
}

/**
 * The spectral concentration of each DPSS taper (the `ratios` of scipy.signal.windows.dpss with return_ratios): the
 * fraction of its energy inside the band |f| < NW / M, close to 1 for the first 2 NW - 1 tapers.
 *
 * @param m - Length.
 * @param nw - Time-halfbandwidth product.
 * @param kmax - Number of tapers.
 * @returns The ratios `[kmax]`.
 * @example
 * ```ts
 * import { dpssRatios } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * (toFlat(dpssRatios(64, 4, 2))[0] ?? 0) > 0.9999 // => true
 * ```
 */
export function dpssRatios(m: number, nw: number, kmax: number): Tensor {
  const tapers = toFlat(dpss(m, nw, { kmax }))
  const w = nw / m
  const r = Float64Array.from({ length: m }, (_, k) =>
    k === 0 ? 2 * w : (4 * w * Math.sin(2 * Math.PI * w * k)) / (2 * Math.PI * w * k),
  )
  return constant(
    Float64Array.from({ length: kmax }, (_, t) => {
      let total = 0
      for (let lag = 0; lag < m; lag++) {
        let acc = 0
        for (let n = 0; n + lag < m; n++) acc += (tapers[t * m + n] ?? 0) * (tapers[t * m + n + lag] ?? 0)
        total += acc * (r[lag] ?? 0)
      }
      return total
    }),
  )
}

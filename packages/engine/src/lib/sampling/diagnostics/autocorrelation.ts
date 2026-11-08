/**
 * The autocovariance and autocorrelation of a chain of draws at every lag, computed by FFT as ArviZ's `autocov` and
 * `autocorr`: the raw material of effective sample sizes and of autocorrelation plots.
 */

import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { drawsOf, isMatrix, rowAutocovariance } from './_draws.ts'

/**
 * The autocovariance of draws at every lag, gamma_k = (1/n) sum_t (x_t - xbar)(x_{t+k} - xbar) for k = 0..n-1, by FFT
 * with zero padding (ArviZ `autocov`). Rows of a `[chain, draw]` input are treated separately.
 *
 * @param x - Draws `[draw]` or `[chain, draw]`.
 * @returns The autocovariances, the same shape as the input.
 * @throws OmaiError with code `shape` for inputs with more than two axes.
 * @concept autocorrelation
 * @example
 * ```ts
 * import { autocovariance } from 'omai-engine/sampling'
 * import { toArray } from 'omai-engine/tensor'
 * (toArray(autocovariance([1, -1, 1, -1])) as number[]).map((v) => Math.round(v * 1e12) / 1e12) // => [1, -0.75, 0.5, -0.25]
 * ```
 */
export function autocovariance(x: TensorLike): Tensor<'float64'> {
  const d = drawsOf(x, 'autocovariance')
  return fromStorage(rowAutocovariance(d), isMatrix(x) ? [d.chains, d.draws] : [d.draws], 'float64')
}

/**
 * The autocorrelation of draws at every lag, rho_k = gamma_k / gamma_0 (ArviZ `autocorr`), by FFT. Rows of a
 * `[chain, draw]` input are treated separately; a constant row gives NaN.
 *
 * @param x - Draws `[draw]` or `[chain, draw]`.
 * @returns The autocorrelations, the same shape as the input (lag 0 is 1).
 * @throws OmaiError with code `shape` for inputs with more than two axes.
 * @concept autocorrelation
 * @example
 * ```ts
 * import { autocorrelation } from 'omai-engine/sampling'
 * import { toArray } from 'omai-engine/tensor'
 * (toArray(autocorrelation([1, 2, 3, 4])) as number[]).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 0.25, -0.3, -0.45]
 * ```
 */
export function autocorrelation(x: TensorLike): Tensor<'float64'> {
  const d = drawsOf(x, 'autocorrelation')
  const cov = rowAutocovariance(d)
  const out = new Float64Array(cov.length)
  for (let c = 0; c < d.chains; c++) {
    const g0 = cov[c * d.draws] ?? Number.NaN
    for (let k = 0; k < d.draws; k++) out[c * d.draws + k] = (cov[c * d.draws + k] ?? 0) / g0
  }
  return fromStorage(out, isMatrix(x) ? [d.chains, d.draws] : [d.draws], 'float64')
}

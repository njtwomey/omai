/**
 * The theory of ARMA processes x_t = sum phi_j x_(t-j) + e_t + sum theta_j e_(t-j): the moving-average and
 * autoregressive weights, the exact autocovariance, autocorrelation and partial autocorrelation functions, the
 * stationarity and invertibility checks, and the innovations algorithm (Brockwell and Davis) that turns an
 * autocovariance sequence into one-step predictors, one lag per step.
 */

import { OmaiError } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { polyroots } from '../numerics/polynomial/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { arToPacf } from './_params.ts'

/**
 * The weights psi_j of the moving-average representation x_t = sum_(j>=0) psi_j e_(t-j) of a causal ARMA (statsmodels'
 * `arma2ma`): psi_0 = 1 and psi_j = theta_j + sum_i phi_i psi_(j-i).
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param n - How many weights.
 * @returns psi_0..psi_(n-1).
 * @example
 * ```ts
 * import { armaPsiWeights } from 'omai-engine/timeseries'
 * Array.from(armaPsiWeights([0.5], [0.4], 4)) // => [1, 0.9, 0.45, 0.225]
 * ```
 */
export function armaPsiWeights(ar: ArrayLike<number>, ma: ArrayLike<number>, n: number): Float64Array {
  const psi = new Float64Array(n)
  if (n > 0) psi[0] = 1
  for (let j = 1; j < n; j++) {
    let s = j <= ma.length ? (ma[j - 1] ?? 0) : 0
    for (let i = 1; i <= Math.min(ar.length, j); i++) s += (ar[i - 1] ?? 0) * (psi[j - i] ?? 0)
    psi[j] = s
  }
  return psi
}

/**
 * The weights pi_j of the autoregressive representation e_t = x_t - sum_(j>=1) pi_j x_(t-j) of an invertible ARMA
 * (statsmodels' `arma2ar`, with the sign of x_t = sum pi_j x_(t-j) + e_t): the expansion of phi(B) / theta(B).
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param n - How many weights.
 * @returns pi_1..pi_n.
 * @example
 * ```ts
 * import { armaPiWeights } from 'omai-engine/timeseries'
 * // An MA(1) with theta = 0.5 inverts to pi_j = -(-0.5)^j.
 * Array.from(armaPiWeights([], [0.5], 3)) // => [0.5, -0.25, 0.125]
 * ```
 */
export function armaPiWeights(ar: ArrayLike<number>, ma: ArrayLike<number>, n: number): Float64Array {
  // phi(B) / theta(B) = 1 - sum pi_j B^j with theta(B) = 1 + sum theta_j B^j: c_j = phi_j - sum_i theta_i c_(j-i) on the
  // coefficients c of 1 - pi(B) = phi(B) / theta(B).
  const c = new Float64Array(n + 1)
  c[0] = 1
  for (let j = 1; j <= n; j++) {
    let s = j <= ar.length ? -(ar[j - 1] ?? 0) : 0
    for (let i = 1; i <= Math.min(ma.length, j); i++) s -= (ma[i - 1] ?? 0) * (c[j - i] ?? 0)
    c[j] = s
  }
  return Float64Array.from({ length: n }, (_, j) => -(c[j + 1] ?? 0))
}

/**
 * Whether the autoregressive polynomial 1 - phi_1 z - ... - phi_p z^p has every root outside the unit circle.
 *
 * @param ar - The autoregressive coefficients.
 * @returns True for a causal (stationary) process.
 * @example
 * ```ts
 * import { isStationary } from 'omai-engine/timeseries'
 * [isStationary([0.5, 0.3]), isStationary([1.1]), isStationary([])] // => [true, false, true]
 * ```
 */
export function isStationary(ar: ArrayLike<number>): boolean {
  return rootsOutside([1, ...Array.from(ar, (v) => -v)])
}

/**
 * Whether the moving-average polynomial 1 + theta_1 z + ... + theta_q z^q has every root outside the unit circle.
 *
 * @param ma - The moving-average coefficients.
 * @returns True for an invertible process.
 * @example
 * ```ts
 * import { isInvertible } from 'omai-engine/timeseries'
 * [isInvertible([0.5]), isInvertible([2])] // => [true, false]
 * ```
 */
export function isInvertible(ma: ArrayLike<number>): boolean {
  return rootsOutside([1, ...Array.from(ma)])
}

function rootsOutside(poly: number[]): boolean {
  // A vanishing top coefficient puts a root beyond any radius of interest (and ill-conditions the companion matrix).
  while (poly.length > 1 && Math.abs(poly[poly.length - 1] ?? 0) < 1e-12) poly.pop()
  const roots = polyroots(poly)
  const im = toFlat(roots.im)
  return toFlat(roots.re).every((x, i) => Math.hypot(x, im[i] ?? 0) > 1 + 1e-12)
}

/**
 * The exact autocovariances gamma(0..nlags) of a causal ARMA (statsmodels' `arma_acovf`): the first p + 1 from the
 * linear system gamma(k) - sum phi_j gamma(k - j) = sigma2 sum_(j>=k) theta_j psi_(j-k), the rest by recursion.
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param sigma2 - The innovation variance (default 1).
 * @param nlags - The largest lag.
 * @returns gamma(0), ..., gamma(nlags).
 * @throws OmaiError with code `value` when the process is not stationary.
 * @example
 * ```ts
 * import { armaAutocovariance } from 'omai-engine/timeseries'
 * Array.from(armaAutocovariance([0.5], [], 1, 3)) // => [1.3333333333333333, 0.6666666666666666, 0.3333333333333333, 0.16666666666666666]
 * ```
 */
export function armaAutocovariance(ar: ArrayLike<number>, ma: ArrayLike<number>, sigma2 = 1, nlags = 10): Float64Array {
  if (!isStationary(ar)) throw new OmaiError('value', 'armaAutocovariance: the process is not stationary')
  const p = ar.length
  const q = ma.length
  const theta = [1, ...Array.from(ma)]
  const psi = armaPsiWeights(ar, ma, q + 1)
  const rhs = (k: number): number => {
    let s = 0
    for (let j = k; j <= q; j++) s += (theta[j] ?? 0) * (psi[j - k] ?? 0)
    return sigma2 * s
  }
  const gamma = new Float64Array(Math.max(nlags, p) + 1)
  if (p === 0) {
    for (let k = 0; k <= Math.min(q, gamma.length - 1); k++) gamma[k] = rhs(k)
  } else {
    const a = new Float64Array((p + 1) * (p + 1))
    const b = new Float64Array(p + 1)
    for (let k = 0; k <= p; k++) {
      a[k * (p + 1) + k] = (a[k * (p + 1) + k] ?? 0) + 1
      for (let j = 1; j <= p; j++) {
        const at = k * (p + 1) + Math.abs(k - j)
        a[at] = (a[at] ?? 0) - (ar[j - 1] ?? 0)
      }
      b[k] = rhs(k)
    }
    const g = toFlat(solve(reshapeSquare(a, p + 1), Array.from(b)))
    for (let k = 0; k <= p; k++) gamma[k] = g[k] ?? 0
    for (let k = p + 1; k < gamma.length; k++) {
      let s = rhs(k)
      for (let j = 1; j <= p; j++) s += (ar[j - 1] ?? 0) * (gamma[k - j] ?? 0)
      gamma[k] = s
    }
    return gamma.slice(0, nlags + 1)
  }
  for (let k = p + 1; k < gamma.length; k++) {
    let s = rhs(k)
    for (let j = 1; j <= p; j++) s += (ar[j - 1] ?? 0) * (gamma[k - j] ?? 0)
    gamma[k] = s
  }
  return gamma.slice(0, nlags + 1)
}

function reshapeSquare(a: Float64Array, n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from(a.subarray(i * n, (i + 1) * n)))
}

/**
 * The theoretical autocorrelation function rho(0..nlags) of a causal ARMA.
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param nlags - The largest lag.
 * @returns rho(0) = 1, ..., rho(nlags).
 * @example
 * ```ts
 * import { armaAcf } from 'omai-engine/timeseries'
 * Array.from(armaAcf([0.5], [], 3)) // => [1, 0.5, 0.25, 0.125]
 * ```
 */
export function armaAcf(ar: ArrayLike<number>, ma: ArrayLike<number>, nlags = 10): Float64Array {
  const g = armaAutocovariance(ar, ma, 1, nlags)
  const g0 = g[0] ?? 1
  return Float64Array.from(g, (v) => v / g0)
}

/**
 * The theoretical partial autocorrelation function phi_kk (k = 0..nlags, with 1 at lag 0) of a causal ARMA, by the
 * Durbin-Levinson recursion on its autocorrelations.
 *
 * @param ar - The autoregressive coefficients.
 * @param ma - The moving-average coefficients.
 * @param nlags - The largest lag.
 * @returns The partial autocorrelations at lags 0..nlags.
 * @example
 * ```ts
 * import { armaPacf } from 'omai-engine/timeseries'
 * // An AR(1) has a single non-zero partial autocorrelation.
 * Array.from(armaPacf([0.5], [], 3)).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 0.5, 0, 0]
 * ```
 */
export function armaPacf(ar: ArrayLike<number>, ma: ArrayLike<number>, nlags = 10): Float64Array {
  const rho = armaAcf(ar, ma, nlags)
  const out = new Float64Array(nlags + 1)
  out[0] = 1
  let phi: Float64Array = new Float64Array(0)
  for (let k = 1; k <= nlags; k++) {
    let num = rho[k] ?? 0
    let den = 1
    for (let j = 1; j < k; j++) {
      num -= (phi[j - 1] ?? 0) * (rho[k - j] ?? 0)
      den -= (phi[j - 1] ?? 0) * (rho[j] ?? 0)
    }
    const kk = num / den
    const next = new Float64Array(k)
    for (let j = 1; j < k; j++) next[j - 1] = (phi[j - 1] ?? 0) - kk * (phi[k - j - 1] ?? 0)
    next[k - 1] = kk
    phi = next
    out[k] = kk
  }
  return out
}

/**
 * The partial autocorrelations of the AR(p) with the given coefficients, p of them (the reflection coefficients of the
 * step-down recursion; NaN from the first non-stationary one).
 *
 * @param ar - The autoregressive coefficients.
 * @returns phi_11, ..., phi_pp.
 * @example
 * ```ts
 * import { arToPartialAutocorrelations } from 'omai-engine/timeseries'
 * Array.from(arToPartialAutocorrelations([0.5, 0.2])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.625, 0.2]
 * ```
 */
export function arToPartialAutocorrelations(ar: ArrayLike<number>): Float64Array {
  return arToPacf(ar)
}

/** The state of {@link innovationsAlgorithm}: the predictor coefficients row by row and the prediction variances. */
export interface InnovationsState {
  /** Rows computed so far. */
  readonly n: number
  /** `theta[m][j - 1]` is theta_(m,j), the weight of the innovation j steps back in the m-step-ahead predictor. */
  readonly theta: number[][]
  /** The mean squared errors v_0, v_1, ... of the one-step predictors. */
  readonly v: number[]
}

/**
 * The innovations algorithm (Brockwell and Davis, Prop. 5.2.2) for a stationary series with autocovariances `gamma`:
 * the one-step predictor of X_(m+1) is sum_j theta_(m,j) (X_(m+1-j) - its predictor), with mean squared error v_m.
 * Each step computes one more row (one more lag). For an invertible MA(q), theta_(m,j) tends to theta_j and v_m to the
 * innovation variance.
 *
 * @param gamma - Autocovariances gamma(0..N); the algorithm runs up to row N.
 * @returns The algorithm; its state has the rows computed so far.
 * @example
 * ```ts
 * import { innovationsAlgorithm } from 'omai-engine/timeseries'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // MA(1), theta = 0.5, sigma2 = 1: gamma = (1.25, 0.5, 0, 0).
 * const s = runAlgorithm(innovationsAlgorithm([1.25, 0.5, 0, 0]), { maxSteps: 3 }).state
 * s.v.map((v) => Math.round(v * 1e6) / 1e6) // => [1.25, 1.05, 1.011905, 1.002941]
 * ```
 */
export function innovationsAlgorithm(gamma: ArrayLike<number>): Algorithm<InnovationsState> {
  const N = gamma.length - 1
  const g = (h: number): number => gamma[Math.abs(h)] ?? 0
  return {
    name: 'innovations',
    config: { rows: N },
    schema: channels({
      metrics: { v: { description: 'mean squared error of the one-step predictor of this row', direction: 'lower' } },
      tensors: { theta: { role: 'state', description: 'the predictor coefficients theta_(m,1..m) of the newest row' } },
    }),
    init: () => ({ n: 1, theta: [[]], v: [g(0)] }),
    step: (s) => {
      const m = s.n
      const row = new Array<number>(m).fill(0)
      // theta_(m, m-k) for k = 0..m-1 (BD 5.2.4), with theta[k][j - 1] = theta_(k, j).
      for (let k = 0; k < m; k++) {
        let acc = g(m - k)
        for (let j = 0; j < k; j++) acc -= (s.theta[k]?.[k - j - 1] ?? 0) * (row[m - j - 1] ?? 0) * (s.v[j] ?? 0)
        row[m - k - 1] = acc / (s.v[k] ?? 1)
      }
      let vm = g(0)
      for (let j = 0; j < m; j++) vm -= (row[m - j - 1] ?? 0) ** 2 * (s.v[j] ?? 0)
      return { n: m + 1, theta: [...s.theta, row], v: [...s.v, vm] }
    },
    done: (s) => (s.n > N ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { v: s.v[s.v.length - 1] ?? 0 },
      tensors: { theta: { role: 'state', value: s.theta[s.theta.length - 1] ?? [], shape: [s.theta.length - 1] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as InnovationsState,
  }
}

/**
 * Run the innovations algorithm to the end: the predictor coefficients and prediction variances up to row N.
 *
 * @param gamma - Autocovariances gamma(0..N).
 * @returns `theta` (row m has m entries) and `v` (N + 1 entries).
 * @example
 * ```ts
 * import { innovations } from 'omai-engine/timeseries'
 * innovations([2, 1, 0]).v // => [2, 1.5, 1.3333333333333333]
 * ```
 */
export function innovations(gamma: ArrayLike<number>): { readonly theta: number[][]; readonly v: number[] } {
  const { state } = runAlgorithm(innovationsAlgorithm(gamma), { maxSteps: gamma.length - 1 })
  return { theta: state.theta, v: state.v }
}

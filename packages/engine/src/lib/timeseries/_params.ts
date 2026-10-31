/**
 * Parameter transforms of the time-series fits: partial autocorrelations to and from autoregressive coefficients (the
 * Durbin-Levinson recursion), and the map from unconstrained reals to a stationary polynomial that statsmodels uses
 * to keep the optimiser inside the stationary (or invertible) region.
 */

/** The AR coefficients (x_t = sum phi_j x_(t-j) + e_t) of a stationary process with the given partial autocorrelations. */
export function pacfToAr(pacf: ArrayLike<number>): Float64Array {
  let phi = new Float64Array(0)
  for (let n = 0; n < pacf.length; n++) {
    const r = pacf[n] ?? 0
    const next = new Float64Array(n + 1)
    for (let k = 0; k < n; k++) next[k] = (phi[k] ?? 0) - r * (phi[n - 1 - k] ?? 0)
    next[n] = r
    phi = next
  }
  return phi
}

/** The partial autocorrelations of the AR coefficients (the step-down recursion); NaN from the first |r| >= 1. */
export function arToPacf(ar: ArrayLike<number>): Float64Array {
  const out = new Float64Array(ar.length)
  let phi = Float64Array.from(ar)
  for (let n = ar.length - 1; n >= 0; n--) {
    const r = phi[n] ?? 0
    out[n] = r
    if (!(Math.abs(r) < 1)) {
      for (let k = 0; k < n; k++) out[k] = Number.NaN
      break
    }
    const prev = new Float64Array(n)
    for (let k = 0; k < n; k++) prev[k] = ((phi[k] ?? 0) + r * (phi[n - 1 - k] ?? 0)) / (1 - r * r)
    phi = prev
  }
  return out
}

/** Unconstrained reals to the coefficients of a stationary AR polynomial: u -> u / sqrt(1 + u^2) -> Durbin-Levinson. */
export function constrainStationary(u: ArrayLike<number>): Float64Array {
  return pacfToAr(Float64Array.from(u, (v) => v / Math.sqrt(1 + v * v)))
}

/** The inverse of {@link constrainStationary}; undefined when the coefficients are not stationary. */
export function unconstrainStationary(ar: ArrayLike<number>): Float64Array | undefined {
  const pacf = arToPacf(ar)
  if (pacf.some((r) => !Number.isFinite(r) || Math.abs(r) >= 1)) return undefined
  return Float64Array.from(pacf, (r) => r / Math.sqrt(1 - r * r))
}

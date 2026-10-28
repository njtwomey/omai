// The internals of the ARIMA family: the model specification, the regression columns (trend and exogenous), the layout
// of the parameter vector, the expansion of seasonal polynomials, and the two objectives (conditional sum of squares
// and the exact concentrated likelihood) on the differenced series.
import { MdmlError } from '../base/index.ts'
import { polymul } from '../numerics/polynomial/index.ts'
import { constrainStationary } from './_params.ts'
import { difference, differencePolynomial } from './differencing.ts'
import { armaForm, filterForm } from './state-space.ts'

/** The deterministic part of the mean: none, a constant, a linear trend, or both (statsmodels' `trend`). */
export type Trend = 'n' | 'c' | 't' | 'ct'

/** An ARIMA(p, d, q)(P, D, Q)_s specification with optional trend and exogenous regressors. */
export interface ArimaSpec {
  /** (p, d, q). */
  readonly order: readonly [number, number, number]
  /** The seasonal part (P, D, Q) and its period s. */
  readonly seasonal?: { readonly order: readonly [number, number, number]; readonly period: number }
  /** The trend (default `'c'` when nothing is differenced and `'n'` otherwise). */
  readonly trend?: Trend
  /** Exogenous regressors, one row per observation. */
  readonly exog?: readonly (readonly number[])[]
}

/** A specification with its defaults filled in. */
export interface ResolvedSpec {
  readonly p: number
  readonly d: number
  readonly q: number
  readonly P: number
  readonly D: number
  readonly Q: number
  readonly s: number
  readonly trend: Trend
  readonly kExog: number
}

/** Fill in and validate a specification. */
export function resolveSpec(spec: ArimaSpec): ResolvedSpec {
  const [p, d, q] = spec.order
  const [P, D, Q] = spec.seasonal?.order ?? [0, 0, 0]
  const s = spec.seasonal?.period ?? 1
  for (const v of [p, d, q, P, D, Q])
    if (!(Number.isInteger(v) && v >= 0)) throw new MdmlError('value', 'ARIMA orders must be non-negative integers')
  if ((P > 0 || D > 0 || Q > 0) && !(Number.isInteger(s) && s >= 2))
    throw new MdmlError('value', 'a seasonal ARIMA needs a period of at least 2')
  const trend = spec.trend ?? (d + D === 0 ? 'c' : 'n')
  return { p, d, q, P, D, Q, s, trend, kExog: spec.exog?.[0]?.length ?? 0 }
}

/** The names of the regression coefficients, trend first. */
export function regressionNames(rs: ResolvedSpec): string[] {
  const names: string[] = []
  if (rs.trend === 'c' || rs.trend === 'ct') names.push('const')
  if (rs.trend === 't' || rs.trend === 'ct') names.push('drift')
  for (let i = 0; i < rs.kExog; i++) names.push(`x${i + 1}`)
  return names
}

/** The regression columns (trend then exogenous) over `n` observations, the first being observation number `start`. */
export function regressionColumns(n: number, rs: ResolvedSpec, exog: ArimaSpec['exog'], start = 0): number[][] {
  const cols: number[][] = []
  if (rs.trend === 'c' || rs.trend === 'ct') cols.push(new Array<number>(n).fill(1))
  if (rs.trend === 't' || rs.trend === 'ct') cols.push(Array.from({ length: n }, (_, i) => start + i + 1))
  for (let k = 0; k < rs.kExog; k++) {
    if (exog === undefined || exog.length !== n) throw new MdmlError('shape', `the exogenous regressors need ${n} rows`)
    cols.push(exog.map((row) => row[k] ?? Number.NaN))
  }
  return cols
}

/** Apply the regular and seasonal differences of a specification to a series. */
export function differenceSeries(x: ArrayLike<number>, rs: ResolvedSpec): number[] {
  return difference(difference(Array.from(x), rs.d), rs.D, rs.s)
}

/** The full differencing polynomial (1 - B)^d (1 - B^s)^D. */
export function differencingOf(rs: ResolvedSpec): number[] {
  return differencePolynomial(rs.d, rs.D, rs.s)
}

/** The parameters of an ARIMA in their natural (constrained) form. */
export interface NaturalParameters {
  readonly beta: number[]
  readonly ar: number[]
  readonly ma: number[]
  readonly sar: number[]
  readonly sma: number[]
}

/** Split a flat vector (regression, ar, ma, sar, sma) into its parts. */
export function splitVector(x: ArrayLike<number>, rs: ResolvedSpec, kReg: number): NaturalParameters {
  let at = 0
  const take = (k: number): number[] => {
    const part = Array.from({ length: k }, (_, i) => x[at + i] ?? 0)
    at += k
    return part
  }
  return { beta: take(kReg), ar: take(rs.p), ma: take(rs.q), sar: take(rs.P), sma: take(rs.Q) }
}

/** The unconstrained reals to the natural parameters: stationary AR and seasonal AR, invertible MA and seasonal MA. */
export function constrain(u: ArrayLike<number>, rs: ResolvedSpec, kReg: number): NaturalParameters {
  const parts = splitVector(u, rs, kReg)
  return {
    beta: parts.beta,
    ar: Array.from(constrainStationary(parts.ar)),
    ma: Array.from(constrainStationary(parts.ma), (v) => -v),
    sar: Array.from(constrainStationary(parts.sar)),
    sma: Array.from(constrainStationary(parts.sma), (v) => -v),
  }
}

/** The AR and MA polynomials with the seasonal factors multiplied in: phi*(B) = phi(B) Phi(B^s), likewise theta*. */
export function expandPolynomials(
  ar: readonly number[],
  ma: readonly number[],
  sar: readonly number[],
  sma: readonly number[],
  s: number,
): { readonly ar: number[]; readonly ma: number[] } {
  const seasonalPoly = (coeffs: readonly number[], sign: number): number[] => {
    const poly = new Array<number>(coeffs.length * s + 1).fill(0)
    poly[0] = 1
    coeffs.forEach((v, k) => {
      poly[(k + 1) * s] = sign * v
    })
    return poly
  }
  let arPoly = [1, ...ar.map((v) => -v)]
  if (sar.length > 0) arPoly = polymul(arPoly, seasonalPoly(sar, -1))
  let maPoly = [1, ...ma]
  if (sma.length > 0) maPoly = polymul(maPoly, seasonalPoly(sma, 1))
  return { ar: arPoly.slice(1).map((v) => -v), ma: maPoly.slice(1) }
}

/** The differenced data a fit works on: the series and regression columns after differencing. */
export interface Prepared {
  readonly rs: ResolvedSpec
  /** The differenced series. */
  readonly z: Float64Array
  /** The differenced regression columns. */
  readonly columns: Float64Array[]
}

/** Difference the series and its regression columns once. */
export function prepare(y: ArrayLike<number>, spec: ArimaSpec): Prepared {
  const rs = resolveSpec(spec)
  if ((rs.trend === 'c' || rs.trend === 'ct') && rs.d + rs.D > 0)
    throw new MdmlError('value', "ARIMA: a differenced series has no constant; use trend 't' for a drift")
  const columns = regressionColumns(y.length, rs, spec.exog).map((c) => Float64Array.from(differenceSeries(c, rs)))
  for (const c of columns)
    if (c.length > 0 && c.every((v) => v === 0))
      throw new MdmlError('value', 'ARIMA: a regressor is zero after differencing')
  const z = Float64Array.from(differenceSeries(y, rs))
  const parameters = columns.length + rs.p + rs.q + rs.P + rs.Q + 1
  if (z.length < parameters + 1) throw new MdmlError('value', 'ARIMA: the series is too short for the model')
  return { rs, z, columns }
}

/** z - sum beta_k column_k: the differenced series with the regression removed. */
export function regressionResidual(prep: Prepared, beta: readonly number[]): Float64Array {
  const out = Float64Array.from(prep.z)
  prep.columns.forEach((c, k) => {
    const b = beta[k] ?? 0
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) - b * (c[i] ?? 0)
  })
  return out
}

/** The concentrated exact Gaussian log-likelihood on the differenced scale (NaN when the filter breaks down). */
export function exactLogLikelihood(prep: Prepared, nat: NaturalParameters): number {
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, prep.rs.s)
  const f = filterForm(regressionResidual(prep, nat.beta), armaForm(ar, ma))
  const n = f.nobs
  return -0.5 * (n * (Math.log(2 * Math.PI) + 1 + Math.log(f.sumSquares / n)) + f.sumLogF)
}

/** The conditional residuals: the ARMA recursion started at zero after the first p* observations. */
export function conditionalResiduals(
  z: ArrayLike<number>,
  ar: readonly number[],
  ma: readonly number[],
): { readonly residuals: Float64Array; readonly start: number } {
  const n = z.length
  const start = ar.length
  const e = new Float64Array(n)
  for (let t = start; t < n; t++) {
    let v = z[t] ?? 0
    for (let j = 1; j <= ar.length; j++) v -= (ar[j - 1] ?? 0) * (z[t - j] ?? 0)
    for (let j = 1; j <= ma.length; j++) if (t - j >= start) v -= (ma[j - 1] ?? 0) * (e[t - j] ?? 0)
    e[t] = v
  }
  return { residuals: e, start }
}

/** The CSS log-likelihood with the variance concentrated out: -m/2 (log 2 pi + 1 + log(css / m)). */
export function cssLogLikelihood(prep: Prepared, nat: NaturalParameters): number {
  const { ar, ma } = expandPolynomials(nat.ar, nat.ma, nat.sar, nat.sma, prep.rs.s)
  const { residuals, start } = conditionalResiduals(regressionResidual(prep, nat.beta), ar, ma)
  let css = 0
  for (let t = start; t < residuals.length; t++) css += (residuals[t] ?? 0) ** 2
  const m = residuals.length - start
  return -0.5 * m * (Math.log(2 * Math.PI) + 1 + Math.log(css / m))
}

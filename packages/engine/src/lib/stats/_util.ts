// Private helpers of the stats domain: reading samples and tables once, building tensors, and scalar views of the
// special-function CDFs that p-values and intervals need.
import { MdmlError } from '../base/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { betainc, chdtr, chdtrc, fdtr, fdtrc, lgamma, ndtr, ndtri, stdtr, stdtrit } from '../numerics/special/index.ts'
import { asTensor, fromStorage, item, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/**
 * Read a sample as a flat array, requiring at least `min` values. With `rejectNaN` a NaN is an error: routines that
 * sort, rank or compare the data would otherwise treat it as a tie and answer with a finite statistic.
 */
export function sample(name: string, x: TensorLike, min = 1, rejectNaN = false): Float64Array {
  const data = Float64Array.from(toFlat(asTensor(x)))
  if (data.length < min) throw new MdmlError('value', `${name}: needs at least ${min} observations`)
  if (rejectNaN)
    for (const v of data) if (Number.isNaN(v)) throw new MdmlError('value', `${name}: NaN values are not allowed`)
  return data
}

/** Read several samples (groups). */
export function groupsOf(name: string, groups: readonly TensorLike[], min = 1, minGroups = 2): Float64Array[] {
  if (groups.length < minGroups) throw new MdmlError('value', `${name}: needs at least ${minGroups} groups`)
  return groups.map((g) => sample(name, g, min))
}

/** Read two samples of equal length. */
export function paired(
  name: string,
  x: TensorLike,
  y: TensorLike,
  min = 1,
  rejectNaN = false,
): [Float64Array, Float64Array] {
  const a = sample(name, x, min, rejectNaN)
  const b = sample(name, y, min, rejectNaN)
  if (a.length !== b.length) throw new MdmlError('shape', `${name}: samples differ in length`)
  return [a, b]
}

/** A dense row-major matrix read from an input. */
export interface Rows {
  readonly rows: number
  readonly cols: number
  readonly data: Float64Array
}

/** Read a 2-D input (a 1-D input is one column). */
export function rowsOf(name: string, x: TensorLike): Rows {
  const t = asTensor(x)
  const data = Float64Array.from(toFlat(t))
  if (t.shape.length === 1) return { rows: t.shape[0] ?? 0, cols: 1, data }
  if (t.shape.length !== 2) throw new MdmlError('shape', `${name}: expected a 1-D or 2-D array`)
  return { rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0, data }
}

/** Column `j` of a row-major matrix. */
export function column(m: Rows, j: number): Float64Array {
  const out = new Float64Array(m.rows)
  for (let i = 0; i < m.rows; i++) out[i] = m.data[i * m.cols + j] ?? 0
  return out
}

/** A 1-D float64 tensor. */
export function vec(xs: ArrayLike<number>): Tensor {
  return fromStorage(Float64Array.from(xs), [xs.length], 'float64')
}

/** A 2-D float64 tensor from row-major data. */
export function mat(data: ArrayLike<number>, rows: number, cols: number): Tensor {
  return fromStorage(Float64Array.from(data), [rows, cols], 'float64')
}

/** A polynomial with coefficients in ascending order at `x`, by Horner's rule. */
export function poly(c: readonly number[], x: number): number {
  let r = 0
  for (let i = c.length - 1; i >= 0; i--) r = r * x + (c[i] ?? 0)
  return r
}

/** An ascending copy. */
export function sortedCopy(xs: ArrayLike<number>): Float64Array {
  return Float64Array.from(xs).sort()
}

export const normCdf = (x: number): number => item(ndtr(x))
export const normSf = (x: number): number => item(ndtr(-x))
export const normPpf = (p: number): number => item(ndtri(p))
export const tCdf = (df: number, t: number): number => item(stdtr(df, t))
export const tSf = (df: number, t: number): number => item(stdtr(df, -t))
export const tPpf = (df: number, p: number): number => item(stdtrit(df, p))
export const chi2Cdf = (df: number, x: number): number => item(chdtr(df, x))
export const chi2Sf = (df: number, x: number): number => item(chdtrc(df, x))
export const fCdf = (d1: number, d2: number, x: number): number => item(fdtr(d1, d2, x))
export const fSf = (d1: number, d2: number, x: number): number => item(fdtrc(d1, d2, x))
export const betaInc = (a: number, b: number, x: number): number => item(betainc(a, b, x))
export const logGamma = (x: number): number => item(lgamma(x))

/**
 * A root of f on [lo, ...): expands the upper end geometrically until the sign changes, then runs Brent's method.
 * Returns `lo` when f(lo) already has the sign of the far end, and `Infinity` when no sign change is found.
 */
export function rootAbove(f: (x: number) => number, lo: number, hi: number, maxExpansions = 60): number {
  const flo = f(lo)
  if (flo === 0) return lo
  let b = hi
  let fb = f(b)
  for (let i = 0; i < maxExpansions && Math.sign(fb) === Math.sign(flo); i++) {
    b = lo + (b - lo) * 2
    fb = f(b)
  }
  if (Math.sign(fb) === Math.sign(flo)) return Number.POSITIVE_INFINITY
  return brentq(f, lo, b, { xtol: 1e-13, rtol: 1e-13, maxIterations: 200 }).root
}

/** The integral of a scalar function by adaptive Gauss-Kronrod quadrature (infinite limits allowed). */
export function integrate(f: (x: number) => number, a: number, b: number, epsrel = 1e-10, limit = 200): number {
  return item(quad((x) => vec(toFlat(x).map(f)), a, b, { epsabs: 0, epsrel, limit }).value)
}

/** The two-sided normal critical value of a confidence level. */
export const zCritical = (level: number): number => normPpf(0.5 + level / 2)

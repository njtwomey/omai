// What the Gaussian-vector sites read their arguments through: a Gaussian side (a point, or a cavity that is shown very wide when it says nothing),
// the rows a plated factor stands for, and the evaluator of the closed-form integral over the unknown vectors at one precision matrix. Shared by the
// Wishart site (`covariance.ts`) and the standard-deviation-and-correlation site (`correlation.ts`).
import { OmaiError } from '../../base/index.ts'
import { choleskyInPlace, choleskyInverse, choleskyLogDet, choleskySolveInPlace } from '../_bartlett.ts'
import type { Arg, Eta, OperatorContext } from '../types.ts'
import { FLAT_CAVITY_PRECISION, ImproperMessageError, numberParam } from './args.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The rows a factor stands for: `count` i.i.d. vectors whose sample mean is the observed `x` and whose scatter about that mean is `scatter`
 * (`sum (y_i - ybar)(y_i - ybar)'`, row-major). A factor of one row has the defaults, `count = 1` and no scatter.
 */
export interface Rows {
  readonly count: number
  readonly scatter: Float64Array
}

/** The rows of a factor from its parameters. */
export function rowsOf(params: OperatorContext['params'], d: number): Rows {
  const count = numberParam(params, 'count', 1)
  if (!(count >= 1)) throw new OmaiError('value', 'mvn-wishart: count is the number of rows, at least 1')
  const given = params['scatter']
  const scatter = given === undefined ? new Float64Array(d * d) : Float64Array.from(given as readonly number[])
  if (scatter.length !== d * d)
    throw new OmaiError('shape', `mvn-wishart: scatter has ${scatter.length} entries, expected ${d * d}`)
  return { count, scatter }
}

/** One of the two Gaussian arguments, as the integral reads it: a point, or a cavity (a very wide one when it carries no information). */
export interface GaussianSide {
  readonly point: boolean
  /** True when the cavity says nothing yet: it is read as a wide one and the message to it is the projection against the flat cavity. */
  readonly flat: boolean
  /** The value of a point, the mean of a cavity. */
  readonly mean: Float64Array
  /** The natural parameters of the cavity: `h` and the precision (row-major); zero for a point. */
  readonly h: Float64Array
  readonly precision: Float64Array
  /** The covariance of the cavity; zero for a point. */
  readonly cov: Float64Array
}

/** Read a Gaussian vector argument. */
export function gaussianSide(a: Arg, d: number): GaussianSide {
  if (a.value !== undefined) {
    const v = typeof a.value === 'number' ? [a.value] : a.value
    if (v.length !== d)
      throw new OmaiError('shape', `the observed vector has ${v.length} entries, the matrix is ${d} by ${d}`)
    return {
      point: true,
      flat: false,
      mean: Float64Array.from(v),
      h: new Float64Array(d),
      precision: new Float64Array(d * d),
      cov: new Float64Array(d * d),
    }
  }
  const eta = a.eta as Eta
  if (eta.every((v) => Math.abs(v) < 1e-12)) {
    const precision = new Float64Array(d * d)
    const cov = new Float64Array(d * d)
    for (let i = 0; i < d; i++) {
      precision[i * d + i] = FLAT_CAVITY_PRECISION
      cov[i * d + i] = 1 / FLAT_CAVITY_PRECISION
    }
    return { point: false, flat: true, mean: new Float64Array(d), h: new Float64Array(d), precision, cov }
  }
  const precision = Float64Array.from(eta.slice(d))
  const l = Float64Array.from(precision)
  if (!choleskyInPlace(l, d)) throw new ImproperMessageError('a Gaussian vector message has no density')
  const cov = choleskyInverse(l, d)
  const h = Float64Array.from(eta.slice(0, d))
  const mean = Float64Array.from(h)
  choleskySolveInPlace(l, d, mean)
  return { point: false, flat: false, mean, h, precision, cov }
}

export const NO_ROWS: Rows = { count: 1, scatter: new Float64Array(0) }

/** What the integral over the unknown Gaussians gives at one precision matrix, and (when asked) the conditional of one of them. */
export type Evaluator = (
  lambda: Float64Array,
  offset: number,
  logDet: number,
  target: -1 | 0 | 1,
  outMean?: Float64Array,
  outCov?: Float64Array,
) => number

/**
 * The evaluator of the Gaussian part of the factor `N(x; mean, L^-1)` against the cavities of `x` and `mean`: for a given
 * precision matrix the log of the integral over the vectors that are not observed, and the mean and covariance of one of them
 * under the tilted conditional.
 *
 * @param x - The vector's side.
 * @param mean - The mean's side.
 * @param rows - The rows the vector stands for (its sample mean, their count and scatter); default one row.
 * @param d - The dimension.
 * @returns The evaluator.
 */
export function evaluatorOf(x: GaussianSide, mean: GaussianSide, d: number, rows: Rows = NO_ROWS): Evaluator {
  const k = rows.count
  const unknown = [x, mean].map((s, i) => ({ s, sign: i === 0 ? 1 : -1, i })).filter((u) => !u.s.point)
  const q = unknown.length * d
  const c = new Float64Array(d)
  for (let i = 0; i < d; i++) c[i] = (x.point ? (x.mean[i] as number) : 0) - (mean.point ? (mean.mean[i] as number) : 0)
  const p = new Float64Array(q * q)
  const h = new Float64Array(q)
  const w = new Float64Array(q)
  const lc = new Float64Array(d)
  const col = new Float64Array(q)
  return (lambda, offset, logDet, target, outMean, outCov) => {
    // L c and c' L c
    let quad = 0
    for (let i = 0; i < d; i++) {
      let s = 0
      for (let j = 0; j < d; j++) s += (lambda[offset + i * d + j] as number) * (c[j] as number)
      lc[i] = s
      quad += s * (c[i] as number)
    }
    // k rows around the vector: (k / 2) log |L| - (k d / 2) log 2 pi - (k / 2) c' L c - tr(L scatter) / 2
    let tr = 0
    if (k > 1)
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) tr += (lambda[offset + i * d + j] as number) * (rows.scatter[j * d + i] as number)
    const base = 0.5 * k * logDet - 0.5 * k * d * LOG_2PI - 0.5 * k * quad - 0.5 * tr
    if (q === 0) return base
    for (let a = 0; a < unknown.length; a++) {
      const ua = unknown[a] as (typeof unknown)[number]
      for (let b = 0; b < unknown.length; b++) {
        const ub = unknown[b] as (typeof unknown)[number]
        for (let i = 0; i < d; i++)
          for (let j = 0; j < d; j++)
            p[(a * d + i) * q + b * d + j] =
              k * ua.sign * ub.sign * (lambda[offset + i * d + j] as number) +
              (a === b ? (ua.s.precision[i * d + j] as number) : 0)
      }
      for (let i = 0; i < d; i++) h[a * d + i] = (ua.s.h[i] as number) - k * ua.sign * (lc[i] as number)
    }
    if (!choleskyInPlace(p, q)) return Number.NEGATIVE_INFINITY
    w.set(h)
    choleskySolveInPlace(p, q, w)
    let hw = 0
    for (let i = 0; i < q; i++) hw += (h[i] as number) * (w[i] as number)
    const logI = base + 0.5 * hw - 0.5 * choleskyLogDet(p, q) + 0.5 * q * LOG_2PI
    if (target >= 0 && outMean !== undefined && outCov !== undefined) {
      const slot = unknown.findIndex((u) => u.i === target)
      for (let i = 0; i < d; i++) outMean[i] = w[slot * d + i] as number
      for (let j = 0; j < d; j++) {
        col.fill(0)
        col[slot * d + j] = 1
        choleskySolveInPlace(p, q, col)
        for (let i = 0; i < d; i++) outCov[i * d + j] = col[slot * d + i] as number
      }
    }
    return logI
  }
}

// What the multiclass sites share: the parameters that name the class and the log probabilities of the softmax and of the multinomial
// probit. `multiclass.ts` (one Gaussian message per class score) and `multiclass-row.ts` (one joint message on the vector the scores are
// made of) both read them, and integrate with the adaptive Gauss-Hermite rule of `numerics/_product.ts`.

import { logNdtr, ndtr } from '../../../numerics/special/_kernels.ts'
import { numberParam } from '../../core/args.ts'
import { ParameterError } from '../../core/errors.ts'
import type { Params } from '../../core/types.ts'
import { gaussianNodes } from '../../numerics/_quadrature.ts'

/** The classes and the outcome from the parameters, checked. */
export function classesOf(params: Params): { readonly classes: number; readonly y: number; readonly probit: boolean } {
  const classes = numberParam(params, 'classes')
  const y = numberParam(params, 'y')
  if (!(Number.isInteger(classes) && classes >= 2))
    throw new ParameterError('multiclass-site: classes is an integer of at least 2')
  if (!(Number.isInteger(y) && y >= 0 && y < classes))
    throw new ParameterError(`multiclass-site: the class must be an integer from 0 to ${classes - 1}`)
  const link = params['link'] ?? 'logit'
  if (link !== 'logit' && link !== 'probit')
    throw new ParameterError("multiclass-site: the link is 'logit' or 'probit'")
  return { classes, y, probit: link === 'probit' }
}

/** log of the sum of `exp(v)` (`-Infinity` for none). */
export function logSumExp(v: readonly number[]): number {
  let top = Number.NEGATIVE_INFINITY
  for (const x of v) if (x > top) top = x
  if (!Number.isFinite(top)) return top
  let s = 0
  for (const x of v) s += Math.exp(x - top)
  return top + Math.log(s)
}

/** log P(y | z) under the softmax. */
export function logSoftmax(z: ArrayLike<number>, y: number): number {
  let top = Number.NEGATIVE_INFINITY
  for (let k = 0; k < z.length; k++) if ((z[k] as number) > top) top = z[k] as number
  let s = 0
  for (let k = 0; k < z.length; k++) s += Math.exp((z[k] as number) - top)
  return (z[y] as number) - top - Math.log(s)
}

const E_NODES = gaussianNodes(0, 1, 20)

/** The log of the standard normal density. */
const logPhi = (t: number): number => -0.5 * t * t - 0.9189385332046727

/** The inverse Mills ratio `phi(t) / Phi(t)`, the derivative of `log Phi`, stable in the lower tail. */
const millsRatio = (t: number): number => Math.exp(logPhi(t) - logNdtr(t))

/**
 * log P(y | z) under the multinomial probit by the same integral over `e_y` as {@link logProbit}, laid at the integrand's own mode: the
 * integrand `phi(e) prod_j Phi(d_j + e)` (`d_j = z_y - z_j`) is log-concave, so Newton's method with a bracket finds its peak, and 20
 * Gauss-Hermite nodes of its curvature there, summed in log space, resolve it wherever it is (the peak of a class that is twenty below
 * the chosen one is at `e = 10`, where `logProbit`'s fixed nodes stop at 7.6 and its product of normal cdfs underflows).
 */
function logProbitAtMode(d: readonly number[]): number {
  let lo = 0
  let hi = 8 - Math.min(...d)
  let e = Math.min(Math.max(0, -Math.min(...d) * 0.5), hi)
  const slope = (t: number): number => d.reduce((s, dj) => s + millsRatio(dj + t), -t)
  for (let it = 0; it < 100; it++) {
    const g = slope(e)
    if (g > 0) lo = e
    else hi = e
    const curvature = -1 - d.reduce((s, dj) => s + millsRatio(dj + e) * (dj + e + millsRatio(dj + e)), 0)
    let next = e - g / curvature
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi)
    if (Math.abs(next - e) <= 1e-12 * (1 + Math.abs(e))) {
      e = next
      break
    }
    e = next
  }
  const precision = 1 + d.reduce((s, dj) => s + millsRatio(dj + e) * (dj + e + millsRatio(dj + e)), 0)
  const sd = 1 / Math.sqrt(precision)
  const logw = Float64Array.from(E_NODES.x, (x, i) => {
    const t = e + sd * x
    return Math.log(E_NODES.w[i] as number) + 0.5 * x * x + logPhi(t) + d.reduce((s, dj) => s + logNdtr(dj + t), 0)
  })
  let top = Number.NEGATIVE_INFINITY
  for (const v of logw) if (v > top) top = v
  let total = 0
  for (const v of logw) total += Math.exp(v - top)
  return top + Math.log(total) + Math.log(sd) + 0.9189385332046727
}

/**
 * The total by which the other classes' scores exceed the chosen one's (`sum_j max(0, z_j - z_y)`) above which {@link logProbit} lays its
 * integral at the mode: the integrand then peaks where its fixed 20 nodes thin out, and the probability is below about `1e-5`. Below
 * it (every ordinary evaluation) the fixed nodes keep the digits they always had.
 */
const TAIL = 6

/** log P(y | z) under the multinomial probit: the largest of `z_k + e_k`, `e_k` standard normal; one integral over `e_y`. */
export function logProbit(z: ArrayLike<number>, y: number): number {
  let deficit = 0
  for (let j = 0; j < z.length; j++) if (j !== y) deficit += Math.max(0, (z[j] as number) - (z[y] as number))
  if (deficit > TAIL) {
    const d: number[] = []
    for (let j = 0; j < z.length; j++) if (j !== y) d.push((z[y] as number) - (z[j] as number))
    return logProbitAtMode(d)
  }
  let total = 0
  for (let i = 0; i < E_NODES.x.length; i++) {
    let p = 1
    for (let j = 0; j < z.length; j++)
      if (j !== y) p *= ndtr((z[y] as number) - (z[j] as number) + (E_NODES.x[i] as number))
    total += (E_NODES.w[i] as number) * p
  }
  return Math.log(total)
}

/** The log probability of the observed class at the scores: the softmax's or the multinomial probit's. */
export function scoreLog(params: Params): (z: readonly number[]) => number {
  const { y, probit } = classesOf(params)
  return probit ? (z) => logProbit(z, y) : (z) => logSoftmax(z, y)
}

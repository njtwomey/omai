// The adaptive Gauss-Hermite rule on a product grid: the log mass, mean and covariance of a density that is a Gaussian cavity times a
// smooth factor over up to six coordinates, with a node for every combination of them and a full-covariance proposal that follows the
// tilted moments (the cavity first, then the Gaussian of the moments just found, three rounds). Exact for a smooth factor and well behaved
// when the cavity is much wider than the factor, as in the first sweeps. The multiclass sites call it over their class scores; the sites
// on a Gaussian vector (`_vector-site.ts`) over the standardised coordinates of the cavity of their coordinates.

import { ImproperMessageError, ParameterError } from '../core/errors.ts'
import { zeroArray } from './_arrays.ts'
import { choleskyInPlace } from './_bartlett.ts'
import { shiftedSum } from './_moments.ts'
import { gaussianNodes } from './_quadrature.ts'

/** The nodes per free score for a number of free scores (the grid is their product): the cap keeps a site at some tens of thousands of evaluations. */
const NODES: Readonly<Record<number, number>> = { 1: 48, 2: 24, 3: 8, 4: 7, 5: 6, 6: 5 }

/** The most free scores the product grid integrates; more go to the auxiliary-variable rule. */
export const MOST_PRODUCT = 6

/** Rounds of adaptation of the proposal: the first uses the cavity, each next the Gaussian of the tilted moments of the one before. */
const ROUNDS = 3

/** The log mass and the moments of the tilted density of the product grid. */
export interface ProductTilt {
  readonly logMass: number
  readonly mean: readonly number[]
  /** The tilted covariance, row-major `r x r`. */
  readonly covariance: readonly number[]
}

/**
 * The log mass, the mean and the covariance of a density proportional to `exp(logCavity(x) + logTarget(x))` over `r` coordinates
 * (one to six), by adaptive Gauss-Hermite quadrature on the product grid with a full-covariance proposal: the Gaussian of `start`
 * first, then each round the Gaussian of the tilted moments just found, up to three rounds. Exact for a smooth target and well
 * behaved when the cavity is much wider than the target, as in the first sweeps.
 *
 * @param r - The number of coordinates.
 * @param start - The first proposal: the cavity's mean and covariance (row-major).
 * @param logCavity - The log density of the cavity, normalised.
 * @param logTarget - The log of the factor (to its power) at a point.
 * @returns The log mass of the target under the normalised cavity, and the tilted moments.
 * @throws ParameterError for more coordinates than the grid reaches.
 * @throws ImproperMessageError when the target has no mass under the cavity or the tilted covariance is not positive definite.
 */
export function productTilt(
  r: number,
  start: { readonly mean: readonly number[]; readonly covariance: readonly number[] },
  logCavity: (x: readonly number[]) => number,
  logTarget: (x: readonly number[]) => number,
): ProductTilt {
  const n = NODES[r]
  if (n === undefined)
    throw new ParameterError(`product grid: ${r} free scores are more than the six the product rule integrates`)
  const grid = gaussianNodes(0, 1, n)
  let mu = [...start.mean]
  let cov = [...start.covariance]
  let result: ProductTilt | undefined
  const total = n ** r
  for (let round = 0; round < ROUNDS; round++) {
    const l = Float64Array.from(cov)
    if (!choleskyInPlace(l, r))
      throw new ImproperMessageError('product grid: the tilted covariance is not positive definite')
    let logDet = 0
    for (let i = 0; i < r; i++) logDet += 2 * Math.log(l[i * r + i] as number)
    // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
    const logw = new Float64Array(total)
    const xs = new Float64Array(total * r)
    const t = zeroArray(r)
    const x = zeroArray(r)
    for (let idx = 0; idx < total; idx++) {
      let rest = idx
      let lw = 0
      for (let i = r - 1; i >= 0; i--) {
        const j = rest % n
        rest = (rest - j) / n
        t[i] = grid.x[j] as number
        lw += Math.log(grid.w[j] as number)
      }
      // x = mu + L t; the proposal's log density at x is -r/2 log 2 pi - logdet/2 - |t|^2 / 2.
      let tt = 0
      for (let i = 0; i < r; i++) {
        let s = mu[i] as number
        for (let k = 0; k <= i; k++) s += (l[i * r + k] as number) * (t[k] as number)
        x[i] = s
        xs[idx * r + i] = s
        tt += (t[i] as number) ** 2
      }
      // The node's weight is w_t f(x) N_cav(x) / N_proposal(x); the Gaussian weights of the product rule already stand for N_proposal.
      const lq = -0.5 * (r * Math.log(2 * Math.PI) + logDet + tt)
      const v = lw + logTarget(x) + logCavity(x) - lq
      logw[idx] = v
    }
    const { top, weights, total: sum } = shiftedSum(logw)
    if (!Number.isFinite(top)) throw new ImproperMessageError('product grid: the factor has no mass under the cavities')
    const m = zeroArray(r)
    for (let idx = 0; idx < total; idx++) {
      const w = weights[idx] as number
      for (let i = 0; i < r; i++) m[i] = (m[i] as number) + w * (xs[idx * r + i] as number)
    }
    for (let i = 0; i < r; i++) m[i] = (m[i] as number) / sum
    const v = zeroArray(r * r)
    for (let idx = 0; idx < total; idx++) {
      const w = weights[idx] as number
      for (let i = 0; i < r; i++) {
        const di = (xs[idx * r + i] as number) - (m[i] as number)
        for (let k = 0; k <= i; k++)
          v[i * r + k] = (v[i * r + k] as number) + w * di * ((xs[idx * r + k] as number) - (m[k] as number))
      }
    }
    for (let i = 0; i < r; i++)
      for (let k = 0; k <= i; k++) {
        const e = (v[i * r + k] as number) / sum
        v[i * r + k] = e
        v[k * r + i] = e
      }
    const done =
      result !== undefined &&
      m.every((mi, i) => Math.abs(mi - (result?.mean[i] as number)) <= 1e-3 * Math.sqrt(v[i * r + i] as number)) &&
      m.every((_, i) => Math.abs((v[i * r + i] as number) / (result?.covariance[i * r + i] as number) - 1) <= 1e-3)
    result = { logMass: top + Math.log(sum), mean: m, covariance: v }
    mu = m
    cov = v
    if (done) break
  }
  return result as ProductTilt
}

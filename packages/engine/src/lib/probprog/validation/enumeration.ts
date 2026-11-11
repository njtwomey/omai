/**
 * Oracles that sum out a discrete part by enumeration: the exact posterior of a spike and slab linear regression with
 * Gaussian noise, every subset of the coefficients in the slab (2^p of them) weighed by its marginal likelihood, the
 * Gaussian coefficients integrated in closed form and an unknown noise precision integrated on a grid. It is the exact
 * inclusion probability and posterior mean of each coefficient that an approximation (EP, NUTS on the mixture) is checked
 * against, feasible for up to about sixteen coefficients.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'

/** The prior and data of a spike and slab regression, `y = X w + noise`, with each `w_j` a mixture of two zero-mean normals. */
export interface SpikeSlabRegression {
  /** The design, one row per observation. */
  readonly X: readonly (readonly number[])[]
  /** The responses. */
  readonly y: readonly number[]
  /** The prior probability that a coefficient is in the slab. */
  readonly inclusion: number
  /** The sd of the slab. */
  readonly slab: number
  /** The sd of the spike. */
  readonly spike: number
  /** The noise: a known sd, or a gamma prior on the precision (`shape`, `rate`). */
  readonly noise: { readonly sd: number } | { readonly shape: number; readonly rate: number }
  /** The grid points over the log precision when it is unknown (default 61). */
  readonly gridPoints?: number
}

/** The exact posterior of a spike and slab regression. */
export interface SpikeSlabPosterior {
  /** P(coefficient j is in the slab | data), one per coefficient. */
  readonly inclusion: readonly number[]
  /** The posterior mean of each coefficient. */
  readonly mean: readonly number[]
  /** The posterior sd of each coefficient. */
  readonly sd: readonly number[]
  /** The posterior mean of the noise precision (the known one when it is known). */
  readonly precision: number
  /** The posterior sd of the noise precision (zero when it is known). */
  readonly precisionSd: number
  /** log p(y) under the prior (the grid's estimate when the precision is unknown). */
  readonly logEvidence: number
  /** How many subsets were summed. */
  readonly subsets: number
}

/** How far the precision grid reaches from its centre, as a factor each way. */
const SPREAD = 10

/** The weight of one (subset, precision) cell and what it needs to give moments afterwards. */
interface Cell {
  readonly subset: number
  readonly point: number
  readonly logWeight: number
}

/**
 * The exact posterior of a spike and slab linear regression, by summing over all 2^p subsets of coefficients in the slab:
 * for a subset and a noise precision tau the coefficients are Gaussian (precision `D^-1 + tau X'X`), so the marginal
 * likelihood is closed-form; the subsets are weighed by it and by the Bernoulli prior, and an unknown precision is
 * integrated on a grid over its log (centred on the precision a ridge fit leaves, spanning a factor of ten each way). The moments
 * of each coefficient are the weighted moments of those Gaussians.
 *
 * @param problem - The design, responses, prior and noise.
 * @returns The inclusion probability, mean and sd of each coefficient, the precision and the evidence.
 * @concept posterior-validation
 * @index function Exact spike and slab regression posterior by enumerating the 2^p subsets of coefficients.
 * @example
 * ```ts
 * import { spikeSlabRegression } from 'omai-engine/probprog/validation'
 * const post = spikeSlabRegression({ X: [[1], [2]], y: [1.1, 2.1], inclusion: 0.5, slab: 2, spike: 0.05, noise: { sd: 0.5 } })
 * [post.subsets, post.inclusion[0]! > 0.99] // => [2, true]
 * ```
 */
export function spikeSlabRegression(problem: SpikeSlabRegression): SpikeSlabPosterior {
  const { X, y, inclusion, slab, spike } = problem
  const n = y.length
  const p = X[0]?.length ?? 0
  if (n === 0 || p === 0) throw new OmaiError('shape', 'spikeSlabRegression: the design is empty')
  if (p > 22)
    throw new OmaiError('value', 'spikeSlabRegression: more than 22 coefficients is too many subsets to enumerate')
  if (!(inclusion > 0 && inclusion < 1 && slab > 0 && spike > 0))
    throw new OmaiError('value', 'spikeSlabRegression: the inclusion probability is in (0, 1), the scales are positive')
  const gram = new Float64Array(p * p)
  const xty = new Float64Array(p)
  let yty = 0
  for (let i = 0; i < n; i++) {
    const row = X[i] as readonly number[]
    const yi = y[i] as number
    yty += yi * yi
    for (let a = 0; a < p; a++) {
      xty[a] = (xty[a] ?? 0) + (row[a] as number) * yi
      for (let b = 0; b < p; b++) gram[a * p + b] = (gram[a * p + b] ?? 0) + (row[a] as number) * (row[b] as number)
    }
  }
  // The precision grid: one point when it is known, else a log grid around the noise of the full least-squares fit.
  const known = 'sd' in problem.noise
  const taus: number[] = []
  const logPrior: number[] = []
  if ('sd' in problem.noise) {
    taus.push(1 / problem.noise.sd ** 2)
    logPrior.push(0)
  } else {
    const { shape, rate } = problem.noise
    const points = problem.gridPoints ?? 61
    // The centre: the precision the posterior mean of a ridge fit (every coefficient with the slab's prior) leaves, a few times over.
    let centre = (n / 2 + shape) / (yty / 2 + rate)
    for (let round = 0; round < 4; round++) {
      const ridge = Float64Array.from(gram)
      for (let j = 0; j < p; j++) ridge[j * p + j] = (ridge[j * p + j] ?? 0) + 1 / (centre * slab * slab)
      const w = dpotrs(dpotrf(ridge, p).factor, xty, p)
      let fit = 0
      for (let j = 0; j < p; j++) fit += (w[j] as number) * (xty[j] as number)
      centre = (n / 2 + shape) / (Math.max(yty - fit, 1e-12) / 2 + rate)
    }
    const lo = Math.log(centre / SPREAD)
    const hi = Math.log(centre * SPREAD)
    const h = (hi - lo) / (points - 1)
    for (let k = 0; k < points; k++) {
      const t = lo + k * h
      taus.push(Math.exp(t))
      // log gamma density at tau, times tau (the Jacobian of the log grid), times the grid step.
      logPrior.push(shape * Math.log(rate) - item(lgamma(shape)) + shape * t - rate * Math.exp(t) + Math.log(h))
    }
  }
  const logPi = Math.log(inclusion)
  const log1Pi = Math.log(1 - inclusion)
  const variances = [spike ** 2, slab ** 2]
  const cells: Cell[] = []
  const a = new Float64Array(p * p)
  const subsets = 2 ** p
  for (let subset = 0; subset < subsets; subset++) {
    let logPrior0 = 0
    let logDetD = 0
    for (let j = 0; j < p; j++) {
      const inSlab = (subset >> j) & 1
      logPrior0 += inSlab === 1 ? logPi : log1Pi
      logDetD += Math.log(variances[inSlab] as number)
    }
    for (let k = 0; k < taus.length; k++) {
      const tau = taus[k] as number
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++)
          a[i * p + j] =
            tau * (gram[i * p + j] as number) + (i === j ? 1 / (variances[(subset >> i) & 1] as number) : 0)
      const { factor, info } = dpotrf(a, p)
      if (info !== 0) continue
      let logDetA = 0
      for (let j = 0; j < p; j++) logDetA += 2 * Math.log(factor[j * p + j] as number)
      const solved = dpotrs(factor, xty, p)
      let quad = 0
      for (let j = 0; j < p; j++) quad += (xty[j] as number) * (solved[j] as number)
      const logLik =
        -0.5 * n * Math.log(2 * Math.PI) +
        0.5 * n * Math.log(tau) -
        0.5 * logDetD -
        0.5 * logDetA -
        0.5 * (tau * yty - tau * tau * quad)
      cells.push({ subset, point: k, logWeight: logPrior0 + (logPrior[k] as number) + logLik })
    }
  }
  const logZ = logSumExpOf(cells.map((c) => c.logWeight))
  const incl = new Float64Array(p)
  const mean = new Float64Array(p)
  const second = new Float64Array(p)
  let tauMean = 0
  let tauSecond = 0
  for (const c of cells) {
    const w = Math.exp(c.logWeight - logZ)
    const tau = taus[c.point] as number
    tauMean += w * tau
    tauSecond += w * tau * tau
    for (let j = 0; j < p; j++) if ((c.subset >> j) & 1) incl[j] = (incl[j] ?? 0) + w
    // The moments of the coefficients need the inverse's diagonal: only for cells that matter.
    if (w < 1e-10) continue
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++)
        a[i * p + j] =
          tau * (gram[i * p + j] as number) + (i === j ? 1 / (variances[(c.subset >> i) & 1] as number) : 0)
    const { factor } = dpotrf(a, p)
    const m = dpotrs(factor, xty, p)
    const unit = new Float64Array(p)
    for (let j = 0; j < p; j++) {
      unit.fill(0)
      unit[j] = 1
      const column = dpotrs(factor, unit, p)
      const mj = tau * (m[j] as number)
      mean[j] = (mean[j] ?? 0) + w * mj
      second[j] = (second[j] ?? 0) + w * (mj * mj + (column[j] as number))
    }
  }
  return {
    inclusion: Array.from(incl),
    mean: Array.from(mean),
    sd: Array.from(mean, (m, j) => Math.sqrt(Math.max(0, (second[j] as number) - m * m))),
    precision: tauMean,
    precisionSd: known ? 0 : Math.sqrt(Math.max(0, tauSecond - tauMean * tauMean)),
    logEvidence: logZ,
    subsets,
  }
}

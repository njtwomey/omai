/**
 * Oracles that sum out a discrete part by enumeration: the exact posterior of a spike and slab linear regression with
 * Gaussian noise, every subset of the coefficients in the slab (2^p of them) weighed by its marginal likelihood, the
 * Gaussian coefficients integrated in closed form and an unknown noise precision integrated on a grid (feasible for up to
 * about sixteen coefficients); of a mixture of linear regressions, every assignment of the rows to classes weighed by its marginal likelihood, each class's coefficients Gaussian given its rows; and of a zero-inflated Poisson regression, the structural zero of each count summed out and
 * the inflation probability and the two coefficients on a refined grid. They are the exact answers that an approximation
 * (EP, NUTS on the mixture) is checked against.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { full, item, toArray, toFlat, zeros } from '../../tensor/index.ts'

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
  // The normal equations are kernel buffers: dpotrf and dpotrs take and return Float64Arrays, once per (subset, precision) cell.
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
  const incl = toFlat(zeros([p]))
  const mean = toFlat(zeros([p]))
  const second = toFlat(zeros([p]))
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
    inclusion: incl,
    mean,
    sd: mean.map((m, j) => Math.sqrt(Math.max(0, (second[j] as number) - m * m))),
    precision: tauMean,
    precisionSd: known ? 0 : Math.sqrt(Math.max(0, tauSecond - tauMean * tauMean)),
    logEvidence: logZ,
    subsets,
  }
}

/** A zero-inflated Poisson regression with one covariate: each count is a structural zero with probability `pi`, else Poisson with rate `exp(b0 + b1 x)`. */
export interface ZeroInflatedPoisson {
  /** The covariate of each count. */
  readonly x: readonly number[]
  /** The counts. */
  readonly y: readonly number[]
  /** The Beta prior of the inflation probability. */
  readonly inflation: { readonly a: number; readonly b: number }
  /** The sd of the zero-mean normal priors of the intercept and the slope. */
  readonly coefficientSd: number
  /** The grid points per axis (default 40). */
  readonly points?: number
}

/** The exact posterior of a zero-inflated Poisson regression. */
export interface ZeroInflatedPosterior {
  /** The mean and sd of the inflation probability, the intercept and the slope. */
  readonly pi: { readonly mean: number; readonly sd: number }
  readonly b0: { readonly mean: number; readonly sd: number }
  readonly b1: { readonly mean: number; readonly sd: number }
  /** For each count, the probability that it is a structural zero (zero for a positive count). */
  readonly structural: readonly number[]
  /** log p(y) under the prior. */
  readonly logEvidence: number
}

/**
 * The exact posterior of a zero-inflated Poisson regression: the structural zero of each count is summed out (the likelihood
 * of a zero is `pi + (1 - pi) exp(-rate)`, of a positive count `(1 - pi)` times its Poisson mass), and the inflation
 * probability and the two coefficients are integrated on a grid, coarse over the prior's range and then refined around where
 * the posterior has its mass (its mean plus or minus five sds), until the cells are fine against its spread.
 *
 * @param problem - The counts, the covariate and the priors.
 * @returns The moments of the three parameters, each count's probability of being a structural zero and the evidence.
 * @concept posterior-validation
 * @index function Exact zero-inflated Poisson regression posterior on a refined grid, structural zeros summed out.
 * @example
 * ```ts
 * import { zeroInflatedPoisson } from 'omai-engine/probprog/validation'
 * const post = zeroInflatedPoisson({ x: [0, 1, 2, 0, 1, 2], y: [0, 0, 0, 1, 3, 7], inflation: { a: 1, b: 1 }, coefficientSd: 2 })
 * [post.structural[0] > 0.2, post.structural[3]] // => [true, 0]
 * ```
 */
export function zeroInflatedPoisson(problem: ZeroInflatedPoisson): ZeroInflatedPosterior {
  const { x, y, inflation, coefficientSd } = problem
  const n = y.length
  const points = problem.points ?? 40
  if (n === 0 || x.length !== n) throw new OmaiError('shape', 'zeroInflatedPoisson: need one covariate per count')
  const logFactorial = y.map((v) => {
    let s = 0
    for (let k = 2; k <= v; k++) s += Math.log(k)
    return s
  })
  const logBeta = item(lgamma(inflation.a)) + item(lgamma(inflation.b)) - item(lgamma(inflation.a + inflation.b))
  type Box = readonly [readonly [number, number], readonly [number, number], readonly [number, number]]
  const sweep = (
    box: Box,
    m: number,
  ): { moments: { mean: number; sd: number }[]; structural: number[]; logZ: number } => {
    const axis = (k: 0 | 1 | 2, i: number): number => box[k][0] + ((i + 0.5) / m) * (box[k][1] - box[k][0])
    const volume = box.reduce((v, [lo, hi]) => (v * (hi - lo)) / m, 1)
    const logWeights: number[] = []
    const where: [number, number, number][] = []
    for (let a = 0; a < m; a++)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < m; j++) {
          const [pi, b0, b1] = [axis(0, a), axis(1, i), axis(2, j)]
          let l =
            (inflation.a - 1) * Math.log(pi) +
            (inflation.b - 1) * Math.log(1 - pi) -
            logBeta -
            (b0 * b0 + b1 * b1) / (2 * coefficientSd ** 2) -
            Math.log(2 * Math.PI * coefficientSd ** 2)
          for (let r = 0; r < n; r++) {
            const rate = Math.exp(b0 + b1 * (x[r] as number))
            const count = y[r] as number
            const poisson = Math.exp(-rate + count * Math.log(rate) - (logFactorial[r] as number))
            l += Math.log((count === 0 ? pi : 0) + (1 - pi) * poisson)
          }
          logWeights.push(l)
          where.push([pi, b0, b1])
        }
    const logZ = logSumExpOf(logWeights) + Math.log(volume)
    const sums = [0, 0, 0]
    const squares = [0, 0, 0]
    const structural = toFlat(zeros([n]))
    logWeights.forEach((lw, g) => {
      const w = Math.exp(lw - logZ) * volume
      const point = where[g] as [number, number, number]
      for (let k = 0; k < 3; k++) {
        sums[k] = (sums[k] as number) + w * (point[k] as number)
        squares[k] = (squares[k] as number) + w * (point[k] as number) ** 2
      }
      if (w < 1e-12) return
      for (let r = 0; r < n; r++) {
        if (y[r] !== 0) continue
        const rate = Math.exp(point[1] + point[2] * (x[r] as number))
        structural[r] = (structural[r] as number) + (w * point[0]) / (point[0] + (1 - point[0]) * Math.exp(-rate))
      }
    })
    const moments = sums.map((m1, k) => ({ mean: m1, sd: Math.sqrt(Math.max(0, (squares[k] as number) - m1 * m1)) }))
    return { moments, structural, logZ }
  }
  // Zoom: a box over the prior's range, then boxes around where the mass was found, until the cells are fine against the spread.
  const reach = Math.max(8, 4 * coefficientSd)
  let box: Box = [
    [0.005, 0.995],
    [-reach, reach],
    [-reach, reach],
  ]
  let result = sweep(box, points)
  for (let round = 0; round < 8; round++) {
    const fine = result.moments.every(
      (m, k) => (box[k as 0 | 1 | 2][1] - box[k as 0 | 1 | 2][0]) / points <= 0.35 * m.sd,
    )
    if (fine) break
    const next = result.moments.map((m, k): readonly [number, number] => {
      const cell = (box[k as 0 | 1 | 2][1] - box[k as 0 | 1 | 2][0]) / points
      const half = Math.max(5 * m.sd, 2 * cell)
      return k === 0
        ? [Math.max(1e-4, m.mean - half), Math.min(1 - 1e-4, m.mean + half)]
        : [m.mean - half, m.mean + half]
    })
    box = next as unknown as Box
    result = sweep(box, points)
  }
  const [pi, b0, b1] = result.moments as [
    { mean: number; sd: number },
    { mean: number; sd: number },
    { mean: number; sd: number },
  ]
  return { pi, b0, b1, structural: result.structural, logEvidence: result.logZ }
}

/** The prior and data of a mixture of linear regressions on a shared design: a class per row, a coefficient vector per class. */
export interface MixtureOfRegressions {
  /** The design, one row per observation. */
  readonly X: readonly (readonly number[])[]
  /** The responses. */
  readonly y: readonly number[]
  /** The prior mean of each class's coefficients, `priorMean[class][feature]`. */
  readonly priorMean: readonly (readonly number[])[]
  /** The prior sd of every coefficient. */
  readonly priorSd: number
  /** The known noise sd. */
  readonly noiseSd: number
  /** The probability of each class a priori (default equal). */
  readonly weights?: readonly number[]
}

/** The exact posterior of a mixture of regressions. */
export interface MixtureOfRegressionsPosterior {
  /** For each class and feature, the mixture over assignments of the Gaussian posterior of the coefficient (`coefficients[class][feature]`). */
  readonly coefficients: readonly (readonly {
    readonly mean: number
    readonly sd: number
    readonly components: readonly { readonly weight: number; readonly mean: number; readonly sd: number }[]
  }[])[]
  /** P(row i is in class k | data), `membership[i][k]`. */
  readonly membership: readonly (readonly number[])[]
  /** log p(y) under the prior. */
  readonly logEvidence: number
  /** How many assignments were summed. */
  readonly assignments: number
}

/**
 * The exact posterior of a mixture of linear regressions with a known noise: every assignment of the rows to classes
 * (`K^n` of them) is weighed by the marginal likelihood of its rows under each class's Gaussian prior (closed form) and the
 * class probabilities; given an assignment each class's coefficients are Gaussian, so the posterior of a coefficient is a mixture of
 * Gaussians over the assignments. The exact answer that EP on the model layer's `pick` gate is checked against.
 *
 * @param problem - The design, responses, prior and noise.
 * @returns The mixture of each coefficient, each row's class probabilities and the evidence.
 * @concept posterior-validation
 * @index function Exact mixture-of-regressions posterior by enumerating the assignments of rows to classes.
 * @example
 * ```ts
 * import { mixtureOfRegressions } from 'omai-engine/probprog/validation'
 * const post = mixtureOfRegressions({ X: [[1], [1]], y: [3, -3], priorMean: [[3], [-3]], priorSd: 1, noiseSd: 0.5 })
 * [post.assignments, post.membership[0]![0]! > 0.99] // => [4, true]
 * ```
 */
export function mixtureOfRegressions(problem: MixtureOfRegressions): MixtureOfRegressionsPosterior {
  const { X, y, priorMean, priorSd, noiseSd } = problem
  const n = y.length
  const classes = priorMean.length
  const p = X[0]?.length ?? 0
  if (n === 0 || p === 0 || classes === 0) throw new OmaiError('shape', 'mixtureOfRegressions: the design is empty')
  const assignments = classes ** n
  if (assignments > 2 ** 20) throw new OmaiError('value', 'mixtureOfRegressions: too many assignments to enumerate')
  const weights = problem.weights ?? toFlat(full([classes], 1 / classes))
  const tau = 1 / noiseSd ** 2
  const prior = 1 / priorSd ** 2
  type Part = { mean: Float64Array; variance: Float64Array }
  const logw = toFlat(zeros([assignments]))
  const parts: Part[][] = []
  for (let a = 0; a < assignments; a++) {
    const rows: number[][] = Array.from({ length: classes }, () => [])
    for (let i = 0, r = a; i < n; i++, r = Math.floor(r / classes)) (rows[r % classes] as number[]).push(i)
    let lw = 0
    const here: Part[] = []
    for (let k = 0; k < classes; k++) {
      const m0 = priorMean[k] as readonly number[]
      const members = rows[k] as number[]
      lw += members.length * Math.log(weights[k] as number)
      // kernel buffers of dpotrf and dpotrs (Float64Arrays), one set per class of every assignment
      const A = new Float64Array(p * p)
      const b = new Float64Array(p)
      let yty = 0
      for (let j = 0; j < p; j++) {
        A[j * p + j] = prior
        b[j] = prior * (m0[j] as number)
      }
      for (const i of members) {
        const row = X[i] as readonly number[]
        yty += (y[i] as number) ** 2
        for (let j = 0; j < p; j++) {
          b[j] = (b[j] as number) + tau * (row[j] as number) * (y[i] as number)
          for (let l = 0; l < p; l++)
            A[j * p + l] = (A[j * p + l] as number) + tau * (row[j] as number) * (row[l] as number)
        }
      }
      const { factor } = dpotrf(A, p)
      let logDetA = 0
      for (let j = 0; j < p; j++) logDetA += 2 * Math.log(factor[j * p + j] as number)
      const mean = dpotrs(factor, b, p)
      let bAb = 0
      let m0m0 = 0
      for (let j = 0; j < p; j++) {
        bAb += (b[j] as number) * (mean[j] as number)
        m0m0 += (m0[j] as number) ** 2
      }
      lw +=
        -0.5 * members.length * Math.log(2 * Math.PI * noiseSd ** 2) -
        0.5 * p * Math.log(priorSd ** 2) -
        0.5 * logDetA -
        0.5 * (tau * yty + prior * m0m0 - bAb)
      const variance = new Float64Array(p)
      const unit = new Float64Array(p)
      for (let j = 0; j < p; j++) {
        unit.fill(0)
        unit[j] = 1
        variance[j] = dpotrs(factor, unit, p)[j] as number
      }
      here.push({ mean, variance })
    }
    logw[a] = lw
    parts.push(here)
  }
  const logEvidence = logSumExpOf(logw)
  const membership = toArray(zeros([n, classes])) as number[][]
  const coefficients = Array.from({ length: classes }, (_, k) =>
    Array.from({ length: p }, (_, j) => {
      const components: { weight: number; mean: number; sd: number }[] = []
      let m1 = 0
      let m2 = 0
      for (let a = 0; a < assignments; a++) {
        const w = Math.exp((logw[a] as number) - logEvidence)
        const part = (parts[a] as Part[])[k] as Part
        if (w > 1e-12)
          components.push({ weight: w, mean: part.mean[j] as number, sd: Math.sqrt(part.variance[j] as number) })
        m1 += w * (part.mean[j] as number)
        m2 += w * ((part.variance[j] as number) + (part.mean[j] as number) ** 2)
      }
      return { mean: m1, sd: Math.sqrt(Math.max(m2 - m1 * m1, 0)), components }
    }),
  )
  for (let a = 0; a < assignments; a++) {
    const w = Math.exp((logw[a] as number) - logEvidence)
    for (let i = 0, r = a; i < n; i++, r = Math.floor(r / classes)) {
      const row = membership[i] as number[]
      row[r % classes] = (row[r % classes] as number) + w
    }
  }
  return { coefficients, membership, logEvidence, assignments }
}

/**
 * Conjugate Bayesian linear regression with a zero-mean Gaussian prior of precision `alpha` on the weights and known
 * noise precision `beta` (Bishop, PRML 3.3): the batch posterior in closed form, the same posterior reached one
 * observation at a time as a traced `Algorithm`, the posterior predictive at new inputs, the log marginal likelihood
 * (the evidence) and its maximisation over `alpha` and `beta`, and a polynomial basis to regress on. The sequential
 * update adds one rank-one term to the precision and one to the information vector, so after every row it is the
 * batch posterior of the rows seen so far, in any order.
 */

import { LOG_2PI, MdmlError } from '../base/index.ts'
import { type MatrixView, matrixView } from '../learn/index.ts'
import { type MinMaxScalerParams, minMaxScaler, polynomialFeatures } from '../learn/preprocessing/index.ts'
import { cholesky, choleskyLogdet, choSolve, ddot, dgemv, eigvalsh } from '../numerics/linalg/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { eye, fromStorage, item, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { evidenceUpdates } from './bayesian-options.ts'
import type { Design } from './design.ts'

/** The prior precision of the weights and the (known) precision of the observation noise. */
export interface BayesianPrior {
  /** The precision of the zero-mean isotropic Gaussian prior on the weights. */
  readonly alpha: number
  /** The precision (inverse variance) of the Gaussian noise on the targets. */
  readonly beta: number
}

/**
 * The posterior over the weights after `n` observations: a Gaussian with the given mean and covariance (the inverse
 * of `precision`), the information vector `precision * mean` it accumulates, and the log marginal likelihood of the
 * `n` targets seen. Plain JSON.
 */
export interface BayesianPosterior {
  /** Observations absorbed. */
  readonly n: number
  readonly mean: number[]
  /** `alpha I + beta sum phi phi^T`, `[p][p]`. */
  readonly precision: number[][]
  /** The inverse of `precision`, `[p][p]`. */
  readonly covariance: number[][]
  /** `beta sum y phi`: what `precision * mean` equals. */
  readonly information: number[]
  /** `log p(y_1..y_n | phi_1..phi_n, alpha, beta)`; 0 before any row. */
  readonly logEvidence: number
}

/** The inputs of a regression: features `x` and targets `y` (weights are not used). */
export type BayesianDesign = Pick<Design, 'x' | 'y'>
const flatten = (m: readonly (readonly number[])[]): Float64Array => Float64Array.from(m.flat())

const nest = (flat: ArrayLike<number>, p: number): number[][] =>
  Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => flat[i * p + j] ?? 0))

function checkPrior(prior: BayesianPrior, who: string): void {
  if (!(prior.alpha > 0 && Number.isFinite(prior.alpha) && prior.beta > 0 && Number.isFinite(prior.beta)))
    throw new MdmlError('value', `${who}: alpha and beta must be positive and finite`)
}

/** The mean, covariance and log determinant of a Gaussian from its precision `[p][p]` and information vector. */
function solveGaussian(
  precision: readonly (readonly number[])[],
  information: readonly number[],
): { readonly mean: number[]; readonly covariance: number[][]; readonly logDet: number } {
  const p = information.length
  const l = cholesky(fromStorage(flatten(precision), [p, p], 'float64'))
  const cov = Float64Array.from(toFlat(choSolve(l, eye(p))))
  // The inverse of a symmetric matrix is symmetric: remove the rounding asymmetry.
  for (let i = 0; i < p; i++)
    for (let j = i + 1; j < p; j++) {
      const s = ((cov[i * p + j] ?? 0) + (cov[j * p + i] ?? 0)) / 2
      cov[i * p + j] = s
      cov[j * p + i] = s
    }
  return { mean: Array.from(dgemv(cov, p, p, information)), covariance: nest(cov, p), logDet: item(choleskyLogdet(l)) }
}

/**
 * The prior as a posterior over `p` weights: mean zero, covariance `I / alpha`, no observations.
 *
 * @param p - The number of weights (basis functions).
 * @param prior - The prior and noise precisions.
 * @returns The posterior before any data.
 * @example
 * ```ts
 * import { priorPosterior } from 'mdml-engine/linear'
 * priorPosterior(2, { alpha: 4, beta: 1 }).covariance // => [[0.25, 0], [0, 0.25]]
 * ```
 */
export function priorPosterior(p: number, prior: BayesianPrior): BayesianPosterior {
  checkPrior(prior, 'priorPosterior')
  if (!(Number.isInteger(p) && p >= 1)) throw new MdmlError('value', 'priorPosterior: p must be a positive integer')
  const eyeScaled = (v: number): number[][] =>
    Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? v : 0)))
  return {
    n: 0,
    mean: new Array<number>(p).fill(0),
    precision: eyeScaled(prior.alpha),
    covariance: eyeScaled(1 / prior.alpha),
    information: new Array<number>(p).fill(0),
    logEvidence: 0,
  }
}

/**
 * Absorb one observation: the precision gains `beta phi phi^T`, the information `beta y phi`, and the log evidence the
 * log density of `y` under the posterior predictive before the update (the chain rule of the marginal likelihood).
 *
 * @param posterior - The posterior so far.
 * @param phi - The basis functions at the new input, length `p`.
 * @param y - The new target.
 * @param beta - The noise precision.
 * @returns The posterior with the observation absorbed.
 * @example
 * ```ts
 * import { bayesianUpdate, priorPosterior } from 'mdml-engine/linear'
 * const next = bayesianUpdate(priorPosterior(1, { alpha: 1, beta: 1 }), [1], 2, 1)
 * next.mean[0].toFixed(4) // => '1.0000'
 * ```
 */
export function bayesianUpdate(
  posterior: BayesianPosterior,
  phi: ArrayLike<number>,
  y: number,
  beta: number,
): BayesianPosterior {
  const p = posterior.mean.length
  if (phi.length !== p) throw new MdmlError('shape', `bayesianUpdate: expected ${p} basis functions, got ${phi.length}`)
  const variance = 1 / beta + ddot(phi, dgemv(flatten(posterior.covariance), p, p, phi))
  const residual = y - ddot(phi, posterior.mean)
  const logPredictive = -0.5 * (LOG_2PI + Math.log(variance) + (residual * residual) / variance)
  const precision = posterior.precision.map((row, i) => row.map((v, j) => v + beta * (phi[i] ?? 0) * (phi[j] ?? 0)))
  const information = posterior.information.map((v, i) => v + beta * y * (phi[i] ?? 0))
  const { mean, covariance } = solveGaussian(precision, information)
  return {
    n: posterior.n + 1,
    mean,
    precision,
    covariance,
    information,
    logEvidence: posterior.logEvidence + logPredictive,
  }
}

/** The residual sum of squares of weights on a design. */
function residualSumOfSquares(design: BayesianDesign, mean: ArrayLike<number>): number {
  const { x, y } = design
  const fit = dgemv(x.data, x.n, x.d, mean)
  let s = 0
  for (let i = 0; i < x.n; i++) s += ((y[i] ?? 0) - (fit[i] ?? 0)) ** 2
  return s
}

/**
 * The batch posterior in closed form (Bishop 3.50-3.54): precision `alpha I + beta X^T X`, mean `beta S X^T y`, and the
 * log evidence `p/2 log alpha + n/2 log beta - E(m) - 1/2 log |precision| - n/2 log 2 pi`, with
 * `E(m) = beta/2 |y - X m|^2 + alpha/2 |m|^2`.
 *
 * @param design - Features `[n, p]` (already expanded into basis functions) and targets.
 * @param prior - The prior and noise precisions.
 * @returns The posterior after all rows.
 * @example
 * ```ts
 * import { bayesianLinearFit, designOf } from 'mdml-engine/linear'
 * const design = designOf({ x: [[1], [1]], y: [2, 2] }, 'example')
 * bayesianLinearFit(design, { alpha: 1, beta: 1 }).mean[0].toFixed(4) // => '1.3333'
 * ```
 */
export function bayesianLinearFit(design: BayesianDesign, prior: BayesianPrior): BayesianPosterior {
  checkPrior(prior, 'bayesianLinearFit')
  const { x, y } = design
  const { n, d: p } = x
  const gram = gramOf(x)
  const precision = nest(
    gram.map((v, k) => prior.beta * v + (Math.floor(k / p) === k % p ? prior.alpha : 0)),
    p,
  )
  const information = Array.from(dgemv(x.data, n, p, y, true), (v) => prior.beta * v)
  const { mean, covariance, logDet } = solveGaussian(precision, information)
  const energy = (prior.beta / 2) * residualSumOfSquares(design, mean) + (prior.alpha / 2) * ddot(mean, mean)
  const logEvidence =
    (p / 2) * Math.log(prior.alpha) + (n / 2) * Math.log(prior.beta) - energy - logDet / 2 - (n / 2) * LOG_2PI
  return { n, mean, precision, covariance, information, logEvidence }
}

/** `X^T X` of a design, `[p * p]` row-major. */
function gramOf(x: MatrixView): Float64Array {
  const { n, d: p, data } = x
  const g = new Float64Array(p * p)
  for (let i = 0; i < n; i++) {
    const row = data.subarray(i * p, (i + 1) * p)
    for (let a = 0; a < p; a++) {
      const ra = row[a] ?? 0
      for (let b = a; b < p; b++) g[a * p + b] = (g[a * p + b] ?? 0) + ra * (row[b] ?? 0)
    }
  }
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) g[a * p + b] = g[b * p + a] ?? 0
  return g
}

/** The predictive mean and variance of each row of a matrix view. */
function predictiveRows(
  posterior: Pick<BayesianPosterior, 'mean' | 'covariance'>,
  m: MatrixView,
  beta: number,
): { readonly mean: number[]; readonly variance: number[] } {
  const p = posterior.mean.length
  const cov = flatten(posterior.covariance)
  const mean: number[] = []
  const variance: number[] = []
  for (let i = 0; i < m.n; i++) {
    const row = m.data.subarray(i * m.d, (i + 1) * m.d)
    mean.push(ddot(row, posterior.mean))
    variance.push(1 / beta + ddot(row, dgemv(cov, p, p, row)))
  }
  return { mean, variance }
}

/**
 * The posterior predictive of a new target at each row (Bishop 3.58-3.59): mean `m^T phi` and variance
 * `1 / beta + phi^T S phi`, the noise plus the uncertainty about the weights.
 *
 * @param posterior - The posterior.
 * @param x - Basis functions at the new inputs, `[m, p]` (a tensor, nested arrays or a matrix view).
 * @param beta - The noise precision.
 * @returns The predictive mean and variance of each row.
 * @example
 * ```ts
 * import { posteriorPredictive, priorPosterior } from 'mdml-engine/linear'
 * posteriorPredictive(priorPosterior(1, { alpha: 1, beta: 4 }), [[2]], 4).variance // => [4.25]
 * ```
 */
export function posteriorPredictive(
  posterior: Pick<BayesianPosterior, 'mean' | 'covariance'>,
  x: TensorLike | MatrixView,
  beta: number,
): { readonly mean: number[]; readonly variance: number[] } {
  const view = isMatrixView(x) ? x : matrixView(x, 'posteriorPredictive', { nFeatures: posterior.mean.length })
  return predictiveRows(posterior, view, beta)
}

const isMatrixView = (x: TensorLike | MatrixView): x is MatrixView =>
  typeof x === 'object' && x !== null && !Array.isArray(x) && 'n' in x && 'd' in x && !('shape' in x)

/**
 * Bayesian linear regression as a traced algorithm that takes one observation per step, in the given order: each step
 * adds a rank-one term to the precision and the information vector, re-solves for the mean and covariance, and adds
 * the predictive log density of the new target to the log evidence. After `t` steps the state is the batch posterior of
 * the first `t` rows of `order`. The tape records the log evidence, the mean predictive standard deviation over the
 * design's rows (falling as data arrive), and the mean and covariance.
 *
 * @param design - Features `[n, p]` (basis functions) and targets.
 * @param prior - The prior and noise precisions.
 * @param order - The row indices to absorb, in order.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { bayesianLinearAlgorithm, designOf } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const design = designOf({ x: [[1], [1], [1]], y: [2, 2, 2] }, 'example')
 * const alg = bayesianLinearAlgorithm(design, { alpha: 1, beta: 1 }, [0, 1, 2])
 * runAlgorithm(alg, { maxSteps: 10 }).state.n // => 3
 * ```
 */
export function bayesianLinearAlgorithm(
  design: BayesianDesign,
  prior: BayesianPrior,
  order: readonly number[],
): Algorithm<BayesianPosterior> {
  checkPrior(prior, 'bayesianLinearAlgorithm')
  const { x, y } = design
  const p = x.d
  for (const i of order)
    if (!(Number.isInteger(i) && i >= 0 && i < x.n))
      throw new MdmlError('value', `bayesianLinearAlgorithm: row ${i} is outside 0..${x.n - 1}`)
  return {
    name: 'bayesian-linear-regression',
    config: { alpha: prior.alpha, beta: prior.beta, rows: order.length },
    schema: channels({
      metrics: {
        logEvidence: { description: 'the log marginal likelihood of the targets seen so far', direction: 'higher' },
        meanPredictiveStd: {
          description: 'the mean predictive standard deviation over the training rows',
          direction: 'lower',
        },
      },
      tensors: {
        mean: { role: 'param', description: 'the posterior mean of the weights', shape: [p] },
        covariance: { role: 'param', description: 'the posterior covariance of the weights, row-major', shape: [p, p] },
      },
    }),
    init: () => priorPosterior(p, prior),
    step: (s) => {
      const i = order[s.n] as number
      return bayesianUpdate(s, x.data.subarray(i * p, (i + 1) * p), y[i] ?? 0, prior.beta)
    },
    done: (s) => (s.n >= order.length ? 'converged' : undefined),
    observe: (s) => {
      const { variance } = predictiveRows(s, x, prior.beta)
      return {
        metrics: {
          logEvidence: s.logEvidence,
          meanPredictiveStd: variance.reduce((a, v) => a + Math.sqrt(v), 0) / Math.max(1, variance.length),
        },
        tensors: {
          mean: { role: 'param', value: s.mean, shape: [p] },
          covariance: { role: 'param', value: s.covariance.flat(), shape: [p, p] },
        },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BayesianPosterior,
  }
}

/** The paths {@link maximiseEvidenceAlgorithm} logs. */
export type EvidencePath = 'evidence/alpha' | 'evidence/beta' | 'evidence/gamma' | 'evidence/logEvidence'

/** What a stepped evidence maximisation logs: the precisions, the effective parameters and the log evidence. */
export const evidenceContract: Contract<unknown, EvidencePath> = defineContract({
  'evidence/alpha': scalar({ description: 'the prior precision alpha used in the iteration' }),
  'evidence/beta': scalar({ description: 'the noise precision beta used in the iteration' }),
  'evidence/gamma': scalar({ description: 'the effective number of well-determined weights at alpha and beta' }),
  'evidence/logEvidence': scalar({ description: 'the log marginal likelihood at alpha and beta', direction: 'higher' }),
})

/**
 * The state of {@link maximiseEvidenceAlgorithm}: the current precisions, the effective number of parameters and log
 * evidence at the previous ones, the iterations done, and whether both precisions moved by less than the tolerance.
 */
export interface EvidenceState extends BayesianPrior {
  readonly gamma: number
  readonly logEvidence: number
  readonly iterations: number
  readonly converged: boolean
}

/** Settings of {@link maximiseEvidenceAlgorithm}. */
export interface EvidenceSettings {
  /** The most iterations the function runs (default 100). */
  readonly maxIter?: number
  /** The relative tolerance on both precisions (default 1e-8). */
  readonly tol?: number
  /** The update, a value of `evidenceUpdates.space` (default MacKay's fixed point). */
  readonly update?: SpaceValue
}

/**
 * Evidence maximisation as a stepped algorithm: each step fits the posterior at the current `alpha` and `beta`, finds
 * the effective number of parameters `gamma = sum lambda_i / (alpha + lambda_i)` (with `lambda_i = beta e_i`, `e_i` the
 * eigenvalues of `X^T X`) and the log evidence, and updates the precisions by the chosen update (MacKay's fixed point
 * or EM). It is done when both precisions change by less than `tol` in relative terms. The state serialises.
 *
 * @param design - Features `[n, p]` and targets.
 * @param start - The starting `alpha` and `beta`.
 * @param settings - tol and the update (the step budget is the caller's).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a non-positive precision.
 * @concept bayesian-linear-regression
 * @example
 * ```ts
 * import { designOf, maximiseEvidenceAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const design = designOf({ x: [[1], [1], [1], [1]], y: [1.9, 2.1, 2, 2.05] }, 'example')
 * runAlgorithm(maximiseEvidenceAlgorithm(design, { alpha: 1, beta: 1 }), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function maximiseEvidenceAlgorithm(
  design: BayesianDesign,
  start: BayesianPrior,
  settings: EvidenceSettings = {},
): Algorithm<EvidenceState, EvidencePath> {
  checkPrior(start, 'maximiseEvidence')
  const { x } = design
  const { n, d: p } = x
  const gram = gramOf(x)
  const eigenvalues = toFlat(eigvalsh(fromStorage(gram, [p, p], 'float64')))
  const tol = settings.tol ?? 1e-8
  const next = evidenceUpdates.make(settings.update)
  return {
    name: 'linear/maximise-evidence',
    config: { alpha: start.alpha, beta: start.beta, tol, update: evidenceUpdates.optionOf(settings.update).name },
    schema: channels({
      metrics: { logEvidence: { description: 'the log marginal likelihood', direction: 'higher' } },
    }),
    init: () => ({
      alpha: start.alpha,
      beta: start.beta,
      gamma: 0,
      logEvidence: Number.NaN,
      iterations: 0,
      converged: false,
    }),
    step: (s, ctx) => {
      const { alpha, beta } = s
      const posterior = bayesianLinearFit(design, { alpha, beta })
      const gamma = eigenvalues.reduce((a, e) => a + (beta * Math.max(e, 0)) / (alpha + beta * Math.max(e, 0)), 0)
      const updated = next({
        n,
        p,
        mean: posterior.mean,
        covariance: posterior.covariance,
        gram,
        residual: residualSumOfSquares(design, posterior.mean),
        gamma,
      })
      const converged = Math.abs(updated.alpha - alpha) <= tol * alpha && Math.abs(updated.beta - beta) <= tol * beta
      ctx.log('evidence/alpha', alpha)
      ctx.log('evidence/beta', beta)
      ctx.log('evidence/gamma', gamma)
      ctx.log('evidence/logEvidence', posterior.logEvidence)
      return {
        alpha: updated.alpha,
        beta: updated.beta,
        gamma,
        logEvidence: posterior.logEvidence,
        iterations: s.iterations + 1,
        converged,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isNaN(s.logEvidence) ? {} : { logEvidence: s.logEvidence } }),
    serialise: (s) => stateToJson({ ...s, logEvidence: Number.isNaN(s.logEvidence) ? null : s.logEvidence }),
    deserialise: (j) => {
      const o = stateFromJson(j) as Omit<EvidenceState, 'logEvidence'> & { readonly logEvidence: number | null }
      return { ...o, logEvidence: o.logEvidence ?? Number.NaN }
    },
  }
}

/** The outcome of {@link maximiseEvidence}. */
export interface EvidenceFit extends BayesianPrior {
  /** The effective number of well-determined weights, `sum lambda_i / (alpha + lambda_i)`. */
  readonly gamma: number
  /** The posterior at the final `alpha` and `beta`. */
  readonly posterior: BayesianPosterior
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Evidence maximisation (MacKay's fixed point, Bishop 3.5.2): from the current `alpha` and `beta`, find the posterior
 * mean `m`, the eigenvalues `lambda_i = beta e_i` of `beta X^T X` and `gamma = sum lambda_i / (alpha + lambda_i)`, then
 * set `alpha = gamma / |m|^2` and `beta = (n - gamma) / |y - X m|^2`, until both change by less than `tol` in relative
 * terms. This is {@link maximiseEvidenceAlgorithm} run to the end (or `maxIter` steps).
 *
 * @param design - Features `[n, p]` and targets.
 * @param start - The starting `alpha` and `beta`.
 * @param options - The most iterations (default 100), the relative tolerance (default 1e-8) and the update.
 * @returns The maximising precisions, `gamma`, and the posterior there.
 * @example
 * ```ts
 * import { designOf, maximiseEvidence } from 'mdml-engine/linear'
 * const design = designOf({ x: [[1], [1], [1], [1]], y: [1.9, 2.1, 2, 2.05] }, 'example')
 * maximiseEvidence(design, { alpha: 1, beta: 1 }).converged // => true
 * ```
 */
export function maximiseEvidence(
  design: BayesianDesign,
  start: BayesianPrior,
  options: EvidenceSettings = {},
): EvidenceFit {
  const run = runAlgorithm(maximiseEvidenceAlgorithm(design, start, options), { maxSteps: options.maxIter ?? 100 })
  const { alpha, beta, gamma, iterations, converged } = run.state
  return { alpha, beta, gamma, posterior: bayesianLinearFit(design, { alpha, beta }), iterations, converged }
}

/** A polynomial basis of the features, scaled to `[-1, 1]` first so that high powers stay well conditioned. */
export interface PolynomialBasis {
  readonly degree: number
  /** The fitted min-max scaler to `[-1, 1]` (plain JSON). */
  readonly scaler: MinMaxScalerParams
  /** Expand features `[m, d]` into the basis `[m, p]`: a bias column, then every monomial up to the degree. */
  readonly expand: (x: TensorLike) => MatrixView
}

/**
 * The polynomial basis of the features of `x`: each feature scaled to `[-1, 1]` by the range of `x`, then expanded by
 * the library's `polynomialFeatures` (a bias column, powers and cross products up to `degree`).
 *
 * @param degree - The highest total degree.
 * @param x - The training features, whose range sets the scaling.
 * @returns The basis.
 * @example
 * ```ts
 * import { polynomialBasis } from 'mdml-engine/linear'
 * Array.from(polynomialBasis(2, [[0], [2]]).expand([[1]]).data) // => [1, 0, 0]
 * ```
 */
export function polynomialBasis(degree: number, x: TensorLike): PolynomialBasis {
  const scaler = minMaxScaler({ featureRange: [-1, 1] }).fit({ x })
  return polynomialBasisOf(degree, scaler.params)
}

/**
 * A polynomial basis from its fitted scaler: what {@link polynomialBasis} returns, rebuilt from its JSON.
 *
 * @param degree - The highest total degree.
 * @param scaler - The scaler parameters of a {@link PolynomialBasis}.
 * @returns The basis.
 * @example
 * ```ts
 * import { polynomialBasis, polynomialBasisOf } from 'mdml-engine/linear'
 * const b = polynomialBasis(3, [[0], [4]])
 * Array.from(polynomialBasisOf(3, b.scaler).expand([[4]]).data) // => [1, 1, 1, 1]
 * ```
 */
export function polynomialBasisOf(degree: number, scaler: MinMaxScalerParams): PolynomialBasis {
  if (!(Number.isInteger(degree) && degree >= 0)) throw new MdmlError('value', 'polynomialBasis: degree must be >= 0')
  const d = scaler.scale.length
  const scale = minMaxScaler({ featureRange: [-1, 1] }).revive(scaler)
  const powers = polynomialFeatures({ degree, includeBias: true }).fit({ x: [new Array<number>(d).fill(0)] })
  return {
    degree,
    scaler,
    expand: (x) => matrixView(powers.transform(scale.transform(x)), 'polynomialBasis'),
  }
}

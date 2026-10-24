import { type Json, meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { maternKernel } from '../kernels/index.ts'
import { choleskyJitter, choSolve, solveTriangular } from '../numerics/linalg/index.ts'
import { erfc, erfcx, logNdtr, ndtr, normalPdf } from '../numerics/special/index.ts'
import { lbfgsb } from '../optim/index.ts'
import { foldIn, type Key, scrambledSobol } from '../random/index.ts'
import { sample } from '../space/index.ts'
import { conditionGaussianProcess } from '../stochastic/index.ts'
import {
  add,
  asTensor,
  diagonal,
  div,
  exp,
  expandDims,
  eye,
  less,
  log,
  matmul,
  maximum,
  mul,
  neg,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { decodeInto, type Leaf, leafCoordinate, leavesOf, nativeOf } from './_leaves.ts'
import type { Sampler } from './samplers.ts'
import { lossOf, type StudyDirection } from './study.ts'

const SQRT_HALF = Math.SQRT1_2
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const MIN_NOISE = 1e-6

// ---------------------------------------------------------------------------------------------------------------
// Acquisition functions (minimisation: smaller objective values are better)

function standardised(mean: TensorLike, std: TensorLike, best: number, xi: number): Value {
  return div(sub(sub(best, mean), xi), std)
}

/**
 * Expected improvement below the best value, EI = (f* - mu - xi) Phi(z) + sigma phi(z) with
 * z = (f* - mu - xi) / sigma, for a Gaussian prediction N(mu, sigma^2) of a minimised objective.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations (positive).
 * @param best - The best (smallest) value so far, f*.
 * @param xi - Exploration margin (default 0).
 * @returns EI, differentiable in the predictions.
 * @concept bayesian-optimisation
 * @reference Jones, Schonlau and Welch (1998), Efficient global optimization of expensive black-box functions.
 * @example
 * ```ts
 * import { expectedImprovement } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(expectedImprovement([0], [1], 0)) // => [0.398942]
 * ```
 */
export function expectedImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  const z = standardised(mean, std, best, xi)
  return add(mul(mul(z, std), ndtr(z)), mul(std, normalPdf(z)))
}

/**
 * The logarithm of expected improvement, computed stably for very negative z (Ament et al. 2023, as Optuna's
 * `logei`): log(z Phi(z) + phi(z)) + log sigma, switching below z = -25 to
 * -z^2 / 2 - log sqrt(2 pi) + log(1 + sqrt(pi / 2) z erfcx(-z / sqrt 2)).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Exploration margin (default 0).
 * @returns log EI.
 * @concept bayesian-optimisation
 * @reference Ament et al. (2023), Unexpected improvements to expected improvement for Bayesian optimization.
 * @example
 * ```ts
 * import { logExpectedImprovement } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * (toFlat(logExpectedImprovement([40], [1], 0))[0] ?? Number.NaN) > -Infinity // => true
 * ```
 */
export function logExpectedImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  const z = standardised(mean, std, best, xi)
  const small = less(z, -25)
  // Each branch gets inputs from its own range, so the unused one never produces NaN gradients.
  const zl = where(small, -25, z)
  const zs = where(small, z, -26)
  const regular = log(
    add(mul(mul(zl, 0.5), erfc(mul(zl, -SQRT_HALF))), mul(exp(mul(square(zl), -0.5)), 1 / Math.sqrt(2 * Math.PI))),
  )
  const tail = add(
    sub(mul(square(zs), -0.5), LOG_SQRT_2PI),
    log(add(1, mul(mul(zs, Math.sqrt(Math.PI / 2)), erfcx(mul(zs, -SQRT_HALF))))),
  )
  return add(where(small, tail, regular), log(std))
}

/**
 * Probability of improvement below the best value, Phi((f* - mu - xi) / sigma).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Margin (default 0).
 * @returns PI.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { probabilityOfImprovement } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(probabilityOfImprovement([0], [1], 0)) // => [0.5]
 * ```
 */
export function probabilityOfImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  return ndtr(standardised(mean, std, best, xi))
}

/**
 * log probability of improvement, accurate far in the tail.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param best - The best value so far.
 * @param xi - Margin (default 0).
 * @returns log PI.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { logProbabilityOfImprovement } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(logProbabilityOfImprovement([0], [1], 0)) // => [-0.693147]
 * ```
 */
export function logProbabilityOfImprovement(mean: TensorLike, std: TensorLike, best: number, xi = 0): Value {
  return logNdtr(standardised(mean, std, best, xi))
}

/**
 * The lower confidence bound mu - beta sigma of a minimised objective (GP-UCB of Srinivas et al. for maximisation is
 * the upper bound of the negated objective); smaller is more promising.
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param beta - Width in standard deviations (default 2).
 * @returns The bounds.
 * @concept bayesian-optimisation
 * @reference Srinivas et al. (2010), Gaussian process optimization in the bandit setting.
 * @example
 * ```ts
 * import { lowerConfidenceBound } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(lowerConfidenceBound([1], [0.5], 2)) // => [0]
 * ```
 */
export function lowerConfidenceBound(mean: TensorLike, std: TensorLike, beta = 2): Value {
  return sub(mean, mul(std, beta))
}

/**
 * The upper confidence bound mu + beta sigma (GP-UCB for a maximised objective).
 *
 * @param mean - Predicted means.
 * @param std - Predicted standard deviations.
 * @param beta - Width in standard deviations (default 2).
 * @returns The bounds.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { upperConfidenceBound } from 'whitebox-engine/tune'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(upperConfidenceBound([1], [0.5], 2)) // => [2]
 * ```
 */
export function upperConfidenceBound(mean: TensorLike, std: TensorLike, beta = 2): Value {
  return add(mean, mul(std, beta))
}

// ---------------------------------------------------------------------------------------------------------------
// The Gaussian-process surrogate

/** Hyperparameters of the surrogate's kernel s Matern-5/2(r), r^2 = sum_j d_j^2 / l_j^2, plus observation noise. */
export interface GpHyperparameters {
  /** One length scale per input column (per leaf). */
  readonly lengthScales: readonly number[]
  readonly kernelScale: number
  readonly noise: number
}

/** A fitted Gaussian-process surrogate: predictions in the units of the observations. */
export interface GpSurrogate {
  readonly hyper: GpHyperparameters
  /** Columns compared by Hamming distance (options) rather than by difference, and their numbers of options. */
  readonly categorical: readonly boolean[]
  readonly sizes: readonly number[]
  readonly x: readonly (readonly number[])[]
  readonly y: readonly number[]
  /** The log marginal likelihood of the (standardised, when fitted) observations at the hyperparameters. */
  readonly logMarginalLikelihood: number
  /** The log marginal likelihood plus the log prior of the hyperparameters (the MAP objective of the fit). */
  readonly logPosterior: number
  /** Posterior mean and standard deviation of the latent function at points (rows of columns). */
  predict(points: readonly (readonly number[])[]): { readonly mean: number[]; readonly std: number[] }
  /** The same, differentiably, for points `[m, columns]` (a traced value inside an acquisition optimiser). */
  predictValue(points: Value): { readonly mean: Value; readonly std: Value }
}

/** Embedding of the columns: continuous ones as they are, categorical ones one-hot scaled by 1/sqrt 2 (Hamming). */
interface Embedding {
  readonly width: number
  readonly embed: (row: readonly number[]) => number[]
  /** [width, columns]: spreads one length scale per column over its embedded coordinates. */
  readonly spread: number[][]
  /** [width, columns]: places column values (continuous ones) into the embedding (categorical columns get 0). */
  readonly place: number[][]
}

function embedding(categorical: readonly boolean[], sizes: readonly number[]): Embedding {
  const starts: number[] = []
  let width = 0
  for (const [j, c] of categorical.entries()) {
    starts.push(width)
    width += c ? (sizes[j] ?? 1) : 1
  }
  const spread = Array.from({ length: width }, () => new Array<number>(categorical.length).fill(0))
  const place = Array.from({ length: width }, () => new Array<number>(categorical.length).fill(0))
  for (const [j, c] of categorical.entries()) {
    const s = starts[j] as number
    for (let k = 0; k < (c ? (sizes[j] ?? 1) : 1); k++) (spread[s + k] as number[])[j] = 1
    if (!c) (place[s] as number[])[j] = 1
  }
  return {
    width,
    spread,
    place,
    embed: (row) => {
      const out = new Array<number>(width).fill(0)
      for (const [j, c] of categorical.entries()) {
        const s = starts[j] as number
        const v = row[j] ?? 0
        if (!c) out[s] = v
        else if (Number.isInteger(v) && v >= 0 && v < (sizes[j] ?? 1)) out[s + v] = SQRT_HALF
      }
      return out
    },
  }
}

/** Optuna's log prior of the kernel hyperparameters (without its normalising constants). */
function logPriorOf(hyper: GpHyperparameters): number {
  let lp = 0
  for (const l of hyper.lengthScales) {
    const lambda = 1 / (l * l)
    lp -= 0.1 / lambda + 0.1 * lambda
  }
  lp += Math.log(hyper.kernelScale) - hyper.kernelScale
  lp += 0.1 * Math.log(hyper.noise) - 30 * hyper.noise
  return lp
}

function covariance(lengthScales: Value, kernelScale: Value | number): (a: Value, b: Value) => Value {
  const kernel = maternKernel({ nu: 2.5, lengthScale: lengthScales })
  return (a, b) => mul(kernel.k(a, b), kernelScale)
}

/** Options of {@link fitGpSurrogate} and {@link gpPosterior}. */
export interface GpOptions {
  /** Which columns are options (compared by Hamming distance), with `sizes` their number of options. */
  readonly categorical?: readonly boolean[]
  readonly sizes?: readonly number[]
  /** L-BFGS-B iterations for the hyperparameters (default 50). */
  readonly maxIterations?: number
}

function buildSurrogate(
  x: readonly (readonly number[])[],
  y: readonly number[],
  hyper: GpHyperparameters,
  categorical: readonly boolean[],
  sizes: readonly number[],
  standardise: boolean,
): GpSurrogate {
  const mu = standardise ? meanOf(y) : 0
  const sd = standardise ? Math.sqrt(varianceOf(y)) || 1 : 1
  const yn = y.map((v) => (v - mu) / sd)
  const emb = embedding(categorical, sizes)
  const xe = asTensor(x.map(emb.embed))
  const ell = reshape(matmul(asTensor(emb.spread), asTensor(hyper.lengthScales)), [emb.width])
  const k = covariance(ell, hyper.kernelScale)
  const n = x.length
  const kxx = add(k(xe, xe), mul(eye(n), hyper.noise))
  const factor = choleskyJitter(kxx).l
  const alpha = choSolve(factor, asTensor(yn))
  const fit = toFlat(sum(mul(asTensor(yn), alpha)))[0] ?? 0
  const logDet = toFlat(sum(log(diagonal(factor))))[0] ?? 0
  const lml = -0.5 * fit - logDet - n * LOG_SQRT_2PI
  const predictValue = (points: Value): { mean: Value; std: Value } => {
    const kxs = k(xe, points)
    const mean = reshape(matmul(expandDims(alpha, 0), kxs), [-1])
    const v = solveTriangular(factor, kxs, { lower: true })
    const variance = maximum(sub(hyper.kernelScale, sum(square(v), 0)), 1e-12)
    return { mean: add(mul(mean, sd), mu), std: mul(sqrt(variance), sd) }
  }
  return {
    hyper,
    categorical,
    sizes,
    x,
    y,
    logMarginalLikelihood: lml,
    logPosterior: lml + logPriorOf(hyper),
    predictValue,
    predict: (points) => {
      const p = predictValue(asTensor(points.map(emb.embed)))
      return { mean: toFlat(p.mean), std: toFlat(p.std) }
    },
  }
}

/**
 * The posterior of a Gaussian process with fixed hyperparameters (no standardisation): mean and standard deviation of
 * the latent function at test points, for the kernel s Matern-5/2 with one length scale per column (Hamming distance
 * on categorical columns) and noise variance `noise` (scikit-learn's `GaussianProcessRegressor` with
 * `ConstantKernel(s) * Matern(l, nu=2.5)`, `alpha=noise` and `optimizer=None`).
 *
 * @param x - Training rows.
 * @param y - Training values.
 * @param points - Test rows.
 * @param hyper - Length scales, kernel scale and noise.
 * @param options - Categorical columns and their sizes.
 * @returns Means and standard deviations.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { gpPosterior } from 'whitebox-engine/tune'
 * const p = gpPosterior([[0], [1]], [0, 1], [[0]], { lengthScales: [1], kernelScale: 1, noise: 1e-10 })
 * Math.abs(p.mean[0] ?? 1) < 1e-6 && (p.std[0] ?? 1) < 1e-4 // => true
 * ```
 */
export function gpPosterior(
  x: readonly (readonly number[])[],
  y: readonly number[],
  points: readonly (readonly number[])[],
  hyper: GpHyperparameters,
  options: GpOptions = {},
): { readonly mean: number[]; readonly std: number[] } {
  const d = x[0]?.length ?? 0
  return buildSurrogate(
    x,
    y,
    hyper,
    options.categorical ?? new Array<boolean>(d).fill(false),
    options.sizes ?? new Array<number>(d).fill(1),
    false,
  ).predict(points)
}

/**
 * Fit a Gaussian-process surrogate to observations as Optuna's `GPSampler` does: values standardised, kernel
 * s Matern-5/2 with automatic relevance determination (Hamming distance for options), noise at least 1e-6, and the
 * hyperparameters at the maximum a posteriori point of the log marginal likelihood plus Optuna's priors (on inverse
 * squared length scales lambda: -(0.1 / lambda + 0.1 lambda); Gamma(2, 1) on s; Gamma(1.1, 30) on the noise), found
 * by L-BFGS-B in log coordinates with gradients by automatic differentiation.
 *
 * @param x - Rows of columns in the unit cube (option indices for categorical columns).
 * @param y - Observed values.
 * @param options - Categorical columns, their sizes and the optimiser's iterations.
 * @returns The surrogate.
 * @throws WhiteboxError with code `shape` when x and y differ in length or x is empty.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { fitGpSurrogate } from 'whitebox-engine/tune'
 * const xs = [0, 0.25, 0.5, 0.75, 1].map((v) => [v])
 * const gp = fitGpSurrogate(xs, xs.map(([v]) => Math.sin(6 * (v ?? 0))))
 * Math.abs((gp.predict([[0.5]]).mean[0] ?? 0) - Math.sin(3)) < 0.05 // => true
 * ```
 */
export function fitGpSurrogate(
  x: readonly (readonly number[])[],
  y: readonly number[],
  options: GpOptions = {},
): GpSurrogate {
  const n = x.length
  if (n === 0 || y.length !== n) throw new WhiteboxError('shape', 'fitGpSurrogate: need rows with one value each')
  const d = x[0]?.length ?? 0
  const categorical = options.categorical ?? new Array<boolean>(d).fill(false)
  const sizes = options.sizes ?? new Array<number>(d).fill(1)
  const mu = meanOf(y)
  const sd = Math.sqrt(varianceOf(y)) || 1
  const yn = asTensor(y.map((v) => (v - mu) / sd))
  const emb = embedding(categorical, sizes)
  const xe = asTensor(x.map(emb.embed))
  const spread = asTensor(emb.spread)
  const anchor = sliceAxis(xe, 0, 0, 1)
  const objective = (theta: unknown): Value => {
    const t = theta as Value
    const inverseSquared = exp(sliceAxis(t, 0, 0, d))
    const scale = exp(sliceAxis(t, 0, d, d + 1))
    const noise = add(exp(sliceAxis(t, 0, d + 1, d + 2)), MIN_NOISE)
    const ell = reshape(matmul(spread, reshape(div(1, sqrt(inverseSquared)), [d, 1])), [emb.width])
    const k = covariance(ell, reshape(scale, []))
    const post = conditionGaussianProcess(k, xe, yn, anchor, { noise: reshape(noise, []) })
    const prior = add(
      add(neg(sum(add(div(0.1, inverseSquared), mul(inverseSquared, 0.1)))), sum(sub(log(scale), scale))),
      sum(sub(mul(log(noise), 0.1), mul(noise, 30))),
    )
    return neg(add(post.logMarginalLikelihood, prior))
  }
  // Optuna's start: unit inverse squared length scales, kernel scale and noise.
  const theta0 = [...new Array<number>(d).fill(0), 0, Math.log(1 - 0.99 * MIN_NOISE)]
  const fit = lbfgsb(objective, theta0, {
    bounds: { lower: -12, upper: 12 },
    maxIterations: options.maxIterations ?? 50,
    gradTol: 1e-3,
  })
  const theta = toFlat(fit.x as Value)
  const hyper: GpHyperparameters = {
    lengthScales: theta.slice(0, d).map((v) => 1 / Math.sqrt(Math.exp(v))),
    kernelScale: Math.exp(theta[d] ?? 0),
    noise: Math.exp(theta[d + 1] ?? 0) + MIN_NOISE,
  }
  return buildSurrogate(x, y, hyper, categorical, sizes, true)
}

// ---------------------------------------------------------------------------------------------------------------
// Acquisition maximisation and the sampler

/** An acquisition: from predicted means and standard deviations (and the best value) to scores to maximise. */
export type Acquisition = (mean: Value, std: Value, best: number) => Value

/**
 * The acquisitions {@link gpSampler} knows by name, each maximised: log expected improvement, expected improvement,
 * log probability of improvement and the negated lower confidence bound (beta 2).
 *
 * @example
 * ```ts
 * import { acquisitions } from 'whitebox-engine/tune'
 * import { asTensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(acquisitions.LCB(asTensor([2]), asTensor([0.5]), 0)) // => [-1]
 * Object.keys(acquisitions) // => ['logEI', 'EI', 'PI', 'LCB']
 * ```
 */
export const acquisitions: Readonly<Record<'logEI' | 'EI' | 'PI' | 'LCB', Acquisition>> = {
  logEI: (m, s, best) => logExpectedImprovement(m, s, best),
  EI: (m, s, best) => expectedImprovement(m, s, best),
  PI: (m, s, best) => logProbabilityOfImprovement(m, s, best),
  LCB: (m, s) => neg(lowerConfidenceBound(m, s, 2)),
}

/** Options of {@link maximizeAcquisition}. */
export interface AcquisitionSearchOptions {
  /** Quasi-random candidates scored first (default 512). */
  readonly nCandidates?: number
  /** Best candidates refined by L-BFGS-B over the continuous columns (default 3). */
  readonly nLocal?: number
  /** L-BFGS-B iterations per refinement (default 30). */
  readonly maxIterations?: number
  /** Points to add to the candidates (the best observed configurations). */
  readonly seeds?: readonly (readonly number[])[]
}

/**
 * Maximise an acquisition over the unit cube (option indices for categorical columns), as Optuna's GP sampler does:
 * score scrambled Sobol' candidates (options read through equal cells) and seed points, then refine the best few by
 * L-BFGS-B over the continuous columns with the options fixed, gradients by automatic differentiation.
 *
 * @param surrogate - The surrogate.
 * @param acquisition - The acquisition (maximised).
 * @param best - The best observed value (passed to the acquisition).
 * @param key - The key of the candidates.
 * @param options - Candidates, refinements and seeds.
 * @returns The best point and its acquisition value.
 * @concept bayesian-optimisation
 * @example
 * ```ts
 * import { acquisitions, fitGpSurrogate, maximizeAcquisition } from 'whitebox-engine/tune'
 * import { key } from 'whitebox-engine/random'
 * const xs = [[0.1], [0.4], [0.9]]
 * const gp = fitGpSurrogate(xs, [1, 0, 1])
 * const { point } = maximizeAcquisition(gp, acquisitions.EI, 0, key(0))
 * point.length // => 1
 * ```
 */
export function maximizeAcquisition(
  surrogate: GpSurrogate,
  acquisition: Acquisition,
  best: number,
  key: Key,
  options: AcquisitionSearchOptions = {},
): { readonly point: number[]; readonly value: number } {
  const d = surrogate.categorical.length
  const sizes = surrogate.sizes
  const m = options.nCandidates ?? 512
  const u = scrambledSobol(key, m, Math.max(1, d)).data
  const candidates: number[][] = Array.from({ length: m }, (_, i) =>
    surrogate.categorical.map((c, j) => {
      const v = u[i * Math.max(1, d) + j] ?? 0.5
      return c ? Math.min((sizes[j] as number) - 1, Math.floor(v * (sizes[j] as number))) : v
    }),
  )
  for (const s of options.seeds ?? []) candidates.push([...s])
  const score = (rows: readonly (readonly number[])[]): number[] => {
    const p = surrogate.predict(rows)
    return toFlat(acquisition(asTensor(p.mean), asTensor(p.std), best))
  }
  const scores = score(candidates)
  const order = scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0) || a - b)
  let bestPoint = candidates[order[0] ?? 0] as number[]
  let bestValue = scores[order[0] ?? 0] ?? Number.NEGATIVE_INFINITY
  const continuous = surrogate.categorical.flatMap((c, j) => (c ? [] : [j]))
  if (continuous.length > 0) {
    const emb = embedding(surrogate.categorical, sizes)
    const place = asTensor(emb.place)
    for (const i of order.slice(0, options.nLocal ?? 3)) {
      const start = candidates[i] as number[]
      const fixed = asTensor([emb.embed(start.map((v, j) => (surrogate.categorical[j] === true ? v : 0)))])
      const select = asTensor(continuous.map((j) => surrogate.categorical.map((_, k) => (k === j ? 1 : 0))))
      const f = (z: unknown): Value => {
        const full = matmul(reshape(z as Value, [1, continuous.length]), select)
        const row = add(fixed, matmul(full, transpose(place)))
        const p = surrogate.predictValue(row)
        return neg(sum(acquisition(p.mean, p.std, best)))
      }
      const r = lbfgsb(
        f,
        continuous.map((j) => start[j] as number),
        { bounds: { lower: 0, upper: 1 }, maxIterations: options.maxIterations ?? 30 },
      )
      if (-r.fun > bestValue) {
        const z = toFlat(r.x as Value)
        bestPoint = start.map((v, j) => (surrogate.categorical[j] === true ? v : (z[continuous.indexOf(j)] ?? v)))
        bestValue = -r.fun
      }
    }
  }
  return { point: bestPoint, value: bestValue }
}

/** Options of {@link gpSampler}. */
export interface GpSamplerOptions {
  /** Random trials before the surrogate is used (default 10). */
  readonly nStartupTrials?: number
  /** The acquisition (default `'logEI'`). */
  readonly acquisition?: keyof typeof acquisitions | Acquisition
  /** Acquisition search settings. */
  readonly search?: AcquisitionSearchOptions
  /** Hyperparameter fitting iterations (default 50). */
  readonly maxIterations?: number
}

/**
 * Bayesian optimisation with a Gaussian-process surrogate (Optuna's `GPSampler`): after `nStartupTrials` random
 * trials, fit {@link fitGpSurrogate} to the complete trials in the space's unit-cube coordinates (options by Hamming
 * distance), and propose the maximiser of the acquisition (log expected improvement by default) found by
 * {@link maximizeAcquisition}, seeded with the best trials so far. Single objective.
 * TODO(tune): multi-objective acquisition (log expected hypervolume improvement) as in Optuna's GPSampler.
 *
 * @param options - Startup trials, acquisition, search and fitting settings.
 * @returns The sampler.
 * @concept bayesian-optimisation
 * @reference Snoek, Larochelle and Adams (2012), Practical Bayesian optimization of machine learning algorithms.
 * @example
 * ```ts
 * import { gpSampler, functionObjective, tune } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const f = functionObjective((x) => ((x as number) - 0.7) ** 2)
 * (tune(f, real(0, 1), { sampler: gpSampler({ nStartupTrials: 4 }), nTrials: 10, seed: 0 }).best?.values[0] ?? 1) < 1e-3 // => true
 * ```
 */
export function gpSampler(options: GpSamplerOptions = {}): Sampler {
  const nStartup = options.nStartupTrials ?? 10
  const acq = options.acquisition ?? 'logEI'
  const acquisition = typeof acq === 'function' ? acq : acquisitions[acq]
  const config: Json = {
    nStartupTrials: nStartup,
    acquisition: typeof acq === 'function' ? 'custom' : acq,
    maxIterations: options.maxIterations ?? 50,
  }
  return {
    name: 'gp',
    config,
    multiObjective: false,
    ask: (study, key) => {
      const complete = study.trials.filter((t) => t.state === 'complete' && Number.isFinite(t.values[0] ?? Number.NaN))
      if (complete.length < nStartup) return sample(study.space, key)
      const leaves = leavesOf(study.space)
      const d = study.directions[0] as StudyDirection
      const columns = (native: readonly number[]): number[] =>
        leaves.map((l, j) => columnOf(l, native[j] ?? Number.NaN))
      const x = complete.map((t) => columns(nativeOf(study.space, t.params)))
      const y = complete.map((t) => lossOf(d, t.values[0] ?? 0))
      const categorical = leaves.map((l) => l.kind === 'categorical')
      const sizes = leaves.map((l) => (l.kind === 'categorical' ? l.n : 1))
      const surrogate = fitGpSurrogate(x, y, {
        categorical,
        sizes,
        ...(options.maxIterations === undefined ? {} : { maxIterations: options.maxIterations }),
      })
      const best = y.reduce((a, b) => Math.min(a, b))
      const order = y.map((_, i) => i).sort((a, b) => (y[a] ?? 0) - (y[b] ?? 0))
      const seeds = order.slice(0, 3).map((i) => x[i] as number[])
      const { point } = maximizeAcquisition(surrogate, acquisition, best, foldIn(key, 0), { ...options.search, seeds })
      // Continuous columns are unit-cube coordinates already; options go through their coordinate.
      return decodeInto(
        study.space,
        leaves.map((l, j) =>
          l.kind === 'categorical' ? leafCoordinate(l, Math.round(point[j] ?? 0)) : (point[j] ?? 0.5),
        ),
      )
    },
  }
}

/** A leaf's surrogate column: its unit-cube coordinate, or its option index. */
function columnOf(leaf: Leaf, native: number): number {
  if (leaf.kind === 'categorical') return Number.isNaN(native) ? -1 : native
  return leafCoordinate(leaf, native)
}

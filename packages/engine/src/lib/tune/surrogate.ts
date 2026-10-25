/**
 * The Gaussian-process surrogate of Bayesian optimisation and the importance evaluators: a Matern-5/2 kernel with one
 * length scale per parameter (automatic relevance determination, Hamming distance for options), fitted to standardised
 * observations as Optuna's `GPSampler` does, with the posterior mean and standard deviation at new points.
 */

import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { maternKernel } from '../kernels/index.ts'
import { choleskyJitter, choSolve, solveTriangular } from '../numerics/linalg/index.ts'
import { lbfgsb } from '../optim/index.ts'
import { conditionGaussianProcess } from '../stochastic/index.ts'
import {
  add,
  asTensor,
  diagonal,
  div,
  exp,
  expandDims,
  eye,
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
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'

const SQRT_HALF = Math.SQRT1_2
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
const MIN_NOISE = 1e-6

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
  /**
   * The prediction at one point, differentiably in its continuous columns: `base` gives every column (options fixed),
   * `continuous` the (possibly traced) values of the continuous columns in order, as an acquisition optimiser moves them.
   */
  predictAt(base: readonly number[], continuous: Value): { readonly mean: Value; readonly std: Value }
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
  const columns = categorical.flatMap((c, j) => (c ? [] : [j]))
  const select = asTensor(columns.map((j) => categorical.map((_, k) => (k === j ? 1 : 0))))
  const place = transpose(asTensor(emb.place))
  return {
    hyper,
    categorical,
    sizes,
    x,
    y,
    logMarginalLikelihood: lml,
    logPosterior: lml + logPriorOf(hyper),
    predictAt: (base, continuous) => {
      const fixed = asTensor([emb.embed(base.map((v, j) => (categorical[j] === true ? v : 0)))])
      const full = matmul(reshape(continuous, [1, columns.length]), select)
      return predictValue(add(fixed, matmul(full, place)))
    },
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

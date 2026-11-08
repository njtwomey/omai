/**
 * Gaussian processes as random functions: covariance functions of Brownian motion, the Brownian bridge and the
 * Ornstein-Uhlenbeck process, sampling paths at given points, conditioning on noisy observations to obtain the
 * posterior, and the Karhunen-Loeve expansion of a process into orthogonal eigenfunctions with independent Gaussian
 * coefficients (numerical, and exact for Brownian motion).
 */

import { OmaiError } from '../base/index.ts'
import { choleskyJitter, choSolve, eigh, solveTriangular } from '../numerics/linalg/index.ts'
import { type Key, normal } from '../random/index.ts'
import {
  abs,
  add,
  asFloat,
  asTensor,
  diagonal,
  div,
  exp,
  expandDims,
  eye,
  flip,
  fromStorage,
  log,
  matmul,
  maximum,
  minimum,
  mul,
  neg,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { positiveInteger, primalNumbers } from './_shared.ts'

/**
 * A covariance function k(x, y) of a random function, evaluated on two sets of points `[n, d]` and `[m, d]` to give the
 * `[n, m]` matrix of covariances. Written with tensor operations, so everything built on it differentiates in its
 * parameters. (Stationary kernels such as the squared exponential live in `kernels`; the process covariances of
 * Brownian motion, bridges, Ornstein-Uhlenbeck and fractional Brownian motion live here.)
 */
export type CovarianceFunction = (x: Value, y: Value) => Value

/** Points as `[n, d]`: a vector of times `[n]` becomes `[n, 1]`. */
function points(x: TensorLike): Value {
  const v = asFloat(asTensor(x))
  if (v.shape.length === 1) return expandDims(v, -1)
  if (v.shape.length !== 2) throw new OmaiError('shape', 'gaussian process: points must be [n] or [n, d]')
  return v
}

/** Pairwise first coordinates s_i and t_j as `[n, 1]` and `[1, m]`. */
function times(x: Value, y: Value): [Value, Value] {
  return [sliceAxis(x, 1, 0, 1), transpose(sliceAxis(y, 1, 0, 1))]
}

/**
 * The covariance matrix K_ij = k(x_i, y_j) of a covariance function on two sets of points (y defaults to x).
 *
 * @param k - The covariance function.
 * @param x - Points `[n]` or `[n, d]`.
 * @param y - Points `[m]` or `[m, d]` (default x).
 * @returns `[n, m]`.
 * @concept gaussian-processes
 * @example
 * ```ts
 * import { brownianCovariance, covarianceMatrix } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(covarianceMatrix(brownianCovariance(), [1, 2])) // => [[1, 1], [1, 2]]
 * ```
 */
export function covarianceMatrix(k: CovarianceFunction, x: TensorLike, y?: TensorLike): Value {
  const px = points(x)
  return k(px, y === undefined ? px : points(y))
}

/**
 * The covariance of Brownian motion, sigma^2 min(s, t) (times on the first coordinate).
 *
 * @param sigma - The volatility (default 1).
 * @returns The covariance function.
 * @concept brownian-motion
 * @example
 * ```ts
 * import { brownianCovariance } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(brownianCovariance(2)([[0.5]], [[3]])) // => 2
 * ```
 */
export function brownianCovariance(sigma: TensorLike = 1): CovarianceFunction {
  const [s2] = promoteFloat(mul(sigma, sigma)) as [Value]
  return (x, y) => {
    const [s, t] = times(asTensor(x), asTensor(y))
    return mul(s2, minimum(s, t))
  }
}

/**
 * The covariance of a Brownian bridge pinned at 0 and T, sigma^2 (min(s, t) - s t / T).
 *
 * @param horizon - The pinning time T (default 1).
 * @param sigma - The volatility (default 1).
 * @returns The covariance function.
 * @concept brownian-bridge
 * @example
 * ```ts
 * import { bridgeCovariance } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(bridgeCovariance()([[0.5]], [[0.5]])) // => 0.25
 * ```
 */
export function bridgeCovariance(horizon = 1, sigma: TensorLike = 1): CovarianceFunction {
  if (!(horizon > 0)) throw new OmaiError('value', 'bridgeCovariance: the horizon must be positive')
  const [s2] = promoteFloat(mul(sigma, sigma)) as [Value]
  return (x, y) => {
    const [s, t] = times(asTensor(x), asTensor(y))
    return mul(s2, sub(minimum(s, t), div(mul(s, t), horizon)))
  }
}

/**
 * The covariance of the stationary Ornstein-Uhlenbeck process, sigma^2 / (2 theta) exp(-theta |s - t|) (the Matern
 * covariance with smoothness 1/2, in time).
 *
 * @param params - Rate theta and volatility sigma.
 * @returns The covariance function.
 * @concept ornstein-uhlenbeck
 * @example
 * ```ts
 * import { ouCovariance } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(ouCovariance({ theta: 0.5, sigma: 1 })([[0]], [[0]])) // => 1
 * ```
 */
export function ouCovariance(params: { readonly theta: TensorLike; readonly sigma: TensorLike }): CovarianceFunction {
  const [theta, sigma] = promoteFloat(params.theta, params.sigma) as [Value, Value]
  const scale = div(square(sigma), mul(theta, 2))
  return (x, y) => {
    const [s, t] = times(asTensor(x), asTensor(y))
    return mul(scale, exp(neg(mul(theta, abs(sub(s, t))))))
  }
}

/** Options of {@link sampleGaussianProcess}. */
export interface GpSampleOptions {
  /** The mean function on the points `[n, d]` (default zero), or mean values `[n]`. */
  readonly mean?: ((x: Value) => Value) | TensorLike
  /** The number of sample functions (default 1). */
  readonly paths?: number
  /** The largest relative jitter tried for a stable Cholesky factor (default 1e-4; the smallest that works is used). */
  readonly maxJitter?: number
}

function meanValues(mean: GpSampleOptions['mean'], x: Value): Value {
  if (mean === undefined) return fromStorage(new Float64Array(x.shape[0] ?? 0), [x.shape[0] ?? 0], 'float64')
  return typeof mean === 'function' ? reshape(mean(x), [x.shape[0] ?? 0]) : asFloat(asTensor(mean))
}

/**
 * Sample paths of a Gaussian process at given points: f = m + L z with L L^T = K (Cholesky, with jitter for
 * semidefinite covariances such as Brownian motion's at t = 0). Reparameterised, so samples differentiate in the
 * covariance's parameters.
 *
 * @param key - The key.
 * @param k - The covariance function.
 * @param x - Points `[n]` or `[n, d]`.
 * @param options - Mean, number of paths and the jitter ceiling.
 * @returns Samples `[paths, n]`.
 * @concept gaussian-processes
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianCovariance, sampleGaussianProcess } from 'omai-engine/stochastic'
 * sampleGaussianProcess(key(0), brownianCovariance(), [0.1, 0.5, 1], { paths: 4 }).shape // => [4, 3]
 * ```
 */
export function sampleGaussianProcess(
  key: Key,
  k: CovarianceFunction,
  x: TensorLike,
  options: GpSampleOptions = {},
): Value {
  const px = points(x)
  const n = px.shape[0] ?? 0
  const paths = positiveInteger(options.paths ?? 1, 'paths', 'sampleGaussianProcess')
  const { l } = choleskyJitter(k(px, px), options.maxJitter ?? 1e-4)
  const z = normal(key, [paths, n])
  return add(matmul(z, transpose(l)), expandDims(meanValues(options.mean, px), 0))
}

/** A Gaussian process conditioned on observations. */
export interface GaussianProcessPosterior {
  /** The posterior mean at the test points `[m]`. */
  readonly mean: Value
  /** The posterior covariance `[m, m]`. */
  readonly covariance: Value
  /** The posterior standard deviations `[m]`. */
  readonly std: Value
  /** The log marginal likelihood of the observations. */
  readonly logMarginalLikelihood: Value
}

/**
 * Condition a Gaussian process on noisy observations y = f(X) + eps, eps ~ N(0, noise): the posterior at test points
 * has mean m* + K*^T (K + noise I)^{-1} (y - m) and covariance K** - K*^T (K + noise I)^{-1} K* (as
 * sklearn.gaussian_process.GaussianProcessRegressor with a fixed kernel), with the log marginal likelihood
 * -1/2 r^T (K + noise I)^{-1} r - sum log L_ii - n/2 log 2 pi. Differentiable in the observations and the covariance's
 * parameters.
 *
 * @param k - The covariance function.
 * @param xTrain - Observed points `[n]` or `[n, d]`.
 * @param yTrain - Observations `[n]`.
 * @param xTest - Test points `[m]` or `[m, d]`.
 * @param options - Observation noise variance (default 1e-10) and mean function.
 * @returns The posterior mean, covariance, standard deviations and log marginal likelihood.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { brownianCovariance, conditionGaussianProcess } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * // Brownian motion seen at t = 1: at t = 0.5 the mean halves and the variance is the bridge's, 1/4
 * const post = conditionGaussianProcess(brownianCovariance(), [1], [2], [0.5], { noise: 0 })
 * toArray(post.mean) // => [1]
 * toArray(post.covariance) // => [[0.25]]
 * ```
 */
export function conditionGaussianProcess(
  k: CovarianceFunction,
  xTrain: TensorLike,
  yTrain: TensorLike,
  xTest: TensorLike,
  options: { readonly noise?: TensorLike; readonly mean?: GpSampleOptions['mean'] } = {},
): GaussianProcessPosterior {
  const xa = points(xTrain)
  const xb = points(xTest)
  const n = xa.shape[0] ?? 0
  const y = asFloat(asTensor(yTrain))
  if (y.shape.length !== 1 || y.shape[0] !== n) throw new OmaiError('shape', 'conditionGaussianProcess: y is [n]')
  const noise = options.noise ?? 1e-10
  const kxx = add(k(xa, xa), mul(eye(n), noise))
  const factor = choleskyJitter(kxx).l
  const r = sub(y, meanValues(options.mean, xa))
  const alpha = choSolve(factor, r)
  const kxs = k(xa, xb)
  const mean = add(meanValues(options.mean, xb), reshape(matmul(swapaxes(kxs, 0, 1), expandDims(alpha, -1)), [-1]))
  const v = solveTriangular(factor, kxs, { lower: true })
  const covariance = sub(k(xb, xb), matmul(transpose(v), v))
  const logMarginalLikelihood = sub(
    sub(mul(sum(mul(r, alpha)), -0.5), sum(log(diagonal(factor)))),
    0.5 * n * Math.log(2 * Math.PI),
  )
  return { mean, covariance, std: sqrt(maximum(diagonal(covariance), 0)), logMarginalLikelihood }
}

/**
 * Sample functions from a Gaussian process posterior (or any Gaussian with a mean and covariance).
 *
 * @param key - The key.
 * @param posterior - A posterior from {@link conditionGaussianProcess}.
 * @param paths - The number of samples (default 1).
 * @returns `[paths, m]`.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianCovariance, conditionGaussianProcess, samplePosterior } from 'omai-engine/stochastic'
 * const post = conditionGaussianProcess(brownianCovariance(), [1], [2], [0.25, 0.5, 0.75], { noise: 1e-8 })
 * samplePosterior(key(0), post, 3).shape // => [3, 3]
 * ```
 */
export function samplePosterior(key: Key, posterior: GaussianProcessPosterior, paths = 1): Value {
  positiveInteger(paths, 'paths', 'samplePosterior')
  const m = posterior.mean.shape[0] ?? 0
  const { l } = choleskyJitter(posterior.covariance)
  return add(matmul(normal(key, [paths, m]), transpose(l)), expandDims(posterior.mean, 0))
}

/** A Karhunen-Loeve expansion on a grid: f(t) = sum_k sqrt(lambda_k) xi_k phi_k(t). */
export interface KarhunenLoeve {
  /** The grid. */
  readonly times: Float64Array
  /** The eigenvalues lambda_k, decreasing. */
  readonly eigenvalues: Value
  /** The eigenfunctions on the grid, one per row `[terms, n]`, orthonormal in L^2 (by the quadrature weights). */
  readonly eigenfunctions: Value
}

function trapezoidWeights(t: Float64Array): Float64Array {
  const n = t.length
  const w = new Float64Array(n)
  for (let i = 0; i + 1 < n; i++) {
    const h = (t[i + 1] ?? 0) - (t[i] ?? 0)
    w[i] = (w[i] ?? 0) + h / 2
    w[i + 1] = (w[i + 1] ?? 0) + h / 2
  }
  return w
}

/**
 * The Karhunen-Loeve expansion of a process on an interval by the Nystrom method: the eigenpairs of the covariance
 * operator, from the symmetric eigenproblem W^{1/2} K W^{1/2} with trapezoid weights W. The eigenvalues are the
 * variances of the uncorrelated coefficients, decreasing; for Brownian motion on [0, T] they approach
 * T^2 / ((k - 1/2)^2 pi^2) ({@link brownianKarhunenLoeve}).
 *
 * @param k - The covariance function.
 * @param times - The grid (increasing).
 * @param terms - The number of terms kept (default all).
 * @returns The eigenvalues and eigenfunctions on the grid.
 * @concept karhunen-loeve
 * @example
 * ```ts
 * import { brownianCovariance, karhunenLoeve, timeGrid } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * const kl = karhunenLoeve(brownianCovariance(), timeGrid(1, 400), 2)
 * Math.abs(toArray(kl.eigenvalues)[0] - 4 / Math.PI ** 2) < 1e-4 // => true
 * ```
 */
export function karhunenLoeve(k: CovarianceFunction, times: ArrayLike<number>, terms?: number): KarhunenLoeve {
  const t = Float64Array.from(times)
  const n = t.length
  if (n < 2) throw new OmaiError('value', 'karhunenLoeve: needs at least two grid points')
  const keep = terms === undefined ? n : positiveInteger(Math.min(terms, n), 'terms', 'karhunenLoeve')
  const w = trapezoidWeights(t)
  const root = fromStorage(Float64Array.from(w, Math.sqrt), [n], 'float64')
  const kk = covarianceMatrix(k, fromStorage(t, [n], 'float64'))
  const scaled = mul(mul(kk, expandDims(root, 1)), expandDims(root, 0))
  const sym = mul(add(scaled, transpose(scaled)), 0.5)
  const { eigenvalues, eigenvectors } = eigh(sym)
  // eigh is ascending: reverse, keep the largest `keep`, and undo the weighting of the eigenvectors
  const lambda = sliceAxis(flip(eigenvalues, 0), 0, 0, keep)
  const rows = sliceAxis(transpose(flip(eigenvectors, 1)), 0, 0, keep)
  const functions = div(rows, expandDims(root, 0))
  // fix the sign of each eigenfunction: its first clearly non-zero value is positive (it starts increasing)
  const f = primalNumbers(functions)
  const signs = Array.from({ length: keep }, (_, r) => {
    let big = 0
    for (let j = 0; j < n; j++) big = Math.max(big, Math.abs(f[r * n + j] ?? 0))
    for (let j = 0; j < n; j++) {
      const v = f[r * n + j] ?? 0
      if (Math.abs(v) > 1e-6 * big) return v < 0 ? -1 : 1
    }
    return 1
  })
  return {
    times: t,
    eigenvalues: lambda,
    eigenfunctions: mul(functions, fromStorage(Float64Array.from(signs), [keep, 1], 'float64')),
  }
}

/**
 * The exact Karhunen-Loeve expansion of Brownian motion on [0, T]: lambda_k = T^2 / ((k - 1/2)^2 pi^2) and
 * phi_k(t) = sqrt(2 / T) sin((k - 1/2) pi t / T), k = 1, 2, ...
 *
 * @param times - The grid in [0, T] (T is its last point).
 * @param terms - The number of terms.
 * @returns The eigenvalues and eigenfunctions on the grid.
 * @concept karhunen-loeve
 * @example
 * ```ts
 * import { brownianKarhunenLoeve } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(brownianKarhunenLoeve([0, 1], 1).eigenvalues)[0] === 4 / Math.PI ** 2 // => true
 * ```
 */
export function brownianKarhunenLoeve(times: ArrayLike<number>, terms: number): KarhunenLoeve {
  positiveInteger(terms, 'terms', 'brownianKarhunenLoeve')
  const t = Float64Array.from(times)
  const horizon = t[t.length - 1] ?? 1
  if (!(horizon > 0)) throw new OmaiError('value', 'brownianKarhunenLoeve: the grid must end after 0')
  const n = t.length
  const freq = Array.from({ length: terms }, (_, k) => ((k + 0.5) * Math.PI) / horizon)
  const phi = new Float64Array(terms * n)
  for (let k = 0; k < terms; k++) {
    for (let j = 0; j < n; j++) phi[k * n + j] = Math.sqrt(2 / horizon) * Math.sin((freq[k] ?? 0) * (t[j] ?? 0))
  }
  return {
    times: t,
    eigenvalues: fromStorage(
      Float64Array.from(freq, (w) => 1 / (w * w)),
      [terms],
      'float64',
    ),
    eigenfunctions: fromStorage(phi, [terms, n], 'float64'),
  }
}

/**
 * Sample paths from a truncated Karhunen-Loeve expansion, f = sum_k sqrt(lambda_k) xi_k phi_k with independent
 * standard normal xi_k: a smooth approximation whose error is the discarded variance sum_{k > K} lambda_k.
 *
 * @param key - The key.
 * @param kl - The expansion.
 * @param paths - The number of paths (default 1).
 * @returns `[paths, n]` on the expansion's grid.
 * @concept karhunen-loeve
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { brownianKarhunenLoeve, sampleKarhunenLoeve, timeGrid } from 'omai-engine/stochastic'
 * sampleKarhunenLoeve(key(0), brownianKarhunenLoeve(timeGrid(1, 50), 20), 5).shape // => [5, 51]
 * ```
 */
export function sampleKarhunenLoeve(key: Key, kl: KarhunenLoeve, paths = 1): Value {
  positiveInteger(paths, 'paths', 'sampleKarhunenLoeve')
  const terms = kl.eigenvalues.shape[0] ?? 0
  const xi = mul(normal(key, [paths, terms]), expandDims(sqrt(maximum(kl.eigenvalues, 0)), 0))
  return matmul(xi, kl.eigenfunctions)
}

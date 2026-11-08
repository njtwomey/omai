/**
 * The exact log marginal likelihood of Gaussian-process regression and its gradient with respect to the kernel's
 * log-parameterised hyperparameters (scikit-learn's `log_marginal_likelihood(theta, eval_gradient=True)`), through
 * `gramGradient`; and the objective the hyperparameters are fitted on, a differentiable function of theta with that
 * gradient as its derivative rule, so that any `optim` minimiser can use it.
 */

import { OmaiError } from '../base/index.ts'
import { gramGradient, type Kernel, theta as thetaOf, withTheta } from '../kernels/index.ts'
import { cholesky, choSolve } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  diagonal,
  eye,
  log,
  matmul,
  mul,
  outer,
  promoteFloat,
  reshape,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { negatedObjective } from './_search.ts'

/** Options of {@link logMarginalLikelihood}. */
export interface MarginalLikelihoodOptions {
  /** Noise variance added to the diagonal: one value, or one per observation (default 1e-10, scikit-learn's). */
  readonly alpha?: number | TensorLike
  /** Evaluate at these log-space hyperparameters instead of the kernel's own. */
  readonly theta?: TensorLike
}

/** The log marginal likelihood and its gradient in theta. */
export interface MarginalLikelihood {
  readonly value: number
  /** d value / d theta, in the order of `theta(kernel)` (empty when every hyperparameter is fixed). */
  readonly gradient: Float64Array
}

const diagonalNoise = (alpha: number | TensorLike, n: number): Value => {
  if (typeof alpha === 'number') return mul(eye(n), alpha)
  const a = promoteFloat(alpha)[0] as Value
  if (a.shape.length === 0) return mul(eye(n), a)
  return mul(eye(n), reshape(a, [1, n]))
}

/**
 * The log marginal likelihood of observations y at points X under a zero-mean Gaussian process with the kernel plus
 * independent noise, log p(y | X, theta) = -1/2 y^T K^-1 y - 1/2 log |K| - n/2 log 2 pi with K = k(X, X) + alpha I,
 * and its gradient, d/d theta_j = 1/2 tr((a a^T - K^-1) dK/d theta_j) with a = K^-1 y and dK from
 * {@link gramGradient}. A kernel matrix that is not positive definite gives minus infinity and a zero gradient, as
 * scikit-learn does.
 *
 * @param kernel - The kernel.
 * @param x - Inputs `[n, d]`.
 * @param y - Observations `[n]`.
 * @param options - The noise, and a theta to evaluate at.
 * @returns The value and the gradient.
 * @throws OmaiError `shape` when y is not `[n]`.
 * @concept gaussian-process-regression
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), equation 5.9.
 * @example
 * ```ts
 * import { logMarginalLikelihood } from 'omai-engine/gp'
 * import { rbfKernel } from 'omai-engine/kernels'
 * const { value, gradient } = logMarginalLikelihood(rbfKernel({ lengthScale: 1 }), [[0], [1]], [0, 1], { alpha: 0.1 })
 * [Math.round(value * 1e6) / 1e6, gradient.length] // => [-2.405074, 1]
 * ```
 */
export function logMarginalLikelihood(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: MarginalLikelihoodOptions = {},
): MarginalLikelihood {
  const at = options.theta === undefined ? kernel : withTheta(kernel, options.theta)
  const target = asTensor(y)
  const n = target.shape[0] ?? 0
  if (target.shape.length !== 1)
    throw new OmaiError('shape', `logMarginalLikelihood: y must be [n], got [${target.shape.join(', ')}]`)
  const { gram, gradient } = gramGradient(at, x)
  const p = gradient.shape[2] ?? 0
  const K = add(gram, diagonalNoise(options.alpha ?? 1e-10, n))
  let L: Value
  try {
    L = cholesky(K)
  } catch {
    return { value: -Infinity, gradient: new Float64Array(p) }
  }
  const yv = promoteFloat(target)[0] as Value
  const a = choSolve(L, yv)
  const value = sub(sub(mul(sum(mul(yv, a)), -0.5), sum(log(diagonal(L)))), 0.5 * n * Math.log(2 * Math.PI))
  if (p === 0) return { value: toFlat(value)[0] ?? Number.NaN, gradient: new Float64Array(0) }
  const inverse = choSolve(L, eye(n))
  const weight = sub(outer(a, a), inverse)
  const g = mul(matmul(reshape(weight, [1, n * n]), reshape(gradient, [n * n, p])), 0.5)
  return { value: toFlat(value)[0] ?? Number.NaN, gradient: Float64Array.from(toFlat(g)) }
}

/**
 * The negative log marginal likelihood as a differentiable function of theta, for a minimiser: the value is exact and
 * the derivative rule is the analytic gradient of {@link logMarginalLikelihood}, evaluated once per point however many
 * times the value and the gradient are asked for.
 *
 * @param kernel - The kernel (its structure; theta replaces its hyperparameters).
 * @param x - Inputs `[n, d]`.
 * @param y - Observations `[n]`.
 * @param options - The noise.
 * @returns A function of theta `[p]` giving a scalar.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { negativeLogMarginalLikelihood } from 'omai-engine/gp'
 * import { rbfKernel } from 'omai-engine/kernels'
 * import { grad } from 'omai-engine/autodiff'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const f = negativeLogMarginalLikelihood(rbfKernel(), [[0], [1], [2]], [0, 1, 0], { alpha: 0.1 })
 * toArray(grad(f)(tensor([0]))).length // => 1
 * ```
 */
export function negativeLogMarginalLikelihood(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: Pick<MarginalLikelihoodOptions, 'alpha'> = {},
): (t: Value) => Value {
  return negatedObjective('gp-negative-log-marginal-likelihood', (t) =>
    logMarginalLikelihood(kernel, x, y, { ...options, theta: t }),
  )
}

/**
 * The number of free hyperparameters of a kernel (the length of its theta).
 *
 * @param kernel - The kernel.
 * @returns The count.
 * @concept gaussian-process-regression
 * @example
 * ```ts
 * import { hyperparameterCount } from 'omai-engine/gp'
 * import { rationalQuadraticKernel } from 'omai-engine/kernels'
 * hyperparameterCount(rationalQuadraticKernel()) // => 2
 * ```
 */
export function hyperparameterCount(kernel: Kernel): number {
  return thetaOf(kernel).shape[0] ?? 0
}

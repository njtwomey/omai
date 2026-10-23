import { WhiteboxError } from '../../base/index.ts'
import { cholesky, inv, solve, solveTriangular } from '../../numerics/linalg/index.ts'
import { normal as normalNoise } from '../../random/index.ts'
import {
  add,
  diagonal,
  log,
  matmul,
  mul,
  promoteFloat,
  reshape,
  square,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'
import { type Distribution, defineDistribution } from '../distribution.ts'

const LOG_2PI = Math.log(2 * Math.PI)

/** How a multivariate normal's covariance is given. */
export type CovarianceParams =
  | { readonly covariance: TensorLike }
  | { readonly precision: TensorLike }
  | { readonly scaleTril: TensorLike }

/** A multivariate normal distribution, with its covariance and Gaussian conditioning and marginalisation. */
export interface MultivariateNormal extends Distribution {
  readonly loc: Value
  /** The lower Cholesky factor L of the covariance (covariance = L L^T). */
  readonly scaleTril: Value
  covariance(): Value
  /** The marginal distribution of some coordinates. */
  marginal(indices: readonly number[]): MultivariateNormal
  /** The conditional distribution of the other coordinates given observed values of `indices`. */
  condition(indices: readonly number[], values: TensorLike): MultivariateNormal
}

/**
 * The multivariate normal N(loc, Sigma) over the last axis (scipy.stats.multivariate_normal,
 * torch.distributions.MultivariateNormal), parameterised by a covariance, a precision or a lower Cholesky factor. All
 * computations go through the Cholesky factor (triangular solves and its diagonal), so densities and samples are
 * differentiable in the mean and the covariance.
 *
 * @param loc - Mean `[..., d]`.
 * @param params - `{ covariance }`, `{ precision }` or `{ scaleTril }` (`[d, d]`).
 * @returns The distribution; `rsample` is loc + L eps.
 * @concept distributions
 * @example
 * ```ts
 * import { multivariateNormal } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(multivariateNormal([0, 0], { covariance: [[1, 0], [0, 1]] }).logProb([0, 0])) + Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function multivariateNormal(loc: TensorLike, params: CovarianceParams): MultivariateNormal {
  const [mu] = promoteFloat(loc) as [Value]
  const scaleTril =
    'scaleTril' in params
      ? (promoteFloat(params.scaleTril)[0] as Value)
      : 'covariance' in params
        ? cholesky(params.covariance)
        : cholesky(inv(params.precision))
  if (scaleTril.shape.length !== 2)
    throw new WhiteboxError('shape', 'multivariateNormal: the covariance must be a single [d, d] matrix')
  const d = scaleTril.shape[0] ?? 0
  if ((mu.shape[mu.shape.length - 1] ?? -1) !== d)
    throw new WhiteboxError('shape', `multivariateNormal: loc must end in ${d}`)
  const batchShape = mu.shape.slice(0, -1)
  const logDet = sum(log(diagonal(scaleTril)))
  const cov = (): Value => matmul(scaleTril, swapaxes(scaleTril, 0, 1))
  const base = defineDistribution({
    name: 'multivariateNormal',
    params: { loc: mu, scaleTril },
    support: { kind: 'real' },
    batchShape,
    eventShape: [d],
    logProb: (x) => {
      const diff = sub(x, mu)
      const lead = diff.shape.slice(0, -1)
      const rows = lead.reduce((a, b) => a * b, 1)
      // z = L^{-1} (x - mu) for every row at once: solve against the rows as columns.
      const z = solveTriangular(scaleTril, swapaxes(reshape(diff, [rows, d]), 0, 1), { lower: true })
      const maha = reshape(sum(square(z), 0), lead)
      return sub(mul(add(maha, d * LOG_2PI), -0.5), logDet)
    },
    rsample: (key, shape) => add(mu, matmul(normalNoise(key, [...shape, ...batchShape, d]), swapaxes(scaleTril, 0, 1))),
    mean: () => mu,
    variance: () => add(mul(mu, 0), diagonal(cov())),
    mode: () => mu,
    // The same for every batch member: broadcast against the batch shape through loc.
    entropy: () => add(sum(mul(mu, 0), -1), add(logDet, 0.5 * d * (1 + LOG_2PI))),
  })
  const self: MultivariateNormal = {
    ...base,
    loc: mu,
    scaleTril,
    covariance: cov,
    marginal: (indices) => {
      const idx = [...indices]
      return multivariateNormal(take(mu, idx, -1), { covariance: take(take(cov(), idx, 0), idx, 1) })
    },
    condition: (indices, values) => {
      const observed = [...indices]
      const rest = Array.from({ length: d }, (_, i) => i).filter((i) => !observed.includes(i))
      const s = cov()
      const block = (r: number[], c: number[]): Value => take(take(s, r, 0), c, 1)
      const sab = block(rest, observed)
      const sbb = block(observed, observed)
      const residual = sub(promoteFloat(values)[0] as Value, take(mu, observed, -1))
      // mu_a + S_ab S_bb^{-1} (x_b - mu_b) and S_aa - S_ab S_bb^{-1} S_ba.
      const lead = residual.shape.slice(0, -1)
      const rows = lead.reduce((a, b) => a * b, 1)
      const columns = swapaxes(reshape(residual, [rows, observed.length]), 0, 1)
      const shift = swapaxes(matmul(sab, solve(sbb, columns)), 0, 1)
      const condMean = add(take(mu, rest, -1), reshape(shift, [...lead, rest.length]))
      const condCov = sub(block(rest, rest), matmul(sab, solve(sbb, swapaxes(sab, 0, 1))))
      return multivariateNormal(condMean, { covariance: mul(add(condCov, swapaxes(condCov, 0, 1)), 0.5) })
    },
  }
  return self
}

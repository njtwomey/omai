/**
 * The multivariate normal distribution, parameterised by a covariance, a precision or a Cholesky factor (scipy.stats
 * and torch.distributions), with Gaussian conditioning and marginalisation, plus the low-rank-plus-diagonal form whose
 * densities cost O(d r^2) and the matrix normal distribution with separate row and column covariances.
 */

import { MdmlError } from '../../base/index.ts'
import { cholesky, choSolve, inv, solve, solveTriangular } from '../../numerics/linalg/index.ts'
import { foldIn, normal as normalNoise } from '../../random/index.ts'
import {
  add,
  broadcastShapes,
  diagonal,
  div,
  expandDims,
  eye,
  log,
  matmul,
  mul,
  promoteFloat,
  reshape,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  take,
  type Value,
} from '../../tensor/index.ts'
import { broadcastBatch, type Distribution, defineDistribution } from '../distribution.ts'

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
  precision(): Value
  /** The marginal distribution of some coordinates. */
  marginal(indices: readonly number[]): MultivariateNormal
  /** The conditional distribution of the other coordinates given observed values of `indices`. */
  condition(indices: readonly number[], values: TensorLike): MultivariateNormal
}

const transposeLast = (m: Value): Value => swapaxes(m, -1, -2)

/**
 * The lower Cholesky factor of a covariance given in any of the three forms.
 *
 * @param params - `{ covariance }`, `{ precision }` or `{ scaleTril }`.
 * @returns The factor L with covariance L L^T.
 * @example
 * ```ts
 * import { scaleTrilOf } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(scaleTrilOf({ covariance: [[4, 0], [0, 9]] })) // => [[2, 0], [0, 3]]
 * ```
 */
export function scaleTrilOf(params: CovarianceParams): Value {
  if ('scaleTril' in params) return promoteFloat(params.scaleTril)[0] as Value
  if ('covariance' in params) return cholesky(params.covariance)
  return cholesky(inv(params.precision))
}

/**
 * The multivariate normal N(loc, Sigma) over the last axis (scipy.stats.multivariate_normal,
 * torch.distributions.MultivariateNormal), parameterised by a covariance, a precision or a lower Cholesky factor, each
 * possibly batched (`[..., d, d]`, broadcasting against loc's batch). All computations go through the Cholesky factor
 * (triangular solves and its diagonal), so densities and samples are differentiable in the mean and the covariance.
 *
 * @param loc - Mean `[..., d]`.
 * @param params - `{ covariance }`, `{ precision }` or `{ scaleTril }` (`[..., d, d]`).
 * @returns The distribution; `rsample` is loc + L eps.
 * @throws MdmlError with code `shape` when the shapes disagree.
 * @concept distributions
 * @example
 * ```ts
 * import { multivariateNormal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(multivariateNormal([0, 0], { covariance: [[1, 0], [0, 1]] }).logProb([0, 0])) + Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function multivariateNormal(loc: TensorLike, params: CovarianceParams): MultivariateNormal {
  const [mu] = promoteFloat(loc) as [Value]
  const scaleTril = scaleTrilOf(params)
  if (scaleTril.shape.length < 2)
    throw new MdmlError('shape', 'multivariateNormal: the covariance must be a [..., d, d] matrix')
  const d = scaleTril.shape[scaleTril.shape.length - 1] ?? 0
  if ((mu.shape[mu.shape.length - 1] ?? -1) !== d)
    throw new MdmlError('shape', `multivariateNormal: loc must end in ${d}`)
  const batchShape = broadcastShapes(mu.shape.slice(0, -1), scaleTril.shape.slice(0, -2))
  const single = scaleTril.shape.length === 2
  const logDet = sum(log(diagonal(scaleTril, 0, -2, -1)), -1)
  const cov = (): Value => matmul(scaleTril, transposeLast(scaleTril))
  // Squared Mahalanobis distances |L^{-1} (x - mu)|^2 over the batch.
  const mahalanobis = (x: Value): Value => {
    const diff = sub(x, mu)
    if (single) {
      const lead = diff.shape.slice(0, -1)
      const rows = lead.reduce((a, b) => a * b, 1)
      // Solve against every row at once, as the columns of one right-hand side.
      const z = solveTriangular(scaleTril, transposeLast(reshape(diff, [rows, d])), { lower: true })
      return reshape(sum(square(z), 0), lead)
    }
    const z = solveTriangular(scaleTril, expandDims(diff, -1), { lower: true })
    return sum(square(squeeze(z, -1)), -1)
  }
  const base = defineDistribution({
    name: 'multivariateNormal',
    params: { loc: mu, scaleTril },
    support: { kind: 'real' },
    batchShape,
    eventShape: [d],
    logProb: (x) => sub(mul(add(mahalanobis(x), d * LOG_2PI), -0.5), logDet),
    rsample: (key, shape) => {
      const eps = normalNoise(key, [...shape, ...batchShape, d])
      return add(mu, squeeze(matmul(scaleTril, expandDims(eps, -1)), -1))
    },
    mean: () => broadcastBatch(mu, [...batchShape, d]),
    variance: () => broadcastBatch(sum(square(scaleTril), -1), [...batchShape, d]),
    mode: () => broadcastBatch(mu, [...batchShape, d]),
    entropy: () => broadcastBatch(add(logDet, 0.5 * d * (1 + LOG_2PI)), batchShape),
  })
  const block = (s: Value, r: readonly number[], c: readonly number[]): Value => take(take(s, [...r], -2), [...c], -1)
  return {
    ...base,
    loc: mu,
    scaleTril,
    covariance: cov,
    precision: () => choSolve(scaleTril, broadcastBatch(eye(d), [...scaleTril.shape.slice(0, -2), d, d])),
    marginal: (indices) => {
      const idx = [...indices]
      return multivariateNormal(take(mu, idx, -1), { covariance: block(cov(), idx, idx) })
    },
    condition: (indices, values) => {
      const observed = [...indices]
      const rest = Array.from({ length: d }, (_, i) => i).filter((i) => !observed.includes(i))
      const s = cov()
      const sab = block(s, rest, observed)
      const sbb = block(s, observed, observed)
      const residual = sub(promoteFloat(values)[0] as Value, take(mu, observed, -1))
      // K = S_ab S_bb^{-1} (S_bb is symmetric); mean mu_a + K (x_b - mu_b), covariance S_aa - K S_ba.
      const gain = transposeLast(solve(sbb, transposeLast(sab)))
      const condMean = add(take(mu, rest, -1), squeeze(matmul(gain, expandDims(residual, -1)), -1))
      const condCov = sub(block(s, rest, rest), matmul(gain, transposeLast(sab)))
      return multivariateNormal(condMean, { covariance: mul(add(condCov, transposeLast(condCov)), 0.5) })
    },
  }
}

/** A multivariate normal whose covariance is low rank plus diagonal: W W^T + diag(D). */
export interface LowRankMultivariateNormal extends Distribution {
  readonly loc: Value
  readonly covFactor: Value
  readonly covDiag: Value
  covariance(): Value
}

/**
 * The multivariate normal with covariance W W^T + diag(D) for a factor W `[..., d, r]` and a positive diagonal D
 * `[..., d]` (torch.distributions.LowRankMultivariateNormal): densities cost O(d r^2) through the Woodbury identity and
 * the matrix determinant lemma, never forming the d x d covariance.
 *
 * @param loc - Mean `[..., d]`.
 * @param covFactor - Factor W `[..., d, r]`.
 * @param covDiag - Diagonal D `[..., d]` (positive).
 * @returns The distribution; `rsample` is loc + W eps_r + sqrt(D) eps_d.
 * @example
 * ```ts
 * import { lowRankMultivariateNormal } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lowRankMultivariateNormal([0, 0], [[1], [2]], [1, 1]).variance()) // => [2, 5]
 * ```
 */
export function lowRankMultivariateNormal(
  loc: TensorLike,
  covFactor: TensorLike,
  covDiag: TensorLike,
): LowRankMultivariateNormal {
  const [mu, w, dg] = promoteFloat(loc, covFactor, covDiag) as [Value, Value, Value]
  if (w.shape.length < 2) throw new MdmlError('shape', 'lowRankMultivariateNormal: the factor must be [..., d, r]')
  const d = w.shape[w.shape.length - 2] ?? 0
  const r = w.shape[w.shape.length - 1] ?? 0
  if ((mu.shape[mu.shape.length - 1] ?? -1) !== d || (dg.shape[dg.shape.length - 1] ?? -1) !== d)
    throw new MdmlError('shape', `lowRankMultivariateNormal: loc and the diagonal must end in ${d}`)
  const batchShape = broadcastShapes(
    broadcastShapes(mu.shape.slice(0, -1), w.shape.slice(0, -2)),
    dg.shape.slice(0, -1),
  )
  // Capacitance C = I + W^T D^{-1} W and its Cholesky factor.
  const wOverD = div(w, expandDims(dg, -1))
  const capacitance = add(eye(r), matmul(transposeLast(w), wOverD))
  const lc = cholesky(capacitance)
  const logDet = add(mul(sum(log(diagonal(lc, 0, -2, -1)), -1), 2), sum(log(dg), -1))
  const cov = (): Value => add(matmul(w, transposeLast(w)), mul(eye(d), expandDims(dg, -1)))
  const base = defineDistribution({
    name: 'lowRankMultivariateNormal',
    params: { loc: mu, covFactor: w, covDiag: dg },
    support: { kind: 'real' },
    batchShape,
    eventShape: [d],
    logProb: (x) => {
      const diff = sub(x, mu)
      // (x - mu)^T Sigma^{-1} (x - mu) = |diff|^2_{D^{-1}} - |L_C^{-1} W^T D^{-1} diff|^2.
      const first = sum(div(square(diff), dg), -1)
      const v = matmul(transposeLast(wOverD), expandDims(diff, -1))
      const z = solveTriangular(lc, v, { lower: true })
      const maha = sub(first, sum(square(squeeze(z, -1)), -1))
      return mul(add(add(maha, logDet), d * LOG_2PI), -0.5)
    },
    rsample: (key, shape) => {
      const full = [...shape, ...batchShape]
      const e1 = normalNoise(foldIn(key, 0), [...full, r])
      const e2 = normalNoise(foldIn(key, 1), [...full, d])
      return add(add(mu, squeeze(matmul(w, expandDims(e1, -1)), -1)), mul(sqrt(dg), e2))
    },
    mean: () => broadcastBatch(mu, [...batchShape, d]),
    variance: () => broadcastBatch(add(sum(square(w), -1), dg), [...batchShape, d]),
    mode: () => broadcastBatch(mu, [...batchShape, d]),
    entropy: () => broadcastBatch(mul(add(logDet, d * (1 + LOG_2PI)), 0.5), batchShape),
  })
  return { ...base, loc: mu, covFactor: w, covDiag: dg, covariance: cov }
}

/**
 * The matrix normal distribution MN(M, U, V) on n x p matrices (scipy.stats.matrix_normal): vec(X) ~
 * N(vec(M), V kron U), with row covariance U and column covariance V. Computed through Cholesky factors of U and V.
 *
 * @param mean - Mean M `[..., n, p]`.
 * @param rowCov - Row covariance U `[n, n]`.
 * @param colCov - Column covariance V `[p, p]`.
 * @returns The distribution with event shape `[n, p]`; `rsample` is M + L_U Z L_V^T.
 * @example
 * ```ts
 * import { matrixNormal } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(matrixNormal([[0, 0]], [[2]], [[1, 0], [0, 3]]).variance()) // => [[2, 6]]
 * ```
 */
export function matrixNormal(mean: TensorLike, rowCov: TensorLike, colCov: TensorLike): Distribution {
  const [m, u, v] = promoteFloat(mean, rowCov, colCov) as [Value, Value, Value]
  if (m.shape.length < 2) throw new MdmlError('shape', 'matrixNormal: the mean must be [..., n, p]')
  const n = m.shape[m.shape.length - 2] ?? 0
  const p = m.shape[m.shape.length - 1] ?? 0
  const lu = cholesky(u)
  const lv = cholesky(v)
  const batchShape = m.shape.slice(0, -2)
  const logDetU = mul(sum(log(diagonal(lu, 0, -2, -1)), -1), 2)
  const logDetV = mul(sum(log(diagonal(lv, 0, -2, -1)), -1), 2)
  const logNorm = add(add(mul(logDetU, p), mul(logDetV, n)), n * p * LOG_2PI)
  return defineDistribution({
    name: 'matrixNormal',
    params: { mean: m, rowCov: u, colCov: v },
    support: { kind: 'real' },
    batchShape,
    eventShape: [n, p],
    logProb: (x) => {
      // A = L_U^{-1} (X - M) L_V^{-T}; the exponent is |A|_F^2.
      const y = solveTriangular(lu, sub(x, m), { lower: true })
      const a = solveTriangular(lv, transposeLast(y), { lower: true })
      return mul(add(sum(sum(square(a), -1), -1), logNorm), -0.5)
    },
    rsample: (key, shape) => {
      const z = normalNoise(key, [...shape, ...batchShape, n, p])
      return add(m, matmul(matmul(lu, z), transposeLast(lv)))
    },
    mean: () => m,
    variance: () =>
      broadcastBatch(mul(expandDims(diagonal(u, 0, -2, -1), -1), expandDims(diagonal(v, 0, -2, -1), -2)), [
        ...batchShape,
        n,
        p,
      ]),
    mode: () => m,
    entropy: () => broadcastBatch(mul(add(logNorm, n * p), 0.5), batchShape),
  })
}

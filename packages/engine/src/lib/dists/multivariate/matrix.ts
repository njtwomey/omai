/**
 * Distributions over matrices: the Wishart and inverse Wishart distributions on positive-definite matrices (the
 * conjugate priors of a normal precision and covariance, as scipy.stats), and the LKJ distribution over correlation
 * matrices and their Cholesky factors, with the log multivariate gamma function that normalises them.
 */

import { WhiteboxError } from '../../base/index.ts'
import { cholesky, inv, solveTriangular } from '../../numerics/linalg/index.ts'
import { digamma, lgamma } from '../../numerics/special/index.ts'
import { foldIn, normal as normalNoise } from '../../random/index.ts'
import {
  add,
  arange,
  broadcastShapes,
  concat,
  diagonal,
  div,
  expandDims,
  eye,
  greater,
  log,
  matmul,
  mul,
  norm,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  tril,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { gammaSample } from '../_sampling.ts'
import { beta } from '../continuous/bounded.ts'
import {
  broadcastBatch,
  type Distribution,
  defineDistribution,
  inSupport,
  unavailable,
  validateParams,
} from '../distribution.ts'
import { type CovarianceParams, scaleTrilOf } from './normal.ts'

const transposeLast = (m: Value): Value => swapaxes(m, -1, -2)
const logDetFromTril = (l: Value): Value => mul(sum(log(diagonal(l, 0, -2, -1)), -1), 2)

/**
 * The log multivariate gamma function log Gamma_p(a) = p (p - 1) / 4 log pi + sum_{j=1}^p log Gamma(a + (1 - j) / 2)
 * (scipy.special.multigammaln), the normaliser of Wishart densities.
 *
 * @param a - Argument (above (p - 1) / 2).
 * @param p - Dimension.
 * @returns log Gamma_p(a), differentiable in a.
 * @example
 * ```ts
 * import { multigammaln } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(multigammaln(3, 1)) - Math.log(2)) < 1e-15 // => true
 * ```
 */
export function multigammaln(a: TensorLike, p: number): Value {
  const [x] = promoteFloat(a) as [Value]
  let total: Value = add(mul(x, 0), ((p * (p - 1)) / 4) * Math.log(Math.PI))
  for (let j = 1; j <= p; j++) total = add(total, lgamma(add(x, (1 - j) / 2)))
  return total
}

/** The multivariate digamma psi_p(a) = sum_{j=1}^p psi(a + (1 - j) / 2). */
function multidigamma(a: Value, p: number): Value {
  let total: Value = mul(a, 0)
  for (let j = 1; j <= p; j++) total = add(total, digamma(add(a, (1 - j) / 2)))
  return total
}

/**
 * Bartlett factors: lower-triangular A with A_ii = sqrt(chi-square(df - i)) (0-based i) and standard normals below the
 * diagonal, `[...shape, ...batch, p, p]`, differentiable in df through the implicitly reparameterised gamma.
 */
function bartlett(
  key: Parameters<typeof foldIn>[0],
  df: Value,
  p: number,
  shape: readonly number[],
  batch: readonly number[],
): Value {
  const offsets = arange(p)
  const halfDf = mul(sub(expandDims(broadcastBatch(df, batch), -1), offsets), 0.5)
  const chi = mul(gammaSample(foldIn(key, 0), halfDf, shape), 2)
  const below = tril(normalNoise(foldIn(key, 1), [...shape, ...batch, p, p]), -1)
  return add(below, mul(expandDims(sqrt(chi), -1), eye(p)))
}

/**
 * The Wishart distribution W_p(df, Sigma) on p x p positive-definite matrices (scipy.stats.wishart,
 * torch.distributions.Wishart): the scatter matrix of df independent N(0, Sigma) vectors, and the conjugate prior of
 * a normal precision.
 *
 * @param df - Degrees of freedom (above p - 1).
 * @param params - The scale Sigma as `{ covariance }`, `{ precision }` or `{ scaleTril }` (`[..., p, p]`).
 * @returns The distribution with event shape `[p, p]`; `rsample` is the Bartlett decomposition L A A^T L^T, with
 *   chi-square diagonals by implicit reparameterisation, so draws differentiate in df and the scale.
 * @example
 * ```ts
 * import { wishart } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(wishart(5, { covariance: [[1, 0], [0, 4]] }).mean()) // => [[5, 0], [0, 20]]
 * ```
 */
export function wishart(df: TensorLike, params: CovarianceParams): Distribution {
  const [nu] = promoteFloat(df) as [Value]
  const l = scaleTrilOf(params)
  const p = l.shape[l.shape.length - 1] ?? 0
  validateParams('wishart', { df: [sub(nu, p - 1), 'positive'] })
  const batchShape = broadcastShapes(nu.shape, l.shape.slice(0, -2))
  const sigma = (): Value => matmul(l, transposeLast(l))
  const logDetSigma = logDetFromTril(l)
  const halfNu = mul(nu, 0.5)
  const lift = (v: Value): Value => expandDims(expandDims(v, -1), -1)
  return defineDistribution({
    name: 'wishart',
    params: { df: nu, scaleTril: l },
    support: { kind: 'positiveDefinite' },
    batchShape,
    eventShape: [p, p],
    logProb: (x) => {
      const lx = cholesky(x)
      // tr(Sigma^{-1} X) = |L^{-1} L_X|_F^2.
      const m = solveTriangular(l, lx, { lower: true })
      const trace = sum(sum(square(m), -1), -1)
      const lp = sub(
        sub(
          sub(mul(sub(nu, p + 1), mul(logDetFromTril(lx), 0.5)), mul(trace, 0.5)),
          add(mul(nu, (p / 2) * Math.LN2), mul(halfNu, logDetSigma)),
        ),
        multigammaln(halfNu, p),
      )
      return where(inSupport({ kind: 'positiveDefinite' }, x), lp, Number.NEGATIVE_INFINITY)
    },
    rsample: (key, shape) => {
      const a = bartlett(key, nu, p, shape, batchShape)
      const la = matmul(l, a)
      return matmul(la, transposeLast(la))
    },
    mean: () => mul(lift(nu), sigma()),
    variance: () => {
      const s = sigma()
      const d = diagonal(s, 0, -2, -1)
      return mul(lift(nu), add(square(s), mul(expandDims(d, -1), expandDims(d, -2))))
    },
    mode: () => mul(lift(sub(nu, p + 1)), sigma()),
    entropy: () =>
      add(
        sub(
          add(add(mul(logDetSigma, (p + 1) / 2), ((p * (p + 1)) / 2) * Math.LN2), multigammaln(halfNu, p)),
          mul(mul(sub(nu, p + 1), 0.5), multidigamma(halfNu, p)),
        ),
        mul(nu, p / 2),
      ),
  })
}

/**
 * The inverse Wishart distribution W_p^{-1}(df, Psi) (scipy.stats.invwishart): X^{-1} ~ W_p(df, Psi^{-1}), the
 * conjugate prior of a normal covariance.
 *
 * @param df - Degrees of freedom (above p - 1).
 * @param scale - The scale matrix Psi `[..., p, p]` (positive definite).
 * @returns The distribution with event shape `[p, p]`; `rsample` inverts a reparameterised Wishart draw. The entropy
 *   is the Wishart's minus (p + 1) E[log |W|].
 * @example
 * ```ts
 * import { inverseWishart } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(inverseWishart(5, [[2, 0], [0, 4]]).mean()) // => [[1, 0], [0, 2]]
 * ```
 */
export function inverseWishart(df: TensorLike, scale: TensorLike): Distribution {
  const [nu, psi] = promoteFloat(df, scale) as [Value, Value]
  const lp = cholesky(psi)
  const p = psi.shape[psi.shape.length - 1] ?? 0
  validateParams('inverseWishart', { df: [sub(nu, p - 1), 'positive'] })
  const batchShape = broadcastShapes(nu.shape, psi.shape.slice(0, -2))
  const logDetPsi = logDetFromTril(lp)
  const halfNu = mul(nu, 0.5)
  const lift = (v: Value): Value => expandDims(expandDims(v, -1), -1)
  // X = W^{-1} for W ~ Wishart(df, Psi^{-1}).
  const inner = wishart(nu, { precision: psi })
  return defineDistribution({
    name: 'inverseWishart',
    params: { df: nu, scale: psi },
    support: { kind: 'positiveDefinite' },
    batchShape,
    eventShape: [p, p],
    logProb: (x) => {
      const lx = cholesky(x)
      // tr(Psi X^{-1}) = |L_X^{-1} L_Psi|_F^2.
      const m = solveTriangular(lx, lp, { lower: true })
      const trace = sum(sum(square(m), -1), -1)
      const value = sub(
        sub(
          sub(mul(halfNu, logDetPsi), mul(nu, (p / 2) * Math.LN2)),
          add(multigammaln(halfNu, p), mul(mul(add(nu, p + 1), 0.5), logDetFromTril(lx))),
        ),
        mul(trace, 0.5),
      )
      return where(inSupport({ kind: 'positiveDefinite' }, x), value, Number.NEGATIVE_INFINITY)
    },
    rsample: (key, shape) => inv(inner.rsample(key, shape)),
    mean: () => div(psi, lift(sub(nu, p + 1))),
    variance: () => {
      const d = diagonal(psi, 0, -2, -1)
      const num = add(
        mul(lift(sub(nu, p - 1)), square(psi)),
        mul(lift(sub(nu, p + 1)), mul(expandDims(d, -1), expandDims(d, -2))),
      )
      return div(num, lift(mul(mul(sub(nu, p), square(sub(nu, p + 1))), sub(nu, p + 3))))
    },
    mode: () => div(psi, lift(add(nu, p + 1))),
    entropy: () => {
      // E[log |W|] = psi_p(df / 2) + p log 2 - log |Psi| for W ~ Wishart(df, Psi^{-1}).
      const expectedLogDet = sub(add(multidigamma(halfNu, p), p * Math.LN2), logDetPsi)
      return sub(inner.entropy(), mul(expectedLogDet, p + 1))
    },
  })
}

/** log of the LKJ normaliser for dimension d and concentration eta (shared by the matrix and Cholesky forms). */
function lkjLogNormaliser(eta: Value, d: number): Value {
  const dm1 = d - 1
  const alpha = add(eta, 0.5 * dm1)
  return add(sub(multigammaln(sub(alpha, 0.5), dm1), mul(lgamma(alpha), dm1)), 0.5 * dm1 * Math.log(Math.PI))
}

/** Onion-method draws of Cholesky factors of LKJ correlation matrices, `[...shape, ...batch, d, d]`. */
function onion(
  key: Parameters<typeof foldIn>[0],
  eta: Value,
  d: number,
  shape: readonly number[],
  batch: readonly number[],
): Value {
  // Row i (i >= 1) has squared length y_i ~ Beta(i / 2, eta + (d - 2) / 2 - (i - 1) / 2) off the diagonal.
  const offset = concat([zeros([1]), arange(d - 1)], -1)
  const marginal = add(broadcastBatch(eta, batch), 0.5 * (d - 2))
  const a = add(offset, 0.5)
  const b = sub(expandDims(marginal, -1), mul(offset, 0.5))
  const y = expandDims(beta(a, b).rsample(foldIn(key, 0), shape), -1)
  const u = tril(normalNoise(foldIn(key, 1), [...shape, ...batch, d, d]), -1)
  // Unit rows (row 0 is empty and stays zero).
  const rowNorm = norm(u, 2, -1, true)
  const unit = div(u, where(greater(rowNorm, 0), rowNorm, 1))
  const w = mul(sqrt(y), unit)
  const diag = sqrt(sub(1, sum(square(w), -1)))
  return add(w, mul(expandDims(diag, -1), eye(d)))
}

/**
 * The LKJ distribution over Cholesky factors of d x d correlation matrices (Lewandowski, Kurowicka and Joe, 2009;
 * torch.distributions.LKJCholesky): log p(L) = sum_{i>=1} (d - i - 1 + 2 (eta - 1)) log L_ii - log c(eta), uniform
 * over correlation matrices at eta = 1 and concentrated at the identity for large eta.
 *
 * @param dim - The dimension d (at least 2).
 * @param concentration - eta (positive).
 * @returns The distribution with event shape `[d, d]`; `rsample` is the onion method with reparameterised beta
 *   draws, so samples differentiate in eta. Its mode is the identity (eta >= 1); other moments throw.
 * @example
 * ```ts
 * import { lkjCholesky } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * // Uniform over 2 x 2 correlations: the correlation r is uniform on (-1, 1), so p(L) = 1 / 2 at L_11 = 1.
 * Math.abs(item(lkjCholesky(2, 1).logProb([[1, 0], [0, 1]])) - Math.log(0.5)) < 1e-14 // => true
 * ```
 */
export function lkjCholesky(dim: number, concentration: TensorLike = 1): Distribution {
  if (!(Number.isInteger(dim) && dim >= 2)) throw new WhiteboxError('value', 'lkjCholesky: dim must be at least 2')
  const [eta] = promoteFloat(concentration) as [Value]
  validateParams('lkjCholesky', { concentration: [eta, 'positive'] })
  const batchShape = [...eta.shape]
  const logNorm = lkjLogNormaliser(eta, dim)
  const order = sub(add(expandDims(mul(sub(eta, 1), 2), -1), dim), arange(2, dim + 1))
  return defineDistribution({
    name: 'lkjCholesky',
    params: { concentration: eta },
    support: { kind: 'correlationCholesky' },
    batchShape,
    eventShape: [dim, dim],
    logProb: (l) => {
      const diag = sliceAxis(diagonal(l, 0, -2, -1), -1, 1, dim)
      return sub(sum(mul(order, log(diag)), -1), logNorm)
    },
    rsample: (key, shape) => onion(key, eta, dim, shape, batchShape),
    mean: () => unavailable('lkjCholesky', 'mean (use lkj for moments of the correlation matrix)'),
    variance: () => unavailable('lkjCholesky', 'variance (use lkj for moments of the correlation matrix)'),
    mode: () => broadcastBatch(eye(dim), [...batchShape, dim, dim]),
    entropy: () => unavailable('lkjCholesky', 'entropy (no closed form)'),
  })
}

/**
 * The LKJ distribution over d x d correlation matrices: p(C) proportional to det(C)^(eta - 1).
 *
 * @param dim - The dimension d (at least 2).
 * @param concentration - eta (positive).
 * @returns The distribution with event shape `[d, d]`: mean and mode the identity, off-diagonal variance
 *   1 / (2 eta + d - 1); `rsample` is L L^T for an onion-method Cholesky draw.
 * @example
 * ```ts
 * import { lkj } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(lkj(2, 1).variance()) // => [[0, 0.3333333333333333], [0.3333333333333333, 0]]
 * ```
 */
export function lkj(dim: number, concentration: TensorLike = 1): Distribution {
  const chol = lkjCholesky(dim, concentration)
  const eta = chol.params['concentration'] as Value
  const batchShape = [...eta.shape]
  const logNorm = lkjLogNormaliser(eta, dim)
  const lift = (v: Value): Value => expandDims(expandDims(v, -1), -1)
  const offDiagonal = sub(1, eye(dim))
  return defineDistribution({
    name: 'lkj',
    params: { concentration: eta },
    support: { kind: 'correlation' },
    batchShape,
    eventShape: [dim, dim],
    logProb: (c) => {
      const l = cholesky(c)
      const value = sub(mul(sub(eta, 1), logDetFromTril(l)), logNorm)
      return where(inSupport({ kind: 'correlation' }, c), value, Number.NEGATIVE_INFINITY)
    },
    rsample: (key, shape) => {
      const l = chol.rsample(key, shape)
      return matmul(l, transposeLast(l))
    },
    mean: () => broadcastBatch(eye(dim), [...batchShape, dim, dim]),
    variance: () => mul(div(1, lift(add(mul(eta, 2), dim - 1))), offDiagonal),
    mode: () => broadcastBatch(eye(dim), [...batchShape, dim, dim]),
    entropy: () => unavailable('lkj', 'entropy (no closed form)'),
  })
}

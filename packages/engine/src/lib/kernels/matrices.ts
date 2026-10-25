/**
 * Statistics and methods on kernel matrices: centring in feature space, kernel and kernel-target alignment (and centred
 * kernel alignment), the Hilbert-Schmidt independence criterion (HSIC), the squared maximum mean discrepancy (MMD), the
 * effective dimension of a regularised kernel matrix, kernel PCA, and scikit-learn's `pairwise_kernels` by metric name.
 */

import { MdmlError } from '../base/index.ts'
import { eigh, solve } from '../numerics/linalg/index.ts'
import {
  add,
  div,
  eye,
  fromStorage,
  lessEqual,
  matmul,
  mean,
  mul,
  neg,
  outer,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  take,
  trace,
  type Value,
  where,
} from '../tensor/index.ts'
import { cosineSimilarityKernel, linearKernel, polynomialKernel, sigmoidKernel } from './dot-product.ts'
import type { Kernel } from './kernel.ts'
import { laplacianKernel, rbfKernel } from './stationary.ts'
import { chi2Distance, chi2Kernel } from './structured.ts'

function matrix(k: TensorLike, name: string): Value {
  const [v] = promoteFloat(k) as [Value]
  if (v.shape.length !== 2) throw new MdmlError('shape', `${name}: expected a matrix, got [${v.shape.join(', ')}]`)
  return v
}

function square2(k: TensorLike, name: string): Value {
  const v = matrix(k, name)
  if (v.shape[0] !== v.shape[1]) throw new MdmlError('shape', `${name}: expected a square matrix`)
  return v
}

/**
 * Centre a kernel matrix in feature space (scikit-learn KernelCenterer): K_c = H K H with H = I - 1/n, the Gram matrix
 * of the features minus their mean. With a reference (training) Gram matrix, a cross-kernel `[m, n]` between new points
 * and the reference points is centred with the training mean.
 *
 * @param k - A Gram matrix `[n, n]`, or a cross-kernel `[m, n]` when `reference` is given.
 * @param reference - The training Gram matrix `[n, n]`.
 * @returns The centred matrix, differentiable.
 * @concept kernel-matrices
 * @example
 * ```ts
 * import { centerKernel } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(centerKernel([[1, 0], [0, 1]])) // => [[0.5, -0.5], [-0.5, 0.5]]
 * ```
 */
export function centerKernel(k: TensorLike, reference?: TensorLike): Value {
  if (reference === undefined) {
    const v = square2(k, 'centerKernel')
    return add(sub(sub(v, mean(v, 0, true)), mean(v, 1, true)), mean(v))
  }
  const fit = square2(reference, 'centerKernel')
  const v = matrix(k, 'centerKernel')
  if (v.shape[1] !== fit.shape[0]) throw new MdmlError('shape', 'centerKernel: the cross-kernel must be [m, n]')
  return add(sub(sub(v, mean(fit, 0, true)), mean(v, 1, true)), mean(fit))
}

/** Options of {@link kernelAlignment}. */
export interface AlignmentOptions {
  /** Centre both matrices first (centred kernel alignment, CKA; default false). */
  readonly centred?: boolean
}

/**
 * The alignment of two kernel matrices, <K_1, K_2>_F / (|K_1|_F |K_2|_F), a cosine between kernels (Cristianini et
 * al.); centred, it is centred kernel alignment (Cortes et al.), equal to linear CKA of the feature maps.
 *
 * @param k1 - A Gram matrix `[n, n]`.
 * @param k2 - A Gram matrix `[n, n]`.
 * @param options - Whether to centre.
 * @returns The scalar alignment in [-1, 1], differentiable.
 * @concept kernel-matrices
 * @reference N. Cristianini et al., On kernel-target alignment (NeurIPS 2001); C. Cortes, M. Mohri and A. Rostamizadeh,
 *   Algorithms for learning kernels based on centered alignment (JMLR 13, 2012).
 * @example
 * ```ts
 * import { kernelAlignment } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(kernelAlignment([[1, 0], [0, 1]], [[2, 0], [0, 2]])) // => 1
 * ```
 */
export function kernelAlignment(k1: TensorLike, k2: TensorLike, options: AlignmentOptions = {}): Value {
  const a = options.centred === true ? centerKernel(k1) : square2(k1, 'kernelAlignment')
  const b = options.centred === true ? centerKernel(k2) : square2(k2, 'kernelAlignment')
  if (a.shape[0] !== b.shape[0]) throw new MdmlError('shape', 'kernelAlignment: the matrices differ in size')
  return div(sum(mul(a, b)), sqrt(mul(sum(square(a)), sum(square(b)))))
}

/**
 * Kernel-target alignment: the alignment of K with the ideal kernel y y^T of the targets (labels in {-1, +1} for
 * classification).
 *
 * @param k - A Gram matrix `[n, n]`.
 * @param y - Targets `[n]`.
 * @param options - Whether to centre.
 * @returns The scalar alignment.
 * @concept kernel-matrices
 * @example
 * ```ts
 * import { targetAlignment } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(targetAlignment([[1, -1], [-1, 1]], [1, -1])) // => 1
 * ```
 */
export function targetAlignment(k: TensorLike, y: TensorLike, options: AlignmentOptions = {}): Value {
  const [t] = promoteFloat(y) as [Value]
  return kernelAlignment(k, outer(t, t), options)
}

/** Options of {@link hsic}. */
export interface HsicOptions {
  /** The biased V-statistic tr(K H L H) / (n - 1)^2 (default) or the unbiased U-statistic of Song et al. (n >= 4). */
  readonly estimator?: 'biased' | 'unbiased'
  /** Divide by sqrt(HSIC(K, K) HSIC(L, L)) (normalised HSIC; the biased one is CKA). Default false. */
  readonly normalise?: boolean
}

function offDiagonal(k: Value): Value {
  return where(eye(k.shape[0] ?? 0), 0, k)
}

function hsicValue(k: Value, l: Value, estimator: 'biased' | 'unbiased'): Value {
  const n = k.shape[0] ?? 0
  if (estimator === 'biased') return div(sum(mul(centerKernel(k), centerKernel(l))), (n - 1) * (n - 1))
  if (n < 4) throw new MdmlError('value', 'hsic: the unbiased estimator needs at least 4 points')
  const kt = offDiagonal(k)
  const lt = offDiagonal(l)
  const first = sum(mul(kt, lt))
  const second = div(mul(sum(kt), sum(lt)), (n - 1) * (n - 2))
  const third = mul(sum(mul(sum(kt, 1), sum(lt, 0))), 2 / (n - 2))
  return div(sub(add(first, second), third), n * (n - 3))
}

/**
 * The Hilbert-Schmidt independence criterion between two samples from their Gram matrices: the squared norm of the
 * cross-covariance operator, zero exactly when the variables are independent (for characteristic kernels).
 *
 * @param k - The Gram matrix of the first variable `[n, n]`.
 * @param l - The Gram matrix of the second variable `[n, n]` (same sample order).
 * @param options - Estimator and normalisation.
 * @returns The scalar statistic, differentiable in the Gram matrices.
 * @throws MdmlError `shape` for matrices of different sizes; `value` for the unbiased estimator with n < 4.
 * @concept kernel-independence
 * @reference A. Gretton et al., Measuring statistical dependence with Hilbert-Schmidt norms (ALT 2005); L. Song et al.,
 *   Feature selection via dependence maximization (JMLR 13, 2012).
 * @example
 * ```ts
 * import { hsic } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * const k = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
 * Math.round(item(hsic(k, k)) * 1e12) / 1e12 // => 0.444444444444
 * ```
 */
export function hsic(k: TensorLike, l: TensorLike, options: HsicOptions = {}): Value {
  const a = square2(k, 'hsic')
  const b = square2(l, 'hsic')
  if (a.shape[0] !== b.shape[0]) throw new MdmlError('shape', 'hsic: the Gram matrices differ in size')
  const estimator = options.estimator ?? 'biased'
  const value = hsicValue(a, b, estimator)
  if (options.normalise !== true) return value
  return div(value, sqrt(mul(hsicValue(a, a, estimator), hsicValue(b, b, estimator))))
}

/** Options of {@link mmdSquared}. */
export interface MmdStatisticOptions {
  /**
   * The biased V-statistic (default), the unbiased U-statistic (diagonals left out), or the linear-time statistic of
   * Gretton et al. over consecutive pairs (equal sample sizes).
   */
  readonly estimator?: 'biased' | 'unbiased' | 'linear'
}

/**
 * The squared maximum mean discrepancy between two samples from their kernel matrices,
 * MMD^2 = E k(x, x') + E k(y, y') - 2 E k(x, y), the squared distance between kernel mean embeddings (the statistic of
 * the kernel two-sample test; `losses.mmd` is the training loss built from samples).
 *
 * @param kxx - Gram matrix of the first sample `[m, m]`.
 * @param kyy - Gram matrix of the second sample `[n, n]`.
 * @param kxy - Cross-kernel `[m, n]`.
 * @param options - The estimator.
 * @returns The scalar statistic (the unbiased and linear ones can be negative).
 * @throws MdmlError `shape` for inconsistent matrices or, for the linear statistic, unequal or too small samples.
 * @concept kernel-two-sample
 * @reference A. Gretton et al., A kernel two-sample test (JMLR 13, 2012).
 * @example
 * ```ts
 * import { mmdSquared } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(mmdSquared([[1, 0], [0, 1]], [[1, 1], [1, 1]], [[0.5, 0.5], [0.5, 0.5]])) // => 0.5
 * ```
 */
export function mmdSquared(
  kxx: TensorLike,
  kyy: TensorLike,
  kxy: TensorLike,
  options: MmdStatisticOptions = {},
): Value {
  const a = square2(kxx, 'mmdSquared')
  const b = square2(kyy, 'mmdSquared')
  const c = matrix(kxy, 'mmdSquared')
  const m = a.shape[0] ?? 0
  const n = b.shape[0] ?? 0
  if (c.shape[0] !== m || c.shape[1] !== n) throw new MdmlError('shape', 'mmdSquared: kxy must be [m, n]')
  const estimator = options.estimator ?? 'biased'
  if (estimator === 'biased') return sub(add(mean(a), mean(b)), mul(mean(c), 2))
  if (estimator === 'unbiased') {
    if (m < 2 || n < 2) throw new MdmlError('shape', 'mmdSquared: the unbiased statistic needs two points a side')
    const within = (k: Value, size: number): Value => div(sum(offDiagonal(k)), size * (size - 1))
    return sub(add(within(a, m), within(b, n)), mul(mean(c), 2))
  }
  if (m !== n || m < 2) throw new MdmlError('shape', 'mmdSquared: the linear statistic needs equal samples of >= 2')
  const pairs = Math.floor(m / 2)
  const flat = (k: Value, i: number, j: number): Value => sum(sliceAxis(sliceAxis(k, 0, i, i + 1), 1, j, j + 1))
  let total: Value | undefined
  for (let p = 0; p < pairs; p++) {
    const i = 2 * p
    const j = 2 * p + 1
    const h = sub(add(flat(a, i, j), flat(b, i, j)), add(flat(c, i, j), flat(c, j, i)))
    total = total === undefined ? h : add(total, h)
  }
  return div(total as Value, pairs)
}

/**
 * The effective dimension (degrees of freedom) of a kernel matrix at regularisation lambda,
 * d(lambda) = tr(K (K + lambda I)^(-1)) = sum_i mu_i / (mu_i + lambda): the number of directions kernel ridge regression
 * fits at that penalty, which governs its statistical rate and the number of random features or Nyström landmarks it
 * needs. (For the penalty of a mean loss over n points, pass n lambda.)
 *
 * @param k - A Gram matrix `[n, n]`.
 * @param lambda - The regularisation lambda > 0.
 * @returns The scalar, differentiable.
 * @concept kernel-matrices
 * @reference T. Zhang, Learning bounds for kernel regression using effective data dimensionality (Neural Computation
 *   17, 2005); A. Rudi and L. Rosasco, Generalization properties of learning with random features (NeurIPS 2017).
 * @example
 * ```ts
 * import { effectiveDimension } from 'mdml-engine/kernels'
 * import { item } from 'mdml-engine/tensor'
 * item(effectiveDimension([[3, 0], [0, 1]], 1)) // => 1.25
 * ```
 */
export function effectiveDimension(k: TensorLike, lambda: TensorLike): Value {
  const v = square2(k, 'effectiveDimension')
  const [l] = promoteFloat(lambda) as [Value]
  return trace(solve(add(v, mul(eye(v.shape[0] ?? 0), l)), v))
}

/** A fitted kernel PCA. */
export interface KernelPca {
  /** Eigenvalues of the centred Gram matrix, descending `[k]` (negative rounding errors set to 0). */
  readonly eigenvalues: Value
  /** The matching unit eigenvectors as columns `[n, k]` (largest-magnitude entry positive, as scikit-learn). */
  readonly eigenvectors: Value
  /** The training points' projections `[n, k]` (eigenvectors times sqrt(eigenvalues)). */
  readonly projections: Value
  /** Project new points from their cross-kernel with the training points `[m, n]`. */
  transform(kCross: TensorLike): Value
}

/** Options of {@link kernelPca}. */
export interface KernelPcaOptions {
  /** The number of components (default all n). */
  readonly nComponents?: number
}

/**
 * Kernel principal component analysis on a Gram matrix (scikit-learn KernelPCA with a precomputed kernel): the
 * eigendecomposition of the centred Gram matrix K_c = V diag(lambda) V^T gives the projections V sqrt(lambda) of the
 * training points onto the principal axes in feature space, and new points project as K_c(new, train) V / sqrt(lambda).
 *
 * @param k - A Gram matrix `[n, n]`.
 * @param options - The number of components.
 * @returns Eigenvalues, eigenvectors, projections and the projection of new points.
 * @throws MdmlError `value` for more components than points.
 * @concept kernel-pca
 * @reference B. Schölkopf, A. Smola and K.-R. Müller, Nonlinear component analysis as a kernel eigenvalue problem
 *   (Neural Computation 10, 1998).
 * @example
 * ```ts
 * import { kernelPca } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(kernelPca([[2, 0], [0, 2]], { nComponents: 1 }).eigenvalues).map((v) => Math.round(v * 1e9) / 1e9) // => [2]
 * ```
 */
export function kernelPca(k: TensorLike, options: KernelPcaOptions = {}): KernelPca {
  const fit = square2(k, 'kernelPca')
  const n = fit.shape[0] ?? 0
  const count = options.nComponents ?? n
  if (!(Number.isInteger(count) && count >= 1 && count <= n))
    throw new MdmlError('value', `kernelPca: nComponents must be in 1..${n}, got ${count}`)
  const { eigenvalues, eigenvectors } = eigh(centerKernel(fit))
  const order = fromStorage(
    Int32Array.from({ length: count }, (_, i) => n - 1 - i),
    [count],
    'int32',
  )
  const raw = take(eigenvalues, order, 0)
  const mask = lessEqual(raw, 0)
  const values = where(mask, 0, raw)
  const vectors = take(eigenvectors, order, 1)
  const roots = sqrt(where(mask, 1, values))
  return {
    eigenvalues: values,
    eigenvectors: vectors,
    projections: where(mask, 0, mul(vectors, roots)),
    transform(kCross: TensorLike): Value {
      return where(mask, 0, div(matmul(centerKernel(kCross, fit), vectors), roots))
    },
  }
}

/** The kernels of scikit-learn's pairwise_kernels, by name. */
export type PairwiseKernelMetric =
  | 'additive_chi2'
  | 'chi2'
  | 'linear'
  | 'poly'
  | 'polynomial'
  | 'rbf'
  | 'laplacian'
  | 'sigmoid'
  | 'cosine'

/** Parameters of {@link pairwiseKernels}, in scikit-learn's names. */
export interface PairwiseKernelOptions {
  /** gamma (default 1 / n_features, or 1 for chi2). */
  readonly gamma?: number
  /** The polynomial degree (default 3). */
  readonly degree?: number
  /** The offset c_0 (default 1). */
  readonly coef0?: number
}

/**
 * The kernel matrix between rows of x and y by scikit-learn's metric names and parameterisation
 * (sklearn.metrics.pairwise.pairwise_kernels): rbf exp(-gamma |x - y|^2), laplacian exp(-gamma |x - y|_1),
 * polynomial (gamma x . y + c_0)^degree, sigmoid tanh(gamma x . y + c_0), chi2 exp(-gamma chi2(x, y)),
 * additive_chi2 -chi2(x, y), linear and cosine; or any {@link Kernel}.
 *
 * @param x - Rows `[m, d]`.
 * @param y - Rows `[n, d]` (default x).
 * @param metric - A metric name or a kernel.
 * @param options - gamma, degree and coef0.
 * @returns `[m, n]`.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { pairwiseKernels } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(pairwiseKernels([[1, 1]], [[2, 0]], 'polynomial', { degree: 2, coef0: 0, gamma: 1 })) // => [[4]]
 * ```
 */
export function pairwiseKernels(
  x: TensorLike,
  y: TensorLike | undefined,
  metric: PairwiseKernelMetric | Kernel,
  options: PairwiseKernelOptions = {},
): Value {
  if (typeof metric !== 'string') return y === undefined ? metric.k(x) : metric.k(x, y)
  const a = matrix(x, 'pairwiseKernels')
  const b = y === undefined ? undefined : matrix(y, 'pairwiseKernels')
  const gamma = options.gamma ?? 1 / (a.shape[1] ?? 1)
  const evaluate = (kernel: Kernel): Value => (b === undefined ? kernel.k(a) : kernel.k(a, b))
  switch (metric) {
    case 'rbf':
      return evaluate(rbfKernel({ lengthScale: 1 / Math.sqrt(2 * gamma) }))
    case 'laplacian':
      return evaluate(laplacianKernel({ lengthScale: 1 / gamma }))
    case 'poly':
    case 'polynomial':
      return evaluate(polynomialKernel({ degree: options.degree ?? 3, gamma, coef0: options.coef0 ?? 1 }))
    case 'sigmoid':
      return evaluate(sigmoidKernel({ gamma, coef0: options.coef0 ?? 1 }))
    case 'linear':
      return evaluate(linearKernel())
    case 'cosine':
      return evaluate(cosineSimilarityKernel())
    case 'chi2':
      return evaluate(chi2Kernel({ gamma: options.gamma ?? 1 }))
    case 'additive_chi2':
      return neg(chi2Distance(a, b))
  }
}

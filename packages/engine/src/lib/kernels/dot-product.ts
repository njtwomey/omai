/**
 * Dot-product kernels, which depend on inner products rather than distances: the linear, polynomial, sigmoid and
 * cosine-similarity kernels, the arc-cosine kernels of Cho and Saul, the neural tangent and NNGP kernels of an
 * infinitely wide fully connected network (as neural-tangents), and the empirical neural tangent kernel of a finite
 * network at its current parameters.
 */

import { jacrev } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import {
  add,
  arcsin,
  concat,
  div,
  einsum,
  expandDims,
  eye,
  matmul,
  mul,
  ones,
  pow,
  promoteFloat,
  reshape,
  sizeOf,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  tanh,
  treeFlatten,
  treeUnflatten,
  type Value,
  where,
} from '../tensor/index.ts'
import { arcCosineJ, safeDivide, safeSqrt } from './_safe.ts'
import { asRows, defineKernel, type Kernel } from './kernel.ts'

function param(x: TensorLike): Value {
  return promoteFloat(x)[0] as Value
}

/** Inner products of rows, `[m, n]`. */
function dots(x: Value, y: Value | undefined): Value {
  return matmul(x, swapaxes(y ?? x, 0, 1))
}

/** Squared norms of rows, `[m]`. */
function squaredNorms(x: Value): Value {
  return sum(square(x), 1)
}

/** For a Gram matrix (y undefined) set the diagonal to exactly 1: a cosine of a point with itself. */
function unitDiagonal(c: Value, gram: boolean): Value {
  if (!gram) return c
  const n = c.shape[0] ?? 0
  return where(eye(n), 1, c)
}

/** Options of {@link linearKernel}. */
export interface LinearOptions {
  /** A weight on the inner product (GPyTorch LinearKernel's variance); omitted, the weight is 1 and not a hyperparameter. */
  readonly variance?: TensorLike
  /** An inhomogeneity sigma_0, adding sigma_0^2 (scikit-learn DotProduct); omitted, there is none. */
  readonly sigma0?: TensorLike
}

/**
 * The linear (dot-product) kernel k(x, y) = sigma_0^2 + v x . y (scikit-learn DotProduct and linear_kernel, GPyTorch
 * LinearKernel): Bayesian linear regression as a Gaussian process. Not stationary.
 *
 * @param options - The optional weight v and offset sigma_0.
 * @returns The kernel, hyperparameters the options given.
 * @concept dot-product-kernels
 * @example
 * ```ts
 * import { linearKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(linearKernel({ sigma0: 2 }).k([1, 2], [3, 4])) // => 15
 * ```
 */
export function linearKernel(options: LinearOptions = {}): Kernel {
  const params: Record<string, Value> = {}
  if (options.variance !== undefined) params['variance'] = param(options.variance)
  if (options.sigma0 !== undefined) params['sigma0'] = param(options.sigma0)
  const affine = (p: Record<string, Value>, v: Value): Value => {
    const weighted = p['variance'] === undefined ? v : mul(p['variance'], v)
    return p['sigma0'] === undefined ? weighted : add(weighted, square(p['sigma0']))
  }
  return defineKernel({
    name: 'linear',
    params,
    matrix: (p, x, y) => affine(p, dots(x, y)),
    diag: (p, x) => affine(p, squaredNorms(x)),
  })
}

/** Options of {@link polynomialKernel}. */
export interface PolynomialOptions {
  /** The degree (default 3, as scikit-learn). */
  readonly degree?: number
  /** The scale gamma of the inner product (default 1). */
  readonly gamma?: TensorLike
  /** The offset c_0 >= 0 (default 1). */
  readonly coef0?: TensorLike
}

/**
 * The polynomial kernel k(x, y) = (gamma x . y + c_0)^degree (scikit-learn polynomial_kernel, GPyTorch
 * PolynomialKernel): the inner product of all monomials up to the degree, weighted by c_0.
 *
 * @param options - Degree, gamma and offset.
 * @returns The kernel, hyperparameters `{ coef0, gamma }` (the degree is a setting).
 * @throws WhiteboxError `domain` for a degree that is not a positive integer.
 * @concept dot-product-kernels
 * @example
 * ```ts
 * import { polynomialKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(polynomialKernel({ degree: 2 }).k([1, 2], [3, 1])) // => 36
 * ```
 */
export function polynomialKernel(options: PolynomialOptions = {}): Kernel {
  const degree = options.degree ?? 3
  if (!(Number.isInteger(degree) && degree >= 1))
    throw new WhiteboxError('domain', `polynomialKernel: the degree must be a positive integer, got ${degree}`)
  return defineKernel({
    name: 'polynomial',
    params: { coef0: param(options.coef0 ?? 1), gamma: param(options.gamma ?? 1) },
    config: { degree },
    matrix: (p, x, y) => pow(add(mul(dots(x, y), p.gamma), p.coef0), degree),
    diag: (p, x) => pow(add(mul(squaredNorms(x), p.gamma), p.coef0), degree),
  })
}

/** Options of {@link sigmoidKernel}. */
export interface SigmoidOptions {
  /** The scale gamma (default 1). */
  readonly gamma?: TensorLike
  /** The offset c_0, any real number (default 1). */
  readonly coef0?: TensorLike
}

/**
 * The sigmoid (hyperbolic tangent, multilayer perceptron) kernel k(x, y) = tanh(gamma x . y + c_0) (scikit-learn
 * sigmoid_kernel). It is not positive semi-definite in general; it is kept for compatibility with SVM practice.
 *
 * @param options - gamma and c_0.
 * @returns The kernel, hyperparameters `{ coef0 (real), gamma }`.
 * @concept dot-product-kernels
 * @example
 * ```ts
 * import { sigmoidKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(sigmoidKernel({ coef0: 0 }).k([1, 0], [0, 1])) // => 0
 * ```
 */
export function sigmoidKernel(options: SigmoidOptions = {}): Kernel {
  return defineKernel({
    name: 'sigmoid',
    params: { coef0: param(options.coef0 ?? 1), gamma: param(options.gamma ?? 1) },
    constraints: { coef0: 'real', gamma: 'positive' },
    matrix: (p, x, y) => tanh(add(mul(dots(x, y), p.gamma), p.coef0)),
    diag: (p, x) => tanh(add(mul(squaredNorms(x), p.gamma), p.coef0)),
  })
}

/**
 * The cosine-similarity kernel k(x, y) = x . y / (|x| |y|) (scikit-learn cosine_similarity, the 'cosine' metric of
 * pairwise_kernels): the linear kernel of the normalised points; a zero vector has similarity 0 with everything.
 *
 * @returns The kernel (no hyperparameters).
 * @concept dot-product-kernels
 * @example
 * ```ts
 * import { cosineSimilarityKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(cosineSimilarityKernel().k([3, 0], [2, 2])) - Math.SQRT1_2) < 1e-15 // => true
 * ```
 */
export function cosineSimilarityKernel(): Kernel {
  const norms = (x: Value): Value => safeSqrt(squaredNorms(x))
  return defineKernel({
    name: 'cosine-similarity',
    params: {},
    matrix: (_p, x, y) => {
      const nx = norms(x)
      const ny = y === undefined ? nx : norms(y)
      return safeDivide(dots(x, y), mul(expandDims(nx, 1), expandDims(ny, 0)))
    },
    diag: (_p, x) => safeDivide(squaredNorms(x), squaredNorms(x)),
  })
}

/** Options of {@link arcCosineKernel}. */
export interface ArcCosineOptions {
  /** The order n: 0 (step), 1 (ReLU) or 2 (squared ReLU) (default 1). */
  readonly order?: 0 | 1 | 2
  /** The number of compositions (default 1): the kernel of a deeper infinite network. */
  readonly depth?: number
}

/**
 * The arc-cosine kernel of Cho and Saul, k_n(x, y) = (1 / pi) |x|^n |y|^n J_n(theta) with theta the angle between x and
 * y, J_0 = pi - theta, J_1 = sin(theta) + (pi - theta) cos(theta) and
 * J_2 = 3 sin(theta) cos(theta) + (pi - theta)(1 + 2 cos^2(theta)): twice the expected product of the step, ReLU or
 * squared-ReLU features of x and y under Gaussian weights, 2 E[Theta(w.x) Theta(w.y) (w.x)^n (w.y)^n]. Composing it
 * `depth` times gives the kernel of a deep network of infinite width. The angle is undefined at the origin, so points
 * should be non-zero (a zero vector is treated as orthogonal to everything). On a Gram matrix the diagonal angle is
 * exactly 0; between separately given equal points, arccos near 1 costs about sqrt(eps) of accuracy.
 *
 * @param options - Order and depth.
 * @returns The kernel (no hyperparameters).
 * @throws WhiteboxError `domain` for an order other than 0, 1, 2 or a depth below 1.
 * @concept dot-product-kernels
 * @reference Y. Cho and L. K. Saul, Kernel methods for deep learning (NeurIPS 2009).
 * @example
 * ```ts
 * import { arcCosineKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(arcCosineKernel({ order: 0 }).k([1, 0], [0, 1])) // => 0.5
 * ```
 */
export function arcCosineKernel(options: ArcCosineOptions = {}): Kernel {
  const order = options.order ?? 1
  const depth = options.depth ?? 1
  if (order !== 0 && order !== 1 && order !== 2)
    throw new WhiteboxError('domain', `arcCosineKernel: the order must be 0, 1 or 2, got ${String(order)}`)
  if (!(Number.isInteger(depth) && depth >= 1))
    throw new WhiteboxError('domain', `arcCosineKernel: the depth must be a positive integer, got ${depth}`)
  // J_n(0) / pi: the factor by which a layer maps k(x, x) to k(x, x)^n.
  const self = [1, 1, 3][order] as number
  const layer = (k: Value, dx: Value, dy: Value, gram: boolean): Value => {
    const outer = mul(expandDims(dx, 1), expandDims(dy, 0))
    const c = unitDiagonal(safeDivide(k, safeSqrt(outer)), gram)
    const scale = order === 0 ? 1 : order === 1 ? safeSqrt(outer) : outer
    return div(mul(arcCosineJ(order, c), scale), Math.PI)
  }
  const nextSelf = (d: Value): Value => (order === 0 ? mul(ones([d.shape[0] ?? 0]), 1) : mul(pow(d, order), self))
  return defineKernel({
    name: 'arc-cosine',
    params: {},
    config: { order, depth },
    matrix: (_p, x, y) => {
      let k = dots(x, y)
      let dx = squaredNorms(x)
      let dy = y === undefined ? dx : squaredNorms(y)
      for (let l = 0; l < depth; l++) {
        k = layer(k, dx, dy, y === undefined)
        dx = nextSelf(dx)
        dy = y === undefined ? dx : nextSelf(dy)
      }
      return k
    },
    diag: (_p, x) => {
      let d = squaredNorms(x)
      for (let l = 0; l < depth; l++) d = nextSelf(d)
      return d
    },
  })
}

/** Options of {@link neuralTangentKernel}. */
export interface NeuralTangentOptions {
  /** The number of hidden layers (default 1). */
  readonly depth?: number
  /** The activation (default 'relu'). */
  readonly activation?: 'relu' | 'erf'
  /** The weight variance sigma_w^2 (default 1; 2 keeps ReLU activations at unit scale). */
  readonly weightVariance?: TensorLike
  /** The bias variance sigma_b^2 (default 0, then fixed). */
  readonly biasVariance?: TensorLike
  /** The tangent kernel (default) or the NNGP kernel (the network output covariance at initialisation). */
  readonly kind?: 'ntk' | 'nngp'
}

/** E[phi(u) phi(v)] and E[phi'(u) phi'(v)] for (u, v) Gaussian with covariance [[sxx, sxy], [sxy, syy]]. */
function activationMoments(
  activation: 'relu' | 'erf',
  sxy: Value,
  sxx: Value,
  syy: Value,
  gram: boolean,
): { value: Value; derivative: Value } {
  const outer = mul(expandDims(sxx, 1), expandDims(syy, 0))
  if (activation === 'relu') {
    const c = unitDiagonal(safeDivide(sxy, safeSqrt(outer)), gram)
    return {
      value: div(mul(arcCosineJ(1, c), safeSqrt(outer)), 2 * Math.PI),
      derivative: div(arcCosineJ(0, c), 2 * Math.PI),
    }
  }
  const a = add(mul(sxx, 2), 1)
  const b = add(mul(syy, 2), 1)
  const ab = mul(expandDims(a, 1), expandDims(b, 0))
  return {
    value: mul(arcsin(div(mul(sxy, 2), sqrt(ab))), 2 / Math.PI),
    derivative: div(4 / Math.PI, sqrt(sub(ab, mul(square(sxy), 4)))),
  }
}

/** The same moments on the diagonal (u = v). */
function activationSelf(activation: 'relu' | 'erf', s: Value): { value: Value; derivative: Value } {
  if (activation === 'relu') return { value: mul(s, 0.5), derivative: mul(ones([s.shape[0] ?? 0]), 0.5) }
  const a = add(mul(s, 2), 1)
  return {
    value: mul(arcsin(div(mul(s, 2), a)), 2 / Math.PI),
    derivative: div(4 / Math.PI, sqrt(add(mul(s, 4), 1))),
  }
}

/**
 * The neural tangent kernel (and the NNGP kernel) of an infinitely wide fully connected network
 * x -> Dense -> (phi -> Dense) x depth, in the NTK parameterisation of neural-tangents (Dense layers W_std / sqrt(n)
 * with sigma_w^2 = W_std^2 and sigma_b^2 = b_std^2). The recursion is Sigma <- sigma_w^2 E[phi phi] + sigma_b^2 and
 * Theta <- sigma_w^2 Theta E[phi' phi'] + Sigma, starting from Sigma = Theta = sigma_w^2 x . y / d + sigma_b^2. Gradient
 * descent on a wide network follows kernel regression with Theta; Sigma is its output covariance at initialisation.
 *
 * @param options - Depth, activation, variances and which kernel.
 * @returns The kernel, hyperparameters `{ biasVariance, weightVariance }`.
 * @throws WhiteboxError `domain` for a depth below 1.
 * @concept neural-tangent-kernel
 * @reference A. Jacot, F. Gabriel and C. Hongler, Neural tangent kernel (NeurIPS 2018); S. Arora et al., On exact
 *   computation with an infinitely wide neural net (NeurIPS 2019); R. Novak et al., Neural Tangents (ICLR 2020).
 * @example
 * ```ts
 * import { neuralTangentKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(neuralTangentKernel({ weightVariance: 2 }).k([1, 1], [1, 1])) // => 4
 * ```
 */
export function neuralTangentKernel(options: NeuralTangentOptions = {}): Kernel {
  const depth = options.depth ?? 1
  const activation = options.activation ?? 'relu'
  const kind = options.kind ?? 'ntk'
  if (!(Number.isInteger(depth) && depth >= 1))
    throw new WhiteboxError('domain', `neuralTangentKernel: the depth must be a positive integer, got ${depth}`)
  return defineKernel({
    name: kind,
    params: { biasVariance: param(options.biasVariance ?? 0), weightVariance: param(options.weightVariance ?? 1) },
    config: { depth, activation, kind },
    matrix: (p, x, y) => {
      const wv = p.weightVariance
      const bv = p.biasVariance
      const dim = x.shape[1] ?? 1
      const dense = (s: Value): Value => add(mul(s, wv), bv)
      const gram = y === undefined
      let sxy = dense(div(dots(x, y), dim))
      let sxx = dense(div(squaredNorms(x), dim))
      let syy = gram ? sxx : dense(div(squaredNorms(y), dim))
      let ntk = sxy
      for (let l = 0; l < depth; l++) {
        const m = activationMoments(activation, sxy, sxx, syy, gram)
        sxy = dense(m.value)
        sxx = dense(activationSelf(activation, sxx).value)
        syy = gram ? sxx : dense(activationSelf(activation, syy).value)
        ntk = add(mul(mul(ntk, m.derivative), wv), sxy)
      }
      return kind === 'ntk' ? ntk : sxy
    },
    diag: (p, x) => {
      const wv = p.weightVariance
      const bv = p.biasVariance
      const dense = (s: Value): Value => add(mul(s, wv), bv)
      let s = dense(div(squaredNorms(x), x.shape[1] ?? 1))
      let ntk = s
      for (let l = 0; l < depth; l++) {
        const m = activationSelf(activation, s)
        s = dense(m.value)
        ntk = add(mul(mul(ntk, m.derivative), wv), s)
      }
      return kind === 'ntk' ? ntk : s
    },
  })
}

/**
 * The empirical neural tangent kernel of a network at its current parameters,
 * Theta(x_i, y_j) = sum_k <d f_k(x_i) / d params, d f_k(y_j) / d params>, the outputs summed over (neural-tangents'
 * empirical_ntk_fn with trace_axes=(-1,)). As the width grows at random initialisation it approaches
 * {@link neuralTangentKernel}.
 *
 * @param f - The network, (params, x `[n, d]`) => outputs `[n]` or `[n, k]`.
 * @param params - The parameters (a pytree of concrete values).
 * @param x - Points `[m, d]`.
 * @param y - Points `[n, d]` (default x).
 * @returns Theta `[m, n]`.
 * @concept neural-tangent-kernel
 * @example
 * ```ts
 * import { empiricalNtk } from 'whitebox-engine/kernels'
 * import { matmul, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const f = (w: Value, x: Value) => matmul(x, w)
 * toArray(empiricalNtk(f, tensor([0.3, -1]), [[1, 2], [0, 1]])) // => [[5, 2], [2, 1]]
 * ```
 */
export function empiricalNtk<P>(f: (params: P, x: Value) => Value, params: P, x: TensorLike, y?: TensorLike): Value {
  const { leaves, def } = treeFlatten(params)
  const sizes = leaves.map((leaf) => sizeOf(leaf.shape))
  const flat = concat(
    leaves.map((leaf) => reshape(leaf, [-1])),
    0,
  )
  const unflatten = (v: Value): P => {
    let offset = 0
    const rebuilt = leaves.map((leaf, i) => {
      const size = sizes[i] ?? 0
      const piece = reshape(sliceAxis(v, 0, offset, offset + size), leaf.shape)
      offset += size
      return piece
    })
    return treeUnflatten(def, rebuilt) as P
  }
  const jacobianAt = (points: Value): Value => {
    const n = points.shape[0] ?? 0
    const j = jacrev((v: Value) => reshape(f(unflatten(v), points), [-1]))(flat)
    return reshape(j, [n, -1, sizeOf(flat.shape)])
  }
  const a = asRows(x, 'empiricalNtk').rows
  const ja = jacobianAt(a)
  const jb = y === undefined ? ja : jacobianAt(asRows(y, 'empiricalNtk').rows)
  return einsum('ikp,jkp->ij', ja, jb)
}

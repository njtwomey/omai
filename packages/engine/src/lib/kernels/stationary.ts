import { WhiteboxError } from '../base/index.ts'
import { cdist } from '../numerics/geometry/index.ts'
import { kv, lgamma } from '../numerics/special/index.ts'
import {
  add,
  cos,
  div,
  equal,
  exp,
  expandDims,
  item,
  less,
  mul,
  neg,
  ones,
  pow,
  promoteFloat,
  reshape,
  sin,
  sliceAxis,
  square,
  sub,
  sum,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { defineKernel, type Kernel } from './kernel.ts'

function param(x: TensorLike): Value {
  return promoteFloat(x)[0] as Value
}

/** Rows scaled by length scales (a scalar, or one per dimension). */
function scaled(x: Value, y: Value | undefined, lengthScale: Value): [Value, Value] {
  const a = div(x, lengthScale)
  return [a, y === undefined ? a : div(y, lengthScale)]
}

/** Squared distances between scaled rows. */
function scaledSquared(x: Value, y: Value | undefined, lengthScale: Value): Value {
  return cdist(...scaled(x, y, lengthScale), 'sqeuclidean')
}

/** Euclidean distances between scaled rows (zero gradient at zero distance). */
function scaledDistance(x: Value, y: Value | undefined, lengthScale: Value): Value {
  return cdist(...scaled(x, y, lengthScale), 'euclidean')
}

/** A diagonal that is the same constant for every row. */
function constantDiag(x: Value, value: Value | number): Value {
  return mul(ones([x.shape[0] ?? 0]), value)
}

function isotropic(name: string, lengthScale: Value): void {
  if (lengthScale.shape.length !== 0)
    throw new WhiteboxError('shape', `${name}: the length scale must be a scalar (the kernel is isotropic)`)
}

/** Options of {@link rbfKernel}. */
export interface LengthScaleOptions {
  /** The length scale l, a scalar or one per input dimension (default 1). */
  readonly lengthScale?: TensorLike
}

/**
 * The squared-exponential (Gaussian, RBF) kernel k(x, y) = exp(-|x - y|^2 / (2 l^2)) (scikit-learn RBF, GPyTorch
 * RBFKernel). Infinitely differentiable sample paths; with one length scale per dimension it is the ARD kernel.
 *
 * @param options - The length scale.
 * @returns The kernel, hyperparameters `{ lengthScale }`.
 * @concept stationary-kernels
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), section 4.2.
 * @example
 * ```ts
 * import { rbfKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(rbfKernel({ lengthScale: 2 }).k([0, 0], [0, 2])) === Math.exp(-0.5) // => true
 * ```
 */
export function rbfKernel(options: LengthScaleOptions = {}): Kernel {
  return defineKernel({
    name: 'rbf',
    params: { lengthScale: param(options.lengthScale ?? 1) },
    stationary: true,
    matrix: (p, x, y) => exp(mul(scaledSquared(x, y, p.lengthScale), -0.5)),
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link maternKernel}. */
export interface MaternOptions extends LengthScaleOptions {
  /** Smoothness nu > 0 (default 1.5): 0.5, 1.5 and 2.5 have closed forms, Infinity is the RBF kernel. */
  readonly nu?: number
}

/**
 * The Matérn kernel k(r) = 2^(1 - nu) / Gamma(nu) (sqrt(2 nu) r)^nu K_nu(sqrt(2 nu) r), r = |x - y| / l (scikit-learn
 * Matern, GPyTorch MaternKernel): sample paths are ceil(nu) - 1 times differentiable. nu = 1/2 is the exponential
 * kernel, nu = 3/2 is (1 + sqrt(3) r) exp(-sqrt(3) r), nu = 5/2 is (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r), and other
 * orders use the Bessel function K_nu.
 *
 * @param options - Length scale and smoothness.
 * @returns The kernel, hyperparameters `{ lengthScale }` (nu is a setting).
 * @throws WhiteboxError `domain` for nu <= 0.
 * @concept stationary-kernels
 * @reference B. Matérn, Spatial Variation (1960); Rasmussen and Williams (2006), section 4.2.1.
 * @example
 * ```ts
 * import { maternKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(maternKernel({ nu: 0.5 }).k([0], [1])) === Math.exp(-1) // => true
 * ```
 */
export function maternKernel(options: MaternOptions = {}): Kernel {
  const nu = options.nu ?? 1.5
  if (!(nu > 0)) throw new WhiteboxError('domain', `maternKernel: nu must be positive, got ${nu}`)
  const profile = (r: Value): Value => {
    if (nu === 0.5) return exp(neg(r))
    if (nu === 1.5) {
      const s = mul(r, Math.sqrt(3))
      return mul(add(s, 1), exp(neg(s)))
    }
    if (nu === 2.5) {
      const s = mul(r, Math.sqrt(5))
      return mul(add(add(s, 1), div(square(s), 3)), exp(neg(s)))
    }
    // The general order: k = 1 - O(z^min(2 nu, 2)) as z -> 0, so below the z where that term is under rounding k = 1
    // is set directly (K_nu itself overflows near 0).
    const z = mul(r, Math.sqrt(2 * nu))
    const zero = less(z, 10 ** (-16 / Math.min(2 * nu, 2)))
    const safe = where(zero, 1, z)
    const c = Math.exp((1 - nu) * Math.LN2 - item(lgamma(nu)))
    return where(zero, 1, mul(mul(pow(safe, nu), kv(nu, safe)), c))
  }
  return defineKernel({
    name: 'matern',
    params: { lengthScale: param(options.lengthScale ?? 1) },
    stationary: true,
    config: { nu: Number.isFinite(nu) ? nu : 'inf' },
    matrix: (p, x, y) =>
      nu === Number.POSITIVE_INFINITY
        ? exp(mul(scaledSquared(x, y, p.lengthScale), -0.5))
        : profile(scaledDistance(x, y, p.lengthScale)),
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link rationalQuadraticKernel}. */
export interface RationalQuadraticOptions extends LengthScaleOptions {
  /** The scale-mixture parameter alpha (default 1). */
  readonly alpha?: TensorLike
}

/**
 * The rational quadratic kernel k(x, y) = (1 + |x - y|^2 / (2 alpha l^2))^(-alpha) (scikit-learn RationalQuadratic,
 * GPyTorch RQKernel): a scale mixture of RBF kernels with gamma-distributed inverse squared length scales, tending to
 * the RBF kernel as alpha grows.
 *
 * @param options - Length scale and alpha.
 * @returns The kernel, hyperparameters `{ alpha, lengthScale }`.
 * @concept stationary-kernels
 * @reference Rasmussen and Williams (2006), equation 4.19.
 * @example
 * ```ts
 * import { rationalQuadraticKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(rationalQuadraticKernel({ alpha: 1 }).k([0], [Math.sqrt(2)])) // => 0.5
 * ```
 */
export function rationalQuadraticKernel(options: RationalQuadraticOptions = {}): Kernel {
  return defineKernel({
    name: 'rational-quadratic',
    params: { alpha: param(options.alpha ?? 1), lengthScale: param(options.lengthScale ?? 1) },
    stationary: true,
    matrix: (p, x, y) => {
      const alpha = p.alpha as Value
      const base = add(div(scaledSquared(x, y, p.lengthScale as Value), mul(alpha, 2)), 1)
      return pow(base, neg(alpha))
    },
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link exponentialKernel}. */
export interface ExponentialOptions extends LengthScaleOptions {
  /** The exponent gamma in (0, 2] (default 1, the Ornstein-Uhlenbeck kernel). */
  readonly power?: number
}

/**
 * The (gamma-)exponential kernel k(x, y) = exp(-(|x - y| / l)^gamma), positive definite for 0 < gamma <= 2: gamma = 1
 * is the Ornstein-Uhlenbeck kernel (Matérn 1/2), gamma = 2 is the RBF kernel with length scale l / sqrt(2).
 *
 * @param options - Length scale and exponent.
 * @returns The kernel, hyperparameters `{ lengthScale }` (the exponent is a setting).
 * @throws WhiteboxError `domain` for an exponent outside (0, 2].
 * @concept stationary-kernels
 * @reference Rasmussen and Williams (2006), equation 4.18.
 * @example
 * ```ts
 * import { exponentialKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(exponentialKernel({ lengthScale: 2 }).k([0], [2])) === Math.exp(-1) // => true
 * ```
 */
export function exponentialKernel(options: ExponentialOptions = {}): Kernel {
  const power = options.power ?? 1
  if (!(power > 0 && power <= 2))
    throw new WhiteboxError('domain', `exponentialKernel: the exponent must be in (0, 2], got ${power}`)
  return defineKernel({
    name: 'exponential',
    params: { lengthScale: param(options.lengthScale ?? 1) },
    stationary: true,
    config: { power },
    matrix: (p, x, y) => {
      const r = scaledDistance(x, y, p.lengthScale)
      if (power === 1) return exp(neg(r))
      const zero = equal(r, 0)
      return where(zero, 1, exp(neg(pow(where(zero, 1, r), power))))
    },
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/**
 * The Laplacian kernel k(x, y) = exp(-sum_d |x_d - y_d| / l_d), the product of one-dimensional exponential kernels
 * (scikit-learn laplacian_kernel with gamma = 1 / l).
 *
 * @param options - The length scale (a scalar, or one per dimension).
 * @returns The kernel, hyperparameters `{ lengthScale }`.
 * @concept stationary-kernels
 * @example
 * ```ts
 * import { laplacianKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(laplacianKernel().k([0, 0], [1, -1])) === Math.exp(-2) // => true
 * ```
 */
export function laplacianKernel(options: LengthScaleOptions = {}): Kernel {
  return defineKernel({
    name: 'laplacian',
    params: { lengthScale: param(options.lengthScale ?? 1) },
    stationary: true,
    matrix: (p, x, y) => exp(neg(cdist(...scaled(x, y, p.lengthScale), 'cityblock'))),
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link periodicKernel}. */
export interface PeriodicOptions {
  /** The length scale l (a scalar; default 1). */
  readonly lengthScale?: TensorLike
  /** The period p (default 1). */
  readonly period?: TensorLike
}

function periodicPart(d: Value, lengthScale: Value, period: Value): Value {
  const s = sin(div(mul(d, Math.PI), period))
  return exp(mul(div(square(s), square(lengthScale)), -2))
}

/**
 * The periodic (exp-sine-squared) kernel k(x, y) = exp(-2 sin^2(pi |x - y| / p) / l^2) (scikit-learn ExpSineSquared,
 * MacKay's periodic kernel): functions that repeat exactly with period p.
 * Positive semi-definite for one-dimensional inputs; with the Euclidean distance of several dimensions it need not be
 * (as for scikit-learn's), so use `productKernel` of one-dimensional periodic kernels for a PSD multi-dimensional version.
 *
 * @param options - Length scale and period.
 * @returns The kernel, hyperparameters `{ lengthScale, period }`.
 * @throws WhiteboxError `shape` for a vector length scale (the kernel is isotropic).
 * @concept stationary-kernels
 * @reference D. J. C. MacKay, Introduction to Gaussian processes (1998).
 * @example
 * ```ts
 * import { periodicKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(periodicKernel({ period: 2 }).k([0.3], [4.3])) - 1) < 1e-12 // => true
 * ```
 */
export function periodicKernel(options: PeriodicOptions = {}): Kernel {
  const lengthScale = param(options.lengthScale ?? 1)
  isotropic('periodicKernel', lengthScale)
  return defineKernel({
    name: 'periodic',
    params: { lengthScale, period: param(options.period ?? 1) },
    stationary: true,
    config: { isotropic: true },
    matrix: (p, x, y) => periodicPart(cdist(x, y ?? x, 'euclidean'), p.lengthScale as Value, p.period as Value),
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link locallyPeriodicKernel}. */
export interface LocallyPeriodicOptions extends PeriodicOptions {
  /** The length scale of the RBF envelope (default 1). */
  readonly decay?: TensorLike
}

/**
 * The locally periodic kernel, a periodic kernel under an RBF envelope:
 * k(x, y) = exp(-2 sin^2(pi r / p) / l^2) exp(-r^2 / (2 decay^2)), r = |x - y|: patterns that repeat while slowly
 * changing shape (scikit-learn ExpSineSquared * RBF).
 * Like the periodic kernel, it is positive semi-definite for one-dimensional inputs only.
 *
 * @param options - Length scale, period and envelope length scale.
 * @returns The kernel, hyperparameters `{ decay, lengthScale, period }`.
 * @throws WhiteboxError `shape` for a vector length scale.
 * @concept stationary-kernels
 * @reference D. Duvenaud, Automatic Model Construction with Gaussian Processes (PhD thesis, 2014), chapter 2.
 * @example
 * ```ts
 * import { locallyPeriodicKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(locallyPeriodicKernel({ period: 1, decay: 1 }).k([0], [1])) - Math.exp(-0.5)) < 1e-12 // => true
 * ```
 */
export function locallyPeriodicKernel(options: LocallyPeriodicOptions = {}): Kernel {
  const lengthScale = param(options.lengthScale ?? 1)
  isotropic('locallyPeriodicKernel', lengthScale)
  return defineKernel({
    name: 'locally-periodic',
    params: { decay: param(options.decay ?? 1), lengthScale, period: param(options.period ?? 1) },
    stationary: true,
    config: { isotropic: true },
    matrix: (p, x, y) => {
      const decay = p.decay as Value
      const d = cdist(x, y ?? x, 'euclidean')
      const envelope = exp(mul(scaledSquared(x, y, decay), -0.5))
      return mul(periodicPart(d, p.lengthScale as Value, p.period as Value), envelope)
    },
    diag: (_p, x) => constantDiag(x, 1),
  })
}

/** Options of {@link spectralMixtureKernel}. */
export interface SpectralMixtureOptions {
  /** Mixture weights w_q `[Q]`. */
  readonly weights: TensorLike
  /** Spectral means mu_q `[Q, d]` (frequencies). */
  readonly means: TensorLike
  /** Spectral scales s_q `[Q, d]` (standard deviations of the spectral Gaussians, inverse length scales). */
  readonly scales: TensorLike
}

/**
 * The spectral mixture kernel of Wilson and Adams: the spectral density is a mixture of Gaussians, so
 * k(tau) = sum_q w_q prod_d exp(-2 pi^2 tau_d^2 s_qd^2) cos(2 pi tau_d mu_qd), tau = x - y. It can approximate any
 * stationary kernel. (GPyTorch's SpectralMixtureKernel sums over components before the product over dimensions; the
 * two agree in one dimension.)
 *
 * @param options - Weights, means and scales.
 * @returns The kernel, hyperparameters `{ means, scales, weights }`.
 * @throws WhiteboxError `shape` when the shapes disagree.
 * @concept stationary-kernels
 * @reference A. G. Wilson and R. P. Adams, Gaussian process kernels for pattern discovery and extrapolation (ICML 2013).
 * @example
 * ```ts
 * import { spectralMixtureKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * const k = spectralMixtureKernel({ weights: [2], means: [[0.5]], scales: [[0]] })
 * Math.abs(item(k.k([0], [1])) + 2) < 1e-12 // => true
 * ```
 */
export function spectralMixtureKernel(options: SpectralMixtureOptions): Kernel {
  const weights = param(options.weights)
  const means = param(options.means)
  const scales = param(options.scales)
  const q = weights.shape[0] ?? 0
  if (
    weights.shape.length !== 1 ||
    means.shape.length !== 2 ||
    means.shape[0] !== q ||
    scales.shape.join(',') !== means.shape.join(',')
  )
    throw new WhiteboxError('shape', 'spectralMixtureKernel: weights [Q], means [Q, d] and scales [Q, d]')
  const dim = means.shape[1] ?? 0
  return defineKernel({
    name: 'spectral-mixture',
    params: { means, scales, weights },
    stationary: true,
    config: { components: q },
    matrix: (p, x, y) => {
      if ((x.shape[1] ?? 0) !== dim)
        throw new WhiteboxError('shape', `spectralMixtureKernel: points must be [n, ${dim}]`)
      const other = y ?? x
      // tau [1, m, n, d] against the components [Q, 1, 1, d].
      const tau = expandDims(sub(expandDims(x, 1), expandDims(other, 0)), 0)
      const component = (v: Value): Value => reshape(v, [-1, 1, 1, dim])
      const envelope = exp(mul(sum(square(mul(tau, component(p.scales as Value))), -1), -2 * Math.PI * Math.PI))
      const phases = cos(mul(mul(tau, component(p.means as Value)), 2 * Math.PI))
      let waves = sum(sliceAxis(phases, 3, 0, 1), 3)
      for (let j = 1; j < dim; j++) waves = mul(waves, sum(sliceAxis(phases, 3, j, j + 1), 3))
      return sum(mul(mul(envelope, waves), reshape(p.weights as Value, [-1, 1, 1])), 0)
    },
    diag: (p, x) => constantDiag(x, sum(p.weights as Value)),
  })
}

/** Options of {@link cosineKernel}. */
export interface CosineOptions {
  /** The period p (default 1). */
  readonly period?: TensorLike
}

/**
 * The cosine kernel k(x, y) = cos(pi |x - y| / p) (GPyTorch CosineKernel). Positive semi-definite in one dimension
 * (cos(a - b) = cos a cos b + sin a sin b) but not in general for several.
 *
 * @param options - The period.
 * @returns The kernel, hyperparameters `{ period }`.
 * @concept stationary-kernels
 * @example
 * ```ts
 * import { cosineKernel } from 'whitebox-engine/kernels'
 * import { item } from 'whitebox-engine/tensor'
 * item(cosineKernel({ period: 2 }).k([0], [2])) // => -1
 * ```
 */
export function cosineKernel(options: CosineOptions = {}): Kernel {
  return defineKernel({
    name: 'cosine',
    params: { period: param(options.period ?? 1) },
    stationary: true,
    matrix: (p, x, y) => cos(div(mul(cdist(x, y ?? x, 'euclidean'), Math.PI), p.period)),
    diag: (_p, x) => constantDiag(x, 1),
  })
}

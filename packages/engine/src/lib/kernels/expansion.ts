/**
 * Explicit feature maps that approximate a kernel by an inner product of finite feature vectors, so kernel methods
 * scale to large data: random Fourier and orthogonal random features (Rahimi and Recht), the Nystrom approximation from
 * landmarks, additive and skewed chi-square samplers, tensor and polynomial count sketches (as scikit-learn's
 * kernel_approximation), and the errors of the approximate Gram matrix.
 */

import { assertFinite, MdmlError } from '../base/index.ts'
import { type ContextOptions, resolveKey } from '../context/index.ts'
import { fft, ifft } from '../numerics/fourier/index.ts'
import { qr, svd, svdvals } from '../numerics/linalg/index.ts'
import { cauchy, chisquare, foldIn, gamma, type Key, normal, permutation, randint, uniform } from '../random/index.ts'
import {
  abs,
  add,
  cMul,
  concat,
  cos,
  diagonal,
  div,
  expandDims,
  fromStorage,
  greater,
  isnan,
  isTensor,
  item,
  log,
  matmul,
  max,
  mul,
  ones,
  primalOf,
  promoteFloat,
  real,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  tan,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { asRows, type Kernel } from './kernel.ts'

/** An explicit feature map z(x), with z(x) . z(y) approximating (or equal to) a kernel k(x, y). */
export interface FeatureMap {
  readonly name: string
  /** Features of rows `[n, d]` (or one point `[d]`), `[n, D]`. */
  transform(x: TensorLike): Value
}

/** A size or count that must be a positive whole number (a zero or NaN here hangs, wraps or raises a typed-array error later). */
function positiveInteger(name: string, what: string, v: number): number {
  if (!Number.isInteger(v) || v < 1)
    throw new MdmlError('value', `${name}: ${what} must be a positive integer, got ${v}`)
  return v
}

/** A setting that must be finite and above `min` (strictly when `open`). */
function finiteAbove(name: string, what: string, v: number, min: number, open: boolean): number {
  if (!Number.isFinite(v) || (open ? v <= min : v < min))
    throw new MdmlError('value', `${name}: ${what} must be a finite number ${open ? '>' : '>='} ${min}, got ${v}`)
  return v
}

function rowsOf(x: TensorLike, name: string): { rows: Value; single: boolean } {
  return asRows(x, name)
}

function finish(z: Value, single: boolean): Value {
  return single ? sum(z, 0) : z
}

function concrete(x: TensorLike): Tensor<'float64'> {
  return primalOf(promoteFloat(x)[0] as Value) as Tensor<'float64'>
}

/** Random Fourier features z(x) = sqrt(2 s / D) cos(x W + b), from their draws. */
export interface RandomFourierFeatures extends FeatureMap {
  /** Frequencies W `[d, D]`. */
  readonly weights: Tensor<'float64'>
  /** Phases b `[D]`. */
  readonly offset: Tensor<'float64'>
  /** The output scale s (default 1). */
  readonly outputScale: number
}

/** Options of {@link randomFourierFeatures}: either draw frequencies for a kernel, or give them. */
export type RandomFourierOptions =
  | (ContextOptions & {
      /** The input dimension d. */
      readonly inputDim: number
      /** The number of features D (default 100). */
      readonly nComponents?: number
      /** The stationary kernel (default rbfKernel()): rbf, matern, exponential (power 1), laplacian, rational quadratic, or scaleKernel of one. */
      readonly kernel?: Kernel
    })
  | {
      readonly weights: TensorLike
      readonly offset: TensorLike
      readonly outputScale?: number
    }

/** Per-column multipliers of Gaussian frequencies drawn from a kernel's spectral density, as `[d, D]` frequencies. */
function spectralFrequencies(k: Key, kernel: Kernel, d: number, n: number): { weights: Tensor; scale: number } {
  if (kernel.name === 'scale') {
    const inner = spectralFrequencies(k, kernel.parts[0] as Kernel, d, n)
    return { weights: inner.weights, scale: inner.scale * item(primalOf(kernel.params['outputScale'] as Value)) }
  }
  const l = concrete(kernel.params['lengthScale'] as Value)
  const lengthScale = l.shape.length === 0 ? (l as Value) : reshape(l, [d, 1])
  const z = normal(foldIn(k, 0), [d, n])
  // Scale mixtures of Gaussians: column j of z is a Gaussian frequency with random precision tau_j.
  const frequencies = (w: Value): Tensor => primalOf(div(w, lengthScale)) as Tensor
  // A multivariate t with nu' degrees of freedom: w = z / sqrt(tau), tau = chi2(nu') / nu'.
  const studentT = (dof: number): Tensor => frequencies(div(z, sqrt(div(chisquare(foldIn(k, 1), [1, n], dof), dof))))
  switch (kernel.name) {
    case 'rbf':
      return { weights: frequencies(z), scale: 1 }
    case 'matern': {
      const nu = kernel.config['nu']
      // The Matérn spectral density is a multivariate t with 2 nu degrees of freedom and scale 1 / l.
      return { weights: nu === 'inf' ? frequencies(z) : studentT(2 * Number(nu)), scale: 1 }
    }
    case 'exponential':
      if (kernel.config['power'] !== 1)
        throw new MdmlError('not-implemented', 'randomFourierFeatures: only the exponent 1 (Matérn 1/2) is sampled')
      return { weights: studentT(1), scale: 1 }
    case 'rational-quadratic': {
      const alpha = item(primalOf(kernel.params['alpha'] as Value))
      // RQ is a mixture of RBF kernels over the precision tau ~ Gamma(alpha, rate alpha): w = z sqrt(tau) / l.
      return { weights: frequencies(mul(z, sqrt(div(gamma(foldIn(k, 1), [1, n], alpha), alpha)))), scale: 1 }
    }
    case 'laplacian': {
      // A product of one-dimensional exponential kernels: independent Cauchy frequencies.
      return { weights: frequencies(cauchy(foldIn(k, 2), [d, n])), scale: 1 }
    }
    default:
      throw new MdmlError('not-implemented', `randomFourierFeatures: no spectral sampler for ${kernel.name}`)
  }
}

function fourierMap(name: string, weights: Tensor<'float64'>, offset: Tensor<'float64'>, scale: number) {
  const n = offset.shape[0] ?? 0
  if (weights.shape.length !== 2 || weights.shape[1] !== n || offset.shape.length !== 1)
    throw new MdmlError('shape', `${name}: weights [d, D] and offset [D]`)
  const factor = Math.sqrt((2 * scale) / n)
  return {
    name,
    weights,
    offset,
    outputScale: scale,
    transform(x: TensorLike): Value {
      const { rows, single } = rowsOf(x, name)
      return finish(mul(cos(add(matmul(rows, weights), offset)), factor), single)
    },
  }
}

/**
 * Random Fourier features (Rahimi and Recht): by Bochner's theorem a stationary kernel is the Fourier transform of a
 * probability density p(w), so z(x) = sqrt(2 / D) cos(x W + b) with the columns of W drawn from p and b uniform on
 * [0, 2 pi) gives E[z(x) . z(y)] = k(x, y). The RBF kernel has Gaussian frequencies (scikit-learn RBFSampler, with
 * gamma = 1 / (2 l^2)), Matérn ones are Student-t, the Laplacian kernel's are Cauchy and the rational quadratic's are a
 * gamma scale mixture of Gaussians. Given `weights` and `offset`, the map uses those draws instead.
 *
 * @param options - Input dimension, number of features, kernel and key; or the draws.
 * @returns The feature map with its frequencies and phases.
 * @throws MdmlError `not-implemented` for a kernel without a spectral sampler; `value` for an input dimension or a
 *   number of features that is not a positive integer.
 * @concept random-features
 * @reference A. Rahimi and B. Recht, Random features for large-scale kernel machines (NeurIPS 2007).
 * @example
 * ```ts
 * import { randomFourierFeatures } from 'mdml-engine/kernels'
 * import { key } from 'mdml-engine/random'
 * randomFourierFeatures({ inputDim: 3, nComponents: 50, key: key(0) }).transform([[1, 2, 3]]).shape // => [1, 50]
 * ```
 */
export function randomFourierFeatures(options: RandomFourierOptions): RandomFourierFeatures {
  if ('weights' in options)
    return fourierMap('random-fourier', concrete(options.weights), concrete(options.offset), options.outputScale ?? 1)
  const n = positiveInteger('randomFourierFeatures', 'nComponents', options.nComponents ?? 100)
  positiveInteger('randomFourierFeatures', 'inputDim', options.inputDim)
  const k = resolveKey(options, 'randomFourierFeatures')
  const kernel = options.kernel
  const { weights, scale } =
    kernel === undefined
      ? { weights: normal(foldIn(k, 0), [options.inputDim, n]), scale: 1 }
      : spectralFrequencies(k, kernel, options.inputDim, n)
  const offset = uniform(foldIn(k, 9), [n], { high: 2 * Math.PI })
  return fourierMap('random-fourier', weights as Tensor<'float64'>, offset, scale)
}

/** Options of {@link orthogonalRandomFeatures}. */
export interface OrthogonalRandomOptions extends ContextOptions {
  readonly inputDim: number
  /** The number of features D (default 100). */
  readonly nComponents?: number
  /** The RBF length scale (default 1). */
  readonly lengthScale?: number
}

/**
 * Orthogonal random features for the RBF kernel (Yu et al.): the Gaussian frequency matrix is replaced, block by
 * block of d frequencies, by a uniformly random orthogonal matrix with chi-distributed row norms, W = S Q / l. Each
 * frequency keeps its Gaussian marginal, so the estimate stays unbiased, while orthogonality lowers its variance.
 *
 * @param options - Input dimension, number of features, length scale and key.
 * @returns The feature map with its frequencies and phases.
 * @throws MdmlError `value` for an input dimension or number of features that is not a positive integer, or a length
 *   scale that is not positive and finite.
 * @concept random-features
 * @reference F. X. Yu et al., Orthogonal random features (NeurIPS 2016).
 * @example
 * ```ts
 * import { orthogonalRandomFeatures } from 'mdml-engine/kernels'
 * import { key } from 'mdml-engine/random'
 * orthogonalRandomFeatures({ inputDim: 2, nComponents: 5, key: key(1) }).weights.shape // => [2, 5]
 * ```
 */
export function orthogonalRandomFeatures(options: OrthogonalRandomOptions): RandomFourierFeatures {
  const d = positiveInteger('orthogonalRandomFeatures', 'inputDim', options.inputDim)
  const n = positiveInteger('orthogonalRandomFeatures', 'nComponents', options.nComponents ?? 100)
  const l = finiteAbove('orthogonalRandomFeatures', 'lengthScale', options.lengthScale ?? 1, 0, true)
  const k = resolveKey(options, 'orthogonalRandomFeatures')
  const blocks: Value[] = []
  for (let b = 0; b * d < n; b++) {
    const { q, r } = qr(normal(foldIn(k, 2 * b), [d, d]))
    // Q diag(sign(diag R)) is Haar-distributed; the row norms of a d x d Gaussian matrix are chi(d) distributed.
    const haar = mul(q, where(greater(diagonal(r), 0), 1, -1))
    const norms = sqrt(sum(square(normal(foldIn(k, 2 * b + 1), [d, d])), 1))
    blocks.push(mul(swapaxes(haar, 0, 1), expandDims(norms, 1)))
  }
  const rows = sliceAxis(concat(blocks, 0), 0, 0, n)
  const weights = primalOf(div(swapaxes(rows, 0, 1), l)) as Tensor<'float64'>
  const offset = uniform(foldIn(k, 1 << 20), [n], { high: 2 * Math.PI })
  return fourierMap('orthogonal-random', weights, offset, 1)
}

/** A fitted Nyström map. */
export interface Nystroem extends FeatureMap {
  readonly kernel: Kernel
  /** The landmark points `[c, d]`. */
  readonly components: Value
  /** Their row indices in the fitting data. */
  readonly componentIndices: readonly number[]
  /** K_cc^(-1/2) `[c, c]` (scikit-learn's normalization_). */
  readonly normalization: Value
}

/** Options of {@link nystroem}. */
export interface NystroemOptions extends ContextOptions {
  /** The number of landmarks (default 100, at most the number of points). */
  readonly nComponents?: number
  /** Landmark row indices (default: a random subset). */
  readonly indices?: readonly number[]
}

/**
 * The Nyström approximation (scikit-learn Nystroem): with landmarks C drawn from the data,
 * z(x) = K_cc^(-1/2) k(C, x), so that z(x) . z(y) = k(x, C) K_cc^+ k(C, y), the kernel projected onto the span of the
 * landmarks. K_cc^(-1/2) comes from its SVD with singular values clipped at 1e-12, as scikit-learn.
 *
 * @param kernel - Any kernel.
 * @param x - The data `[n, d]` to draw landmarks from.
 * @param options - Number of landmarks or their indices, and the key.
 * @returns The fitted map.
 * @throws MdmlError `value` for a number of landmarks that is not a positive integer, or landmark indices outside the rows.
 * @concept random-features
 * @reference C. K. I. Williams and M. Seeger, Using the Nyström method to speed up kernel machines (NeurIPS 2001).
 * @example
 * ```ts
 * import { linearKernel, nystroem } from 'mdml-engine/kernels'
 * nystroem(linearKernel(), [[1, 0], [0, 1], [1, 1]], { indices: [0, 1] }).transform([[2, 3]]).shape // => [1, 2]
 * ```
 */
export function nystroem(kernel: Kernel, x: TensorLike, options: NystroemOptions = {}): Nystroem {
  const { rows } = rowsOf(x, 'nystroem')
  const n = rows.shape[0] ?? 0
  if (options.indices !== undefined) {
    for (const i of options.indices) {
      if (!Number.isInteger(i) || i < 0 || i >= n)
        throw new MdmlError('value', `nystroem: landmark ${i} is not a row in [0, ${n})`)
    }
  } else positiveInteger('nystroem', 'nComponents', options.nComponents ?? 100)
  const indices =
    options.indices ??
    Array.from(permutation(resolveKey(options, 'nystroem'), n).data).slice(0, Math.min(n, options.nComponents ?? 100))
  const components = take(rows, fromStorage(Int32Array.from(indices), [indices.length], 'int32'), 0)
  const { u, s, vh } = svd(kernel.k(components))
  const clipped = toFlat(primalOf(s)).map((v) => 1 / Math.sqrt(Math.max(v, 1e-12)))
  const normalization = matmul(mul(u, fromStorage(Float64Array.from(clipped), [clipped.length], 'float64')), vh)
  return {
    name: 'nystroem',
    kernel,
    components,
    componentIndices: indices,
    normalization,
    transform(points: TensorLike): Value {
      const r = rowsOf(points, 'nystroem')
      return finish(matmul(kernel.k(r.rows, components), swapaxes(normalization, 0, 1)), r.single)
    },
  }
}

/** Options of {@link additiveChi2Sampler}. */
export interface AdditiveChi2Options {
  /** The number of frequency samples (default 2). */
  readonly sampleSteps?: number
  /** The sampling interval L (default 0.8, 0.5, 0.4 for 1, 2, 3 steps; required otherwise). */
  readonly sampleInterval?: number
}

/**
 * The additive chi-square feature map (scikit-learn AdditiveChi2Sampler): a deterministic sampling of the Fourier
 * transform of the chi-square kernel's spectrum at frequencies 0, L, ..., (n - 1) L, giving d (2 n - 1) features per
 * point whose inner products approximate the additive chi-square kernel sum_i 2 x_i y_i / (x_i + y_i). Inputs must be
 * non-negative; zero features stay zero.
 *
 * @param options - Steps and interval.
 * @returns The feature map.
 * @throws MdmlError `value` for more than 3 steps without an interval, or an interval that is not positive and finite;
 *   `domain` from `transform` for a negative feature (a NaN feature gives NaN features).
 * @concept random-features
 * @reference A. Vedaldi and A. Zisserman, Efficient additive kernels via explicit feature maps (TPAMI 2012).
 * @example
 * ```ts
 * import { additiveChi2Sampler } from 'mdml-engine/kernels'
 * additiveChi2Sampler({ sampleSteps: 2 }).transform([[1, 2, 0]]).shape // => [1, 9]
 * ```
 */
export function additiveChi2Sampler(options: AdditiveChi2Options = {}): FeatureMap {
  const steps = options.sampleSteps ?? 2
  const defaults: Readonly<Record<number, number>> = { 1: 0.8, 2: 0.5, 3: 0.4 }
  const interval = options.sampleInterval ?? defaults[steps]
  if (!(Number.isInteger(steps) && steps >= 1))
    throw new MdmlError('value', `additiveChi2Sampler: sampleSteps must be a positive integer, got ${steps}`)
  if (interval === undefined)
    throw new MdmlError('value', 'additiveChi2Sampler: give sampleInterval for more than 3 sample steps')
  finiteAbove('additiveChi2Sampler', 'sampleInterval', interval, 0, true)
  return {
    name: 'additive-chi2',
    transform(x: TensorLike): Value {
      const { rows, single } = rowsOf(x, 'additiveChi2Sampler')
      if (isTensor(rows) && Array.from(rows.data).some((v) => v < 0))
        throw new MdmlError('domain', 'additiveChi2Sampler: features must be non-negative')
      const nonzero = greater(rows, 0)
      const missing = isnan(rows)
      // A zero feature stays zero, a NaN feature gives NaN features (NaN > 0 is false and must not read as zero).
      const keep = (v: Value): Value => where(missing, Number.NaN, where(nonzero, v, 0))
      const safe = where(nonzero, rows, 1)
      const blocks = [keep(sqrt(mul(safe, interval)))]
      const logStep = mul(log(safe), interval)
      const step = mul(safe, 2 * interval)
      for (let j = 1; j < steps; j++) {
        const factor = sqrt(div(step, Math.cosh(Math.PI * j * interval)))
        blocks.push(keep(mul(factor, cos(mul(logStep, j)))))
        blocks.push(keep(mul(factor, sin(mul(logStep, j)))))
      }
      return finish(concat(blocks, 1), single)
    },
  }
}

/** Skewed chi-square random features, with their draws. */
export interface SkewedChi2Features extends FeatureMap {
  readonly weights: Tensor<'float64'>
  readonly offset: Tensor<'float64'>
  readonly skewedness: number
}

/** Options of {@link skewedChi2Sampler}: either draw for a dimension, or give the draws. */
export type SkewedChi2SamplerOptions =
  | (ContextOptions & {
      readonly inputDim: number
      /** The number of features D (default 100). */
      readonly nComponents?: number
      /** The skewedness c (default 1). */
      readonly skewedness?: number
    })
  | { readonly weights: TensorLike; readonly offset: TensorLike; readonly skewedness?: number }

/**
 * Random features for the skewed chi-square kernel (scikit-learn SkewedChi2Sampler): z(x) =
 * sqrt(2 / D) cos(log(x + c) W + b), with W hyperbolic-secant distributed ((1 / pi) log tan(pi u / 2), u uniform) and
 * b uniform on [0, 2 pi). Inputs must exceed -c.
 *
 * @param options - Input dimension, number of features, skewedness and key; or the draws.
 * @returns The feature map with its draws.
 * @throws MdmlError `value` for an input dimension or number of features that is not a positive integer, or a skewedness
 *   that is not finite.
 * @concept random-features
 * @reference F. Li, C. Ionescu and C. Sminchisescu, Random Fourier approximations for skewed multiplicative histogram
 *   kernels (DAGM 2010).
 * @example
 * ```ts
 * import { skewedChi2Sampler } from 'mdml-engine/kernels'
 * import { key } from 'mdml-engine/random'
 * skewedChi2Sampler({ inputDim: 2, nComponents: 8, key: key(0) }).transform([0.5, 1]).shape // => [8]
 * ```
 */
export function skewedChi2Sampler(options: SkewedChi2SamplerOptions): SkewedChi2Features {
  const c = assertFinite(options.skewedness ?? 1, 'skewedChi2Sampler: skewedness')
  let weights: Tensor<'float64'>
  let offset: Tensor<'float64'>
  if ('weights' in options) {
    weights = concrete(options.weights)
    offset = concrete(options.offset)
  } else {
    const n = positiveInteger('skewedChi2Sampler', 'nComponents', options.nComponents ?? 100)
    positiveInteger('skewedChi2Sampler', 'inputDim', options.inputDim)
    const k = resolveKey(options, 'skewedChi2Sampler')
    const u = uniform(foldIn(k, 0), [options.inputDim, n])
    weights = primalOf(div(log(tan(mul(u, Math.PI / 2))), Math.PI)) as Tensor<'float64'>
    offset = uniform(foldIn(k, 1), [n], { high: 2 * Math.PI })
  }
  const n = offset.shape[0] ?? 0
  return {
    name: 'skewed-chi2',
    weights,
    offset,
    skewedness: c,
    transform(x: TensorLike): Value {
      const { rows, single } = rowsOf(x, 'skewedChi2Sampler')
      if (isTensor(rows) && Array.from(rows.data).some((v) => v <= -c))
        throw new MdmlError('domain', `skewedChi2Sampler: features must exceed -${c}`)
      const projection = add(matmul(log(add(rows, c)), weights), offset)
      return finish(mul(cos(projection), Math.sqrt(2 / n)), single)
    },
  }
}

/** A tensor sketch with its hash functions. */
export interface TensorSketch {
  /** The bucket of each input coordinate, one row per factor. */
  readonly indexHash: readonly (readonly number[])[]
  /** The sign (+1 or -1) of each input coordinate, one row per factor. */
  readonly bitHash: readonly (readonly number[])[]
  readonly nComponents: number
  /** The sketch `[n, D]` of the tensor product x_1 ⊗ ... ⊗ x_p of one row from each input. */
  sketch(...xs: readonly TensorLike[]): Value
}

/** Options of {@link tensorSketch}: either draw hashes for the factor dimensions, or give them. */
export type TensorSketchOptions =
  | (ContextOptions & { readonly inputDims: readonly number[]; readonly nComponents: number })
  | {
      readonly indexHash: readonly (readonly number[])[]
      readonly bitHash: readonly (readonly number[])[]
      readonly nComponents: number
    }

/** The count-sketch matrix S `[d, D]` with S[j, h(j)] = s(j). */
function countSketchMatrix(index: readonly number[], signs: readonly number[], n: number): Tensor<'float64'> {
  const d = index.length
  const data = new Float64Array(d * n)
  for (let j = 0; j < d; j++) data[j * n + (index[j] ?? 0)] = signs[j] ?? 0
  return fromStorage(data, [d, n], 'float64')
}

/**
 * The tensor sketch of Pham and Pagh: the count sketch of a tensor product x_1 ⊗ ... ⊗ x_p is the circular
 * convolution of the count sketches of the factors, computed with FFTs in O(p (d + D log D)). Inner products of
 * sketches estimate inner products of the tensor products without bias, so sketching feature maps of several kernels
 * approximates their product kernel. Differentiable in the inputs.
 *
 * @param options - Factor dimensions, sketch size and key; or the hashes.
 * @returns The sketch with its hashes.
 * @throws MdmlError `shape` when hashes and inputs disagree; `value` for a sketch size or factor dimension that is not a
 *   positive integer, or an index hash outside the sketch.
 * @concept random-features
 * @reference N. Pham and R. Pagh, Fast and scalable polynomial kernels via explicit feature maps (KDD 2013).
 * @example
 * ```ts
 * import { tensorSketch } from 'mdml-engine/kernels'
 * import { toArray } from 'mdml-engine/tensor'
 * const t = tensorSketch({ indexHash: [[0, 1], [1, 0]], bitHash: [[1, 1], [1, -1]], nComponents: 2 })
 * toArray(t.sketch([[1, 2]], [[3, 4]])).map((r) => r.map(Math.round)) // => [[2, -5]]
 * ```
 */
export function tensorSketch(options: TensorSketchOptions): TensorSketch {
  const n = positiveInteger('tensorSketch', 'nComponents', options.nComponents)
  let indexHash: (readonly number[])[]
  let bitHash: (readonly number[])[]
  if ('indexHash' in options) {
    indexHash = options.indexHash.map((r) => [...r])
    bitHash = options.bitHash.map((r) => [...r])
  } else {
    for (const d of options.inputDims) positiveInteger('tensorSketch', 'every input dimension', d)
    const k = resolveKey(options, 'tensorSketch')
    indexHash = options.inputDims.map((d, p) => Array.from(randint(foldIn(k, 2 * p), [d], 0, n).data))
    bitHash = options.inputDims.map((d, p) =>
      Array.from(randint(foldIn(k, 2 * p + 1), [d], 0, 2).data, (b) => (b === 0 ? -1 : 1)),
    )
  }
  if (indexHash.length !== bitHash.length || indexHash.some((r, p) => r.length !== bitHash[p]?.length))
    throw new MdmlError('shape', 'tensorSketch: indexHash and bitHash must have the same shape')
  for (const r of indexHash) {
    if (r.some((h) => !Number.isInteger(h) || h < 0 || h >= n))
      throw new MdmlError('value', `tensorSketch: every index hash must be an integer in [0, ${n})`)
  }
  const matrices = indexHash.map((r, p) => countSketchMatrix(r, bitHash[p] ?? [], n))
  return {
    indexHash,
    bitHash,
    nComponents: n,
    sketch(...xs: readonly TensorLike[]): Value {
      if (xs.length !== matrices.length)
        throw new MdmlError('shape', `tensorSketch: expected ${matrices.length} inputs, got ${xs.length}`)
      const parts = xs.map((x) => rowsOf(x, 'tensorSketch'))
      const spectra = parts.map((part, p) => fft(matmul(part.rows, matrices[p] as Value), undefined, -1))
      const product = spectra.slice(1).reduce((acc, z) => cMul(acc, z), spectra[0] as (typeof spectra)[number])
      return finish(
        real(ifft(product, undefined, -1)),
        parts.every((part) => part.single),
      )
    },
  }
}

/** A polynomial count sketch with its hashes. */
export interface PolynomialCountSketch extends FeatureMap {
  readonly indexHash: readonly (readonly number[])[]
  readonly bitHash: readonly (readonly number[])[]
}

/** Options of {@link polynomialCountSketch}. */
export type PolynomialCountSketchOptions = {
  /** The degree p (default 2). */
  readonly degree?: number
  /** gamma (default 1). */
  readonly gamma?: number
  /** c_0 >= 0 (default 0). */
  readonly coef0?: number
  /** The sketch size D (default 100). */
  readonly nComponents?: number
} & (
  | (ContextOptions & { readonly inputDim: number })
  | { readonly indexHash: readonly (readonly number[])[]; readonly bitHash: readonly (readonly number[])[] }
)

/**
 * The polynomial count sketch (scikit-learn PolynomialCountSketch): the tensor sketch of x' ⊗ ... ⊗ x' (p times) with
 * x' = [sqrt(gamma) x, sqrt(c_0)], whose inner products estimate the polynomial kernel (gamma x . y + c_0)^p.
 *
 * @param options - Degree, gamma, c_0, sketch size and the input dimension and key, or the hashes `[p, d (+ 1)]`.
 * @returns The feature map with its hashes.
 * @throws MdmlError `value` for a degree, sketch size or input dimension that is not a positive integer, a gamma that is
 *   not positive and finite, or a coef0 that is negative or not finite.
 * @concept random-features
 * @reference N. Pham and R. Pagh, Fast and scalable polynomial kernels via explicit feature maps (KDD 2013).
 * @example
 * ```ts
 * import { polynomialCountSketch } from 'mdml-engine/kernels'
 * import { key } from 'mdml-engine/random'
 * polynomialCountSketch({ inputDim: 3, degree: 2, nComponents: 16, key: key(0) }).transform([[1, 2, 3]]).shape // => [1, 16]
 * ```
 */
export function polynomialCountSketch(options: PolynomialCountSketchOptions): PolynomialCountSketch {
  const degree = positiveInteger('polynomialCountSketch', 'degree', options.degree ?? 2)
  const g = finiteAbove('polynomialCountSketch', 'gamma', options.gamma ?? 1, 0, true)
  const c0 = finiteAbove('polynomialCountSketch', 'coef0', options.coef0 ?? 0, 0, false)
  const n = positiveInteger('polynomialCountSketch', 'nComponents', options.nComponents ?? 100)
  if ('inputDim' in options) positiveInteger('polynomialCountSketch', 'inputDim', options.inputDim)
  const sketcher =
    'indexHash' in options
      ? tensorSketch({ indexHash: options.indexHash, bitHash: options.bitHash, nComponents: n })
      : tensorSketch({
          ...(options.key === undefined ? {} : { key: options.key }),
          ...(options.ctx === undefined ? {} : { ctx: options.ctx }),
          inputDims: Array.from({ length: degree }, () => options.inputDim + (c0 === 0 ? 0 : 1)),
          nComponents: n,
        })
  return {
    name: 'polynomial-count-sketch',
    indexHash: sketcher.indexHash,
    bitHash: sketcher.bitHash,
    transform(x: TensorLike): Value {
      const { rows, single } = rowsOf(x, 'polynomialCountSketch')
      const scaled = mul(rows, Math.sqrt(g))
      const lifted = c0 === 0 ? scaled : concat([scaled, mul(ones([rows.shape[0] ?? 0, 1]), Math.sqrt(c0))], 1)
      return finish(sketcher.sketch(...Array.from({ length: sketcher.indexHash.length }, () => lifted)), single)
    },
  }
}

/**
 * The kernel matrix implied by a feature map, Z_x Z_y^T.
 *
 * @param map - The feature map.
 * @param x - Points `[m, d]`.
 * @param y - Points `[n, d]` (default x).
 * @returns `[m, n]`.
 * @concept random-features
 * @example
 * ```ts
 * import { additiveChi2Sampler, approximateGram } from 'mdml-engine/kernels'
 * approximateGram(additiveChi2Sampler(), [[1, 2], [2, 1]]).shape // => [2, 2]
 * ```
 */
export function approximateGram(map: FeatureMap, x: TensorLike, y?: TensorLike): Value {
  const zx = map.transform(asRows(x, 'approximateGram').rows)
  const zy = y === undefined ? zx : map.transform(asRows(y, 'approximateGram').rows)
  return matmul(zx, swapaxes(zy, 0, 1))
}

/** How far an approximate kernel matrix is from the exact one. */
export interface ApproximationError {
  /** |K - K~|_F. */
  readonly frobenius: number
  /** |K - K~|_F / |K|_F. */
  readonly relative: number
  /** The largest singular value of K - K~. */
  readonly spectral: number
  /** max_ij |K_ij - K~_ij|. */
  readonly max: number
}

/**
 * Errors of a kernel approximation: Frobenius (absolute and relative), spectral and entrywise maximum norms of the
 * difference, the quantities in which random-feature and Nyström bounds are stated.
 *
 * @param exact - The exact kernel matrix.
 * @param approximate - Its approximation (for example from {@link approximateGram}).
 * @returns The errors.
 * @throws MdmlError `shape` when the shapes differ.
 * @concept random-features
 * @example
 * ```ts
 * import { approximationError } from 'mdml-engine/kernels'
 * Math.round(approximationError([[2, 0], [0, 2]], [[1, 0], [0, 2]]).relative * 1e9) / 1e9 // => 0.353553391
 * ```
 */
export function approximationError(exact: TensorLike, approximate: TensorLike): ApproximationError {
  const [a, b] = promoteFloat(exact, approximate) as [Value, Value]
  if (a.shape.join(',') !== b.shape.join(','))
    throw new MdmlError('shape', 'approximationError: the matrices must have the same shape')
  const diff = sub(a, b)
  const frobenius = Math.sqrt(item(sum(square(diff))))
  const scale = Math.sqrt(item(sum(square(a))))
  return {
    frobenius,
    relative: scale === 0 ? (frobenius === 0 ? 0 : Number.POSITIVE_INFINITY) : frobenius / scale,
    spectral: toFlat(svdvals(diff))[0] ?? 0,
    max: item(max(abs(diff))),
  }
}

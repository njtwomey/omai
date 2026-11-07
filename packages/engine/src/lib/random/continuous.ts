/**
 * Continuous random samplers that take a key first, JAX style: uniform and normal draws, the truncated normal (Robert's
 * rejection samplers), exponential, Laplace, logistic, Gumbel, Cauchy, gamma (Marsaglia-Tsang), beta, chi-square,
 * Student-t, Dirichlet, multivariate normal and von Mises, plus `uniformStream` for code that needs an unknown number
 * of draws.
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  matmul,
  type Shape,
  sizeOf,
  type Tensor,
  type TensorLike,
  toVector,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { bitsBlock } from './_bits.ts'
import { assertPositive, assertScale } from './_check.ts'
import { unitDouble, unitDoubles } from './_unit.ts'
import { gammaOne, logGammaOne, truncatedOne } from './_variates.ts'
import { foldIn, type Key, split } from './key.ts'

function fill(shape: Shape, draw: (i: number) => number): Tensor<'float64'> {
  const out = new Float64Array(sizeOf(shape))
  for (let i = 0; i < out.length; i++) out[i] = draw(i)
  return fromStorage(out, [...shape], 'float64')
}

/**
 * Uniform samples in `[low, high)` (like `jax.random.uniform`).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param options - `low` (default 0) and `high` (default 1).
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` unless `low` and `high` are finite with `low <= high`.
 * @concept counter-based-rng
 * @index function Uniform samples from a key and a shape.
 * @example
 * ```ts
 * import { key, uniform } from 'mdml-engine/random'
 * uniform(key(0), [1000]).data.every((u) => u >= 0 && u < 1) // => true
 * ```
 */
export function uniform(
  k: Key,
  shape: Shape,
  options: { readonly low?: number; readonly high?: number } = {},
): Tensor<'float64'> {
  const low = options.low ?? 0
  const high = options.high ?? 1
  if (!(Number.isFinite(low) && Number.isFinite(high) && high >= low)) {
    throw new MdmlError('value', `uniform: need finite low <= high, got [${low}, ${high}]`)
  }
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => low + (high - low) * (u[i] ?? 0))
}

/**
 * An endless stream of uniform draws in `[0, 1)` from one key, for code that needs an unknown number of draws
 * (rejection samplers, random walks, procedural generators). The draws come in blocks: block `b` is
 * `uniform(foldIn(k, b), [block])`, so the stream is a pure function of the key and the block size. A stream can start
 * at any draw (`from`), which is how a steppable sampler that keeps only the number of draws it has used resumes the
 * same stream after a checkpoint.
 *
 * @param k - The random key.
 * @param block - Draws per block (default 256); the same key and block size give the same stream.
 * @param from - The index of the first draw to return (default 0): the stream is the tail of the one from 0.
 * @returns A function returning the next draw on each call.
 * @concept counter-based-rng
 * @example
 * ```ts
 * import { foldIn, key, uniform, uniformStream } from 'mdml-engine/random'
 * const next = uniformStream(key(0), 4)
 * const draws = [next(), next(), next(), next(), next()]
 * draws[4] === uniform(foldIn(key(0), 1), [4]).data[0] // => true
 * draws.every((u) => u >= 0 && u < 1) // => true
 * draws.push(next())
 * uniformStream(key(0), 4, 5)() === draws[5] // => true
 * ```
 */
export function uniformStream(k: Key, block = 256, from = 0): () => number {
  if (!(Number.isInteger(block) && block >= 1))
    throw new MdmlError('value', 'uniformStream: block must be a positive integer')
  if (!(Number.isInteger(from) && from >= 0))
    throw new MdmlError('value', 'uniformStream: from must be a non-negative integer')
  // Block b is unitDoubles(foldIn(k, b), block), whose draws 2j and 2j + 1 come from Philox block j alone: the stream
  // computes those two on demand, so a sampler that uses a handful of draws pays for a handful of Philox blocks.
  let b = Math.floor(from / block)
  let blockKey = foldIn(k, b)
  let i = from - b * block
  let pair = -1
  let even = 0
  let odd = 0
  return () => {
    if (i >= block) {
      b += 1
      blockKey = foldIn(k, b)
      i = 0
      pair = -1
    }
    const j = Math.floor(i / 2)
    if (j !== pair) {
      const w = bitsBlock(blockKey.words, j)
      even = unitDouble(w[0], w[1])
      odd = unitDouble(w[2], w[3])
      pair = j
    }
    return i++ % 2 === 0 ? even : odd
  }
}

/**
 * Normal samples by the Box-Muller transform (like `jax.random.normal`).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param options - `loc` (default 0) and `scale` (default 1).
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite `scale`.
 * @concept normal-distribution
 * @index function Normal samples from a key and a shape, like jax.random.normal.
 * @example
 * ```ts
 * import { key, normal } from 'mdml-engine/random'
 * const x = normal(key(1), [20000])
 * Math.abs(x.data.reduce((a, b) => a + b, 0) / 20000) < 0.03 // => true
 * ```
 */
export function normal(
  k: Key,
  shape: Shape,
  options: { readonly loc?: number; readonly scale?: number } = {},
): Tensor<'float64'> {
  const loc = options.loc ?? 0
  const scale = options.scale ?? 1
  assertScale('normal', 'scale', scale)
  const n = sizeOf(shape)
  const u = unitDoubles(k, 2 * Math.ceil(n / 2))
  const out = new Float64Array(n)
  for (let i = 0; i < n; i += 2) {
    // 1 - u lies in (0, 1], so the logarithm is finite.
    const r = Math.sqrt(-2 * Math.log(1 - (u[i] ?? 0)))
    const theta = 2 * Math.PI * (u[i + 1] ?? 0)
    out[i] = loc + scale * r * Math.cos(theta)
    if (i + 1 < n) out[i + 1] = loc + scale * r * Math.sin(theta)
  }
  return fromStorage(out, [...shape], 'float64')
}

/**
 * Standard normal samples truncated to `[lower, upper]`, by Robert's (1995) rejection samplers: an exponential
 * proposal in the tails, a uniform proposal on short intervals and normal rejection on wide ones, so every interval,
 * however narrow or far out, costs a few uniforms per draw.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param lower - Lower bound (may be `-Infinity`).
 * @param upper - Upper bound (may be `Infinity`).
 * @returns A float64 tensor of standard truncated normals.
 * @throws MdmlError with code `value` unless `upper > lower`.
 * @reference Robert (1995), Simulation of truncated normal variables, Statistics and Computing 5.
 * @example
 * ```ts
 * import { key, truncatedNormal } from 'mdml-engine/random'
 * truncatedNormal(key(0), [100], 1, 2).data.every((v) => v >= 1 && v <= 2) // => true
 * truncatedNormal(key(0), [100], 8, Infinity).data.every((v) => v >= 8) // => true
 * ```
 */
export function truncatedNormal(k: Key, shape: Shape, lower: number, upper: number): Tensor<'float64'> {
  if (!(upper > lower)) throw new MdmlError('value', 'truncatedNormal: need upper > lower')
  const next = uniformStream(k)
  return fill(shape, () => truncatedOne(lower, upper, next))
}

/**
 * Exponential samples with a given rate.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param rate - The rate (default 1).
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` unless `rate` is finite and positive.
 * @example
 * ```ts
 * import { key, exponential } from 'mdml-engine/random'
 * exponential(key(0), [5]).data.every((v) => v >= 0) // => true
 * ```
 */
export function exponential(k: Key, shape: Shape, rate = 1): Tensor<'float64'> {
  assertPositive('exponential', 'rate', rate)
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => -Math.log(1 - (u[i] ?? 0)) / rate)
}

/**
 * Laplace samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param loc - Location.
 * @param scale - Scale.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite `scale`.
 * @example
 * ```ts
 * import { key, laplace } from 'mdml-engine/random'
 * laplace(key(0), [3]).shape // => [3]
 * ```
 */
export function laplace(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  assertScale('laplace', 'scale', scale)
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => {
    // Shift by half an ulp of the grid so v lies in (-0.5, 0.5) and the logarithm is finite.
    const v = (u[i] ?? 0) - 0.5 + 2 ** -54
    return loc - scale * Math.sign(v) * Math.log(1 - 2 * Math.abs(v))
  })
}

/**
 * Logistic samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param loc - Location.
 * @param scale - Scale.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite `scale`.
 * @example
 * ```ts
 * import { key, logistic } from 'mdml-engine/random'
 * logistic(key(0), [3]).shape // => [3]
 * ```
 */
export function logistic(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  assertScale('logistic', 'scale', scale)
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => {
    const v = Math.min(Math.max(u[i] ?? 0, 1e-300), 1 - 1e-16)
    return loc + scale * Math.log(v / (1 - v))
  })
}

/**
 * Gumbel samples (used for the Gumbel-max trick).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param loc - Location.
 * @param scale - Scale.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite `scale`.
 * @example
 * ```ts
 * import { key, gumbel } from 'mdml-engine/random'
 * gumbel(key(0), [3]).shape // => [3]
 * ```
 */
export function gumbel(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  assertScale('gumbel', 'scale', scale)
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => loc - scale * Math.log(-Math.log(Math.max(u[i] ?? 0, 1e-300))))
}

/**
 * Cauchy samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param loc - Location.
 * @param scale - Scale.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite `scale`.
 * @example
 * ```ts
 * import { key, cauchy } from 'mdml-engine/random'
 * cauchy(key(0), [3]).shape // => [3]
 * ```
 */
export function cauchy(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  assertScale('cauchy', 'scale', scale)
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => loc + scale * Math.tan(Math.PI * ((u[i] ?? 0) - 0.5)))
}

/**
 * Gamma samples with shape `alpha` and scale 1 (Marsaglia-Tsang).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param alpha - The shape parameter (positive).
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` for non-positive `alpha`.
 * @example
 * ```ts
 * import { key, gamma } from 'mdml-engine/random'
 * gamma(key(0), [4], 2).data.every((v) => v > 0) // => true
 * ```
 */
export function gamma(k: Key, shape: Shape, alpha: number): Tensor<'float64'> {
  if (!(alpha > 0)) throw new MdmlError('value', `gamma: alpha must be positive, got ${alpha}`)
  const next = uniformStream(k)
  return fill(shape, () => gammaOne(alpha, next))
}

/**
 * One gamma variate (shape `alpha_i`, scale 1) for each entry of `alphas`, each from its own stream `foldIn(k, i)`, so
 * the draw for entry i does not depend on the others (the samplers of batched gamma, beta and Dirichlet distributions).
 *
 * @param k - The random key.
 * @param alphas - Shape parameters (positive).
 * @returns The variates.
 * @throws MdmlError with code `value` for a non-positive shape.
 * @example
 * ```ts
 * import { key, gammaVariates } from 'mdml-engine/random'
 * gammaVariates(key(0), [0.5, 2, 30]).every((v) => v > 0) // => true
 * ```
 */
export function gammaVariates(k: Key, alphas: ArrayLike<number>): Float64Array {
  const out = new Float64Array(alphas.length)
  for (let i = 0; i < alphas.length; i++) {
    const alpha = alphas[i] ?? Number.NaN
    if (!(alpha > 0)) throw new MdmlError('value', `gammaVariates: alpha must be positive, got ${alpha}`)
    out[i] = gammaOne(alpha, uniformStream(foldIn(k, i)))
  }
  return out
}

/**
 * Beta samples, as `X / (X + Y)` for independent gamma variates, formed from their logarithms so that small shapes
 * (where both variates underflow) still give the right mixture of values near 0 and 1.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param a - First shape parameter (positive).
 * @param b - Second shape parameter (positive).
 * @returns A float64 tensor in [0, 1].
 * @throws MdmlError with code `value` for non-positive shapes.
 * @example
 * ```ts
 * import { key, beta } from 'mdml-engine/random'
 * beta(key(0), [4], 2, 3).data.every((v) => v >= 0 && v <= 1) // => true
 * ```
 */
export function beta(k: Key, shape: Shape, a: number, b: number): Tensor<'float64'> {
  if (!(a > 0 && b > 0)) throw new MdmlError('value', `beta: shapes must be positive, got ${a} and ${b}`)
  const next = uniformStream(k)
  return fill(shape, () => {
    const lx = logGammaOne(a, next)
    const ly = logGammaOne(b, next)
    return 1 / (1 + Math.exp(ly - lx))
  })
}

/**
 * Chi-square samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param df - Degrees of freedom.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` unless `df` is finite and positive.
 * @example
 * ```ts
 * import { key, chisquare } from 'mdml-engine/random'
 * chisquare(key(0), [3], 2).shape // => [3]
 * ```
 */
export function chisquare(k: Key, shape: Shape, df: number): Tensor<'float64'> {
  assertPositive('chisquare', 'df', df)
  const g = gamma(k, shape, df / 2)
  return fill(shape, (i) => 2 * (g.data[i] ?? 0))
}

/**
 * Student-t samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param df - Degrees of freedom.
 * @returns A float64 tensor.
 * @throws MdmlError with code `value` unless `df` is finite and positive.
 * @example
 * ```ts
 * import { key, studentT } from 'mdml-engine/random'
 * studentT(key(0), [3], 5).shape // => [3]
 * ```
 */
export function studentT(k: Key, shape: Shape, df: number): Tensor<'float64'> {
  assertPositive('studentT', 'df', df)
  const [kz, kc] = split(k) as [Key, Key]
  const z = normal(kz, shape)
  const c = chisquare(kc, shape, df)
  return fill(shape, (i) => (z.data[i] ?? 0) / Math.sqrt((c.data[i] ?? 1) / df))
}

/**
 * Dirichlet samples: each row sums to one (normalised gamma variates, formed from their logarithms so that small
 * concentrations do not underflow). Takes tensors (migration, D-009: `alpha` was `readonly number[]`; nested arrays
 * still work).
 *
 * @param k - The random key.
 * @param alpha - Concentrations (a vector of length K, positive): a tensor, a typed array or a nested array.
 * @param n - Number of samples.
 * @returns A float64 tensor of shape `[n, K]`.
 * @throws MdmlError with code `value` for a non-positive concentration, `shape` unless `alpha` is a vector and
 * `not-differentiable` for a traced `alpha` (a sample is not differentiated through its concentrations).
 * @example
 * ```ts
 * import { key, dirichlet } from 'mdml-engine/random'
 * Math.abs(Array.from(dirichlet(key(0), [1, 2, 3], 1).data).reduce((a, b) => a + b) - 1) < 1e-12 // => true
 * ```
 */
export function dirichlet(k: Key, alpha: TensorLike, n: number): Tensor<'float64'> {
  const a = toVector(alpha, 'dirichlet')
  if (a.some((x) => !(x > 0))) throw new MdmlError('value', 'dirichlet: concentrations must be positive')
  const K = a.length
  const next = uniformStream(k)
  const out = new Float64Array(n * K)
  for (let r = 0; r < n; r++) {
    let top = -Infinity
    for (let j = 0; j < K; j++) {
      const g = logGammaOne(a[j] ?? 1, next)
      out[r * K + j] = g
      if (g > top) top = g
    }
    let total = 0
    for (let j = 0; j < K; j++) {
      const e = Math.exp((out[r * K + j] ?? 0) - top)
      out[r * K + j] = e
      total += e
    }
    for (let j = 0; j < K; j++) out[r * K + j] = (out[r * K + j] ?? 0) / total
  }
  return fromStorage(out, [n, K], 'float64')
}

/**
 * Multivariate normal samples `mean + L z`, given the lower Cholesky factor `L` of the covariance (factorise with
 * `numerics/linalg.cholesky`).
 *
 * @param k - The random key.
 * @param mean - The mean, length d.
 * @param scaleTril - The lower-triangular factor, `[d, d]`.
 * @param n - Number of samples.
 * @returns A value of shape `[n, d]`.
 * @example
 * ```ts
 * import { key, multivariateNormal } from 'mdml-engine/random'
 * multivariateNormal(key(0), [0, 0], [[1, 0], [0.5, 1]], 4).shape // => [4, 2]
 * ```
 */
export function multivariateNormal(k: Key, mean: TensorLike, scaleTril: TensorLike, n: number): Value {
  const m = asTensor(mean)
  const d = m.shape[0] ?? 0
  const z = normal(k, [n, d])
  // Rows are z L^T, so each row has covariance L L^T.
  return add(matmul(z, transpose(asTensor(scaleTril))), m)
}

/**
 * Von Mises samples on the circle (Best and Fisher, 1979).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param mu - Mean direction.
 * @param kappa - Concentration (non-negative).
 * @returns A float64 tensor of angles in (-pi, pi].
 * @throws MdmlError with code `value` for a negative or non-finite `kappa`.
 * @example
 * ```ts
 * import { key, vonMises } from 'mdml-engine/random'
 * vonMises(key(0), [10], 0, 4).data.every((v) => Math.abs(v) <= Math.PI) // => true
 * ```
 */
export function vonMises(k: Key, shape: Shape, mu: number, kappa: number): Tensor<'float64'> {
  assertScale('vonMises', 'kappa', kappa)
  const next = uniformStream(k)
  if (kappa < 1e-8) return fill(shape, () => Math.PI * (2 * next() - 1))
  const tau = 1 + Math.sqrt(1 + 4 * kappa * kappa)
  const rho = (tau - Math.sqrt(2 * tau)) / (2 * kappa)
  const r = (1 + rho * rho) / (2 * rho)
  return fill(shape, () => {
    for (;;) {
      const z = Math.cos(Math.PI * next())
      const f = (1 + r * z) / (r + z)
      const c = kappa * (r - f)
      const u2 = next()
      if (c * (2 - c) - u2 > 0 || Math.log(c / u2) + 1 - c >= 0) {
        const theta = mu + Math.sign(next() - 0.5) * Math.acos(f)
        return Math.atan2(Math.sin(theta), Math.cos(theta))
      }
    }
  })
}

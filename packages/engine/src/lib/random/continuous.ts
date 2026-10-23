import { WhiteboxError } from '../base/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  matmul,
  type Shape,
  sizeOf,
  type Tensor,
  type TensorLike,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { normalFrom, uniformStream, unitDoubles } from './_unit.ts'
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
 * @concept counter-based-rng
 * @example
 * ```ts
 * import { key, uniform } from 'whitebox-engine/random'
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
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => low + (high - low) * (u[i] ?? 0))
}

/**
 * Normal samples by the Box-Muller transform (like `jax.random.normal`).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param options - `loc` (default 0) and `scale` (default 1).
 * @returns A float64 tensor.
 * @concept normal-distribution
 * @example
 * ```ts
 * import { key, normal } from 'whitebox-engine/random'
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
 * Normal samples truncated to `[lower, upper]` (in standard units), by inversion of the normal CDF restricted to the
 * interval (accurate in the tails through log-space bounds).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param lower - Lower bound (may be `-Infinity`).
 * @param upper - Upper bound (may be `Infinity`).
 * @returns A float64 tensor of standard truncated normals.
 * @example
 * ```ts
 * import { key, truncatedNormal } from 'whitebox-engine/random'
 * truncatedNormal(key(0), [100], 1, 2).data.every((v) => v >= 1 && v <= 2) // => true
 * ```
 */
export function truncatedNormal(k: Key, shape: Shape, lower: number, upper: number): Tensor<'float64'> {
  if (!(upper > lower)) throw new WhiteboxError('value', 'truncatedNormal: need upper > lower')
  const next = uniformStream(k, foldIn)
  // Rejection from a uniform on finite intervals is slow in the tails; use the exponential proposal of Robert (1995)
  // for one-sided tails and normal rejection otherwise.
  const draw = (): number => {
    if (lower >= 0.5) {
      const a = (lower + Math.sqrt(lower * lower + 4)) / 2
      for (;;) {
        const z = lower - Math.log(1 - next()) / a
        if (z <= upper && next() <= Math.exp(-((z - a) ** 2) / 2)) return z
      }
    }
    if (upper <= -0.5) {
      const a = (-upper + Math.sqrt(upper * upper + 4)) / 2
      for (;;) {
        const z = -upper - Math.log(1 - next()) / a
        if (-z >= lower && next() <= Math.exp(-((z - a) ** 2) / 2)) return -z
      }
    }
    for (;;) {
      const z = normalFrom(next)
      if (z >= lower && z <= upper) return z
    }
  }
  return fill(shape, () => draw())
}

/**
 * Exponential samples with a given rate.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param rate - The rate (default 1).
 * @returns A float64 tensor.
 * @example
 * ```ts
 * import { key, exponential } from 'whitebox-engine/random'
 * exponential(key(0), [5]).data.every((v) => v >= 0) // => true
 * ```
 */
export function exponential(k: Key, shape: Shape, rate = 1): Tensor<'float64'> {
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
 * @example
 * ```ts
 * import { key, laplace } from 'whitebox-engine/random'
 * laplace(key(0), [3]).shape // => [3]
 * ```
 */
export function laplace(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => {
    const v = (u[i] ?? 0) - 0.5
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
 * @example
 * ```ts
 * import { key, logistic } from 'whitebox-engine/random'
 * logistic(key(0), [3]).shape // => [3]
 * ```
 */
export function logistic(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
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
 * @example
 * ```ts
 * import { key, gumbel } from 'whitebox-engine/random'
 * gumbel(key(0), [3]).shape // => [3]
 * ```
 */
export function gumbel(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
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
 * @example
 * ```ts
 * import { key, cauchy } from 'whitebox-engine/random'
 * cauchy(key(0), [3]).shape // => [3]
 * ```
 */
export function cauchy(k: Key, shape: Shape, loc = 0, scale = 1): Tensor<'float64'> {
  const u = unitDoubles(k, sizeOf(shape))
  return fill(shape, (i) => loc + scale * Math.tan(Math.PI * ((u[i] ?? 0) - 0.5)))
}

function gammaOne(alpha: number, next: () => number): number {
  // Marsaglia and Tsang (2000); for alpha < 1 boost with U^(1/alpha), done in log space to avoid underflow.
  if (alpha < 1) return Math.exp(Math.log(gammaOne(alpha + 1, next)) + Math.log(next() || Number.MIN_VALUE) / alpha)
  const d = alpha - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    let x: number
    let v: number
    do {
      x = normalFrom(next)
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = next()
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

/**
 * Gamma samples with shape `alpha` and scale 1 (Marsaglia-Tsang).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param alpha - The shape parameter (positive).
 * @returns A float64 tensor.
 * @throws WhiteboxError with code `value` for non-positive `alpha`.
 * @example
 * ```ts
 * import { key, gamma } from 'whitebox-engine/random'
 * gamma(key(0), [4], 2).data.every((v) => v > 0) // => true
 * ```
 */
export function gamma(k: Key, shape: Shape, alpha: number): Tensor<'float64'> {
  if (!(alpha > 0)) throw new WhiteboxError('value', `gamma: alpha must be positive, got ${alpha}`)
  const next = uniformStream(k, foldIn)
  return fill(shape, () => gammaOne(alpha, next))
}

/**
 * One gamma variate (shape `alpha_i`, scale 1) for each entry of `alphas`, each from its own stream `foldIn(k, i)`, so
 * the draw for entry i does not depend on the others (the samplers of batched gamma, beta and Dirichlet distributions).
 *
 * @param k - The random key.
 * @param alphas - Shape parameters (positive).
 * @returns The variates.
 * @throws WhiteboxError with code `value` for a non-positive shape.
 * @example
 * ```ts
 * import { key, gammaVariates } from 'whitebox-engine/random'
 * gammaVariates(key(0), [0.5, 2, 30]).every((v) => v > 0) // => true
 * ```
 */
export function gammaVariates(k: Key, alphas: ArrayLike<number>): Float64Array {
  const out = new Float64Array(alphas.length)
  for (let i = 0; i < alphas.length; i++) {
    const alpha = alphas[i] ?? Number.NaN
    if (!(alpha > 0)) throw new WhiteboxError('value', `gammaVariates: alpha must be positive, got ${alpha}`)
    out[i] = gammaOne(alpha, uniformStream(foldIn(k, i), foldIn))
  }
  return out
}

/**
 * Beta samples, as `X / (X + Y)` for gamma variates (computed in log space when both shapes are small).
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param a - First shape parameter.
 * @param b - Second shape parameter.
 * @returns A float64 tensor in [0, 1].
 * @example
 * ```ts
 * import { key, beta } from 'whitebox-engine/random'
 * beta(key(0), [4], 2, 3).data.every((v) => v >= 0 && v <= 1) // => true
 * ```
 */
export function beta(k: Key, shape: Shape, a: number, b: number): Tensor<'float64'> {
  const [ka, kb] = split(k) as [Key, Key]
  const x = gamma(ka, shape, a)
  const y = gamma(kb, shape, b)
  return fill(shape, (i) => {
    const xi = x.data[i] ?? 0
    const yi = y.data[i] ?? 0
    return xi + yi === 0 ? (a >= b ? 1 : 0) : xi / (xi + yi)
  })
}

/**
 * Chi-square samples.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param df - Degrees of freedom.
 * @returns A float64 tensor.
 * @example
 * ```ts
 * import { key, chisquare } from 'whitebox-engine/random'
 * chisquare(key(0), [3], 2).shape // => [3]
 * ```
 */
export function chisquare(k: Key, shape: Shape, df: number): Tensor<'float64'> {
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
 * @example
 * ```ts
 * import { key, studentT } from 'whitebox-engine/random'
 * studentT(key(0), [3], 5).shape // => [3]
 * ```
 */
export function studentT(k: Key, shape: Shape, df: number): Tensor<'float64'> {
  const [kz, kc] = split(k) as [Key, Key]
  const z = normal(kz, shape)
  const c = chisquare(kc, shape, df)
  return fill(shape, (i) => (z.data[i] ?? 0) / Math.sqrt((c.data[i] ?? 1) / df))
}

/**
 * Dirichlet samples: each row sums to one (normalised gammas, computed in log space for small concentrations).
 *
 * @param k - The random key.
 * @param alpha - Concentrations (length K).
 * @param n - Number of samples.
 * @returns A float64 tensor of shape `[n, K]`.
 * @example
 * ```ts
 * import { key, dirichlet } from 'whitebox-engine/random'
 * Math.abs(Array.from(dirichlet(key(0), [1, 2, 3], 1).data).reduce((a, b) => a + b) - 1) < 1e-12 // => true
 * ```
 */
export function dirichlet(k: Key, alpha: readonly number[], n: number): Tensor<'float64'> {
  const K = alpha.length
  const next = uniformStream(k, foldIn)
  const out = new Float64Array(n * K)
  for (let r = 0; r < n; r++) {
    let total = 0
    for (let j = 0; j < K; j++) {
      const g = gammaOne(alpha[j] ?? 1, next)
      out[r * K + j] = g
      total += g
    }
    for (let j = 0; j < K; j++) out[r * K + j] = total > 0 ? (out[r * K + j] ?? 0) / total : 1 / K
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
 * import { key, multivariateNormal } from 'whitebox-engine/random'
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
 * @example
 * ```ts
 * import { key, vonMises } from 'whitebox-engine/random'
 * vonMises(key(0), [10], 0, 4).data.every((v) => Math.abs(v) <= Math.PI) // => true
 * ```
 */
export function vonMises(k: Key, shape: Shape, mu: number, kappa: number): Tensor<'float64'> {
  const next = uniformStream(k, foldIn)
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

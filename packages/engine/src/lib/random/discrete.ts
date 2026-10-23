import { WhiteboxError } from '../base/index.ts'
import { fromStorage, type Shape, sizeOf, type Tensor } from '../tensor/index.ts'
import { logGammaSmall, uniformStream, unitDoubles } from './_unit.ts'
import { foldIn, type Key } from './key.ts'

function ints(shape: Shape, draw: (i: number) => number): Tensor<'int32'> {
  const out = new Int32Array(sizeOf(shape))
  for (let i = 0; i < out.length; i++) out[i] = draw(i)
  return fromStorage(out, [...shape], 'int32')
}

/**
 * Uniform random integers in `[low, high)`. The bias from scaling a 53-bit uniform is below 2^-53 relative to the range.
 *
 * @param k - The random key.
 * @param shape - Output shape.
 * @param low - Smallest value.
 * @param high - Exclusive upper bound.
 * @returns An int32 tensor.
 * @throws WhiteboxError with code `value` unless `high > low`.
 * @example
 * ```ts
 * import { key, randint } from 'whitebox-engine/random'
 * randint(key(0), [5], 0, 3).data.every((v) => v >= 0 && v < 3) // => true
 * ```
 */
export function randint(k: Key, shape: Shape, low: number, high: number): Tensor<'int32'> {
  if (!(high > low)) throw new WhiteboxError('value', `randint: need high > low, got [${low}, ${high})`)
  const u = unitDoubles(k, sizeOf(shape))
  return ints(shape, (i) => low + Math.floor((u[i] ?? 0) * (high - low)))
}

/**
 * Bernoulli samples (1 with probability `p`).
 *
 * @param k - The random key.
 * @param p - Success probability in [0, 1].
 * @param shape - Output shape.
 * @returns A bool tensor.
 * @example
 * ```ts
 * import { key, bernoulli } from 'whitebox-engine/random'
 * Array.from(bernoulli(key(0), 1, [3]).data) // => [1, 1, 1]
 * ```
 */
export function bernoulli(k: Key, p: number, shape: Shape): Tensor<'bool'> {
  const u = unitDoubles(k, sizeOf(shape))
  const out = new Uint8Array(u.length)
  for (let i = 0; i < u.length; i++) out[i] = (u[i] ?? 0) < p ? 1 : 0
  return fromStorage(out, [...shape], 'bool')
}

function cumulative(weights: readonly number[], name: string): number[] {
  const total = weights.reduce((a, b) => a + b, 0)
  if (!(total > 0) || !Number.isFinite(total) || weights.some((w) => w < 0 || Number.isNaN(w))) {
    throw new WhiteboxError('value', `${name}: weights must be non-negative with a positive finite sum`)
  }
  let acc = 0
  return weights.map((w) => {
    acc += w / total
    return acc
  })
}

function invert(cdf: readonly number[], weights: readonly number[], u: number): number {
  let lo = 0
  let hi = cdf.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (u < (cdf[mid] ?? 1)) hi = mid
    else lo = mid + 1
  }
  // Never return a zero-weight category, even at the CDF boundaries.
  while (lo > 0 && (weights[lo] ?? 0) === 0) lo--
  while (lo < cdf.length - 1 && (weights[lo] ?? 0) === 0) lo++
  return lo
}

/**
 * Samples from a categorical distribution given probabilities or logits (by inverse CDF).
 *
 * @param k - The random key.
 * @param weights - Non-negative weights, or logits when `options.logits` is set.
 * @param shape - Output shape.
 * @param options - `logits`: treat `weights` as unnormalised log-probabilities.
 * @returns An int32 tensor of category indices.
 * @throws WhiteboxError with code `value` when the weights have no positive finite sum.
 * @example
 * ```ts
 * import { key, categorical } from 'whitebox-engine/random'
 * Array.from(categorical(key(0), [0, 1, 0], [4]).data) // => [1, 1, 1, 1]
 * ```
 */
export function categorical(
  k: Key,
  weights: readonly number[],
  shape: Shape,
  options: { readonly logits?: boolean } = {},
): Tensor<'int32'> {
  let w = weights
  if (options.logits === true) {
    let m = -Infinity
    for (const x of weights) if (x > m) m = x
    w = weights.map((x) => Math.exp(x - m))
  }
  const cdf = cumulative(w, 'categorical')
  const u = unitDoubles(k, sizeOf(shape))
  return ints(shape, (i) => invert(cdf, w, u[i] ?? 0))
}

/**
 * Multinomial counts: `n` draws into categories.
 *
 * @param k - The random key.
 * @param n - Number of draws.
 * @param probs - Category probabilities.
 * @param samples - Number of independent count vectors.
 * @returns An int32 tensor of shape `[samples, K]`, each row summing to `n`.
 * @example
 * ```ts
 * import { key, multinomial } from 'whitebox-engine/random'
 * Array.from(multinomial(key(0), 10, [0.5, 0.5], 1).data).reduce((a, b) => a + b) // => 10
 * ```
 */
export function multinomial(k: Key, n: number, probs: readonly number[], samples: number): Tensor<'int32'> {
  const K = probs.length
  const draws = categorical(k, probs, [samples, n])
  const out = new Int32Array(samples * K)
  for (let s = 0; s < samples; s++) {
    for (let j = 0; j < n; j++) {
      const idx = s * K + (draws.data[s * n + j] ?? 0)
      out[idx] = (out[idx] ?? 0) + 1
    }
  }
  return fromStorage(out, [samples, K], 'int32')
}

function poissonOne(lambda: number, next: () => number): number {
  if (lambda === 0) return 0
  if (lambda < 30) {
    // Knuth: multiply uniforms until the product drops below e^-lambda.
    const limit = Math.exp(-lambda)
    let p = 1
    let count = -1
    do {
      count += 1
      p *= next()
    } while (p > limit)
    return count
  }
  // Hormann's transformed rejection with squeeze (PTRS), exact for lambda >= 10.
  const slam = Math.sqrt(lambda)
  const loglam = Math.log(lambda)
  const b = 0.931 + 2.53 * slam
  const a = -0.059 + 0.02483 * b
  const invalpha = 1.1239 + 1.1328 / (b - 3.4)
  const vr = 0.9277 - 3.6224 / (b - 2)
  for (;;) {
    const u = next() - 0.5
    const v = next()
    const us = 0.5 - Math.abs(u)
    const kk = Math.floor(((2 * a) / us + b) * u + lambda + 0.43)
    if (us >= 0.07 && v <= vr) return kk
    if (kk < 0 || (us < 0.013 && v > us)) continue
    if (
      Math.log(v) + Math.log(invalpha) - Math.log(a / (us * us) + b) <=
      -lambda + kk * loglam - logGammaSmall(kk + 1)
    ) {
      return kk
    }
  }
}

/**
 * Poisson samples (Knuth's method for small rates, Hormann's PTRS for large ones).
 *
 * @param k - The random key.
 * @param lambda - The rate (non-negative).
 * @param shape - Output shape.
 * @returns An int32 tensor.
 * @throws WhiteboxError with code `value` for a negative rate.
 * @example
 * ```ts
 * import { key, poisson } from 'whitebox-engine/random'
 * poisson(key(0), 3, [5]).data.every((v) => v >= 0) // => true
 * ```
 */
export function poisson(k: Key, lambda: number, shape: Shape): Tensor<'int32'> {
  if (!(lambda >= 0)) throw new WhiteboxError('value', `poisson: rate must be non-negative, got ${lambda}`)
  const next = uniformStream(k, foldIn)
  return ints(shape, () => poissonOne(lambda, next))
}

/**
 * Binomial samples, exact for any `n`, by geometric waiting times (cost proportional to `n min(p, 1 - p)`).
 *
 * @param k - The random key.
 * @param n - Number of trials.
 * @param p - Success probability.
 * @param shape - Output shape.
 * @returns An int32 tensor.
 * @example
 * ```ts
 * import { key, binomial } from 'whitebox-engine/random'
 * binomial(key(0), 10, 0.3, [5]).data.every((v) => v >= 0 && v <= 10) // => true
 * ```
 */
export function binomial(k: Key, n: number, p: number, shape: Shape): Tensor<'int32'> {
  if (!(p >= 0 && p <= 1) || !Number.isInteger(n) || n < 0) {
    throw new WhiteboxError('value', 'binomial: need an integer n >= 0 and p in [0, 1]')
  }
  const next = uniformStream(k, foldIn)
  const q = Math.min(p, 1 - p)
  const draw = (): number => {
    if (q === 0) return 0
    const logq = Math.log(1 - q)
    let count = 0
    let position = 0
    for (;;) {
      position += Math.floor(Math.log(1 - next()) / logq) + 1
      if (position > n) break
      count += 1
    }
    return p <= 0.5 ? count : n - count
  }
  return ints(shape, draw)
}

/**
 * Geometric samples: the number of trials up to and including the first success.
 *
 * @param k - The random key.
 * @param p - Success probability in (0, 1].
 * @param shape - Output shape.
 * @returns An int32 tensor of values at least 1.
 * @example
 * ```ts
 * import { key, geometric } from 'whitebox-engine/random'
 * Array.from(geometric(key(0), 1, [3]).data) // => [1, 1, 1]
 * ```
 */
export function geometric(k: Key, p: number, shape: Shape): Tensor<'int32'> {
  if (!(p > 0 && p <= 1)) throw new WhiteboxError('value', `geometric: p must be in (0, 1], got ${p}`)
  const u = unitDoubles(k, sizeOf(shape))
  return ints(shape, (i) => (p === 1 ? 1 : Math.floor(Math.log(1 - (u[i] ?? 0)) / Math.log(1 - p)) + 1))
}

/** A Vose alias table for O(1) sampling from a fixed discrete distribution. */
export interface AliasTable {
  readonly probability: Float64Array
  readonly alias: Int32Array
}

/**
 * Build a Vose alias table.
 *
 * @param weights - Non-negative weights.
 * @returns The table.
 * @example
 * ```ts
 * import { aliasTable } from 'whitebox-engine/random'
 * aliasTable([1, 1]).alias.length // => 2
 * ```
 */
export function aliasTable(weights: readonly number[]): AliasTable {
  const n = weights.length
  cumulative(weights, 'aliasTable')
  const total = weights.reduce((a, b) => a + b, 0)
  const scaled = weights.map((w) => (w * n) / total)
  const probability = new Float64Array(n)
  const alias = new Int32Array(n)
  const small: number[] = []
  const large: number[] = []
  scaled.forEach((s, i) => {
    ;(s < 1 ? small : large).push(i)
  })
  while (small.length > 0 && large.length > 0) {
    const s = small.pop() as number
    const l = large.pop() as number
    probability[s] = scaled[s] ?? 0
    alias[s] = l
    scaled[l] = (scaled[l] ?? 0) + (scaled[s] ?? 0) - 1
    ;((scaled[l] ?? 0) < 1 ? small : large).push(l)
  }
  for (const i of [...small, ...large]) {
    probability[i] = 1
    alias[i] = i
  }
  return { probability, alias }
}

/**
 * Sample from an alias table.
 *
 * @param k - The random key.
 * @param table - The table.
 * @param shape - Output shape.
 * @returns An int32 tensor of category indices.
 * @example
 * ```ts
 * import { key, aliasTable, aliasSample } from 'whitebox-engine/random'
 * Array.from(aliasSample(key(0), aliasTable([0, 1]), [3]).data) // => [1, 1, 1]
 * ```
 */
export function aliasSample(k: Key, table: AliasTable, shape: Shape): Tensor<'int32'> {
  const n = table.probability.length
  const u = unitDoubles(k, 2 * sizeOf(shape))
  return ints(shape, (i) => {
    const column = Math.floor((u[2 * i] ?? 0) * n)
    return (u[2 * i + 1] ?? 0) < (table.probability[column] ?? 1) ? column : (table.alias[column] ?? column)
  })
}

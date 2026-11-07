/**
 * Discrete random samplers that take a key first: integers, Bernoulli, categorical (from probabilities or logits),
 * multinomial, Poisson and binomial (the same algorithms as numpy, so costs stay bounded for large counts) and
 * geometric draws, plus Vose alias tables for O(1) sampling from a fixed discrete distribution.
 */

import { MdmlError } from '../base/index.ts'
import { fromStorage, type Shape, sizeOf, type Tensor, type TensorLike } from '../tensor/index.ts'
import { vectorNumbers } from './_check.ts'
import { unitDoubles } from './_unit.ts'
import { binomialOne, poissonOne } from './_variates.ts'
import { uniformStream } from './continuous.ts'
import type { Key } from './key.ts'

const INT32_MAX = 2147483647

/** Fill an int32 tensor from a draw; a draw that does not fit throws instead of wrapping around. */
function ints(name: string, shape: Shape, draw: (i: number) => number): Tensor<'int32'> {
  const out = new Int32Array(sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    const v = draw(i)
    if (!(v <= INT32_MAX)) throw new MdmlError('value', `${name}: a draw of ${v} does not fit an int32`)
    out[i] = v
  }
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
 * @throws MdmlError with code `value` unless `high > low`.
 * @example
 * ```ts
 * import { key, randint } from 'mdml-engine/random'
 * randint(key(0), [5], 0, 3).data.every((v) => v >= 0 && v < 3) // => true
 * ```
 */
export function randint(k: Key, shape: Shape, low: number, high: number): Tensor<'int32'> {
  if (!(high > low)) throw new MdmlError('value', `randint: need high > low, got [${low}, ${high})`)
  const u = unitDoubles(k, sizeOf(shape))
  return ints('randint', shape, (i) => low + Math.floor((u[i] ?? 0) * (high - low)))
}

/**
 * Bernoulli samples (1 with probability `p`).
 *
 * @param k - The random key.
 * @param p - Success probability in [0, 1].
 * @param shape - Output shape.
 * @returns A bool tensor.
 * @throws MdmlError with code `value` unless `p` is in [0, 1].
 * @example
 * ```ts
 * import { key, bernoulli } from 'mdml-engine/random'
 * Array.from(bernoulli(key(0), 1, [3]).data) // => [1, 1, 1]
 * ```
 */
export function bernoulli(k: Key, p: number, shape: Shape): Tensor<'bool'> {
  if (!(p >= 0 && p <= 1)) throw new MdmlError('value', `bernoulli: p must be in [0, 1], got ${p}`)
  const u = unitDoubles(k, sizeOf(shape))
  const out = new Uint8Array(u.length)
  for (let i = 0; i < u.length; i++) out[i] = (u[i] ?? 0) < p ? 1 : 0
  return fromStorage(out, [...shape], 'bool')
}

function cumulative(weights: readonly number[], name: string): number[] {
  const total = weights.reduce((a, b) => a + b, 0)
  if (!(total > 0) || !Number.isFinite(total) || weights.some((w) => w < 0 || Number.isNaN(w))) {
    throw new MdmlError('value', `${name}: weights must be non-negative with a positive finite sum`)
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
 * Samples from a categorical distribution given probabilities or logits (by inverse CDF). Takes tensors (migration,
 * D-009: `weights` was `readonly number[]`; nested arrays still work).
 *
 * @param k - The random key.
 * @param weights - Non-negative weights, or logits when `options.logits` is set: a vector (a tensor, a typed array or
 * a nested array).
 * @param shape - Output shape.
 * @param options - `logits`: treat `weights` as unnormalised log-probabilities.
 * @returns An int32 tensor of category indices.
 * @throws MdmlError with code `value` when the weights have no positive finite sum, `shape` unless `weights` is a
 * vector and `not-differentiable` for traced weights.
 * @index function Samples from a categorical distribution given probabilities or logits.
 * @example
 * ```ts
 * import { key, categorical } from 'mdml-engine/random'
 * Array.from(categorical(key(0), [0, 1, 0], [4]).data) // => [1, 1, 1, 1]
 * ```
 */
export function categorical(
  k: Key,
  weights: TensorLike,
  shape: Shape,
  options: { readonly logits?: boolean } = {},
): Tensor<'int32'> {
  const given = vectorNumbers('categorical', 'the weights', weights)
  let w = given
  if (options.logits === true) {
    let m = -Infinity
    for (const x of given) if (x > m) m = x
    w = given.map((x) => Math.exp(x - m))
  }
  const cdf = cumulative(w, 'categorical')
  const u = unitDoubles(k, sizeOf(shape))
  return ints('categorical', shape, (i) => invert(cdf, w, u[i] ?? 0))
}

/**
 * Multinomial counts: `n` draws into categories, by conditional binomials (category j takes a binomial share of the
 * draws the earlier categories left), so the cost does not grow with `n`. Takes tensors (migration, D-009: `probs` was
 * `readonly number[]`; nested arrays still work).
 *
 * @param k - The random key.
 * @param n - Number of draws (a non-negative integer).
 * @param probs - Category probabilities (a vector of non-negative weights, normalised): a tensor, a typed array or a
 * nested array.
 * @param samples - Number of independent count vectors.
 * @returns An int32 tensor of shape `[samples, K]`, each row summing to `n`.
 * @throws MdmlError with code `value` for a bad `n` or weights without a positive finite sum, `shape` unless `probs`
 * is a vector and `not-differentiable` for traced `probs`.
 * @example
 * ```ts
 * import { key, multinomial } from 'mdml-engine/random'
 * Array.from(multinomial(key(0), 10, [0.5, 0.5], 1).data).reduce((a, b) => a + b) // => 10
 * ```
 */
export function multinomial(k: Key, n: number, probs: TensorLike, samples: number): Tensor<'int32'> {
  if (!Number.isInteger(n) || n < 0) throw new MdmlError('value', `multinomial: n must be a non-negative integer`)
  const p = vectorNumbers('multinomial', 'the probabilities', probs)
  const cdf = cumulative(p, 'multinomial')
  const K = p.length
  const next = uniformStream(k)
  const out = new Int32Array(samples * K)
  for (let s = 0; s < samples; s++) {
    let left = n
    let before = 0
    for (let j = 0; j < K - 1 && left > 0; j++) {
      const pj = (cdf[j] ?? 1) - before
      const rest = 1 - before
      const share = rest > 0 ? Math.min(1, Math.max(0, pj / rest)) : 0
      const c = binomialOne(left, share, next)
      out[s * K + j] = c
      left -= c
      before = cdf[j] ?? 1
    }
    out[s * K + K - 1] = (out[s * K + K - 1] ?? 0) + left
  }
  return fromStorage(out, [samples, K], 'int32')
}

/**
 * Poisson samples (Knuth's method for rates below 30, Hormann's PTRS above, as numpy).
 *
 * @param k - The random key.
 * @param lambda - The rate (non-negative).
 * @param shape - Output shape.
 * @returns An int32 tensor.
 * @throws MdmlError with code `value` for a negative or non-finite rate, or a draw beyond the int32 range.
 * @reference Hormann (1993), The transformed rejection method for generating Poisson random variables.
 * @example
 * ```ts
 * import { key, poisson } from 'mdml-engine/random'
 * poisson(key(0), 3, [5]).data.every((v) => v >= 0) // => true
 * ```
 */
export function poisson(k: Key, lambda: number, shape: Shape): Tensor<'int32'> {
  if (!(lambda >= 0 && lambda < Infinity)) {
    throw new MdmlError('value', `poisson: rate must be finite and non-negative, got ${lambda}`)
  }
  const next = uniformStream(k)
  return ints('poisson', shape, () => poissonOne(lambda, next))
}

/**
 * Binomial samples, exact for any `n`: geometric waiting times between successes when `n min(p, 1 - p) < 30`, and
 * Kachitvichyanukul and Schmeiser's BTPE otherwise (as numpy), so the cost stays bounded for large `n`.
 *
 * @param k - The random key.
 * @param n - Number of trials (a non-negative integer).
 * @param p - Success probability.
 * @param shape - Output shape.
 * @returns An int32 tensor.
 * @throws MdmlError with code `value` unless `n` is a non-negative integer and `p` is in [0, 1].
 * @reference Kachitvichyanukul and Schmeiser (1988), Binomial random variate generation, CACM 31(2).
 * @example
 * ```ts
 * import { key, binomial } from 'mdml-engine/random'
 * binomial(key(0), 10, 0.3, [5]).data.every((v) => v >= 0 && v <= 10) // => true
 * binomial(key(0), 1e6, 0.5, [3]).data.every((v) => Math.abs(v - 5e5) < 5000) // => true
 * ```
 */
export function binomial(k: Key, n: number, p: number, shape: Shape): Tensor<'int32'> {
  if (!(p >= 0 && p <= 1) || !Number.isInteger(n) || n < 0) {
    throw new MdmlError('value', 'binomial: need an integer n >= 0 and p in [0, 1]')
  }
  const next = uniformStream(k)
  return ints('binomial', shape, () => binomialOne(n, p, next))
}

/**
 * Geometric samples: the number of trials up to and including the first success.
 *
 * @param k - The random key.
 * @param p - Success probability in (0, 1].
 * @param shape - Output shape.
 * @returns An int32 tensor of values at least 1.
 * @throws MdmlError with code `value` unless `p` is in (0, 1], or when a draw does not fit an int32 (tiny `p`).
 * @example
 * ```ts
 * import { key, geometric } from 'mdml-engine/random'
 * Array.from(geometric(key(0), 1, [3]).data) // => [1, 1, 1]
 * ```
 */
export function geometric(k: Key, p: number, shape: Shape): Tensor<'int32'> {
  if (!(p > 0 && p <= 1)) throw new MdmlError('value', `geometric: p must be in (0, 1], got ${p}`)
  const u = unitDoubles(k, sizeOf(shape))
  return ints('geometric', shape, (i) => (p === 1 ? 1 : Math.floor(Math.log1p(-(u[i] ?? 0)) / Math.log1p(-p)) + 1))
}

/** A Vose alias table for O(1) sampling from a fixed discrete distribution. */
export interface AliasTable {
  /** The probability of keeping each column, a float64 vector. */
  readonly probability: Tensor<'float64'>
  /** The category each column falls back to, an int32 vector. */
  readonly alias: Tensor<'int32'>
}

/**
 * Build a Vose alias table. Takes and returns tensors (migration, D-009: `weights` was `readonly number[]` and the
 * table held a `Float64Array` and an `Int32Array`; nested arrays still work as input).
 *
 * @param weights - Non-negative weights: a vector (a tensor, a typed array or a nested array).
 * @returns The table, whose `probability` and `alias` are tensors of shape `[n]`.
 * @throws MdmlError with code `value` for weights without a positive finite sum, `shape` unless `weights` is a
 * vector and `not-differentiable` for traced weights.
 * @example
 * ```ts
 * import { aliasTable } from 'mdml-engine/random'
 * aliasTable([1, 1]).alias.shape // => [2]
 * ```
 */
export function aliasTable(weights: TensorLike): AliasTable {
  const w = vectorNumbers('aliasTable', 'the weights', weights)
  const n = w.length
  cumulative(w, 'aliasTable')
  const total = w.reduce((a, b) => a + b, 0)
  const scaled = w.map((x) => (x * n) / total)
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
  return { probability: fromStorage(probability, [n], 'float64'), alias: fromStorage(alias, [n], 'int32') }
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
 * import { key, aliasTable, aliasSample } from 'mdml-engine/random'
 * Array.from(aliasSample(key(0), aliasTable([0, 1]), [3]).data) // => [1, 1, 1]
 * ```
 */
export function aliasSample(k: Key, table: AliasTable, shape: Shape): Tensor<'int32'> {
  const probability = table.probability.data
  const alias = table.alias.data
  const n = probability.length
  const u = unitDoubles(k, 2 * sizeOf(shape))
  return ints('aliasSample', shape, (i) => {
    const column = Math.floor((u[2 * i] ?? 0) * n)
    return (u[2 * i + 1] ?? 0) < (probability[column] ?? 1) ? column : (alias[column] ?? column)
  })
}

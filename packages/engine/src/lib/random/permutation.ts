/**
 * Random permutations and selections from a key: Fisher-Yates permutations and differentiable row shuffles, numpy-style
 * `choice` with or without replacement, bootstrap indices, Floyd subsampling, reservoir sampling from a stream of
 * unknown length, random splits by fraction, and keys for parallel tasks.
 */

import { MdmlError } from '../base/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, take, type Value } from '../tensor/index.ts'
import { unitDoubles } from './_unit.ts'
import { uniformStream } from './continuous.ts'
import { categorical } from './discrete.ts'
import { type Key, split } from './key.ts'

/**
 * A uniformly random permutation of `0..n-1` (Fisher-Yates).
 *
 * @param k - The random key.
 * @param n - Length.
 * @returns An int32 tensor of shape `[n]`.
 * @index function A uniformly random permutation of 0..n-1.
 * @example
 * ```ts
 * import { key, permutation } from 'mdml-engine/random'
 * Array.from(permutation(key(3), 5).data).sort() // => [0, 1, 2, 3, 4]
 * ```
 */
export function permutation(k: Key, n: number): Tensor<'int32'> {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) out[i] = i
  const u = unitDoubles(k, n)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor((u[i] ?? 0) * (i + 1))
    const tmp = out[i] ?? 0
    out[i] = out[j] ?? 0
    out[j] = tmp
  }
  return fromStorage(out, [n], 'int32')
}

/**
 * Shuffle the rows (axis 0) of a value (a gather, so it is differentiable inside a transform).
 *
 * @param k - The random key.
 * @param x - The value: a tensor, typed array or nested array with at least one dimension.
 * @returns The rows in random order.
 * @throws MdmlError with code `value` for a scalar.
 * @example
 * ```ts
 * import { key, shuffle } from 'mdml-engine/random'
 * import { tensor } from 'mdml-engine/tensor'
 * shuffle(key(0), tensor([[1], [2], [3]])).shape // => [3, 1]
 * Array.from(shuffle(key(0), new Float64Array([1, 2, 3])).shape) // => [3]
 * ```
 */
export function shuffle(k: Key, x: TensorLike): Value {
  const v = asTensor(x)
  const n = v.shape[0]
  if (n === undefined) throw new MdmlError('value', 'shuffle: need at least one dimension')
  return take(v, permutation(k, n))
}

/**
 * Choose `size` indices from `0..n-1`, with or without replacement, optionally weighted (numpy `choice`).
 *
 * @param k - The random key.
 * @param n - Population size.
 * @param size - How many to choose.
 * @param options - `replace` (default true) and `weights`.
 * @returns An int32 tensor of indices.
 * @throws MdmlError with code `value` when choosing more than `n` (or more than the positive weights) without
 * replacement, or when `weights` does not have length `n`.
 * @index function Choose indices with or without replacement, optionally weighted.
 * @example
 * ```ts
 * import { key, choice } from 'mdml-engine/random'
 * new Set(choice(key(0), 5, 5, { replace: false }).data).size // => 5
 * ```
 */
export function choice(
  k: Key,
  n: number,
  size: number,
  options: { readonly replace?: boolean; readonly weights?: readonly number[] } = {},
): Tensor<'int32'> {
  const replace = options.replace ?? true
  if (options.weights !== undefined && options.weights.length !== n) {
    throw new MdmlError('value', `choice: ${options.weights.length} weights for a population of ${n}`)
  }
  if (replace) {
    const w = options.weights ?? new Array<number>(n).fill(1)
    return categorical(k, w, [size])
  }
  if (size > n) throw new MdmlError('value', `choice: cannot take ${size} of ${n} without replacement`)
  if (options.weights === undefined) {
    return fromStorage(permutation(k, n).data.slice(0, size), [size], 'int32')
  }
  const positive = options.weights.filter((w) => w > 0).length
  if (size > positive) {
    throw new MdmlError('value', `choice: cannot take ${size} without replacement from ${positive} positive weights`)
  }
  // Weighted sampling without replacement by exponential keys (Efraimidis-Spirakis): keep the largest u^(1/w), that
  // is the smallest E / w for exponential E, compared as log E - log w so that tiny weights stay finite.
  const u = unitDoubles(k, n)
  const keys = options.weights.map((w, i) => (w > 0 ? Math.log(-Math.log(1 - (u[i] ?? 0))) - Math.log(w) : Infinity))
  const order = keys.map((_, i) => i).sort((a, b) => (keys[a] ?? 0) - (keys[b] ?? 0))
  return fromStorage(Int32Array.from(order.slice(0, size)), [size], 'int32')
}

/**
 * Indices for one bootstrap resample of `n` items (with replacement).
 *
 * @param k - The random key.
 * @param n - Number of items.
 * @returns An int32 tensor of length `n`.
 * @example
 * ```ts
 * import { key, bootstrapIndices } from 'mdml-engine/random'
 * bootstrapIndices(key(0), 10).shape // => [10]
 * ```
 */
export function bootstrapIndices(k: Key, n: number): Tensor<'int32'> {
  return choice(k, n, n)
}

/**
 * A uniformly random subset of `size` of the indices `0..n-1`, in increasing order (Floyd's algorithm: `size` draws
 * and memory proportional to `size`, however large `n` is). Use {@link choice} for the indices in random order.
 *
 * @param k - The random key.
 * @param n - Population size.
 * @param size - Subset size, at most `n`.
 * @returns An int32 tensor of `size` distinct sorted indices.
 * @throws MdmlError with code `value` unless `0 <= size <= n` are integers.
 * @reference Bentley and Floyd (1987), A sample of brilliance, CACM 30(9).
 * @example
 * ```ts
 * import { key, subsample } from 'mdml-engine/random'
 * const s = Array.from(subsample(key(0), 1000000, 5).data)
 * s.length === 5 && s.every((v, i) => i === 0 || v > (s[i - 1] ?? -1)) // => true
 * Array.from(subsample(key(0), 4, 4).data) // => [0, 1, 2, 3]
 * ```
 */
export function subsample(k: Key, n: number, size: number): Tensor<'int32'> {
  if (!Number.isInteger(n) || !Number.isInteger(size) || size < 0 || size > n) {
    throw new MdmlError('value', `subsample: need integers 0 <= size <= n, got size ${size} of ${n}`)
  }
  const u = unitDoubles(k, size)
  const chosen = new Set<number>()
  for (let i = 0; i < size; i++) {
    const j = n - size + i
    const t = Math.floor((u[i] ?? 0) * (j + 1))
    chosen.add(chosen.has(t) ? j : t)
  }
  return fromStorage(Int32Array.from(chosen).sort(), [size], 'int32')
}

/**
 * A uniform random sample of `size` items from a stream whose length is not known in advance, in one pass with
 * memory for `size` items (reservoir sampling). Every subset of `size` items is equally likely. Method `'L'` (Li's
 * Algorithm L, the default) draws how many items to skip, so it costs `O(size (1 + log(N / size)))` uniforms;
 * method `'R'` (Waterman's Algorithm R, as in Knuth) draws one uniform per item. Both are deterministic for a key;
 * on an array, Algorithm L jumps over the skipped items without visiting them.
 *
 * @param k - The random key.
 * @param items - The stream: any iterable, or an array-like.
 * @param size - Reservoir size (a non-negative integer).
 * @param options - `method`: `'L'` (default) or `'R'`.
 * @returns The sampled items (all of them, in order, when the stream is shorter than `size`); the order of a full
 * reservoir is its slot order, not the stream order.
 * @throws MdmlError with code `value` for a negative or non-integer `size`.
 * @reference Li (1994), Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))), ACM TOMS 20(4);
 * Vitter (1985), Random sampling with a reservoir, ACM TOMS 11(1).
 * @example
 * ```ts
 * import { key, reservoir } from 'mdml-engine/random'
 * reservoir(key(0), ['a', 'b'], 3) // => ['a', 'b']
 * reservoir(key(0), Array.from({ length: 100 }, (_, i) => i), 5).length // => 5
 * reservoir(key(0), new Set([1, 2, 3, 4]), 2, { method: 'R' }).length // => 2
 * ```
 */
export function reservoir<T>(
  k: Key,
  items: Iterable<T> | ArrayLike<T>,
  size: number,
  options: { readonly method?: 'L' | 'R' } = {},
): T[] {
  if (!Number.isInteger(size) || size < 0) {
    throw new MdmlError('value', `reservoir: size must be a non-negative integer, got ${size}`)
  }
  const method = options.method ?? 'L'
  const next = uniformStream(k)
  const out: T[] = []
  // Arrays, typed arrays and other array-likes are indexed directly (strings are iterated, by code point).
  const array =
    typeof items !== 'string' && typeof (items as ArrayLike<T>).length === 'number'
      ? (items as ArrayLike<T>)
      : undefined
  if (array !== undefined) {
    const n = array.length
    for (let i = 0; i < Math.min(size, n); i++) out.push(array[i] as T)
    if (size === 0 || n <= size) return out
    if (method === 'R') {
      for (let i = size; i < n; i++) {
        const j = Math.floor(next() * (i + 1))
        if (j < size) out[j] = array[i] as T
      }
      return out
    }
    let w = Math.exp(Math.log(1 - next()) / size)
    for (let i = size - 1; ; ) {
      i += Math.floor(Math.log(1 - next()) / Math.log1p(-w)) + 1
      if (!(i < n)) return out
      out[Math.floor(next() * size)] = array[i] as T
      w *= Math.exp(Math.log(1 - next()) / size)
    }
  }
  const it = (items as Iterable<T>)[Symbol.iterator]()
  while (out.length < size) {
    const r = it.next()
    if (r.done === true) return out
    out.push(r.value)
  }
  if (size === 0) return out
  if (method === 'R') {
    for (let i = size, r = it.next(); r.done !== true; i++, r = it.next()) {
      const j = Math.floor(next() * (i + 1))
      if (j < size) out[j] = r.value
    }
    return out
  }
  let w = Math.exp(Math.log(1 - next()) / size)
  for (;;) {
    const skip = Math.floor(Math.log(1 - next()) / Math.log1p(-w))
    for (let s = 0; s < skip; s++) if (it.next().done === true) return out
    const r = it.next()
    if (r.done === true) return out
    out[Math.floor(next() * size)] = r.value
    w *= Math.exp(Math.log(1 - next()) / size)
  }
}

/**
 * Split `0..n-1` into disjoint random parts with given fractions (the last part takes the remainder).
 *
 * @param k - The random key.
 * @param n - Number of items.
 * @param fractions - Fractions of the parts (summing to at most 1).
 * @returns One int32 index tensor per part.
 * @throws MdmlError with code `value` for negative fractions or fractions summing to more than 1.
 * @example
 * ```ts
 * import { key, randomSplit } from 'mdml-engine/random'
 * randomSplit(key(0), 10, [0.8, 0.2]).map((p) => p.shape[0]) // => [8, 2]
 * ```
 */
export function randomSplit(k: Key, n: number, fractions: readonly number[]): Tensor<'int32'>[] {
  const total = fractions.reduce((a, b) => a + b, 0)
  if (fractions.some((f) => !(f >= 0)) || total > 1 + 1e-9) {
    throw new MdmlError('value', `randomSplit: fractions must be non-negative with a sum of at most 1`)
  }
  const order = permutation(k, n).data
  const parts: Tensor<'int32'>[] = []
  let start = 0
  fractions.forEach((f, i) => {
    const end = i === fractions.length - 1 ? n : Math.min(n, start + Math.round(f * n))
    parts.push(fromStorage(order.slice(start, end), [end - start], 'int32'))
    start = end
  })
  return parts
}

/**
 * Independent keys for `n` parallel tasks (a convenience over `split`).
 *
 * @param k - The random key.
 * @param n - Number of keys.
 * @returns The keys.
 * @example
 * ```ts
 * import { key, keys } from 'mdml-engine/random'
 * keys(key(0), 3).length // => 3
 * ```
 */
export function keys(k: Key, n: number): Key[] {
  return split(k, n)
}

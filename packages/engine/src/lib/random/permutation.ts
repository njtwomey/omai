import { WhiteboxError } from '../base/index.ts'
import { fromStorage, type Tensor, type TensorLike, take, type Value } from '../tensor/index.ts'
import { unitDoubles } from './_unit.ts'
import { categorical } from './discrete.ts'
import { type Key, split } from './key.ts'

/**
 * A uniformly random permutation of `0..n-1` (Fisher-Yates).
 *
 * @param k - The random key.
 * @param n - Length.
 * @returns An int32 tensor of shape `[n]`.
 * @example
 * ```ts
 * import { key, permutation } from 'whitebox-engine/random'
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
 * Shuffle the rows (axis 0) of a value.
 *
 * @param k - The random key.
 * @param x - The value.
 * @returns The rows in random order.
 * @example
 * ```ts
 * import { key, shuffle } from 'whitebox-engine/random'
 * import { tensor } from 'whitebox-engine/tensor'
 * shuffle(key(0), tensor([[1], [2], [3]])).shape // => [3, 1]
 * ```
 */
export function shuffle(k: Key, x: TensorLike): Value {
  const v = typeof x === 'object' && x !== null && 'shape' in x ? (x as Value) : undefined
  const n = v?.shape[0] ?? (Array.isArray(x) ? x.length : 0)
  return take(x, permutation(k, n))
}

/**
 * Choose `size` indices from `0..n-1`, with or without replacement, optionally weighted (numpy `choice`).
 *
 * @param k - The random key.
 * @param n - Population size.
 * @param size - How many to choose.
 * @param options - `replace` (default true) and `weights`.
 * @returns An int32 tensor of indices.
 * @throws WhiteboxError with code `value` when choosing more than `n` without replacement.
 * @example
 * ```ts
 * import { key, choice } from 'whitebox-engine/random'
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
  if (replace) {
    const w = options.weights ?? new Array<number>(n).fill(1)
    return categorical(k, w, [size])
  }
  if (size > n) throw new WhiteboxError('value', `choice: cannot take ${size} of ${n} without replacement`)
  if (options.weights === undefined) {
    return fromStorage(permutation(k, n).data.slice(0, size), [size], 'int32')
  }
  // Weighted sampling without replacement by exponential keys (Efraimidis-Spirakis): keep the largest u^(1/w).
  const u = unitDoubles(k, n)
  const keys = options.weights.map((w, i) => (w > 0 ? Math.log(Math.max(u[i] ?? 0, 1e-300)) / w : -Infinity))
  const order = keys.map((_, i) => i).sort((a, b) => (keys[b] ?? 0) - (keys[a] ?? 0))
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
 * import { key, bootstrapIndices } from 'whitebox-engine/random'
 * bootstrapIndices(key(0), 10).shape // => [10]
 * ```
 */
export function bootstrapIndices(k: Key, n: number): Tensor<'int32'> {
  return choice(k, n, n)
}

/**
 * Split `0..n-1` into disjoint random parts with given fractions (the last part takes the remainder).
 *
 * @param k - The random key.
 * @param n - Number of items.
 * @param fractions - Fractions of the parts (summing to at most 1).
 * @returns One int32 index tensor per part.
 * @example
 * ```ts
 * import { key, randomSplit } from 'whitebox-engine/random'
 * randomSplit(key(0), 10, [0.8, 0.2]).map((p) => p.shape[0]) // => [8, 2]
 * ```
 */
export function randomSplit(k: Key, n: number, fractions: readonly number[]): Tensor<'int32'>[] {
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
 * import { key, keys } from 'whitebox-engine/random'
 * keys(key(0), 3).length // => 3
 * ```
 */
export function keys(k: Key, n: number): Key[] {
  return split(k, n)
}

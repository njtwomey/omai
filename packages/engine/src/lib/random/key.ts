/**
 * Immutable random keys in the style of JAX PRNG keys: two 32-bit words, created from a seed and derived with `split`
 * (several independent keys) and `foldIn` (a key per step or episode). Random bits come from the Philox counter-based
 * generator, and `keyFromPath` derives a key from a seed and a path of names.
 */

import { WhiteboxError } from '../base/index.ts'
import { philox4x32 } from './philox.ts'

/**
 * An immutable random key (two 32-bit words), in the style of JAX PRNG keys.
 *
 * Functions that need randomness take a key argument and never advance hidden state. To make several independent draws,
 * {@link split} the key; reusing a key reproduces the same draw.
 *
 * @concept counter-based-rng
 */
export interface Key {
  readonly kind: 'key'
  readonly words: readonly [number, number]
}

// Domain tags in the fourth counter word keep key derivation and bit generation from ever sharing a block.
const SPLIT_TAG = 0x53504c54
const FOLD_TAG = 0x464f4c44
const BITS_TAG = 0

/**
 * Create a key from an integer seed.
 *
 * @param seed - A non-negative safe integer.
 * @returns The key.
 * @throws WhiteboxError with code `value` for negative or non-integer seeds.
 * @concept counter-based-rng
 * @example
 * ```ts
 * import { key } from 'whitebox-engine/random'
 * key(42).words // => [42, 0]
 * ```
 */
export function key(seed: number): Key {
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new WhiteboxError('value', `key: seed must be a non-negative safe integer, got ${seed}`, { seed })
  }
  return { kind: 'key', words: [seed >>> 0, Math.floor(seed / 2 ** 32) >>> 0] }
}

/**
 * Derive `n` independent keys from one key.
 *
 * @param k - The parent key; do not use it again for draws after splitting.
 * @param n - Number of child keys (default 2).
 * @returns The child keys.
 * @example
 * ```ts
 * import { key, split } from 'whitebox-engine/random'
 * const [a, b] = split(key(0))
 * a?.words[0] === b?.words[0] // => false
 * ```
 */
export function split(k: Key, n = 2): Key[] {
  const out: Key[] = []
  for (let i = 0; i < n; i++) {
    const [w0, w1] = philox4x32([i, 0, 0, SPLIT_TAG], k.words)
    out.push({ kind: 'key', words: [w0, w1] })
  }
  return out
}

/**
 * Derive a key from a parent key and an integer, such as a step or episode number.
 *
 * @param k - The parent key.
 * @param data - An unsigned 32-bit integer.
 * @returns The derived key; the same inputs always give the same key.
 * @example
 * ```ts
 * import { key, foldIn } from 'whitebox-engine/random'
 * foldIn(key(0), 3).words[0] === foldIn(key(0), 3).words[0] // => true
 * ```
 */
export function foldIn(k: Key, data: number): Key {
  const [w0, w1] = philox4x32([data >>> 0, 0, 0, FOLD_TAG], k.words)
  return { kind: 'key', words: [w0, w1] }
}

/**
 * `n` random unsigned 32-bit words from a key.
 *
 * @param k - The key.
 * @param n - Number of words.
 * @returns The words.
 * @example
 * ```ts
 * import { key, randomBits } from 'whitebox-engine/random'
 * randomBits(key(0), 4).length // => 4
 * ```
 */
export function randomBits(k: Key, n: number): Uint32Array {
  const out = new Uint32Array(n)
  for (let block = 0; block * 4 < n; block++) {
    const words = philox4x32([block >>> 0, Math.floor(block / 2 ** 32) >>> 0, 0, BITS_TAG], k.words)
    for (let j = 0; j < 4 && block * 4 + j < n; j++) out[block * 4 + j] = words[j] ?? 0
  }
  return out
}

/**
 * Test whether a value is a {@link Key}.
 *
 * @param value - Any value.
 * @returns `true` for keys.
 * @example
 * ```ts
 * import { isKey, key } from 'whitebox-engine/random'
 * isKey(key(1)) // => true
 * ```
 */
export function isKey(value: unknown): value is Key {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'key'
}

// FNV-1a: a stable 32-bit hash of a name, so key paths are reproducible across platforms.
function hashName(name: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

/**
 * Derive a key from a seed and a path of names and numbers. The same path always gives the same key, whatever else
 * has been drawn, which is how contexts give every part of a program its own reproducible randomness.
 *
 * @param seed - The root seed.
 * @param path - Names (hashed) and non-negative integers (folded in directly).
 * @returns The derived key.
 * @concept counter-based-rng
 * @example
 * ```ts
 * import { keyFromPath } from 'whitebox-engine/random'
 * keyFromPath(0, 'init', 3).words[0] === keyFromPath(0, 'init', 3).words[0] // => true
 * keyFromPath(0, 'init').words[0] === keyFromPath(0, 'data').words[0] // => false
 * ```
 */
export function keyFromPath(seed: number, ...path: readonly (string | number)[]): Key {
  let k = key(seed)
  for (const part of path) k = foldIn(k, typeof part === 'number' ? part : hashName(part))
  return k
}

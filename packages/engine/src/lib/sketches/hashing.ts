/**
 * Deterministic hashing for the sketches: a seeded 32-bit hash of strings and numbers (FNV-1a mixing finished by the
 * Murmur3 avalanche), a 64-bit pair for double hashing, and the derivation of a hash seed from a random `Key`, so a
 * sketch is reproducible from its key exactly as every other random object in the library. Nothing here reads a clock
 * or `Math.random`, and the numbers hash through their IEEE-754 bytes in a fixed byte order.
 */

import { isKey, type Key } from '../random/index.ts'

/** An item the sketches accept: a string or a number. */
export type SketchItem = string | number

/** A hash seed: an unsigned integer, or a random `Key` from which one is derived. */
export type SketchSeed = number | Key

const BYTES = new DataView(new ArrayBuffer(8))

/**
 * The Murmur3 32-bit finaliser (`fmix32`): a bijection of the unsigned 32-bit integers in which every input bit flips
 * about half of the output bits.
 *
 * @param h - An integer (taken modulo 2^32).
 * @returns The mixed unsigned 32-bit integer.
 * @concept hashing
 * @example
 * ```ts
 * import { fmix32 } from 'omai-engine/sketches'
 * fmix32(0) // => 0
 * fmix32(1) // => 1364076727
 * ```
 */
export function fmix32(h: number): number {
  let x = h >>> 0
  x ^= x >>> 16
  x = Math.imul(x, 0x85ebca6b) >>> 0
  x ^= x >>> 13
  x = Math.imul(x, 0xc2b2ae35) >>> 0
  x ^= x >>> 16
  return x >>> 0
}

/**
 * Resolve a {@link SketchSeed} to an unsigned 32-bit hash seed. A number is taken modulo 2^32; a `Key` is mixed from
 * both of its words, so two different keys give different seeds (up to 32-bit collisions) and the same key always the
 * same one.
 *
 * @param seed - A number or a `Key` (default 0).
 * @returns The unsigned 32-bit seed.
 * @concept hashing
 * @example
 * ```ts
 * import { seedOf } from 'omai-engine/sketches'
 * import { key } from 'omai-engine/random'
 * seedOf(7) // => 7
 * seedOf(key(7)) === seedOf(key(7)) // => true
 * seedOf(key(7)) === seedOf(key(8)) // => false
 * ```
 */
export function seedOf(seed: SketchSeed = 0): number {
  if (isKey(seed)) return fmix32(fmix32(seed.words[0] ^ 0x9e3779b9) + seed.words[1])
  return seed >>> 0
}

/**
 * Hash a string or number with a seed into an unsigned 32-bit integer. Numbers hash through their IEEE-754 bytes
 * (little-endian), with `-0` equal to `0`, so `hash32(1)` differs from `hash32('1')`.
 *
 * @param item - The item to hash.
 * @param seed - Unsigned 32-bit seed (default 0).
 * @returns An unsigned integer in `[0, 2^32)`.
 * @concept hashing
 * @example
 * ```ts
 * import { hash32 } from 'omai-engine/sketches'
 * hash32('alpha', 1) === hash32('alpha', 1) // => true
 * hash32('alpha', 1) === hash32('alpha', 2) // => false
 * hash32(0) === hash32(-0) // => true
 * ```
 */
export function hash32(item: SketchItem, seed = 0): number {
  let h = (seed ^ 0x811c9dc5) >>> 0
  if (typeof item === 'number') {
    BYTES.setFloat64(0, item === 0 ? 0 : item, true)
    h ^= BYTES.getUint32(0, true)
    h = Math.imul(h, 0x01000193) >>> 0
    h ^= BYTES.getUint32(4, true)
    h = Math.imul(h, 0x01000193) >>> 0
    return fmix32(h)
  }
  for (let i = 0; i < item.length; i++) {
    h ^= item.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return fmix32(h ^ item.length)
}

/**
 * A 64-bit hash as two 32-bit halves `[h1, h2]` (two seeds apart): enough for Kirsch-Mitzenmacher double hashing of
 * any number of probes and for HyperLogLog's 64-bit rank.
 *
 * @param item - The item to hash.
 * @param seed - Unsigned 32-bit seed (default 0).
 * @returns `[h1, h2]`, each an unsigned 32-bit integer.
 * @concept hashing
 * @example
 * ```ts
 * import { hashPair } from 'omai-engine/sketches'
 * hashPair('x', 3).length // => 2
 * hashPair('x', 3)[0] === hashPair('x', 3)[0] // => true
 * ```
 */
export function hashPair(item: SketchItem, seed = 0): readonly [number, number] {
  return [hash32(item, seed), hash32(item, (seed ^ 0x9e3779b9) >>> 0)]
}

/**
 * Probe `i` of the enhanced double hashing family (Kirsch and Mitzenmacher 2006; Dillinger and Manolios 2004):
 * `h1 + i h2 + i^2` modulo `m`, with `h2` forced odd, which behaves like `k` independent hashes for Bloom filters.
 *
 * @param h1 - Primary 32-bit hash.
 * @param h2 - Secondary 32-bit hash.
 * @param i - Probe number `0, 1, ..., k - 1`.
 * @param m - Table size (`m >= 1`).
 * @returns A bucket index in `[0, m)`.
 * @concept hashing
 * @reference A. Kirsch and M. Mitzenmacher, Less hashing, same performance: building a better Bloom filter
 *   (Random Structures and Algorithms 2008).
 * @example
 * ```ts
 * import { doubleHashIndex } from 'omai-engine/sketches'
 * doubleHashIndex(10, 3, 0, 100) // => 10
 * doubleHashIndex(10, 3, 1, 100) // => 14
 * ```
 */
export function doubleHashIndex(h1: number, h2: number, i: number, m: number): number {
  const step = (h2 | 1) >>> 0
  return (((h1 >>> 0) + Math.imul(i, step) + Math.imul(i, i)) >>> 0) % m
}

/**
 * A string that identifies an item by type and value (`'n:1'` for the number 1, `'s:1'` for the string), for exact
 * reference counts kept beside a sketch.
 *
 * @param item - The item.
 * @returns Its key.
 * @concept hashing
 * @example
 * ```ts
 * import { itemKey } from 'omai-engine/sketches'
 * itemKey(1) // => 'n:1'
 * itemKey('1') // => 's:1'
 * ```
 */
export function itemKey(item: SketchItem): string {
  return typeof item === 'number' ? `n:${item}` : `s:${item}`
}

/**
 * The item an {@link itemKey} names.
 *
 * @param key - A key made by `itemKey`.
 * @returns The number or string it was made from.
 * @concept hashing
 * @example
 * ```ts
 * import { itemFromKey, itemKey } from 'omai-engine/sketches'
 * itemFromKey(itemKey(2.5)) // => 2.5
 * itemFromKey(itemKey('a')) // => 'a'
 * ```
 */
export function itemFromKey(key: string): SketchItem {
  return key.startsWith('n:') ? Number(key.slice(2)) : key.slice(2)
}

/**
 * String hash functions shared by the text tools: UTF-8 encoding, MurmurHash3 (32-bit, matching scikit-learn's
 * `murmurhash3_32` used by its hashing vectoriser) and FNV-1a.
 */

let encoder: TextEncoder | undefined

/**
 * The UTF-8 bytes of a string (lone surrogates become U+FFFD, as TextEncoder does).
 *
 * @param text - The string.
 * @returns Its UTF-8 encoding.
 * @example
 * ```ts
 * import { utf8Bytes } from 'omai-engine/text'
 * Array.from(utf8Bytes('é')) // => [195, 169]
 * ```
 */
export function utf8Bytes(text: string): Uint8Array {
  encoder ??= new TextEncoder()
  return encoder.encode(text)
}

const bytesOf = (key: string | Uint8Array): Uint8Array => (typeof key === 'string' ? utf8Bytes(key) : key)

/**
 * MurmurHash3 (x86, 32-bit) of the UTF-8 bytes of a string, as scikit-learn's `murmurhash3_32` (the hash of
 * HashingVectorizer and FeatureHasher).
 *
 * @param key - The string (or bytes) to hash.
 * @param seed - The seed (default 0).
 * @param positive - Return the unsigned hash (default false: a signed 32-bit integer, as scikit-learn).
 * @returns The hash.
 * @concept feature-hashing
 * @example
 * ```ts
 * import { murmurhash3 } from 'omai-engine/text'
 * [murmurhash3('foo'), murmurhash3('foo', 0, true)] // => [-156908512, 4138058784]
 * ```
 */
export function murmurhash3(key: string | Uint8Array, seed = 0, positive = false): number {
  const data = bytesOf(key)
  const c1 = 0xcc9e2d51
  const c2 = 0x1b873593
  let h = seed >>> 0
  const blocks = data.length >> 2
  for (let i = 0; i < blocks; i++) {
    const o = i * 4
    let k = (data[o] ?? 0) | ((data[o + 1] ?? 0) << 8) | ((data[o + 2] ?? 0) << 16) | ((data[o + 3] ?? 0) << 24)
    k = Math.imul(k, c1)
    k = (k << 15) | (k >>> 17)
    k = Math.imul(k, c2)
    h ^= k
    h = (h << 13) | (h >>> 19)
    h = (Math.imul(h, 5) + 0xe6546b64) | 0
  }
  const tail = blocks * 4
  const rest = data.length & 3
  if (rest > 0) {
    let k = 0
    if (rest === 3) k ^= (data[tail + 2] ?? 0) << 16
    if (rest >= 2) k ^= (data[tail + 1] ?? 0) << 8
    k ^= data[tail] ?? 0
    k = Math.imul(k, c1)
    k = (k << 15) | (k >>> 17)
    k = Math.imul(k, c2)
    h ^= k
  }
  h ^= data.length
  h ^= h >>> 16
  h = Math.imul(h, 0x85ebca6b)
  h ^= h >>> 13
  h = Math.imul(h, 0xc2b2ae35)
  h ^= h >>> 16
  return positive ? h >>> 0 : h | 0
}

/**
 * The 32-bit FNV-1a hash (Fowler, Noll and Vo) of the UTF-8 bytes of a string. With `signedBytes: true`, each byte
 * is sign-extended before the xor, as fastText's (and gensim's) `hash` does, which changes the hash of non-ASCII text.
 *
 * @param key - The string (or bytes) to hash.
 * @param options - Whether to sign-extend bytes (fastText's variant).
 * @returns The unsigned hash.
 * @concept feature-hashing
 * @example
 * ```ts
 * import { fnv1a } from 'omai-engine/text'
 * [fnv1a('a'), fnv1a('é'), fnv1a('é', { signedBytes: true })] // => [3826002220, 513665217, 1023043777]
 * ```
 */
export function fnv1a(key: string | Uint8Array, options: { readonly signedBytes?: boolean } = {}): number {
  const data = bytesOf(key)
  const signed = options.signedBytes === true
  let h = 0x811c9dc5
  for (const b of data) {
    h ^= signed && b > 127 ? b - 256 : b
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

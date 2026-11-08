/**
 * Document fingerprinting and near-duplicate detection: Rabin-Karp rolling hashes and substring search, winnowing (the
 * selection scheme behind the MOSS plagiarism detector), SimHash signatures compared by Hamming distance, MinHash
 * signatures that estimate Jaccard similarity (as datasketch), and locality-sensitive hashing with the optimal banding,
 * combined in `nearDuplicates`.
 */

import { OmaiError } from '../base/index.ts'
import { gaussLegendre } from '../numerics/quadrature/index.ts'
import { type Key, randomBits } from '../random/index.ts'
import { fmix32 } from '../sketches/index.ts'
import { toFlat } from '../tensor/index.ts'
import { murmurhash3, utf8Bytes } from './hashing.ts'
import { shingles } from './shingles.ts'

// ---------------------------------------------------------------------------------------------------------------
// Rabin-Karp rolling hashes.

/** Parameters of a polynomial rolling hash (kept below 2^53 so every product is exact). */
export interface RollingHashOptions {
  /** The base (default 257). */
  readonly base?: number
  /** The modulus (default 1,000,000,007, a prime). */
  readonly modulus?: number
}

function codes(s: string | readonly number[]): number[] {
  return typeof s === 'string' ? Array.from(s, (c) => c.codePointAt(0) ?? 0) : [...s]
}

function mulmod(a: number, b: number, m: number): number {
  // Split b to keep partial products below 2^53.
  const hi = Math.floor(b / 65536)
  const lo = b % 65536
  return (((a * hi) % m) * 65536 + a * lo) % m
}

/**
 * The polynomial rolling hash (Karp and Rabin 1987) of every window of k symbols: h = sum c_i base^(k-1-i) mod m,
 * updated in constant time per step.
 *
 * @param s - A string (hashed by code point) or symbol codes.
 * @param k - The window length.
 * @param options - Base and modulus.
 * @returns The hash of each window, by start position.
 * @throws OmaiError with code `value` for k below 1.
 * @concept fingerprinting
 * @example
 * ```ts
 * import { rollingHashes } from 'omai-engine/text'
 * rollingHashes('abab', 2) // => [25027, 25283, 25027]
 * ```
 */
export function rollingHashes(s: string | readonly number[], k: number, options: RollingHashOptions = {}): number[] {
  if (!Number.isInteger(k) || k < 1) throw new OmaiError('value', 'rollingHashes: k must be a positive integer')
  const base = options.base ?? 257
  const m = options.modulus ?? 1_000_000_007
  const c = codes(s)
  if (c.length < k) return []
  let top = 1
  for (let i = 1; i < k; i++) top = mulmod(top, base, m)
  let h = 0
  for (let i = 0; i < k; i++) h = (mulmod(h, base, m) + ((c[i] ?? 0) % m)) % m
  const out = [h]
  for (let i = k; i < c.length; i++) {
    h = (h - mulmod((c[i - k] ?? 0) % m, top, m) + m) % m
    h = (mulmod(h, base, m) + ((c[i] ?? 0) % m)) % m
    out.push(h)
  }
  return out
}

/**
 * Exact substring search by Rabin-Karp: compare the rolling hash of each window with the pattern's, verifying hits.
 *
 * @param pattern - The pattern.
 * @param text - The text.
 * @param options - Base and modulus.
 * @returns The start positions (in code points) of every occurrence.
 * @concept fingerprinting
 * @example
 * ```ts
 * import { rabinKarp } from 'omai-engine/text'
 * rabinKarp('aba', 'abababa') // => [0, 2, 4]
 * ```
 */
export function rabinKarp(pattern: string, text: string, options: RollingHashOptions = {}): number[] {
  const p = Array.from(pattern)
  const t = Array.from(text)
  if (p.length === 0) return Array.from({ length: t.length + 1 }, (_, i) => i)
  const target = rollingHashes(p.join(''), p.length, options)[0]
  const out: number[] = []
  for (const [i, h] of rollingHashes(t.join(''), p.length, options).entries())
    if (h === target && p.every((ch, j) => t[i + j] === ch)) out.push(i)
  return out
}

/** A fingerprint selected by winnowing: a k-gram hash and the position of its k-gram. */
export interface Fingerprint {
  readonly hash: number
  readonly position: number
}

/**
 * Winnowing (Schleimer, Wilkerson and Aiken 2003, MOSS): hash every k-gram, slide a window of w hashes and keep the
 * minimum of each window (the rightmost on ties), recording it once while it stays the minimum. Any shared substring
 * of at least w + k - 1 characters yields a shared fingerprint.
 *
 * @param text - The text (or symbol codes, e.g. normalised tokens).
 * @param k - The k-gram length (noise threshold).
 * @param w - The window size.
 * @param options - Rolling hash parameters.
 * @returns The fingerprints, by position.
 * @throws OmaiError with code `value` for k or w below 1.
 * @concept fingerprinting
 * @example
 * ```ts
 * import { winnow } from 'omai-engine/text'
 * winnow('adorunrunrunadorunrun', 5, 4).length // => 6
 * ```
 */
export function winnow(
  text: string | readonly number[],
  k: number,
  w: number,
  options: RollingHashOptions = {},
): Fingerprint[] {
  if (!Number.isInteger(w) || w < 1) throw new OmaiError('value', 'winnow: w must be a positive integer')
  const hashes = rollingHashes(text, k, options)
  const out: Fingerprint[] = []
  let last = -1
  for (let start = 0; start + w <= Math.max(hashes.length, w); start++) {
    const end = Math.min(start + w, hashes.length)
    if (end <= start) break
    let best = start
    for (let i = start; i < end; i++) if ((hashes[i] ?? 0) <= (hashes[best] ?? 0)) best = i
    if (best !== last) {
      out.push({ hash: hashes[best] ?? 0, position: best })
      last = best
    }
    if (end === hashes.length) break
  }
  return out
}

// ---------------------------------------------------------------------------------------------------------------
// SimHash.

/**
 * SimHash (Charikar 2002): a 64-bit signature of weighted features, each bit the sign of the sum of the weights of
 * features whose hash has that bit set; similar texts have signatures at small Hamming distance. Features are hashed
 * with two 32-bit MurmurHash3 lanes.
 *
 * @param features - Features with weights (a map) or a list (weight 1 each, repeats add up).
 * @returns The signature as a 64-bit unsigned integer.
 * @concept fingerprinting
 * @example
 * ```ts
 * import { simhash, hammingDistance64 } from 'omai-engine/text'
 * const a = simhash(['the', 'quick', 'brown', 'fox', 'jumps'])
 * const b = simhash(['the', 'quick', 'brown', 'fox', 'leaps'])
 * hammingDistance64(a, b) < hammingDistance64(a, simhash(['lorem', 'ipsum', 'dolor', 'sit', 'amet'])) // => true
 * ```
 */
export function simhash(features: ReadonlyMap<string, number> | Iterable<string>): bigint {
  const weights = new Map<string, number>()
  if (features instanceof Map) for (const [f, w] of features) weights.set(f, w)
  else for (const f of features as Iterable<string>) weights.set(f, (weights.get(f) ?? 0) + 1)
  const v = new Float64Array(64)
  for (const [f, w] of weights) {
    const lo = murmurhash3(f, 0, true)
    const hi = murmurhash3(f, 0x9747b28c, true)
    for (let b = 0; b < 32; b++) {
      v[b] = (v[b] ?? 0) + ((lo >>> b) & 1 ? w : -w)
      v[b + 32] = (v[b + 32] ?? 0) + ((hi >>> b) & 1 ? w : -w)
    }
  }
  let out = 0n
  for (let b = 63; b >= 0; b--) out = (out << 1n) | ((v[b] ?? 0) > 0 ? 1n : 0n)
  return out
}

/**
 * The number of differing bits of two 64-bit signatures.
 *
 * @param a - The first signature.
 * @param b - The second signature.
 * @returns The Hamming distance.
 * @concept fingerprinting
 * @example
 * ```ts
 * import { hammingDistance64 } from 'omai-engine/text'
 * hammingDistance64(0b1011n, 0b0001n) // => 2
 * ```
 */
export function hammingDistance64(a: bigint, b: bigint): number {
  let x = BigInt.asUintN(64, a ^ b)
  let n = 0
  while (x > 0n) {
    x &= x - 1n
    n++
  }
  return n
}

// ---------------------------------------------------------------------------------------------------------------
// MinHash and LSH (datasketch's affine32 scheme).

/** The affine permutations of a MinHash: odd multipliers a and offsets b, both 32-bit. */
export interface MinhashPermutations {
  readonly a: Uint32Array
  readonly b: Uint32Array
}

/**
 * Random MinHash permutations (odd 32-bit multipliers and 32-bit offsets) from a key.
 *
 * @param key - The key.
 * @param numPerm - The number of permutations (default 128).
 * @returns The permutations.
 * @throws OmaiError with code `value` for a non-positive count.
 * @concept minhash
 * @example
 * ```ts
 * import { minhashPermutations } from 'omai-engine/text'
 * import { key } from 'omai-engine/random'
 * Array.from(minhashPermutations(key(0), 4).a).every((x) => x % 2 === 1) // => true
 * ```
 */
export function minhashPermutations(key: Key, numPerm = 128): MinhashPermutations {
  if (!Number.isInteger(numPerm) || numPerm < 1)
    throw new OmaiError('value', 'minhash: numPerm must be a positive integer')
  const bits = randomBits(key, 2 * numPerm)
  return {
    a: Uint32Array.from({ length: numPerm }, (_, i) => ((bits[2 * i] ?? 0) | 1) >>> 0),
    b: Uint32Array.from({ length: numPerm }, (_, i) => (bits[2 * i + 1] ?? 0) >>> 0),
  }
}

/**
 * A MinHash signature (Broder 1997) as datasketch's default (affine32) scheme computes it: each item's 32-bit hash is
 * mixed by MurmurHash3's finalizer, then permuted by h -> a h + b mod 2^32, and the signature keeps the minimum of
 * each permutation (2^32 - 1 for an empty set).
 *
 * @param items - The items (shingles, tokens).
 * @param permutations - The permutations ({@link minhashPermutations}, or datasketch's).
 * @param hash - The item hash, a 32-bit unsigned integer (default MurmurHash3 of the UTF-8 bytes).
 * @returns The signature.
 * @concept minhash
 * @example
 * ```ts
 * import { minhash, minhashPermutations, minhashJaccard } from 'omai-engine/text'
 * import { key } from 'omai-engine/random'
 * const p = minhashPermutations(key(1), 256)
 * minhashJaccard(minhash(['a', 'b', 'c'], p), minhash(['a', 'b', 'c'], p)) // => 1
 * ```
 */
export function minhash(
  items: Iterable<string>,
  permutations: MinhashPermutations,
  hash: (item: string) => number = (s) => murmurhash3(utf8Bytes(s), 0, true),
): Uint32Array {
  const n = permutations.a.length
  const sig = new Uint32Array(n).fill(0xffffffff)
  for (const it of new Set(items)) {
    const hv = fmix32(hash(it))
    for (let i = 0; i < n; i++) {
      const v = (Math.imul(permutations.a[i] ?? 0, hv) + (permutations.b[i] ?? 0)) >>> 0
      if (v < (sig[i] ?? 0)) sig[i] = v
    }
  }
  return sig
}

/**
 * The Jaccard similarity estimated from two MinHash signatures: the fraction of equal entries.
 *
 * @param a - The first signature.
 * @param b - The second signature.
 * @returns The estimate in [0, 1].
 * @throws OmaiError with code `shape` for signatures of different lengths.
 * @concept minhash
 * @example
 * ```ts
 * import { minhashJaccard } from 'omai-engine/text'
 * minhashJaccard(Uint32Array.of(1, 2, 3, 4), Uint32Array.of(1, 2, 0, 0)) // => 0.5
 * ```
 */
export function minhashJaccard(a: Uint32Array, b: Uint32Array): number {
  if (a.length !== b.length) throw new OmaiError('shape', 'minhashJaccard: signatures differ in length')
  let same = 0
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++
  return same / a.length
}

/**
 * The LSH banding of MinHash signatures (b bands of r rows) that minimises the weighted false positive and false
 * negative probabilities at a Jaccard threshold (datasketch's `_optimal_param`): the integrals of 1 - (1 - s^r)^b below
 * and of (1 - s^r)^b above the threshold, computed exactly by Gauss-Legendre quadrature.
 *
 * @param threshold - The Jaccard threshold.
 * @param numPerm - The signature length.
 * @param weights - The weights of false positives and false negatives (default 0.5 each).
 * @returns The number of bands and rows per band.
 * @throws OmaiError with code `value` for a threshold outside [0, 1] or weights not summing to 1.
 * @concept locality-sensitive-hashing
 * @example
 * ```ts
 * import { lshParameters } from 'omai-engine/text'
 * lshParameters(0.5, 32) // => { bands: 8, rows: 4 }
 * ```
 */
export function lshParameters(
  threshold: number,
  numPerm: number,
  weights: readonly [number, number] = [0.5, 0.5],
): { readonly bands: number; readonly rows: number } {
  if (!(threshold >= 0 && threshold <= 1)) throw new OmaiError('value', 'lshParameters: threshold must be in [0, 1]')
  if (Math.abs(weights[0] + weights[1] - 1) > 1e-9) throw new OmaiError('value', 'lshParameters: weights must sum to 1')
  // The integrands are polynomials of degree at most bands * rows <= numPerm, so Gauss-Legendre with
  // ceil((numPerm + 1) / 2) points integrates them exactly (scipy's quad, which datasketch uses, agrees to rounding).
  const rule = gaussLegendre(Math.max(1, Math.ceil((numPerm + 1) / 2)))
  const nodes = toFlat(rule.nodes)
  const ws = toFlat(rule.weights)
  const integrate = (f: (s: number) => number, lo: number, hi: number): number => {
    const half = (hi - lo) / 2
    const mid = (hi + lo) / 2
    let acc = 0
    for (const [i, x] of nodes.entries()) acc += (ws[i] ?? 0) * f(mid + half * x)
    return acc * half
  }
  let best = Number.POSITIVE_INFINITY
  let opt = { bands: 0, rows: 0 }
  for (let b = 1; b <= numPerm; b++) {
    for (let r = 1; r <= Math.floor(numPerm / b); r++) {
      const miss = (x: number): number => (1 - x ** r) ** b
      const fp = integrate((x) => 1 - miss(x), 0, threshold)
      const fn = integrate(miss, threshold, 1)
      const err = fp * weights[0] + fn * weights[1]
      if (err < best) {
        best = err
        opt = { bands: b, rows: r }
      }
    }
  }
  return opt
}

/**
 * Candidate near-duplicate pairs by LSH banding (Indyk and Motwani 1998; datasketch's MinHashLSH): signatures that
 * agree on every row of at least one band.
 *
 * @param signatures - MinHash signatures.
 * @param bands - The number of bands.
 * @param rows - Rows per band (bands times rows at most the signature length).
 * @returns The candidate pairs [i, j] with i < j, in order.
 * @throws OmaiError with code `value` when the bands do not fit the signatures.
 * @concept locality-sensitive-hashing
 * @example
 * ```ts
 * import { lshCandidates } from 'omai-engine/text'
 * lshCandidates([Uint32Array.of(1, 2, 3, 4), Uint32Array.of(1, 2, 9, 9), Uint32Array.of(7, 7, 7, 7)], 2, 2) // => [[0, 1]]
 * ```
 */
export function lshCandidates(signatures: readonly Uint32Array[], bands: number, rows: number): [number, number][] {
  const n = signatures[0]?.length ?? 0
  if (!(bands >= 1 && rows >= 1 && bands * rows <= n))
    throw new OmaiError('value', 'lshCandidates: bands * rows must fit the signature')
  const pairs = new Set<string>()
  for (let band = 0; band < bands; band++) {
    const buckets = new Map<string, number[]>()
    for (const [i, s] of signatures.entries()) {
      const key = Array.from(s.subarray(band * rows, (band + 1) * rows)).join(',')
      const list = buckets.get(key) ?? []
      for (const j of list) pairs.add(`${j},${i}`)
      list.push(i)
      buckets.set(key, list)
    }
  }
  return [...pairs].map((p) => p.split(',').map(Number) as [number, number]).sort((x, y) => x[0] - y[0] || x[1] - y[1])
}

/**
 * Near-duplicate detection: character k-shingles, MinHash signatures, LSH candidates with the optimal banding for the
 * threshold, and the pairs whose estimated Jaccard similarity reaches it.
 *
 * @param documents - The documents.
 * @param options - Threshold (default 0.5), shingle size (default 5), signature length (default 128) and key.
 * @returns The near-duplicate pairs with their estimated similarity.
 * @concept near-duplicate-detection
 * @example
 * ```ts
 * import { nearDuplicates } from 'omai-engine/text'
 * import { key } from 'omai-engine/random'
 * const docs = ['the quick brown fox jumps over the lazy dog', 'the quick brown fox jumped over the lazy dog', 'an entirely different sentence about cats']
 * nearDuplicates(docs, { key: key(3), threshold: 0.5 }).map((d) => d.pair) // => [[0, 1]]
 * ```
 */
export function nearDuplicates(
  documents: readonly string[],
  options: { readonly key: Key; readonly threshold?: number; readonly shingleSize?: number; readonly numPerm?: number },
): { readonly pair: [number, number]; readonly similarity: number }[] {
  const threshold = options.threshold ?? 0.5
  const numPerm = options.numPerm ?? 128
  const perms = minhashPermutations(options.key, numPerm)
  const sigs = documents.map((d) => minhash(shingles(d, { size: options.shingleSize ?? 5 }), perms))
  const { bands, rows } = lshParameters(threshold, numPerm)
  const out: { pair: [number, number]; similarity: number }[] = []
  for (const [i, j] of lshCandidates(sigs, bands, rows)) {
    const sim = minhashJaccard(sigs[i] as Uint32Array, sigs[j] as Uint32Array)
    if (sim >= threshold) out.push({ pair: [i, j], similarity: sim })
  }
  return out
}

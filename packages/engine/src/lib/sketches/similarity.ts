/**
 * Set similarity sketches and locality-sensitive hashing: Broder's MinHash
 * Jaccard signature and banded LSH index with S-curve collision probability.
 */

import { MdmlError } from '../base/index.ts'
import { gaussLegendre } from '../numerics/quadrature/index.ts'
import { fmix32, hash32, type SketchItem, type SketchSeed, seedOf } from './hashing.ts'

/** Options for computing a {@link MinHashSignature}. */
export interface MinHashOptions {
  /** Number of independent hash functions `k`. Defaults to `128`. */
  readonly numHashes?: number
  /** Base hash seed, a number or a `Key`. Defaults to `0`. */
  readonly seed?: SketchSeed
}

/**
 * Broder's MinHash signature of length `numHashes` representing a set as its
 * coordinatewise minimum hash values under `k` independent hash permutations.
 */
export interface MinHashSignature {
  /** Number of hash coordinates `k`. */
  readonly numHashes: number
  /** Seed used to generate the hash family. */
  readonly seed: number
  /** Coordinatewise minimum 32-bit unsigned hashes of length `numHashes`. */
  readonly values: Uint32Array
  /** Update the signature in place with an additional element. */
  add(item: SketchItem): void
  /** Estimate Jaccard similarity `|A cap B| / |A cup B|` with another signature. */
  jaccard(other: MinHashSignature): number
  /** Merge two MinHash signatures by coordinatewise minimum (`min(sigA[i], sigB[i])`). */
  merge(other: MinHashSignature): MinHashSignature
}

/** Options for constructing a banded {@link LshIndex} over MinHash signatures. */
export interface LshIndexOptions {
  /** Number of bands `b`. */
  readonly bands: number
  /** Number of rows (hashes) per band `r` (`bands * rows <= signature.numHashes`). */
  readonly rows: number
}

/**
 * Banded Locality-Sensitive Hashing (LSH) index over MinHash signatures.
 * Two signatures collide in at least one band with probability
 * `1 - (1 - s^r)^b`, producing a sharp S-curve threshold around `(1 / b)^(1 / r)`.
 */
export interface LshIndex {
  /** Number of bands `b`. */
  readonly bands: number
  /** Number of rows per band `r`. */
  readonly rows: number
  /** Approximate Jaccard similarity threshold `(1 / b)^(1 / r)`. */
  readonly threshold: number
  /** Insert a labelled MinHash signature into the banded hash tables. */
  add(id: string, signature: MinHashSignature): void
  /** Return candidate document IDs that share at least one band bucket with `signature`. */
  query(signature: MinHashSignature): readonly string[]
  /** Return all unique candidate pairs `[idA, idB]` sharing at least one band bucket. */
  candidatePairs(): readonly (readonly [string, string])[]
  /** Theoretical probability `1 - (1 - s^r)^b` that a pair with Jaccard `s` becomes a candidate. */
  collisionProbability(jaccardSimilarity: number): number
}

/**
 * Compute a MinHash signature over an iterable of set elements.
 *
 * @param items - Initial elements of the set (optional)
 * @param options - Number of hash functions `numHashes` (default `128`) and `seed`
 * @returns {@link MinHashSignature} supporting incremental `add`, `jaccard`, and `merge`
 *
 * @example
 * ```ts
 * import { minHash } from 'mdml-engine/sketches'
 * const sigA = minHash(['a', 'b', 'c', 'd'], { numHashes: 256 })
 * const sigB = minHash(['b', 'c', 'd', 'e'], { numHashes: 256 })
 * Math.abs(sigA.jaccard(sigB) - 3 / 5) < 0.15 // => true
 * ```
 */
export function minHash(items: Iterable<SketchItem> = [], options: MinHashOptions = {}): MinHashSignature {
  const numHashes = options.numHashes ?? 128
  const seed = seedOf(options.seed)
  if (!Number.isInteger(numHashes) || numHashes < 1) {
    throw new MdmlError('value', `minHash: numHashes must be a positive integer, got ${numHashes}`)
  }

  function build(values: Uint32Array): MinHashSignature {
    const sig: MinHashSignature = {
      numHashes,
      seed,
      values,
      add(item: SketchItem): void {
        const h1 = hash32(item, seed)
        const h2 = hash32(item, (seed ^ 0x9e3779b9) >>> 0)
        for (let i = 0; i < numHashes; i++) {
          const combined = fmix32((h1 + Math.imul(i + 1, h2 | 1) + Math.imul(i, 0x85ebca6b)) >>> 0)
          const cur = values[i] ?? 0xffffffff
          if (combined < cur) {
            values[i] = combined
          }
        }
      },
      jaccard(other: MinHashSignature): number {
        return minHashJaccard(sig, other)
      },
      merge(other: MinHashSignature): MinHashSignature {
        if (other.numHashes !== numHashes || other.seed !== seed) {
          throw new MdmlError('value', 'minHash.merge: numHashes and seed must match')
        }
        const merged = new Uint32Array(numHashes)
        for (let i = 0; i < numHashes; i++) {
          const a = values[i] ?? 0xffffffff
          const b = other.values[i] ?? 0xffffffff
          merged[i] = Math.min(a, b)
        }
        return build(merged)
      },
    }
    return sig
  }

  const initValues = new Uint32Array(numHashes)
  initValues.fill(0xffffffff)
  const sig = build(initValues)
  for (const item of items) {
    sig.add(item)
  }
  return sig
}

/**
 * Estimate Jaccard similarity `|A cap B| / |A cup B|` from two MinHash signatures
 * as the fraction of coordinates where `sigA.values[i] === sigB.values[i]`.
 *
 * @param sigA - First MinHash signature
 * @param sigB - Second MinHash signature
 * @returns Unbiased Jaccard similarity estimate in `[0, 1]`
 *
 * @example
 * ```ts
 * import { minHash, minHashJaccard } from 'mdml-engine/sketches'
 * const s1 = minHash([1, 2, 3], { numHashes: 64 })
 * const s2 = minHash([1, 2, 3], { numHashes: 64 })
 * minHashJaccard(s1, s2) // => 1
 * ```
 */
export function minHashJaccard(sigA: MinHashSignature, sigB: MinHashSignature): number {
  if (sigA.numHashes !== sigB.numHashes) {
    throw new MdmlError(
      'value',
      `minHashJaccard: signature lengths must match (${sigA.numHashes} vs ${sigB.numHashes})`,
    )
  }
  let matches = 0
  for (let i = 0; i < sigA.numHashes; i++) {
    if (sigA.values[i] === sigB.values[i]) matches++
  }
  return matches / sigA.numHashes
}

/**
 * Evaluate the banded LSH S-curve collision probability `1 - (1 - s^r)^b` that
 * two sets with true Jaccard similarity `s` agree on all `r` rows in at least
 * one of `b` bands.
 *
 * @param jaccardSimilarity - True Jaccard similarity `s` in `[0, 1]`
 * @param bands - Number of bands `b`
 * @param rows - Number of rows per band `r`
 * @returns Probability in `[0, 1]` of colliding in at least one band
 *
 * @example
 * ```ts
 * import { lshCollisionProbability } from 'mdml-engine/sketches'
 * lshCollisionProbability(0.8, 20, 5).toFixed(4) // => '0.9996'
 * ```
 */
export function lshCollisionProbability(jaccardSimilarity: number, bands: number, rows: number): number {
  if (jaccardSimilarity <= 0) return 0
  if (jaccardSimilarity >= 1) return 1
  return 1 - (1 - jaccardSimilarity ** rows) ** bands
}

function bandBucketKey(values: Uint32Array, band: number, rows: number): number {
  let h = (0x811c9dc5 ^ band) >>> 0
  const offset = band * rows
  for (let r = 0; r < rows; r++) {
    const v = values[offset + r] ?? 0
    h = fmix32((h ^ v) + Math.imul(r + 1, 0x9e3779b1))
  }
  return h
}

/**
 * Create a banded Locality-Sensitive Hashing (LSH) index over MinHash signatures.
 *
 * @param options - Number of `bands` (`b`) and `rows` per band (`r`)
 * @returns Mutable {@link LshIndex}
 *
 * @example
 * ```ts
 * import { lshIndex, minHash } from 'mdml-engine/sketches'
 * const idx = lshIndex({ bands: 16, rows: 4 })
 * idx.add('doc1', minHash(['a', 'b', 'c', 'd'], { numHashes: 64 }))
 * idx.add('doc2', minHash(['a', 'b', 'c', 'd'], { numHashes: 64 }))
 * idx.candidatePairs() // => [['doc1', 'doc2']]
 * ```
 */
export function lshIndex(options: LshIndexOptions): LshIndex {
  const { bands, rows } = options
  if (!Number.isInteger(bands) || bands < 1) {
    throw new MdmlError('value', `lshIndex: bands must be a positive integer, got ${bands}`)
  }
  if (!Number.isInteger(rows) || rows < 1) {
    throw new MdmlError('value', `lshIndex: rows must be a positive integer, got ${rows}`)
  }

  const tables: Map<number, string[]>[] = Array.from({ length: bands }, () => new Map<number, string[]>())

  return {
    bands,
    rows,
    threshold: (1 / bands) ** (1 / rows),
    add(id: string, signature: MinHashSignature): void {
      if (signature.numHashes < bands * rows) {
        throw new MdmlError(
          'value',
          `lshIndex.add: signature has ${signature.numHashes} hashes, need at least bands * rows = ${bands * rows}`,
        )
      }
      for (let b = 0; b < bands; b++) {
        const key = bandBucketKey(signature.values, b, rows)
        const table = tables[b]
        if (!table) continue
        const bucket = table.get(key)
        if (bucket) {
          if (!bucket.includes(id)) bucket.push(id)
        } else {
          table.set(key, [id])
        }
      }
    },
    query(signature: MinHashSignature): readonly string[] {
      if (signature.numHashes < bands * rows) {
        throw new MdmlError(
          'value',
          `lshIndex.query: signature has ${signature.numHashes} hashes, need at least bands * rows = ${bands * rows}`,
        )
      }
      const candidates = new Set<string>()
      for (let b = 0; b < bands; b++) {
        const key = bandBucketKey(signature.values, b, rows)
        const bucket = tables[b]?.get(key)
        if (!bucket) continue
        for (const id of bucket) {
          candidates.add(id)
        }
      }
      return Array.from(candidates)
    },
    candidatePairs(): readonly (readonly [string, string])[] {
      const pairMap = new Map<string, readonly [string, string]>()
      for (let b = 0; b < bands; b++) {
        const table = tables[b]
        if (!table) continue
        for (const bucket of table.values()) {
          if (bucket.length < 2) continue
          for (let i = 0; i < bucket.length; i++) {
            for (let j = i + 1; j < bucket.length; j++) {
              const id1 = bucket[i]
              const id2 = bucket[j]
              if (id1 === undefined || id2 === undefined) continue
              const pair: readonly [string, string] = id1 <= id2 ? [id1, id2] : [id2, id1]
              pairMap.set(`${pair[0]}\0${pair[1]}`, pair)
            }
          }
        }
      }
      return Array.from(pairMap.values())
    },
    collisionProbability(jaccardSimilarity: number): number {
      return lshCollisionProbability(jaccardSimilarity, bands, rows)
    },
  }
}

/**
 * The exact Jaccard similarity `|A intersect B| / |A union B|` of two collections (as sets; 1 for two empty ones), the
 * value a MinHash estimates.
 *
 * @param a - First collection.
 * @param b - Second collection.
 * @returns The similarity in `[0, 1]`.
 * @concept minhash
 * @example
 * ```ts
 * import { exactJaccard } from 'mdml-engine/sketches'
 * exactJaccard(['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e']) // => 0.6
 * ```
 */
export function exactJaccard(a: Iterable<SketchItem>, b: Iterable<SketchItem>): number {
  const sa = new Set(a)
  const sb = new Set(b)
  if (sa.size === 0 && sb.size === 0) return 1
  let both = 0
  for (const x of sa) if (sb.has(x)) both += 1
  return both / (sa.size + sb.size - both)
}

/** The banding chosen by {@link lshOptimalParams}. */
export interface LshParams {
  /** Number of bands `b`. */
  readonly bands: number
  /** Rows per band `r`. */
  readonly rows: number
  /** The weighted error `w_fp FP + w_fn FN` at this banding. */
  readonly error: number
}

/**
 * The banding `(b, r)` with `b r <= numHashes` that minimises the weighted area of the S-curve's errors for a
 * similarity threshold `t`: the false positives `int_0^t (1 - (1 - s^r)^b) ds` (pairs below the threshold that
 * collide) and the false negatives `int_t^1 (1 - s^r)^b ds` (pairs above it that do not). The integrands are
 * polynomials, integrated exactly by Gauss-Legendre. The same criterion as datasketch's `MinHashLSH`.
 *
 * @param threshold - The Jaccard threshold `t` in `(0, 1)`.
 * @param numHashes - The signature length.
 * @param weights - `[falsePositive, falseNegative]` weights (default `[0.5, 0.5]`).
 * @returns The best `bands`, `rows` and their error.
 * @throws MdmlError with code `value` for a threshold outside `(0, 1)` or `numHashes < 1`.
 * @concept locality-sensitive-hashing
 * @reference J. Leskovec, A. Rajaraman and J. Ullman, Mining of Massive Datasets, section 3.4.
 * @example
 * ```ts
 * import { lshOptimalParams } from 'mdml-engine/sketches'
 * const p = lshOptimalParams(0.5, 128)
 * p.bands * p.rows <= 128 // => true
 * ```
 */
export function lshOptimalParams(
  threshold: number,
  numHashes: number,
  weights: readonly [number, number] = [0.5, 0.5],
): LshParams {
  if (!(threshold > 0 && threshold < 1)) {
    throw new MdmlError('value', `lshOptimalParams: threshold must lie in (0, 1), got ${threshold}`)
  }
  if (!(Number.isInteger(numHashes) && numHashes >= 1)) {
    throw new MdmlError('value', `lshOptimalParams: numHashes must be a positive integer, got ${numHashes}`)
  }
  const rule = gaussLegendre(Math.min(128, Math.ceil(numHashes / 2) + 1))
  const xs = Array.from(rule.nodes.data)
  const ws = Array.from(rule.weights.data)
  const integrate = (f: (s: number) => number, lo: number, hi: number): number => {
    let total = 0
    for (let i = 0; i < xs.length; i++) total += (ws[i] ?? 0) * f(((xs[i] ?? 0) + 1) * 0.5 * (hi - lo) + lo)
    return total * 0.5 * (hi - lo)
  }
  let best: LshParams = { bands: 1, rows: 1, error: Number.POSITIVE_INFINITY }
  for (let b = 1; b <= numHashes; b++) {
    for (let r = 1; r <= Math.floor(numHashes / b); r++) {
      const fp = integrate((s) => 1 - (1 - s ** r) ** b, 0, threshold)
      const fn = integrate((s) => (1 - s ** r) ** b, threshold, 1)
      const error = fp * weights[0] + fn * weights[1]
      if (error < best.error) best = { bands: b, rows: r, error }
    }
  }
  return best
}

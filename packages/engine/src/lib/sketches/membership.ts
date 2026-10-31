/**
 * Approximate set membership with no false negatives: the Bloom filter with its optimal sizing and a stepped insertion
 * algorithm that watches the false-positive rate climb toward the theory, the counting Bloom filter that supports
 * deletion, and the cuckoo filter that stores short fingerprints in two candidate buckets and evicts by cuckoo kicks,
 * with its own stepped algorithm. All hashing is seeded (a number or a `Key`), so a filter is reproducible.
 */

import { type Json, MdmlError, popcount } from '../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { doubleHashIndex, fmix32, hashPair, itemKey, type SketchItem, type SketchSeed, seedOf } from './hashing.ts'

/** Bit count, hash count and expected false-positive rate of a Bloom filter. */
export interface OptimalBloomParams {
  /** Number of bits `m = ceil(-n ln(p) / (ln 2)^2)`. */
  readonly bits: number
  /** Number of hash functions `k = max(1, round((m / n) ln 2))`. */
  readonly hashes: number
  /** Theoretical false-positive probability `(1 - exp(-k n / m))^k` at `n` items. */
  readonly expectedFpr: number
}

/**
 * The optimal bit array length `m` and hash count `k` for a Bloom filter holding `expectedItems` elements at a target
 * false-positive probability `targetFpr` (Bloom 1970): `m = -n ln p / (ln 2)^2`, `k = (m / n) ln 2`.
 *
 * @param expectedItems - Expected number of distinct insertions `n >= 1`.
 * @param targetFpr - Target false-positive rate in `(0, 1)`.
 * @returns `{ bits, hashes, expectedFpr }`, the last the rate the rounded sizes actually give.
 * @throws MdmlError with code `value` when `expectedItems < 1` or `targetFpr` is outside `(0, 1)`.
 * @concept bloom-filter
 * @example
 * ```ts
 * import { optimalBloomParams } from 'mdml-engine/sketches'
 * const p = optimalBloomParams(100, 0.01)
 * [p.bits, p.hashes] // => [959, 7]
 * ```
 */
export function optimalBloomParams(expectedItems: number, targetFpr: number): OptimalBloomParams {
  if (!(expectedItems >= 1 && Number.isFinite(expectedItems))) {
    throw new MdmlError('value', `optimalBloomParams: expectedItems must be >= 1, got ${expectedItems}`)
  }
  if (!(targetFpr > 0 && targetFpr < 1)) {
    throw new MdmlError('value', `optimalBloomParams: targetFpr must lie in (0, 1), got ${targetFpr}`)
  }
  const bits = Math.max(1, Math.ceil((-expectedItems * Math.log(targetFpr)) / (Math.LN2 * Math.LN2)))
  const hashes = Math.max(1, Math.round((bits / expectedItems) * Math.LN2))
  return { bits, hashes, expectedFpr: bloomFalsePositiveRate(bits, hashes, expectedItems) }
}

/**
 * The classical false-positive probability of a Bloom filter with `m` bits and `k` hashes after `n` insertions,
 * `(1 - exp(-k n / m))^k`.
 *
 * @param bits - Number of bits `m`.
 * @param hashes - Number of hash functions `k`.
 * @param items - Number of insertions `n`.
 * @returns The probability that a new item tests positive.
 * @concept bloom-filter
 * @example
 * ```ts
 * import { bloomFalsePositiveRate } from 'mdml-engine/sketches'
 * bloomFalsePositiveRate(959, 7, 100).toFixed(4) // => '0.0100'
 * ```
 */
export function bloomFalsePositiveRate(bits: number, hashes: number, items: number): number {
  return (1 - Math.exp((-hashes * items) / bits)) ** hashes
}

/** Configuration of {@link bloomFilter} and {@link countingBloomFilter}. */
export interface BloomFilterOptions {
  /** Number of bits (or counters) `m`; derived from `expectedItems` and `falsePositiveRate` when omitted (else 1024). */
  readonly bits?: number
  /** Number of hash functions `k`; derived from the same when omitted (else 7). */
  readonly hashes?: number
  /** Expected number of items, used with `falsePositiveRate` when `bits` is omitted. */
  readonly expectedItems?: number
  /** Target false-positive rate in `(0, 1)` for `expectedItems` (default 0.01). */
  readonly falsePositiveRate?: number
  /** Hash seed, a number or a `Key` (default 0). */
  readonly seed?: SketchSeed
}

interface BloomShape {
  readonly bits: number
  readonly hashes: number
  readonly seed: number
}

function resolveBloom(options: BloomFilterOptions, who: string): BloomShape {
  const seed = seedOf(options.seed)
  let bits = options.bits
  let hashes = options.hashes
  if (bits === undefined && options.expectedItems !== undefined) {
    const opt = optimalBloomParams(options.expectedItems, options.falsePositiveRate ?? 0.01)
    bits = opt.bits
    hashes ??= opt.hashes
  }
  bits ??= 1024
  hashes ??= 7
  if (!(Number.isInteger(bits) && bits >= 1))
    throw new MdmlError('value', `${who}: bits must be a positive integer, got ${bits}`)
  if (!(Number.isInteger(hashes) && hashes >= 1)) {
    throw new MdmlError('value', `${who}: hashes must be a positive integer, got ${hashes}`)
  }
  return { bits, hashes, seed }
}

function setBloomBits(words: Uint32Array, shape: BloomShape, item: SketchItem): void {
  const [h1, h2] = hashPair(item, shape.seed)
  for (let i = 0; i < shape.hashes; i++) {
    const bit = doubleHashIndex(h1, h2, i, shape.bits)
    words[bit >>> 5] = (words[bit >>> 5] ?? 0) | (1 << (bit & 31))
  }
}

function testBloomBits(words: Uint32Array, shape: BloomShape, item: SketchItem): boolean {
  const [h1, h2] = hashPair(item, shape.seed)
  for (let i = 0; i < shape.hashes; i++) {
    const bit = doubleHashIndex(h1, h2, i, shape.bits)
    if (((words[bit >>> 5] ?? 0) & (1 << (bit & 31))) === 0) return false
  }
  return true
}

function countSetBits(words: Uint32Array): number {
  let set = 0
  for (let i = 0; i < words.length; i++) set += popcount(words[i] ?? 0)
  return set
}

/** A Bloom filter: a bit array and `k` hash probes per item. */
export interface BloomFilter {
  /** Number of bits `m`. */
  readonly bits: number
  /** Number of hash functions `k`. */
  readonly hashes: number
  /** The resolved 32-bit hash seed. */
  readonly seed: number
  /** The packed bit array (`ceil(m / 32)` words). */
  readonly words: Uint32Array
  /** Insertions so far (repeats count). */
  readonly inserted: number
  /** Insert an item. */
  add(item: SketchItem): void
  /** Whether an item may be in the set: never false for an inserted item, false positives at the theoretical rate. */
  has(item: SketchItem): boolean
  /** Fraction of the bits that are set. */
  fillRatio(): number
  /** `(1 - exp(-k n / m))^k` for `n` items (default the number inserted). */
  theoreticalFpr(items?: number): number
  /** The false-positive rate implied by the fill: `fillRatio^k`, which needs no count of distinct items. */
  observedFpr(): number
  /** The distinct-item estimate of Swamidass and Baldi, `-(m / k) ln(1 - X / m)` for `X` set bits. */
  estimateCount(): number
  /** The union of two filters of the same shape and seed (bitwise OR), as a new filter. */
  merge(other: BloomFilter): BloomFilter
  /** Clear every bit. */
  clear(): void
}

function makeBloom(shape: BloomShape, words: Uint32Array, insertedAtStart: number): BloomFilter {
  let inserted = insertedAtStart
  const fill = (): number => countSetBits(words) / shape.bits
  return {
    bits: shape.bits,
    hashes: shape.hashes,
    seed: shape.seed,
    words,
    get inserted() {
      return inserted
    },
    add: (item) => {
      setBloomBits(words, shape, item)
      inserted += 1
    },
    has: (item) => testBloomBits(words, shape, item),
    fillRatio: fill,
    theoreticalFpr: (items = inserted) => bloomFalsePositiveRate(shape.bits, shape.hashes, items),
    observedFpr: () => fill() ** shape.hashes,
    estimateCount: () => {
      const x = countSetBits(words)
      return x >= shape.bits ? Number.POSITIVE_INFINITY : -(shape.bits / shape.hashes) * Math.log(1 - x / shape.bits)
    },
    merge: (other) => {
      if (other.bits !== shape.bits || other.hashes !== shape.hashes || other.seed !== shape.seed) {
        throw new MdmlError('shape', 'bloomFilter.merge: filters must share bits, hashes and seed')
      }
      const merged = new Uint32Array(words.length)
      for (let i = 0; i < merged.length; i++) merged[i] = (words[i] ?? 0) | (other.words[i] ?? 0)
      return makeBloom(shape, merged, inserted + other.inserted)
    },
    clear: () => {
      words.fill(0)
      inserted = 0
    },
  }
}

/**
 * A Bloom filter (Bloom 1970) over a packed bit array with Kirsch-Mitzenmacher double hashing: `add` sets `k` bits,
 * `has` tests them. Items inserted always test positive; an item never inserted tests positive with probability
 * `(1 - exp(-k n / m))^k` after `n` insertions.
 *
 * @param options - `bits` and `hashes`, or `expectedItems` and `falsePositiveRate` to size them, and `seed`.
 * @returns The filter.
 * @throws MdmlError with code `value` for non-positive sizes.
 * @concept bloom-filter
 * @reference B. H. Bloom, Space/time trade-offs in hash coding with allowable errors (Communications of the ACM, 1970).
 * @example
 * ```ts
 * import { bloomFilter } from 'mdml-engine/sketches'
 * const bf = bloomFilter({ expectedItems: 100, falsePositiveRate: 0.01 })
 * bf.add('alpha')
 * bf.add('beta')
 * [bf.has('alpha'), bf.has('beta'), bf.hashes] // => [true, true, 7]
 * ```
 */
export function bloomFilter(options: BloomFilterOptions = {}): BloomFilter {
  const shape = resolveBloom(options, 'bloomFilter')
  return makeBloom(shape, new Uint32Array(Math.ceil(shape.bits / 32)), 0)
}

/** The state of {@link bloomFilterAlgorithm}. */
export interface BloomFilterStepState {
  /** The packed bit array. */
  readonly words: Uint32Array
  /** Items inserted so far. */
  readonly inserted: number
  /** Fraction of bits set. */
  readonly fillRatio: number
  /** The theoretical false-positive rate at `inserted` items. */
  readonly theoreticalFpr: number
  /** The measured rate on the negative probes. */
  readonly empiricalFpr: number
}

/** Input of {@link bloomFilterAlgorithm}: a Bloom filter's options and the stream. */
export interface BloomFilterAlgorithmOptions extends BloomFilterOptions {
  /** The stream, one item inserted per step. */
  readonly items: readonly SketchItem[]
  /** Items known not to be in the stream, to measure the false-positive rate on (default 400 generated strings). */
  readonly queries?: readonly SketchItem[]
}

/**
 * Bloom filter insertion as an algorithm, one item per step, logging the fill ratio, the theoretical false-positive
 * rate and the rate measured on negative probes, so the measured curve can be watched following the theoretical one.
 *
 * @param options - The filter's shape and seed, the stream, and optionally the negative probes.
 * @returns The algorithm.
 * @concept bloom-filter
 * @example
 * ```ts
 * import { bloomFilterAlgorithm } from 'mdml-engine/sketches'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const items = Array.from({ length: 30 }, (_, i) => `item-${i}`)
 * const res = runAlgorithm(bloomFilterAlgorithm({ items, bits: 512, hashes: 4 }), { maxSteps: 100 })
 * [res.steps, res.state.inserted, res.state.fillRatio > 0.1] // => [30, 30, true]
 * ```
 */
export function bloomFilterAlgorithm(options: BloomFilterAlgorithmOptions): Algorithm<BloomFilterStepState> {
  const shape = resolveBloom(options, 'bloomFilterAlgorithm')
  const items = options.items
  const inStream = new Set(items.map(itemKey))
  const probes: readonly SketchItem[] =
    options.queries ??
    Array.from({ length: 400 }, (_, i) => `~negative-probe-${i}`).filter((q) => !inStream.has(itemKey(q)))
  const measure = (words: Uint32Array, inserted: number): BloomFilterStepState => {
    let hits = 0
    for (const q of probes) if (testBloomBits(words, shape, q)) hits += 1
    return {
      words,
      inserted,
      fillRatio: countSetBits(words) / shape.bits,
      theoreticalFpr: bloomFalsePositiveRate(shape.bits, shape.hashes, inserted),
      empiricalFpr: probes.length === 0 ? 0 : hits / probes.length,
    }
  }
  return {
    name: 'bloom-filter',
    config: { bits: shape.bits, hashes: shape.hashes, seed: shape.seed, items: items.length, probes: probes.length },
    schema: channels({
      metrics: {
        fillRatio: { description: 'fraction of the bits that are set' },
        theoreticalFpr: { description: 'the false-positive rate (1 - exp(-k n / m))^k at n items' },
        empiricalFpr: { description: 'the false-positive rate measured on items never inserted' },
      },
    }),
    init: () => measure(new Uint32Array(Math.ceil(shape.bits / 32)), 0),
    step: (s) => {
      const item = items[s.inserted]
      if (item === undefined) return s
      const words = Uint32Array.from(s.words)
      setBloomBits(words, shape, item)
      return measure(words, s.inserted + 1)
    },
    done: (s) => (s.inserted >= items.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { fillRatio: s.fillRatio, theoreticalFpr: s.theoreticalFpr, empiricalFpr: s.empiricalFpr },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as BloomFilterStepState,
  }
}

/** A counting Bloom filter: counters in place of bits, so items can be removed. */
export interface CountingBloomFilter {
  /** Number of counters `m`. */
  readonly buckets: number
  /** Number of hash functions `k`. */
  readonly hashes: number
  /** The resolved 32-bit hash seed. */
  readonly seed: number
  /** The counters. */
  readonly counts: Uint32Array
  /** Net insertions (added minus removed). */
  readonly size: number
  /** Increment the `k` counters of an item. */
  add(item: SketchItem): void
  /** Decrement the `k` counters of an item that tests positive; `false` (and no change) for one that does not. */
  remove(item: SketchItem): boolean
  /** Whether all `k` counters of an item are positive. */
  has(item: SketchItem): boolean
  /** The smallest of an item's `k` counters: an upper bound on how many times it was inserted net of removals. */
  count(item: SketchItem): number
}

/**
 * A counting Bloom filter (Fan, Cao, Almeida and Broder 2000): the bits become counters, so removing an item that was
 * inserted leaves no false negative for the others. Removing an item that was never inserted but tests positive
 * (a false positive) corrupts the filter, as in the original.
 *
 * @param options - As {@link bloomFilter}; `bits` is the number of counters.
 * @returns The filter.
 * @throws MdmlError with code `value` for non-positive sizes.
 * @concept bloom-filter
 * @reference L. Fan, P. Cao, J. Almeida and A. Z. Broder, Summary cache: a scalable wide-area web cache sharing
 *   protocol (IEEE/ACM Transactions on Networking, 2000).
 * @example
 * ```ts
 * import { countingBloomFilter } from 'mdml-engine/sketches'
 * const cbf = countingBloomFilter({ bits: 256, hashes: 3 })
 * cbf.add('x')
 * cbf.add('y')
 * cbf.remove('x')
 * [cbf.has('x'), cbf.has('y'), cbf.size] // => [false, true, 1]
 * ```
 */
export function countingBloomFilter(options: BloomFilterOptions = {}): CountingBloomFilter {
  const { bits: buckets, hashes, seed } = resolveBloom(options, 'countingBloomFilter')
  const counts = new Uint32Array(buckets)
  let size = 0
  const slots = (item: SketchItem): number[] => {
    const [h1, h2] = hashPair(item, seed)
    return Array.from({ length: hashes }, (_, i) => doubleHashIndex(h1, h2, i, buckets))
  }
  const count = (item: SketchItem): number => {
    let min = Number.POSITIVE_INFINITY
    for (const idx of slots(item)) min = Math.min(min, counts[idx] ?? 0)
    return min
  }
  return {
    buckets,
    hashes,
    seed,
    counts,
    get size() {
      return size
    },
    add: (item) => {
      for (const idx of slots(item)) counts[idx] = (counts[idx] ?? 0) + 1
      size += 1
    },
    remove: (item) => {
      if (count(item) === 0) return false
      for (const idx of slots(item)) counts[idx] = Math.max(0, (counts[idx] ?? 0) - 1)
      size = Math.max(0, size - 1)
      return true
    },
    has: (item) => count(item) > 0,
    count,
  }
}

/** Configuration of {@link cuckooFilter}. */
export interface CuckooFilterOptions {
  /** Target number of items; the bucket count is the next power of two of `capacity / bucketSize` (default 256). */
  readonly capacity?: number
  /** Slots per bucket `b` (default 4). */
  readonly bucketSize?: number
  /** Bits per fingerprint `f` in `[4, 16]` (default 12): the false-positive rate is at most about `2 b / 2^f`. */
  readonly fingerprintBits?: number
  /** Most relocations of one insertion before it is refused (default 500). */
  readonly maxKicks?: number
  /** Seed of the hashes and of the eviction choices, a number or a `Key` (default 0). */
  readonly seed?: SketchSeed
}

interface CuckooGeometry {
  readonly numBuckets: number
  readonly bucketSize: number
  readonly fingerprintBits: number
  readonly maxKicks: number
  readonly seed: number
}

function resolveCuckoo(options: CuckooFilterOptions): CuckooGeometry {
  const capacity = options.capacity ?? 256
  const bucketSize = options.bucketSize ?? 4
  const fingerprintBits = options.fingerprintBits ?? 12
  const maxKicks = options.maxKicks ?? 500
  if (!(Number.isInteger(capacity) && capacity >= 1)) {
    throw new MdmlError('value', `cuckooFilter: capacity must be a positive integer, got ${capacity}`)
  }
  if (!(Number.isInteger(bucketSize) && bucketSize >= 1)) {
    throw new MdmlError('value', `cuckooFilter: bucketSize must be a positive integer, got ${bucketSize}`)
  }
  if (!(Number.isInteger(fingerprintBits) && fingerprintBits >= 4 && fingerprintBits <= 16)) {
    throw new MdmlError('value', `cuckooFilter: fingerprintBits must be in [4, 16], got ${fingerprintBits}`)
  }
  if (!(Number.isInteger(maxKicks) && maxKicks >= 0)) {
    throw new MdmlError('value', `cuckooFilter: maxKicks must be a non-negative integer, got ${maxKicks}`)
  }
  let numBuckets = 2
  while (numBuckets * bucketSize < capacity) numBuckets *= 2
  return { numBuckets, bucketSize, fingerprintBits, maxKicks, seed: seedOf(options.seed) }
}

interface CuckooSites {
  readonly i1: number
  readonly i2: number
  readonly fp: number
}

function cuckooAlt(g: CuckooGeometry, bucket: number, fp: number): number {
  const mask = g.numBuckets - 1
  // The offset depends on the fingerprint only, so the map is an involution: alt(alt(i)) = i.
  const offset = fmix32(Math.imul(fp, 0x5bd1e995) ^ g.seed) & mask || 1
  return (bucket ^ offset) & mask
}

function cuckooSites(g: CuckooGeometry, item: SketchItem): CuckooSites {
  const [h1, h2] = hashPair(item, g.seed)
  const raw = h2 & ((1 << g.fingerprintBits) - 1)
  const fp = raw === 0 ? 1 : raw
  const i1 = h1 & (g.numBuckets - 1)
  return { i1, i2: cuckooAlt(g, i1, fp), fp }
}

function slotOf(slots: Uint16Array, g: CuckooGeometry, bucket: number, fp: number): number {
  const base = bucket * g.bucketSize
  for (let s = 0; s < g.bucketSize; s++) if (slots[base + s] === fp) return base + s
  return -1
}

function placeInBucket(slots: Uint16Array, g: CuckooGeometry, bucket: number, fp: number): boolean {
  const at = slotOf(slots, g, bucket, 0)
  if (at < 0) return false
  slots[at] = fp
  return true
}

interface CuckooInsertion {
  readonly ok: boolean
  readonly kicks: number
  readonly rng: number
}

// Insert a fingerprint by partial-key cuckoo hashing. A failed insertion is rolled back, so a refusal never loses an
// item that was already stored (the textbook version drops the last evicted fingerprint).
function cuckooInsert(
  slots: Uint16Array,
  g: CuckooGeometry,
  { i1, i2, fp }: CuckooSites,
  rngIn: number,
): CuckooInsertion {
  if (placeInBucket(slots, g, i1, fp) || placeInBucket(slots, g, i2, fp)) return { ok: true, kicks: 0, rng: rngIn }
  let rng = fmix32(rngIn + 0x9e3779b9)
  let bucket = (rng & 1) === 0 ? i1 : i2
  let current = fp
  const undo: [number, number][] = []
  for (let k = 1; k <= g.maxKicks; k++) {
    rng = fmix32(rng + k)
    const at = bucket * g.bucketSize + (rng % g.bucketSize)
    const evicted = slots[at] ?? 1
    undo.push([at, evicted])
    slots[at] = current
    current = evicted
    bucket = cuckooAlt(g, bucket, current)
    if (placeInBucket(slots, g, bucket, current)) return { ok: true, kicks: k, rng }
  }
  for (let j = undo.length - 1; j >= 0; j--) {
    const [at, old] = undo[j] as [number, number]
    slots[at] = old
  }
  return { ok: false, kicks: g.maxKicks, rng }
}

/** A cuckoo filter: fingerprints in two candidate buckets, with deletion. */
export interface CuckooFilter {
  /** Number of buckets (a power of two). */
  readonly numBuckets: number
  /** Slots per bucket. */
  readonly bucketSize: number
  /** Bits per fingerprint. */
  readonly fingerprintBits: number
  /** Most relocations of one insertion. */
  readonly maxKicks: number
  /** Fingerprints stored. */
  readonly size: number
  /** The slots (0 is empty). */
  readonly buckets: Uint16Array
  /** Relocations done by the most recent `add`. */
  readonly lastKicks: number
  /** Insert an item: `true`, or `false` (leaving the filter unchanged) when `maxKicks` relocations did not find room. */
  add(item: SketchItem): boolean
  /** Whether an item's fingerprint is in either of its two buckets. */
  has(item: SketchItem): boolean
  /** Delete one copy of an item's fingerprint if present. Only items actually inserted may be removed. */
  remove(item: SketchItem): boolean
  /** Fraction of the slots occupied. */
  loadFactor(): number
}

/**
 * A cuckoo filter (Fan, Andersen, Kaminsky and Mitzenmacher 2014): an item is a non-zero fingerprint `fp` stored in
 * bucket `i1 = h(x)` or `i2 = i1 xor h(fp)`, so the second bucket is computable from the fingerprint alone and
 * evictions need no access to the original items. With four slots per bucket the table fills to about 95% before
 * insertions fail; the false-positive rate is about `2 b alpha / 2^f` at load `alpha`.
 *
 * @param options - Capacity, bucket size, fingerprint bits, maximum kicks and seed.
 * @returns The filter.
 * @throws MdmlError with code `value` for invalid sizes.
 * @concept cuckoo-filter
 * @reference B. Fan, D. G. Andersen, M. Kaminsky and M. D. Mitzenmacher, Cuckoo filter: practically better than Bloom
 *   (CoNEXT 2014).
 * @example
 * ```ts
 * import { cuckooFilter } from 'mdml-engine/sketches'
 * const cf = cuckooFilter({ capacity: 64 })
 * cf.add('key-1')
 * const present = cf.has('key-1')
 * cf.remove('key-1')
 * [present, cf.has('key-1'), cf.size] // => [true, false, 0]
 * ```
 */
export function cuckooFilter(options: CuckooFilterOptions = {}): CuckooFilter {
  const g = resolveCuckoo(options)
  const slots = new Uint16Array(g.numBuckets * g.bucketSize)
  let size = 0
  let lastKicks = 0
  let rng = fmix32(g.seed ^ 0x2545f491)
  return {
    numBuckets: g.numBuckets,
    bucketSize: g.bucketSize,
    fingerprintBits: g.fingerprintBits,
    maxKicks: g.maxKicks,
    buckets: slots,
    get size() {
      return size
    },
    get lastKicks() {
      return lastKicks
    },
    add: (item) => {
      const r = cuckooInsert(slots, g, cuckooSites(g, item), rng)
      rng = r.rng
      lastKicks = r.kicks
      if (r.ok) size += 1
      return r.ok
    },
    has: (item) => {
      const { i1, i2, fp } = cuckooSites(g, item)
      return slotOf(slots, g, i1, fp) >= 0 || slotOf(slots, g, i2, fp) >= 0
    },
    remove: (item) => {
      const { i1, i2, fp } = cuckooSites(g, item)
      let at = slotOf(slots, g, i1, fp)
      if (at < 0) at = slotOf(slots, g, i2, fp)
      if (at < 0) return false
      slots[at] = 0
      size -= 1
      return true
    },
    loadFactor: () => size / slots.length,
  }
}

/** The state of {@link cuckooFilterAlgorithm}. */
export interface CuckooStepState {
  /** The slots (0 is empty). */
  readonly slots: Uint16Array
  /** Items offered so far. */
  readonly offered: number
  /** Fingerprints stored. */
  readonly size: number
  /** Insertions refused. */
  readonly failures: number
  /** Relocations done by the last insertion. */
  readonly kicks: number
  /** The state of the eviction generator. */
  readonly rng: number
}

/** Input of {@link cuckooFilterAlgorithm}: a cuckoo filter's options and the stream. */
export interface CuckooFilterAlgorithmOptions extends CuckooFilterOptions {
  /** The stream, one item offered per step. */
  readonly items: readonly SketchItem[]
}

/**
 * Cuckoo filter insertion as an algorithm, one item per step, logging the load factor, the relocations the insertion
 * needed and the cumulative refusals: the kicks stay near zero until the table is nearly full and then climb.
 *
 * @param options - The filter's options and the stream.
 * @returns The algorithm.
 * @concept cuckoo-filter
 * @example
 * ```ts
 * import { cuckooFilterAlgorithm } from 'mdml-engine/sketches'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const items = Array.from({ length: 40 }, (_, i) => `k${i}`)
 * const res = runAlgorithm(cuckooFilterAlgorithm({ items, capacity: 64 }), { maxSteps: 100 })
 * [res.state.size, res.state.failures] // => [40, 0]
 * ```
 */
export function cuckooFilterAlgorithm(options: CuckooFilterAlgorithmOptions): Algorithm<CuckooStepState> {
  const g = resolveCuckoo(options)
  const items = options.items
  const total = g.numBuckets * g.bucketSize
  return {
    name: 'cuckoo-filter',
    config: { ...g, items: items.length },
    schema: channels({
      metrics: {
        loadFactor: { description: 'fraction of the slots occupied' },
        kicks: { description: 'relocations the last insertion needed' },
        failures: { description: 'insertions refused so far' },
      },
    }),
    init: () => ({
      slots: new Uint16Array(total),
      offered: 0,
      size: 0,
      failures: 0,
      kicks: 0,
      rng: fmix32(g.seed ^ 0x2545f491),
    }),
    step: (s) => {
      const item = items[s.offered]
      if (item === undefined) return s
      const slots = Uint16Array.from(s.slots)
      const r = cuckooInsert(slots, g, cuckooSites(g, item), s.rng)
      return {
        slots: r.ok ? slots : s.slots,
        offered: s.offered + 1,
        size: s.size + (r.ok ? 1 : 0),
        failures: s.failures + (r.ok ? 0 : 1),
        kicks: r.kicks,
        rng: r.rng,
      }
    },
    done: (s) => (s.offered >= items.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: { loadFactor: s.size / total, kicks: s.kicks, failures: s.failures } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as CuckooStepState,
  }
}

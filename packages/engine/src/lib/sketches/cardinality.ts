/**
 * Distinct counting with HyperLogLog (Flajolet, Fusy, Gandouet and Meunier 2007): `2^p` registers each keeping the
 * longest run of leading zeros seen in the hashes routed to it, estimated either by the classical harmonic mean with
 * linear counting for small counts or by Ertl's improved estimator, which needs no empirical bias tables and is
 * accurate from empty to saturation. Sketches merge by register-wise maximum, so the union of streams is a merge.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { hashPair, itemKey, type SketchItem, type SketchSeed, seedOf } from './hashing.ts'

/** Configuration of {@link hyperLogLog}. */
export interface HyperLogLogOptions {
  /** Precision `p` in `[4, 18]`, giving `m = 2^p` registers (default 10). */
  readonly precision?: number
  /** Hash seed, a number or a `Key` (default 0). */
  readonly seed?: SketchSeed
}

/** Which estimator {@link HyperLogLog.count} uses. */
export type HyperLogLogEstimator = 'improved' | 'classic'

/** A HyperLogLog sketch. */
export interface HyperLogLog {
  /** The precision `p`. */
  readonly precision: number
  /** Number of registers `m = 2^p`. */
  readonly numRegisters: number
  /** The registers: the longest leading-zero run plus one seen by each (0 if none). */
  readonly registers: Uint8Array
  /** The resolved 32-bit hash seed. */
  readonly seed: number
  /** The asymptotic relative standard error `1.04 / sqrt(m)` of the classical estimator. */
  readonly standardError: number
  /** Insert an item (repeats change nothing). */
  add(item: SketchItem): void
  /** The estimated number of distinct items: `'improved'` (Ertl 2017, default) or `'classic'` (harmonic mean, linear counting below `5m/2`). */
  count(estimator?: HyperLogLogEstimator): number
  /** The union of this sketch and another of the same precision and seed, as a new sketch. */
  merge(other: HyperLogLog): HyperLogLog
}

const MAX_PRECISION = 18

function resolvePrecision(precision: number | undefined, who: string): number {
  const p = precision ?? 10
  if (!(Number.isInteger(p) && p >= 4 && p <= MAX_PRECISION)) {
    throw new OmaiError('value', `${who}: precision must be an integer in [4, ${MAX_PRECISION}], got ${p}`)
  }
  return p
}

// The register and rank of an item: the top p bits of the 64-bit hash pick the register, the rank is the position of
// the first 1-bit in the remaining 64 - p bits (so in [1, 65 - p]).
function locate(item: SketchItem, p: number, seed: number): { readonly index: number; readonly rank: number } {
  const [h1, h2] = hashPair(item, seed)
  const index = h1 >>> (32 - p)
  const lead = Math.clz32((h1 << p) >>> 0)
  return { index, rank: lead < 32 - p ? lead + 1 : 32 - p + Math.clz32(h2) + 1 }
}

function alphaOf(m: number): number {
  if (m === 16) return 0.673
  if (m === 32) return 0.697
  if (m === 64) return 0.709
  return 0.7213 / (1 + 1.079 / m)
}

// Ertl's sigma(x) = x + sum_{k>=1} x^(2^k) 2^(k-1), the small-count correction.
function ertlSigma(x: number): number {
  if (x === 1) return Number.POSITIVE_INFINITY
  let y = 1
  let z = x
  let t = x
  let previous: number
  do {
    t *= t
    previous = z
    z += t * y
    y += y
  } while (z !== previous)
  return z
}

// Ertl's tau(x) = (1 - x - sum_{k>=1} (1 - x^(2^-k))^2 2^-k) / 3, the saturation correction.
function ertlTau(x: number): number {
  if (x === 0 || x === 1) return 0
  let y = 1
  let z = 1 - x
  let t = x
  let previous: number
  do {
    t = Math.sqrt(t)
    previous = z
    y *= 0.5
    z -= (1 - t) ** 2 * y
  } while (z !== previous)
  return z / 3
}

/**
 * The HyperLogLog cardinality estimate of a register array, by Ertl's improved estimator
 * (`m^2 / (2 ln 2) / (m sigma(C_0 / m) + sum_k C_k 2^-k + m tau(1 - C_{q+1} / m) 2^-q)`) or by the classical harmonic
 * mean with linear counting when it is at most `5m/2`.
 *
 * @param registers - The `m = 2^p` registers, hashes of `64 - p` bits assumed.
 * @param estimator - `'improved'` (default) or `'classic'`.
 * @returns The estimated number of distinct items.
 * @concept hyperloglog
 * @reference O. Ertl, New cardinality estimation algorithms for HyperLogLog sketches (arXiv 1702.01284, 2017).
 * @example
 * ```ts
 * import { hyperLogLogEstimate } from 'omai-engine/sketches'
 * hyperLogLogEstimate(new Uint8Array(16)) // => 0
 * hyperLogLogEstimate(Uint8Array.from([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'classic').toFixed(3) // => '2.137'
 * ```
 */
export function hyperLogLogEstimate(registers: Uint8Array, estimator: HyperLogLogEstimator = 'improved'): number {
  const m = registers.length
  if (estimator === 'classic') {
    let harmonic = 0
    let zeros = 0
    for (const r of registers) {
      harmonic += 2 ** -r
      if (r === 0) zeros += 1
    }
    const raw = (alphaOf(m) * m * m) / harmonic
    return raw <= 2.5 * m && zeros > 0 ? m * Math.log(m / zeros) : raw
  }
  const q = 64 - Math.log2(m)
  const counts = new Float64Array(q + 2)
  for (const r of registers) counts[r] = (counts[r] ?? 0) + 1
  let z = m * ertlTau(1 - (counts[q + 1] ?? 0) / m)
  for (let k = q; k >= 1; k--) z = 0.5 * (z + (counts[k] ?? 0))
  z += m * ertlSigma((counts[0] ?? 0) / m)
  return z === Number.POSITIVE_INFINITY ? 0 : m ** 2 / (2 * Math.LN2 * z)
}

function makeHll(precision: number, seed: number, registers: Uint8Array): HyperLogLog {
  const m = registers.length
  return {
    precision,
    numRegisters: m,
    registers,
    seed,
    standardError: 1.04 / Math.sqrt(m),
    add: (item) => {
      const { index, rank } = locate(item, precision, seed)
      if (rank > (registers[index] ?? 0)) registers[index] = rank
    },
    count: (estimator = 'improved') => hyperLogLogEstimate(registers, estimator),
    merge: (other) => hllMerge(makeHll(precision, seed, registers), other),
  }
}

/**
 * A HyperLogLog distinct-count sketch with `2^p` registers: relative standard error about `1.04 / sqrt(2^p)` (about 3%
 * at the default precision 10, in a kilobyte), whatever the number of distinct items up to 2^(64-p).
 *
 * @param options - Precision `p` and seed.
 * @returns The sketch.
 * @throws OmaiError with code `value` for a precision outside `[4, 18]`.
 * @concept hyperloglog
 * @reference P. Flajolet, E. Fusy, O. Gandouet and F. Meunier, HyperLogLog: the analysis of a near-optimal cardinality
 *   estimation algorithm (AofA 2007).
 * @index structure HyperLogLog distinct-count sketch.
 * @example
 * ```ts
 * import { hyperLogLog } from 'omai-engine/sketches'
 * const hll = hyperLogLog({ precision: 12 })
 * for (let i = 0; i < 5000; i++) hll.add(`user-${i % 2000}`)
 * Math.abs(hll.count() - 2000) < 100 // => true
 * ```
 */
export function hyperLogLog(options: HyperLogLogOptions = {}): HyperLogLog {
  const precision = resolvePrecision(options.precision, 'hyperLogLog')
  return makeHll(precision, seedOf(options.seed), new Uint8Array(1 << precision))
}

/**
 * Merge two HyperLogLog sketches of the same precision and seed by the register-wise maximum, which is exactly the
 * sketch of the union of the two streams.
 *
 * @param a - First sketch.
 * @param b - Second sketch.
 * @returns A new sketch whose count estimates `|A union B|`.
 * @throws OmaiError with code `shape` when the precisions or seeds differ.
 * @concept hyperloglog
 * @example
 * ```ts
 * import { hllMerge, hyperLogLog } from 'omai-engine/sketches'
 * const a = hyperLogLog()
 * const b = hyperLogLog()
 * for (let i = 0; i < 300; i++) a.add(i)
 * for (let i = 200; i < 500; i++) b.add(i)
 * Math.abs(hllMerge(a, b).count() - 500) < 40 // => true
 * ```
 */
export function hllMerge(a: HyperLogLog, b: HyperLogLog): HyperLogLog {
  if (a.precision !== b.precision || a.seed !== b.seed) {
    throw new OmaiError('shape', 'hllMerge: sketches must share precision and seed')
  }
  const merged = new Uint8Array(a.numRegisters)
  for (let i = 0; i < merged.length; i++) merged[i] = Math.max(a.registers[i] ?? 0, b.registers[i] ?? 0)
  return makeHll(a.precision, a.seed, merged)
}

/** The state of {@link hyperLogLogAlgorithm}. */
export interface HyperLogLogState {
  /** The registers. */
  readonly registers: Uint8Array
  /** Items seen. */
  readonly seen: number
  /** The number of distinct items among them (exact, for reference). */
  readonly trueCardinality: number
  /** The estimate. */
  readonly estimate: number
  /** `|estimate - true| / max(1, true)`. */
  readonly relativeError: number
}

/** Input of {@link hyperLogLogAlgorithm}. */
export interface HyperLogLogAlgorithmOptions extends HyperLogLogOptions {
  /** The stream, one item per step. */
  readonly items: readonly SketchItem[]
  /** The estimator (default `'improved'`). */
  readonly estimator?: HyperLogLogEstimator
}

/**
 * HyperLogLog over a stream as an algorithm, one item per step, logging the estimate, the exact distinct count kept
 * beside it and the relative error (which should stay within a few multiples of `1.04 / sqrt(m)`).
 *
 * @param options - Precision, seed, estimator and the stream.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a precision outside `[4, 18]`.
 * @concept hyperloglog
 * @example
 * ```ts
 * import { hyperLogLogAlgorithm } from 'omai-engine/sketches'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const res = runAlgorithm(hyperLogLogAlgorithm({ items: [1, 2, 3, 2, 1, 4], precision: 6 }), { maxSteps: 10 })
 * res.state.trueCardinality // => 4
 * ```
 */
export function hyperLogLogAlgorithm(options: HyperLogLogAlgorithmOptions): Algorithm<HyperLogLogState> {
  const precision = resolvePrecision(options.precision, 'hyperLogLogAlgorithm')
  const seed = seedOf(options.seed)
  const estimator = options.estimator ?? 'improved'
  const items = options.items
  const distinctAfter: number[] = [0]
  const seenKeys = new Set<string>()
  for (const item of items) {
    seenKeys.add(itemKey(item))
    distinctAfter.push(seenKeys.size)
  }
  const report = (registers: Uint8Array, seen: number): HyperLogLogState => {
    const trueCardinality = distinctAfter[seen] ?? 0
    const estimate = hyperLogLogEstimate(registers, estimator)
    return {
      registers,
      seen,
      trueCardinality,
      estimate,
      relativeError: Math.abs(estimate - trueCardinality) / Math.max(1, trueCardinality),
    }
  }
  return {
    name: 'hyperloglog',
    config: { precision, seed, estimator, items: items.length },
    schema: channels({
      metrics: {
        estimate: { description: 'the estimated number of distinct items' },
        trueCardinality: { description: 'the exact number of distinct items so far' },
        relativeError: { description: '|estimate - true| / max(1, true)', direction: 'lower' },
      },
      tensors: { registers: { role: 'state', description: 'the 2^p registers', shape: [1 << precision] } },
    }),
    init: () => report(new Uint8Array(1 << precision), 0),
    step: (s) => {
      const item = items[s.seen]
      if (item === undefined) return s
      const registers = Uint8Array.from(s.registers)
      const { index, rank } = locate(item, precision, seed)
      if (rank > (registers[index] ?? 0)) registers[index] = rank
      return report(registers, s.seen + 1)
    },
    done: (s) => (s.seen >= items.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { estimate: s.estimate, trueCardinality: s.trueCardinality, relativeError: s.relativeError },
      tensors: { registers: { role: 'state', value: s.registers } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as HyperLogLogState,
  }
}

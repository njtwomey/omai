/**
 * Frequency estimation and heavy hitters over a stream: the Count-Min sketch (one-sided error, with the conservative
 * update), the Count sketch (signed, unbiased, median of rows), and the counter-based summaries Misra-Gries and
 * Space-Saving, whose errors are bounded by `N / (k + 1)` and `N / k` for `k` counters. The table sketches merge by
 * addition; each has its error guarantee stated in its documentation and checked in its property tests.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { fmix32, hash32, itemFromKey, itemKey, type SketchItem, type SketchSeed, seedOf } from './hashing.ts'

/** Configuration of {@link countMinSketch}. */
export interface CountMinOptions {
  /** Buckets per row `w`; default `ceil(e / epsilon)`. */
  readonly width?: number
  /** Rows `d`; default `ceil(ln(1 / delta))`. */
  readonly depth?: number
  /** The error factor: the overestimate is at most `epsilon N` (default 0.01), used when `width` is omitted. */
  readonly epsilon?: number
  /** The failure probability of that bound (default 0.01), used when `depth` is omitted. */
  readonly delta?: number
  /** Conservative update: raise only the counters at the current minimum. Never worse, usually much better; not mergeable. */
  readonly conservative?: boolean
  /** Hash seed, a number or a `Key` (default 0). */
  readonly seed?: SketchSeed
}

interface TableShape {
  readonly width: number
  readonly depth: number
  readonly seed: number
}

function rowSeed(seed: number, row: number): number {
  return fmix32(seed + Math.imul(row + 1, 0x9e3779b1))
}

function resolveCountMin(options: CountMinOptions): TableShape & { readonly epsilon: number; readonly delta: number } {
  const epsilon = options.epsilon ?? 0.01
  const delta = options.delta ?? 0.01
  if (options.width === undefined && !(epsilon > 0 && epsilon < 1)) {
    throw new MdmlError('value', `countMinSketch: epsilon must lie in (0, 1), got ${epsilon}`)
  }
  if (options.depth === undefined && !(delta > 0 && delta < 1)) {
    throw new MdmlError('value', `countMinSketch: delta must lie in (0, 1), got ${delta}`)
  }
  const width = options.width ?? Math.ceil(Math.E / epsilon)
  const depth = options.depth ?? Math.max(1, Math.ceil(Math.log(1 / delta)))
  if (!(Number.isInteger(width) && width >= 1)) {
    throw new MdmlError('value', `countMinSketch: width must be a positive integer, got ${width}`)
  }
  if (!(Number.isInteger(depth) && depth >= 1)) {
    throw new MdmlError('value', `countMinSketch: depth must be a positive integer, got ${depth}`)
  }
  return { width, depth, seed: seedOf(options.seed), epsilon: Math.E / width, delta: Math.exp(-depth) }
}

function cmCells(shape: TableShape, item: SketchItem): number[] {
  return Array.from(
    { length: shape.depth },
    (_, r) => r * shape.width + (hash32(item, rowSeed(shape.seed, r)) % shape.width),
  )
}

function cmEstimate(table: Float64Array, shape: TableShape, item: SketchItem): number {
  let min = Number.POSITIVE_INFINITY
  for (const cell of cmCells(shape, item)) min = Math.min(min, table[cell] ?? 0)
  return min
}

function cmAdd(table: Float64Array, shape: TableShape, item: SketchItem, count: number, conservative: boolean): void {
  const cells = cmCells(shape, item)
  if (conservative) {
    const target = cmEstimate(table, shape, item) + count
    for (const cell of cells) table[cell] = Math.max(table[cell] ?? 0, target)
  } else {
    for (const cell of cells) table[cell] = (table[cell] ?? 0) + count
  }
}

/**
 * A Count-Min sketch (Cormode and Muthukrishnan 2005): a `d x w` table of counters, one hash per row. The estimate is
 * the row-minimum, never below the true count, and above it by at most `epsilon N` with probability at least
 * `1 - delta`, where `w = ceil(e / epsilon)`, `d = ceil(ln(1 / delta))` and `N` is the total count added.
 */
export interface CountMinSketch {
  /** Buckets per row. */
  readonly width: number
  /** Rows. */
  readonly depth: number
  /** The error factor `e / w`: overestimates are at most `epsilon N` with probability `1 - delta`. */
  readonly epsilon: number
  /** The failure probability `exp(-d)`. */
  readonly delta: number
  /** Whether updates are conservative. */
  readonly conservative: boolean
  /** Total count added, `N`. */
  readonly totalCount: number
  /** The counters, row-major `[depth * width]`. */
  readonly table: Float64Array
  /** Add `count` (non-negative, default 1) occurrences of an item. */
  add(item: SketchItem, count?: number): void
  /** The estimated count of an item: at least the true count. */
  estimate(item: SketchItem): number
  /** The sketch of the combined streams, for sketches of the same shape and seed (cell-wise sum). */
  merge(other: CountMinSketch): CountMinSketch
}

function makeCountMin(
  shape: ReturnType<typeof resolveCountMin>,
  conservative: boolean,
  table: Float64Array,
  startTotal: number,
): CountMinSketch {
  let total = startTotal
  return {
    width: shape.width,
    depth: shape.depth,
    epsilon: shape.epsilon,
    delta: shape.delta,
    conservative,
    table,
    get totalCount() {
      return total
    },
    add: (item, count = 1) => {
      if (!(count >= 0 && Number.isFinite(count))) {
        throw new MdmlError('value', `countMinSketch.add: count must be non-negative and finite, got ${count}`)
      }
      total += count
      cmAdd(table, shape, item, count, conservative)
    },
    estimate: (item) => cmEstimate(table, shape, item),
    merge: (other) => {
      if (other.width !== shape.width || other.depth !== shape.depth) {
        throw new MdmlError('shape', 'countMinSketch.merge: dimensions must match')
      }
      if (conservative || other.conservative) {
        throw new MdmlError(
          'value',
          'countMinSketch.merge: conservative sketches do not merge (the sum overestimates more)',
        )
      }
      const merged = new Float64Array(table.length)
      for (let i = 0; i < merged.length; i++) merged[i] = (table[i] ?? 0) + (other.table[i] ?? 0)
      return makeCountMin(shape, false, merged, total + other.totalCount)
    },
  }
}

/**
 * Create a Count-Min sketch.
 *
 * @param options - `width` and `depth`, or `epsilon` and `delta`; `conservative`; `seed`.
 * @returns The sketch.
 * @throws MdmlError with code `value` for invalid dimensions.
 * @concept count-min-sketch
 * @reference G. Cormode and S. Muthukrishnan, An improved data stream summary: the count-min sketch and its
 *   applications (Journal of Algorithms, 2005).
 * @index structure Count-Min sketch for stream frequencies.
 * @example
 * ```ts
 * import { countMinSketch } from 'mdml-engine/sketches'
 * const cms = countMinSketch({ epsilon: 0.01, delta: 0.01 })
 * cms.add('a', 5)
 * cms.add('b', 2)
 * [cms.estimate('a'), cms.estimate('b'), cms.width, cms.depth] // => [5, 2, 272, 5]
 * ```
 */
export function countMinSketch(options: CountMinOptions = {}): CountMinSketch {
  const shape = resolveCountMin(options)
  return makeCountMin(shape, options.conservative ?? false, new Float64Array(shape.width * shape.depth), 0)
}

/** The state of {@link countMinAlgorithm}. */
export interface CountMinState {
  /** The counters. */
  readonly table: Float64Array
  /** Items seen. */
  readonly seen: number
  /** The exact counts so far, by `itemKey`. */
  readonly truth: Readonly<Record<string, number>>
  /** The largest overestimate over the distinct items seen. */
  readonly maxOverestimate: number
  /** The mean overestimate over the distinct items seen. */
  readonly meanOverestimate: number
  /** The guaranteed bound `epsilon N`. */
  readonly bound: number
}

/** Input of {@link countMinAlgorithm}. */
export interface CountMinAlgorithmOptions extends CountMinOptions {
  /** The stream, one item per step. */
  readonly items: readonly SketchItem[]
}

/**
 * Count-Min over a stream as an algorithm, one item per step, logging the largest and mean overestimate against the
 * exact counts kept beside it, and the bound `epsilon N` they should stay under.
 *
 * @param options - The sketch's shape and the stream.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for invalid dimensions.
 * @concept count-min-sketch
 * @example
 * ```ts
 * import { countMinAlgorithm } from 'mdml-engine/sketches'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const res = runAlgorithm(countMinAlgorithm({ items: ['a', 'b', 'a', 'c', 'a'], width: 64, depth: 4 }), { maxSteps: 9 })
 * [res.state.seen, res.state.truth['s:a'], res.state.maxOverestimate] // => [5, 3, 0]
 * ```
 */
export function countMinAlgorithm(options: CountMinAlgorithmOptions): Algorithm<CountMinState> {
  const shape = resolveCountMin(options)
  const conservative = options.conservative ?? false
  const items = options.items
  const measure = (table: Float64Array, seen: number, truth: Readonly<Record<string, number>>): CountMinState => {
    let max = 0
    let sum = 0
    const keys = Object.keys(truth)
    for (const key of keys) {
      const over = cmEstimate(table, shape, itemFromKey(key)) - (truth[key] ?? 0)
      max = Math.max(max, over)
      sum += over
    }
    return {
      table,
      seen,
      truth,
      maxOverestimate: max,
      meanOverestimate: keys.length === 0 ? 0 : sum / keys.length,
      bound: shape.epsilon * seen,
    }
  }
  return {
    name: 'count-min-sketch',
    config: { width: shape.width, depth: shape.depth, seed: shape.seed, conservative, items: items.length },
    schema: channels({
      metrics: {
        maxOverestimate: {
          description: 'largest estimate minus true count over the distinct items seen',
          direction: 'lower',
        },
        meanOverestimate: {
          description: 'mean estimate minus true count over the distinct items seen',
          direction: 'lower',
        },
        bound: {
          description: 'the guaranteed overestimate bound epsilon N (holds with probability 1 - delta per item)',
        },
      },
    }),
    init: () => measure(new Float64Array(shape.width * shape.depth), 0, {}),
    step: (s) => {
      const item = items[s.seen]
      if (item === undefined) return s
      const table = Float64Array.from(s.table)
      cmAdd(table, shape, item, 1, conservative)
      const key = itemKey(item)
      return measure(table, s.seen + 1, { ...s.truth, [key]: (s.truth[key] ?? 0) + 1 })
    },
    done: (s) => (s.seen >= items.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { maxOverestimate: s.maxOverestimate, meanOverestimate: s.meanOverestimate, bound: s.bound },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as CountMinState,
  }
}

/** Configuration of {@link countSketch}. */
export interface CountSketchOptions {
  /** Buckets per row `w`; default `ceil(3 / epsilon^2)`. */
  readonly width?: number
  /** Rows `d` (made odd, for a median); default `ceil(ln(1 / delta))`. */
  readonly depth?: number
  /** The L2 error factor (default 0.1), used when `width` is omitted. */
  readonly epsilon?: number
  /** The failure probability (default 0.01), used when `depth` is omitted. */
  readonly delta?: number
  /** Hash seed, a number or a `Key` (default 0). */
  readonly seed?: SketchSeed
}

/**
 * A Count sketch (Charikar, Chen and Farach-Colton 2002): like Count-Min but each row multiplies the update by a random
 * sign, so a row's estimate `s(x) T[h(x)]` is unbiased and the median over rows concentrates: the error is at most
 * `epsilon ||f||_2` with probability `1 - delta` for `w = 3 / epsilon^2`. Updates may be negative.
 */
export interface CountSketch {
  /** Buckets per row. */
  readonly width: number
  /** Rows (odd). */
  readonly depth: number
  /** The signed counters, row-major. */
  readonly table: Float64Array
  /** Add `count` (any sign, default 1) occurrences of an item. */
  add(item: SketchItem, count?: number): void
  /** The median over rows of the signed counter: unbiased for the item's count. */
  estimate(item: SketchItem): number
  /** The estimated second moment `sum_x f_x^2`: the median over rows of the sum of squared counters. */
  secondMoment(): number
  /** The sketch of the combined streams, for sketches of the same shape and seed (cell-wise sum). */
  merge(other: CountSketch): CountSketch
}

function makeCountSketch(width: number, depth: number, seed: number, table: Float64Array): CountSketch {
  const cell = (item: SketchItem, r: number): { readonly at: number; readonly sign: number } => {
    const s = rowSeed(seed, r)
    return {
      at: r * width + (hash32(item, s) % width),
      sign: (hash32(item, s ^ 0x5bd1e995) & 1) === 0 ? 1 : -1,
    }
  }
  return {
    width,
    depth,
    table,
    add: (item, count = 1) => {
      for (let r = 0; r < depth; r++) {
        const { at, sign } = cell(item, r)
        table[at] = (table[at] ?? 0) + sign * count
      }
    },
    estimate: (item) => {
      const rows = Array.from({ length: depth }, (_, r) => {
        const { at, sign } = cell(item, r)
        return sign * (table[at] ?? 0)
      }).sort((a, b) => a - b)
      return rows[depth >> 1] ?? 0
    },
    secondMoment: () => {
      const rows = Array.from({ length: depth }, (_, r) => {
        let s = 0
        for (let c = 0; c < width; c++) s += (table[r * width + c] ?? 0) ** 2
        return s
      }).sort((a, b) => a - b)
      return rows[depth >> 1] ?? 0
    },
    merge: (other) => {
      if (other.width !== width || other.depth !== depth) {
        throw new MdmlError('shape', 'countSketch.merge: dimensions must match')
      }
      const merged = new Float64Array(table.length)
      for (let i = 0; i < merged.length; i++) merged[i] = (table[i] ?? 0) + (other.table[i] ?? 0)
      return makeCountSketch(width, depth, seed, merged)
    },
  }
}

/**
 * Create a Count sketch.
 *
 * @param options - `width` and `depth`, or `epsilon` and `delta`; `seed`.
 * @returns The sketch.
 * @throws MdmlError with code `value` for invalid dimensions.
 * @concept count-sketch
 * @reference M. Charikar, K. Chen and M. Farach-Colton, Finding frequent items in data streams (ICALP 2002).
 * @example
 * ```ts
 * import { countSketch } from 'mdml-engine/sketches'
 * const cs = countSketch({ width: 256, depth: 5 })
 * cs.add('alpha', 10)
 * cs.add('beta', -3)
 * [cs.estimate('alpha'), cs.estimate('beta')] // => [10, -3]
 * ```
 */
export function countSketch(options: CountSketchOptions = {}): CountSketch {
  const epsilon = options.epsilon ?? 0.1
  const delta = options.delta ?? 0.01
  const width = options.width ?? Math.ceil(3 / (epsilon * epsilon))
  const rawDepth = options.depth ?? Math.max(1, Math.ceil(Math.log(1 / delta)))
  const depth = rawDepth % 2 === 0 ? rawDepth + 1 : rawDepth
  if (!(Number.isInteger(width) && width >= 1)) {
    throw new MdmlError('value', `countSketch: width must be a positive integer, got ${width}`)
  }
  if (!(Number.isInteger(rawDepth) && rawDepth >= 1)) {
    throw new MdmlError('value', `countSketch: depth must be a positive integer, got ${rawDepth}`)
  }
  return makeCountSketch(width, depth, seedOf(options.seed), new Float64Array(width * depth))
}

/** A monitored item of a counter-based summary. */
export interface SpaceSavingEntry {
  /** The item. */
  readonly item: SketchItem
  /** Its counter: `true count <= count <= true count + error`. */
  readonly count: number
  /** The most the counter can overestimate (the counter it replaced). */
  readonly error: number
}

/**
 * Space-Saving (Metwally, Agrawal and El Abbadi 2005): `k` counters; an unmonitored item replaces the smallest counter
 * and inherits its value as the error. Every counter overestimates by at most its `error <= N / k`, every item with
 * true frequency above `N / k` is monitored, and the counters sum to exactly `N`.
 */
export interface SpaceSavingTopK {
  /** Number of counters. */
  readonly k: number
  /** Total weight added, `N`. */
  readonly totalCount: number
  /** The guarantee `N / k` on every counter's error. */
  readonly guaranteedBound: number
  /** Add `count` (positive, default 1) occurrences of an item. */
  add(item: SketchItem, count?: number): void
  /** The bounds on an item's count: a monitored item's counter and error; an unmonitored one is at most the smallest counter. */
  estimate(item: SketchItem): { readonly count: number; readonly error: number; readonly lowerBound: number }
  /** The monitored entries, largest counter first. */
  entries(): readonly SpaceSavingEntry[]
  /** Every entry that may have frequency above `phi N` (counter above it): no true heavy hitter is missing. */
  heavyHitters(phi: number): readonly SpaceSavingEntry[]
  /** The entries certainly above `phi N` (counter minus error above it): no false positive. */
  guaranteedHeavyHitters(phi: number): readonly SpaceSavingEntry[]
}

function spaceSavingAdd(entries: SpaceSavingEntry[], k: number, item: SketchItem, count: number): void {
  const at = entries.findIndex((e) => e.item === item)
  if (at >= 0) {
    const e = entries[at] as SpaceSavingEntry
    entries[at] = { ...e, count: e.count + count }
    return
  }
  if (entries.length < k) {
    entries.push({ item, count, error: 0 })
    return
  }
  let min = 0
  for (let i = 1; i < entries.length; i++) if ((entries[i]?.count ?? 0) < (entries[min]?.count ?? 0)) min = i
  const floor = entries[min]?.count ?? 0
  entries[min] = { item, count: floor + count, error: floor }
}

function sortedEntries(entries: readonly SpaceSavingEntry[]): SpaceSavingEntry[] {
  return [...entries].sort((a, b) => b.count - a.count)
}

/**
 * Create a Space-Saving heavy-hitters summary with `k` counters.
 *
 * @param k - Number of counters (`k >= 1`).
 * @returns The summary.
 * @throws MdmlError with code `value` for a non-positive or non-integer `k`.
 * @concept heavy-hitters
 * @reference A. Metwally, D. Agrawal and A. El Abbadi, Efficient computation of frequent and top-k elements in data
 *   streams (ICDT 2005).
 * @example
 * ```ts
 * import { spaceSavingTopK } from 'mdml-engine/sketches'
 * const ss = spaceSavingTopK(3)
 * for (const x of ['a', 'a', 'a', 'b', 'b', 'c', 'd']) ss.add(x)
 * ss.entries().map((e) => e.item) // => ['a', 'b', 'd']
 * ```
 */
export function spaceSavingTopK(k: number): SpaceSavingTopK {
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `spaceSavingTopK: k must be a positive integer, got ${k}`)
  const entries: SpaceSavingEntry[] = []
  let total = 0
  const smallest = (): number => (entries.length < k ? 0 : Math.min(...entries.map((e) => e.count)))
  return {
    k,
    get totalCount() {
      return total
    },
    get guaranteedBound() {
      return total / k
    },
    add: (item, count = 1) => {
      if (!(count > 0 && Number.isFinite(count))) {
        throw new MdmlError('value', `spaceSavingTopK.add: count must be positive and finite, got ${count}`)
      }
      total += count
      spaceSavingAdd(entries, k, item, count)
    },
    estimate: (item) => {
      const e = entries.find((x) => x.item === item)
      if (e !== undefined) return { count: e.count, error: e.error, lowerBound: e.count - e.error }
      const floor = smallest()
      return { count: floor, error: floor, lowerBound: 0 }
    },
    entries: () => sortedEntries(entries),
    heavyHitters: (phi) => sortedEntries(entries).filter((e) => e.count > phi * total),
    guaranteedHeavyHitters: (phi) => sortedEntries(entries).filter((e) => e.count - e.error > phi * total),
  }
}

/** The state of {@link spaceSavingAlgorithm}. */
export interface SpaceSavingState {
  /** The monitored entries. */
  readonly entries: readonly SpaceSavingEntry[]
  /** Items seen. */
  readonly seen: number
  /** The exact counts so far, by `itemKey`. */
  readonly truth: Readonly<Record<string, number>>
  /** The largest `counter - true count` over the monitored items. */
  readonly maxError: number
  /** The guarantee `N / k`. */
  readonly bound: number
}

/** Input of {@link spaceSavingAlgorithm}. */
export interface SpaceSavingAlgorithmOptions {
  /** The stream, one item per step. */
  readonly items: readonly SketchItem[]
  /** Number of counters. */
  readonly k: number
}

/**
 * Space-Saving over a stream as an algorithm, one item per step, logging the largest counter error against the
 * exact counts and the guarantee `N / k` above it, and the entries as a probe.
 *
 * @param options - The stream and the number of counters.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a non-positive or non-integer `k`.
 * @concept heavy-hitters
 * @example
 * ```ts
 * import { spaceSavingAlgorithm } from 'mdml-engine/sketches'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const res = runAlgorithm(spaceSavingAlgorithm({ items: ['x', 'x', 'y', 'z', 'x'], k: 2 }), { maxSteps: 9 })
 * res.state.entries.map((e) => e.item) // => ['x', 'z']
 * ```
 */
export function spaceSavingAlgorithm(options: SpaceSavingAlgorithmOptions): Algorithm<SpaceSavingState> {
  const { items, k } = options
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `spaceSavingAlgorithm: k must be a positive integer, got ${k}`)
  const measure = (
    entries: readonly SpaceSavingEntry[],
    seen: number,
    truth: Readonly<Record<string, number>>,
  ): SpaceSavingState => {
    let maxError = 0
    for (const e of entries) maxError = Math.max(maxError, e.count - (truth[itemKey(e.item)] ?? 0))
    return { entries, seen, truth, maxError, bound: seen / k }
  }
  return {
    name: 'space-saving',
    config: { k, items: items.length },
    schema: channels({
      metrics: {
        maxError: { description: 'largest counter minus true count over the monitored items', direction: 'lower' },
        bound: { description: 'the guaranteed bound N / k on every counter error' },
        monitored: { description: 'number of monitored items' },
      },
    }),
    init: () => measure([], 0, {}),
    step: (s) => {
      const item = items[s.seen]
      if (item === undefined) return s
      const entries = [...s.entries]
      spaceSavingAdd(entries, k, item, 1)
      const key = itemKey(item)
      return measure(entries, s.seen + 1, { ...s.truth, [key]: (s.truth[key] ?? 0) + 1 })
    },
    done: (s) => (s.seen >= items.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: { maxError: s.maxError, bound: s.bound, monitored: s.entries.length } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as SpaceSavingState,
  }
}

/**
 * A Misra-Gries summary (Misra and Gries 1982; Karp, Shenker and Papadimitriou 2003): at most `k` counters; an item with
 * no counter, when all `k` are taken, decrements every counter by one (dropping zeros). Each counter underestimates its
 * item by at most `N / (k + 1)`, and every item with frequency above `N / (k + 1)` keeps a counter. Summaries merge
 * (Agarwal et al. 2012) without losing the bound.
 */
export interface MisraGries {
  /** Maximum number of counters. */
  readonly k: number
  /** Total number of items added, `N`. */
  readonly totalCount: number
  /** The guarantee `N / (k + 1)` on every underestimate. */
  readonly guaranteedBound: number
  /** Add one occurrence of an item. */
  add(item: SketchItem): void
  /** The counter of an item (0 if none): `true count - N/(k+1) <= estimate <= true count`. */
  estimate(item: SketchItem): number
  /** The counters, largest first. */
  counters(): readonly { readonly item: SketchItem; readonly count: number }[]
  /** Combine with another summary of the same `k`: add the counters, subtract the `(k+1)`-th largest, keep positives. */
  merge(other: MisraGries): MisraGries
}

function makeMisraGries(k: number, start: Map<SketchItem, number>, startTotal: number): MisraGries {
  const counts = start
  let total = startTotal
  const counters = (): { readonly item: SketchItem; readonly count: number }[] =>
    [...counts].map(([item, count]) => ({ item, count })).sort((a, b) => b.count - a.count)
  return {
    k,
    get totalCount() {
      return total
    },
    get guaranteedBound() {
      return total / (k + 1)
    },
    add: (item) => {
      total += 1
      const have = counts.get(item)
      if (have !== undefined) counts.set(item, have + 1)
      else if (counts.size < k) counts.set(item, 1)
      else {
        for (const [key, c] of [...counts]) {
          if (c <= 1) counts.delete(key)
          else counts.set(key, c - 1)
        }
      }
    },
    estimate: (item) => counts.get(item) ?? 0,
    counters,
    merge: (other) => {
      if (other.k !== k) throw new MdmlError('shape', 'misraGries.merge: summaries must share k')
      const sum = new Map(counts)
      for (const { item, count } of other.counters()) sum.set(item, (sum.get(item) ?? 0) + count)
      const sorted = [...sum].sort((a, b) => b[1] - a[1])
      const cut = sorted[k]?.[1] ?? 0
      const kept = new Map<SketchItem, number>()
      for (const [item, c] of sorted) if (c - cut > 0) kept.set(item, c - cut)
      return makeMisraGries(k, kept, total + other.totalCount)
    },
  }
}

/**
 * Create a Misra-Gries heavy-hitters summary with at most `k` counters.
 *
 * @param k - Maximum number of counters (`k >= 1`).
 * @returns The summary.
 * @throws MdmlError with code `value` for a non-positive or non-integer `k`.
 * @concept heavy-hitters
 * @reference J. Misra and D. Gries, Finding repeated elements (Science of Computer Programming, 1982).
 * @example
 * ```ts
 * import { misraGries } from 'mdml-engine/sketches'
 * const mg = misraGries(2)
 * for (const x of ['a', 'a', 'a', 'b', 'c', 'a']) mg.add(x)
 * [mg.estimate('a'), mg.estimate('b')] // => [3, 0]
 * ```
 */
export function misraGries(k: number): MisraGries {
  if (!(Number.isInteger(k) && k >= 1))
    throw new MdmlError('value', `misraGries: k must be a positive integer, got ${k}`)
  return makeMisraGries(k, new Map(), 0)
}

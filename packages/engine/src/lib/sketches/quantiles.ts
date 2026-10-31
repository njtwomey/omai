/**
 * Quantile sketches that summarise a stream of numbers in small space: the Greenwald-Khanna summary (deterministic,
 * every rank within `epsilon n`), the merging t-digest (relative accuracy that tightens toward both tails), and the KLL
 * compactor hierarchy (randomised, rank error about `1 / k` with high probability, mergeable). Each answers `quantile`,
 * `cdf` and `rank`; the randomised one draws its coins from a `Key`.
 */

import { MdmlError } from '../base/index.ts'
import { type Key, uniformStream } from '../random/index.ts'

/** What the quantile sketches share. */
export interface QuantileSketch {
  /** Number of values (total weight) added. */
  readonly count: number
  /** The smallest value added (`NaN` if none). */
  readonly min: number
  /** The largest value added (`NaN` if none). */
  readonly max: number
  /** Add a value. */
  add(value: number): void
  /** The value at fraction `q` of the sorted stream, `q` in `[0, 1]` (`NaN` if empty). */
  quantile(q: number): number
  /** The estimated fraction of the stream at most `x`. */
  cdf(x: number): number
}

function checkFinite(value: number, who: string): void {
  if (!Number.isFinite(value)) throw new MdmlError('value', `${who}: value must be finite, got ${value}`)
}

function checkQuantile(q: number, who: string): void {
  if (!(q >= 0 && q <= 1)) throw new MdmlError('value', `${who}: q must lie in [0, 1], got ${q}`)
}

/** A tuple of the Greenwald-Khanna summary: a value, the rank gap `g` to its predecessor, and the rank slack `delta`. */
export interface GkTuple {
  /** The stored value. */
  readonly value: number
  /** `rmin(v_i) - rmin(v_{i-1})`. */
  readonly g: number
  /** `rmax(v_i) - rmin(v_i)`. */
  readonly delta: number
}

/** The Greenwald-Khanna summary: every quantile is answered with rank error at most `epsilon n`. */
export interface GkSummary extends QuantileSketch {
  /** The rank accuracy `epsilon`. */
  readonly epsilon: number
  /** The stored tuples in increasing value order. */
  tuples(): readonly GkTuple[]
  /** The value whose true rank is within `epsilon n` of `ceil(q n)`. */
  quantile(q: number): number
  /** An estimate of the number of values at most `x`, within `epsilon n`. */
  rank(x: number): number
}

/**
 * The Greenwald-Khanna quantile summary (2001) with accuracy `epsilon`. Tuples `(v, g, delta)` keep
 * `g + delta <= 2 epsilon n`, so `rmin(v_i) = sum_{j<=i} g_j` and `rmax = rmin + delta` bracket the rank of `v_i` within
 * `2 epsilon n`, and the quantile query returns a value whose rank is within `epsilon n` of the target. It is
 * deterministic; this version merges adjacent tuples by the simple rule (the paper's banded rule gives the
 * `O(log(epsilon n) / epsilon)` space bound, which this rule does not claim).
 *
 * @param epsilon - The rank error as a fraction of `n`, in `(0, 1)`.
 * @returns The summary.
 * @throws MdmlError with code `value` for `epsilon` outside `(0, 1)`.
 * @concept quantile-sketch
 * @reference M. Greenwald and S. Khanna, Space-efficient online computation of quantile summaries (SIGMOD 2001).
 * @example
 * ```ts
 * import { gkSummary } from 'mdml-engine/sketches'
 * const gk = gkSummary(0.01)
 * for (let i = 1; i <= 1000; i++) gk.add(i)
 * Math.abs(gk.quantile(0.5) - 500) <= 0.01 * 1000 // => true
 * ```
 */
export function gkSummary(epsilon: number): GkSummary {
  if (!(epsilon > 0 && epsilon < 1))
    throw new MdmlError('value', `gkSummary: epsilon must lie in (0, 1), got ${epsilon}`)
  const tuples: { value: number; g: number; delta: number }[] = []
  let n = 0
  let min = Number.NaN
  let max = Number.NaN
  const period = Math.max(1, Math.floor(1 / (2 * epsilon)))
  const compress = (): void => {
    const cap = Math.floor(2 * epsilon * n)
    for (let i = tuples.length - 2; i >= 1; i--) {
      const here = tuples[i]
      const next = tuples[i + 1]
      if (here !== undefined && next !== undefined && here.g + next.g + next.delta <= cap) {
        next.g += here.g
        tuples.splice(i, 1)
      }
    }
  }
  const ranks = (): { rmin: number; rmax: number; value: number }[] => {
    let r = 0
    return tuples.map((t) => {
      r += t.g
      return { rmin: r, rmax: r + t.delta, value: t.value }
    })
  }
  // x lies between two stored values: its rank is at least the predecessor's rmin and below the successor's rmax, a
  // bracket of width at most 2 epsilon n, so its midpoint is within epsilon n.
  const rank = (x: number): number => {
    const r = ranks()
    const after = r.findIndex((t) => t.value > x)
    const lower = after === 0 ? 0 : (r[(after < 0 ? r.length : after) - 1]?.rmin ?? 0)
    const upper = after < 0 ? n : (r[after]?.rmax ?? n) - 1
    return (lower + upper) / 2
  }
  return {
    epsilon,
    get count() {
      return n
    },
    get min() {
      return min
    },
    get max() {
      return max
    },
    tuples: () => tuples.map((t) => ({ ...t })),
    add: (value) => {
      checkFinite(value, 'gkSummary.add')
      let at = 0
      while (at < tuples.length && (tuples[at]?.value ?? 0) <= value) at++
      // A new minimum or maximum is known exactly; otherwise its rank slack is its successor's, less itself.
      const succ = tuples[at]
      const delta = at === 0 || succ === undefined ? 0 : succ.g + succ.delta - 1
      tuples.splice(at, 0, { value, g: 1, delta })
      n += 1
      if (Number.isNaN(min) || value < min) min = value
      if (Number.isNaN(max) || value > max) max = value
      if (n % period === 0) compress()
    },
    quantile: (q) => {
      checkQuantile(q, 'gkSummary.quantile')
      if (n === 0) return Number.NaN
      const target = Math.max(1, Math.ceil(q * n))
      const slack = epsilon * n
      const r = ranks()
      for (const t of r) if (target - t.rmin <= slack && t.rmax - target <= slack) return t.value
      let best = r[0]
      for (const t of r) {
        if (
          best === undefined ||
          Math.abs((t.rmin + t.rmax) / 2 - target) < Math.abs((best.rmin + best.rmax) / 2 - target)
        )
          best = t
      }
      return best?.value ?? Number.NaN
    },
    rank,
    cdf: (x) => (n === 0 ? Number.NaN : x < min ? 0 : x >= max ? 1 : rank(x) / n),
  }
}

/** A centroid of a t-digest: a mean and the weight it stands for. */
export interface TDigestCentroid {
  /** Mean of the values merged into the centroid. */
  readonly mean: number
  /** Total weight of those values. */
  readonly weight: number
}

/** Configuration of {@link tDigest}. */
export interface TDigestOptions {
  /** The compression `delta` (default 100): about `delta` centroids are kept, and the rank error is about `1 / delta`. */
  readonly compression?: number
}

/** The merging t-digest. */
export interface TDigest extends QuantileSketch {
  /** The compression. */
  readonly compression: number
  /** Total weight added. */
  readonly totalWeight: number
  /** Add a value with a positive weight (default 1). */
  add(value: number, weight?: number): void
  /** The compressed centroids in increasing mean order. */
  centroids(): readonly TDigestCentroid[]
  /** The digest of both streams (the centroids of both, recompressed). */
  merge(other: TDigest): TDigest
}

// The k_1 scale function k(q) = delta / (2 pi) asin(2q - 1): steep at both ends, so tail centroids stay small.
const scaleK = (q: number, compression: number): number =>
  (compression / (2 * Math.PI)) * Math.asin(2 * Math.min(1, Math.max(0, q)) - 1)

function compressCentroids(input: readonly TDigestCentroid[], total: number, compression: number): TDigestCentroid[] {
  if (input.length <= 1) return input.slice()
  const sorted = input.slice().sort((a, b) => a.mean - b.mean)
  const out: TDigestCentroid[] = []
  let current = sorted[0] as TDigestCentroid
  let before = 0
  let limit = scaleK(0, compression) + 1
  for (let i = 1; i < sorted.length; i++) {
    const c = sorted[i] as TDigestCentroid
    const weight = current.weight + c.weight
    if (scaleK((before + weight) / total, compression) <= limit) {
      current = { mean: (current.mean * current.weight + c.mean * c.weight) / weight, weight }
    } else {
      out.push(current)
      before += current.weight
      limit = scaleK(before / total, compression) + 1
      current = c
    }
  }
  out.push(current)
  return out
}

function makeTDigest(
  compression: number,
  start: TDigestCentroid[],
  startWeight: number,
  startMin: number,
  startMax: number,
): TDigest {
  let centroids = start
  const buffer: TDigestCentroid[] = []
  let total = startWeight
  let min = startMin
  let max = startMax
  const limit = Math.ceil(compression * 5)
  const flush = (): void => {
    if (buffer.length === 0) return
    centroids = compressCentroids(centroids.concat(buffer), total, compression)
    buffer.length = 0
  }
  const cdf = (x: number): number => {
    flush()
    if (total === 0) return Number.NaN
    if (x < min) return 0
    if (x >= max) return 1
    const first = centroids[0] as TDigestCentroid
    const last = centroids[centroids.length - 1] as TDigestCentroid
    if (x < first.mean) {
      // Between the minimum and the first centre: half the first centroid's weight is spread over it.
      return first.mean > min ? ((first.weight / 2) * ((x - min) / (first.mean - min))) / total : 0
    }
    if (x >= last.mean) {
      return last.mean < max ? 1 - ((last.weight / 2) * ((max - x) / (max - last.mean))) / total : 1
    }
    let cum = 0
    for (let i = 0; i < centroids.length - 1; i++) {
      const a = centroids[i] as TDigestCentroid
      const b = centroids[i + 1] as TDigestCentroid
      const leftCentre = cum + a.weight / 2
      const gap = (a.weight + b.weight) / 2
      if (x < b.mean) {
        return (leftCentre + (b.mean > a.mean ? ((x - a.mean) / (b.mean - a.mean)) * gap : gap / 2)) / total
      }
      cum += a.weight
    }
    return 1
  }
  return {
    compression,
    get count() {
      return total
    },
    get totalWeight() {
      return total
    },
    get min() {
      return min
    },
    get max() {
      return max
    },
    add: (value, weight = 1) => {
      checkFinite(value, 'tDigest.add')
      if (!(weight > 0 && Number.isFinite(weight))) {
        throw new MdmlError('value', `tDigest.add: weight must be positive and finite, got ${weight}`)
      }
      buffer.push({ mean: value, weight })
      total += weight
      if (Number.isNaN(min) || value < min) min = value
      if (Number.isNaN(max) || value > max) max = value
      if (buffer.length >= limit) flush()
    },
    quantile: (q) => {
      checkQuantile(q, 'tDigest.quantile')
      flush()
      if (total === 0) return Number.NaN
      const first = centroids[0] as TDigestCentroid
      const last = centroids[centroids.length - 1] as TDigestCentroid
      if (centroids.length === 1) return first.mean
      const index = q * total
      if (index <= 1) return min
      if (index >= total - 1) return max
      if (index < first.weight / 2)
        return min + ((first.mean - min) * (index - 1)) / Math.max(first.weight / 2 - 1, 1e-300)
      if (index > total - last.weight / 2) {
        return max - ((max - last.mean) * (total - index - 1)) / Math.max(last.weight / 2 - 1, 1e-300)
      }
      let cum = 0
      for (let i = 0; i < centroids.length - 1; i++) {
        const a = centroids[i] as TDigestCentroid
        const b = centroids[i + 1] as TDigestCentroid
        const leftCentre = cum + a.weight / 2
        const gap = (a.weight + b.weight) / 2
        if (index < leftCentre + gap) return a.mean + ((b.mean - a.mean) * (index - leftCentre)) / gap
        cum += a.weight
      }
      return max
    },
    cdf,
    centroids: () => {
      flush()
      return centroids.slice()
    },
    merge: (other) => {
      flush()
      const weight = total + other.totalWeight
      const both = compressCentroids(centroids.concat(other.centroids()), weight, compression)
      const lo = Number.isNaN(min) ? other.min : Number.isNaN(other.min) ? min : Math.min(min, other.min)
      const hi = Number.isNaN(max) ? other.max : Number.isNaN(other.max) ? max : Math.max(max, other.max)
      return makeTDigest(compression, both, weight, lo, hi)
    },
  }
}

/**
 * A merging t-digest (Dunning and Ertl 2019): values are buffered, then sorted with the centroids and merged left to
 * right while each merged centroid spans at most one unit of the scale `k(q) = delta / (2 pi) asin(2q - 1)`. Centroids
 * near `q = 0` and `q = 1` hold a handful of values, so tail quantiles are far more accurate (in relative terms) than
 * the middle ones, which carry weight about `n / delta`.
 *
 * @param options - The `compression` (default 100, at least 10).
 * @returns The digest.
 * @throws MdmlError with code `value` for a compression below 10.
 * @concept quantile-sketch
 * @reference T. Dunning and O. Ertl, Computing extremely accurate quantiles using t-digests (arXiv 1902.04023, 2019).
 * @example
 * ```ts
 * import { tDigest } from 'mdml-engine/sketches'
 * const td = tDigest({ compression: 100 })
 * for (let i = 1; i <= 10000; i++) td.add(i)
 * Math.abs(td.quantile(0.5) - 5000) < 100 // => true
 * ```
 */
export function tDigest(options: TDigestOptions = {}): TDigest {
  const compression = options.compression ?? 100
  if (!(compression >= 10)) throw new MdmlError('value', `tDigest: compression must be at least 10, got ${compression}`)
  return makeTDigest(compression, [], 0, Number.NaN, Number.NaN)
}

/** Configuration of {@link kllSketch}. */
export interface KllSketchOptions {
  /** The size parameter `k >= 8` (default 200): rank error about `1.65 / k` at 99% confidence, space about `3 k` values. */
  readonly k?: number
  /** The key the compaction coins come from. */
  readonly key: Key
}

/** The KLL sketch. */
export interface KllSketch extends QuantileSketch {
  /** The size parameter. */
  readonly k: number
  /** Number of values stored (at most about `3 k`). */
  readonly size: number
  /** The compactors from the lowest level (weight 1) upward (weight `2^level`). */
  levels(): readonly (readonly number[])[]
  /** The estimated number of values at most `x`. */
  rank(x: number): number
  /** The sketch of both streams (the levels concatenated, then compacted), keeping this sketch's coins. */
  merge(other: KllSketch): KllSketch
}

function makeKll(
  k: number,
  key: Key,
  start: number[][],
  startCount: number,
  startMin: number,
  startMax: number,
): KllSketch {
  const levels = start.map((l) => l.slice())
  if (levels.length === 0) levels.push([])
  const coin = uniformStream(key, 64)
  let n = startCount
  let min = startMin
  let max = startMax
  const capacity = (h: number): number => Math.max(2, Math.ceil(k * (2 / 3) ** (levels.length - 1 - h)))
  const totalCapacity = (): number => levels.reduce((s, _, h) => s + capacity(h), 0)
  const size = (): number => levels.reduce((s, l) => s + l.length, 0)
  const compress = (): void => {
    while (size() >= totalCapacity()) {
      let h = 0
      while (h < levels.length && (levels[h]?.length ?? 0) < capacity(h)) h++
      if (h >= levels.length) return
      const buf = (levels[h] as number[]).sort((a, b) => a - b)
      if (h + 1 >= levels.length) levels.push([])
      const up = levels[h + 1] as number[]
      // An odd item stays at this level; the pairs lose one member each, chosen by a fair coin.
      const odd = buf.length % 2 === 1 ? buf.pop() : undefined
      const offset = coin() < 0.5 ? 0 : 1
      for (let i = offset; i < buf.length; i += 2) up.push(buf[i] as number)
      levels[h] = odd === undefined ? [] : [odd]
    }
  }
  const weighted = (): { value: number; weight: number }[] => {
    const out: { value: number; weight: number }[] = []
    levels.forEach((l, h) => {
      for (const value of l) out.push({ value, weight: 2 ** h })
    })
    return out.sort((a, b) => a.value - b.value)
  }
  compress()
  const rank = (x: number): number => {
    let r = 0
    for (const { value, weight } of weighted()) if (value <= x) r += weight
    return r
  }
  return {
    k,
    get count() {
      return n
    },
    get size() {
      return size()
    },
    get min() {
      return min
    },
    get max() {
      return max
    },
    add: (value) => {
      checkFinite(value, 'kllSketch.add')
      levels[0]?.push(value)
      n += 1
      if (Number.isNaN(min) || value < min) min = value
      if (Number.isNaN(max) || value > max) max = value
      compress()
    },
    quantile: (q) => {
      checkQuantile(q, 'kllSketch.quantile')
      if (n === 0) return Number.NaN
      const items = weighted()
      const total = items.reduce((s, it) => s + it.weight, 0)
      const target = q * total
      let cum = 0
      for (const it of items) {
        cum += it.weight
        if (cum >= target) return it.value
      }
      return max
    },
    cdf: (x) => {
      if (n === 0) return Number.NaN
      const items = weighted()
      const total = items.reduce((s, it) => s + it.weight, 0)
      return rank(x) / total
    },
    rank: (x) => {
      const items = weighted()
      const total = items.reduce((s, it) => s + it.weight, 0)
      return (rank(x) / total) * n
    },
    levels: () => levels.map((l) => l.slice()),
    merge: (other) => {
      const theirs = other.levels()
      const height = Math.max(levels.length, theirs.length)
      const combined = Array.from({ length: height }, (_, h) => [...(levels[h] ?? []), ...(theirs[h] ?? [])])
      const lo = Number.isNaN(min) ? other.min : Number.isNaN(other.min) ? min : Math.min(min, other.min)
      const hi = Number.isNaN(max) ? other.max : Number.isNaN(other.max) ? max : Math.max(max, other.max)
      return makeKll(k, key, combined, n + other.count, lo, hi)
    },
  }
}

/**
 * A KLL quantile sketch (Karnin, Lang and Liberty 2016): a stack of compactors, level `h` holding values of weight
 * `2^h` with capacity `k (2/3)^(top - h)` (at least 2). When the sketch is full the lowest full compactor is sorted and
 * every other value is promoted to the next level (the discarded half chosen by a fair coin), which leaves every rank
 * query unbiased with error about `1 / k`. It stays mergeable, and its error does not grow with `n`.
 *
 * @param options - `k` and the `key` for the coins.
 * @returns The sketch.
 * @throws MdmlError with code `value` for `k < 8`.
 * @concept quantile-sketch
 * @reference Z. Karnin, K. Lang and E. Liberty, Optimal quantile approximation in streams (FOCS 2016).
 * @example
 * ```ts
 * import { kllSketch } from 'mdml-engine/sketches'
 * import { key } from 'mdml-engine/random'
 * const kll = kllSketch({ k: 200, key: key(1) })
 * for (let i = 1; i <= 20000; i++) kll.add(i)
 * Math.abs(kll.quantile(0.5) - 10000) < 0.03 * 20000 // => true
 * ```
 */
export function kllSketch(options: KllSketchOptions): KllSketch {
  const k = options.k ?? 200
  if (!(Number.isInteger(k) && k >= 8))
    throw new MdmlError('value', `kllSketch: k must be an integer of at least 8, got ${k}`)
  return makeKll(k, options.key, [[]], 0, Number.NaN, Number.NaN)
}

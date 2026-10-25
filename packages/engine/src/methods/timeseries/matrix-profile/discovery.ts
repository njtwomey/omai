/**
 * Pattern discovery from a matrix profile, as in stumpy: all matches of a query, the top motifs (repeated patterns, the
 * windows with the closest neighbours) with their matches, the top discords (the most unusual windows), and time-series
 * chains of windows that evolve step by step (anchored and all chains).
 */

import { WhiteboxError } from '../../../lib/base/index.ts'
import { applyExclusionZone, exclusionZone, mass, seriesOf } from '../../../lib/signal/index.ts'
import type { TensorLike } from '../../../lib/tensor/index.ts'

// Discovery on a matrix profile, ported from STUMPY: matches of a query (match), motifs (the windows with the
// smallest profile values and their neighbours), discords (the largest), and time-series chains from the left and
// right nearest neighbours (atsc, allc).

/** A match of a query: its distance and the start of the matching window. */
export interface Match {
  readonly distance: number
  readonly index: number
}

/** A distance threshold: a number, or a function of the distance profile. */
export type MaxDistance = number | ((d: Float64Array) => number)

/** STUMPY's default threshold: max(mean - 2 std, min) of the finite distances. */
function defaultThreshold(d: Float64Array): number {
  let n = 0
  let mean = 0
  let lo = Number.POSITIVE_INFINITY
  for (const v of d)
    if (Number.isFinite(v)) {
      n++
      mean += v
      lo = Math.min(lo, v)
    }
  if (n === 0) return Number.NaN
  mean /= n
  let s = 0
  for (const v of d) if (Number.isFinite(v)) s += (v - mean) ** 2
  return Math.max(mean - 2 * Math.sqrt(s / n), lo)
}

function argmin(d: Float64Array): number {
  let best = 0
  for (let i = 1; i < d.length; i++) if ((d[i] ?? 0) < (d[best] ?? 0)) best = i
  return best
}

/** STUMPY's _find_matches: repeatedly take the closest window, then exclude its zone. */
function findMatches(
  profile: Float64Array,
  zone: number,
  maxDistance: MaxDistance | undefined,
  maxMatches: number,
  queryIdx: number | undefined,
  atol: number,
): Match[] {
  const d = Float64Array.from(profile)
  const limit = typeof maxDistance === 'number' ? maxDistance : (maxDistance ?? defaultThreshold)(d)
  let candidate = queryIdx ?? argmin(d)
  const out: Match[] = []
  for (let r = 0; r < d.length; r++) {
    const v = d[candidate] ?? Number.POSITIVE_INFINITY
    if (v > atol + limit || !Number.isFinite(v) || out.length >= maxMatches) break
    out.push({ distance: v, index: candidate })
    applyExclusionZone(d, candidate, zone)
    candidate = argmin(d)
  }
  return out
}

/** Options for {@link match}. */
export interface MatchOptions {
  /** Largest distance to accept (default max(mean - 2 std, min) of the distance profile). */
  readonly maxDistance?: MaxDistance
  /** Most matches to return (default all). */
  readonly maxMatches?: number
  /** Absolute tolerance added to the threshold (default 1e-8). */
  readonly atol?: number
  /** The query's own position in the series, matched first at distance 0. */
  readonly queryIdx?: number
}

/**
 * All non-overlapping matches of a query in a series (stumpy.match): the closest window, then repeatedly the closest
 * outside the exclusion zones of those already taken, while the distance is below the threshold.
 *
 * @param q - Query `[m]`.
 * @param t - Series.
 * @param options - Threshold, count, tolerance and the query's own position.
 * @returns The matches, closest first.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { match } from 'whitebox-engine/methods/timeseries'
 * const t = [0, 1, 2, 1, 0, 5, 3, 0, 1, 2, 1, 0, 4]
 * match([0, 1, 2, 1], t, { maxDistance: 0.1 }).map((r) => r.index) // => [0, 7]
 * ```
 */
export function match(q: TensorLike, t: TensorLike, options: MatchOptions = {}): Match[] {
  const qa = seriesOf(q, 'match')
  if (!qa.every(Number.isFinite)) throw new WhiteboxError('value', 'match: the query contains NaN or infinity')
  const d = mass(qa, seriesOf(t, 'match'))
  if (options.queryIdx !== undefined) d[options.queryIdx] = 0
  return findMatches(
    d,
    exclusionZone(qa.length),
    options.maxDistance,
    options.maxMatches ?? Number.POSITIVE_INFINITY,
    options.queryIdx,
    options.atol ?? 1e-8,
  )
}

/** Options for {@link motifs}. */
export interface MotifOptions {
  /** A motif needs more than this many matches (default 1). */
  readonly minNeighbors?: number
  /** Largest distance of a match (default as {@link match}). */
  readonly maxDistance?: MaxDistance
  /** Largest profile value of a motif (default max(mean - 2 std, min) of the profile). */
  readonly cutoff?: number
  /** Matches per motif (default 10). */
  readonly maxMatches?: number
  /** Motifs to return (default 1). */
  readonly maxMotifs?: number
  /** Tolerance of the matching (default 1e-8). */
  readonly atol?: number
}

/** Motifs: per motif, its matches' distances and starts (each row padded with NaN and -1 to the longest). */
export interface Motifs {
  readonly distances: number[][]
  readonly indices: number[][]
}

/**
 * Top motifs from a matrix profile (stumpy.motifs): repeatedly the window with the smallest profile value (below the
 * cutoff) and all its matches, the exclusion zones of every match removed from the profile before the next motif.
 *
 * @param t - Series.
 * @param P - Its matrix profile (nearest-neighbour distances, `[n - m + 1]`).
 * @param options - Neighbours, thresholds and counts.
 * @returns Distances and starts of each motif's matches.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { motifs, stump } from 'whitebox-engine/methods/timeseries'
 * const t = [0, 1, 2, 1, 0, 5, 3, 0, 1, 2, 1, 0, 4, 9, 2, 6, 3]
 * motifs(t, stump(t, 4).P, { maxDistance: 0.1 }).indices // => [[0, 7]]
 * ```
 */
export function motifs(t: TensorLike, P: ArrayLike<number>, options: MotifOptions = {}): Motifs {
  const ta = seriesOf(t, 'motifs')
  const profile = Float64Array.from(P)
  const m = ta.length - profile.length + 1
  const zone = exclusionZone(m)
  const maxMotifs = Math.max(1, options.maxMotifs ?? 1)
  const maxMatches = options.maxMatches ?? 10
  const cutoff = options.cutoff ?? defaultThreshold(profile)
  const minNeighbors = options.minNeighbors ?? 1
  const distances: number[][] = []
  const indices: number[][] = []
  let candidate = argmin(profile)
  for (let r = 0; r < profile.length; r++) {
    if (distances.length >= maxMotifs) break
    const v = profile[candidate] ?? Number.POSITIVE_INFINITY
    if (v > cutoff || !Number.isFinite(v)) break
    if (typeof options.maxDistance === 'number' && v > options.maxDistance) break
    const found = match(ta.slice(candidate, candidate + m), ta, {
      maxMatches,
      queryIdx: candidate,
      atol: options.atol ?? 1e-8,
      ...(options.maxDistance === undefined ? {} : { maxDistance: options.maxDistance }),
    })
    if (found.length > minNeighbors) {
      distances.push(found.slice(0, maxMatches).map((x) => x.distance))
      indices.push(found.slice(0, maxMatches).map((x) => x.index))
    }
    const exclude = found.length === 0 ? [candidate] : found.map((x) => x.index)
    for (const idx of exclude) applyExclusionZone(profile, idx, zone)
    candidate = argmin(profile)
  }
  const width = indices.reduce((w, row) => Math.max(w, row.length), 0)
  return {
    distances: distances.map((row) => [...row, ...Array.from({ length: width - row.length }, () => Number.NaN)]),
    indices: indices.map((row) => [...row, ...Array.from({ length: width - row.length }, () => -1)]),
  }
}

/**
 * Top discords from a matrix profile: repeatedly the window with the largest finite profile value (the most
 * unusual: farthest from its nearest neighbour), each one's exclusion zone removed before the next.
 *
 * @param P - Matrix profile `[l]`.
 * @param m - Window length.
 * @param k - Discords to return (default 1).
 * @returns Discord starts and their profile values, most unusual first.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { discords } from 'whitebox-engine/methods/timeseries'
 * discords([0.1, 0.2, 3, 0.1, 0.3, 0.2, 2, 0.1], 4, 2).indices // => [2, 6]
 * ```
 */
export function discords(
  P: ArrayLike<number>,
  m: number,
  k = 1,
): { readonly indices: number[]; readonly distances: number[] } {
  const p = Float64Array.from(P, (v) => (Number.isFinite(v) ? v : Number.NEGATIVE_INFINITY))
  const zone = exclusionZone(m)
  const indices: number[] = []
  const distances: number[] = []
  for (let r = 0; r < k; r++) {
    let best = 0
    for (let i = 1; i < p.length; i++) if ((p[i] ?? 0) > (p[best] ?? 0)) best = i
    const v = p[best] ?? Number.NEGATIVE_INFINITY
    if (!Number.isFinite(v)) break
    indices.push(best)
    distances.push(v)
    applyExclusionZone(p, best, zone, Number.NEGATIVE_INFINITY)
  }
  return { indices, distances }
}

/**
 * The anchored time-series chain from window j (stumpy.atsc): follow right nearest neighbours while each link is
 * mutual (the next window's left neighbour points back).
 *
 * @param IL - Left nearest-neighbour indices.
 * @param IR - Right nearest-neighbour indices.
 * @param j - Anchor.
 * @returns The chain's window starts.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { atsc } from 'whitebox-engine/methods/timeseries'
 * atsc([-1, -1, 0, 1, 2], [2, 3, 4, -1, -1], 0) // => [0, 2, 4]
 * ```
 */
export function atsc(IL: ArrayLike<number>, IR: ArrayLike<number>, j: number): number[] {
  const out = [j]
  let cur = j
  for (let i = 0; i < IL.length; i++) {
    const next = IR[cur] ?? -1
    if (next === -1 || (IL[next] ?? -1) !== cur) break
    cur = next
    out.push(cur)
  }
  return out
}

/**
 * All time-series chains (stumpy.allc): every maximal chain of mutual left-right neighbours, and the longest
 * (unanchored) one.
 *
 * @param IL - Left nearest-neighbour indices.
 * @param IR - Right nearest-neighbour indices.
 * @returns All chains (in order of their first window) and the longest.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { allc } from 'whitebox-engine/methods/timeseries'
 * allc([-1, -1, 0, 1, 2], [2, 3, 4, -1, -1]).chains // => [[0, 2, 4], [1, 3]]
 * ```
 */
export function allc(
  IL: ArrayLike<number>,
  IR: ArrayLike<number>,
): { readonly chains: number[][]; readonly longest: number[] } {
  const n = IL.length
  const L = new Int32Array(n).fill(1)
  const chains: number[][] = []
  const seen = new Set<string>()
  for (let i = 0; i < n; i++) {
    if (L[i] !== 1) continue
    const c = [i]
    let j = i
    for (let r = 0; r < n; r++) {
      const next = IR[j] ?? -1
      if (next === -1 || (IL[next] ?? -1) !== j) break
      j = next
      L[j] = -1
      L[i] = (L[i] ?? 0) + 1
      c.push(j)
    }
    const id = c.join(',')
    if (!seen.has(id)) {
      seen.add(id)
      chains.push(c)
    }
  }
  let best = 0
  for (let i = 1; i < n; i++) if ((L[i] ?? 0) > (L[best] ?? 0)) best = i
  return { chains, longest: atsc(IL, IR, best) }
}

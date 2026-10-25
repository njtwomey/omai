/**
 * Summaries of time series by representative windows: MPdist, a matrix-profile distance between series robust to their
 * lengths, snippets (Imani et al., 2018) that cover a series with a few typical windows and the regimes each
 * represents, and the ostinato consensus motif shared by several series. As in stumpy.
 */

import { WhiteboxError } from '../../../lib/base/index.ts'
import { mass, seriesOf } from '../../../lib/signal/index.ts'
import type { TensorLike } from '../../../lib/tensor/index.ts'
import { stump } from './algorithms.ts'

// Distances between whole series built from matrix profiles, and what they find, ported from STUMPY: MPdist (the
// k-th smallest value of the joined AB and BA profiles), time-series snippets (the representative windows that best
// cover a series under MPdist) and the consensus motif across several series (ostinato).

/** STUMPY's _select_P_ABBA_value: the k-th smallest value, or the largest finite one among the k smallest. */
function selectK(values: Float64Array, k: number): number {
  const kk = Math.min(Math.trunc(k), values.length - 1)
  const sorted = Float64Array.from(values).sort()
  const v = sorted[kk] ?? Number.POSITIVE_INFINITY
  if (Number.isFinite(v)) return v
  let finite = 0
  for (let i = 0; i < kk; i++) if (Number.isFinite(sorted[i] ?? Number.POSITIVE_INFINITY)) finite++
  return sorted[Math.max(0, finite - 1)] ?? Number.POSITIVE_INFINITY
}

/** Options for {@link mpdist}. */
export interface MpdistOptions {
  /** The k-th smallest as a fraction of the combined length (default 0.05). */
  readonly percentage?: number
  /** The rank to take explicitly (overrides percentage). */
  readonly k?: number
}

/**
 * The matrix-profile distance between two series (stumpy.mpdist): the k-th smallest of the AB-join and BA-join
 * profiles together, k = ceil(percentage (nA + nB)); robust to series of different lengths and to some unmatched
 * structure.
 *
 * @param tA - First series.
 * @param tB - Second series.
 * @param m - Window length.
 * @param options - Percentage or rank.
 * @returns The distance.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { mpdist } from 'whitebox-engine/methods/timeseries'
 * const a = Array.from({ length: 30 }, (_, i) => Math.sin(i / 2))
 * mpdist(a, a.slice(5), 6) < 1e-6 // => true
 * ```
 */
export function mpdist(tA: TensorLike, tB: TensorLike, m: number, options: MpdistOptions = {}): number {
  const a = seriesOf(tA, 'mpdist')
  const b = seriesOf(tB, 'mpdist')
  const ab = stump(a, m, { tB: b }).P
  const ba = stump(b, m, { tB: a }).P
  const p = new Float64Array(ab.length + ba.length)
  p.set(ab)
  p.set(ba, ab.length)
  const k =
    options.k !== undefined
      ? Math.min(Math.trunc(options.k), p.length - 1)
      : Math.min(Math.ceil(Math.min(1, Math.max(0, options.percentage ?? 0.05)) * (a.length + b.length)), p.length - 1)
  return selectK(p, k)
}

/** The MPdist of a query against every window of its length in a series (stumpy's _mpdist_vect). */
function mpdistProfile(
  q: Float64Array,
  t: Float64Array,
  s: number,
  percentage: number,
  k: number | undefined,
  queryIdx: number,
): Float64Array {
  const j = q.length - s + 1
  const l = t.length - s + 1
  const dm = Array.from({ length: j }, (_, i) => {
    const w = q.subarray(i, i + s)
    if (!w.every(Number.isFinite)) return new Float64Array(l).fill(Number.POSITIVE_INFINITY)
    const d = mass(w.slice(), t)
    d[queryIdx + i] = 0
    return d
  })
  const kk = Math.min(Math.trunc(k ?? Math.min(Math.ceil(percentage * 2 * q.length), 2 * j - 1)), 2 * j - 1)
  const colMin = Float64Array.from({ length: l }, (_, c) => {
    let v = Number.POSITIVE_INFINITY
    for (const row of dm) v = Math.min(v, row[c] ?? Number.POSITIVE_INFINITY)
    return v
  })
  const out = new Float64Array(t.length - q.length + 1)
  const pabba = new Float64Array(2 * j)
  for (let i = 0; i < out.length; i++) {
    for (let r = 0; r < j; r++) {
      let v = Number.POSITIVE_INFINITY
      const row = dm[r] as Float64Array
      for (let c = i; c < i + j; c++) v = Math.min(v, row[c] ?? Number.POSITIVE_INFINITY)
      pabba[r] = v
    }
    for (let r = 0; r < j; r++) pabba[j + r] = colMin[i + r] ?? Number.POSITIVE_INFINITY
    out[i] = selectK(pabba, kk)
  }
  return out
}

/** Options for {@link snippets}. */
export interface SnippetOptions {
  /** Inner window as a fraction of m (default 1). */
  readonly percentage?: number
  /** Inner window length (overrides percentage). */
  readonly s?: number
  /** MPdist percentage (default 0.05). */
  readonly mpdistPercentage?: number
  /** MPdist rank (overrides the percentage). */
  readonly mpdistK?: number
}

/** Time-series snippets: the windows, their starts, MPdist profiles, the fraction of the series each covers, their
 * areas and regimes (rows of snippet, start, stop). */
export interface Snippets {
  readonly snippets: Float64Array[]
  readonly indices: number[]
  readonly profiles: Float64Array[]
  readonly fractions: number[]
  readonly areas: number[]
  readonly regimes: [number, number, number][]
}

/**
 * Time-series snippets (Imani et al. 2018, stumpy.snippets): split the series into non-overlapping windows of length
 * m, compute each one's MPdist profile against the whole series, then greedily pick the k windows whose profiles'
 * pointwise minimum has the least area; each snippet covers the positions where it is the closest.
 *
 * @param t - Series (length at least 2m).
 * @param m - Snippet length.
 * @param k - Number of snippets.
 * @param options - Inner window and MPdist settings.
 * @returns The snippets.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { snippets } from 'whitebox-engine/methods/timeseries'
 * const t = Array.from({ length: 64 }, (_, i) => (i < 32 ? Math.sin(i) : Math.sign(Math.sin(i / 2))))
 * snippets(t, 8, 2).indices.length // => 2
 * ```
 */
export function snippets(t: TensorLike, m: number, k: number, options: SnippetOptions = {}): Snippets {
  const ta = seriesOf(t, 'snippets')
  if (m > Math.floor(ta.length / 2))
    throw new WhiteboxError('value', 'snippets: m must be at most half the series length')
  const s =
    options.s !== undefined
      ? Math.min(Math.trunc(options.s), m)
      : Math.min(Math.ceil(Math.min(1, Math.max(0, options.percentage ?? 1)) * m), m)
  const windows = Math.floor(ta.length / m)
  const rightPad = ta.length % m === 0 ? 0 : m * Math.ceil(ta.length / m) - ta.length
  const padded = new Float64Array(ta.length + rightPad).fill(Number.NaN)
  padded.set(ta)
  const width = padded.length - m + 1 - rightPad
  const D = Array.from({ length: windows }, (_, i) =>
    mpdistProfile(
      padded.subarray(i * m, (i + 1) * m),
      padded,
      s,
      options.mpdistPercentage ?? 0.05,
      options.mpdistK,
      i * m,
    ).slice(0, width),
  )
  const Q = new Float64Array(width).fill(Number.POSITIVE_INFINITY)
  const out: Snippets = { snippets: [], indices: [], profiles: [], fractions: [], areas: [], regimes: [] }
  for (let r = 0; r < k; r++) {
    let best = 0
    let bestArea = Number.POSITIVE_INFINITY
    for (let i = 0; i < windows; i++) {
      let area = 0
      const row = D[i] as Float64Array
      for (let c = 0; c < width; c++) area += Math.min(row[c] ?? 0, Q[c] ?? 0)
      if (area < bestArea) {
        bestArea = area
        best = i
      }
    }
    const row = D[best] as Float64Array
    out.snippets.push(ta.slice(best * m, best * m + m))
    out.indices.push(best * m)
    out.profiles.push(Float64Array.from(row))
    let area = 0
    for (let c = 0; c < width; c++) {
      area += Math.min(row[c] ?? 0, Q[c] ?? 0)
      Q[c] = Math.min(row[c] ?? 0, Q[c] ?? 0)
    }
    out.areas.push(area)
  }
  const totalMin = Float64Array.from({ length: width }, (_, c) => {
    let v = Number.POSITIVE_INFINITY
    for (const p of out.profiles) v = Math.min(v, p[c] ?? Number.POSITIVE_INFINITY)
    return v
  })
  for (let r = 0; r < k; r++) {
    const p = out.profiles[r] as Float64Array
    let count = 0
    let start = -1
    for (let c = 0; c <= width; c++) {
      const inside = c < width && (p[c] ?? 0) <= (totalMin[c] ?? 0)
      if (inside) count++
      if (inside && start < 0) start = c
      if (!inside && start >= 0) {
        out.regimes.push([r, start, c])
        start = -1
      }
    }
    for (let c = 0; c < width; c++) if ((p[c] ?? 0) <= (totalMin[c] ?? 0)) totalMin[c] = (totalMin[c] ?? 0) - 1
    out.fractions.push(count / width)
  }
  return out
}

/** The consensus motif of several series: its radius, the series it is in and its start. */
export interface Ostinato {
  readonly radius: number
  readonly series: number
  readonly index: number
}

function nearestAcross(
  ts: readonly Float64Array[],
  si: number,
  idx: number,
  m: number,
): { radii: Float64Array; at: Int32Array } {
  const q = (ts[si] as Float64Array).slice(idx, idx + m)
  const radii = new Float64Array(ts.length)
  const at = new Int32Array(ts.length)
  for (let i = 0; i < ts.length; i++) {
    const d = mass(q, ts[i] as Float64Array)
    let best = 0
    for (let j = 1; j < d.length; j++) if ((d[j] ?? 0) < (d[best] ?? 0)) best = j
    radii[i] = d[best] ?? Number.POSITIVE_INFINITY
    at[i] = best
  }
  return { radii, at }
}

const isClose = (a: number, b: number): boolean => Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b)

/**
 * The consensus motif (Kamgar et al. 2019, stumpy.ostinato): the window, in any of the series, whose largest
 * distance to its nearest neighbour in every other series (its radius) is smallest; among windows with the same
 * radius, the most central one (smallest mean distance).
 *
 * @param series - Two or more series.
 * @param m - Window length.
 * @returns The radius, series index and start of the consensus motif.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { ostinato } from 'whitebox-engine/methods/timeseries'
 * const motif = [0, 3, 1, 4, 1]
 * const a = [5, 2, 7, ...motif, 2, 2, 9, 1]
 * const b = [1, 8, 2, 6, 3, 7, ...motif, 4]
 * const c = [...motif, 9, 0, 5, 2, 8, 3]
 * ostinato([a, b, c], 5).radius < 1e-6 // => true
 * ```
 */
export function ostinato(series: readonly TensorLike[], m: number): Ostinato {
  const ts = series.map((s) => seriesOf(s, 'ostinato'))
  const k = ts.length
  if (k < 2) throw new WhiteboxError('value', 'ostinato: need at least two series')
  let bsfRadius = Number.POSITIVE_INFINITY
  let bsfSeries = 0
  let bsfIdx = 0
  for (let j = 0; j < k; j++) {
    const h = j < k - 1 ? j + 1 : 0
    const P = stump(ts[j] as Float64Array, m, { tB: ts[h] as Float64Array }).P
    const order = Array.from(P, (_, i) => i).sort((x, y) => (P[x] ?? 0) - (P[y] ?? 0))
    for (const q of order) {
      let radius = P[q] ?? Number.POSITIVE_INFINITY
      if (radius >= bsfRadius) break
      for (let i = 0; i < k; i++) {
        if (i === j || i === h) continue
        const d = mass((ts[j] as Float64Array).slice(q, q + m), ts[i] as Float64Array)
        let lo = Number.POSITIVE_INFINITY
        for (const v of d) lo = Math.min(lo, v)
        radius = Math.max(radius, lo)
        if (radius >= bsfRadius) break
      }
      if (radius < bsfRadius) {
        bsfRadius = radius
        bsfSeries = j
        bsfIdx = q
      }
    }
  }
  const base = nearestAcross(ts, bsfSeries, bsfIdx, m)
  const mean = (a: Float64Array): number => a.reduce((s, v) => s + v, 0) / a.length
  let bestMean = mean(base.radii)
  let series0 = bsfSeries
  let index0 = bsfIdx
  for (let i = 0; i < k; i++) {
    if (!isClose(base.radii[i] ?? 0, bsfRadius)) continue
    const cand = nearestAcross(ts, i, base.at[i] ?? 0, m)
    let hi = Number.NEGATIVE_INFINITY
    for (const v of cand.radii) hi = Math.max(hi, v)
    if (isClose(hi, bsfRadius) && mean(cand.radii) < bestMean) {
      series0 = i
      index0 = base.at[i] ?? 0
      bestMean = mean(cand.radii)
    }
  }
  return { radius: bsfRadius, series: series0, index: index0 }
}

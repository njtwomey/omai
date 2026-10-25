/**
 * The multidimensional matrix profile (mSTAMP and mSTOMP, as stumpy's `mstump`): for multichannel series, the distance
 * from each window to its best match using its k most similar dimensions, for every k. Also the subspace a motif spans
 * and minimum description length for choosing how many dimensions a motif really has.
 */

import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import { ndtri } from '../../../lib/numerics/special/index.ts'
import {
  applyExclusionZone,
  exclusionZone,
  type SubsequenceStats,
  slidingDotProductDirect,
  squaredZNormalizedDistance,
  subsequenceStats,
  zNormalize,
} from '../../../lib/signal/index.ts'
import { asTensor, primalOf, type TensorLike, toFlat } from '../../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../../lib/trace/index.ts'
import type { MatrixProfileTrace } from './algorithms.ts'

// Multidimensional matrix profiles (Yeh et al. 2017, stumpy.mstump): for every window and every k, the best
// k-dimensional match, found by sorting the per-dimension distance profiles column by column and averaging the k
// smallest; with dimensions forced in (include) or the largest distances instead (discords). And the subspace and
// minimum-description-length analyses that pick the dimensions a motif actually spans.

/** Rows of a `[d, n]` series. */
function seriesRows(t: TensorLike, name: string): Float64Array[] {
  const v = asTensor(t)
  if (v.shape.length !== 2) throw new WhiteboxError('shape', `${name}: expected a [d, n] series`)
  const d = v.shape[0] ?? 0
  const n = v.shape[1] ?? 0
  const data = primalOf(v).data
  return Array.from({ length: d }, (_, i) => Float64Array.from({ length: n }, (_, j) => Number(data[i * n + j] ?? 0)))
}

/** A multidimensional matrix profile: row k - 1 holds the k-dimensional profile and its indices. */
export interface MultiProfile {
  readonly m: number
  readonly d: number
  /** Distances `[d * l]` (row-major). */
  readonly P: Float64Array
  /** Indices `[d * l]`. */
  readonly I: Int32Array
}

/** Options of the multidimensional profiles. */
export interface MultiOptions {
  /** Dimensions always included in every k-dimensional match (they fill the first rows). */
  readonly include?: readonly number[]
  /** Sort the distances in descending order (multidimensional discords). */
  readonly discords?: boolean
}

function uniqueInclude(include: readonly number[] | undefined, d: number, name: string): number[] {
  if (include === undefined) return []
  const out: number[] = []
  for (const i of include) {
    if (!Number.isInteger(i) || i < 0 || i >= d) throw new WhiteboxError('value', `${name}: include out of range`)
    if (!out.includes(i)) out.push(i)
  }
  return out
}

/**
 * Turn per-dimension distance rows (one column per window) into the k-dimensional profile rows: included
 * dimensions first, the rest sorted per column (ascending, or descending for discords), then cumulative means.
 */
function combine(D: Float64Array[], include: readonly number[], discords: boolean): Float64Array[] {
  const d = D.length
  const l = D[0]?.length ?? 0
  const rest = Array.from({ length: d }, (_, i) => i).filter((i) => !include.includes(i))
  const out = Array.from({ length: d }, () => new Float64Array(l))
  const col = new Float64Array(rest.length)
  for (let j = 0; j < l; j++) {
    for (let r = 0; r < rest.length; r++) col[r] = D[rest[r] ?? 0]?.[j] ?? 0
    col.sort()
    if (discords) col.reverse()
    let acc = 0
    for (let r = 0; r < d; r++) {
      const v = r < include.length ? (D[include[r] ?? 0]?.[j] ?? 0) : (col[r - include.length] ?? 0)
      acc += v
      ;(out[r] as Float64Array)[j] = acc / (r + 1)
    }
  }
  return out
}

function distanceRow(stats: SubsequenceStats, qt: Float64Array, i: number, m: number): Float64Array {
  const mu = (j: number): number => (stats.isFinite[j] === 1 ? (stats.mean[j] ?? 0) : Number.POSITIVE_INFINITY)
  return Float64Array.from(qt, (q, j) =>
    Math.sqrt(
      squaredZNormalizedDistance(
        m,
        q,
        mu(i),
        stats.std[i] ?? 0,
        mu(j),
        stats.std[j] ?? 0,
        stats.isConstant[i] === 1,
        stats.isConstant[j] === 1,
      ),
    ),
  )
}

/**
 * The multidimensional distance profile of the window at `idx` (stumpy.multi_distance_profile): row k - 1 is, for
 * every other window, the mean of its k smallest per-dimension z-normalised distances (with the exclusion zone).
 *
 * @param t - Series `[d, n]`.
 * @param m - Window length.
 * @param idx - Query window start.
 * @param options - Included dimensions and discord ordering.
 * @returns The profiles `[d, l]` as rows.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { multiDistanceProfile } from 'whitebox-engine/methods/timeseries'
 * const t = [[0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], [1, 2, 0, 4, 2, 9, 3, 1, 7, 0, 5, 2, 8]]
 * multiDistanceProfile(t, 4, 2).length // => 2
 * ```
 */
export function multiDistanceProfile(
  t: TensorLike,
  m: number,
  idx: number,
  options: MultiOptions = {},
): Float64Array[] {
  const rows = seriesRows(t, 'multiDistanceProfile')
  const include = uniqueInclude(options.include, rows.length, 'multiDistanceProfile')
  const D = rows.map((r) => {
    const s = subsequenceStats(r, m)
    const d = distanceRow(s, slidingDotProductDirect(s.t.subarray(idx, idx + m), s.t), idx, m)
    return d
  })
  const out = combine(D, include, options.discords ?? false)
  const zone = exclusionZone(m)
  for (const r of out) applyExclusionZone(r, idx, zone)
  return out
}

/** The state of {@link mstumpAlgorithm}: the profile so far, rows done and each dimension's dot products. */
export interface MstumpState {
  readonly P: Float64Array
  readonly I: Int32Array
  readonly done: number
  readonly qt: readonly Float64Array[]
}

/**
 * mSTAMP/mSTOMP as an algorithm (stumpy.mstump): rows in order, each dimension's dot products updated in O(l) from
 * the previous row's, the per-dimension distance profiles combined into k-dimensional ones by sorting and averaging.
 *
 * @param t - Series `[d, n]`.
 * @param m - Window length.
 * @param options - Included dimensions, discord ordering and rows per step.
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { mstumpAlgorithm } from 'whitebox-engine/methods/timeseries'
 * mstumpAlgorithm([[0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], [1, 2, 0, 4, 2, 9, 3, 1, 7, 0, 5, 2, 8]], 4).name // => 'mstump'
 * ```
 */
export function mstumpAlgorithm(
  t: TensorLike,
  m: number,
  options: MultiOptions & { readonly batch?: number } = {},
): Algorithm<MstumpState> {
  const rows = seriesRows(t, 'mstump')
  const d = rows.length
  const stats = rows.map((r) => subsequenceStats(r, m))
  const n = rows[0]?.length ?? 0
  const l = n - m + 1
  const include = uniqueInclude(options.include, d, 'mstump')
  const discords = options.discords ?? false
  const zone = exclusionZone(m)
  const batch = options.batch ?? Math.max(1, Math.ceil(l / 10))
  const first = stats.map((s) => slidingDotProductDirect(s.t.subarray(0, m), s.t))
  return {
    name: 'mstump',
    config: { m, d, include, discords, batch } as Json,
    init: () => ({
      P: new Float64Array(d * l).fill(Number.POSITIVE_INFINITY),
      I: new Int32Array(d * l).fill(-1),
      done: 0,
      qt: stats.map(() => new Float64Array(l)),
    }),
    step: (st) => {
      const P = Float64Array.from(st.P)
      const I = Int32Array.from(st.I)
      const qt = st.qt.map((q) => Float64Array.from(q))
      const stop = Math.min(l, st.done + batch)
      for (let idx = st.done; idx < stop; idx++) {
        const D = stats.map((s, dim) => {
          const q = qt[dim] as Float64Array
          if (idx === 0) q.set(first[dim] as Float64Array)
          else {
            for (let j = l - 1; j > 0; j--)
              q[j] =
                (q[j - 1] ?? 0) -
                (s.t[idx - 1] ?? 0) * (s.t[j - 1] ?? 0) +
                (s.t[idx + m - 1] ?? 0) * (s.t[j + m - 1] ?? 0)
            q[0] = first[dim]?.[idx] ?? 0
          }
          const row = distanceRow(s, q, idx, m)
          applyExclusionZone(row, idx, zone)
          return row
        })
        const combined = combine(D, include, discords)
        for (let r = 0; r < d; r++) {
          const row = combined[r] as Float64Array
          let best = 0
          for (let j = 1; j < l; j++) if ((row[j] ?? 0) < (row[best] ?? 0)) best = j
          const v = row[best] ?? Number.POSITIVE_INFINITY
          P[r * l + idx] = v
          I[r * l + idx] = Number.isFinite(v) ? best : -1
        }
      }
      return { P, I, done: stop, qt }
    },
    done: (st) => (st.done >= l ? 'converged' : undefined),
    observe: (st) => ({ metrics: { fraction: st.done / l } }),
  }
}

/**
 * The multidimensional matrix profile (stumpy.mstump): row k - 1 holds, for every window, the distance to its best
 * k-dimensional match (the mean of the k smallest per-dimension distances) and that match's index.
 *
 * @param t - Series `[d, n]`.
 * @param m - Window length.
 * @param options - Included dimensions, discord ordering, rows per step and tracing.
 * @returns The profile.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { mstump } from 'whitebox-engine/methods/timeseries'
 * const t = [[0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], [1, 2, 0, 4, 2, 9, 3, 1, 7, 0, 5, 2, 8]]
 * mstump(t, 4).P.length // => 20
 * ```
 */
export function mstump(
  t: TensorLike,
  m: number,
  options: MultiOptions & { readonly batch?: number } & MatrixProfileTrace = {},
): MultiProfile {
  const d = seriesRows(t, 'mstump').length
  const st = runAlgorithm(mstumpAlgorithm(t, m, options), {
    maxSteps: Number.MAX_SAFE_INTEGER,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  return { m, d, P: st.P, I: st.I }
}

/** The bin edges of STUMPY's discretisation: standard normal quantiles of (1 .. 2^b - 1) / 2^b. */
function inverseNormBins(bits: number): Float64Array {
  const count = 2 ** bits
  const p = Array.from({ length: count - 1 }, (_, i) => (i + 1) / count)
  return Float64Array.from(toFlat(ndtri(p)))
}

/** np.digitize(x, bins, right=True): the index i with bins[i-1] < x <= bins[i]. */
function digitize(x: number, bins: Float64Array): number {
  let lo = 0
  let hi = bins.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((bins[mid] ?? 0) < x) lo = mid + 1
    else hi = mid
  }
  return lo
}

function discretised(rows: readonly Float64Array[], idx: number, m: number, bins: Float64Array): Int32Array[] {
  return rows.map((r) => {
    const s = subsequenceStats(r, m)
    const w = s.isConstant[idx] === 1 ? new Float64Array(m) : zNormalize(s.t.slice(idx, idx + m))
    return Int32Array.from(w, (v) => digitize(v, bins))
  })
}

function subspaceOf(D: Float64Array, k: number, include: readonly number[], discords: boolean): number[] {
  const order = Array.from(D, (_, i) => i).sort(
    (a, b) => (discords ? (D[b] ?? 0) - (D[a] ?? 0) : (D[a] ?? 0) - (D[b] ?? 0)) || a - b,
  )
  const inc = order.filter((i) => include.includes(i))
  const exc = order.filter((i) => !include.includes(i))
  return [...inc, ...exc].slice(0, k + 1)
}

/** Options for {@link subspace} and {@link mdl}. */
export interface SubspaceOptions extends MultiOptions {
  /** Bits of the discretisation (default 8). */
  readonly bits?: number
}

/**
 * The k + 1 dimensions a motif pair spans (stumpy.subspace): both windows z-normalised and discretised to 2^bits
 * normal-quantile levels, then the dimensions ordered by the Euclidean distance between them (smallest first, or
 * largest for discords), included dimensions first.
 *
 * @param t - Series `[d, n]`.
 * @param m - Window length.
 * @param subseqIdx - Start of the motif window.
 * @param nnIdx - Start of its neighbour.
 * @param k - The dimensionality minus one (as in the k-dimensional profile row).
 * @param options - Included dimensions, discords and bits.
 * @returns The dimension indices.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { subspace } from 'whitebox-engine/methods/timeseries'
 * // The second dimension repeats with period 6, so the windows at 0 and 6 match there.
 * const t = [[0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], [0, 1, 3, 2, 9, 1, 0, 1, 3, 2, 9, 1, 0]]
 * subspace(t, 4, 0, 6, 0) // => [1]
 * ```
 */
export function subspace(
  t: TensorLike,
  m: number,
  subseqIdx: number,
  nnIdx: number,
  k: number,
  options: SubspaceOptions = {},
): number[] {
  const rows = seriesRows(t, 'subspace')
  const bins = inverseNormBins(options.bits ?? 8)
  const a = discretised(rows, subseqIdx, m, bins)
  const b = discretised(rows, nnIdx, m, bins)
  const D = Float64Array.from(a, (r, i) => {
    let s = 0
    for (let q = 0; q < m; q++) s += ((r[q] ?? 0) - (b[i]?.[q] ?? 0)) ** 2
    return Math.sqrt(s)
  })
  return subspaceOf(D, k, uniqueInclude(options.include, rows.length, 'subspace'), options.discords ?? false)
}

/**
 * Minimum description length of each k-dimensional motif (stumpy.mdl): for k = 1 .. d, the bits to store the series
 * with the motif's k-dimensional subspace coded by its differences to the neighbour; the k with the fewest bits is
 * the motif's natural dimensionality.
 *
 * @param t - Series `[d, n]`.
 * @param m - Window length.
 * @param subseqIdx - Motif start for each k (length d).
 * @param nnIdx - Neighbour start for each k (length d).
 * @param options - Included dimensions, discords and bits.
 * @returns The bit sizes and subspaces for each k.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { mdl } from 'whitebox-engine/methods/timeseries'
 * const t = [[0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], [0, 1, 3, 2, 9, 1, 0, 1, 3, 2, 9, 1, 0]]
 * mdl(t, 4, [0, 0], [6, 6]).subspaces // => [[1], [1, 0]]
 * ```
 */
export function mdl(
  t: TensorLike,
  m: number,
  subseqIdx: readonly number[],
  nnIdx: readonly number[],
  options: SubspaceOptions = {},
): { readonly bits: Float64Array; readonly subspaces: number[][] } {
  const rows = seriesRows(t, 'mdl')
  const d = rows.length
  const nBit = options.bits ?? 8
  const bins = inverseNormBins(nBit)
  const include = uniqueInclude(options.include, d, 'mdl')
  const bits = new Float64Array(d)
  const subspaces: number[][] = []
  for (let k = 0; k < d; k++) {
    const a = discretised(rows, subseqIdx[k] ?? 0, m, bins)
    const b = discretised(rows, nnIdx[k] ?? 0, m, bins)
    const D = Float64Array.from(a, (r, i) => {
      let s = 0
      for (let q = 0; q < m; q++) s += ((r[q] ?? 0) - (b[i]?.[q] ?? 0)) ** 2
      return Math.sqrt(s)
    })
    const S = subspaceOf(D, k, include, options.discords ?? false)
    subspaces.push(S)
    const values = new Set<number>()
    for (const i of S) for (let q = 0; q < m; q++) values.add((a[i]?.[q] ?? 0) - (b[i]?.[q] ?? 0))
    const nVal = values.size
    const sub = S.length
    bits[k] = nBit * (2 * d * m - sub * m) + sub * m * Math.log2(nVal) + nVal * nBit
  }
  return { bits, subspaces }
}

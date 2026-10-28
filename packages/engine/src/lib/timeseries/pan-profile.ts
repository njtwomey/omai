/**
 * The pan matrix profile (stumpy's `stimp`): matrix profiles across a range of window lengths, computed in
 * breadth-first order as an anytime algorithm so a coarse picture over all lengths appears first, with `panImage` for
 * the normalised, displayable image.
 */

import type { Json } from '../base/index.ts'
import { seriesOf } from '../signal/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { type MatrixProfileTrace, stump } from './matrix-profile.ts'
import { scrump } from './scrimp.ts'

// The pan matrix profile (Madrid et al. 2019, stumpy.stimp): matrix profiles for a range of window lengths, computed
// in breadth-first order of a balanced binary tree over the lengths so that a partial run already spans the range
// coarsely, then normalised (by 2 sqrt(m), the largest z-normalised distance), contrasted and binarised for display.

/**
 * Level-order indices of the balanced binary search tree over 0 .. n - 1 (stumpy's _bfs_indices): the order in which
 * the pan profile visits window lengths.
 *
 * @param n - Number of nodes.
 * @returns The indices.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { bfsIndices } from 'mdml-engine/timeseries'
 * bfsIndices(10) // => [5, 2, 8, 1, 4, 7, 9, 0, 3, 6]
 * ```
 */
export function bfsIndices(n: number): number[] {
  if (n === 1) return [0]
  const out: number[] = []
  let level: [number, number][] = [[0, n]]
  while (out.length < n && level.length > 0) {
    const next: [number, number][] = []
    for (const [lo, hi] of level) {
      const mid = Math.floor((lo + hi) / 2)
      out.push(mid)
      if (lo < mid) next.push([lo, mid])
      if (mid + 1 < hi) next.push([mid + 1, hi])
    }
    level = next
  }
  return out
}

/** The largest sensible window for a self-join of a series of length n (stumpy's get_max_window_size). */
function maxWindow(n: number): number {
  return Math.trunc(n - Math.floor(Math.floor((n + 3) / 5))) - 1
}

/** Options for {@link stimpAlgorithm}. */
export interface StimpOptions {
  /** Smallest window (default 3). */
  readonly minM?: number
  /** Largest window (default the largest sensible one). */
  readonly maxM?: number
  /** Window step (default 1). */
  readonly step?: number
  /** Fraction of diagonals per profile (below 1: SCRIMP++ approximations; default 0.01). */
  readonly percentage?: number
  /** Seed each approximate profile by PreSCRIMP (default true). */
  readonly preScrump?: boolean
}

/** The state of {@link stimpAlgorithm}: the windows in visiting order, the profiles so far (rows by window order). */
export interface StimpState {
  readonly windows: Int32Array
  readonly order: Int32Array
  readonly pan: Float64Array
  readonly processed: number
}

/**
 * The pan matrix profile as an anytime algorithm (stumpy.stimp): each step computes the profile for the next window
 * length in breadth-first order (exactly with percentage 1, else by SCRIMP++ from the step's key).
 *
 * @param t - Series.
 * @param options - Window range, step and approximation.
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stimpAlgorithm } from 'mdml-engine/timeseries'
 * stimpAlgorithm(Array.from({ length: 40 }, (_, i) => Math.sin(i / 3)), { maxM: 8 }).name // => 'stimp'
 * ```
 */
export function stimpAlgorithm(t: TensorLike, options: StimpOptions = {}): Algorithm<StimpState> {
  const ta = seriesOf(t, 'stimp')
  const n = ta.length
  const minM = options.minM ?? 3
  const step = options.step ?? 1
  const ms: number[] = []
  if (options.maxM === undefined) {
    const hi = Math.max(minM + 1, maxWindow(n))
    for (let m = minM; m <= hi; m += step) ms.push(m)
  } else {
    const lo = Math.min(minM, options.maxM)
    const hi = Math.max(minM, options.maxM)
    for (let m = Math.max(3, lo); m <= Math.min(maxWindow(n), hi); m += step) ms.push(m)
  }
  const order = Int32Array.from(bfsIndices(ms.length))
  const windows = Int32Array.from(order, (i) => ms[i] ?? 3)
  const percentage = Math.min(1, Math.max(0, options.percentage ?? 0.01))
  return {
    name: 'stimp',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as StimpState,
    config: { n, windows: Array.from(windows), percentage } as Json,
    init: () => ({ windows, order, pan: new Float64Array(ms.length * n).fill(Number.POSITIVE_INFINITY), processed: 0 }),
    step: (st, ctx) => {
      const m = windows[st.processed] ?? 3
      const P =
        percentage < 1
          ? scrump(ta, m, { percentage, preScrump: options.preScrump ?? true, key: ctx.key }).P
          : stump(ta, m).P
      const pan = Float64Array.from(st.pan)
      pan.set(P, (order[st.processed] ?? 0) * n)
      return { ...st, pan, processed: st.processed + 1 }
    },
    done: (st) => (st.processed >= ms.length ? 'converged' : undefined),
    observe: (st) => ({ metrics: { processed: st.processed, fraction: st.processed / ms.length } }),
  }
}

/** Options for {@link panImage}. */
export interface PanOptions {
  /** Threshold of the contrast and binarisation (default 0.2). */
  readonly threshold?: number
  /** Divide each row by 2 sqrt(m) (default true). */
  readonly normalize?: boolean
  /** Map values through a sigmoid of their percentile ranks (default true). */
  readonly contrast?: boolean
  /** Threshold to 0 and 1 (default true). */
  readonly binary?: boolean
  /** Clip to [0, 1] (default true). */
  readonly clip?: boolean
}

/**
 * The displayable pan matrix profile (stumpy.stimp.pan): rows (one per window length, ascending) normalised,
 * contrasted, binarised and clipped; rows not yet computed copy the next computed row above them; missing values
 * take the largest value.
 *
 * @param state - A stimp state (complete or partial).
 * @param options - Display transformations.
 * @returns The image `[windows, n]` as rows.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { panImage, stimp } from 'mdml-engine/timeseries'
 * const s = stimp(Array.from({ length: 40 }, (_, i) => Math.sin(i / 3) + 0.1 * Math.cos(i * 2.3)), { maxM: 6, percentage: 1 })
 * panImage(s).length // => 4
 * ```
 */
export function panImage(state: StimpState, options: PanOptions = {}): Float64Array[] {
  const rowsCount = state.order.length
  const n = state.pan.length / Math.max(1, rowsCount)
  const threshold = options.threshold ?? 0.2
  const processed = Array.from(state.order.subarray(0, state.processed))
  const rows = Array.from({ length: rowsCount }, (_, r) =>
    Float64Array.from(state.pan.subarray(r * n, (r + 1) * n), (v) => (v === Number.POSITIVE_INFINITY ? Number.NaN : v)),
  )
  if (options.normalize ?? true)
    for (let p = 0; p < processed.length; p++) {
      const row = rows[processed[p] ?? 0] as Float64Array
      const norm = 1 / (2 * Math.sqrt(state.windows[p] ?? 3))
      for (let c = 0; c < n; c++) row[c] = Math.min(1, (row[c] ?? 0) * norm)
    }
  if (options.contrast ?? true) {
    // Ranks over all processed values (NaN last, stable), mapped to evenly spaced percentiles, then a sigmoid.
    const l = processed.length * n
    const flat: { v: number; i: number }[] = []
    for (let p = 0; p < processed.length; p++) {
      const row = rows[processed[p] ?? 0] as Float64Array
      for (let c = 0; c < n; c++) flat.push({ v: row[c] ?? Number.NaN, i: p * n + c })
    }
    flat.sort((a, b) => {
      const an = Number.isNaN(a.v)
      const bn = Number.isNaN(b.v)
      if (an || bn) return an === bn ? a.i - b.i : an ? 1 : -1
      return a.v - b.v || a.i - b.i
    })
    const pct = new Float64Array(l)
    for (let r = 0; r < l; r++) pct[(flat[r] as { i: number }).i] = l === 1 ? 0 : r / (l - 1)
    for (let p = 0; p < processed.length; p++) {
      const row = rows[processed[p] ?? 0] as Float64Array
      for (let c = 0; c < n; c++) row[c] = 1 / (1 + Math.exp(-10 * ((pct[p * n + c] ?? 0) - threshold)))
    }
  }
  if (options.binary ?? true)
    for (const r of processed) {
      const row = rows[r] as Float64Array
      for (let c = 0; c < n; c++) row[c] = (row[c] ?? Number.NaN) <= threshold ? 0 : 1
    }
  if (options.clip ?? true)
    for (const r of processed) {
      const row = rows[r] as Float64Array
      for (let c = 0; c < n; c++) row[c] = Math.min(1, Math.max(0, row[c] ?? 0))
    }
  const sorted = [...processed].sort((a, b) => a - b)
  let prev = -1
  for (const r of sorted) {
    for (let q = prev + 1; q < r; q++) rows[q] = Float64Array.from(rows[r] as Float64Array)
    prev = r
  }
  let hi = Number.NEGATIVE_INFINITY
  for (const row of rows) for (const v of row) if (!Number.isNaN(v)) hi = Math.max(hi, v)
  for (const row of rows) for (let c = 0; c < n; c++) if (Number.isNaN(row[c] ?? 0)) row[c] = hi
  return rows
}

/**
 * The pan matrix profile (stumpy.stimp, run to completion).
 *
 * @param t - Series.
 * @param options - Window range, approximation and tracing.
 * @returns The final state (see {@link panImage}).
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stimp } from 'mdml-engine/timeseries'
 * Array.from(stimp(Array.from({ length: 40 }, (_, i) => Math.sin(i / 3)), { maxM: 6, percentage: 1 }).windows) // => [5, 4, 6, 3]
 * ```
 */
export function stimp(t: TensorLike, options: StimpOptions & MatrixProfileTrace = {}): StimpState {
  return runAlgorithm(stimpAlgorithm(t, options), {
    maxSteps: Number.MAX_SAFE_INTEGER,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
    ...(options.key === undefined ? {} : { key: options.key }),
    ...(options.seed === undefined ? {} : { seed: options.seed }),
  }).state
}

/**
 * Semantic segmentation of a time series from its matrix profile: FLUSS (as stumpy's `fluss`) counts the
 * nearest-neighbour arcs crossing each position, corrects them against the ideal arc curve and reads regime boundaries
 * from its minima; FLOSS does the same on a stream, one value at a time, as a traced algorithm.
 */

import { type Json, MdmlError } from '../../../lib/base/index.ts'
import {
  exclusionZone,
  mass,
  rollingIsFinite,
  seriesOf,
  subsequenceStats,
  zNormalize,
} from '../../../lib/signal/index.ts'
import type { TensorLike } from '../../../lib/tensor/index.ts'
import type { Algorithm } from '../../../lib/trace/index.ts'
import type { MatrixProfile } from './algorithms.ts'

// Semantic segmentation from the matrix-profile index (Gharghabi et al. 2017, stumpy.fluss and stumpy.floss): arcs
// from every window to its nearest neighbour rarely cross a regime change, so the corrected arc curve (arc counts
// over their expectation under no structure) dips there. FLUSS works on a whole series with bidirectional
// neighbours; FLOSS streams with right neighbours only.
//
// STUMPY estimates the idealised arc curve by fitting a beta density to randomly drawn neighbour indices. Here it is
// the exact expectation of the random model: 2 (x + 1)(w - 1 - x) / w when every neighbour is uniform over all
// windows, and sum_{i <= x} (w - 1 - x) / (w - 1 - i) when it is uniform over the windows to its right. Either can be
// overridden (the goldens pass this curve to STUMPY as its custom_iac).

/**
 * Arc counts (stumpy's _nnmark): how many nearest-neighbour arcs (i to I[i]) pass over each position.
 *
 * @param I - Nearest-neighbour indices (negative means none: the arc is empty).
 * @returns Counts `[l]`.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { arcCounts } from 'mdml-engine/methods/timeseries'
 * Array.from(arcCounts([2, 3, 0, 1])) // => [2, 4, 2, 0]
 * ```
 */
export function arcCounts(I: ArrayLike<number>): Int32Array {
  const k = I.length
  const diff = new Int32Array(k + 1)
  for (let i = 0; i < k; i++) {
    const j = (I[i] ?? -1) < 0 ? i : (I[i] ?? i)
    diff[Math.min(i, j)] = (diff[Math.min(i, j)] ?? 0) + 1
    diff[Math.max(i, j)] = (diff[Math.max(i, j)] ?? 0) - 1
  }
  const out = new Int32Array(k)
  let acc = 0
  for (let i = 0; i < k; i++) {
    acc += diff[i] ?? 0
    out[i] = acc
  }
  return out
}

/**
 * The idealised arc curve: the expected arc counts when neighbours are uniformly random (bidirectional) or
 * uniformly random to the right (unidirectional, for streaming).
 *
 * @param width - Number of windows.
 * @param bidirectional - Neighbours on both sides (default true).
 * @returns The expected counts `[width]`.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { idealArcCurve } from 'mdml-engine/methods/timeseries'
 * Array.from(idealArcCurve(4)) // => [1.5, 2, 1.5, 0]
 * ```
 */
export function idealArcCurve(width: number, bidirectional = true): Float64Array {
  const w = width
  if (bidirectional) return Float64Array.from({ length: w }, (_, x) => (2 * (x + 1) * (w - 1 - x)) / w)
  const out = new Float64Array(w)
  let harmonic = 0
  for (let x = 0; x < w - 1; x++) {
    harmonic += 1 / (w - 1 - x)
    out[x] = (w - 1 - x) * harmonic
  }
  return out
}

/** Options for {@link correctedArcCurve} and {@link fluss}. */
export interface ArcCurveOptions {
  /** Neighbours on both sides (default true). */
  readonly bidirectional?: boolean
  /** Edge exclusion in multiples of L (default 5): the curve is 1 within L * factor of either end. */
  readonly exclusionFactor?: number
  /** A custom idealised arc curve. */
  readonly iac?: ArrayLike<number>
}

/**
 * The corrected arc curve (stumpy's _cac): arc counts over the idealised arc curve, capped at 1, set to 1 near the
 * ends; low values mark regime boundaries.
 *
 * @param I - Nearest-neighbour indices.
 * @param L - Typical regime length scale (the subsequence length).
 * @param options - Direction, edge exclusion and idealised curve.
 * @returns The curve `[l]` in [0, 1].
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { correctedArcCurve } from 'mdml-engine/methods/timeseries'
 * correctedArcCurve([1, 0, 3, 2, 5, 4, 7, 6], 1, { exclusionFactor: 1 })[3] // => 0
 * ```
 */
export function correctedArcCurve(I: ArrayLike<number>, L: number, options: ArcCurveOptions = {}): Float64Array {
  const k = I.length
  const ac = arcCounts(I)
  const iac =
    options.iac === undefined ? idealArcCurve(k, options.bidirectional ?? true) : Float64Array.from(options.iac)
  if (iac.length !== k) throw new MdmlError('shape', 'correctedArcCurve: the idealised curve has the wrong length')
  const cac = Float64Array.from(ac, (a, i) => Math.min(1, a / ((iac[i] ?? 0) === 0 ? 1e-10 : (iac[i] ?? 0))))
  const factor = options.exclusionFactor ?? 5
  if (factor > 0) {
    const edge = Math.min(k, L * factor)
    cac.fill(1, 0, edge)
    cac.fill(1, Math.max(0, k - edge))
  }
  return cac
}

/**
 * Regime extraction (stumpy's _rea): the n - 1 lowest points of the corrected arc curve, each excluding L * factor
 * around it.
 *
 * @param cac - Corrected arc curve.
 * @param regimes - Number of regimes.
 * @param L - Length scale.
 * @param exclusionFactor - Exclusion around each boundary in multiples of L (default 5).
 * @returns Boundary positions.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { regimeLocations } from 'mdml-engine/methods/timeseries'
 * regimeLocations([1, 0.9, 0.2, 0.8, 1, 0.5, 1], 3, 1, 1) // => [2, 5]
 * ```
 */
export function regimeLocations(cac: ArrayLike<number>, regimes: number, L: number, exclusionFactor = 5): number[] {
  const tmp = Float64Array.from(cac)
  const out: number[] = []
  for (let r = 0; r < regimes - 1; r++) {
    let best = 0
    for (let i = 1; i < tmp.length; i++) if ((tmp[i] ?? 0) < (tmp[best] ?? 0)) best = i
    out.push(best)
    tmp.fill(1, Math.max(best - exclusionFactor * L, 0), Math.min(best + exclusionFactor * L, tmp.length))
  }
  return out
}

/**
 * Fast low-cost unipotent semantic segmentation (stumpy.fluss): the corrected arc curve of the matrix-profile index
 * and the n - 1 regime boundaries read from it.
 *
 * @param I - Matrix-profile indices (nearest neighbours).
 * @param L - Length scale (usually the window length).
 * @param regimes - Number of regimes.
 * @param options - Edge exclusion and idealised curve.
 * @returns The corrected arc curve and the boundaries.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { fluss } from 'mdml-engine/methods/timeseries'
 * fluss([1, 0, 3, 2, 5, 4, 7, 6], 1, 2, { exclusionFactor: 1 }).regimes // => [1]
 * ```
 */
export function fluss(
  I: ArrayLike<number>,
  L: number,
  regimes: number,
  options: Omit<ArcCurveOptions, 'bidirectional'> = {},
): { readonly cac: Float64Array; readonly regimes: number[] } {
  const cac = correctedArcCurve(I, L, { ...options, bidirectional: true })
  return { cac, regimes: regimeLocations(cac, regimes, L, options.exclusionFactor ?? 5) }
}

/** The state of streaming segmentation ({@link flossInit}, {@link flossUpdate}). */
export interface FlossState {
  readonly t: Float64Array
  readonly m: number
  readonly L: number
  readonly exclusionFactor: number
  readonly iac: Float64Array
  /** Distances to the right nearest neighbours. */
  readonly P: Float64Array
  /** Right nearest-neighbour indices in stream coordinates. */
  readonly I: Float64Array
  readonly lastIdx: number
  readonly appended: number
  readonly cac: Float64Array
}

/**
 * Start streaming segmentation (stumpy.floss): from a matrix profile's right nearest neighbours, the right-neighbour
 * distances and the unidirectional idealised arc curve.
 *
 * @param mp - Matrix profile of `t` (its right indices are used).
 * @param t - The series it was computed from.
 * @param L - Length scale.
 * @param options - Edge exclusion and idealised curve.
 * @returns The state.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { flossInit, stump } from 'mdml-engine/methods/timeseries'
 * const t = Array.from({ length: 40 }, (_, i) => Math.sin(i / 2) + (i > 20 ? Math.sin(i * 1.3) : 0))
 * flossInit(stump(t, 5), t, 5).P.length // => 36
 * ```
 */
export function flossInit(
  mp: MatrixProfile,
  t: TensorLike,
  L: number,
  options: Omit<ArcCurveOptions, 'bidirectional'> = {},
): FlossState {
  const ta = seriesOf(t, 'flossInit')
  const m = mp.m
  const k = mp.PL.length
  const stats = subsequenceStats(ta, m)
  const P = new Float64Array(k)
  const I = new Float64Array(k)
  for (let i = 0; i < k; i++) {
    const j = mp.IR[i] ?? -1
    if (j < 0) {
      P[i] = Number.POSITIVE_INFINITY
      I[i] = i
      continue
    }
    I[i] = j
    const ci = stats.isConstant[i] === 1
    const cj = stats.isConstant[j] === 1
    if (ci && cj) P[i] = 0
    else if (ci || cj) P[i] = Math.sqrt(m)
    else {
      const a = zNormalize(ta.slice(i, i + m))
      const b = zNormalize(ta.slice(j, j + m))
      let s = 0
      for (let q = 0; q < m; q++) s += ((a[q] ?? 0) - (b[q] ?? 0)) ** 2
      P[i] = Math.sqrt(s)
    }
  }
  const iac = options.iac === undefined ? idealArcCurve(k, false) : Float64Array.from(options.iac)
  return {
    t: Float64Array.from(ta),
    m,
    L,
    exclusionFactor: options.exclusionFactor ?? 5,
    iac,
    P,
    I,
    lastIdx: ta.length - m + 1,
    appended: 0,
    cac: new Float64Array(k).fill(-1),
  }
}

/**
 * Append one value to streaming segmentation (stumpy.floss.update): the window slides by one, the new window's
 * distances update every older window's right neighbour, and the corrected arc curve is recomputed.
 *
 * @param state - The state.
 * @param value - The new value.
 * @returns The new state; `cac` is the corrected arc curve over the current windows.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { flossInit, flossUpdate, stump } from 'mdml-engine/methods/timeseries'
 * const t = Array.from({ length: 40 }, (_, i) => Math.sin(i / 2))
 * flossUpdate(flossInit(stump(t, 5), t, 5, { exclusionFactor: 1 }), 0.5).cac.length // => 36
 * ```
 */
export function flossUpdate(state: FlossState, value: number): FlossState {
  const { m } = state
  const n = state.t.length
  const t = new Float64Array(n)
  t.set(state.t.subarray(1))
  t[n - 1] = value
  const k = state.P.length
  const zoneStart = Math.max(0, k - exclusionZone(m))
  const P = new Float64Array(k)
  const I = new Float64Array(k)
  P.set(state.P.subarray(1))
  I.set(state.I.subarray(1))
  P[k - 1] = Number.POSITIVE_INFINITY
  I[k - 1] = state.lastIdx
  const filled = Float64Array.from(t, (v) => (Number.isFinite(v) ? v : 0))
  const D = mass(filled.slice(n - m), filled)
  D.fill(Number.POSITIVE_INFINITY, zoneStart)
  const finite = rollingIsFinite(t, m)
  for (let i = 0; i < k; i++) if (finite[i] !== 1 || finite[k - 1] !== 1) D[i] = Number.POSITIVE_INFINITY
  for (let i = 0; i < k; i++)
    if ((D[i] ?? 0) < (P[i] ?? 0)) {
      P[i] = D[i] ?? 0
      I[i] = state.lastIdx
    }
  const local = Float64Array.from(I, (v) => v - state.appended - 1)
  const cac = correctedArcCurve(local, state.L, {
    bidirectional: false,
    exclusionFactor: state.exclusionFactor,
    iac: state.iac,
  })
  return { ...state, t, P, I, cac, lastIdx: state.lastIdx + 1, appended: state.appended + 1 }
}

/**
 * Streaming segmentation as an algorithm: each step appends the next value by {@link flossUpdate}; the trace shows
 * the corrected arc curve's minimum (a regime change entering the window) over time.
 *
 * @param mp - Matrix profile of the initial series.
 * @param t - Initial series.
 * @param L - Length scale.
 * @param stream - Values to append, one per step.
 * @param options - Edge exclusion and idealised curve.
 * @returns The algorithm.
 * @concept regime-segmentation
 * @example
 * ```ts
 * import { flossAlgorithm, stump } from 'mdml-engine/methods/timeseries'
 * const t = Array.from({ length: 40 }, (_, i) => Math.sin(i / 2))
 * flossAlgorithm(stump(t, 5), t, 5, [1, 2]).name // => 'floss'
 * ```
 */
export function flossAlgorithm(
  mp: MatrixProfile,
  t: TensorLike,
  L: number,
  stream: ArrayLike<number>,
  options: Omit<ArcCurveOptions, 'bidirectional'> = {},
): Algorithm<FlossState & { readonly consumed: number }> {
  const init = flossInit(mp, t, L, options)
  return {
    name: 'floss',
    config: { m: mp.m, L, stream: stream.length, exclusionFactor: init.exclusionFactor } as Json,
    init: () => ({ ...init, consumed: 0 }),
    step: (st) => ({ ...flossUpdate(st, stream[st.consumed] ?? Number.NaN), consumed: st.consumed + 1 }),
    done: (st) => (st.consumed >= stream.length ? 'converged' : undefined),
    observe: (st) => {
      let lo = Number.POSITIVE_INFINITY
      let at = -1
      for (let i = 0; i < st.cac.length; i++)
        if ((st.cac[i] ?? 1) < lo) {
          lo = st.cac[i] ?? 1
          at = i
        }
      return { metrics: { cacMin: lo, cacArgmin: at } }
    },
  }
}

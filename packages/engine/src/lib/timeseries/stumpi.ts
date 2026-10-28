/**
 * The incremental matrix profile of a stream (STUMPY's `stumpi`): the state after a series, the O(n) update when one
 * more value arrives (optionally dropping the oldest, so the window slides), and the update as a traced `Algorithm`
 * with one value per step.
 */

import type { Json } from '../base/index.ts'
import {
  applyExclusionZone,
  exclusionZone,
  seriesOf,
  slidingDotProductDirect,
  squaredZNormalizedDistance,
  subsequenceStats,
} from '../signal/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { observeProfile } from './_matrix-profile.ts'
import { insertRight, type MatrixProfile } from './_profile.ts'
import { stump } from './matrix-profile.ts'

/** The state of an incremental matrix profile ({@link stumpiInit}, {@link stumpiUpdate}). */
export interface StumpiState {
  /** The series (with egress, the last n values; non-finite values stored as 0). */
  readonly t: Float64Array
  readonly isFinite: Uint8Array
  readonly m: number
  readonly k: number
  readonly egress: boolean
  readonly exclusion: number
  readonly appended: number
  readonly mean: Float64Array
  readonly std: Float64Array
  readonly isConstant: Uint8Array
  /** Dot products of the last window against every window. */
  readonly qt: Float64Array
  readonly profile: MatrixProfile
}

/**
 * Start an incremental matrix profile (stumpy.stumpi) from a series: its STUMP profile plus the statistics needed to
 * update it in O(n) per new value.
 *
 * @param t - Initial series.
 * @param m - Window length.
 * @param options - k, and egress (default true: keep a fixed-length window, dropping the oldest value).
 * @returns The state.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stumpiInit } from 'mdml-engine/timeseries'
 * stumpiInit([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).profile.P.length // => 10
 * ```
 */
export function stumpiInit(
  t: TensorLike,
  m: number,
  options: { readonly k?: number; readonly egress?: boolean } = {},
): StumpiState {
  const raw = seriesOf(t, 'stumpiInit')
  const profile = stump(raw, m, { k: options.k ?? 1 })
  const stats = subsequenceStats(raw, m)
  const mean = Float64Array.from(stats.mean, (v, i) => (stats.isFinite[i] === 1 ? v : Number.POSITIVE_INFINITY))
  return {
    t: stats.t,
    isFinite: Uint8Array.from(raw, (v) => (Number.isFinite(v) ? 1 : 0)),
    m,
    k: options.k ?? 1,
    egress: options.egress ?? true,
    exclusion: exclusionZone(m),
    appended: 0,
    mean,
    std: stats.std,
    isConstant: stats.isConstant,
    qt: slidingDotProductDirect(stats.t.subarray(stats.t.length - m), stats.t),
    profile,
  }
}

function lastWindowStats(
  t: Float64Array,
  finite: Uint8Array,
  m: number,
): { mu: number; sigma: number; constant: boolean } {
  const n = t.length
  for (let i = n - m; i < n; i++)
    if (finite[i] !== 1) return { mu: Number.POSITIVE_INFINITY, sigma: Number.NaN, constant: false }
  const w = t.subarray(n - m)
  const stats = subsequenceStats(w, m)
  return { mu: stats.mean[0] ?? 0, sigma: stats.std[0] ?? 0, constant: stats.isConstant[0] === 1 }
}

/**
 * Append one value to an incremental matrix profile (stumpy.stumpi.update): the new window's distance profile from the
 * O(n) dot-product update, merged into every row it improves (top-k), and the new row; with egress the oldest value
 * and its window are dropped first.
 *
 * @param state - The incremental state.
 * @param value - The new value (a non-finite value makes the windows containing it infinite).
 * @returns The new state.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stumpiInit, stumpiUpdate } from 'mdml-engine/timeseries'
 * const s = stumpiUpdate(stumpiInit([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4, { egress: false }), 3)
 * s.profile.P.length // => 11
 * ```
 */
export function stumpiUpdate(state: StumpiState, value: number): StumpiState {
  const { m, k, exclusion } = state
  const finiteValue = Number.isFinite(value)
  const v = finiteValue ? value : 0
  const egress = state.egress
  const nOld = state.t.length
  const lOld = nOld - m + 1
  // New series (egress: shift left by one), dot products, statistics and profile rows.
  const t = egress ? new Float64Array(nOld) : new Float64Array(nOld + 1)
  const finite = new Uint8Array(t.length)
  if (egress) {
    t.set(state.t.subarray(1))
    finite.set(state.isFinite.subarray(1))
  } else {
    t.set(state.t)
    finite.set(state.isFinite)
  }
  t[t.length - 1] = v
  finite[t.length - 1] = finiteValue ? 1 : 0
  const n = t.length
  const l = n - m + 1
  const q = lastWindowStats(t, finite, m)
  // The value leaving the last window: the first value of the old last window.
  const tDrop = state.t[lOld - 1] ?? 0
  const qtNew = new Float64Array(l)
  if (egress) {
    // STUMPY: QT shifted left, then QT_new[1:] = QT[:l-1] - T[:l-1] t_drop + T[m:] t on the shifted series.
    const shifted = state.qt.subarray(1)
    for (let j = 1; j < l; j++) qtNew[j] = (shifted[j - 1] ?? 0) - (t[j - 1] ?? 0) * tDrop + (t[j - 1 + m] ?? 0) * v
  } else
    for (let j = 1; j < l; j++) qtNew[j] = (state.qt[j - 1] ?? 0) - (t[j - 1] ?? 0) * tDrop + (t[j - 1 + m] ?? 0) * v
  let s0 = 0
  for (let r = 0; r < m; r++) s0 += (t[r] ?? 0) * (t[n - m + r] ?? 0)
  qtNew[0] = s0
  const shift = (a: Float64Array | Uint8Array | Int32Array, fill: number): typeof a => {
    const out = new (a.constructor as Float64ArrayConstructor)(egress ? a.length : a.length + 1) as typeof a
    if (egress) out.set(a.subarray(1))
    else out.set(a)
    out[out.length - 1] = fill
    return out
  }
  const mean = shift(state.mean, q.mu) as Float64Array
  const std = shift(state.std, q.sigma) as Float64Array
  const isConstant = shift(state.isConstant, q.constant ? 1 : 0) as Uint8Array
  const shiftRows = <A extends Float64Array | Int32Array>(a: A, fill: number): A => {
    const out = new (a.constructor as Float64ArrayConstructor)(l * k) as unknown as A
    if (egress) out.set(a.subarray(k) as never)
    else out.set(a as never)
    out.fill(fill as never, (l - 1) * k)
    return out
  }
  const P = shiftRows(state.profile.P as Float64Array, Number.POSITIVE_INFINITY)
  const I = shiftRows(state.profile.I as Int32Array, -1)
  const PL = shift(state.profile.PL as Float64Array, Number.POSITIVE_INFINITY) as Float64Array
  const IL = shift(state.profile.IL as Int32Array, -1) as Int32Array
  const PR = shift(state.profile.PR as Float64Array, Number.POSITIVE_INFINITY) as Float64Array
  const IR = shift(state.profile.IR as Int32Array, -1) as Int32Array
  // Indices stored in the profile are in the coordinates of the whole stream (STUMPY's n_appended offset).
  const appended = egress ? state.appended + 1 : 0
  const D = new Float64Array(l)
  const lastFinite = Number.isFinite(q.mu)
  for (let j = 0; j < l; j++) {
    const d2 = squaredZNormalizedDistance(
      m,
      qtNew[j] ?? 0,
      q.mu,
      q.sigma,
      mean[j] ?? 0,
      std[j] ?? 0,
      q.constant,
      isConstant[j] === 1,
    )
    D[j] = lastFinite ? Math.sqrt(d2) : Number.POSITIVE_INFINITY
  }
  applyExclusionZone(D, l - 1, exclusion)
  for (let i = 0; i < l; i++) {
    const d = D[i] ?? Number.POSITIVE_INFINITY
    if (d < (P[i * k + k - 1] ?? 0)) insertRight(P, I, i * k, k, d, l + appended - 1, 'right')
    // The new window lies to the right of every other: it may be a better right neighbour.
    if (i < l - 1 && d < (PR[i] ?? 0)) {
      PR[i] = d
      IR[i] = l + appended - 1
    }
  }
  P.fill(Number.POSITIVE_INFINITY, (l - 1) * k)
  I.fill(-1, (l - 1) * k)
  for (let i = 0; i < l; i++)
    if ((D[i] ?? 0) < (P[(l - 1) * k + k - 1] ?? 0)) insertRight(P, I, (l - 1) * k, k, D[i] ?? 0, i + appended, 'right')
  PL[l - 1] = P[(l - 1) * k] ?? Number.POSITIVE_INFINITY
  IL[l - 1] = I[(l - 1) * k] ?? -1
  return {
    ...state,
    t,
    isFinite: finite,
    appended,
    mean,
    std,
    isConstant,
    qt: qtNew,
    profile: { ...state.profile, P, I, PL, IL, PR, IR },
  }
}

/**
 * An incremental matrix profile over a stream as an algorithm: each step appends the next value by
 * {@link stumpiUpdate}, so the profile's evolution can be traced.
 *
 * @param t - Initial series.
 * @param m - Window length.
 * @param stream - Values to append, one per step.
 * @param options - k and egress.
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stumpiAlgorithm } from 'mdml-engine/timeseries'
 * stumpiAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4, [1, 2]).name // => 'stumpi'
 * ```
 */
export function stumpiAlgorithm(
  t: TensorLike,
  m: number,
  stream: ArrayLike<number>,
  options: { readonly k?: number; readonly egress?: boolean } = {},
): Algorithm<StumpiState & { readonly consumed: number }> {
  const init = stumpiInit(t, m, options)
  return {
    name: 'stumpi',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as StumpiState & { readonly consumed: number },
    config: { m, k: init.k, egress: init.egress, stream: stream.length } as Json,
    init: () => ({ ...init, consumed: 0 }),
    step: (st) => ({ ...stumpiUpdate(st, stream[st.consumed] ?? Number.NaN), consumed: st.consumed + 1 }),
    done: (st) => (st.consumed >= stream.length ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, st.consumed / Math.max(1, stream.length)),
  }
}

/**
 * The approximate and anytime matrix profiles of the SCRIMP family: PreSCRIMP, which seeds a profile from the exact
 * distance profiles of a sample of rows and propagates each neighbour along its diagonal, and SCRIMP++, which then
 * traverses the diagonals in random order, a batch per step, converging to the exact profile.
 */

import type { Json } from '../base/index.ts'
import { key as makeKey, permutation } from '../random/index.ts'
import {
  applyExclusionZone,
  slidingDotProductDirect,
  squaredZNormalizedDistance,
  subsequenceStats,
} from '../signal/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  type DiagonalState,
  diagonalProfile,
  emptyProfile,
  type MatrixProfileOptions,
  type MatrixProfileTrace,
  mergeProfiles,
  observeProfile,
  runOptions,
  type StumpOptions,
  setup,
} from './_matrix-profile.ts'
import type { MatrixProfile } from './_profile.ts'
import { diagonals, insertRight } from './_profile.ts'

/** Options for {@link prescrump}. */
export interface PrescrumpOptions extends Omit<MatrixProfileOptions, 'normalize' | 'p'> {
  /** Sampling interval of the rows (default the exclusion zone, ceil(m / 4)). */
  readonly s?: number
}

/**
 * PreSCRIMP (stumpy.prescrump): an approximate profile from the exact distance profiles of every s-th row (in a random
 * order from the key), each nearest neighbour propagated s - 1 steps along its diagonal both ways by the O(1)
 * dot-product update. Distances are upper bounds of the exact profile.
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - AB-join series, k and sampling interval, and the key for the row order.
 * @returns The approximate matrix profile (left and right neighbours are not computed).
 * @concept matrix-profile
 * @example
 * ```ts
 * import { prescrump } from 'omai-engine/timeseries'
 * import { key } from 'omai-engine/random'
 * const t = [0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7]
 * Array.from(prescrump(t, 4, { s: 1, key: key(0) }).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function prescrump(
  tA: TensorLike,
  m: number,
  options: PrescrumpOptions & Pick<MatrixProfileTrace, 'key' | 'seed'> = {},
): MatrixProfile {
  const st = setup(tA, m, options, 'prescrump')
  const sa = subsequenceStats(st.a, m)
  const sb = st.selfJoin ? sa : subsequenceStats(st.b, m)
  const k = st.k
  const l = st.lA
  const w = st.lB
  const excl = st.selfJoin ? st.exclusion : undefined
  const s = Math.max(1, options.s ?? st.exclusion)
  const rowsAll: number[] = []
  for (let i = 0; i < l; i += s) rowsAll.push(i)
  const key = options.key ?? (options.seed === undefined ? undefined : makeKey(options.seed))
  const perm = key === undefined ? rowsAll.map((_, i) => i) : Array.from(permutation(key, rowsAll.length).data)
  const P2 = new Float64Array(l * k).fill(Number.POSITIVE_INFINITY)
  const I = new Int32Array(l * k).fill(-1)
  const has = (row: number, j: number): boolean => {
    for (let q = 0; q < k; q++) if (I[row * k + q] === j) return true
    return false
  }
  const offer = (row: number, j: number, v: number): void => {
    if (v < (P2[row * k + k - 1] ?? 0) && !has(row, j)) insertRight(P2, I, row * k, k, v, j, 'right')
  }
  const mu = (x: typeof sa, i: number): number => (x.isFinite[i] === 1 ? (x.mean[i] ?? 0) : Number.POSITIVE_INFINITY)
  const dist = (qt: number, i: number, j: number): number =>
    squaredZNormalizedDistance(
      m,
      qt,
      mu(sa, i),
      sa.std[i] ?? 0,
      mu(sb, j),
      sb.std[j] ?? 0,
      sa.isConstant[i] === 1,
      sb.isConstant[j] === 1,
    )
  for (const r of perm) {
    const i = rowsAll[r] ?? 0
    const qtRow = slidingDotProductDirect(sa.t.subarray(i, i + m), sb.t)
    const d = Float64Array.from(qtRow, (qt, j) => dist(qt, i, j))
    if (sa.isFinite[i] !== 1) d.fill(Number.POSITIVE_INFINITY)
    if (excl !== undefined) applyExclusionZone(d, i, excl)
    let nn = 0
    for (let j = 1; j < w; j++) if ((d[j] ?? 0) < (d[nn] ?? 0)) nn = j
    if (!Number.isFinite(d[nn] ?? Number.POSITIVE_INFINITY)) continue
    offer(i, nn, d[nn] ?? 0)
    const j = nn
    let qt = qtRow[j] ?? 0
    for (let g = 1; g < Math.min(s, l - i, w - j); g++) {
      qt =
        qt - (sb.t[j + g - 1] ?? 0) * (sa.t[i + g - 1] ?? 0) + (sb.t[j + g + m - 1] ?? 0) * (sa.t[i + g + m - 1] ?? 0)
      const v = dist(qt, i + g, j + g)
      offer(i + g, j + g, v)
      if (excl !== undefined) offer(j + g, i + g, v)
    }
    qt = qtRow[j] ?? 0
    for (let g = 1; g < Math.min(s, i + 1, j + 1); g++) {
      qt = qt - (sb.t[j - g + m] ?? 0) * (sa.t[i - g + m] ?? 0) + (sb.t[j - g] ?? 0) * (sa.t[i - g] ?? 0)
      const v = dist(qt, i - g, j - g)
      offer(i - g, j - g, v)
      if (excl !== undefined) offer(j - g, i - g, v)
    }
    if (excl !== undefined)
      for (let jj = 0; jj < w; jj++)
        if ((d[jj] ?? Number.POSITIVE_INFINITY) < (P2[jj * k + k - 1] ?? 0)) offer(jj, i, d[jj] ?? 0)
  }
  const out = emptyProfile(l, k, m, st.exclusion)
  ;(out.P as Float64Array).set(Float64Array.from(P2, Math.sqrt))
  ;(out.I as Int32Array).set(I)
  return out
}

/** Options for {@link scrumpAlgorithm}. */
export interface ScrumpOptions extends StumpOptions {
  /** Fraction of the diagonals to traverse (default 0.01). */
  readonly percentage?: number
  /** Seed the profile by {@link prescrump} first (SCRIMP++; default true). */
  readonly preScrump?: boolean
  /** PreSCRIMP's sampling interval. */
  readonly s?: number
}

/**
 * SCRIMP++ as an anytime algorithm (stumpy.scrump): optionally seeded by PreSCRIMP, then diagonals in a random order
 * (from the run's key), a batch per step, merged into the profile. With percentage 1 the result is exact.
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - AB-join series, k, normalisation, fraction, seeding and batch size.
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { scrumpAlgorithm } from 'omai-engine/timeseries'
 * scrumpAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).name // => 'scrump'
 * ```
 */
export function scrumpAlgorithm(tA: TensorLike, m: number, options: ScrumpOptions = {}): Algorithm<DiagonalState> {
  const s = setup(tA, m, options, 'scrump')
  const all = diagonals(s.lA, s.lB, s.selfJoin, s.exclusion)
  const fraction = Math.min(1, Math.max(0, options.percentage ?? 0.01))
  const count = Math.max(1, Math.ceil(fraction * all.length))
  const batch = options.batch ?? Math.max(1, Math.ceil(count / 10))
  const seed = (options.preScrump ?? true) && s.normalize
  return {
    name: 'scrump',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as DiagonalState,
    config: { m, k: s.k, normalize: s.normalize, fraction, preScrump: seed, batch } as Json,
    init: (key) => {
      const perm = permutation(key, all.length).data
      const order = Array.from({ length: count }, (_, i) => all[perm[i] ?? 0] ?? 0)
      const base = emptyProfile(s.lA, s.k, m, s.exclusion)
      const profile = seed
        ? mergeProfiles(
            base,
            prescrump(s.a, m, {
              k: s.k,
              key,
              ...(options.tB === undefined ? {} : { tB: s.b }),
              ...(options.s === undefined ? {} : { s: options.s }),
            }),
          )
        : base
      return { profile, done: 0, order }
    },
    step: (st) => {
      const next = Math.min(st.order.length, st.done + batch)
      const part = diagonalProfile(s, m, st.order.slice(st.done, next))
      return { profile: mergeProfiles(st.profile, part), done: next, order: st.order }
    },
    done: (st) => (st.done >= st.order.length ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, st.done / Math.max(1, all.length)),
  }
}

/**
 * An approximate matrix profile by SCRIMP++ (exact with percentage 1).
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - SCRIMP++ options, key and tracing.
 * @returns The matrix profile.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { scrump } from 'omai-engine/timeseries'
 * Array.from(scrump([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4, { percentage: 1 }).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function scrump(tA: TensorLike, m: number, options: ScrumpOptions & MatrixProfileTrace = {}): MatrixProfile {
  return runAlgorithm(scrumpAlgorithm(tA, m, options), runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

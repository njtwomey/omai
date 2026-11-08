/**
 * The exact matrix profile and its anytime relatives, as in STUMPY: for every window of a series the distance to its
 * nearest neighbouring windows (z-normalised, or raw p-norm distances as AAMP), by diagonals (STUMP), by rows with the
 * dot-product update (STOMP) or by random rows (STAMP), each a traced `Algorithm` whose state is the profile so far and
 * whose step is a batch of diagonals or rows. Self-joins ignore trivial matches within the exclusion zone; an AB-join
 * profiles one series against another.
 *
 * @index function Exact matrix profile and anytime relatives (STUMP, STOMP, STAMP).
 */

import type { Json } from '../base/index.ts'
import { permutation } from '../random/index.ts'
import {
  applyExclusionZone,
  mass,
  slidingDotProductDirect,
  squaredZNormalizedDistance,
  subsequenceStats,
} from '../signal/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import type { MatrixProfileTrace } from './_matrix-profile.ts'
import {
  cloneProfile,
  type DiagonalState,
  diagonalProfile,
  emptyProfile,
  mergeProfiles,
  observeProfile,
  type RowState,
  runOptions,
  type StumpOptions,
  setup,
  writeRow,
} from './_matrix-profile.ts'
import type { MatrixProfile } from './_profile.ts'
import { diagonals } from './_profile.ts'

export type {
  DiagonalState,
  MatrixProfileOptions,
  MatrixProfileTrace,
  RowState,
  StumpOptions,
} from './_matrix-profile.ts'
export type { MatrixProfile } from './_profile.ts'

/**
 * STUMP as an algorithm: the exact matrix profile by traversing the diagonals of the distance matrix in batches, each
 * diagonal in O(l) by the covariance update (z-normalised) or the p-norm update (AAMP). Each step adds a batch.
 *
 * @param tA - Series.
 * @param m - Window length (at least 3).
 * @param options - AB-join series, k, normalisation, norm and batch size.
 * @returns The algorithm; its state holds the profile so far.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stumpAlgorithm } from 'omai-engine/timeseries'
 * stumpAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).name // => 'stump'
 * ```
 */
export function stumpAlgorithm(tA: TensorLike, m: number, options: StumpOptions = {}): Algorithm<DiagonalState> {
  const s = setup(tA, m, options, 'stump')
  const order = diagonals(s.lA, s.lB, s.selfJoin, s.exclusion)
  const batch = options.batch ?? Math.max(1, Math.ceil(order.length / 10))
  return {
    name: 'stump',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as DiagonalState,
    config: { m, k: s.k, normalize: s.normalize, p: s.p, selfJoin: s.selfJoin, batch } as Json,
    init: () => ({ profile: emptyProfile(s.lA, s.k, m, s.exclusion), done: 0, order }),
    step: (st) => {
      const next = Math.min(order.length, st.done + batch)
      const part = diagonalProfile(s, m, order.slice(st.done, next))
      return { profile: mergeProfiles(st.profile, part), done: next, order }
    },
    done: (st) => (st.done >= order.length ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, order.length === 0 ? 1 : st.done / order.length),
  }
}

/**
 * The exact matrix profile by STUMP (stumpy.stump, or stumpy.aamp when not normalised): for every window its k
 * nearest neighbours (excluding trivial matches within ceil(m / 4) in a self-join), and its left and right nearest
 * neighbours.
 *
 * @param tA - Series.
 * @param m - Window length (at least 3).
 * @param options - AB-join series, k, normalisation, norm, batch size and tracing.
 * @returns The matrix profile.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stump } from 'omai-engine/timeseries'
 * const mp = stump([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4)
 * Array.from(mp.I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function stump(tA: TensorLike, m: number, options: StumpOptions & MatrixProfileTrace = {}): MatrixProfile {
  const alg = stumpAlgorithm(tA, m, options)
  return runAlgorithm(alg, runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

/**
 * STOMP as an algorithm: the exact matrix profile row by row, each row's dot products updated in O(l) from the
 * previous row's (QT[i][j] = QT[i-1][j-1] - a[i-1] b[j-1] + a[i+m-1] b[j+m-1]), so each step costs O(l) per row.
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - AB-join series, k and rows per step (z-normalised only).
 * @returns The algorithm; its state holds the profile so far.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stompAlgorithm } from 'omai-engine/timeseries'
 * stompAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).name // => 'stomp'
 * ```
 */
export function stompAlgorithm(
  tA: TensorLike,
  m: number,
  options: Omit<StumpOptions, 'normalize' | 'p'> = {},
): Algorithm<RowState> {
  const s = setup(tA, m, options, 'stomp')
  const sa = subsequenceStats(s.a, m)
  const sb = s.selfJoin ? sa : subsequenceStats(s.b, m)
  const batch = options.batch ?? Math.max(1, Math.ceil(s.lA / 10))
  // Dot products of every window of a against the first window of b (the first column).
  const firstColumn = slidingDotProductDirect(sb.t.subarray(0, m), sa.t)
  const order = Int32Array.from({ length: s.lA }, (_, i) => i)
  return {
    name: 'stomp',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as RowState,
    config: { m, k: s.k, selfJoin: s.selfJoin, batch } as Json,
    init: () => ({ profile: emptyProfile(s.lA, s.k, m, s.exclusion), done: 0, order, qt: new Float64Array(s.lB) }),
    step: (st) => {
      const profile = cloneProfile(st.profile)
      const qt = Float64Array.from(st.qt)
      const stop = Math.min(s.lA, st.done + batch)
      const d = new Float64Array(s.lB)
      for (let i = st.done; i < stop; i++) {
        if (i === 0) qt.set(slidingDotProductDirect(sa.t.subarray(0, m), sb.t))
        else {
          for (let j = s.lB - 1; j > 0; j--)
            qt[j] =
              (qt[j - 1] ?? 0) -
              (sa.t[i - 1] ?? 0) * (sb.t[j - 1] ?? 0) +
              (sa.t[i + m - 1] ?? 0) * (sb.t[j + m - 1] ?? 0)
          qt[0] = firstColumn[i] ?? 0
        }
        const muQ = sa.isFinite[i] === 1 ? (sa.mean[i] ?? 0) : Number.POSITIVE_INFINITY
        for (let j = 0; j < s.lB; j++) {
          const muT = sb.isFinite[j] === 1 ? (sb.mean[j] ?? 0) : Number.POSITIVE_INFINITY
          const v = squaredZNormalizedDistance(
            m,
            qt[j] ?? 0,
            muQ,
            sa.std[i] ?? 0,
            muT,
            sb.std[j] ?? 0,
            sa.isConstant[i] === 1,
            sb.isConstant[j] === 1,
          )
          d[j] = v < 1e-14 ? 0 : v
        }
        if (s.selfJoin) applyExclusionZone(d, i, s.exclusion)
        writeRow(profile, i, d, s.selfJoin, Math.sqrt)
      }
      return { profile, done: stop, order, qt }
    },
    done: (st) => (st.done >= s.lA ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, st.done / s.lA),
  }
}

/**
 * The exact matrix profile by STOMP (row order); equal to {@link stump}'s.
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - AB-join series, k, rows per step and tracing.
 * @returns The matrix profile.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stomp } from 'omai-engine/timeseries'
 * Array.from(stomp([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function stomp(
  tA: TensorLike,
  m: number,
  options: Omit<StumpOptions, 'normalize' | 'p'> & MatrixProfileTrace = {},
): MatrixProfile {
  return runAlgorithm(stompAlgorithm(tA, m, options), runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

/** Options for {@link stampAlgorithm}. */
export interface StampOptions extends StumpOptions {
  /** Fraction of rows to evaluate (default 1, the exact profile). */
  readonly percentage?: number
  /** Exclusion-zone half-width (default ceil(m / 4); STUMPY's reference stamp uses ceil(m / 2)). */
  readonly exclusion?: number
}

/**
 * STAMP as an anytime algorithm: rows in a random order (from the run's key), each by a MASS distance profile; in a
 * self-join every evaluated row also improves the columns it touches, so a partial run is already a useful upper
 * bound of the profile.
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - AB-join series, normalisation, fraction of rows, exclusion and rows per step (k = 1).
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stampAlgorithm } from 'omai-engine/timeseries'
 * stampAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).name // => 'stamp'
 * ```
 */
export function stampAlgorithm(tA: TensorLike, m: number, options: StampOptions = {}): Algorithm<RowState> {
  const s = setup(tA, m, { ...options, k: 1 }, 'stamp')
  const exclusion = s.selfJoin ? (options.exclusion ?? s.exclusion) : 0
  const rows = Math.max(1, Math.ceil(Math.min(1, Math.max(0, options.percentage ?? 1)) * s.lA))
  const batch = options.batch ?? Math.max(1, Math.ceil(rows / 10))
  const massOptions = { normalize: s.normalize, p: s.p }
  return {
    name: 'stamp',
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as RowState,
    config: { m, selfJoin: s.selfJoin, rows, exclusion, normalize: s.normalize, batch } as Json,
    init: (key) => ({
      profile: emptyProfile(s.lA, 1, m, exclusion),
      done: 0,
      order: Int32Array.from(permutation(key, s.lA).data),
      qt: new Float64Array(0),
    }),
    step: (st) => {
      const profile = cloneProfile(st.profile)
      const P = profile.P as Float64Array
      const I = profile.I as Int32Array
      const stop = Math.min(rows, st.done + batch)
      for (let r = st.done; r < stop; r++) {
        const i = st.order[r] ?? 0
        const d = mass(s.a.slice(i, i + m), s.b, massOptions)
        if (s.selfJoin) applyExclusionZone(d, i, exclusion)
        const squared = Float64Array.from(d, (v) => v * v)
        writeRow(profile, i, squared, s.selfJoin, Math.sqrt)
        if (s.selfJoin)
          for (let j = 0; j < d.length; j++)
            if ((d[j] ?? Number.POSITIVE_INFINITY) < (P[j] ?? 0)) {
              P[j] = d[j] ?? 0
              I[j] = i
            }
      }
      return { ...st, profile, done: stop }
    },
    done: (st) => (st.done >= rows ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, st.done / s.lA),
  }
}

/**
 * The matrix profile by STAMP (exact with the default percentage 1; an upper bound otherwise).
 *
 * @param tA - Series.
 * @param m - Window length.
 * @param options - STAMP options, key and tracing.
 * @returns The matrix profile (k = 1).
 * @concept matrix-profile
 * @example
 * ```ts
 * import { stamp } from 'omai-engine/timeseries'
 * Array.from(stamp([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function stamp(tA: TensorLike, m: number, options: StampOptions & MatrixProfileTrace = {}): MatrixProfile {
  return runAlgorithm(stampAlgorithm(tA, m, options), runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

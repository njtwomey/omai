/**
 * Algorithms for the matrix profile, as in the stumpy library: for every window of a time series, the distance to its
 * nearest neighbouring window (z-normalised or not), from which motifs, discords and regimes can be read. Exact STUMP
 * (by diagonals), STOMP (row by row) and STAMP, approximate and anytime PreSCRIMP and SCRIMP++, and the incremental
 * STUMPI for streams, each as a traced algorithm.
 */

import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import { key as makeKey, permutation } from '../../../lib/random/index.ts'
import {
  applyExclusionZone,
  exclusionZone,
  mass,
  rollingIsFinite,
  seriesOf,
  slidingDotProductDirect,
  squaredZNormalizedDistance,
  subsequenceStats,
} from '../../../lib/signal/index.ts'
import type { TensorLike } from '../../../lib/tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../../../lib/trace/index.ts'
import {
  checkSelfJoin,
  correlationDiagonals,
  correlationState,
  diagonals,
  distanceState,
  insertRight,
  type MatrixProfile,
  pNormDiagonals,
  prepareDiagonal,
  profileFromCorrelations,
  profileFromPNorms,
} from './_profile.ts'

// Matrix-profile algorithms, ported from STUMPY: STUMP (diagonal traversal with the O(1) covariance update, exact),
// STOMP (row order with the O(1) dot-product update, exact), STAMP (anytime, random rows by MASS), PreSCRIMP and
// SCRIMP++ (anytime, random diagonals seeded by approximate neighbours), the incremental STUMPI, non-normalised AAMP
// (p-norms) and top-k neighbours. Every iterative one is an Algorithm, so its progress can be traced and resumed.

export type { MatrixProfile } from './_profile.ts'

/** Options shared by the matrix-profile algorithms. */
export interface MatrixProfileOptions {
  /** A second series for an AB-join (the profile of every window of the first against the second). */
  readonly tB?: TensorLike
  /** Neighbours per window (default 1). */
  readonly k?: number
  /** z-normalised distances (default true) or raw p-norm distances (AAMP). */
  readonly normalize?: boolean
  /** The Minkowski norm for non-normalised distances (default 2). */
  readonly p?: number
}

/** Tracing options passed through to {@link runAlgorithm}. */
export type MatrixProfileTrace = Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'key' | 'seed'>>

interface Setup {
  readonly a: Float64Array
  readonly b: Float64Array
  readonly selfJoin: boolean
  readonly k: number
  readonly normalize: boolean
  readonly p: number
  readonly exclusion: number
  readonly lA: number
  readonly lB: number
}

function setup(tA: TensorLike, m: number, options: MatrixProfileOptions, name: string): Setup {
  const a = seriesOf(tA, name)
  const selfJoin = options.tB === undefined
  const b = options.tB === undefined ? a : seriesOf(options.tB, name)
  if (!Number.isInteger(m) || m < 3) throw new WhiteboxError('value', `${name}: the window must be at least 3`)
  if (m > Math.min(a.length, b.length)) throw new WhiteboxError('value', `${name}: the window is longer than a series`)
  const exclusion = selfJoin ? checkSelfJoin(m, a.length) : 0
  const k = options.k ?? 1
  if (!Number.isInteger(k) || k < 1) throw new WhiteboxError('value', `${name}: k must be a positive integer`)
  return {
    a,
    b,
    selfJoin,
    k,
    normalize: options.normalize ?? true,
    p: options.p ?? 2,
    exclusion,
    lA: a.length - m + 1,
    lB: b.length - m + 1,
  }
}

function runOptions(trace: MatrixProfileTrace, maxSteps: number): RunOptions {
  return {
    maxSteps,
    ...(trace.tracer === undefined ? {} : { tracer: trace.tracer }),
    ...(trace.policy === undefined ? {} : { policy: trace.policy }),
    ...(trace.parent === undefined ? {} : { parent: trace.parent }),
    ...(trace.key === undefined ? {} : { key: trace.key }),
    ...(trace.seed === undefined ? {} : { seed: trace.seed }),
  }
}

function emptyProfile(l: number, k: number, m: number, exclusion: number): MatrixProfile {
  return {
    m,
    k,
    exclusion,
    P: new Float64Array(l * k).fill(Number.POSITIVE_INFINITY),
    I: new Int32Array(l * k).fill(-1),
    PL: new Float64Array(l).fill(Number.POSITIVE_INFINITY),
    IL: new Int32Array(l).fill(-1),
    PR: new Float64Array(l).fill(Number.POSITIVE_INFINITY),
    IR: new Int32Array(l).fill(-1),
  }
}

/** STUMPY's _merge_topk_PI: merge sorted neighbour rows keeping the k smallest, skipping indices already present. */
function mergeProfiles(into: MatrixProfile, from: MatrixProfile): MatrixProfile {
  const k = into.k
  const l = into.PL.length
  const P = Float64Array.from(into.P)
  const I = Int32Array.from(into.I)
  const tmpP = new Float64Array(k)
  const tmpI = new Int32Array(k)
  for (let i = 0; i < l; i++) {
    const o = i * k
    const overlap = new Set<number>()
    for (let q = 0; q < k; q++) {
      const ib = from.I[o + q] ?? -1
      for (let r = 0; r < k; r++) if ((into.I[o + r] ?? -1) === ib) overlap.add(ib)
    }
    let aj = 0
    let bj = 0
    let idx = 0
    for (let step = 0; step < 2 * k && idx < k; step++) {
      if (bj < k && (from.P[o + bj] ?? 0) < (into.P[o + aj] ?? 0)) {
        const ib = from.I[o + bj] ?? -1
        if (!overlap.has(ib)) {
          tmpP[idx] = from.P[o + bj] ?? 0
          tmpI[idx] = ib
          idx++
        }
        bj++
      } else {
        tmpP[idx] = into.P[o + aj] ?? 0
        tmpI[idx] = into.I[o + aj] ?? -1
        idx++
        aj++
      }
    }
    P.set(tmpP, o)
    I.set(tmpI, o)
  }
  const PL = Float64Array.from(into.PL)
  const IL = Int32Array.from(into.IL)
  const PR = Float64Array.from(into.PR)
  const IR = Int32Array.from(into.IR)
  for (let i = 0; i < l; i++) {
    if ((from.PL[i] ?? 0) < (PL[i] ?? 0)) {
      PL[i] = from.PL[i] ?? 0
      IL[i] = from.IL[i] ?? -1
    }
    if ((from.PR[i] ?? 0) < (PR[i] ?? 0)) {
      PR[i] = from.PR[i] ?? 0
      IR[i] = from.IR[i] ?? -1
    }
  }
  return { ...into, P, I, PL, IL, PR, IR }
}

/** Run the diagonal kernel over some diagonals into a fresh profile. */
function diagonalProfile(s: Setup, m: number, diags: readonly number[]): MatrixProfile {
  if (s.normalize) {
    const a = prepareDiagonal(s.a, m)
    const b = s.selfJoin ? a : prepareDiagonal(s.b, m)
    const st = correlationState(s.lA, s.k)
    correlationDiagonals(a, b, diags, st, s.k, s.selfJoin)
    return profileFromCorrelations(st, m, s.k, s.exclusion)
  }
  const fill = (x: Float64Array): { t: Float64Array; isFinite: Uint8Array } => ({
    t: Float64Array.from(x, (v) => (Number.isFinite(v) ? v : 0)),
    isFinite: rollingIsFinite(x, m),
  })
  const a = fill(s.a)
  const b = s.selfJoin ? a : fill(s.b)
  const st = distanceState(s.lA, s.k)
  pNormDiagonals(a, b, m, diags, st, s.k, s.selfJoin, s.p)
  return profileFromPNorms(st, m, s.k, s.exclusion, s.p)
}

/** The state of {@link stumpAlgorithm} and {@link scrumpAlgorithm}: the profile so far and the diagonals done. */
export interface DiagonalState {
  readonly profile: MatrixProfile
  readonly done: number
  readonly order: readonly number[]
}

function observeProfile(profile: MatrixProfile, fraction: number): { metrics: Record<string, number> } {
  let best = Number.POSITIVE_INFINITY
  let finite = 0
  for (let i = 0; i < profile.PL.length; i++) {
    const v = profile.P[i * profile.k] ?? Number.POSITIVE_INFINITY
    if (Number.isFinite(v)) {
      finite++
      best = Math.min(best, v)
    }
  }
  return { metrics: { fraction, motif: best, covered: finite / profile.PL.length } }
}

/** Options for {@link stumpAlgorithm}. */
export interface StumpOptions extends MatrixProfileOptions {
  /** Diagonals per step (default: all of them in ten steps). */
  readonly batch?: number
}

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
 * import { stumpAlgorithm } from 'whitebox-engine/methods/timeseries'
 * stumpAlgorithm([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).name // => 'stump'
 * ```
 */
export function stumpAlgorithm(tA: TensorLike, m: number, options: StumpOptions = {}): Algorithm<DiagonalState> {
  const s = setup(tA, m, options, 'stump')
  const order = diagonals(s.lA, s.lB, s.selfJoin, s.exclusion)
  const batch = options.batch ?? Math.max(1, Math.ceil(order.length / 10))
  return {
    name: 'stump',
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
 * import { stump } from 'whitebox-engine/methods/timeseries'
 * const mp = stump([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4)
 * Array.from(mp.I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function stump(tA: TensorLike, m: number, options: StumpOptions & MatrixProfileTrace = {}): MatrixProfile {
  const alg = stumpAlgorithm(tA, m, options)
  return runAlgorithm(alg, runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

/** The state of the row-wise algorithms ({@link stompAlgorithm}, {@link stampAlgorithm}). */
export interface RowState {
  readonly profile: MatrixProfile
  readonly done: number
  readonly order: Int32Array
  /** STOMP's dot products of the last row against every window of the second series. */
  readonly qt: Float64Array
}

/** Write row i's top-k, left and right neighbours from its squared (or p-th power) distance profile. */
function writeRow(
  profile: MatrixProfile,
  i: number,
  d: Float64Array,
  selfJoin: boolean,
  root: (v: number) => number,
): void {
  const k = profile.k
  const P = profile.P as Float64Array
  const I = profile.I as Int32Array
  P.fill(Number.POSITIVE_INFINITY, i * k, (i + 1) * k)
  I.fill(-1, i * k, (i + 1) * k)
  for (let j = 0; j < d.length; j++) {
    const v = d[j] ?? Number.POSITIVE_INFINITY
    if (v < (P[i * k + k - 1] ?? 0)) insertRight(P, I, i * k, k, v, j)
  }
  for (let q = 0; q < k; q++) {
    const v = P[i * k + q] ?? Number.POSITIVE_INFINITY
    P[i * k + q] = Number.isFinite(v) ? root(v) : v
    if (!Number.isFinite(v)) I[i * k + q] = -1
  }
  if (!selfJoin) return
  let bl = Number.POSITIVE_INFINITY
  let il = -1
  for (let j = 0; j < i; j++)
    if ((d[j] ?? Number.POSITIVE_INFINITY) < bl) {
      bl = d[j] ?? 0
      il = j
    }
  let br = Number.POSITIVE_INFINITY
  let ir = -1
  for (let j = i + 1; j < d.length; j++)
    if ((d[j] ?? Number.POSITIVE_INFINITY) < br) {
      br = d[j] ?? 0
      ir = j
    }
  ;(profile.PL as Float64Array)[i] = Number.isFinite(bl) ? root(bl) : bl
  ;(profile.IL as Int32Array)[i] = il
  ;(profile.PR as Float64Array)[i] = Number.isFinite(br) ? root(br) : br
  ;(profile.IR as Int32Array)[i] = ir
}

function cloneProfile(p: MatrixProfile): MatrixProfile {
  return {
    ...p,
    P: Float64Array.from(p.P),
    I: Int32Array.from(p.I),
    PL: Float64Array.from(p.PL),
    IL: Int32Array.from(p.IL),
    PR: Float64Array.from(p.PR),
    IR: Int32Array.from(p.IR),
  }
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
 * import { stompAlgorithm } from 'whitebox-engine/methods/timeseries'
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
 * import { stomp } from 'whitebox-engine/methods/timeseries'
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
 * import { stampAlgorithm } from 'whitebox-engine/methods/timeseries'
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
 * import { stamp } from 'whitebox-engine/methods/timeseries'
 * Array.from(stamp([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function stamp(tA: TensorLike, m: number, options: StampOptions & MatrixProfileTrace = {}): MatrixProfile {
  return runAlgorithm(stampAlgorithm(tA, m, options), runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

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
 * import { prescrump } from 'whitebox-engine/methods/timeseries'
 * import { key } from 'whitebox-engine/random'
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
 * import { scrumpAlgorithm } from 'whitebox-engine/methods/timeseries'
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
 * import { scrump } from 'whitebox-engine/methods/timeseries'
 * Array.from(scrump([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], 4, { percentage: 1 }).I) // => [9, 8, 9, 1, 9, 2, 3, 4, 1, 0]
 * ```
 */
export function scrump(tA: TensorLike, m: number, options: ScrumpOptions & MatrixProfileTrace = {}): MatrixProfile {
  return runAlgorithm(scrumpAlgorithm(tA, m, options), runOptions(options, Number.MAX_SAFE_INTEGER)).state.profile
}

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
 * import { stumpiInit } from 'whitebox-engine/methods/timeseries'
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
 * import { stumpiInit, stumpiUpdate } from 'whitebox-engine/methods/timeseries'
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
 * import { stumpiAlgorithm } from 'whitebox-engine/methods/timeseries'
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
    config: { m, k: init.k, egress: init.egress, stream: stream.length } as Json,
    init: () => ({ ...init, consumed: 0 }),
    step: (st) => ({ ...stumpiUpdate(st, stream[st.consumed] ?? Number.NaN), consumed: st.consumed + 1 }),
    done: (st) => (st.consumed >= stream.length ? 'converged' : undefined),
    observe: (st) => observeProfile(st.profile, st.consumed / Math.max(1, stream.length)),
  }
}

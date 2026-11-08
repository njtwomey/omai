/**
 * The shared kernels of the matrix-profile algorithms, private to the matrix-profile methods: STUMPY's preprocessing
 * for diagonal traversal, the z-normalised diagonal kernel (Pearson correlations updated along each diagonal, with
 * top-k insertion), the non-normalised p-norm kernel, and the profile container.
 */

import { OmaiError } from '../base/index.ts'
import { checkWindow, exclusionZone, rollingMeanStd, subsequenceStats } from '../signal/index.ts'

/** A matrix profile: for every window, its k nearest-neighbour distances and indices, and its left and right
 * nearest neighbours (before and after it in time). Rows of P and I are ascending in distance. */
export interface MatrixProfile {
  /** Window length. */
  readonly m: number
  /** Neighbours per window. */
  readonly k: number
  /** Exclusion-zone half-width used for self-joins (0 for AB-joins). */
  readonly exclusion: number
  /** Distances `[l * k]` (infinity where there is no neighbour). */
  readonly P: Float64Array
  /** Indices `[l * k]` (-1 where there is no neighbour). */
  readonly I: Int32Array
  readonly PL: Float64Array
  readonly IL: Int32Array
  readonly PR: Float64Array
  readonly IR: Int32Array
}

/** STUMPY's preprocessing for diagonal traversal (preprocess_diagonal). */
export interface DiagonalStats {
  readonly t: Float64Array
  readonly n: number
  readonly m: number
  readonly mean: Float64Array
  readonly sigmaInverse: Float64Array
  readonly meanM1: Float64Array
  readonly isFinite: Uint8Array
  readonly isConstant: Uint8Array
}

/** Prepare a series for diagonal traversal: zero-filled values, window means, inverse deviations (constant windows
 * get 1), means of windows of m - 1 for the covariance update, finiteness and constancy. */
export function prepareDiagonal(t: Float64Array, m: number): DiagonalStats {
  checkWindow(m, t.length)
  const s = subsequenceStats(t, m)
  const sigmaInverse = Float64Array.from(s.std, (v, i) => 1 / (s.isConstant[i] === 1 ? 1 : v))
  const meanM1 = rollingMeanStd(s.t, m - 1).mean
  return { t: s.t, n: t.length, m, mean: s.mean, sigmaInverse, meanM1, isFinite: s.isFinite, isConstant: s.isConstant }
}

/** Insert v into an ascending row segment, dropping the smallest (STUMPY's shift-left insertion for correlations). */
function insertLeft(rho: Float64Array, idx: Int32Array, off: number, k: number, v: number, j: number): void {
  let pos = 0
  while (pos < k && (rho[off + pos] ?? 0) < v) pos++
  // pos is the searchsorted index; shift [1, pos) left by one and place v at pos - 1.
  for (let q = 0; q < pos - 1; q++) {
    rho[off + q] = rho[off + q + 1] ?? 0
    idx[off + q] = idx[off + q + 1] ?? 0
  }
  rho[off + pos - 1] = v
  idx[off + pos - 1] = j
}

/** Insert v into an ascending row segment, dropping the largest (STUMPY's shift-right insertion for distances). */
export function insertRight(
  p: Float64Array,
  idx: Int32Array,
  off: number,
  k: number,
  v: number,
  j: number,
  side: 'left' | 'right' = 'left',
): void {
  let pos = 0
  while (pos < k && (side === 'left' ? (p[off + pos] ?? 0) < v : (p[off + pos] ?? 0) <= v)) pos++
  if (pos >= k) return
  for (let q = k - 1; q > pos; q--) {
    p[off + q] = p[off + q - 1] ?? 0
    idx[off + q] = idx[off + q - 1] ?? 0
  }
  p[off + pos] = v
  idx[off + pos] = j
}

/** The diagonals of a self-join (beyond the exclusion zone) or of an AB-join (all of them). */
export function diagonals(lA: number, lB: number, selfJoin: boolean, exclusion: number): number[] {
  const out: number[] = []
  if (selfJoin) for (let g = exclusion + 1; g < lA; g++) out.push(g)
  else for (let g = -lA + 1; g < lB; g++) out.push(g)
  return out
}

/** Correlation state of the z-normalised kernel, so it can be run diagonal batch by diagonal batch. */
export interface CorrelationState {
  readonly rho: Float64Array
  readonly I: Int32Array
  readonly rhoL: Float64Array
  readonly IL: Int32Array
  readonly rhoR: Float64Array
  readonly IR: Int32Array
}

/** A fresh correlation state for l windows and k neighbours. */
export function correlationState(l: number, k: number): CorrelationState {
  return {
    rho: new Float64Array(l * k).fill(Number.NEGATIVE_INFINITY),
    I: new Int32Array(l * k).fill(-1),
    rhoL: new Float64Array(l).fill(Number.NEGATIVE_INFINITY),
    IL: new Int32Array(l).fill(-1),
    rhoR: new Float64Array(l).fill(Number.NEGATIVE_INFINITY),
    IR: new Int32Array(l).fill(-1),
  }
}

/**
 * STUMPY's z-normalised diagonal kernel (_compute_diagonal): along each diagonal j = i + g, the covariance of the
 * windows by the O(1) update cov += (m - 1) / m^2 (a_j b_i - c_j d_i), the Pearson correlation, top-k insertion for
 * row i (and row j for self-joins) and the left and right neighbours.
 */
export function correlationDiagonals(
  a: DiagonalStats,
  b: DiagonalStats,
  diags: readonly number[],
  state: CorrelationState,
  k: number,
  selfJoin: boolean,
): void {
  const m = a.m
  const lA = a.n - m + 1
  const lB = b.n - m + 1
  const mInv = 1 / m
  const constant = (m - 1) * mInv * mInv
  const covA = (j: number): number => (b.t[j + m - 1] ?? 0) - (b.meanM1[j] ?? 0)
  const covB = (i: number): number => (a.t[i + m - 1] ?? 0) - (a.meanM1[i] ?? 0)
  const covC = (j: number): number => (j === 0 ? (b.t[b.n - 1] ?? 0) : (b.t[j - 1] ?? 0)) - (b.meanM1[j] ?? 0)
  const covD = (i: number): number => (i === 0 ? (a.t[a.n - 1] ?? 0) : (a.t[i - 1] ?? 0)) - (a.meanM1[i] ?? 0)
  const { rho, I, rhoL, IL, rhoR, IR } = state
  for (const g of diags) {
    const start = g >= 0 ? 0 : -g
    const stop = Math.min(lA, lB - g)
    let cov = 0
    for (let i = start; i < stop; i++) {
      const j = i + g
      if (i === 0 || j === 0) {
        const mb = b.mean[j] ?? 0
        const ma = a.mean[i] ?? 0
        let s = 0
        for (let q = 0; q < m; q++) s += ((b.t[j + q] ?? 0) - mb) * ((a.t[i + q] ?? 0) - ma)
        cov = s * mInv
      } else cov += constant * (covA(j) * covB(i) - covC(j) * covD(i))
      if (b.isFinite[j] !== 1 || a.isFinite[i] !== 1) continue
      let pearson: number
      if (b.isConstant[j] === 1 && a.isConstant[i] === 1) pearson = 1
      else if (b.isConstant[j] === 1 || a.isConstant[i] === 1) pearson = 0.5
      else pearson = Math.min(1, cov * (b.sigmaInverse[j] ?? 0) * (a.sigmaInverse[i] ?? 0))
      if (pearson > (rho[i * k] ?? 0)) insertLeft(rho, I, i * k, k, pearson, j)
      if (selfJoin) {
        if (pearson > (rho[j * k] ?? 0)) insertLeft(rho, I, j * k, k, pearson, i)
        if (i < j) {
          if (pearson > (rhoL[j] ?? 0)) {
            rhoL[j] = pearson
            IL[j] = i
          }
          if (pearson > (rhoR[i] ?? 0)) {
            rhoR[i] = pearson
            IR[i] = j
          }
        }
      }
    }
  }
}

/** Convert correlations to distances sqrt(|2 m (1 - rho)|) with STUMPY's 1e-14 floor, rows ascending in distance. */
export function profileFromCorrelations(
  state: CorrelationState,
  m: number,
  k: number,
  exclusion: number,
): MatrixProfile {
  const l = state.rhoL.length
  const dist = (r: number): number => {
    const p = Math.abs(2 * m * (1 - r))
    return Math.sqrt(p < 1e-14 ? 0 : p)
  }
  const P = new Float64Array(l * k)
  const I = new Int32Array(l * k)
  for (let i = 0; i < l; i++) {
    for (let q = 0; q < k; q++) {
      P[i * k + q] = dist(state.rho[i * k + k - 1 - q] ?? Number.NEGATIVE_INFINITY)
      I[i * k + q] = state.I[i * k + k - 1 - q] ?? -1
    }
  }
  return {
    m,
    k,
    exclusion,
    P,
    I,
    PL: Float64Array.from(state.rhoL, dist),
    IL: Int32Array.from(state.IL),
    PR: Float64Array.from(state.rhoR, dist),
    IR: Int32Array.from(state.IR),
  }
}

/** Distance state of the non-normalised kernel. */
export interface DistanceState {
  readonly P: Float64Array
  readonly I: Int32Array
  readonly PL: Float64Array
  readonly IL: Int32Array
  readonly PR: Float64Array
  readonly IR: Int32Array
}

/** A fresh distance state (p-norms raised to the p-th power while accumulating). */
export function distanceState(l: number, k: number): DistanceState {
  return {
    P: new Float64Array(l * k).fill(Number.POSITIVE_INFINITY),
    I: new Int32Array(l * k).fill(-1),
    PL: new Float64Array(l).fill(Number.POSITIVE_INFINITY),
    IL: new Int32Array(l).fill(-1),
    PR: new Float64Array(l).fill(Number.POSITIVE_INFINITY),
    IR: new Int32Array(l).fill(-1),
  }
}

/** STUMPY's non-normalised diagonal kernel (aamp): the p-th power of the p-norm updated in O(1) along diagonals. */
export function pNormDiagonals(
  a: { readonly t: Float64Array; readonly isFinite: Uint8Array },
  b: { readonly t: Float64Array; readonly isFinite: Uint8Array },
  m: number,
  diags: readonly number[],
  state: DistanceState,
  k: number,
  selfJoin: boolean,
  p: number,
): void {
  const lA = a.t.length - m + 1
  const lB = b.t.length - m + 1
  const { P, I, PL, IL, PR, IR } = state
  for (const g of diags) {
    const start = g >= 0 ? 0 : -g
    const stop = Math.min(lA, lB - g)
    let pnorm = 0
    for (let i = start; i < stop; i++) {
      const j = i + g
      if (i === 0 || j === 0) {
        let s = 0
        for (let q = 0; q < m; q++) s += Math.abs((b.t[j + q] ?? 0) - (a.t[i + q] ?? 0)) ** p
        pnorm = s
      } else
        pnorm = Math.abs(
          pnorm -
            Math.abs((b.t[j - 1] ?? 0) - (a.t[i - 1] ?? 0)) ** p +
            Math.abs((b.t[j + m - 1] ?? 0) - (a.t[i + m - 1] ?? 0)) ** p,
        )
      if (pnorm < 1e-14) pnorm = 0
      if (a.isFinite[i] !== 1 || b.isFinite[j] !== 1) continue
      if (pnorm < (P[i * k + k - 1] ?? 0)) insertRight(P, I, i * k, k, pnorm, j)
      if (selfJoin) {
        if (pnorm < (P[j * k + k - 1] ?? 0)) insertRight(P, I, j * k, k, pnorm, i)
        if (i < j) {
          if (pnorm < (PL[j] ?? 0)) {
            PL[j] = pnorm
            IL[j] = i
          }
          if (pnorm < (PR[i] ?? 0)) {
            PR[i] = pnorm
            IR[i] = j
          }
        }
      }
    }
  }
}

/** Distances from accumulated p-th powers. */
export function profileFromPNorms(
  state: DistanceState,
  m: number,
  k: number,
  exclusion: number,
  p: number,
): MatrixProfile {
  const root = (v: number): number => v ** (1 / p)
  return {
    m,
    k,
    exclusion,
    P: Float64Array.from(state.P, root),
    I: Int32Array.from(state.I),
    PL: Float64Array.from(state.PL, root),
    IL: Int32Array.from(state.IL),
    PR: Float64Array.from(state.PR, root),
    IR: Int32Array.from(state.IR),
  }
}

/** Check a series is long enough for a self-join with the exclusion zone (STUMPY's check_window_size with n). */
export function checkSelfJoin(m: number, n: number): number {
  checkWindow(m, n)
  const zone = exclusionZone(m)
  if (Math.floor((n - m + 1) / 2) <= zone)
    throw new OmaiError('value', `the series (length ${n}) is too short for a self-join with window ${m}`)
  return zone
}

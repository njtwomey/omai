/**
 * The shared parts of the matrix-profile algorithms: the options and the validated setup of a join, the empty profile,
 * the merge of two profiles (STUMPY's top-k merge), one batch of diagonals into a fresh profile, and the row writer and
 * clone used by the row-wise algorithms. Public algorithms re-export the option types.
 */
import { MdmlError } from '../base/index.ts'
import { rollingIsFinite, seriesOf } from '../signal/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import type { RunOptions } from '../trace/index.ts'
import {
  checkSelfJoin,
  correlationDiagonals,
  correlationState,
  distanceState,
  insertRight,
  type MatrixProfile,
  pNormDiagonals,
  prepareDiagonal,
  profileFromCorrelations,
  profileFromPNorms,
} from './_profile.ts'

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

export interface Setup {
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

export function setup(tA: TensorLike, m: number, options: MatrixProfileOptions, name: string): Setup {
  const a = seriesOf(tA, name)
  const selfJoin = options.tB === undefined
  const b = options.tB === undefined ? a : seriesOf(options.tB, name)
  if (!Number.isInteger(m) || m < 3) throw new MdmlError('value', `${name}: the window must be at least 3`)
  if (m > Math.min(a.length, b.length)) throw new MdmlError('value', `${name}: the window is longer than a series`)
  const exclusion = selfJoin ? checkSelfJoin(m, a.length) : 0
  const k = options.k ?? 1
  if (!Number.isInteger(k) || k < 1) throw new MdmlError('value', `${name}: k must be a positive integer`)
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

export function runOptions(trace: MatrixProfileTrace, maxSteps: number): RunOptions {
  return {
    maxSteps,
    ...(trace.tracer === undefined ? {} : { tracer: trace.tracer }),
    ...(trace.policy === undefined ? {} : { policy: trace.policy }),
    ...(trace.parent === undefined ? {} : { parent: trace.parent }),
    ...(trace.key === undefined ? {} : { key: trace.key }),
    ...(trace.seed === undefined ? {} : { seed: trace.seed }),
  }
}

export function emptyProfile(l: number, k: number, m: number, exclusion: number): MatrixProfile {
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
export function mergeProfiles(into: MatrixProfile, from: MatrixProfile): MatrixProfile {
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
export function diagonalProfile(s: Setup, m: number, diags: readonly number[]): MatrixProfile {
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

export function observeProfile(profile: MatrixProfile, fraction: number): { metrics: Record<string, number> } {
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

/** The state of the row-wise algorithms ({@link stompAlgorithm}, {@link stampAlgorithm}). */
export interface RowState {
  readonly profile: MatrixProfile
  readonly done: number
  readonly order: Int32Array
  /** STOMP's dot products of the last row against every window of the second series. */
  readonly qt: Float64Array
}

/** Write row i's top-k, left and right neighbours from its squared (or p-th power) distance profile. */
export function writeRow(
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

export function cloneProfile(p: MatrixProfile): MatrixProfile {
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

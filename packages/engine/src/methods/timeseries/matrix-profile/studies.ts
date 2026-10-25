/**
 * Studies of matrix-profile methods on planted ground truth: series with known regimes, motifs and a discord, scored
 * for motif recall, discord overlap and segmentation accuracy, and the anytime convergence of STAMP and SCRIMP++
 * towards the exact profile, traced batch by batch.
 */

import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import { foldIn, type Key, normal } from '../../../lib/random/index.ts'
import { exclusionZone } from '../../../lib/signal/index.ts'
import type { TensorLike } from '../../../lib/tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../../../lib/trace/index.ts'
import {
  type DiagonalState,
  type MatrixProfile,
  type RowState,
  scrumpAlgorithm,
  stampAlgorithm,
  stump,
} from './algorithms.ts'
import { discords, motifs } from './discovery.ts'
import { fluss } from './segmentation.ts'

// Matrix-profile studies on tunable data with planted ground truth: a series of regimes with a motif planted several
// times and an anomaly (a spike on a flat stretch), then whether motif discovery, discords and segmentation recover what was planted; and
// the anytime convergence of STAMP and SCRIMP++ towards the exact profile.

/** Options for {@link plantedSeries}. */
export interface PlantedOptions {
  /** Length (default 400). */
  readonly n?: number
  /** Motif length (default 20). */
  readonly m?: number
  /** Occurrences of the planted motif (default 3). */
  readonly occurrences?: number
  /** Noise standard deviation (default 0.1). */
  readonly noise?: number
  /** Number of regimes, each a different oscillation (default 2). */
  readonly regimes?: number
  /** Plant an anomaly (default true). */
  readonly discord?: boolean
}

/** A planted series and its ground truth. */
export interface PlantedSeries {
  readonly t: Float64Array
  readonly m: number
  readonly motifStarts: readonly number[]
  /** Start of the anomaly (-1 without one). */
  readonly discordStart: number
  /** Regime boundaries. */
  readonly boundaries: readonly number[]
}

/**
 * A series with planted structure: equal-length regimes of different oscillations plus noise, a motif (one period
 * of a chirp) planted at well-separated positions and a spike on a flat stretch as a discord.
 *
 * @param key - Randomness.
 * @param options - Length, motif length, occurrences, noise, regimes and discord.
 * @returns The series and its ground truth.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { plantedSeries } from 'whitebox-engine/methods/timeseries'
 * import { key } from 'whitebox-engine/random'
 * plantedSeries(key(0), { n: 200, m: 10 }).motifStarts // => [17, 83, 149]
 * ```
 */
export function plantedSeries(key: Key, options: PlantedOptions = {}): PlantedSeries {
  const n = options.n ?? 400
  const m = options.m ?? 20
  const occurrences = options.occurrences ?? 3
  const noise = options.noise ?? 0.1
  const regimes = options.regimes ?? 2
  if (occurrences * 3 * m > n) throw new WhiteboxError('value', 'plantedSeries: the series is too short for the motifs')
  const e = normal(foldIn(key, 0), [n]).data
  const boundaries = Array.from({ length: regimes - 1 }, (_, r) => Math.round(((r + 1) * n) / regimes))
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let r = 0
    while (r < boundaries.length && i >= (boundaries[r] ?? n)) r++
    const freq = 0.15 + 0.2 * r
    t[i] = (r % 2 === 0 ? Math.sin(freq * i) : Math.sign(Math.sin(freq * i)) * 0.8) + noise * (e[i] ?? 0)
  }
  const spacing = Math.floor(n / occurrences)
  const motifStarts = Array.from(
    { length: occurrences },
    (_, k) => Math.floor(k * spacing + spacing / 3) - Math.floor(m / 2),
  )
  const shape = Float64Array.from({ length: m }, (_, q) => 2 * Math.sin((Math.PI * q * q) / (m * 2)) * (1 - q / m))
  const motifNoise = normal(foldIn(key, 1), [occurrences * m]).data
  for (let k = 0; k < occurrences; k++) {
    const s = motifStarts[k] ?? 0
    for (let q = 0; q < m; q++) t[s + q] = (shape[q] ?? 0) + 0.2 * noise * (motifNoise[k * m + q] ?? 0)
  }
  let discordStart = -1
  if (options.discord ?? true) {
    discordStart = Math.floor(spacing * (occurrences - 1) + (2 * spacing) / 3)
    // A flat stretch with one sharp spike: no other window looks like it after z-normalisation.
    const level = t[discordStart] ?? 0
    const sign = (normal(foldIn(key, 2), [1]).data[0] ?? 0) < 0 ? -1 : 1
    for (let q = 0; q < m && discordStart + q < n; q++)
      t[discordStart + q] = level + (q === Math.floor(m / 2) ? 3 * sign : 0)
  }
  return { t, m, motifStarts, discordStart, boundaries }
}

/** The result of {@link discoveryStudy}. */
export interface DiscoveryStudy {
  readonly planted: PlantedSeries
  readonly profile: MatrixProfile
  /** Matches of the top motif. */
  readonly motif: readonly number[]
  /** Planted occurrences found (within an exclusion zone of a match). */
  readonly motifRecall: number
  readonly discord: number
  readonly discordFound: boolean
  readonly boundaries: readonly number[]
  /** Mean distance from each planted boundary to the nearest found one. */
  readonly boundaryError: number
}

/**
 * A discovery study: plant structure, compute the exact profile, and score motif discovery (recall of the planted
 * occurrences), the top discord (does it overlap the anomaly) and FLUSS segmentation (boundary error).
 *
 * @param key - Randomness.
 * @param options - The planted series.
 * @returns Ground truth, findings and scores.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { discoveryStudy } from 'whitebox-engine/methods/timeseries'
 * import { key } from 'whitebox-engine/random'
 * const s = discoveryStudy(key(1), { n: 240, m: 12 })
 * [s.motifRecall, s.discordFound] // => [1, true]
 * ```
 */
export function discoveryStudy(key: Key, options: PlantedOptions = {}): DiscoveryStudy {
  const planted = plantedSeries(key, options)
  const { t, m } = planted
  const profile = stump(t, m)
  const found = motifs(t, profile.P, {
    maxMatches: planted.motifStarts.length + 2,
    maxDistance: (d) => 0.25 * medianOf(d),
  })
  const motif = found.indices[0]?.filter((i) => i >= 0) ?? []
  const zone = exclusionZone(m)
  const hit = planted.motifStarts.filter((s) => motif.some((i) => Math.abs(i - s) <= zone)).length
  const top = discords(profile.P, m, 1)
  const discord = top.indices[0] ?? -1
  const discordFound =
    planted.discordStart >= 0 && discord > planted.discordStart - m && discord < planted.discordStart + m / 2
  const seg = fluss(profile.I, m, planted.boundaries.length + 1)
  const boundaryError =
    planted.boundaries.length === 0
      ? 0
      : planted.boundaries.reduce(
          (s, b) => s + seg.regimes.reduce((best, r) => Math.min(best, Math.abs(r - b)), Number.POSITIVE_INFINITY),
          0,
        ) / planted.boundaries.length
  return {
    planted,
    profile,
    motif,
    motifRecall: hit / planted.motifStarts.length,
    discord,
    discordFound,
    boundaries: seg.regimes,
    boundaryError,
  }
}

function medianOf(d: Float64Array): number {
  const f = Array.from(d)
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  return f.length === 0 ? Number.NaN : (f[Math.floor(f.length / 2)] ?? Number.NaN)
}

/** One point of an anytime-convergence curve. */
export interface AnytimePoint {
  /** Fraction of the work done. */
  readonly fraction: number
  readonly meanError: number
  readonly maxError: number
  /** Fraction of windows whose profile value is already exact. */
  readonly exactFraction: number
}

/** The state of {@link anytimeStudyAlgorithm}: the inner anytime state and the error curve so far. */
export interface AnytimeStudyState {
  readonly inner: DiagonalState | RowState
  readonly step: number
  readonly curve: readonly AnytimePoint[]
}

/**
 * The anytime convergence of STAMP or SCRIMP++ as an algorithm: each step advances the inner algorithm one batch and
 * records its mean and largest error against the exact profile and the fraction of windows already exact.
 *
 * @param t - Series.
 * @param m - Window length.
 * @param method - `scrump` (with PreSCRIMP seeding) or `stamp`.
 * @param batches - Number of batches to the full computation (default 20).
 * @returns The algorithm.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { anytimeStudyAlgorithm } from 'whitebox-engine/methods/timeseries'
 * anytimeStudyAlgorithm(Array.from({ length: 60 }, (_, i) => Math.sin(i / 3) + Math.cos(i * 1.1)), 6, 'scrump').name // => 'anytime:scrump'
 * ```
 */
export function anytimeStudyAlgorithm(
  t: TensorLike,
  m: number,
  method: 'scrump' | 'stamp',
  batches = 20,
): Algorithm<AnytimeStudyState> {
  const exact = stump(t, m).P
  const l = exact.length
  const inner: Algorithm<DiagonalState> | Algorithm<RowState> =
    method === 'scrump'
      ? scrumpAlgorithm(t, m, { percentage: 1, batch: Math.max(1, Math.ceil((l - exclusionZone(m) - 1) / batches)) })
      : stampAlgorithm(t, m, { batch: Math.max(1, Math.ceil(l / batches)) })
  const score = (P: Float64Array, fraction: number): AnytimePoint => {
    let sum = 0
    let hi = 0
    let ok = 0
    for (let i = 0; i < l; i++) {
      const e = Math.min(Math.abs((P[i] ?? Number.POSITIVE_INFINITY) - (exact[i] ?? 0)), 2 * Math.sqrt(m))
      sum += e
      hi = Math.max(hi, e)
      if (e < 1e-8) ok++
    }
    return { fraction, meanError: sum / l, maxError: hi, exactFraction: ok / l }
  }
  return {
    name: `anytime:${method}`,
    config: { m, method, batches } as Json,
    init: (key) => {
      const st = (inner as Algorithm<DiagonalState | RowState>).init(key)
      return { inner: st, step: 0, curve: [score(st.profile.P as Float64Array, 0)] }
    },
    step: (st, ctx) => {
      const next = (inner as Algorithm<DiagonalState | RowState>).step(st.inner, ctx)
      const point = score(next.profile.P as Float64Array, next.done / Math.max(1, next.order.length))
      return { inner: next, step: st.step + 1, curve: [...st.curve, point] }
    },
    done: (st) =>
      (inner as Algorithm<DiagonalState | RowState>).done?.(st.inner, st.step) === 'converged'
        ? 'converged'
        : undefined,
    observe: (st) => ({ metrics: { ...(st.curve.at(-1) ?? { fraction: 0 }) } }),
  }
}

/**
 * Run an anytime-convergence study to completion and return the error curve.
 *
 * @param t - Series.
 * @param m - Window length.
 * @param method - `scrump` or `stamp`.
 * @param options - Batches, key and tracing.
 * @returns The curve: per step the fraction done, mean and largest error and the fraction of exact windows.
 * @concept matrix-profile
 * @example
 * ```ts
 * import { anytimeStudy } from 'whitebox-engine/methods/timeseries'
 * const curve = anytimeStudy(Array.from({ length: 60 }, (_, i) => Math.sin(i / 3) + Math.cos(i * 1.1)), 6, 'stamp')
 * curve.at(-1)?.exactFraction // => 1
 * ```
 */
export function anytimeStudy(
  t: TensorLike,
  m: number,
  method: 'scrump' | 'stamp',
  options: { readonly batches?: number } & Partial<
    Pick<RunOptions, 'key' | 'seed' | 'tracer' | 'policy' | 'parent'>
  > = {},
): readonly AnytimePoint[] {
  const alg = anytimeStudyAlgorithm(t, m, method, options.batches ?? 20)
  return runAlgorithm(alg, {
    maxSteps: Number.MAX_SAFE_INTEGER,
    ...(options.key === undefined ? {} : { key: options.key }),
    ...(options.seed === undefined ? {} : { seed: options.seed }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state.curve
}

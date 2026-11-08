/**
 * Choosing the number of clusters: the gap statistic (Tibshirani, Walther and Hastie 2001), which compares the
 * within-cluster dispersion with that of uniform reference data, the stability index (Ben-Hur, Elisseeff and Guyon
 * 2002), which asks whether clusterings of overlapping subsamples agree, and the elbow curve of k-means inertia.
 */

import { maxOf, minOf, OmaiError, toRows } from '../base/index.ts'
import { matrixView } from '../learn/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { foldIn, type Key, subsample, uniform } from '../random/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { initialCentres, lloydAlgorithm, nearestCentres } from './kmeans.ts'

/** k-means by the best of `nInit` starts: the labels and the inertia. */
function bestKMeans(
  x: TensorLike,
  k: number,
  key: Key,
  nInit: number,
): { readonly labels: number[]; readonly inertia: number } {
  let best: { readonly labels: number[]; readonly inertia: number } | undefined
  for (let r = 0; r < nInit; r++) {
    const kr = foldIn(key, r)
    const out = runAlgorithm(lloydAlgorithm(x, initialCentres(x, k, 'k-means++', kr)), { maxSteps: 300, key: kr })
    const fit = nearestCentres(x, out.state.centres)
    if (best === undefined || fit.inertia < best.inertia) best = fit
  }
  return best as { readonly labels: number[]; readonly inertia: number }
}

/**
 * The elbow curve: k-means inertia for each number of clusters, for choosing k by eye.
 *
 * @param x - Points `[n, d]`.
 * @param ks - The numbers of clusters to try.
 * @param key - Randomness for the starts.
 * @param nInit - Starts per k (default 3).
 * @returns The numbers of clusters and the inertia of each fit.
 * @example
 * ```ts
 * import { kmeansElbow } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * const x = [[0], [1], [10], [11], [20], [21]]
 * kmeansElbow(x, [1, 3], key(0)).inertia.map((v) => Math.round(v)) // => [402, 2]
 * ```
 */
export function kmeansElbow(
  x: TensorLike,
  ks: readonly number[],
  key: Key,
  nInit = 3,
): { readonly ks: readonly number[]; readonly inertia: number[] } {
  return { ks, inertia: ks.map((k, i) => bestKMeans(x, k, foldIn(key, i), nInit).inertia) }
}

/** The result of the gap statistic. */
export interface GapStatistic {
  readonly ks: readonly number[]
  /** log of the pooled within-cluster dispersion of the data, per k. */
  readonly logDispersion: number[]
  /** The mean log dispersion of the reference data, per k. */
  readonly expectedLogDispersion: number[]
  /** The gap: expected minus observed log dispersion. */
  readonly gap: number[]
  /** The simulation error of the gap, per k. */
  readonly se: number[]
  /** The smallest k whose gap is within one standard error of the next (Tibshirani's rule); the last k if none. */
  readonly best: number
}

/**
 * The gap statistic (Tibshirani, Walther and Hastie 2001): for each k, how much further the log within-cluster
 * dispersion of k-means on the data falls below its expectation on uniform reference data over the bounding box of
 * the data. The chosen k is the smallest with gap(k) >= gap(k + 1) - se(k + 1).
 *
 * @param x - Points `[n, d]`.
 * @param ks - The numbers of clusters to try, ascending.
 * @param key - Randomness for the reference data and the starts.
 * @param options - Reference datasets (default 10) and starts per fit (default 3).
 * @returns The dispersions, gaps and the chosen k.
 * @index function Gap statistic for choosing the number of clusters.
 * @example
 * ```ts
 * import { gapStatistic } from 'omai-engine/cluster'
 * import { generate } from 'omai-engine/data'
 * import { blobs } from 'omai-engine/datasets'
 * import { key } from 'omai-engine/random'
 * const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } })
 * gapStatistic(data.x, [1, 2, 3, 4, 5], key(0), { references: 8 }).best // => 3
 * ```
 */
export function gapStatistic(
  x: TensorLike,
  ks: readonly number[],
  key: Key,
  options: { readonly references?: number; readonly nInit?: number } = {},
): GapStatistic {
  const m = matrixView(x, 'gapStatistic', { minSamples: 2 })
  const b = options.references ?? 10
  const nInit = options.nInit ?? 3
  if (ks.length === 0 || ks.some((k) => !(Number.isInteger(k) && k >= 1 && k <= m.n)))
    throw new OmaiError('value', `gapStatistic: every k must be an integer in 1..${m.n}`)
  const lows = Array.from({ length: m.d }, (_, q) =>
    minOf(Array.from({ length: m.n }, (_, i) => m.data[i * m.d + q] ?? 0)),
  )
  const highs = Array.from({ length: m.d }, (_, q) =>
    maxOf(Array.from({ length: m.n }, (_, i) => m.data[i * m.d + q] ?? 0)),
  )
  const logW = (data: TensorLike, k: number, kk: Key): number =>
    Math.log(Math.max(bestKMeans(data, k, kk, nInit).inertia, Number.MIN_VALUE))
  const rows = toRows(m.data, m.n, m.d)
  const logDispersion = ks.map((k, i) => logW(rows, k, foldIn(key, i)))
  const refs = Array.from({ length: b }, (_, r) => {
    const u = uniform(foldIn(key, 1000 + r), [m.n, m.d]).data
    return toRows(u, m.n, m.d).map((row) => row.map((v, q) => (lows[q] ?? 0) + v * ((highs[q] ?? 0) - (lows[q] ?? 0))))
  })
  const refLogs = ks.map((k, i) => refs.map((ref, r) => logW(ref, k, foldIn(key, 2000 + i * b + r))))
  const expectedLogDispersion = refLogs.map((v) => v.reduce((a, c) => a + c, 0) / b)
  const gap = expectedLogDispersion.map((e, i) => e - (logDispersion[i] ?? 0))
  const se = refLogs.map((v, i) => {
    const mean = expectedLogDispersion[i] ?? 0
    return Math.sqrt(v.reduce((a, c) => a + (c - mean) ** 2, 0) / b) * Math.sqrt(1 + 1 / b)
  })
  let best = ks[ks.length - 1] as number
  for (let i = 0; i < ks.length - 1; i++)
    if ((gap[i] ?? 0) >= (gap[i + 1] ?? 0) - (se[i + 1] ?? 0)) {
      best = ks[i] as number
      break
    }
  return { ks, logDispersion, expectedLogDispersion, gap, se, best }
}

/** The result of {@link stabilityIndex}. */
export interface StabilityIndex {
  readonly ks: readonly number[]
  /** The mean adjusted Rand index between clusterings of pairs of subsamples, on the rows both contain, per k. */
  readonly stability: number[]
  /** The adjusted Rand index of every pair of subsamples, per k. */
  readonly scores: number[][]
  /** The k with the highest stability (the smallest on ties). */
  readonly best: number
}

/**
 * The stability index (Ben-Hur, Elisseeff and Guyon 2002): cluster pairs of random subsamples of the rows with
 * k-means and compare the two labellings on the rows both contain by the adjusted Rand index. A k that finds real
 * structure gives clusterings that agree; a k that splits a cluster arbitrarily does not.
 *
 * @param x - Points `[n, d]`.
 * @param ks - The numbers of clusters to try.
 * @param key - Randomness for the subsamples and starts.
 * @param options - Pairs of subsamples (default 10), the fraction of rows in each (default 0.8) and starts per fit (default 2).
 * @returns The stability per k and the best k.
 * @example
 * ```ts
 * import { stabilityIndex } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [(i % 3) * 20 + (i % 5) * 0.1, Math.floor(i / 3) * 0.05])
 * stabilityIndex(x, [3, 5], key(0), { pairs: 4 }).best // => 3
 * ```
 */
export function stabilityIndex(
  x: TensorLike,
  ks: readonly number[],
  key: Key,
  options: { readonly pairs?: number; readonly fraction?: number; readonly nInit?: number } = {},
): StabilityIndex {
  const m = matrixView(x, 'stabilityIndex', { minSamples: 4 })
  const pairs = options.pairs ?? 10
  const size = Math.min(m.n, Math.max(2, Math.round((options.fraction ?? 0.8) * m.n)))
  const rows = toRows(m.data, m.n, m.d)
  const scores = ks.map((k, ki) =>
    Array.from({ length: pairs }, (_, p) => {
      const kp = foldIn(key, ki * pairs + p)
      const a = Array.from(subsample(foldIn(kp, 0), m.n, size).data)
      const b = Array.from(subsample(foldIn(kp, 1), m.n, size).data)
      const la = bestKMeans(
        a.map((i) => rows[i] as number[]),
        k,
        foldIn(kp, 2),
        options.nInit ?? 2,
      ).labels
      const lb = bestKMeans(
        b.map((i) => rows[i] as number[]),
        k,
        foldIn(kp, 3),
        options.nInit ?? 2,
      ).labels
      const posA = new Map(a.map((i, r) => [i, r]))
      const shared = b.flatMap((i, r) =>
        posA.has(i) ? [[la[posA.get(i) as number] as number, lb[r] as number] as const] : [],
      )
      if (shared.length < 2) return 1
      return adjustedRandScore(
        shared.map((s) => s[0]),
        shared.map((s) => s[1]),
      ).estimate.value
    }),
  )
  const stability = scores.map((s) => s.reduce((a, c) => a + c, 0) / Math.max(s.length, 1))
  let best = ks[0] as number
  let top = Number.NEGATIVE_INFINITY
  for (const [i, k] of ks.entries())
    if ((stability[i] ?? 0) > top + 1e-12) {
      top = stability[i] ?? 0
      best = k
    }
  return { ks, stability, scores, best }
}

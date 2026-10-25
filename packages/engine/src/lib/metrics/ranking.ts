/**
 * Ranking and recommendation metrics: precision and recall at k, hit rate, R-precision, mean average precision, mean
 * reciprocal rank, DCG and NDCG, expected reciprocal rank and rank-biased precision for search results, and
 * beyond-accuracy measures of recommendation lists (catalogue coverage, intra-list diversity, novelty, serendipity and
 * the Gini coefficient of exposure).
 */

import { type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import type { Grid } from './_scores.ts'
import {
  defineMetric,
  type Metric,
  type MetricFunction,
  type MetricOptions,
  matrix,
  metricResult,
  sampleWeights,
  weightedMean,
} from './metric.ts'

const spec = (
  name: string,
  direction: 'higher' | 'lower' = 'higher',
  range: readonly [number, number] = [0, 1],
): MetricSpec => ({
  name,
  direction,
  range,
  inputs: 'rankings',
  category: 'ranking',
  concept: 'ranking-metrics',
})

/** Options of the ranking metrics: one query per row of relevance and scores. */
export interface RankingOptions extends MetricOptions {
  /** Only the k highest-scored items count (default every item). */
  readonly k?: number
}

/** Rows of a relevance matrix `[Q, L]` (a 1-D input is one query). */
function rows(name: string, yTrue: TensorLike, yScore: TensorLike): { t: Grid; s: Grid; q: number; l: number } {
  const t0 = matrix(yTrue)
  const s0 = matrix(yScore)
  const t = t0.shape.length === 1 ? { shape: [1, t0.shape[0] ?? 0], data: t0.data } : t0
  const s = s0.shape.length === 1 ? { shape: [1, s0.shape[0] ?? 0], data: s0.data } : s0
  if (t.shape[0] !== s.shape[0] || t.shape[1] !== s.shape[1])
    throw new WhiteboxError('shape', `${name}: yTrue and yScore shapes differ`)
  return { t, s, q: t.shape[0] ?? 0, l: t.shape[1] ?? 0 }
}

/** The relevance of each query's items in decreasing score order (ties by position). */
function ranked(t: Grid, s: Grid, i: number, l: number): number[] {
  const order = Array.from({ length: l }, (_, j) => j).sort(
    (a, b) => (s.data[i * l + b] ?? 0) - (s.data[i * l + a] ?? 0) || a - b,
  )
  return order.map((j) => t.data[i * l + j] ?? 0)
}

/** Average a per-query score over queries (with sample weights). */
function perQuery(
  name: string,
  yTrue: TensorLike,
  yScore: TensorLike,
  options: MetricOptions,
  score: (rel: number[], i: number, all: { t: Grid; s: Grid; l: number }) => number,
): MetricResult {
  const r = rows(name, yTrue, yScore)
  const w = sampleWeights(name, r.q, options.sampleWeight)
  const values = Array.from({ length: r.q }, (_, i) => score(ranked(r.t, r.s, i, r.l), i, r))
  return metricResult(name, weightedMean(values, w), r.q)
}

const relevantCount = (rel: readonly number[]): number => rel.reduce((a, v) => a + (v > 0 ? 1 : 0), 0)

/** Define a ranking metric from a per-query score of the ranked relevances. */
function rankingMetric<O extends RankingOptions>(
  name: string,
  score: (rel: number[], options: O, i: number, all: { t: Grid; s: Grid; l: number }) => number,
  range: readonly [number, number] = [0, 1],
): Metric<O> {
  const fn = (yTrue: TensorLike, yScore: TensorLike, options: O = {} as O): MetricResult =>
    perQuery(name, yTrue, yScore, options, (rel, i, all) => score(rel, options, i, all))
  return Object.assign(fn, { spec: spec(name, 'higher', range) })
}

/**
 * Precision at k: the fraction of the k highest-scored items that are relevant (relevance > 0), averaged over
 * queries (trec_eval's P_k; torchmetrics' retrieval_precision).
 *
 * @param yTrue - Relevance `[Q, L]` (or `[L]` for one query).
 * @param yScore - Scores `[Q, L]`.
 * @param options - k (default L) and per-query weights.
 * @returns The metric result.
 * @concept ranking-metrics
 * @example
 * ```ts
 * import { precisionAtK } from 'whitebox-engine/metrics'
 * precisionAtK([[1, 0, 1, 0]], [[0.9, 0.8, 0.7, 0.1]], { k: 2 }).estimate.value // => 0.5
 * ```
 */
export const precisionAtK: Metric<RankingOptions> = rankingMetric('precisionAtK', (rel, o) => {
  const k = o.k ?? rel.length
  return relevantCount(rel.slice(0, k)) / k
})

/**
 * Recall at k: the fraction of a query's relevant items among its k highest-scored (0 for queries without relevant
 * items; torchmetrics' retrieval_recall).
 *
 * @param yTrue - Relevance `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { recallAtK } from 'whitebox-engine/metrics'
 * recallAtK([[1, 0, 1, 0]], [[0.9, 0.8, 0.7, 0.1]], { k: 2 }).estimate.value // => 0.5
 * ```
 */
export const recallAtK: Metric<RankingOptions> = rankingMetric('recallAtK', (rel, o) => {
  const total = relevantCount(rel)
  return total === 0 ? 0 : relevantCount(rel.slice(0, o.k ?? rel.length)) / total
})

/**
 * Hit rate at k: whether any relevant item is among the k highest-scored, averaged over queries.
 *
 * @param yTrue - Relevance `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { hitRate } from 'whitebox-engine/metrics'
 * hitRate([[0, 0, 1], [1, 0, 0]], [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], { k: 2 }).estimate.value // => 0.5
 * ```
 */
export const hitRate: Metric<RankingOptions> = rankingMetric('hitRate', (rel, o) =>
  relevantCount(rel.slice(0, o.k ?? rel.length)) > 0 ? 1 : 0,
)

/**
 * R-precision: precision at R, the number of relevant items of the query (0 when there are none).
 *
 * @param yTrue - Relevance `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - Per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { rPrecision } from 'whitebox-engine/metrics'
 * rPrecision([[1, 0, 1, 0]], [[0.9, 0.8, 0.7, 0.1]]).estimate.value // => 0.5
 * ```
 */
export const rPrecision: Metric<MetricOptions> = rankingMetric('rPrecision', (rel) => {
  const r = relevantCount(rel)
  return r === 0 ? 0 : relevantCount(rel.slice(0, r)) / r
})

/** Options of {@link meanAveragePrecision}. */
export interface AveragePrecisionAtKOptions extends RankingOptions {
  /**
   * The denominator of each query's sum of precisions at its relevant ranks: 'relevant' (the number of relevant
   * items, trec_eval's map and map_cut; default), 'min' (min(k, relevant), common in recommendation) or 'retrieved'
   * (the relevant items within the top k, torchmetrics' retrieval_average_precision).
   */
  readonly denominator?: 'relevant' | 'min' | 'retrieved'
}

/**
 * Mean average precision: for each query, the mean of the precision at each relevant item's rank (within the top k),
 * averaged over queries; queries without relevant items score 0.
 *
 * @param yTrue - Relevance `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k, denominator and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanAveragePrecision } from 'whitebox-engine/metrics'
 * meanAveragePrecision([[1, 0, 1, 0]], [[0.9, 0.8, 0.7, 0.1]]).estimate.value // => 0.8333333333333333
 * ```
 */
export const meanAveragePrecision: Metric<AveragePrecisionAtKOptions> = rankingMetric<AveragePrecisionAtKOptions>(
  'meanAveragePrecision',
  (rel, o) => {
    const k = Math.min(o.k ?? rel.length, rel.length)
    const total = relevantCount(rel)
    let hits = 0
    let sum = 0
    for (let i = 0; i < k; i++) {
      if ((rel[i] ?? 0) > 0) {
        hits++
        sum += hits / (i + 1)
      }
    }
    const d = o.denominator ?? 'relevant'
    const den = d === 'relevant' ? total : d === 'min' ? Math.min(k, total) : hits
    return den === 0 ? 0 : sum / den
  },
)

/**
 * Mean reciprocal rank: 1 / the rank of the first relevant item (0 when none is within the top k), averaged over
 * queries.
 *
 * @param yTrue - Relevance `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { meanReciprocalRank } from 'whitebox-engine/metrics'
 * meanReciprocalRank([[0, 0, 1], [1, 0, 0]], [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]).estimate.value // => 0.6666666666666666
 * ```
 */
export const meanReciprocalRank: Metric<RankingOptions> = rankingMetric('meanReciprocalRank', (rel, o) => {
  const k = Math.min(o.k ?? rel.length, rel.length)
  for (let i = 0; i < k; i++) if ((rel[i] ?? 0) > 0) return 1 / (i + 1)
  return 0
})

/** Options of the discounted cumulative gains. */
export interface DcgOptions extends RankingOptions {
  /** 'linear' gains rel (sklearn, default) or 'exponential' gains 2^rel - 1 (Burges et al., 2005). */
  readonly gain?: 'linear' | 'exponential'
  /** The base of the logarithmic discount (default 2). */
  readonly logBase?: number
  /** Break score ties by position instead of averaging the gains of tied items (default false). */
  readonly ignoreTies?: boolean
}

/** The DCG of one query (sklearn's _dcg_sample_scores, with tie-averaged gains unless ignoreTies). */
function queryDcg(t: Grid, s: Grid, i: number, l: number, o: DcgOptions, byScores: boolean): number {
  const k = Math.min(o.k ?? l, l)
  const base = o.logBase ?? 2
  const discount = Array.from({ length: l }, (_, r) => (r < k ? 1 / (Math.log(r + 2) / Math.log(base)) : 0))
  const g = (v: number): number => ((o.gain ?? 'linear') === 'exponential' ? 2 ** v - 1 : v)
  const rel = Array.from({ length: l }, (_, j) => g(t.data[i * l + j] ?? 0))
  const score = Array.from({ length: l }, (_, j) => (byScores ? (s.data[i * l + j] ?? 0) : (t.data[i * l + j] ?? 0)))
  if (o.ignoreTies === true || !byScores) {
    // sklearn reverses an ascending argsort, so among ties the later item ranks first.
    const order = Array.from({ length: l }, (_, j) => j).sort((a, b) => (score[b] ?? 0) - (score[a] ?? 0) || b - a)
    return order.reduce((acc, j, r) => acc + (rel[j] ?? 0) * (discount[r] ?? 0), 0)
  }
  // Tied items share the mean of their gains over the discounts of the ranks they occupy.
  const distinct = [...new Set(score)].sort((a, b) => b - a)
  let rank = 0
  let total = 0
  for (const v of distinct) {
    const members = score.map((x, j) => (x === v ? j : -1)).filter((j) => j >= 0)
    const meanGain = members.reduce((acc, j) => acc + (rel[j] ?? 0), 0) / members.length
    let disc = 0
    for (let r = rank; r < rank + members.length; r++) disc += discount[r] ?? 0
    total += meanGain * disc
    rank += members.length
  }
  return total
}

/**
 * Discounted cumulative gain (sklearn.metrics.dcg_score): sum over the ranking of gain / log_b(rank + 1), within the
 * top k, with the gains of tied scores averaged; averaged over queries.
 *
 * @param yTrue - Relevance grades `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k, gain, logBase, ignoreTies and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { dcgScore } from 'whitebox-engine/metrics'
 * dcgScore([[3, 2, 0]], [[0.9, 0.8, 0.1]]).estimate.value // => 4.2618595071429155
 * ```
 */
export const dcgScore: Metric<DcgOptions> = rankingMetric<DcgOptions>(
  'dcgScore',
  (_rel, o, i, all) => queryDcg(all.t, all.s, i, all.l, o, true),
  [0, Infinity],
)

/**
 * Normalised discounted cumulative gain (sklearn.metrics.ndcg_score): DCG divided by the ideal DCG of the query;
 * queries whose ideal DCG is zero score 0.
 *
 * @param yTrue - Relevance grades `[Q, L]` (non-negative).
 * @param yScore - Scores `[Q, L]`.
 * @param options - k, gain, ignoreTies and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { ndcgScore } from 'whitebox-engine/metrics'
 * ndcgScore([[3, 2, 0]], [[0.9, 0.8, 0.1]]).estimate.value // => 1
 * ```
 */
export const ndcgScore: Metric<DcgOptions> = rankingMetric<DcgOptions>('ndcgScore', (_rel, o, i, all) => {
  const ideal = queryDcg(all.t, all.s, i, all.l, o, false)
  return ideal === 0 ? 0 : queryDcg(all.t, all.s, i, all.l, o, true) / ideal
})

/**
 * Expected reciprocal rank (Chapelle et al., 2009): sum_r (1 / r) R_r prod_{i < r} (1 - R_i) with stopping
 * probabilities R = (2^g - 1) / 2^gmax from the grades, a cascade model of a user who stops at a satisfying result.
 *
 * @param yTrue - Relevance grades `[Q, L]`.
 * @param yScore - Scores `[Q, L]`.
 * @param options - k, maxGrade (default the largest grade in yTrue) and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { expectedReciprocalRank } from 'whitebox-engine/metrics'
 * expectedReciprocalRank([[1, 0]], [[0.9, 0.1]], { maxGrade: 1 }).estimate.value // => 0.5
 * ```
 */
export const expectedReciprocalRank: Metric<RankingOptions & { readonly maxGrade?: number }> = Object.assign(
  (
    yTrue: TensorLike,
    yScore: TensorLike,
    options: RankingOptions & { readonly maxGrade?: number } = {},
  ): MetricResult => {
    const top = options.maxGrade ?? matrix(yTrue).data.reduce((a, v) => Math.max(a, v), 0)
    return perQuery('expectedReciprocalRank', yTrue, yScore, options, (rel) => {
      const k = Math.min(options.k ?? rel.length, rel.length)
      let stillLooking = 1
      let err = 0
      for (let r = 0; r < k; r++) {
        const p = (2 ** (rel[r] ?? 0) - 1) / 2 ** top
        err += (stillLooking * p) / (r + 1)
        stillLooking *= 1 - p
      }
      return err
    })
  },
  { spec: spec('expectedReciprocalRank') },
)

/**
 * Rank-biased precision (Moffat and Zobel, 2008): (1 - p) sum_r rel_r p^(r - 1) for binary relevance, the expected
 * rate of relevant items seen by a user who continues to the next result with persistence p.
 *
 * @param yTrue - Relevance `[Q, L]` (relevant when > 0).
 * @param yScore - Scores `[Q, L]`.
 * @param options - persistence p (default 0.8), k and per-query weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { rankBiasedPrecision } from 'whitebox-engine/metrics'
 * rankBiasedPrecision([[1, 1]], [[0.9, 0.1]], { persistence: 0.5 }).estimate.value // => 0.75
 * ```
 */
export const rankBiasedPrecision: Metric<RankingOptions & { readonly persistence?: number }> = rankingMetric(
  'rankBiasedPrecision',
  (rel, o: RankingOptions & { readonly persistence?: number }) => {
    const p = o.persistence ?? 0.8
    const k = Math.min(o.k ?? rel.length, rel.length)
    let sum = 0
    for (let r = 0; r < k; r++) if ((rel[r] ?? 0) > 0) sum += p ** r
    return (1 - p) * sum
  },
)

const beyondSpec = (name: string, range: readonly [number, number] = [0, 1]): MetricSpec => ({
  ...spec(name, 'higher', range),
  inputs: 'sets',
  concept: 'recommender-evaluation',
})

/** Recommendation lists: one array of item ids per user. */
export type Recommendations = readonly (readonly number[])[]

/**
 * Catalogue coverage: the fraction of the catalogue's items that appear in at least one recommendation list.
 *
 * @param recommendations - Item ids recommended to each user.
 * @param nItems - The catalogue size.
 * @returns The metric result.
 * @concept recommender-evaluation
 * @example
 * ```ts
 * import { catalogCoverage } from 'whitebox-engine/metrics'
 * catalogCoverage([[0, 1], [1, 2]], 6).estimate.value // => 0.5
 * ```
 */
export const catalogCoverage: MetricFunction<[Recommendations, number]> = defineMetric(
  beyondSpec('catalogCoverage'),
  (recommendations: Recommendations, nItems: number): MetricResult => {
    const seen = new Set(recommendations.flat())
    return metricResult('catalogCoverage', seen.size / nItems, recommendations.length)
  },
)

/**
 * Intra-list diversity (Ziegler et al., 2005): the mean pairwise distance between the items of each recommendation
 * list (cosine distance of item feature vectors by default), averaged over lists.
 *
 * @param recommendations - Item ids per user.
 * @param itemFeatures - Item feature vectors `[nItems, D]`.
 * @param options - distance ('cosine', default, or 'euclidean').
 * @returns The metric result.
 * @example
 * ```ts
 * import { intraListDiversity } from 'whitebox-engine/metrics'
 * intraListDiversity([[0, 1]], [[1, 0], [0, 1]]).estimate.value // => 1
 * ```
 */
export const intraListDiversity: MetricFunction<
  [Recommendations, TensorLike, { readonly distance?: 'cosine' | 'euclidean' }?]
> = defineMetric(
  beyondSpec('intraListDiversity', [0, 2]),
  (
    recommendations: Recommendations,
    itemFeatures: TensorLike,
    options: { readonly distance?: 'cosine' | 'euclidean' } = {},
  ): MetricResult => {
    const f = matrix(itemFeatures)
    const d = f.shape[1] ?? 0
    const vec = (i: number): Float64Array => f.data.subarray(i * d, (i + 1) * d)
    const dist = (a: Float64Array, b: Float64Array): number => {
      let dot = 0
      let na = 0
      let nb = 0
      let sq = 0
      for (let j = 0; j < d; j++) {
        dot += (a[j] ?? 0) * (b[j] ?? 0)
        na += (a[j] ?? 0) ** 2
        nb += (b[j] ?? 0) ** 2
        sq += ((a[j] ?? 0) - (b[j] ?? 0)) ** 2
      }
      return options.distance === 'euclidean' ? Math.sqrt(sq) : 1 - dot / Math.sqrt(na * nb)
    }
    const per = recommendations.map((list) => {
      let total = 0
      let pairs = 0
      for (let a = 0; a < list.length; a++) {
        for (let b = a + 1; b < list.length; b++) {
          total += dist(vec(list[a] ?? 0), vec(list[b] ?? 0))
          pairs++
        }
      }
      return pairs === 0 ? 0 : total / pairs
    })
    return metricResult('intraListDiversity', per.reduce((a, b) => a + b, 0) / per.length, per.length)
  },
)

/**
 * Novelty (Zhou et al., 2010): the mean self-information -log2 p(i) of recommended items, with p(i) the share of
 * users (or interactions) involving item i; recommending the long tail scores higher.
 *
 * @param recommendations - Item ids per user.
 * @param popularity - Interaction counts (or probabilities) per item `[nItems]`.
 * @returns The metric result, in bits.
 * @example
 * ```ts
 * import { novelty } from 'whitebox-engine/metrics'
 * Number(novelty([[0], [1]], [3, 1]).estimate.value.toFixed(6)) // => 1.207519
 * ```
 */
export const novelty: MetricFunction<[Recommendations, TensorLike]> = defineMetric(
  beyondSpec('novelty', [0, Infinity]),
  (recommendations: Recommendations, popularity: TensorLike): MetricResult => {
    const pop = matrix(popularity).data
    const total = pop.reduce((a, b) => a + b, 0)
    const items = recommendations.flat()
    const info = items.map((i) => -Math.log2((pop[i] ?? 0) / total))
    return metricResult('novelty', info.reduce((a, b) => a + b, 0) / items.length, recommendations.length)
  },
)

/**
 * Serendipity (Ge, Delgado-Battenfeld and Jannach, 2010): the fraction of each list's items that are relevant and not
 * among the expected (obvious, for example most-popular) recommendations, averaged over users.
 *
 * @param recommendations - Item ids per user.
 * @param relevant - The relevant item ids of each user.
 * @param expected - The expected item ids of each user (a primitive recommender's output).
 * @returns The metric result.
 * @example
 * ```ts
 * import { serendipity } from 'whitebox-engine/metrics'
 * serendipity([[0, 1, 2, 3]], [[0, 1]], [[0]]).estimate.value // => 0.25
 * ```
 */
export const serendipity: MetricFunction<[Recommendations, Recommendations, Recommendations]> = defineMetric(
  beyondSpec('serendipity'),
  (recommendations: Recommendations, relevant: Recommendations, expected: Recommendations): MetricResult => {
    const per = recommendations.map((list, u) => {
      const rel = new Set(relevant[u] ?? [])
      const obvious = new Set(expected[u] ?? [])
      const hits = list.filter((i) => rel.has(i) && !obvious.has(i)).length
      return list.length === 0 ? 0 : hits / list.length
    })
    return metricResult('serendipity', per.reduce((a, b) => a + b, 0) / per.length, per.length)
  },
)

/**
 * The Gini coefficient of item exposure across recommendation lists (0: every item equally exposed, near 1: a few
 * items get all the attention), with exposure weighted by position ('log': 1 / log2(rank + 1), the DCG discount) or
 * counted uniformly.
 *
 * @param recommendations - Item ids per user, best first.
 * @param nItems - The catalogue size (items never recommended have zero exposure).
 * @param options - position weighting ('uniform', default, or 'log').
 * @returns The metric result.
 * @example
 * ```ts
 * import { exposureGini } from 'whitebox-engine/metrics'
 * exposureGini([[0, 1], [1, 0]], 2).estimate.value // => 0
 * ```
 */
export const exposureGini: MetricFunction<[Recommendations, number, { readonly position?: 'uniform' | 'log' }?]> =
  defineMetric(
    { ...beyondSpec('exposureGini'), direction: 'lower' },
    (
      recommendations: Recommendations,
      nItems: number,
      options: { readonly position?: 'uniform' | 'log' } = {},
    ): MetricResult => {
      const exposure = new Float64Array(nItems)
      for (const list of recommendations) {
        for (const [r, i] of list.entries())
          exposure[i] = (exposure[i] ?? 0) + (options.position === 'log' ? 1 / Math.log2(r + 2) : 1)
      }
      const sorted = Array.from(exposure).sort((a, b) => a - b)
      const total = sorted.reduce((a, b) => a + b, 0)
      const n = sorted.length
      // Gini = sum_i (2i - n - 1) x_(i) / (n sum x), for the ascending order statistics.
      const g = total === 0 ? 0 : sorted.reduce((a, x, i) => a + (2 * (i + 1) - n - 1) * x, 0) / (n * total)
      return metricResult('exposureGini', g, recommendations.length)
    },
  )

/**
 * The shared machinery of the outlier detectors: the score-and-label verbs (scores higher for more anomalous, labels -1
 * outlier and 1 inlier), thresholds by contamination fraction, quantile, peaks-over-threshold (generalised Pareto tail
 * fit), Tukey's fences or a robust z-score, the exposition every detector returns, and score normalisation (z-score,
 * min-max, rank, Gaussian unification) and combination (average, maximization, average-of-maximum, maximum-of-average)
 * for ensembles of detectors (Kriegel et al. 2011; Aggarwal and Sathe 2015).
 */

import { MdmlError, maxOf, minOf } from '../base/index.ts'
import { type AnomalyScores, anomalyScores, type Exposition } from '../exposition/index.ts'
import { decisionSurfaceOf, type Fitted, type LearnData, matrixView } from '../learn/index.ts'
import { special } from '../numerics/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { fitGpd, peaksOverThreshold, quantile, ranksOf } from '../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** The verbs of an outlier detector: anomaly scores (higher is more anomalous) and labels (`-1` outlier, `1` inlier). */
export interface OutlierVerbs {
  /** Anomaly scores `[m]`, higher is more anomalous. */
  readonly scoreSamples: (x: TensorLike) => Tensor
  /** `threshold - score`: negative for outliers (scikit-learn's sign convention). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** `-1` for outliers, `1` for inliers. */
  readonly predict: (x: TensorLike) => Tensor
  readonly score?: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/**
 * Build the standard outlier verbs (`scoreSamples`, `decisionFunction`, `predict`, `nFeaturesIn`) of a scorer with a
 * threshold.
 *
 * @param scores - Anomaly scores of new rows (higher is more anomalous).
 * @param threshold - The score threshold above which a row is flagged as an outlier (`-1`).
 * @param d - The number of input features.
 * @returns The outlier verbs.
 * @example
 * ```ts
 * import { outlierVerbs } from 'mdml-engine/anomaly'
 * import { toFlat } from 'mdml-engine/tensor'
 * const v = outlierVerbs(() => [0.2, 0.8], 0.5, 1)
 * toFlat(v.predict([[0], [1]])) // => [1, -1]
 * ```
 */
export function outlierVerbs(scores: (x: TensorLike) => number[], threshold: number, d: number): OutlierVerbs {
  const vec = (v: readonly number[]): Tensor => fromStorage(Float64Array.from(v), [v.length], 'float64')
  return {
    scoreSamples: (x) => vec(scores(x)),
    decisionFunction: (x) => vec(scores(x).map((s) => threshold - s)),
    predict: (x) => {
      const s = scores(x)
      return fromStorage(
        Int32Array.from(s, (v) => (v > threshold ? -1 : 1)),
        [s.length],
        'int32',
      )
    },
    nFeaturesIn: d,
  }
}

/**
 * The threshold on anomaly scores that flags the given contamination fraction of training rows.
 *
 * @param scores - Training anomaly scores.
 * @param contamination - The fraction flagged in `(0, 0.5]` (default 0.1).
 * @returns The `(1 - contamination)` quantile of `scores`.
 * @example
 * ```ts
 * import { contaminationThreshold } from 'mdml-engine/anomaly'
 * contaminationThreshold([1, 2, 3, 4, 5], 0.25) // => 4
 * ```
 */
export function contaminationThreshold(scores: readonly number[], contamination: number = 0.1): number {
  if (!(contamination > 0 && contamination <= 0.5)) throw new MdmlError('value', 'contamination must be in (0, 0.5]')
  return quantile(scores as number[], 1 - contamination)
}

/**
 * The threshold at a fixed quantile of the anomaly scores.
 *
 * @param scores - Anomaly scores.
 * @param q - Quantile level in `[0, 1]` (default 0.9).
 * @returns The `q` quantile of `scores`.
 * @example
 * ```ts
 * import { quantileThreshold } from 'mdml-engine/anomaly'
 * quantileThreshold([0, 2, 4, 6, 8], 0.75) // => 6
 * ```
 */
export function quantileThreshold(scores: readonly number[], q: number = 0.9): number {
  if (!(q >= 0 && q <= 1)) throw new MdmlError('value', 'quantileThreshold: q must lie in [0, 1]')
  return quantile(scores as number[], q)
}

/** Options of {@link potThreshold}. */
export interface PotThresholdOptions {
  /** Target exceedance probability (tail risk `q`, default 0.01). */
  readonly risk?: number
  /** Initial high quantile `q0` defining the peaks-over-threshold base level `u` (default 0.8). */
  readonly initQuantile?: number
}

/**
 * Extreme-value threshold by Peaks-Over-Threshold (Siffer et al. 2017, SPOT): fits a generalised Pareto distribution
 * to exceedances above an initial high quantile `u = quantile(scores, initQuantile)` and returns the level exceeded
 * with probability `risk` (falling back to the empirical `(1 - risk)` quantile when too few exceedances exist).
 *
 * @param scores - Anomaly scores.
 * @param options - Target tail risk (`risk`, default 0.01) and initial threshold quantile (`initQuantile`, default 0.8).
 * @returns The extreme-value threshold.
 * @example
 * ```ts
 * import { potThreshold } from 'mdml-engine/anomaly'
 * const s = Array.from({ length: 50 }, (_, i) => (i + 1) / 10)
 * potThreshold(s, { risk: 0.02, initQuantile: 0.8 }) > 4 // => true
 * ```
 */
export function potThreshold(scores: readonly number[], options: PotThresholdOptions = {}): number {
  if (scores.length === 0) throw new MdmlError('value', 'potThreshold: scores must not be empty')
  const risk = options.risk ?? 0.01
  const q0 = options.initQuantile ?? 0.8
  if (!(risk > 0 && risk < 1)) throw new MdmlError('value', 'potThreshold: risk must lie in (0, 1)')
  if (!(q0 > 0 && q0 < 1)) throw new MdmlError('value', 'potThreshold: initQuantile must lie in (0, 1)')
  const u = quantile(scores as number[], q0)
  const pot = peaksOverThreshold(scores as number[], u)
  const exc = toFlat(pot.exceedances)
  if (exc.length < 3 || !(pot.rate > risk)) {
    return quantile(scores as number[], 1 - risk)
  }
  try {
    const fit = fitGpd(exc, { threshold: 0 })
    const { scale, shape } = fit.params
    const ratio = pot.rate / risk
    const z = Math.abs(shape) < 1e-10 ? u + scale * Math.log(ratio) : u + (scale / shape) * (ratio ** shape - 1)
    return Number.isFinite(z) && z >= u ? z : quantile(scores as number[], 1 - risk)
  } catch {
    return quantile(scores as number[], 1 - risk)
  }
}

/**
 * Tukey's fences: the upper fence `Q3 + k * IQR` of anomaly scores.
 *
 * @param scores - Anomaly scores.
 * @param k - The interquartile multiplier (default 1.5).
 * @returns The threshold.
 * @example
 * ```ts
 * import { iqrThreshold } from 'mdml-engine/anomaly'
 * iqrThreshold([1, 2, 3, 4, 5], 1.5) // => 7
 * ```
 */
export function iqrThreshold(scores: readonly number[], k: number = 1.5): number {
  const q1 = quantile(scores as number[], 0.25)
  const q3 = quantile(scores as number[], 0.75)
  return q3 + k * (q3 - q1)
}

/**
 * A robust z threshold: `median + k * 1.4826 * MAD` of anomaly scores.
 *
 * @param scores - Anomaly scores.
 * @param k - The number of robust standard deviations (default 3.5).
 * @returns The threshold.
 * @example
 * ```ts
 * import { madThreshold } from 'mdml-engine/anomaly'
 * madThreshold([1, 2, 3, 4, 5], 0) // => 3
 * ```
 */
export function madThreshold(scores: readonly number[], k: number = 3.5): number {
  const med = quantile(scores as number[], 0.5)
  const mad = quantile(
    scores.map((v) => Math.abs(v - med)),
    0.5,
  )
  return med + k * 1.4826 * mad
}

/**
 * Flag rows whose anomaly score exceeds `threshold`.
 *
 * @param scores - Anomaly scores.
 * @param threshold - The cut-off threshold.
 * @returns A boolean mask (`true` where `score > threshold`).
 * @example
 * ```ts
 * import { flagOutliers } from 'mdml-engine/anomaly'
 * flagOutliers([0.1, 0.4, 0.9], 0.5) // => [false, false, true]
 * ```
 */
export function flagOutliers(scores: readonly number[], threshold: number): boolean[] {
  return scores.map((s) => s > threshold)
}

/**
 * An outlier detector's exposition: training scores with the threshold, the per-row evidence, and for one or two
 * features the anomaly score surface over a grid.
 *
 * @param fitted - The fitted detector.
 * @param x - The training data `[n, d]`.
 * @param method - The method name.
 * @param threshold - The score threshold.
 * @param evidence - Per-row evidence.
 * @param training - The training rows' scores, when they differ from rescoring them (leave-self-out detectors).
 * @returns The exposition records.
 * @example
 * ```ts
 * import { outlierExposition, outlierVerbs } from 'mdml-engine/anomaly'
 * const verbs = outlierVerbs(() => [0.1, 0.9], 0.5, 1)
 * const fitted = { estimator: 'demo', kind: 'outlier' as const, params: {}, ...verbs }
 * outlierExposition(fitted, [[0], [1]], 'demo', 0.5, { name: 'score', values: [0.1, 0.9] })[0]?.kind // => 'anomalyScores'
 * ```
 */
export function outlierExposition(
  fitted: Fitted & OutlierVerbs,
  x: TensorLike,
  method: string,
  threshold: number,
  evidence: { readonly name: string; readonly values: readonly number[] },
  training?: readonly number[],
): Exposition[] {
  const out: Exposition[] = [
    anomalyScores({ method, scores: training ?? toFlat(fitted.scoreSamples(x)), threshold, evidence }),
  ]
  const d = matrixView(x, method).d
  if (d <= 2)
    out.push(
      decisionSurfaceOf({ ...fitted, predict: fitted.scoreSamples }, x, {
        resolution: 30,
        field: { name: 'anomaly score' },
        // Ground-truth flags, when the rows have them, read 0 inlier and 1 outlier.
        classNames: ['inlier', 'outlier'],
      }),
    )
  return out
}

// ---------------------------------------------------------------------------------------------------------------
// Score normalisation

/** How scores from different detectors are put on one common scale. */
export type ScoreNormalisation = 'minmax' | 'zscore' | 'rank' | 'unify' | 'gaussian'

/**
 * Standardise anomaly scores to zero mean and unit standard deviation (`z-score` normalisation).
 *
 * @param scores - Raw anomaly scores.
 * @returns Standardised scores.
 * @example
 * ```ts
 * import { zscoreNormalise } from 'mdml-engine/anomaly'
 * zscoreNormalise([1, 2, 3]) // => [-1.224744871391589, 0, 1.224744871391589]
 * ```
 */
export function zscoreNormalise(scores: readonly number[]): number[] {
  const n = scores.length
  if (n === 0) return []
  const mean = scores.reduce((a, v) => a + v, 0) / n
  const sd = Math.sqrt(scores.reduce((a, v) => a + (v - mean) ** 2, 0) / n)
  return scores.map((v) => (sd > 0 ? (v - mean) / sd : 0))
}

/**
 * Scale anomaly scores linearly to `[0, 1]` (`min-max` normalisation).
 *
 * @param scores - Raw anomaly scores.
 * @returns Scores in `[0, 1]`.
 * @example
 * ```ts
 * import { minmaxNormalise } from 'mdml-engine/anomaly'
 * minmaxNormalise([1, 5, 9]) // => [0, 0.5, 1]
 * ```
 */
export function minmaxNormalise(scores: readonly number[]): number[] {
  const n = scores.length
  if (n === 0) return []
  const lo = minOf(scores)
  const hi = maxOf(scores)
  return scores.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0))
}

/**
 * Scale anomaly scores by their rank to `[0, 1]` with average ranks for ties.
 *
 * @param scores - Raw anomaly scores.
 * @returns Rank-scaled scores in `[0, 1]`.
 * @example
 * ```ts
 * import { rankNormalise } from 'mdml-engine/anomaly'
 * rankNormalise([3, 1, 1]) // => [1, 0.25, 0.25]
 * ```
 */
export function rankNormalise(scores: readonly number[]): number[] {
  const n = scores.length
  return Array.from(ranksOf(scores).data, (r) => (n > 1 ? (r - 1) / (n - 1) : 0))
}

/**
 * Unify anomaly scores into outlier probability estimates in `[0, 1]` by Gaussian scaling (Kriegel et al. 2011):
 * `max(0, erf((s - mean) / (std * sqrt(2))))`.
 *
 * @param scores - Raw anomaly scores.
 * @returns Unified outlier probabilities in `[0, 1]`.
 * @example
 * ```ts
 * import { unifyScores } from 'mdml-engine/anomaly'
 * unifyScores([0, 0, 10])[0] // => 0
 * ```
 */
export function unifyScores(scores: readonly number[]): number[] {
  const n = scores.length
  if (n === 0) return []
  const mean = scores.reduce((a, v) => a + v, 0) / n
  const sd = Math.sqrt(scores.reduce((a, v) => a + (v - mean) ** 2, 0) / n)
  if (!(sd > 0)) return new Array<number>(n).fill(0)
  const inv = 1 / (sd * Math.SQRT2)
  const scaled = scores.map((v) => (v - mean) * inv)
  const erfs = toFlat(special.erf(scaled) as Tensor)
  return erfs.map((v) => Math.max(0, Math.min(1, v)))
}

/**
 * Put anomaly scores on a common scale: min-max to `[0, 1]`, z-scores, ranks in `[0, 1]` (ties averaged), or Gaussian
 * probability unification (`'unify'` / `'gaussian'`, Kriegel et al. 2011).
 *
 * @param scores - Scores (higher is more anomalous).
 * @param method - The normalisation (default `'zscore'`).
 * @returns The normalised scores.
 * @example
 * ```ts
 * import { normaliseScores } from 'mdml-engine/anomaly'
 * normaliseScores([1, 2, 3, 10], 'minmax') // => [0, 0.1111111111111111, 0.2222222222222222, 1]
 * ```
 */
export function normaliseScores(scores: readonly number[], method: ScoreNormalisation = 'zscore'): number[] {
  if (method === 'minmax') return minmaxNormalise(scores)
  if (method === 'rank') return rankNormalise(scores)
  if (method === 'unify' || method === 'gaussian') return unifyScores(scores)
  return zscoreNormalise(scores)
}

// ---------------------------------------------------------------------------------------------------------------
// Score combination

/** How normalised scores of several detectors are combined across detectors. */
export type ScoreCombination = 'mean' | 'average' | 'max' | 'maximization' | 'median' | 'aom' | 'moa'

function checkScoreMatrix(who: string, scores: readonly (readonly number[])[]): number {
  if (scores.length === 0) throw new MdmlError('value', `${who}: no detectors`)
  const n = scores[0]?.length ?? 0
  if (scores.some((s) => s.length !== n)) throw new MdmlError('shape', `${who}: detectors disagree on rows`)
  return n
}

function partitionBuckets(k: number, nBuckets?: number, key?: Key): number[][] {
  const b = Math.min(k, Math.max(1, nBuckets ?? Math.max(2, Math.floor(k / 2))))
  const order = key === undefined ? Array.from({ length: k }, (_, i) => i) : Array.from(permutation(key, k).data)
  const buckets: number[][] = Array.from({ length: b }, () => [])
  for (let i = 0; i < k; i++) buckets[i % b]?.push(order[i] ?? i)
  return buckets.filter((g) => g.length > 0)
}

/**
 * Combine detector score vectors by their per-row arithmetic mean (Aggarwal and Sathe 2015).
 *
 * @param scores - `scores[k][i]`: detector `k` on row `i`.
 * @returns The averaged score per row.
 * @example
 * ```ts
 * import { averageScores } from 'mdml-engine/anomaly'
 * averageScores([[1, 2], [3, 6]]) // => [2, 4]
 * ```
 */
export function averageScores(scores: readonly (readonly number[])[]): number[] {
  const n = checkScoreMatrix('averageScores', scores)
  const k = scores.length
  return Array.from({ length: n }, (_, i) => scores.reduce((a, s) => a + (s[i] ?? 0), 0) / k)
}

/**
 * Combine detector score vectors by their per-row maximum (Aggarwal and Sathe 2015).
 *
 * @param scores - `scores[k][i]`: detector `k` on row `i`.
 * @returns The maximum score per row.
 * @example
 * ```ts
 * import { maximizationScores } from 'mdml-engine/anomaly'
 * maximizationScores([[1, 5], [3, 2]]) // => [3, 5]
 * ```
 */
export function maximizationScores(scores: readonly (readonly number[])[]): number[] {
  const n = checkScoreMatrix('maximizationScores', scores)
  return Array.from({ length: n }, (_, i) => maxOf(scores.map((s) => s[i] ?? Number.NEGATIVE_INFINITY)))
}

/**
 * Average-of-Maximum (AOM) score combination (Aggarwal and Sathe 2015): partitions the detectors into `nBuckets`
 * subgroups, takes the per-row maximum within each subgroup, and averages across subgroups to reduce variance.
 *
 * @param scores - `scores[k][i]`: detector `k` on row `i`.
 * @param options - Number of buckets (`nBuckets`) and optional random `key` for subgroup assignment.
 * @returns The AOM combined score per row.
 * @example
 * ```ts
 * import { aomScores } from 'mdml-engine/anomaly'
 * aomScores([[1, 2], [3, 4], [2, 6], [4, 2]], { nBuckets: 2 }) // => [3, 5]
 * ```
 */
export function aomScores(
  scores: readonly (readonly number[])[],
  options: { readonly nBuckets?: number; readonly key?: Key } = {},
): number[] {
  const n = checkScoreMatrix('aomScores', scores)
  const buckets = partitionBuckets(scores.length, options.nBuckets, options.key)
  return Array.from({ length: n }, (_, i) => {
    let sum = 0
    for (const g of buckets) {
      let top = Number.NEGATIVE_INFINITY
      for (const d of g) top = Math.max(top, scores[d]?.[i] ?? Number.NEGATIVE_INFINITY)
      sum += top
    }
    return sum / buckets.length
  })
}

/**
 * Maximization-of-Average (MOA) score combination (Aggarwal and Sathe 2015): partitions the detectors into `nBuckets`
 * subgroups, averages the scores within each subgroup, and takes the per-row maximum across subgroups.
 *
 * @param scores - `scores[k][i]`: detector `k` on row `i`.
 * @param options - Number of buckets (`nBuckets`) and optional random `key` for subgroup assignment.
 * @returns The MOA combined score per row.
 * @example
 * ```ts
 * import { moaScores } from 'mdml-engine/anomaly'
 * moaScores([[1, 2], [3, 8], [2, 4], [4, 2]], { nBuckets: 2 }) // => [3.5, 5]
 * ```
 */
export function moaScores(
  scores: readonly (readonly number[])[],
  options: { readonly nBuckets?: number; readonly key?: Key } = {},
): number[] {
  const n = checkScoreMatrix('moaScores', scores)
  const buckets = partitionBuckets(scores.length, options.nBuckets, options.key)
  return Array.from({ length: n }, (_, i) => {
    let best = Number.NEGATIVE_INFINITY
    for (const g of buckets) {
      const avg = g.reduce((a, d) => a + (scores[d]?.[i] ?? 0), 0) / g.length
      best = Math.max(best, avg)
    }
    return best
  })
}

/** Options of {@link combineScores}. */
export interface CombineScoresOptions {
  /** How each detector's scores are normalised before combining (default `'zscore'`). */
  readonly normalisation?: ScoreNormalisation
  /** How normalised scores are aggregated across detectors (default `'mean'`). */
  readonly combination?: ScoreCombination
  /** Fraction of rows flagged to set the threshold (default 0.1). */
  readonly contamination?: number
  /** Number of detector buckets for `'aom'` and `'moa'`. */
  readonly nBuckets?: number
  /** Random key for bucket assignment in `'aom'` and `'moa'`. */
  readonly key?: Key
}

/**
 * Combine several detectors' scores for the same rows (Aggarwal and Sathe 2015): normalise each detector's score
 * vector (`'zscore'`, `'minmax'`, `'rank'` or `'unify'`), then combine per row (`'mean'`/`'average'`,
 * `'max'`/`'maximization'`, `'median'`, `'aom'` or `'moa'`). Returns an `anomalyScores` record whose evidence is the
 * index of the detector that scored each row highest.
 *
 * @param scores - `scores[k][i]`: detector `k` on row `i`.
 * @param options - Normalisation, combination rule, bucket count and contamination fraction (default 0.1).
 * @returns The combined `anomalyScores` record.
 * @example
 * ```ts
 * import { combineScores } from 'mdml-engine/anomaly'
 * combineScores([[1, 2, 3, 10], [0.1, 0.3, 0.2, 5]], { combination: 'max', contamination: 0.25 }).evidence?.values[3] // => 1
 * ```
 */
export function combineScores(
  scores: readonly (readonly number[])[],
  options: CombineScoresOptions = {},
): AnomalyScores {
  const n = checkScoreMatrix('combineScores', scores)
  const norm = scores.map((s) => normaliseScores(s, options.normalisation ?? 'zscore'))
  const how = options.combination ?? 'mean'
  let combined: number[]
  if (how === 'max' || how === 'maximization') {
    combined = maximizationScores(norm)
  } else if (how === 'aom') {
    combined = aomScores(norm, {
      ...(options.nBuckets === undefined ? {} : { nBuckets: options.nBuckets }),
      ...(options.key === undefined ? {} : { key: options.key }),
    })
  } else if (how === 'moa') {
    combined = moaScores(norm, {
      ...(options.nBuckets === undefined ? {} : { nBuckets: options.nBuckets }),
      ...(options.key === undefined ? {} : { key: options.key }),
    })
  } else if (how === 'median') {
    combined = Array.from({ length: n }, (_, i) => {
      const col = norm.map((s) => s[i] ?? 0).sort((a, b) => a - b)
      const k = col.length
      return k % 2 === 1 ? (col[(k - 1) / 2] ?? 0) : ((col[k / 2 - 1] ?? 0) + (col[k / 2] ?? 0)) / 2
    })
  } else {
    combined = averageScores(norm)
  }
  const leader = Array.from({ length: n }, (_, i) => {
    let best = 0
    for (let k = 1; k < norm.length; k++) if ((norm[k]?.[i] ?? 0) > (norm[best]?.[i] ?? 0)) best = k
    return best
  })
  return anomalyScores({
    method: `combined-${how}`,
    scores: combined,
    threshold: contaminationThreshold(combined, options.contamination ?? 0.1),
    evidence: { name: 'strongestDetector', values: leader },
  })
}

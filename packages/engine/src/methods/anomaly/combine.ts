/**
 * The shared machinery of the outlier detectors: the score-and-label verbs (scores higher for more anomalous, labels -1
 * outlier and 1 inlier), thresholds by contamination fraction, Tukey's fences or a robust z-score, the exposition every
 * detector returns, and score normalisation and combination for ensembles of detectors (Aggarwal and Sathe, 2015).
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { anomalyScores, type Exposition } from '../../lib/exposition/index.ts'
import { decisionSurfaceOf, type Fitted, type LearnData, matrixView } from '../../lib/learn/index.ts'
import { quantile } from '../../lib/stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'

/** The verbs of an outlier detector: anomaly scores (higher is more anomalous) and labels (-1 outlier, 1 inlier). */
export interface OutlierVerbs {
  /** Anomaly scores `[m]`, higher is more anomalous. */
  readonly scoreSamples: (x: TensorLike) => Tensor
  /** threshold - score: negative for outliers (sklearn's sign convention). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** -1 for outliers, 1 for inliers. */
  readonly predict: (x: TensorLike) => Tensor
  readonly score?: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/**
 * The outlier verbs of a scorer with a threshold.
 *
 * @param scores - Anomaly scores of new rows (higher is more anomalous).
 * @param threshold - The threshold.
 * @param d - The number of features.
 * @returns The verbs.
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
 * The threshold on anomaly scores that flags the given fraction of training rows.
 *
 * @param scores - Training anomaly scores.
 * @param contamination - The fraction flagged.
 * @returns The (1 - contamination) quantile.
 */
export function contaminationThreshold(scores: readonly number[], contamination: number): number {
  if (!(contamination > 0 && contamination <= 0.5))
    throw new WhiteboxError('value', 'contamination must be in (0, 0.5]')
  return quantile(scores as number[], 1 - contamination)
}

/**
 * An outlier detector's exposition: training scores with the threshold, the evidence, and for one or two features
 * the score over a grid.
 *
 * @param fitted - The fitted detector.
 * @param x - The training data.
 * @param method - The method name.
 * @param threshold - The score threshold.
 * @param evidence - Per-row evidence.
 * @param training - The training rows' scores, when they differ from rescoring them (leave-self-out detectors).
 * @returns The records.
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

/** How scores from different detectors are put on one scale. */
export type ScoreNormalisation = 'minmax' | 'zscore' | 'rank'

/**
 * Put anomaly scores on a common scale: min-max to [0, 1], z-scores, or ranks scaled to [0, 1] (ties averaged).
 *
 * @param scores - Scores (higher is more anomalous).
 * @param method - The normalisation (default z-scores).
 * @returns The normalised scores.
 */
export function normaliseScores(scores: readonly number[], method: ScoreNormalisation = 'zscore'): number[] {
  const n = scores.length
  if (n === 0) return []
  if (method === 'minmax') {
    const lo = Math.min(...scores)
    const hi = Math.max(...scores)
    return scores.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0))
  }
  if (method === 'rank') {
    const order = scores.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
    const ranks = new Array<number>(n).fill(0)
    for (let i = 0; i < n; ) {
      let j = i
      while (j + 1 < n && order[j + 1]?.[0] === order[i]?.[0]) j++
      for (let k = i; k <= j; k++) ranks[order[k]?.[1] ?? 0] = (i + j) / 2
      i = j + 1
    }
    return ranks.map((r) => (n > 1 ? r / (n - 1) : 0))
  }
  const mean = scores.reduce((a, v) => a + v, 0) / n
  const sd = Math.sqrt(scores.reduce((a, v) => a + (v - mean) ** 2, 0) / n)
  return scores.map((v) => (sd > 0 ? (v - mean) / sd : 0))
}

/** How normalised scores of several detectors are combined. */
export type ScoreCombination = 'mean' | 'max' | 'median'

/**
 * Combine several detectors' scores for the same rows (Aggarwal and Sathe 2015): normalise each, then take the mean,
 * maximum or median per row. Its result is an `anomalyScores` record whose evidence is the index of the detector
 * that scored each row highest.
 *
 * @param scores - scores[k][i]: detector k on row i.
 * @param options - Normalisation, combination, and the fraction flagged (default 0.1).
 * @returns The combined record.
 */
export function combineScores(
  scores: readonly (readonly number[])[],
  options: {
    readonly normalisation?: ScoreNormalisation
    readonly combination?: ScoreCombination
    readonly contamination?: number
  } = {},
): ReturnType<typeof anomalyScores> {
  if (scores.length === 0) throw new WhiteboxError('value', 'combineScores: no detectors')
  const n = scores[0]?.length ?? 0
  if (scores.some((s) => s.length !== n)) throw new WhiteboxError('shape', 'combineScores: detectors disagree on rows')
  const norm = scores.map((s) => normaliseScores(s, options.normalisation ?? 'zscore'))
  const how = options.combination ?? 'mean'
  const combined = Array.from({ length: n }, (_, i) => {
    const col = norm.map((s) => s[i] ?? 0).sort((a, b) => a - b)
    if (how === 'max') return col.at(-1) ?? 0
    if (how === 'median') {
      const k = col.length
      return k % 2 === 1 ? (col[(k - 1) / 2] ?? 0) : ((col[k / 2 - 1] ?? 0) + (col[k / 2] ?? 0)) / 2
    }
    return col.reduce((a, v) => a + v, 0) / col.length
  })
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

/**
 * Tukey's fences: the upper fence Q3 + k IQR of scores.
 *
 * @param scores - Scores.
 * @param k - The multiplier (default 1.5).
 * @returns The threshold.
 */
export function iqrThreshold(scores: readonly number[], k = 1.5): number {
  const q1 = quantile(scores as number[], 0.25)
  const q3 = quantile(scores as number[], 0.75)
  return q3 + k * (q3 - q1)
}

/**
 * A robust z threshold: median + k * 1.4826 MAD of scores.
 *
 * @param scores - Scores.
 * @param k - The number of robust standard deviations (default 3.5).
 * @returns The threshold.
 */
export function madThreshold(scores: readonly number[], k = 3.5): number {
  const med = quantile(scores as number[], 0.5)
  const mad = quantile(
    scores.map((v) => Math.abs(v - med)),
    0.5,
  )
  return med + k * 1.4826 * mad
}

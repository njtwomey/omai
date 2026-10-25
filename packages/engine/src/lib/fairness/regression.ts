/**
 * Fairness of regressors and scores: bounded group loss (each group's mean loss and the worst, fairlearn's
 * BoundedGroupLoss), statistical parity and equal opportunity in distribution (the largest Kolmogorov-Smirnov distance
 * or Wasserstein distance between the groups' score distributions, overall or among the positives), and the gaps in
 * mean prediction and mean residual (calibration in the mean) between groups.
 */

import { WhiteboxError } from '../base/index.ts'
import { labelValues, sortLabels } from '../metrics/_labels.ts'
import type { MetricInput } from '../metrics/metric.ts'
import { ks2samp, wassersteinDistance } from '../stats/tests/index.ts'

/** Rows of each group, by sorted group label. */
function byGroup(name: string, groups: MetricInput, n: number): { names: string[]; rows: number[][] } {
  const g = labelValues(groups)
  if (g.length !== n) throw new WhiteboxError('shape', `${name}: one group per row`)
  const names = sortLabels(g).map(String)
  return { names, rows: names.map((label) => g.flatMap((v, i) => (String(v) === label ? [i] : []))) }
}

/** The loss of a regressor's predictions per row. */
export type RegressionLoss = 'squared' | 'absolute'

/** Each group's mean loss and the worst. */
export interface GroupLoss {
  readonly groups: readonly string[]
  readonly losses: readonly number[]
  readonly worst: { readonly group: string; readonly loss: number }
  readonly overall: number
}

/**
 * Bounded group loss (Agarwal, Dudík and Wu, 2019; fairlearn's `BoundedGroupLoss` moment): each group's mean loss and
 * the worst group's, which a fair regressor keeps below a bound.
 *
 * @param yTrue - Targets `[n]`.
 * @param yPred - Predictions `[n]`.
 * @param groups - Sensitive feature `[n]`.
 * @param options - The loss (default squared).
 * @returns Each group's loss, the worst and the overall loss.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { boundedGroupLoss } from 'whitebox-engine/fairness'
 * boundedGroupLoss([1, 2, 3, 4], [1, 2, 2, 6], ['a', 'a', 'b', 'b']).worst // => { group: 'b', loss: 2.5 }
 * ```
 */
export function boundedGroupLoss(
  yTrue: readonly number[],
  yPred: readonly number[],
  groups: MetricInput,
  options: { readonly loss?: RegressionLoss } = {},
): GroupLoss {
  if (yTrue.length !== yPred.length) throw new WhiteboxError('shape', 'boundedGroupLoss: one prediction per target')
  const { names, rows } = byGroup('boundedGroupLoss', groups, yTrue.length)
  const lossOf = (i: number): number => {
    const e = (yPred[i] ?? 0) - (yTrue[i] ?? 0)
    return options.loss === 'absolute' ? Math.abs(e) : e * e
  }
  const mean = (r: readonly number[]): number => r.reduce((s, i) => s + lossOf(i), 0) / Math.max(1, r.length)
  const losses = rows.map(mean)
  let w = 0
  for (const [k, l] of losses.entries()) if (l > (losses[w] ?? 0)) w = k
  return {
    groups: names,
    losses,
    worst: { group: names[w] ?? '', loss: losses[w] ?? 0 },
    overall: mean(yTrue.map((_, i) => i)),
  }
}

/** The distance between two groups' distributions. */
export type DistributionDistance = 'ks' | 'wasserstein'

/** The largest distance between any two groups' distributions, with every pair's. */
export interface DistributionParity {
  readonly groups: readonly string[]
  /** distances[i][j] between groups i and j. */
  readonly distances: readonly (readonly number[])[]
  readonly value: number
  readonly distance: DistributionDistance
}

function pairwise(
  values: readonly number[],
  names: readonly string[],
  rows: readonly (readonly number[])[],
  distance: DistributionDistance,
): DistributionParity {
  const k = names.length
  const d = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  let worst = 0
  for (let i = 0; i < k; i++)
    for (let j = i + 1; j < k; j++) {
      const a = (rows[i] ?? []).map((r) => values[r] ?? 0)
      const b = (rows[j] ?? []).map((r) => values[r] ?? 0)
      const v =
        a.length === 0 || b.length === 0 ? 0 : distance === 'ks' ? ks2samp(a, b).statistic : wassersteinDistance(a, b)
      ;(d[i] as number[])[j] = v
      ;(d[j] as number[])[i] = v
      worst = Math.max(worst, v)
    }
  return { groups: names, distances: d, value: worst, distance }
}

/**
 * Statistical parity in distribution (for scores and regressors): the largest distance between two groups' score
 * distributions, by the Kolmogorov-Smirnov statistic (the largest gap between their CDFs; 0 means every threshold
 * selects both groups alike) or the Wasserstein distance (how far one group's scores must move to match the other's).
 *
 * @param scores - Scores or predictions `[n]`.
 * @param groups - Sensitive feature `[n]`.
 * @param options - The distance (default `ks`).
 * @returns The largest distance and every pair's.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { distributionParity } from 'whitebox-engine/fairness'
 * distributionParity([0.1, 0.2, 0.8, 0.9], ['a', 'a', 'b', 'b']).value // => 1
 * distributionParity([0.1, 0.2, 0.8, 0.9], ['a', 'a', 'b', 'b'], { distance: 'wasserstein' }).value.toFixed(2) // => '0.70'
 * ```
 */
export function distributionParity(
  scores: readonly number[],
  groups: MetricInput,
  options: { readonly distance?: DistributionDistance } = {},
): DistributionParity {
  const { names, rows } = byGroup('distributionParity', groups, scores.length)
  return pairwise(scores, names, rows, options.distance ?? 'ks')
}

/**
 * Equal opportunity in distribution: statistical parity in distribution among the positives only (those with label
 * 1, or a target at or above `positive` for regression): the groups' qualified members get the same scores.
 *
 * @param scores - Scores or predictions `[n]`.
 * @param yTrue - Labels or targets `[n]`.
 * @param groups - Sensitive feature `[n]`.
 * @param options - The distance and the cut on the target (default 1, so labels of 1).
 * @returns The largest distance among the positives and every pair's.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { equalOpportunityInDistribution } from 'whitebox-engine/fairness'
 * equalOpportunityInDistribution([0.9, 0.2, 0.9, 0.3], [1, 0, 1, 0], ['a', 'a', 'b', 'b']).value // => 0
 * ```
 */
export function equalOpportunityInDistribution(
  scores: readonly number[],
  yTrue: readonly number[],
  groups: MetricInput,
  options: { readonly distance?: DistributionDistance; readonly positive?: number } = {},
): DistributionParity {
  if (scores.length !== yTrue.length)
    throw new WhiteboxError('shape', 'equalOpportunityInDistribution: one target per score')
  const cut = options.positive ?? 1
  const { names, rows } = byGroup('equalOpportunityInDistribution', groups, scores.length)
  return pairwise(
    scores,
    names,
    rows.map((r) => r.filter((i) => (yTrue[i] ?? 0) >= cut)),
    options.distance ?? 'ks',
  )
}

/** The gaps in mean prediction and mean residual between groups. */
export interface RegressionGaps {
  readonly groups: readonly string[]
  readonly meanPrediction: readonly number[]
  /** Each group's mean of target minus prediction: 0 when calibrated in the mean. */
  readonly meanResidual: readonly number[]
  /** The largest minus the smallest mean prediction (statistical parity in the mean). */
  readonly meanDifference: number
  /** The largest minus the smallest mean residual (the calibration gap). */
  readonly calibrationGap: number
}

/**
 * The gaps of a regressor between groups: in mean prediction (statistical parity in the mean) and in mean residual
 * (calibration in the mean: a group whose targets the model under-predicts has a positive residual).
 *
 * @param yTrue - Targets `[n]`.
 * @param yPred - Predictions `[n]`.
 * @param groups - Sensitive feature `[n]`.
 * @returns Each group's mean prediction and residual and the two gaps.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { regressionGaps } from 'whitebox-engine/fairness'
 * const r = regressionGaps([1, 3, 2, 6], [1, 3, 1, 5], ['a', 'a', 'b', 'b'])
 * r.meanDifference // => 1
 * r.calibrationGap // => 1
 * ```
 */
export function regressionGaps(
  yTrue: readonly number[],
  yPred: readonly number[],
  groups: MetricInput,
): RegressionGaps {
  if (yTrue.length !== yPred.length) throw new WhiteboxError('shape', 'regressionGaps: one prediction per target')
  const { names, rows } = byGroup('regressionGaps', groups, yTrue.length)
  const mean = (r: readonly number[], f: (i: number) => number): number =>
    r.reduce((s, i) => s + f(i), 0) / Math.max(1, r.length)
  const meanPrediction = rows.map((r) => mean(r, (i) => yPred[i] ?? 0))
  const meanResidual = rows.map((r) => mean(r, (i) => (yTrue[i] ?? 0) - (yPred[i] ?? 0)))
  const spread = (v: readonly number[]): number => Math.max(...v) - Math.min(...v)
  return {
    groups: names,
    meanPrediction,
    meanResidual,
    meanDifference: spread(meanPrediction),
    calibrationGap: spread(meanResidual),
  }
}

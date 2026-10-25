import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as M from './index.ts'

type Disparity = (y: Tensor, p: Tensor, g: Tensor, o: Record<string, unknown>) => { estimate: { value: number } }
const disparity =
  (m: unknown) =>
  ([y, p, g, w]: readonly [Tensor, Tensor, ...(Tensor | undefined)[]], o: Readonly<Record<string, unknown>>): number =>
    (m as Disparity)(y, p, g as Tensor, { ...o, ...(w === undefined ? {} : { sampleWeight: w }) }).estimate.value

// fairlearn's MetricFrame of accuracy or recall (positive class 1).
const groupScore =
  (metric: unknown) =>
  (t: unknown[], p: unknown[]): number => {
    let tp = 0
    let pos = 0
    let right = 0
    for (const [i, v] of t.entries()) {
      if (v === p[i]) right++
      if (v === 1) {
        pos++
        if (p[i] === 1) tp++
      }
    }
    return metric === 'accuracy' ? right / t.length : pos === 0 ? 0 : tp / pos
  }
const frame = ([y, p, g]: readonly (Tensor | undefined)[], o: Readonly<Record<string, unknown>>): M.MetricFrame =>
  M.metricFrame(toFlat(y as Tensor), toFlat(p as Tensor), g as Tensor, groupScore(o['metric']))

goldenSuite('group fairness metrics against fairlearn', 'fairness/group_metrics', {
  demographicParityDifference: disparity(M.demographicParityDifference),
  demographicParityRatio: disparity(M.demographicParityRatio),
  equalOpportunityDifference: disparity(M.equalOpportunityDifference),
  equalOpportunityRatio: disparity(M.equalOpportunityRatio),
  equalizedOddsDifference: disparity(M.equalizedOddsDifference),
  equalizedOddsRatio: disparity(M.equalizedOddsRatio),
  predictiveParityDifference: disparity(M.predictiveParityDifference),
  groupSelection0: ([y, p, g]) => M.groupRates(y, p, g as Tensor).groups['0']?.selectionRate ?? Number.NaN,
  groupFpr1: ([y, p, g]) => M.groupRates(y, p, g as Tensor).groups['1']?.falsePositiveRate ?? Number.NaN,
  disparateImpact: ([p, g], o) => M.disparateImpact(p, g, o).estimate.value,
  accuracyDifference: disparity(M.accuracyDifference),
  accuracyRatio: disparity(M.accuracyRatio),
  treatmentEqualityDifference: disparity(M.treatmentEqualityDifference),
  conditionalGaps: ([p, g, s]) =>
    tensor(Object.values(M.conditionalDemographicParityDifference(p, g as Tensor, s as Tensor).breakdown ?? {})),
  balanceForPositiveClass: ([y, s, g]) => M.balanceForPositiveClass(y, s, g as Tensor).estimate.value,
  balanceForNegativeClass: ([y, s, g]) => M.balanceForNegativeClass(y, s, g as Tensor).estimate.value,
  frameMin: (inputs, o) => frame(inputs, o).groupMin,
  frameMax: (inputs, o) => frame(inputs, o).groupMax,
  frameRatio: (inputs, o) => frame(inputs, o).ratio,
  frameOverall: (inputs, o) => frame(inputs, o).overall,
})

goldenSuite('minimum detectable difference against statsmodels', 'fairness/inference', {
  effectSize: (_, o) =>
    M.minimumDetectableDifference({
      n1: o['n1'] as number,
      n2: o['n2'] as number,
      alpha: o['alpha'] as number,
      power: o['power'] as number,
    }).effectSize,
})

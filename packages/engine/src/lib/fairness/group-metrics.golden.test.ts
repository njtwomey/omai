import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as M from './index.ts'

type Disparity = (y: Tensor, p: Tensor, g: Tensor, o: Record<string, unknown>) => { estimate: { value: number } }
const disparity =
  (m: unknown) =>
  ([y, p, g, w]: readonly [Tensor, Tensor, ...(Tensor | undefined)[]], o: Readonly<Record<string, unknown>>): number =>
    (m as Disparity)(y, p, g as Tensor, { ...o, ...(w === undefined ? {} : { sampleWeight: w }) }).estimate.value

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
})

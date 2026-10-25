import { describe, expect, it } from 'vitest'
import { expectedCalibrationError } from '../metrics/index.ts'
import * as M from './index.ts'

const v = (r: { readonly estimate: { readonly value: number } }): number => r.estimate.value

describe('fairness metrics', () => {
  it('measures calibration by group as the largest gap between group calibration errors', () => {
    const yTrue = [1, 0, 1, 0, 1, 1, 0, 0]
    const yProb = [0.9, 0.1, 0.4, 0.3, 0.8, 0.6, 0.2, 0.7]
    const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
    const r = M.calibrationByGroup(yTrue, yProb, groups, { nBins: 2 })
    const ece = (idx: number[]): number =>
      v(
        expectedCalibrationError(
          idx.map((i) => yTrue[i] ?? 0),
          idx.map((i) => yProb[i] ?? 0),
          { nBins: 2 },
        ),
      )
    expect(v(r)).toBeCloseTo(Math.abs(ece([0, 1, 2, 3]) - ece([4, 5, 6, 7])), 15)
  })
})

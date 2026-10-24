import { describe, expect, it } from 'vitest'
import { real } from '../space/index.ts'
import {
  createStudy,
  hyperbandPruner,
  medianPruner,
  nopPruner,
  patientPruner,
  percentilePruner,
  successiveHalvingPruner,
  thresholdPruner,
} from './index.ts'

describe('pruner settings', () => {
  it('are validated', () => {
    expect(() => percentilePruner(120)).toThrow()
    expect(() => medianPruner({ intervalSteps: 0 })).toThrow()
    expect(() => medianPruner({ nMinTrials: 0 })).toThrow()
    expect(() => medianPruner({ nStartupTrials: -1 })).toThrow()
    expect(() => thresholdPruner()).toThrow()
    expect(() => thresholdPruner({ lower: 2, upper: 1 })).toThrow()
    expect(() => patientPruner(undefined, -1)).toThrow()
    expect(() => patientPruner(undefined, 1, -1)).toThrow()
    expect(() => successiveHalvingPruner({ reductionFactor: 1 })).toThrow()
    expect(() => successiveHalvingPruner({ minResource: 0 })).toThrow()
    expect(() => successiveHalvingPruner({ bootstrapCount: 1 })).toThrow()
    expect(() => hyperbandPruner({ bootstrapCount: 1 })).toThrow()
    expect(() => hyperbandPruner({ minResource: 4, maxResource: 2 })).toThrow()
  })
  it('need a single objective', () => {
    const two = createStudy(real(0, 1), {
      directions: ['minimize', 'minimize'],
      trials: [
        { number: 0, params: 0, state: 'complete', values: [1, 1], intermediate: [{ step: 0, value: 1 }], source: 'x' },
      ],
    })
    const trial = { number: 1, params: 0, intermediate: [{ step: 0, value: 2 }], rungValues: [] }
    expect(() => medianPruner({ nStartupTrials: 0 }).prune(two, trial)).toThrow(/single objective/)
    expect(nopPruner().prune(two, trial).prune).toBe(false)
  })
})

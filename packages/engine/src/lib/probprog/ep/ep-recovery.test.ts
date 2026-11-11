import { describe, expect, it } from 'vitest'
import { activeSetScore, mixtureInterval, recoverySummary } from './ep-recovery.ts'

describe('mixtureInterval', () => {
  it('is the normal interval for one component and sits between the humps of a mixture', () => {
    const [lo, hi] = mixtureInterval([{ weight: 1, mean: 2, sd: 3 }], 0.95)
    expect(lo).toBeCloseTo(2 - 1.959964 * 3, 4)
    expect(hi).toBeCloseTo(2 + 1.959964 * 3, 4)
    const [a, b] = mixtureInterval([
      { weight: 0.5, mean: -5, sd: 0.1 },
      { weight: 0.5, mean: 5, sd: 0.1 },
    ])
    expect(a).toBeCloseTo(-5.16, 1)
    expect(b).toBeCloseTo(5.16, 1)
  })
})

describe('recoverySummary', () => {
  it('counts coverage and errors, and compares with a reference', () => {
    const s = recoverySummary({
      truth: [0, 1, 2],
      mean: [0.1, 1, 3],
      sd: [0.1, 0.1, 0.1],
      lower: [-0.1, 0.8, 2.8],
      upper: [0.3, 1.2, 3.2],
      reference: { mean: [0.1, 1, 3], sd: [0.1, 0.1, 0.2] },
    })
    expect(s.coverage).toBeCloseTo(2 / 3)
    expect(s.maxZ).toBeCloseTo(10)
    expect(s.reference?.maxMeanZ).toBe(0)
    expect(s.reference?.sdRatio).toEqual([0.5, 1])
  })
})

describe('activeSetScore', () => {
  it('ranks, and counts at the threshold', () => {
    expect(activeSetScore([0.9, 0.6, 0.1], [true, false, false])).toEqual({
      auc: 1,
      found: 1,
      missed: 0,
      falsePositives: 1,
    })
    expect(activeSetScore([0.4, 0.6], [true, false]).auc).toBe(0)
  })
})

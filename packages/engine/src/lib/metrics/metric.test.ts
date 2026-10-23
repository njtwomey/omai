import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

describe('metric infrastructure', () => {
  it('streams batches to the same result as one call', () => {
    const s = M.streaming(M.f1Score)
    s.update([0, 1, 2, 2], [0, 2, 2, 2])
    s.update([1, 0], [1, 1], [2, 1])
    expect(s.count).toBe(6)
    const whole = M.f1Score([0, 1, 2, 2, 1, 0], [0, 2, 2, 2, 1, 1], {
      average: 'macro',
      sampleWeight: [1, 1, 1, 1, 2, 1],
    })
    expect(s.compute({ average: 'macro' }).estimate.value).toBe(whole.estimate.value)
    s.reset()
    expect(() => s.compute()).toThrow()
    const strings = M.streaming(M.accuracyScore)
    strings.update(['a', 'b'], ['a', 'a'])
    strings.update(['c'], ['c'])
    expect(strings.compute().estimate.value).toBeCloseTo(2 / 3, 15)
  })

  it('selects rows of numeric and string inputs', () => {
    expect(
      M.selectRows(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        [2, 0],
      ),
    ).toEqual([
      [5, 6],
      [1, 2],
    ])
    expect(M.selectRows(['a', 'b', 'c'], [1, 1])).toEqual(['b', 'b'])
    expect(
      M.rowCount([
        [1, 2],
        [3, 4],
      ]),
    ).toBe(2)
  })

  it('computes sklearn weighted percentiles', () => {
    expect(M.weightedPercentile([3, 1, 2], undefined, 50)).toBe(2)
    expect(M.weightedPercentile([1, 2, 3, 4], [1, 1, 1, 1], 50, false)).toBe(2)
    expect(M.weightedPercentile([1, 2, 3], [0, 0, 0], 50)).toBeNaN()
  })

  it('rejects binary averaging of multi-class targets and mismatched weights', () => {
    expect(() => M.precisionScore([0, 1, 2], [0, 1, 2])).toThrow()
    expect(() => M.precisionScore([0, 1], [0, 1], { posLabel: 7 })).toThrow()
    expect(() => M.accuracyScore([0, 1], [0, 1], { sampleWeight: [1] })).toThrow()
  })
})

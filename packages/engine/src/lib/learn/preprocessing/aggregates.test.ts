import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toArray, toFlat } from '../../tensor/index.ts'
import { groupAggregates, lagFeatures, rollingFeatures } from './aggregates.ts'

describe('groupAggregates', () => {
  it('gives each row its group statistic, per column, value-major', () => {
    const r = toArray(
      groupAggregates(
        [5, 5, 7, 5],
        [
          [1, 10],
          [3, 20],
          [8, 30],
          [5, 40],
        ],
        ['mean', 'max'],
      ),
    ) as number[][]
    expect(r[0]).toEqual([3, 5, 23.333333333333332, 40])
    expect(r[2]).toEqual([8, 8, 30, 30])
    expect(r[1]).toEqual(r[3])
  })

  it('supports sum, min, median, count, std (n - 1) and leaving the row out', () => {
    const all = toArray(
      groupAggregates([0, 0, 0, 0], [[1], [2], [4], [9]], ['sum', 'min', 'median', 'count', 'std']),
    ) as number[][]
    expect(all[0]?.slice(0, 4)).toEqual([16, 1, 3, 4])
    expect(all[0]?.[4]).toBeCloseTo(Math.sqrt(((1 - 4) ** 2 + (2 - 4) ** 2 + 0 + (9 - 4) ** 2) / 3), 12)
    const loo = toArray(groupAggregates([0, 0, 0], [[1], [2], [6]], ['mean'], { excludeSelf: true })) as number[][]
    expect(loo.map((r) => r[0])).toEqual([4, 3.5, 1.5])
    const alone = toArray(groupAggregates([0, 1], [[1], [2]], ['mean', 'std'], { excludeSelf: true })) as number[][]
    expect(alone[0]?.every(Number.isNaN)).toBe(true)
    expect(() => groupAggregates([0], [[1], [2]], ['mean'])).toThrow()
  })
})

describe('rollingFeatures', () => {
  it('matches a direct computation on every window, with pandas semantics for the start', () => {
    const z = Array.from(toFlat(normal(key(1), [30])))
    const stats = ['mean', 'sum', 'min', 'max', 'std', 'median'] as const
    const r = toArray(rollingFeatures(z, 5, stats)) as number[][]
    for (let i = 0; i < 30; i++) {
      if (i < 4) {
        expect(r[i]?.every(Number.isNaN)).toBe(true)
        continue
      }
      const w = z.slice(i - 4, i + 1)
      const mean = w.reduce((a, b) => a + b, 0) / 5
      expect(r[i]?.[0]).toBeCloseTo(mean, 12)
      expect(r[i]?.[1]).toBeCloseTo(mean * 5, 12)
      expect(r[i]?.[2]).toBe(Math.min(...w))
      expect(r[i]?.[3]).toBe(Math.max(...w))
      expect(r[i]?.[4]).toBeCloseTo(Math.sqrt(w.reduce((a, b) => a + (b - mean) ** 2, 0) / 4), 12)
      expect(r[i]?.[5]).toBe([...w].sort((a, b) => a - b)[2])
    }
  })

  it('honours minPeriods, centre and NaN entries, and handles several series', () => {
    expect(toArray(rollingFeatures([1, 2, 4, 8], 2, ['mean']))).toEqual([[Number.NaN], [1.5], [3], [6]])
    expect(toArray(rollingFeatures([1, 2, 4, 8], 3, ['sum'], { minPeriods: 1 }))).toEqual([[1], [3], [7], [14]])
    expect(toArray(rollingFeatures([1, 2, 4, 8, 16], 3, ['mean'], { center: true, minPeriods: 2 }))).toEqual([
      [1.5],
      [7 / 3],
      [14 / 3],
      [28 / 3],
      [12],
    ])
    expect(toArray(rollingFeatures([1, Number.NaN, 3, 5], 2, ['sum'], { minPeriods: 1 }))).toEqual([[1], [1], [3], [8]])
    const two = toArray(
      rollingFeatures(
        [
          [1, 10],
          [2, 20],
          [3, 30],
        ],
        2,
        ['sum', 'max'],
      ),
    ) as number[][]
    expect(two[2]).toEqual([5, 3, 50, 30])
    expect(() => rollingFeatures([1, 2], 0, ['mean'])).toThrow()
  })
})

describe('lagFeatures', () => {
  it('shifts the series, NaN where there is no history, and looks ahead for a negative lag', () => {
    expect(toArray(lagFeatures([1, 2, 3], [1, 2]))).toEqual([
      [Number.NaN, Number.NaN],
      [1, Number.NaN],
      [2, 1],
    ])
    expect(toArray(lagFeatures([1, 2, 3], [-1, 0]))).toEqual([
      [2, 1],
      [3, 2],
      [Number.NaN, 3],
    ])
  })
})

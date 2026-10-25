import { describe, expect, it } from 'vitest'
import { repairFeatures, reweighingWeights } from './preprocessing.ts'

describe('preprocessing', () => {
  it('reweighing makes the weighted label rate equal across groups and keeps the total weight', () => {
    const labels = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    const groups = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    const { weights, cells } = reweighingWeights(labels, groups)
    const rate = (g: number): number => {
      let num = 0
      let den = 0
      for (const [i, v] of groups.entries())
        if (v === g) {
          den += weights[i] ?? 0
          num += (weights[i] ?? 0) * (labels[i] ?? 0)
        }
      return num / den
    }
    expect(rate(0)).toBeCloseTo(rate(1), 12)
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(10, 12)
    expect(cells).toHaveLength(4)
    expect(() => reweighingWeights([1], [0, 1])).toThrow(/one group/)
  })

  it('full repair gives the groups the same distribution; no repair changes nothing', () => {
    const x = [0, 5, 1, 5, 2, 5, 10, 5, 11, 5, 12, 5]
    const groups = [0, 0, 0, 1, 1, 1]
    const full = repairFeatures(x, 2, groups, { skip: [1] })
    expect(Array.from(full)).toEqual([5, 5, 6, 5, 7, 5, 5, 5, 6, 5, 7, 5])
    expect(Array.from(repairFeatures(x, 2, groups, { level: 0 }))).toEqual(x)
    const half = repairFeatures(x, 2, groups, { level: 0.5, skip: [1] })
    expect(half[0]).toBeCloseTo(2.5, 12)
    expect(() => repairFeatures(x, 2, groups, { level: 2 })).toThrow(/level/)
    expect(() => repairFeatures(x, 3, groups)).toThrow(/values/)
  })
})

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  fairRepresentationMemberships,
  learnFairRepresentation,
  massageLabels,
  optimisedLabelMapping,
  preferentialSample,
  repairFeatures,
  reweighingWeights,
} from './preprocessing.ts'

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

describe('massaging, preferential sampling, label mapping and fair representations', () => {
  it('massaging equalises the label rates as nearly as whole swaps allow, swapping the least certain rows', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.boolean(), fc.double({ min: 0, max: 1, noNaN: true })), {
          minLength: 4,
          maxLength: 60,
        }),
        (rows) => {
          const labels = rows.map(([y]) => (y ? 1 : 0))
          const groups = rows.map(([, g]) => (g ? 1 : 0))
          const scores = rows.map(([, , s]) => s)
          const m = massageLabels(labels, scores, groups)
          const rate = (ls: number[], g: number): number => {
            const idx = groups.flatMap((v, i) => (v === g ? [i] : []))
            return idx.length === 0 ? 0 : idx.reduce((s, i) => s + (ls[i] ?? 0), 0) / idx.length
          }
          const gap = (ls: number[]): number => rate(ls, 0) - rate(ls, 1)
          return m.promoted.length === m.demoted.length && Math.abs(gap(m.labels)) <= Math.abs(gap(labels)) + 1e-12
        },
      ),
      { numRuns: 100 },
    )
  })

  it('preferential sampling makes label and group independent in the sample', () => {
    const labels = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    const groups = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    const scores = labels.map((y, i) => y * 0.5 + i * 0.01)
    const s = preferentialSample(labels, scores, groups)
    const rate = (g: number): number => {
      const rows = s.rows.filter((i) => groups[i] === g)
      return rows.filter((i) => labels[i] === 1).length / rows.length
    }
    expect(rate(0)).toBeCloseTo(rate(1), 12)
    expect(s.cells.map((c) => c.expected)).toEqual([4, 3, 4, 3])
  })

  it('the label mapping meets epsilon and costs nothing when the rates already agree', () => {
    const m = optimisedLabelMapping([1, 0, 1, 0], [0, 0, 1, 1], { epsilon: 0 })
    expect(m.distortion).toBeCloseTo(0, 9)
    const loose = optimisedLabelMapping([1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], { epsilon: 0.1 })
    expect(Math.abs((loose.rates[0] ?? 0) - (loose.rates[1] ?? 0))).toBeLessThanOrEqual(0.2 + 1e-9)
    expect(loose.distortion).toBeCloseTo(0.15, 9)
    expect(() =>
      optimisedLabelMapping([1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], { epsilon: 0, maxFlip: 0.1 }),
    ).toThrow(/constraints/)
  })

  it('fair representations trade parity of memberships for prediction as the parity weight grows', () => {
    const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), (i % 2) * 2 + Math.cos(i)])
    const labels = x.map((r) => ((r[1] ?? 0) > 1 ? 1 : 0))
    const groups = x.map((_, i) => i % 2)
    const free = learnFairRepresentation(x, labels, groups, { prototypes: 3, weights: { parity: 0 } })
    const fair = learnFairRepresentation(x, labels, groups, { prototypes: 3, weights: { parity: 100 } })
    expect(fair.losses.parity).toBeLessThan(free.losses.parity)
    const m = fairRepresentationMemberships(fair, x)
    expect(m.every((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-9)).toBe(true)
  })
})

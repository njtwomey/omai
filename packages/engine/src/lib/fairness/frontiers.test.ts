import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { biasedCredit } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { impossibilityAt, paretoFrontier, selectFairModel, thresholdFrontier } from './frontiers.ts'

describe('frontiers and selection', () => {
  it('no point off the front beats a front point in both coordinates (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.double({ min: 0, max: 1, noNaN: true }), fc.double({ min: 0, max: 1, noNaN: true })), {
          minLength: 1,
          maxLength: 30,
        }),
        (pairs) => {
          const points = pairs.map(([accuracy, violation]) => ({ accuracy, violation }))
          const front = paretoFrontier(points)
          const dominated = (i: number): boolean =>
            points.some(
              (q, j) =>
                j !== i &&
                q.accuracy >= (points[i]?.accuracy ?? 0) &&
                q.violation <= (points[i]?.violation ?? 0) &&
                (q.accuracy > (points[i]?.accuracy ?? 0) || q.violation < (points[i]?.violation ?? 0)),
            )
          return front.length > 0 && front.every((i) => !dominated(i))
        },
      ),
      { numRuns: 200 },
    )
  })

  it('selects within a budget, falls back to the least violating, or prices the violation', () => {
    const p = [
      { accuracy: 0.9, violation: 0.3 },
      { accuracy: 0.8, violation: 0.1 },
      { accuracy: 0.7, violation: 0.05 },
    ]
    expect(selectFairModel(p, { maxViolation: 0.12 })).toBe(1)
    expect(selectFairModel(p, { maxViolation: 0.01 })).toBe(2)
    expect(selectFairModel(p, { price: 2 })).toBe(1)
    expect(() => selectFairModel([])).toThrow(/candidates/)
  })

  it('moving one threshold traces a frontier, and calibrated scores cannot also equalise odds when base rates differ', () => {
    const { data, truth } = generate(biasedCredit, {
      key: key(2),
      knobs: { n: 3000, baseRateGap: 1, labelBias: 0, minorityNoise: 1, proxyStrength: 0 },
    })
    const p = toFlat(truth?.posterior?.(data.x) as Tensor)
    const scores = Array.from({ length: 3000 }, (_, i) => p[2 * i + 1] ?? 0)
    const flat = toFlat(data.x)
    const groups = Array.from({ length: 3000 }, (_, i) => flat[4 * i + 3] ?? 0)
    const labels = Array.from(toFlat(data.y))
    const f = thresholdFrontier(scores, labels, groups, { count: 11 })
    expect(f.group).toBe(1)
    expect(f.front.length).toBeGreaterThan(1)
    const r = impossibilityAt(scores, labels, groups)
    expect(Math.abs((r.baseRates[0] ?? 0) - (r.baseRates[1] ?? 0))).toBeGreaterThan(0.2)
    expect(r.equalisedOdds).toBeGreaterThan(0.1)
    expect(r.equalisedOddsAfter).toBeLessThan(r.equalisedOdds)
    expect(r.predictiveParityAfter).toBeGreaterThan(0.05)
  })
})

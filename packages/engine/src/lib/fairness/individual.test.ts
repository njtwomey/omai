import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { consistency, fairnessThroughAwareness, lipschitzFairness } from './individual.ts'

describe('individual fairness', () => {
  it('consistency is 1 for decisions constant on neighbourhoods and in [0, 1] for any decisions (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.double({ min: -5, max: 5, noNaN: true }), fc.boolean()), { minLength: 3, maxLength: 30 }),
        (rows) => {
          const x = rows.map(([v]) => [v])
          const d = rows.map(([, b]) => (b ? 1 : 0))
          const c = consistency(x, d, { k: 3 })
          return (
            c >= -1e-12 &&
            c <= 1 + 1e-12 &&
            consistency(
              x,
              x.map(() => 1),
              { k: 3 },
            ) === 1
          )
        },
      ),
      { numRuns: 100 },
    )
    expect(() => consistency([[0]], [1, 0])).toThrow(/one decision/)
  })

  it('Lipschitz checks every pair, or a keyed sample of them', () => {
    const x = Array.from({ length: 30 }, (_, i) => [i / 10])
    const smooth = lipschitzFairness(
      x,
      x.map((r) => (r[0] ?? 0) / 2),
    )
    expect(smooth.violationRate).toBe(0)
    expect(smooth.worstRatio).toBeCloseTo(0.5, 12)
    expect(smooth.pairs).toHaveLength(435)
    expect(
      lipschitzFairness(
        x,
        x.map((r) => r[0] ?? 0),
        { pairs: 50 },
      ).pairs,
    ).toHaveLength(50)
  })

  it('fairness through awareness is Lipschitz in the metric and decides alone rows by their scores', () => {
    const x = [[0], [0.1], [0.2], [5]]
    const p = fairnessThroughAwareness(x, [0.9, 0.1, 0.9, 0.2])
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        expect(Math.abs((p[i] ?? 0) - (p[j] ?? 0))).toBeLessThanOrEqual(
          Math.abs((x[i]?.[0] ?? 0) - (x[j]?.[0] ?? 0)) + 1e-9,
        )
    expect(p[3]).toBeCloseTo(0, 9)
  })
})

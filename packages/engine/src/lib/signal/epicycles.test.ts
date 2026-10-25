import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { chainOrder, curveSpectrum, epicycleJoints, keptTerms, partialSum } from './epicycles.ts'

const curve = (n: number, f: (t: number) => [number, number]): [number, number][] =>
  Array.from({ length: n }, (_, i) => f((2 * Math.PI * i) / n))

describe('epicycles', () => {
  it('recovers a curve exactly from all its terms, in any chain order', () => {
    const pts = curve(16, (t) => [Math.cos(t) + 0.3 * Math.cos(3 * t), Math.sin(t) - 0.2 * Math.sin(-2 * t)])
    const s = curveSpectrum(pts)
    const all = keptTerms(s, { count: 15 })
    for (const order of ['magnitude', 'speed', 'frequency', 'random'] as const) {
      const chain = chainOrder(s, all, order, key(0))
      for (const [i, p] of pts.entries()) {
        const j = epicycleJoints(s, chain, i / 16)
        expect(j.x.at(-1)).toBeCloseTo(p[0], 10)
        expect(j.y.at(-1)).toBeCloseTo(p[1], 10)
      }
    }
  })

  it('keeps energy as asked (Parseval), and the partial sum matches the chain', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.double({ min: -1, max: 1, noNaN: true }), fc.double({ min: -1, max: 1, noNaN: true })), {
          minLength: 8,
          maxLength: 8,
        }),
        (raw) => {
          const s = curveSpectrum(raw as [number, number][])
          const mean = raw.reduce((a, p) => [a[0] + p[0] / 8, a[1] + p[1] / 8], [0, 0])
          const spread = raw.reduce((a, p) => a + (p[0] - mean[0]) ** 2 + (p[1] - mean[1]) ** 2, 0) / 8
          expect(s.energy).toBeCloseTo(spread, 9)
          const kept = keptTerms(s, { energy: 0.6 })
          const got = kept.reduce((a, b) => a + (s.magnitude[b] ?? 0) ** 2, 0)
          if (s.energy > 1e-9) expect(got).toBeGreaterThanOrEqual(0.6 * s.energy - 1e-12)
          const sum = partialSum(s, kept, 16)
          const j = epicycleJoints(s, kept, 3 / 16)
          expect(sum.x[3]).toBeCloseTo(j.x.at(-1) ?? 0, 9)
          expect(sum.y[3]).toBeCloseTo(j.y.at(-1) ?? 0, 9)
        },
      ),
      { numRuns: 50 },
    )
  })

  it('rejects a partial sum too short for its frequencies', () => {
    const s = curveSpectrum(curve(16, (t) => [Math.cos(5 * t), Math.sin(5 * t)]))
    expect(() => partialSum(s, keptTerms(s, { count: 1 }), 8)).toThrow()
  })
})

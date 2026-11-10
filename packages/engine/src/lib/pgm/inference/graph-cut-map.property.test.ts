import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { randomPairwise } from '../../../testing/pairwise-models.ts'
import * as P from '../index.ts'

const shape = fc.oneof(
  fc.integer({ min: 2, max: 10 }).map((size) => ({ shape: 'chain' as const, size })),
  fc
    .tuple(fc.integer({ min: 1, max: 3 }), fc.integer({ min: 2, max: 4 }))
    .map(([size, cols]) => ({ shape: 'grid' as const, size, cols })),
  fc.integer({ min: 2, max: 9 }).map((size) => ({ shape: 'random' as const, size })),
)

describe('graph-cut MAP (properties)', () => {
  it('equals the exact MAP on random submodular binary models', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 100000 }), shape, (seed, s) => {
        const model = randomPairwise({ ...s, seed, kind: 'submodular', unaryScale: 2 })
        const r = P.graphCutMap(model)
        expect(r.logValue).toBeCloseTo(P.maxProductElimination(model).logValue, 8)
        expect(r.exact).toBe(true)
      }),
      { numRuns: 60 },
    )
  })

  it('roof duality bounds the optimum and the completed labelling is no better than it', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 100000 }), fc.integer({ min: 2, max: 9 }), (seed, size) => {
        const model = randomPairwise({ shape: 'random', size, seed, kind: 'any', unaryScale: 1.5 })
        const r = P.graphCutMap(model, { nonSubmodular: 'qpbo' })
        const optimum = P.maxProductElimination(model).logValue
        expect(r.lowerBound).toBeLessThanOrEqual(-optimum + 1e-9)
        expect(r.logValue).toBeLessThanOrEqual(optimum + 1e-9)
        if (r.exact) expect(r.logValue).toBeCloseTo(optimum, 8)
      }),
      { numRuns: 60 },
    )
  })

  it('alpha-expansion on Potts models lowers the energy to a local minimum within twice the optimum', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 100000 }), fc.integer({ min: 2, max: 4 }), (seed, labels) => {
        const model = randomPairwise({ shape: 'grid', size: 2, cols: 3, seed, labels, kind: 'potts' })
        const optimum = -P.maxProductElimination(model).logValue
        const r = P.alphaExpansion(model)
        expect(r.converged).toBe(true)
        expect(r.energy).toBeGreaterThanOrEqual(optimum - 1e-9)
        expect(r.energy).toBeLessThanOrEqual(2 * optimum + 1e-9)
        for (let i = 1; i < r.energyTrace.length; i++)
          expect(r.energyTrace[i] ?? 0).toBeLessThan((r.energyTrace[i - 1] ?? 0) + 1e-9)
      }),
      { numRuns: 40 },
    )
  })
})

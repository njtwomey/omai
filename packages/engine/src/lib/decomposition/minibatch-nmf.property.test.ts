import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import { type BetaLoss, betaDivergence, miniBatchNmfAlgorithm } from './minibatch-nmf.ts'

const problem = fc
  .tuple(
    fc.integer({ min: 6, max: 20 }),
    fc.integer({ min: 2, max: 5 }),
    fc.integer({ min: 1, max: 3 }),
    fc.integer({ min: 0, max: 1000 }),
  )
  .map(([n, d, k, seed]) => {
    const f = (i: number, j: number, s: number): number => 0.2 + Math.abs(Math.sin(seed + i * 7.3 + j * 2.9 + s))
    return {
      x: Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => f(i, j, 0))),
      w: Array.from({ length: n }, (_, i) => Array.from({ length: k }, (_, c) => f(i, c, 1))),
      h: Array.from({ length: k }, (_, c) => Array.from({ length: d }, (_, j) => f(c, j, 2))),
    }
  })

describe('mini-batch NMF properties', () => {
  it('factors stay non-negative and finite, every beta, every batch size', () => {
    fc.assert(
      fc.property(
        problem,
        fc.constantFrom<BetaLoss>('frobenius', 'kullback-leibler', 'itakura-saito', 1.5),
        fc.integer({ min: 1, max: 20 }),
        (p, betaLoss, batchSize) => {
          const out = runAlgorithm(
            miniBatchNmfAlgorithm(p.x, p.w, p.h, { betaLoss, batchSize, tol: 0, maxNoImprovement: null }),
            { maxSteps: 12 },
          ).state
          for (const row of [...out.w, ...out.h])
            for (const v of row) {
              expect(Number.isFinite(v)).toBe(true)
              expect(v).toBeGreaterThanOrEqual(0)
            }
          expect(out.step).toBe(12)
        },
      ),
      { numRuns: 30 },
    )
  })

  it('the divergence is non-negative and a full pass of multiplicative updates lowers the Frobenius cost', () => {
    fc.assert(
      fc.property(problem, (p) => {
        const before = betaDivergence(p.x, p.w, p.h, 'frobenius')
        expect(before).toBeGreaterThanOrEqual(0)
        const n = p.x.length
        const out = runAlgorithm(
          miniBatchNmfAlgorithm(p.x, p.w, p.h, { batchSize: n, tol: 0, maxNoImprovement: null, forgetFactor: 0 }),
          { maxSteps: 6 },
        ).state
        expect(betaDivergence(p.x, out.w, out.h, 'frobenius')).toBeLessThanOrEqual(before + 1e-9)
      }),
      { numRuns: 30 },
    )
  })
})

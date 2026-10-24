import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const pvals = fc.array(
  fc.integer({ min: 1, max: 1000 }).map((v) => v / 1000),
  { minLength: 1, maxLength: 20 },
)
const METHODS: readonly T.CorrectionMethod[] = [
  'bonferroni',
  'sidak',
  'holm',
  'holm-sidak',
  'hochberg',
  'hommel',
  'benjamini-hochberg',
  'benjamini-yekutieli',
]

describe('multiple testing', () => {
  it('adjusted p-values are at least the raw ones, at most one, and preserve the order of the raw ones', () => {
    fc.assert(
      fc.property(pvals, fc.constantFrom(...METHODS), (p, method) => {
        const q = toFlat(T.adjustPValues(p, method))
        q.forEach((v, i) => {
          expect(v).toBeGreaterThanOrEqual((p[i] ?? 0) - 1e-15)
          expect(v).toBeLessThanOrEqual(1)
        })
        for (let i = 0; i < p.length; i++)
          for (let j = 0; j < p.length; j++)
            if ((p[i] ?? 0) < (p[j] ?? 0)) expect(q[i] ?? 0).toBeLessThanOrEqual((q[j] ?? 0) + 1e-15)
      }),
    )
  })

  it('orders the procedures by power: Bonferroni <= Holm <= Hochberg <= Hommel, and BY <= BH', () => {
    fc.assert(
      fc.property(pvals, (p) => {
        const r = (m: T.CorrectionMethod): number[] => toFlat(T.adjustPValues(p, m))
        const bonf = r('bonferroni')
        const holm = r('holm')
        const hoch = r('hochberg')
        const homm = r('hommel')
        const bh = r('benjamini-hochberg')
        const by = r('benjamini-yekutieli')
        p.forEach((_, i) => {
          expect(holm[i] ?? 0).toBeLessThanOrEqual((bonf[i] ?? 0) + 1e-12)
          expect(hoch[i] ?? 0).toBeLessThanOrEqual((holm[i] ?? 0) + 1e-12)
          expect(homm[i] ?? 0).toBeLessThanOrEqual((hoch[i] ?? 0) + 1e-12)
          expect(bh[i] ?? 0).toBeLessThanOrEqual((by[i] ?? 0) + 1e-12)
        })
      }),
    )
  })

  it('combined p-values of null p-values are uniform', () => {
    for (const method of ['fisher', 'stouffer', 'tippett', 'pearson', 'mudholkar-george'] as const)
      expectUniformPValues(nullPValues(300, 1, (k) => T.combinePValues(toFlat(uniform(k, [5])), { method }).pValue))
    expect(T.multipleTests([0.001, 0.5]).alphaBonferroni).toBe(0.025)
  })
})

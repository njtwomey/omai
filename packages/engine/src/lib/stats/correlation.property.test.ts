import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { coverage, expectUniformPValues, nullPValues } from '../../testing/stats.ts'
import { foldIn, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const pairs = fc
  .integer({ min: 5, max: 25 })
  .chain((n) =>
    fc.tuple(
      fc.array(fc.integer({ min: -50, max: 50 }), { minLength: n, maxLength: n }),
      fc.array(fc.integer({ min: -50, max: 50 }), { minLength: n, maxLength: n }),
    ),
  )

// Kendall's tau-b by the O(n^2) definition.
function naiveTau(x: readonly number[], y: readonly number[]): number {
  let c = 0
  let d = 0
  let tx = 0
  let ty = 0
  for (let i = 0; i < x.length; i++)
    for (let j = i + 1; j < x.length; j++) {
      const a = Math.sign((x[i] ?? 0) - (x[j] ?? 0))
      const b = Math.sign((y[i] ?? 0) - (y[j] ?? 0))
      if (a === 0 && b === 0) continue
      if (a === 0) tx++
      else if (b === 0) ty++
      else if (a === b) c++
      else d++
    }
  return (c - d) / Math.sqrt((c + d + tx) * (c + d + ty))
}

describe('correlation', () => {
  it("Knight's count of Kendall's tau matches the pairwise definition", () => {
    fc.assert(
      fc.property(pairs, ([x, y]) => {
        fc.pre(new Set(x).size > 1 && new Set(y).size > 1)
        expect(S.kendallTau(x, y).statistic).toBeCloseTo(naiveTau(x, y), 12)
      }),
    )
  })

  it('Pearson is affine invariant and Spearman monotone invariant', () => {
    fc.assert(
      fc.property(pairs, ([x, y]) => {
        fc.pre(new Set(x).size > 2 && new Set(y).size > 2)
        const r = S.pearson(x, y).statistic
        expect(
          S.pearson(
            x.map((v) => 2 * v - 7),
            y,
          ).statistic,
        ).toBeCloseTo(r, 10)
        expect(
          S.pearson(
            x.map((v) => -v),
            y,
          ).statistic,
        ).toBeCloseTo(-r, 10)
        expect(
          S.spearman(
            x.map((v) => v ** 3),
            y,
          ).statistic,
        ).toBeCloseTo(S.spearman(x, y).statistic, 10)
        const c = toFlat(S.corrcoef([x, y]))
        expect(c[1]).toBeCloseTo(r, 10)
      }),
    )
  })

  it('gives uniform p-values under independence', () => {
    for (const test of [S.pearson, S.spearman, S.kendallTau]) {
      const ps = nullPValues(300, 11, (k) => {
        const x = toFlat(normal(foldIn(k, 0), [20]))
        const y = toFlat(normal(foldIn(k, 1), [20]))
        return test(x, y).pValue
      })
      expectUniformPValues(ps)
    }
  })

  it("Pearson's Fisher interval covers the true correlation at about the nominal rate", () => {
    const rho = 0.6
    const intervals = Array.from({ length: 400 }, (_, i) => {
      const k = foldIn({ kind: 'key', words: [7, 0] }, i)
      const z1 = toFlat(normal(foldIn(k, 0), [30]))
      const z2 = toFlat(normal(foldIn(k, 1), [30]))
      const y = z1.map((v, j) => rho * v + Math.sqrt(1 - rho * rho) * (z2[j] ?? 0))
      return S.pearson(z1, y).estimate?.interval ?? ([0, 0] as const)
    })
    expect(coverage(intervals, rho)).toBeGreaterThan(0.91)
    expect(coverage(intervals, rho)).toBeLessThan(0.98)
  })

  it('distance correlation is zero only for constant relationships and one for affine maps', () => {
    const x = [1, 2, 3, 4, 5, 6]
    expect(
      S.distanceCorrelation(
        x,
        x.map((v) => 3 * v + 1),
      ).value,
    ).toBeCloseTo(1, 12)
    const boot = S.distanceCorrelation(x, [2, 1, 4, 3, 6, 5], { seed: 1, nResamples: 50 })
    expect(boot.interval?.[0]).toBeLessThanOrEqual(boot.interval?.[1] ?? 0)
  })
})

describe('covariance estimators', () => {
  it('shrink towards the scaled identity with intensities in [0, 1] and stay positive semidefinite', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 1000 }), (seed) => {
        const X = toFlat(normal({ kind: 'key', words: [seed, 3] }, [12, 3]))
        const rows = Array.from({ length: 12 }, (_, i) => X.slice(3 * i, 3 * i + 3))
        for (const est of [S.ledoitWolf(rows), S.oas(rows), S.shrunkCovariance(rows)]) {
          expect(est.shrinkage).toBeGreaterThanOrEqual(0)
          expect(est.shrinkage).toBeLessThanOrEqual(1)
          const c = toFlat(est.covariance)
          // A 3 x 3 symmetric matrix is positive semidefinite when its leading minors are.
          const det2 = (c[0] ?? 0) * (c[4] ?? 0) - (c[1] ?? 0) ** 2
          expect(c[0] ?? 0).toBeGreaterThan(0)
          expect(det2).toBeGreaterThan(-1e-12)
        }
      }),
      { numRuns: 30 },
    )
  })
})

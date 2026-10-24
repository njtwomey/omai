import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { foldIn, type key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number): number[] => toFlat(normal(foldIn(k, i), [n]))

describe('rank tests', () => {
  it('are valid under the null', () => {
    expectUniformPValues(nullPValues(300, 1, (k) => T.mannWhitneyU(draw(k, 0, 12), draw(k, 1, 15)).pValue))
    expectUniformPValues(
      nullPValues(300, 2, (k) => T.mannWhitneyU(draw(k, 0, 5), draw(k, 1, 6)).pValue),
      { discrete: true },
    )
    expectUniformPValues(nullPValues(300, 3, (k) => T.wilcoxon(draw(k, 0, 20)).pValue))
    expectUniformPValues(nullPValues(300, 4, (k) => T.kruskal([draw(k, 0, 8), draw(k, 1, 8), draw(k, 2, 8)]).pValue))
    expectUniformPValues(
      nullPValues(200, 5, (k) => T.friedman(Array.from({ length: 10 }, (_, i) => draw(k, i, 4))).pValue),
    )
    expectUniformPValues(nullPValues(300, 6, (k) => T.brunnerMunzel(draw(k, 0, 12), draw(k, 1, 15)).pValue))
    expectUniformPValues(
      nullPValues(300, 7, (k) => T.signTest(draw(k, 0, 25)).pValue),
      { discrete: true },
    )
  })

  it('the exact Mann-Whitney distribution is symmetric and sums to one', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 7 }), fc.integer({ min: 1, max: 7 }), (m, n) => {
        const x = Array.from({ length: m }, (_, i) => i)
        const y = Array.from({ length: n }, (_, i) => m + i)
        // Complete separation: U = 0, and the one-sided p-value is 1 / C(m + n, m).
        let c = 1
        for (let i = 1; i <= m; i++) c = (c * (n + i)) / i
        expect(T.mannWhitneyU(x, y, { alternative: 'less' }).pValue).toBeCloseTo(1 / c, 12)
        expect(T.mannWhitneyU(y, x, { alternative: 'greater' }).pValue).toBeCloseTo(1 / c, 12)
      }),
    )
  })

  it('Nemenyi p-values are symmetric with ones on the diagonal', () => {
    const p = toFlat(
      T.nemenyi([
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [7, 8, 9, 10],
      ]).pValue,
    )
    expect(p[0]).toBe(1)
    expect(p[1]).toBeCloseTo(p[3] ?? 0, 12)
    expect(p[2]).toBeLessThan(p[1] ?? 0)
  })
})

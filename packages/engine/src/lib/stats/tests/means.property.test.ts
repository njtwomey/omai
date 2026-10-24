import { describe, expect, it } from 'vitest'
import { coverage, expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { formatResult } from '../../base/index.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number, shift = 0, scale = 1): number[] =>
  toFlat(normal(foldIn(k, i), [n])).map((v) => shift + scale * v)

describe('tests of means', () => {
  it('give uniform p-values under the null', () => {
    expectUniformPValues(nullPValues(300, 1, (k) => T.ttest1samp(draw(k, 0, 12), 0).pValue))
    expectUniformPValues(nullPValues(300, 2, (k) => T.ttestInd(draw(k, 0, 10), draw(k, 1, 14)).pValue))
    expectUniformPValues(
      nullPValues(300, 3, (k) => T.ttestInd(draw(k, 0, 10), draw(k, 1, 14, 0, 3), { equalVar: false }).pValue),
    )
    expectUniformPValues(nullPValues(300, 4, (k) => T.ttestRel(draw(k, 0, 10), draw(k, 1, 10)).pValue))
    expectUniformPValues(nullPValues(300, 5, (k) => T.ztest(draw(k, 0, 30), undefined, { sigma: 1 }).pValue))
    expectUniformPValues(
      nullPValues(200, 6, (k) => T.welchAnova([draw(k, 0, 8), draw(k, 1, 9, 0, 2), draw(k, 2, 10)]).pValue),
    )
  })

  it('one-sided p-values add up and the interval covers the mean', () => {
    const x = draw(key(8), 0, 15, 0.4)
    const less = T.ttest1samp(x, 0, { alternative: 'less' }).pValue
    const greater = T.ttest1samp(x, 0, { alternative: 'greater' }).pValue
    expect(less + greater).toBeCloseTo(1, 12)
    const intervals = Array.from(
      { length: 300 },
      (_, i) => T.ttest1samp(draw(key(9), i, 10, 2), 0).estimate?.interval ?? ([0, 0] as const),
    )
    expect(coverage(intervals, 2)).toBeGreaterThan(0.92)
  })

  it('reports effect sizes, degrees of freedom and assumptions', () => {
    const r = T.ttestInd([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    expect(r.df).toBe(8)
    expect(r.effectSize?.name).toBe("Cohen's d")
    expect(r.assumptions?.['equalVariances']).toBe(true)
    expect(formatResult(r)).toContain("Student's t test")
    const a = T.fOneway([
      [1, 2, 3],
      [2, 3, 4],
      [5, 6, 7],
    ])
    expect(a.df).toEqual([2, 6])
    const t = T.tukeyHsd([
      [1, 2, 3],
      [2, 3, 4],
      [5, 6, 7],
    ])
    expect(t.comparisons.map((c) => [c.i, c.j])).toEqual([
      [0, 1],
      [0, 2],
      [1, 2],
    ])
    expect(t.comparisons[1]?.pValue).toBeLessThan(0.05)
  })

  it('repeated-measures ANOVA has epsilon 1 for two conditions and rejects a strong effect', () => {
    const two = T.rmAnova([
      [1, 2],
      [2, 4],
      [3, 3],
      [1, 3],
    ])
    expect(two.assumptions?.['epsilon']).toBe(1)
    const rows = Array.from({ length: 12 }, (_, i) => draw(key(10), i, 3).map((v, j) => v + 2 * j))
    expect(T.rmAnova(rows).pValue).toBeLessThan(0.001)
  })
})

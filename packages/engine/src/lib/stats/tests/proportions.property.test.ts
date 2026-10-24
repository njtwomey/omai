import { describe, expect, it } from 'vitest'
import { coverage, expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { binomial, foldIn, key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, n: number, p: number): number => toFlat(binomial(k, n, p, []))[0] ?? 0

describe('proportions', () => {
  it('the exact binomial test never over-rejects, and the z tests are close to uniform', () => {
    expectUniformPValues(
      nullPValues(300, 1, (k) => T.binomTest(draw(k, 30, 0.3), 30, 0.3).pValue),
      { discrete: true },
    )
    expectUniformPValues(
      nullPValues(
        300,
        2,
        (k) => T.proportionsZtest([draw(foldIn(k, 0), 200, 0.4), draw(foldIn(k, 1), 200, 0.4)], [200, 200]).pValue,
      ),
      { maxDistance: 0.15 },
    )
  })

  it('every interval method covers the true proportion at roughly the nominal rate', () => {
    for (const method of ['wald', 'wilson', 'agresti-coull', 'clopper-pearson', 'jeffreys'] as const) {
      const intervals = Array.from({ length: 300 }, (_, i) => {
        const k = draw(foldIn(key(3), i), 80, 0.35)
        return (T.proportionConfint(k, 80, { method }).interval ?? [0, 0]) as [number, number]
      })
      expect(coverage(intervals, 0.35)).toBeGreaterThan(0.9)
    }
    const diffs = Array.from({ length: 300 }, (_, i) => {
      const a = draw(foldIn(key(4), 2 * i), 70, 0.5)
      const b = draw(foldIn(key(4), 2 * i + 1), 60, 0.3)
      return (T.proportionsDifference(a, 70, b, 60).estimate?.interval ?? [0, 0]) as [number, number]
    })
    expect(coverage(diffs, 0.2)).toBeGreaterThan(0.9)
  })

  it('Clopper-Pearson reaches the boundaries exactly', () => {
    expect(T.proportionConfint(0, 10, { method: 'clopper-pearson' }).interval?.[0]).toBe(0)
    expect(T.proportionConfint(10, 10, { method: 'clopper-pearson' }).interval?.[1]).toBe(1)
    expect(() => T.binomTest(11, 10)).toThrow()
  })
})

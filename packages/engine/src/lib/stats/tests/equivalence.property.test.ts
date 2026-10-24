import { describe, expect, it } from 'vitest'
import { nullPValues } from '../../../testing/stats.ts'
import { binomial, foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number, shift = 0): number[] =>
  toFlat(normal(foldIn(k, i), [n])).map((v) => v + shift)

describe('equivalence tests', () => {
  it('control the type I error at the equivalence boundary', () => {
    // True difference exactly at the upper bound: the null of non-equivalence holds, so rejections are at most alpha.
    const ps = nullPValues(400, 1, (k) => T.tost(draw(k, 0, 25, 0.5), draw(k, 1, 25), { low: -0.5, upp: 0.5 }).pValue)
    expect(ps.filter((p) => p < 0.05).length / ps.length).toBeLessThan(0.08)
    const qs = nullPValues(
      400,
      2,
      (k) => T.tostProportion(toFlat(binomial(k, 80, 0.6, []))[0] ?? 0, 80, { low: 0.4, upp: 0.6 }).pValue,
    )
    expect(qs.filter((p) => p < 0.05).length / qs.length).toBeLessThan(0.06)
  })

  it('rejects non-equivalence exactly when the (1 - 2 alpha) interval lies inside the bounds', () => {
    for (let i = 0; i < 50; i++) {
      const r = T.tost(draw(key(3), i, 20), undefined, { low: -0.6, upp: 0.6 })
      const [lo, hi] = r.estimate?.interval ?? [0, 0]
      expect(r.pValue < 0.05).toBe(lo > -0.6 && hi < 0.6)
    }
    expect(() => T.tost([1, 2, 3], undefined, { low: 1, upp: -1 })).toThrow()
    expect(T.tostProportions(500, 1000, 498, 1000, { low: -0.1, upp: 0.1 }).pValue).toBeLessThan(1e-4)
  })
})

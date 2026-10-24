import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { toArray } from '../../tensor/index.ts'
import * as T from './index.ts'

const rows = (k: ReturnType<typeof key>, n: number, p: number, shift = 0): number[][] =>
  (toArray(normal(k, [n, p])) as number[][]).map((r) => r.map((v) => v + shift))

describe('multivariate tests', () => {
  it('are valid under the null', () => {
    expectUniformPValues(nullPValues(200, 1, (k) => T.hotellingT2(rows(k, 15, 3)).pValue))
    expectUniformPValues(
      nullPValues(200, 2, (k) => T.hotellingT2(rows(foldIn(k, 0), 12, 2), rows(foldIn(k, 1), 14, 2)).pValue),
    )
    expectUniformPValues(
      nullPValues(
        200,
        3,
        (k) => T.manova([rows(foldIn(k, 0), 10, 2), rows(foldIn(k, 1), 10, 2), rows(foldIn(k, 2), 10, 2)]).wilks.pValue,
      ),
    )
    expectUniformPValues(
      nullPValues(200, 4, (k) => T.boxM([rows(foldIn(k, 0), 20, 2), rows(foldIn(k, 1), 20, 2)]).pValue),
      {
        maxDistance: 0.15,
      },
    )
  })

  it('agree with their univariate counterparts in one dimension', () => {
    const a = [[1], [2], [4], [3], [6], [5]]
    const b = [[4], [6], [5], [8], [7], [9]]
    const t = T.ttestInd(a.flat(), b.flat())
    expect(T.hotellingT2(a, b).statistic).toBeCloseTo(t.statistic ** 2, 10)
    expect(T.hotellingT2(a, b).pValue).toBeCloseTo(t.pValue, 10)
    expect(T.manova([a, b]).wilks.pValue).toBeCloseTo(T.fOneway([a.flat(), b.flat()]).pValue, 10)
  })

  it("Mardia's tests reject heavy tails and pass normal data", () => {
    const k = key(9)
    const ps = [0, 1, 2, 3, 4].map((i) => T.mardia(rows(foldIn(k, 10 + i), 200, 2)).kurtosis.pValue)
    expect(ps.filter((p) => p > 0.05).length).toBeGreaterThanOrEqual(4)
    const heavy = rows(foldIn(k, 1), 200, 2).map((r) => r.map((v) => v ** 3))
    expect(T.mardia(heavy).kurtosis.pValue).toBeLessThan(1e-4)
    expect(T.mardia(heavy).normal).toBe(false)
  })
})

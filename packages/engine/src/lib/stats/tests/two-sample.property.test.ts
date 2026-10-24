import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number, shift = 0): number[] =>
  toFlat(normal(foldIn(k, i), [n])).map((v) => v + shift)

describe('two-sample tests', () => {
  it('are valid under the null', () => {
    expectUniformPValues(
      nullPValues(300, 1, (k) => T.ks2samp(draw(k, 0, 15), draw(k, 1, 18)).pValue),
      { discrete: true },
    )
    expectUniformPValues(nullPValues(200, 2, (k) => T.cramervonmises2samp(draw(k, 0, 25), draw(k, 1, 30)).pValue))
    expectUniformPValues(
      nullPValues(200, 3, (k) => T.eppsSingleton(draw(k, 0, 30), draw(k, 1, 30)).pValue),
      {
        maxDistance: 0.15,
      },
    )
    expectUniformPValues(
      nullPValues(
        100,
        4,
        (k) => T.energyTest(draw(k, 0, 10), draw(k, 1, 10), { key: foldIn(k, 9), nResamples: 99 }).pValue,
      ),
      { maxDistance: 0.2, lo: 0, hi: 0.12 },
    )
  })

  it('detect a shift', () => {
    const k = key(5)
    const x = draw(k, 0, 40)
    const y = draw(k, 1, 40, 1)
    for (const r of [T.ks2samp(x, y), T.cramervonmises2samp(x, y), T.andersonKsamp([x, y]), T.eppsSingleton(x, y)])
      expect(r.pValue).toBeLessThan(0.01)
  })

  it('distances are symmetric, zero on identical samples, and Wasserstein is the mean shift for a translate', () => {
    const x = [1, 4, 2, 8, 5]
    expect(
      T.wassersteinDistance(
        x,
        x.map((v) => v + 3),
      ),
    ).toBeCloseTo(3, 12)
    expect(T.energyDistance(x, x)).toBe(0)
    expect(T.energyDistance(x, [2, 3])).toBeCloseTo(T.energyDistance([2, 3], x), 12)
  })

  it('the classifier test rejects separable samples and not exchangeable ones', () => {
    const nearest = (
      tx: readonly (readonly number[])[],
      ty: readonly number[],
      q: readonly (readonly number[])[],
    ): number[] =>
      q.map((p) => {
        let best = 0
        let d = Number.POSITIVE_INFINITY
        tx.forEach((t, i) => {
          const e = Math.abs((t[0] ?? 0) - (p[0] ?? 0))
          if (e < d) {
            d = e
            best = ty[i] ?? 0
          }
        })
        return best
      })
    const k = key(6)
    expect(T.classifierTwoSampleTest(draw(k, 0, 40), draw(k, 1, 40, 4), nearest).pValue).toBeLessThan(1e-6)
    expect(T.classifierTwoSampleTest(draw(k, 2, 40), draw(k, 3, 40), nearest, { seed: 2 }).pValue).toBeGreaterThan(0.01)
  })
})

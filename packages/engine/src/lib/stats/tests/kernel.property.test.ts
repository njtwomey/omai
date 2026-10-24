import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number): number[] => toFlat(normal(foldIn(k, i), [n]))

describe('kernel tests', () => {
  it('MMD and HSIC permutation tests are valid under the null', () => {
    expectUniformPValues(
      nullPValues(
        80,
        1,
        (k) => T.mmdTest(draw(k, 0, 12), draw(k, 1, 12), { key: foldIn(k, 2), nResamples: 99 }).pValue,
      ),
      { maxDistance: 0.2, lo: 0, hi: 0.15 },
    )
    expectUniformPValues(
      nullPValues(
        80,
        2,
        (k) => T.hsicTest(draw(k, 0, 15), draw(k, 1, 15), { key: foldIn(k, 2), nResamples: 99 }).pValue,
      ),
      { maxDistance: 0.2, lo: 0, hi: 0.15 },
    )
  })

  it('the gamma approximations agree with the permutation nulls on clear signals', () => {
    const k = key(3)
    const x = draw(k, 0, 30)
    const y = draw(k, 1, 30).map((v) => v * 2.5)
    expect(T.mmdTest(x, y, { null: 'gamma' }).pValue).toBeLessThan(0.01)
    expect(T.mmdTest(x, y, { nResamples: 199 }).pValue).toBeLessThan(0.02)
    expect(
      T.hsicTest(
        x,
        x.map((v) => v * v),
        { null: 'gamma' },
      ).pValue,
    ).toBeLessThan(0.01)
    expect(() => T.mmdTest(x, y.slice(1), { null: 'gamma' })).toThrow()
  })

  it('MMD is zero for identical samples and positive otherwise, for any kernel', () => {
    const k = T.gaussianKernel(0.7)
    expect(T.mmd2([1, 2, 3], [1, 2, 3], { kernel: k })).toBeCloseTo(0, 14)
    expect(T.mmd2([1, 2, 3], [4, 5, 6], { kernel: k })).toBeGreaterThan(0)
    const linear = (a: ArrayLike<number>, b: ArrayLike<number>): number => (a[0] ?? 0) * (b[0] ?? 0)
    // With a linear kernel, the biased MMD^2 is the squared difference of the means.
    expect(T.mmd2([1, 2, 3], [4, 5, 9], { kernel: linear })).toBeCloseTo(16, 12)
  })
})

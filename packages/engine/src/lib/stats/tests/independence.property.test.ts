import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

const draw = (k: ReturnType<typeof key>, i: number, n: number): number[] => toFlat(normal(foldIn(k, i), [n]))

describe('independence tests', () => {
  it('are valid under independence', () => {
    const opts = (k: ReturnType<typeof key>): { key: ReturnType<typeof key>; nResamples: number } => ({
      key: foldIn(k, 9),
      nResamples: 99,
    })
    const loose = { maxDistance: 0.2, lo: 0, hi: 0.15 }
    expectUniformPValues(
      nullPValues(80, 1, (k) => T.distanceCovarianceTest(draw(k, 0, 15), draw(k, 1, 15), opts(k)).pValue),
      loose,
    )
    expectUniformPValues(
      nullPValues(80, 2, (k) => T.hoeffdingD(draw(k, 0, 15), draw(k, 1, 15), opts(k)).pValue),
      loose,
    )
    expectUniformPValues(
      nullPValues(80, 3, (k) => T.mutualInformationTest(draw(k, 0, 20), draw(k, 1, 20), opts(k)).pValue),
      loose,
    )
    expectUniformPValues(
      nullPValues(300, 4, (k) => {
        const z = draw(k, 0, 40)
        const x = draw(k, 1, 40).map((v, i) => v + (z[i] ?? 0))
        const y = draw(k, 2, 40).map((v, i) => v - (z[i] ?? 0))
        return T.partialCorrelationTest(
          x.map((v, i) => [v, y[i] ?? 0, z[i] ?? 0]),
          0,
          1,
          [2],
        ).pValue
      }),
    )
  })

  it('detect nonlinear dependence that correlation misses', () => {
    const x = draw(key(5), 0, 40)
    const y = x.map((v) => v * v)
    expect(T.distanceCovarianceTest(x, y, { seed: 1 }).pValue).toBeLessThan(0.01)
    expect(T.hoeffdingD(x, y, { seed: 1 }).pValue).toBeLessThan(0.01)
  })

  it('conditioning removes dependence explained by a common cause', () => {
    const z = draw(key(6), 0, 120)
    const x = z.map((v, i) => v + 0.3 * (draw(key(6), 1, 120)[i] ?? 0))
    const y = z.map((v, i) => v + 0.3 * (draw(key(6), 2, 120)[i] ?? 0))
    const data = x.map((v, i) => [v, y[i] ?? 0, z[i] ?? 0])
    expect(T.partialCorrelationTest(data, 0, 1, []).pValue).toBeLessThan(1e-6)
    expect(T.partialCorrelationTest(data, 0, 1, [2]).pValue).toBeGreaterThan(0.01)
    const disc = data.map((r) => r.map((v) => (v > 0 ? 1 : 0)))
    expect(T.gTestConditional(disc, 0, 1, []).pValue).toBeLessThan(1e-3)
    expect(T.kernelCITest(data.slice(0, 60), 0, 1, [2]).pValue).toBeGreaterThan(0.01)
  })
})

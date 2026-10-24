import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { sliceAxis, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  aggregatedVarianceHurst,
  detrendedFluctuation,
  fgnAutocovariance,
  fractionalBrownianMotion,
  fractionalGaussianNoise,
  hurstRs,
  longRangeDependence,
} from './fractional.ts'
import { valuesAt } from './paths.ts'

describe('fractional Gaussian noise and Brownian motion', () => {
  it('Davies-Harte samples have the fGn autocovariance, like the Cholesky method', () => {
    for (const method of ['daviesHarte', 'cholesky'] as const) {
      const x = fractionalGaussianNoise(key(1), 64, 0.75, { paths: 1500, method })
      const a = toFlat(sliceAxis(x, 1, 10, 11))
      const b = toFlat(sliceAxis(x, 1, 11, 12))
      const c = toFlat(sliceAxis(x, 1, 14, 15))
      expect(Math.abs(varianceOf(a) - 1)).toBeLessThan(5 * Math.sqrt(2 / 1500))
      const lag = (u: number[], v: number[]): number => meanOf(u.map((p, i) => p * (v[i] ?? 0)))
      const g = toFlat(fgnAutocovariance([1, 4], 0.75))
      expect(Math.abs(lag(a, b) - (g[0] ?? 0))).toBeLessThan(0.08)
      expect(Math.abs(lag(a, c) - (g[1] ?? 0))).toBeLessThan(0.08)
    }
  })

  it('fBm is self-similar: Var B_H(t) = t^{2H}', () => {
    const b = fractionalBrownianMotion(key(2), 2, 32, 0.3, { paths: 3000 })
    expect(b.values.shape).toEqual([3000, 33])
    const end = toFlat(valuesAt(b, -1))
    const v = 2 ** 0.6
    expect(Math.abs(varianceOf(end) - v)).toBeLessThan(5 * v * Math.sqrt(2 / 3000))
    expect(toFlat(valuesAt(b, 0)).every((x) => x === 0)).toBe(true)
  })

  it('differentiates sample paths in H and sigma', () => {
    const f = (h: Value, s: Value): Value =>
      sum(fractionalBrownianMotion(key(3), 1, 8, h, { paths: 2, sigma: s }).values, 1)
    expectConformant(f, [tensor(0.6), tensor(1.3)], { rtol: 1e-4 })
    expectConformant((h: Value) => fgnAutocovariance([1, 2, 5], h), [tensor([0.3])], { vmap: false })
  })

  it('rejects Hurst exponents outside (0, 1)', () => {
    expect(() => fractionalGaussianNoise(key(0), 8, 1)).toThrow()
    expect(() => longRangeDependence(0)).toThrow()
  })
})

describe('Hurst estimation and long-range dependence', () => {
  it('estimators recover H from fractional Gaussian noise', () => {
    for (const h of [0.3, 0.7]) {
      const x = fractionalGaussianNoise(key(10 + h * 10), 4096, h)
      expect(Math.abs(detrendedFluctuation(x) - h)).toBeLessThan(0.08)
      expect(Math.abs(aggregatedVarianceHurst(x) - h)).toBeLessThan(0.1)
      expect(Math.abs(hurstRs(x) - h)).toBeLessThan(0.12)
    }
    // DFA of fBm (the integrated series) gives H + 1
    const fbm = toFlat(fractionalBrownianMotion(key(5), 1, 4096, 0.5).values)
    expect(Math.abs(detrendedFluctuation(fbm) - 1.5)).toBeLessThan(0.1)
  })

  it('long memory has non-summable correlations', () => {
    const l = longRangeDependence(0.8)
    expect(l.longMemory).toBe(true)
    expect(l.partialSums[3]).toBeGreaterThan(l.partialSums[1] ?? 0)
    const a = longRangeDependence(0.3)
    expect(a.longMemory).toBe(false)
    expect(a.partialSums[3]).toBeLessThan(0.1)
    expect(() => hurstRs([1, 2, 3])).toThrow()
    expect(Number.isNaN(hurstRs(new Array<number>(64).fill(1), { windows: [8, 16] }))).toBe(true)
  })
})

import { describe, expect, it } from 'vitest'
import { expectUniformPValues, nullPValues } from '../../../testing/stats.ts'
import { exponential as expDist, normal as normalDist } from '../../dists/index.ts'
import { exponential, foldIn, key, normal, uniform } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as T from './index.ts'

describe('goodness-of-fit tests', () => {
  it('are valid under the null', () => {
    expectUniformPValues(nullPValues(300, 1, (k) => T.kstest(toFlat(normal(k, [25])), normalDist()).pValue))
    expectUniformPValues(nullPValues(300, 2, (k) => T.kstest(toFlat(normal(k, [300])), normalDist()).pValue))
    expectUniformPValues(nullPValues(300, 3, (k) => T.shapiro(toFlat(normal(k, [30]))).pValue))
    expectUniformPValues(nullPValues(200, 4, (k) => T.cramervonmises(toFlat(uniform(k, [20])), (u) => u).pValue))
    expectUniformPValues(nullPValues(200, 5, (k) => T.kstest(toFlat(exponential(k, [15])), expDist()).pValue))
  })

  it('detect departures', () => {
    const skewed = toFlat(exponential(key(9), [80]))
    expect(T.shapiro(skewed).pValue).toBeLessThan(0.01)
    expect(T.lilliefors(skewed).pValue).toBeLessThan(0.01)
    expect(T.andersonDarling(skewed).pValue).toBeLessThanOrEqual(0.01)
    expect(T.jarqueBera(skewed).pValue).toBeLessThan(0.01)
    expect(T.kstest(skewed, normalDist(1, 1)).pValue).toBeLessThan(0.05)
  })

  it('simulates Lilliefors p-values reproducibly when the analytic approximation does not apply', () => {
    const x = toFlat(normal(foldIn(key(4), 1), [30]))
    const a = T.lilliefors(x, { seed: 2, nResamples: 300 })
    const b = T.lilliefors(x, { seed: 2, nResamples: 300 })
    expect(a.pValue).toBe(b.pValue)
    expect(a.pValue).toBeGreaterThan(0.1)
    expect(a.method).toBe('simulation')
  })

  it('chi-square goodness of fit needs matching totals', () => {
    expect(() => T.chisquare([1, 2], { expected: [1, 1] })).toThrow()
    expect(T.chisquare([10, 10, 10]).statistic).toBe(0)
  })
})

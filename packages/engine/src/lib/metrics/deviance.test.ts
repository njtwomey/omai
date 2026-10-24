import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

const v = (r: { readonly estimate: { readonly value: number } }): number => r.estimate.value

describe('deviances', () => {
  it('averages the Tweedie unit deviance into the mean deviance', () => {
    const y = [0.5, 1.2, 3, 0.1]
    const mu = [0.7, 1, 2.5, 0.3]
    for (const power of [0, 1, 1.5, 2, 3]) {
      const mean = y.reduce((s, yi, i) => s + M.tweedieUnitDeviance(yi, mu[i] ?? 1, power), 0) / y.length
      expect(v(M.meanTweedieDeviance(y, mu, { power }))).toBeCloseTo(mean, 14)
    }
    expect(M.tweedieUnitDeviance(2, 2, 1.5)).toBe(0)
  })
})

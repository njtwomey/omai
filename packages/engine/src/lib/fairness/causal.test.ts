import { describe, expect, it } from 'vitest'
import { bernoulli, normal } from '../dists/index.ts'
import { sample } from '../probprog/index.ts'
import { key } from '../random/index.ts'
import { add, mul } from '../tensor/index.ts'
import { counterfactualFairness, pathSpecificEffects } from './causal.ts'

// a -> m -> y and a -> y; u is independent of a.
const scm = (): void => {
  const a = sample('a', bernoulli({ probs: 0.4 }))
  const u = sample('u', normal(0, 1))
  const m = sample('m', normal(add(mul(a, 1.5), u), 0.5))
  sample('y', normal(add(mul(a, 0.5), mul(m, 2)), 0.3))
}

describe('causal fairness', () => {
  it('a predictor of the exogenous noise is counterfactually fair; one of a descendant is not', () => {
    const fair = counterfactualFairness(scm, { attribute: 'a', predictor: (s) => s['u'] ?? 0, n: 200, key: key(1) })
    expect(fair.meanGap).toBe(0)
    expect(fair.flipRate).toBe(0)
    const unfair = counterfactualFairness(scm, {
      attribute: 'a',
      predictor: (s) => s['m'] ?? 0,
      n: 200,
      threshold: 1,
      key: key(1),
    })
    expect(unfair.meanGap).toBeCloseTo(1.5, 9)
    expect(unfair.flipRate).toBeGreaterThan(0.2)
    expect(unfair.attribute.filter((a) => a === 1).length).toBeGreaterThan(40)
    expect(() => counterfactualFairness(scm, { attribute: 'z', predictor: () => 0, n: 2 })).toThrow(/no site/)
  })

  it('decomposes the total effect into direct and indirect parts exactly in a linear model', () => {
    const e = pathSpecificEffects(scm, { attribute: 'a', outcome: (s) => s['y'] ?? 0, mediators: ['m'], n: 300 })
    expect(e.total).toBeCloseTo(3.5, 9)
    expect(e.direct).toBeCloseTo(0.5, 9)
    expect(e.indirect).toBeCloseTo(3, 9)
    const none = pathSpecificEffects(scm, { attribute: 'a', outcome: (s) => s['y'] ?? 0, mediators: [], n: 50 })
    expect(none.indirect).toBeCloseTo(0, 9)
    expect(() => pathSpecificEffects(scm, { attribute: 'a', outcome: () => 0, mediators: ['q'], n: 2 })).toThrow(
      /no site/,
    )
  })
})

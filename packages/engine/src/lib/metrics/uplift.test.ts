import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { upliftData } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { auuc, policyValue, qiniCoefficient, upliftAtK, upliftCurve, upliftDeciles } from './uplift.ts'

describe('uplift curves on a hand-made experiment', () => {
  const score = [4, 3, 2, 1]
  const t = [1, 0, 1, 0]
  const y = [1, 0, 0, 0]
  it('the Qini and uplift curves follow their definitions', () => {
    expect(Array.from(upliftCurve(score, t, y, 'qini').value)).toEqual([0, 0, 1, 1, 1])
    // (R_t/N_t - R_c/N_c) k: k = 2 gives (1 - 0) * 2, k = 3 gives (1/2 - 0) * 3, k = 4 gives (1/2 - 0) * 4
    expect(Array.from(upliftCurve(score, t, y, 'uplift').value)).toEqual([0, 0, 2, 1.5, 2])
    const c = upliftCurve(score, t, y, 'qini')
    expect(Array.from(c.random)).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(Array.from(c.treated)).toEqual([0, 1, 1, 2, 2])
  })

  it('the coefficients are areas against the random line', () => {
    expect(qiniCoefficient(score, t, y).estimate.value).toBeCloseTo(0.125, 12)
    expect(qiniCoefficient(score, t, y, { subtractRandom: false }).estimate.value).toBeCloseTo(0.625, 12)
    // trapezoids over the fractions 0, 1/4, 1/2, 3/4, 1 of the values 0, 0, 2, 1.5, 2: 0 + 0.25 + 0.4375 + 0.4375
    expect(auuc(score, t, y).estimate.value).toBeCloseTo(0.125, 12)
    expect(auuc(score, t, y, { subtractRandom: false }).estimate.value).toBeCloseTo(1.125, 12)
  })

  it('only the order of the scores matters', () => {
    const a = upliftCurve(score, t, y).value
    const b = upliftCurve(
      score.map((v) => 10 * v + 3),
      t,
      y,
    ).value
    expect(Array.from(b)).toEqual(Array.from(a))
  })

  it('uplift at k, deciles and the policy value', () => {
    expect(upliftAtK(score, t, y, 0.5).estimate.value).toBe(1)
    expect(upliftDeciles(score, t, y, 2).map((b) => b.uplift)).toEqual([1, 0])
    const p = policyValue(score, t, y, { budget: 0.5, propensity: 0.5 })
    expect(p.estimate.value).toBeCloseTo(0.5, 12)
    expect(p.breakdown?.['treatAll']).toBeCloseTo(0.5, 12)
    expect(p.breakdown?.['treatNone']).toBeCloseTo(0, 12)
  })

  it('refuses mismatched inputs and non-binary treatment', () => {
    expect(() => upliftCurve([1, 2], [1, 0, 1], [1, 0])).toThrow()
    expect(() => upliftCurve([1, 2], [2, 0], [1, 0])).toThrow()
  })
})

describe('uplift curves on a planted experiment', () => {
  const g = generate(upliftData, { key: key(3), knobs: { n: 20000, covariates: 3 } })
  const m = toFlat(g.data.x)
  const n = 20000
  const t = Array.from({ length: n }, (_, i) => m[i * 4 + 3] ?? 0)
  const y = Array.from(toFlat(g.data.y))
  const truthUplift = Array.from(toFlat((g.truth.extra as { sampleUplift: Tensor }).sampleUplift))

  it('ranking by the true uplift beats ranking by chance and by the opposite order', () => {
    const oracle = qiniCoefficient(truthUplift, t, y).estimate.value
    const reversed = qiniCoefficient(
      truthUplift.map((v) => -v),
      t,
      y,
    ).estimate.value
    const chance = qiniCoefficient(
      Array.from(toFlat(g.truth.effects?.ite as Tensor)).map((_, i) => (Math.sin(i * 12.9898) * 43758.5453) % 1),
      t,
      y,
    ).estimate.value
    expect(oracle).toBeGreaterThan(0)
    expect(oracle).toBeGreaterThan(Math.abs(chance) * 5)
    expect(reversed).toBeLessThan(0)
    expect(auuc(truthUplift, t, y).estimate.value).toBeGreaterThan(0)
  })

  it('the policy of treating the predicted persuadable beats treating everyone and no one', () => {
    const p = policyValue(truthUplift, t, y, { budget: 0.4, propensity: 0.5 })
    expect(p.estimate.value).toBeGreaterThan((p.breakdown?.['treatAll'] ?? 0) + 0.005)
    expect(p.estimate.value).toBeGreaterThan((p.breakdown?.['treatNone'] ?? 0) + 0.005)
    const bins = upliftDeciles(truthUplift, t, y, 5)
    expect(bins[0]?.uplift).toBeGreaterThan(bins[4]?.uplift ?? 0)
    expect(bins[4]?.uplift).toBeLessThan(0)
  })
})

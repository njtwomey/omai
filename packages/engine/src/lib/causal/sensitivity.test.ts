import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { treatmentEffects } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import {
  benchmarkStrength,
  eValue,
  matchedPairs,
  matching,
  omittedVariableBias,
  ovbContour,
  partialR2,
  refute,
  regressionAdjustment,
  robustnessValue,
  rosenbaumBounds,
  rosenbaumBoundsBinary,
  sensitivityModel,
  splitTreatment,
} from './index.ts'

describe('E-value', () => {
  it('reproduces the worked examples of VanderWeele and Ding', () => {
    // Risk ratio 3.9 (95% CI 1.8 to 8.7): E-values 7.26 and 3.0.
    const e = eValue(3.9, { lower: 1.8, upper: 8.7 })
    expect(e.point).toBeCloseTo(7.263, 3)
    expect(e.interval).toBeCloseTo(3.0, 5)
    // A protective ratio is inverted: RR 0.5 has the E-value of 2.
    expect(eValue(0.5).point).toBeCloseTo(2 + Math.sqrt(2), 12)
    // An interval that includes the null has E-value 1.
    expect(eValue(1.5, { lower: 0.9, upper: 2.5 }).interval).toBe(1)
    expect(eValue(1).point).toBe(1)
  })

  it('converts odds ratios, hazard ratios and standardised differences to risk ratios', () => {
    expect(eValue(4, { scale: 'or-common' }).riskRatio.estimate).toBeCloseTo(2, 12)
    expect(eValue(0.5, { scale: 'smd' }).riskRatio.estimate).toBeCloseTo(Math.exp(0.455), 12)
    expect(eValue(2, { scale: 'hr-common' }).riskRatio.estimate).toBeCloseTo(
      (1 - 0.5 ** Math.SQRT2) / (1 - 0.5 ** (1 / Math.SQRT2)),
      12,
    )
    expect(eValue(2, { scale: 'hr-rare' }).point).toBeCloseTo(eValue(2).point, 12)
  })
})

describe('omitted-variable bias (Cinelli and Hazlett)', () => {
  // Darfur: the coefficient of directly harmed 0.0973 (se 0.0233, df 783) in the sensemakr documentation.
  const darfur = { estimate: 0.09731582, standardError: 0.02325654, df: 783 }

  it('reproduces the published robustness values and partial R-squared', () => {
    expect(partialR2(darfur)).toBeCloseTo(0.02187, 4)
    expect(robustnessValue(darfur)).toBeCloseTo(0.1388, 4)
    expect(robustnessValue(darfur, { alpha: 0.05 })).toBeCloseTo(0.0763, 3)
    expect(robustnessValue(darfur, { q: 0.5 })).toBeLessThan(robustnessValue(darfur))
  })

  it('a confounder at the robustness value explains the estimate away exactly', () => {
    const rv = robustnessValue(darfur)
    expect(omittedVariableBias(darfur, rv, rv).adjustedEstimate).toBeCloseTo(0, 10)
    const half = omittedVariableBias(darfur, 0.03, 0.03)
    expect(half.adjustedEstimate).toBeLessThan(darfur.estimate)
    expect(half.adjustedEstimate).toBeGreaterThan(0)
  })

  it('the contour is the zero line of explaining away: monotone in both strengths', () => {
    const c = ovbContour(darfur, [0, 0.05, 0.1], [0, 0.05, 0.1])
    expect(c.estimate[0]).toEqual([darfur.estimate, darfur.estimate, darfur.estimate])
    expect(c.estimate[2]?.[2]).toBeLessThan(c.estimate[1]?.[1] ?? 0)
  })

  it('with the true partial R-squared of a dropped covariate the formula recovers the full-model estimate exactly', () => {
    const g = generate(treatmentEffects, {
      key: key(1),
      knobs: { n: 3000, ate: 1, confounding: 1.2, heterogeneity: 0 },
    })
    const full = splitTreatment(g.data)
    const model = sensitivityModel(full)
    const m = toFlat(full.x as Tensor)
    const d = (full.x as Tensor).shape[1] ?? 0
    const reduced = {
      ...full,
      x: Array.from({ length: m.length / d }, (_, i) => Array.from({ length: d - 1 }, (_, j) => m[i * d + 1 + j] ?? 0)),
    }
    const restricted = sensitivityModel(reduced).effect
    expect(restricted.estimate).toBeGreaterThan(model.effect.estimate)
    const adj = omittedVariableBias(restricted, model.r2dxj[0] ?? 0, model.r2yxj[0] ?? 0)
    expect(adj.adjustedEstimate).toBeCloseTo(model.effect.estimate, 8)
    expect(adj.adjustedStandardError).toBeCloseTo(model.effect.standardError, 3)
  })

  it('benchmarking: a confounder as strong as an observed covariate bounds the bias, and is impossible when too strong', () => {
    const b = benchmarkStrength(0.1, 0.05, 1, 1)
    expect(b.r2dz).toBeCloseTo(0.1 / 0.9, 12)
    expect(b.r2yz).toBeGreaterThan(0.05)
    expect(() => benchmarkStrength(0.5, 0.1, 3, 3)).toThrow()
  })
})

describe('Rosenbaum bounds', () => {
  const g = generate(treatmentEffects, {
    key: key(2),
    knobs: { n: 1500, ate: 1, confounding: 1, heterogeneity: 0 },
  })
  const data = splitTreatment(g.data)

  it('the bounds widen with Gamma, are equal at Gamma 1, and a real effect has a critical Gamma above one', () => {
    const pairs = matchedPairs(data, matching(data, { on: 'mahalanobis', replacement: false }))
    const r = rosenbaumBounds(pairs.treated, pairs.control)
    expect(r.rows[0]?.pUpper).toBeCloseTo(r.rows[0]?.pLower ?? 0, 12)
    expect(r.rows[0]?.pUpper).toBeLessThan(1e-6)
    for (let i = 1; i < r.rows.length; i++) expect(r.rows[i]?.pUpper).toBeGreaterThanOrEqual(r.rows[i - 1]?.pUpper ?? 1)
    expect(r.criticalGamma).toBeGreaterThan(1.5)
    const at = rosenbaumBounds(pairs.treated, pairs.control, { gammas: [r.criticalGamma] }).rows[0]
    expect(at?.pUpper).toBeCloseTo(0.05, 6)
  })

  it('no effect: the finding is not significant even without hidden bias', () => {
    const none = splitTreatment(
      generate(treatmentEffects, {
        key: key(3),
        knobs: { n: 1500, ate: 0, heterogeneity: 0, confounding: 0 },
      }).data,
    )
    const pairs = matchedPairs(none, matching(none, { on: 'mahalanobis', replacement: false }))
    expect(rosenbaumBounds(pairs.treated, pairs.control).criticalGamma).toBe(1)
  })

  it('the binary bounds are exact binomial tails', () => {
    const treated = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    const control = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    const r = rosenbaumBoundsBinary(treated, control)
    // b = 8 of m = 9 discordant: P(Bin(9, 1/2) >= 8) = 10 / 512
    expect(r.rows[0]?.pUpper).toBeCloseTo(10 / 512, 12)
    expect(r.criticalGamma).toBeGreaterThan(1)
    expect(() => rosenbaumBoundsBinary([1], [1])).toThrow()
  })
})

describe('refutation', () => {
  const g = generate(treatmentEffects, { key: key(4), knobs: { n: 1500, ate: 2, confounding: 1, heterogeneity: 0 } })
  const data = splitTreatment(g.data)
  const estimator = (d: typeof data): number => regressionAdjustment(d, { mode: 'linear' }).estimate.value

  it('a placebo treatment makes the estimate vanish', () => {
    const r = refute(data, estimator, { test: 'placebo-treatment', trials: 20 })
    expect(Math.abs(r.mean)).toBeLessThan(0.15)
    expect(r.original).toBeGreaterThan(1.5)
    expect(r.pValue).toBeLessThan(0.1)
    expect(r.passed).toBe(true)
  })

  it('a random common cause and a data subset do not move a sound estimate', () => {
    for (const test of ['random-common-cause', 'data-subset'] as const) {
      const r = refute(data, estimator, { test, trials: 20 })
      expect(Math.abs(r.mean - r.original)).toBeLessThan(0.1)
      expect(r.passed, test).toBe(true)
    }
  })
})

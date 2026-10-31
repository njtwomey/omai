import { describe, expect, it } from 'vitest'
import {
  argmaxOf,
  coverageOf,
  firingRule,
  likelihoodRatio,
  literalHolds,
  modelRecord,
  modelText,
  predictCounts,
  type RuleModel,
  ruleQuality,
  ruleSignificance,
  ruleStatistics,
  smoothedProbabilities,
} from './index.ts'

const list: RuleModel = {
  kind: 'list',
  classes: [0, 1],
  nFeatures: 2,
  rules: [
    { literals: [{ feature: 0, op: '>', value: 1 }], label: 1, counts: [1, 5] },
    { literals: [{ feature: 1, op: '<=', value: 0 }], label: 0, counts: [4, 0] },
  ],
  defaultCounts: [2, 2],
}
const x = Float64Array.of(2, 0, 0, 0, 0, 1, 3, 3, Number.NaN, Number.NaN)

describe('literals and coverage', () => {
  it('hold by operator and never on a missing value', () => {
    expect(literalHolds({ feature: 0, op: '==', value: 2 }, 2)).toBe(true)
    expect(literalHolds({ feature: 0, op: '<=', value: 2 }, 2)).toBe(true)
    expect(literalHolds({ feature: 0, op: '>', value: 2 }, 2)).toBe(false)
    expect(literalHolds({ feature: 0, op: '<=', value: 2 }, Number.NaN)).toBe(false)
  })

  it('coverage is the conjunction', () => {
    const c = coverageOf(
      [
        { feature: 0, op: '>', value: 1 },
        { feature: 1, op: '>', value: 2 },
      ],
      x,
      2,
    )
    expect(Array.from(c)).toEqual([0, 0, 0, 1, 0])
  })
})

describe('prediction', () => {
  it('a list takes the first firing rule and falls back to the default', () => {
    expect(Array.from(predictCounts(list, x))).toEqual([1, 5, 4, 0, 2, 2, 1, 5, 2, 2])
    expect(firingRule(list, x)).toEqual([0, 1, -1, 0, -1])
  })

  it('a set sums the counts of every firing rule', () => {
    const set: RuleModel = { ...list, kind: 'set' }
    // Row 0 satisfies only rule 0 (x1 = 0 is <= 0 too): both fire.
    expect(Array.from(predictCounts(set, x).subarray(0, 2))).toEqual([5, 5])
  })

  it('probabilities are Laplace smoothed and argmax breaks ties to the first', () => {
    expect(smoothedProbabilities([3, 0])).toEqual([0.8, 0.2])
    expect(argmaxOf([2, 2, 1])).toBe(0)
  })

  it('writes the model as text and as a record', () => {
    expect(modelText(list, ['a', 'b'], ['no', 'yes'])).toBe('IF a > 1 THEN yes\nIF b <= 0 THEN no\nELSE no')
    const rec = modelRecord(list, ['a', 'b'], ['no', 'yes'], 'demo')
    expect(rec.rules[0]?.precision).toBeCloseTo(5 / 6, 12)
    expect(rec.rules[0]?.coverage).toBeCloseTo(6 / 14, 12)
  })
})

describe('rule quality measures', () => {
  const prior = [10, 10]
  it('match their definitions', () => {
    expect(ruleQuality('accuracy', [8, 2], 0, prior)).toBeCloseTo(0.8, 12)
    expect(ruleQuality('laplace', [8, 2], 0, prior)).toBeCloseTo(9 / 12, 12)
    expect(ruleQuality('m-estimate', [8, 2], 0, prior, 2)).toBeCloseTo((8 + 2 * 0.5) / 12, 12)
    expect(ruleQuality('wracc', [8, 2], 0, prior)).toBeCloseTo((10 / 20) * (0.8 - 0.5), 12)
    // minus the entropy in bits of (0.8, 0.2)
    expect(ruleQuality('entropy', [8, 2], 0, prior)).toBeCloseTo(0.8 * Math.log2(0.8) + 0.2 * Math.log2(0.2), 12)
    expect(ruleQuality('accuracy', [0, 0], 0, prior)).toBe(Number.NEGATIVE_INFINITY)
  })

  it('support, confidence, lift, Laplace accuracy and WRAcc of a rule on data', () => {
    const labels = [0, 0, 1, 1, 1, 1]
    const data = Float64Array.of(0, 0, 1, 1, 1, 0)
    const s = ruleStatistics({ literals: [{ feature: 0, op: '>', value: 0 }], label: 1 }, data, 1, labels, 2)
    expect(s.covered).toBe(3)
    expect(s.correct).toBe(3)
    expect(s.support).toBeCloseTo(0.5, 12)
    expect(s.confidence).toBe(1)
    expect(s.lift).toBeCloseTo(1 / (4 / 6), 12)
    expect(s.laplace).toBeCloseTo(4 / 5, 12)
    expect(s.wracc).toBeCloseTo(0.5 * (1 - 4 / 6), 12)
  })

  it('the likelihood ratio is 2 sum f ln(f / e), zero at the prior, and the p-value falls with evidence', () => {
    expect(likelihoodRatio([5, 5], [10, 10])).toBeCloseTo(0, 12)
    expect(likelihoodRatio([10, 0], [10, 10])).toBeCloseTo(2 * 10 * Math.log(2), 12)
    expect(ruleSignificance([5, 5], [10, 10])).toBeCloseTo(1, 12)
    // chi-squared with one degree of freedom: erfc(sqrt(13.86 / 2))
    expect(ruleSignificance([10, 0], [10, 10])).toBeCloseTo(1.9664e-4, 7)
    expect(ruleSignificance([3, 0], [10, 10])).toBeGreaterThan(ruleSignificance([10, 0], [10, 10]))
  })
})

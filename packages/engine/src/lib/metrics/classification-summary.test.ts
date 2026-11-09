import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  classLikelihoodRatios,
  diagnosticOddsRatio,
  negativeLikelihoodRatio,
  positiveLikelihoodRatio,
} from './classification-summary.ts'

describe('classLikelihoodRatios', () => {
  const yTrue = [1, 1, 1, 0, 0, 0, 0, 0]
  const yPred = [1, 1, 0, 1, 0, 0, 0, 0]

  it('are the sensitivity over the false positive rate and the false negative rate over the specificity', () => {
    const { positive, negative } = classLikelihoodRatios(yTrue, yPred)
    expect(positive.estimate.value).toBeCloseTo(2 / 3 / (1 / 5), 12)
    expect(negative.estimate.value).toBeCloseTo(1 / 3 / (4 / 5), 12)
    expect(positive.metric).toBe('positiveLikelihoodRatio')
  })

  it('agree with the single ratios, and their quotient is the diagnostic odds ratio', () => {
    const { positive, negative } = classLikelihoodRatios(yTrue, yPred)
    expect(positive.estimate.value).toBeCloseTo(positiveLikelihoodRatio(yTrue, yPred).estimate.value, 12)
    expect(negative.estimate.value).toBeCloseTo(negativeLikelihoodRatio(yTrue, yPred).estimate.value, 12)
    expect(positive.estimate.value / negative.estimate.value).toBeCloseTo(
      diagnosticOddsRatio(yTrue, yPred).estimate.value,
      12,
    )
  })

  it('ill-defined ratios are NaN, or what replaceUndefinedBy says', () => {
    const perfect = { yTrue: [1, 0, 1, 0], yPred: [1, 0, 1, 0] }
    expect(classLikelihoodRatios(perfect.yTrue, perfect.yPred).positive.estimate.value).toBeNaN()
    expect(classLikelihoodRatios(perfect.yTrue, perfect.yPred).negative.estimate.value).toBe(0)
    expect(classLikelihoodRatios(perfect.yTrue, perfect.yPred, { replaceUndefinedBy: 1 }).positive.estimate.value).toBe(
      1,
    )
    const pair = classLikelihoodRatios(perfect.yTrue, perfect.yPred, {
      replaceUndefinedBy: { positive: 9, negative: 0.5 },
    })
    expect(pair.positive.estimate.value).toBe(9)
    const noNegatives = classLikelihoodRatios([1, 0, 1, 1], [1, 1, 1, 0], {
      replaceUndefinedBy: { positive: 3, negative: 0.5 },
    })
    expect(noNegatives.negative.estimate.value).toBe(0.5)
  })

  it('rejects replacements out of range and non-binary labels', () => {
    expect(() => classLikelihoodRatios(yTrue, yPred, { replaceUndefinedBy: 2 })).toThrow()
    expect(() =>
      classLikelihoodRatios(yTrue, yPred, { replaceUndefinedBy: { positive: 0.5, negative: 0.5 } }),
    ).toThrow()
    expect(() => classLikelihoodRatios([0, 1, 2], [0, 1, 2])).toThrow()
  })

  it('weights act as repetitions', () => {
    const weighted = classLikelihoodRatios([1, 1, 0, 0], [1, 0, 1, 0], { sampleWeight: [3, 1, 1, 2] })
    const repeated = classLikelihoodRatios([1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0])
    expect(weighted.positive.estimate.value).toBeCloseTo(repeated.positive.estimate.value, 12)
    expect(weighted.negative.estimate.value).toBeCloseTo(repeated.negative.estimate.value, 12)
  })

  it('property: LR+ / LR- is the diagnostic odds ratio whenever all four counts are positive', () => {
    const labels = fc.array(fc.tuple(fc.integer({ min: 0, max: 1 }), fc.integer({ min: 0, max: 1 })), {
      minLength: 8,
      maxLength: 40,
    })
    fc.assert(
      fc.property(labels, (pairs) => {
        const t = pairs.map((p) => p[0])
        const p = pairs.map((q) => q[1])
        const count = (a: number, b: number): number => pairs.filter((q) => q[0] === a && q[1] === b).length
        fc.pre([count(1, 1), count(1, 0), count(0, 1), count(0, 0)].every((c) => c > 0))
        const r = classLikelihoodRatios(t, p)
        expect(r.positive.estimate.value / r.negative.estimate.value).toBeCloseTo(
          diagnosticOddsRatio(t, p).estimate.value,
          8,
        )
        expect(r.positive.estimate.value).toBeGreaterThan(0)
      }),
      { numRuns: 60 },
    )
  })
})

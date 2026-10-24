import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

const yTrue = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
const yPred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
const v = (r: { readonly estimate: { readonly value: number } }): number => r.estimate.value

describe('classification rates and their identities', () => {
  it('reads sensitivity as recall, informedness and markedness as sums of rates', () => {
    expect(v(M.sensitivityScore(yTrue, yPred))).toBe(v(M.recallScore(yTrue, yPred)))
    const tpr = v(M.recallScore(yTrue, yPred))
    const tnr = v(M.specificityScore(yTrue, yPred))
    const ppv = v(M.precisionScore(yTrue, yPred))
    const npv = v(M.negativePredictiveValue(yTrue, yPred))
    expect(v(M.informedness(yTrue, yPred))).toBeCloseTo(tpr + tnr - 1, 15)
    expect(v(M.markedness(yTrue, yPred))).toBeCloseTo(ppv + npv - 1, 15)
    expect(v(M.balancedAccuracyScore(yTrue, yPred, { adjusted: true }))).toBeCloseTo(tpr + tnr - 1, 15)
  })

  it('reports per-class and averaged precision, recall, F-score and support together', () => {
    const t = [0, 1, 2, 2, 1, 0, 2, 1]
    const p = [0, 2, 2, 1, 1, 0, 2, 0]
    const r = M.precisionRecallFscoreSupport(t, p, { average: 'macro' })
    expect(r.support).toEqual([2, 3, 3])
    expect(r.averaged?.precision).toBeCloseTo(v(M.precisionScore(t, p, { average: 'macro' })), 15)
    expect(r.averaged?.recall).toBeCloseTo(v(M.recallScore(t, p, { average: 'macro' })), 15)
    expect(r.averaged?.fscore).toBeCloseTo(v(M.f1Score(t, p, { average: 'macro' })), 15)
    const none = M.precisionRecallFscoreSupport(t, p, { average: 'none' })
    expect(none.averaged).toBeUndefined()
    expect(none.recall[0]).toBe(1)
  })

  it('scores multilabel rows by exact match in subset accuracy', () => {
    const t = [
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
    ]
    const p = [
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [0, 1, 1],
    ]
    expect(v(M.subsetAccuracy(t, p))).toBe(0.5)
  })
})

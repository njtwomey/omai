import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import * as M from './index.ts'

const n = 200
const yTrue = Array.from({ length: n }, (_, i) => (Math.sin(i * 1.7) > 0 ? 1 : 0))
const yPred = yTrue.map((y, i) => (Math.cos(i * 2.3) > 0.6 ? 1 - y : y))
const scores = yTrue.map((y, i) => y + Math.sin(i * 0.37) * 1.3)

describe('intervals', () => {
  it('bootstraps reproducibly, around the estimate, with the binomial standard error', () => {
    const acc = M.withInterval(M.accuracyScore, { key: key(1), nResamples: 400 })
    const r = acc(yTrue, yPred)
    const again = acc(yTrue, yPred)
    expect(r.estimate.interval).toEqual(again.estimate.interval)
    const [lo = 0, hi = 0] = r.estimate.interval ?? []
    expect(lo).toBeLessThan(r.estimate.value)
    expect(hi).toBeGreaterThan(r.estimate.value)
    const p = r.estimate.value
    expect(r.estimate.standardError ?? 0).toBeCloseTo(Math.sqrt((p * (1 - p)) / n), 2)
  })

  it('gives basic and BCa intervals near the percentile interval for a smooth statistic', () => {
    const intervals = (['percentile', 'basic', 'bca'] as const).map(
      (bootstrap) =>
        M.withInterval(M.accuracyScore, { key: key(2), nResamples: 300, bootstrap })(yTrue, yPred).estimate.interval ??
        [],
    )
    for (const iv of intervals) {
      expect((iv[1] ?? 0) - (iv[0] ?? 0)).toBeGreaterThan(0.05)
      expect((iv[1] ?? 0) - (iv[0] ?? 0)).toBeLessThan(0.2)
    }
  })

  it('resamples sample weights with the samples', () => {
    const w = yTrue.map((_, i) => 1 + (i % 3))
    const r = M.withInterval(M.f1Score, { key: key(3), nResamples: 100 })(yTrue, yPred, { sampleWeight: w })
    expect(r.estimate.value).toBe(M.f1Score(yTrue, yPred, { sampleWeight: w }).estimate.value)
    expect(r.estimate.method).toContain('bootstrap')
  })

  it('gives DeLong intervals to the ROC AUC only', () => {
    const auc = M.withInterval(M.rocAucScore, { method: 'delong' })(yTrue, scores)
    const { variance } = M.delongVariance(yTrue, scores)
    expect(auc.estimate.standardError).toBeCloseTo(Math.sqrt(variance), 15)
    expect(() => M.withInterval(M.accuracyScore, { method: 'delong' })(yTrue, yPred)).toThrow()
    expect(() => M.withInterval(M.f1Score, { method: 'wilson' })(yTrue, yPred)).toThrow()
  })
})

describe('comparisons and decisions', () => {
  it('runs the paired bootstrap', () => {
    const better = yTrue.map((y, i) => (i % 10 === 0 ? 1 - y : y))
    const r = M.pairedBootstrap(M.accuracyScore, yTrue, better, yPred, { key: key(4), nResamples: 300 })
    expect(r.statistic).toBeCloseTo(
      M.accuracyScore(yTrue, better).estimate.value - M.accuracyScore(yTrue, yPred).estimate.value,
      14,
    )
    expect(r.pValue).toBeLessThan(0.05)
    const same = M.pairedBootstrap(M.accuracyScore, yTrue, yPred, yPred, { nResamples: 50 })
    expect(same.statistic).toBe(0)
    expect(same.pValue).toBe(1)
  })

  it('evaluates many metrics, with shared intervals on request', () => {
    const r = M.evaluate(yTrue, yPred, [M.accuracyScore, M.precisionScore, M.recallScore], {
      interval: { nResamples: 50 },
    })
    expect(Object.keys(r)).toEqual(['accuracyScore', 'precisionScore', 'recallScore'])
    expect(r['recallScore']?.estimate.interval).toHaveLength(2)
  })

  it('makes Bayes decisions and cost thresholds agree', () => {
    const t = M.costThreshold(1, 4)
    const decisions = M.bayesDecision(
      [
        [1 - t + 0.01, t - 0.01],
        [1 - t - 0.01, t + 0.01],
      ],
      [
        [0, 1],
        [4, 0],
      ],
    )
    expect(decisions).toEqual([0, 1])
    expect(
      M.expectedUtility(
        [0, 1],
        [1, 1],
        [
          [1, -1],
          [-1, 1],
        ],
      ).estimate.value,
    ).toBe(0)
  })
})

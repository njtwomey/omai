import { describe, expect, it } from 'vitest'
import { varianceOf } from '../../base/index.ts'
import { key, normal, randint } from '../../random/index.ts'
import { friedman, ttest1samp } from '../../stats/tests/index.ts'
import { asTensor, full, item, mean, toFlat } from '../../tensor/index.ts'
import { defineEstimator } from '../estimator.ts'
import {
  bayesianCorrelatedTTest,
  compareAlgorithms,
  correctedResampledTTest,
  criticalDifference,
  fiveByTwoCvFTest,
  fiveByTwoCvTTest,
  nemenyiQ,
  pairedCvScores,
  resampledTTest,
} from './compare.ts'

const d5 = [0.1, 0.2, 0.15, 0.05, 0.1]

function shuffled(n: number, seed: number): number[] {
  const u = toFlat(randint(key(seed), [n], 0, 1 << 30))
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => (u[a] ?? 0) - (u[b] ?? 0))
}

/** Squared test error of the mean of the training set's first half minus that of its second half. */
function nullDifference(y: number[], train: number[], test: number[]): number {
  const h = train.length / 2
  const m = (rows: number[]): number => rows.reduce((a, i) => a + (y[i] ?? 0), 0) / rows.length
  const a = m(train.slice(0, h))
  const b = m(train.slice(h))
  const err = (c: number): number => test.reduce((acc, i) => acc + ((y[i] ?? 0) - c) ** 2, 0) / test.length
  return err(a) - err(b)
}

describe('resampled t-tests', () => {
  it('the plain test is the one-sample t-test and the corrected one inflates the variance by 1 + k n2 / n1', () => {
    const plain = resampledTTest(d5)
    expect(plain.statistic).toBeCloseTo(ttest1samp(d5).statistic, 12)
    expect(plain.pValue).toBeCloseTo(ttest1samp(d5).pValue, 12)
    const corrected = correctedResampledTTest(d5, { nTrain: 80, nTest: 20 })
    expect(corrected.statistic).toBeCloseTo(plain.statistic / Math.sqrt(1 + (5 * 20) / 80), 12)
    expect(corrected.pValue).toBeGreaterThan(plain.pValue)
    expect(corrected.df).toBe(4)
  })

  it('one-sided alternatives split the two-sided p-value', () => {
    const two = correctedResampledTTest(d5, { nTrain: 80, nTest: 20 })
    const gt = correctedResampledTTest(d5, { nTrain: 80, nTest: 20, alternative: 'greater' })
    const lt = correctedResampledTTest(d5, { nTrain: 80, nTest: 20, alternative: 'less' })
    expect(gt.pValue + lt.pValue).toBeCloseTo(1, 12)
    expect(two.pValue).toBeCloseTo(2 * gt.pValue, 12)
  })

  it('rejects bad inputs', () => {
    expect(() => correctedResampledTTest([1], { nTrain: 8, nTest: 2 })).toThrow()
    expect(() => correctedResampledTTest(d5, { nTrain: 0, nTest: 2 })).toThrow()
  })

  it('has near-nominal type I error under the null in simulation where the plain test rejects far too often', () => {
    // Two classifiers with equal expected accuracy 1/2 on noise labels: the training majority class, and always '+1'.
    // Their scores on random splits are correlated through the shared data, which is what the correction accounts for.
    const N = 40
    const n1 = 30
    const n2 = N - n1
    const splits = 10
    let plain = 0
    let corrected = 0
    const experiments = 400
    for (let e = 0; e < experiments; e++) {
      const labels = Array.from(toFlat(randint(key(1000 + e), [N], 0, 2)))
      const diffs = Array.from({ length: splits }, (_, s) => {
        const order = shuffled(N, 100000 + e * splits + s)
        const train = order.slice(0, n1)
        const test = order.slice(n1)
        const ones = train.reduce((acc, i) => acc + (labels[i] ?? 0), 0)
        const majority = ones * 2 >= n1 ? 1 : 0
        const accA = test.filter((i) => labels[i] === majority).length / n2
        const accB = test.filter((i) => labels[i] === 1).length / n2
        return accA - accB
      })
      if (varianceOf(diffs, 1) === 0) continue
      if (resampledTTest(diffs).pValue < 0.05) plain++
      if (correctedResampledTTest(diffs, { nTrain: n1, nTest: n2 }).pValue < 0.05) corrected++
    }
    expect(plain / experiments).toBeGreaterThan(0.15)
    expect(corrected / experiments).toBeLessThan(0.08)
  })
})

describe('5x2 cv tests', () => {
  const p = [0.1, 0.12, 0.08, 0.1, 0.11, 0.09, 0.12, 0.1, 0.1, 0.1]
  it('t is the first difference over the root mean replication variance', () => {
    let s = 0
    for (let i = 0; i < 5; i++) {
      const a = p[2 * i] as number
      const b = p[2 * i + 1] as number
      const m = (a + b) / 2
      s += (a - m) ** 2 + (b - m) ** 2
    }
    const r = fiveByTwoCvTTest(p)
    expect(r.statistic).toBeCloseTo(0.1 / Math.sqrt(s / 5), 12)
    expect(r.df).toBe(5)
  })

  it('F is the sum of squares over twice the within-replication variance, on (10, 5) degrees of freedom', () => {
    let num = 0
    let den = 0
    for (let i = 0; i < 5; i++) {
      const a = p[2 * i] as number
      const b = p[2 * i + 1] as number
      num += a * a + b * b
      den += ((a - b) / 2) ** 2 * 2
    }
    const r = fiveByTwoCvFTest(p)
    expect(r.statistic).toBeCloseTo(num / (2 * den), 12)
    expect(r.df).toEqual([10, 5])
    expect(r.pValue).toBeLessThan(0.001)
  })

  it('rejects the wrong number of differences', () => {
    expect(() => fiveByTwoCvFTest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])).toThrow()
  })

  it('has near-nominal type I error under the null in simulation', () => {
    const N = 60
    let f = 0
    let t = 0
    const experiments = 600
    for (let e = 0; e < experiments; e++) {
      const y = toFlat(normal(key(7000 + e), [N]))
      const diffs: number[] = []
      for (let r = 0; r < 5; r++) {
        const order = shuffled(N, 300000 + e * 5 + r)
        const halves = [order.slice(0, N / 2), order.slice(N / 2)]
        for (const [train, test] of [
          [halves[0] ?? [], halves[1] ?? []],
          [halves[1] ?? [], halves[0] ?? []],
        ] as const)
          diffs.push(nullDifference(y, train, test))
      }
      if (fiveByTwoCvFTest(diffs).pValue < 0.05) f++
      if (fiveByTwoCvTTest(diffs).pValue < 0.05) t++
    }
    expect(f / experiments).toBeLessThan(0.09)
    expect(t / experiments).toBeLessThan(0.09)
    expect(f / experiments).toBeGreaterThan(0.01)
  })
})

describe('Bayesian correlated t-test', () => {
  it('has the Student-t posterior with the correlation-inflated scale and probabilities that sum to one', () => {
    const r = bayesianCorrelatedTTest(d5, { rho: 0.2, rope: 0.02 })
    expect(r.df).toBe(4)
    expect(r.scale).toBeCloseTo(Math.sqrt((1 / 5 + 0.25) * varianceOf(d5, 1)), 12)
    expect(r.left + r.rope + r.right).toBeCloseTo(1, 12)
    expect(r.interval[0]).toBeLessThan(r.mean)
    expect(r.interval[1]).toBeGreaterThan(r.mean)
  })

  it('is symmetric for a symmetric sample and shifts with the mean', () => {
    const sym = bayesianCorrelatedTTest([-0.1, 0.1, -0.05, 0.05, 0], { rope: 0 })
    expect(sym.left).toBeCloseTo(0.5, 12)
    expect(sym.right).toBeCloseTo(0.5, 12)
    const up = bayesianCorrelatedTTest([0.1, 0.12, 0.09, 0.11, 0.1, 0.13], { rope: 0.02 })
    expect(up.right).toBeGreaterThan(0.99)
  })

  it('puts the mass in the ROPE when the difference is practically zero', () => {
    const r = bayesianCorrelatedTTest([0.001, -0.002, 0.0, 0.001, -0.001, 0.002], { rope: 0.01, rho: 0.1 })
    expect(r.rope).toBeGreaterThan(0.95)
  })
})

describe('Demsar procedures', () => {
  it('reproduces the critical values q_alpha of Demsar (2006), table 5', () => {
    const published: Record<number, number> = {
      2: 1.96,
      3: 2.343,
      4: 2.569,
      5: 2.728,
      6: 2.85,
      7: 2.949,
      8: 3.031,
      9: 3.102,
      10: 3.164,
    }
    for (const [k, q] of Object.entries(published)) expect(Math.abs(nemenyiQ(Number(k)) - q)).toBeLessThan(1.5e-3)
  })

  it('computes the critical difference of the paper example (k = 4, N = 14: 1.25)', () => {
    expect(criticalDifference(4, 14)).toBeCloseTo(1.2537, 3)
  })

  const scores = [
    [0.9, 0.8, 0.7, 0.5],
    [0.85, 0.8, 0.6, 0.55],
    [0.95, 0.7, 0.75, 0.5],
    [0.9, 0.85, 0.7, 0.6],
    [0.8, 0.75, 0.6, 0.65],
    [0.92, 0.81, 0.7, 0.52],
    [0.88, 0.79, 0.66, 0.51],
    [0.91, 0.8, 0.74, 0.58],
  ]
  it('ranks, tests and draws cliques', () => {
    const r = compareAlgorithms(scores, { names: ['a', 'b', 'c', 'd'] })
    expect(r.averageRanks[0]).toBeCloseTo(1, 12)
    expect(r.averageRanks[1]).toBeCloseTo(17 / 8, 12)
    expect(r.friedman.statistic).toBeCloseTo(friedman(scores).statistic, 12)
    const n = 8
    const k = 4
    const chi2 = r.friedman.statistic
    expect(r.imanDavenport.statistic).toBeCloseTo(((n - 1) * chi2) / (n * (k - 1) - chi2), 12)
    expect(r.criticalDifference).toBeCloseTo(criticalDifference(4, 8), 12)
    // Mean ranks 1, 2, ~3.1, ~3.9: the CD (about 1.66) joins a, b; and b, c; and c, d but not a with c.
    for (const c of r.cliques) expect(c.includes(0) && c.includes(2)).toBe(false)
    expect(r.cliques.length).toBeGreaterThan(0)
    expect(r.control.control).toBe(0)
    expect(r.control.holm[3]).toBeLessThan(0.05)
    expect(r.control.bonferroniDunn[3]).toBeGreaterThanOrEqual(r.control.holm[3] as number)
  })

  it('respects the direction of the scores', () => {
    const lower = compareAlgorithms(scores, { higherIsBetter: false })
    expect(lower.averageRanks[0]).toBeCloseTo(4, 12)
  })
})

describe('pairedCvScores', () => {
  const meanModel = defineEstimator(
    {
      name: 'mean',
      kind: 'regressor',
      fit: (d) => ({ m: item(mean(d.y as never)) }),
      methods: (p) => ({ predict: (x) => full([asTensor(x).shape[0] as number], p.m) }),
    },
    {},
  )
  const zero = defineEstimator(
    {
      name: 'zero',
      kind: 'regressor',
      fit: () => ({}),
      methods: () => ({ predict: (x) => full([asTensor(x).shape[0] as number], 0) }),
    },
    {},
  )
  const data = { x: Array.from({ length: 20 }, (_, i) => [i]), y: Array.from({ length: 20 }, (_, i) => 5 + (i % 3)) }
  it('scores both estimators on the same ten 5x2 folds', () => {
    const r = pairedCvScores(meanModel, zero, data, { scoring: 'neg_mean_absolute_error' })
    expect(r.differences.length).toBe(10)
    expect(r.nTrain).toBe(10)
    expect(r.nTest).toBe(10)
    expect(r.differences.every((v) => v > 0)).toBe(true)
    expect(fiveByTwoCvFTest(r.differences).pValue).toBeLessThan(0.01)
  })
})

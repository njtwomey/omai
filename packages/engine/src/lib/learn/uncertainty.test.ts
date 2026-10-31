import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares } from '../../testing/learn.ts'
import { dropout } from '../nn/index.ts'
import { key, normal } from '../random/index.ts'
import { mul, toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  deepEnsemble,
  deepEnsembleAlgorithm,
  deepEnsemblePredict,
  disagreement,
  ensembleUncertainty,
  mcDropoutAlgorithm,
  mcDropoutPredict,
  monteCarloDropout,
} from './uncertainty.ts'

describe('uncertainty', () => {
  it('decomposes regression deep ensemble uncertainty into epistemic and aleatoric variance', () => {
    const u = deepEnsemblePredict(
      [
        [1, 4],
        [3, 4],
      ],
      {
        memberVariances: [
          [0.5, 0.25],
          [1.5, 0.25],
        ],
      },
    )
    expect(toFlat(u.mean)).toEqual([2, 4])
    expect(toFlat(u.epistemicVariance)).toEqual([1, 0])
    expect(toFlat(u.aleatoricVariance)).toEqual([1, 0.25])
    expect(toFlat(u.totalVariance)).toEqual([2, 0.25])
    expect(toFlat(u.mutualInformation)[0]).toBeGreaterThan(0)
    expect(toFlat(u.mutualInformation)[1]).toBeCloseTo(0, 12)
  })

  it('decomposes classification deep ensemble probabilities into predictive entropy and BALD mutual information', () => {
    const m1 = [
      [0.95, 0.05],
      [0.5, 0.5],
    ]
    const m2 = [
      [0.05, 0.95],
      [0.5, 0.5],
    ]
    const u = deepEnsemblePredict([m1, m2])
    expect(toFlat(u.mean)).toEqual([0.5, 0.5, 0.5, 0.5])
    expect(toFlat(u.predictiveEntropy)[0]).toBeCloseTo(Math.LN2, 12)
    expect(toFlat(u.predictiveEntropy)[1]).toBeCloseTo(Math.LN2, 12)
    // Sample 0 has high epistemic disagreement; sample 1 is pure aleatoric uncertainty.
    expect(toFlat(u.mutualInformation)[0]).toBeGreaterThan(0.45)
    expect(toFlat(u.mutualInformation)[1]).toBeCloseTo(0, 12)
    expect(disagreement([m1, m2])[0]).toBeGreaterThan(0.45)
    expect(disagreement([m1, m2])[1]).toBeCloseTo(0, 12)
  })

  it('leaves the Gaussian entropy terms undefined for regression members without variances', () => {
    const u = deepEnsemblePredict([
      [1, 2],
      [3, 2],
    ])
    expect(toFlat(u.aleatoricVariance)).toEqual([0, 0])
    expect(toFlat(u.expectedEntropy).every(Number.isNaN)).toBe(true)
    expect(toFlat(u.mutualInformation).every(Number.isNaN)).toBe(true)
    expect(toFlat(u.disagreement)).toEqual([1, 0])
    // With the members' variances the information is the Gaussian excess entropy, 1/2 ln(total / geometric mean).
    const v = deepEnsemblePredict([[1], [3]], { memberVariances: [[0.5], [1.5]] })
    expect(toFlat(v.mutualInformation)[0]).toBeCloseTo(0.5 * Math.log(2) - 0.25 * Math.log(0.5 * 1.5), 12)
  })

  it('evaluates fitted estimators in ensembleUncertainty', () => {
    const f1 = leastSquares().fit({ x: [[0], [1], [2]], y: [0, 1, 2] })
    const f2 = leastSquares().fit({ x: [[0], [1], [2]], y: [0, 2, 4] })
    const reg = ensembleUncertainty([f1, f2], [[0], [2]])
    expect(toFlat(reg.mean)[0]).toBeCloseTo(0, 10)
    expect(toFlat(reg.mean)[1]).toBeCloseTo(3, 10)
    expect(toFlat(reg.epistemicVariance)[1]).toBeCloseTo(1, 10)

    const c1 = gaussianNB().fit({ x: [[-2], [-1], [1], [2]], y: [0, 0, 1, 1] })
    const c2 = gaussianNB().fit({ x: [[-3], [-2], [2], [3]], y: [0, 0, 1, 1] })
    const cls = ensembleUncertainty([c1, c2], [[0], [5]])
    expect(cls.mean.shape).toEqual([2, 2])
  })

  it('fits a deep ensemble one member per step, each under its own key, with spread growing away from the data', () => {
    const x = Array.from({ length: 40 }, (_, i) => [(i - 20) / 20])
    const noise = toFlat(normal(key(4), [40]))
    const y = x.map(([v = 0], i) => 2 * v + 0.3 * (noise[i] ?? 0))
    const rec = recorder()
    const ens = deepEnsemble(leastSquares(), { x, y }, { members: 12, bootstrap: true, key: key(9), tracer: rec })
    expect(ens.members.length).toBe(12)
    const u = ens.predict([[0], [0.5], [8]])
    const spread = toFlat(u.epistemicVariance)
    expect(spread[2]).toBeGreaterThan(5 * (spread[0] ?? 0))
    expect(spread[2]).toBeGreaterThan(spread[1] ?? 0)
    expect(toFlat(u.mean)[2]).toBeGreaterThan(14)
    expect(toFlat(u.mean)[2]).toBeLessThan(18)

    // The same key gives the same members; another key gives others.
    const again = deepEnsemble(leastSquares(), { x, y }, { members: 12, bootstrap: true, key: key(9) })
    expect(toFlat(again.predict([[8]]).mean)).toEqual(toFlat(u.mean).slice(2))
    const other = deepEnsemble(leastSquares(), { x, y }, { members: 12, bootstrap: true, key: key(10) })
    expect(toFlat(other.predict([[8]]).mean)[0]).not.toBe(toFlat(u.mean)[2])

    // Without a bootstrap a deterministic estimator gives identical members and no spread.
    const same = deepEnsemble(leastSquares(), { x, y }, { members: 3, key: key(1) })
    expect(toFlat(same.predict([[8]]).epistemicVariance)[0]).toBeCloseTo(0, 20)

    const alg = deepEnsembleAlgorithm(
      leastSquares(),
      { x, y },
      { members: 4, bootstrap: true, key: key(2), evaluation: [[0], [8]] },
    )
    const run = runAlgorithm(alg, { maxSteps: 3 })
    expect(run.state.members.length).toBe(3)
    expect(run.state.disagreement).toBeGreaterThan(0)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(run.state))))
    expect(back?.members.length).toBe(3)
    expect(back?.members[0]?.estimator).toBe('least-squares')
  })

  it('Monte Carlo dropout of a network with the mask kept on matches the analytic mean and variance', () => {
    // y = sum_j w_j h_j m_j / (1 - p) with Bernoulli(1 - p) masks m: mean sum w h, variance p / (1 - p) sum (w h)^2.
    const w = [1, -2, 0.5, 3, -1, 2]
    const h = [0.5, 1, 2, 0.3, 1.5, 0.8]
    const p = 0.4
    const out = mcDropoutPredict(
      (k) => [toFlat(mul(dropout(h, p, { key: k, training: true }), w)).reduce((a, b) => a + b, 0)],
      { passes: 4000, key: key(21) },
    )
    const mean = w.reduce((a, wj, j) => a + wj * (h[j] ?? 0), 0)
    const variance = (p / (1 - p)) * w.reduce((a, wj, j) => a + (wj * (h[j] ?? 0)) ** 2, 0)
    // Four standard errors of the mean of 4000 passes.
    expect(Math.abs((toFlat(out.mean)[0] ?? 0) - mean)).toBeLessThan(4 * Math.sqrt(variance / 4000))
    expect(Math.abs((toFlat(out.epistemicVariance)[0] ?? 0) / variance - 1)).toBeLessThan(0.08)
  })

  it('runs Monte Carlo dropout passes with derived keys and serialises stepped state', () => {
    const rec = recorder()
    const out = mcDropoutPredict(
      (k) => ({
        mean: normal(k, [3], { loc: 2, scale: 0.5 }),
        variance: [0.1, 0.2, 0.3],
      }),
      { passes: 8, key: key(11), tracer: rec },
    )
    expect(out.samples.length).toBe(8)
    expect(toFlat(out.aleatoricVariance)[0]).toBeCloseTo(0.1, 12)
    expect(toFlat(out.aleatoricVariance)[2]).toBeCloseTo(0.3, 12)
    expect(toFlat(out.totalVariance)[0]).toBeCloseTo(
      (toFlat(out.epistemicVariance)[0] ?? 0) + (toFlat(out.aleatoricVariance)[0] ?? 0),
      12,
    )

    const alg = mcDropoutAlgorithm((k) => normal(k, [2]), { passes: 4, key: key(3) })
    const run = runAlgorithm(alg, { maxSteps: 4, key: key(3) })
    const json = alg.serialise?.(run.state)
    expect(json).toBeDefined()
    const back = alg.deserialise?.(json as NonNullable<typeof json>)
    expect(back?.t).toBe(4)
    expect(monteCarloDropout(() => [1, 2], { passes: 2, key: key(0) }).samples.length).toBe(2)
  })
})

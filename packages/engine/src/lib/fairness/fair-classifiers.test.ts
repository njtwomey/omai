import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { biasedCredit } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  adversarialDebiasing,
  fairConstrainedLogistic,
  gerryFairClassifier,
  linearProbabilities,
  metaFairClassifier,
  prejudiceRemover,
} from './fair-classifiers.ts'
import { demographicParityDifference } from './group-metrics.ts'

const { data } = generate(biasedCredit, { key: key(8), knobs: { n: 400, labelBias: 0.4, proxyStrength: 2 } })
const flat = toFlat(data.x)
const rows = Array.from({ length: 400 }, (_, i) => [flat[4 * i] ?? 0, flat[4 * i + 1] ?? 0, flat[4 * i + 2] ?? 0])
const groups = Array.from({ length: 400 }, (_, i) => flat[4 * i + 3] ?? 0)
const labels = Array.from(toFlat(data.y))
const gap = (scores: readonly number[]): number =>
  demographicParityDifference(
    labels,
    scores.map((s) => (s >= 0.5 ? 1 : 0)),
    groups,
  ).estimate.value

describe('fair classifiers', () => {
  it('the prejudice remover lowers its prejudice index and the selection gap as eta grows', () => {
    const scoresOf = (eta: number): number[] => {
      const r = prejudiceRemover(rows, labels, groups, { eta })
      return rows.map(
        (x, i) => linearProbabilities(r.models[groups[i] ?? 0] ?? { intercept: 0, weights: [] }, [x])[0] ?? 0,
      )
    }
    const a = prejudiceRemover(rows, labels, groups, { eta: 0 })
    const b = prejudiceRemover(rows, labels, groups, { eta: 30 })
    expect(b.prejudiceIndex).toBeLessThan(a.prejudiceIndex)
    expect(gap(scoresOf(30))).toBeLessThan(gap(scoresOf(0)))
  })

  it('the covariance constraint holds and narrows the gap', () => {
    const r = fairConstrainedLogistic(rows, labels, groups)
    expect(Math.abs(r.covariance)).toBeLessThan(1e-6)
    expect(Math.abs(r.unconstrainedCovariance)).toBeGreaterThan(0.01)
    expect(gap(linearProbabilities(r.model, rows))).toBeLessThan(0.2)
  })

  it('adversarial debiasing leaves less of the group in the predictor output', () => {
    // The predictor sees the group itself, so alone it uses it.
    const withGroup = rows.map((r, i) => [...r, groups[i] ?? 0])
    const plain = adversarialDebiasing(withGroup, labels, groups, { adversaryWeight: 0, steps: 300, key: key(1) })
    const fair = adversarialDebiasing(withGroup, labels, groups, { adversaryWeight: 1, steps: 300, key: key(1) })
    // The correlation between the predictor's logit and the group.
    const corr = (m: { intercept: number; weights: readonly number[] }): number => {
      const z = withGroup.map((x) => m.intercept + x.reduce((s, v, j) => s + v * (m.weights[j] ?? 0), 0))
      const mz = z.reduce((a, b) => a + b, 0) / z.length
      const mg = groups.reduce((a, b) => a + b, 0) / groups.length
      let c = 0
      let vz = 0
      let vg = 0
      for (const [i, v] of z.entries()) {
        c += (v - mz) * ((groups[i] ?? 0) - mg)
        vz += (v - mz) ** 2
        vg += ((groups[i] ?? 0) - mg) ** 2
      }
      return Math.abs(c / Math.sqrt(vz * vg))
    }
    expect(corr(fair.model)).toBeLessThan(corr(plain.model) / 3)
    expect(fair.steps.every((s) => Number.isFinite(s.predictorLoss) && Number.isFinite(s.adversaryLoss))).toBe(true)
  })

  it('the meta fair classifier meets tau and gives up accuracy only as tau tightens', () => {
    const post = rows.map((x) => 1 / (1 + Math.exp(-((x[0] ?? 0) + (x[1] ?? 0)))))
    const loose = metaFairClassifier(post, groups, { tau: 0.5 })
    const tight = metaFairClassifier(post, groups, { tau: 0.95 })
    expect(tight.ratio).toBeGreaterThanOrEqual(0.95 - 1e-9)
    expect(tight.expectedAccuracy).toBeLessThanOrEqual(loose.expectedAccuracy + 1e-12)
    expect(() => metaFairClassifier(post, groups, { tau: 0 })).toThrow(/tau/)
  })

  it('GerryFair averages its classifiers and records the audit each round', () => {
    const r = gerryFairClassifier(
      rows,
      groups.map((g) => [g]),
      labels,
      { iterations: 8 },
    )
    expect(r.classifiers).toHaveLength(7)
    expect(r.decisions.every((p) => p >= 0 && p <= 1)).toBe(true)
    expect(r.steps.every((s) => s.violation >= 0 && s.error >= 0 && s.error <= 1)).toBe(true)
  })
})

import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { fromStorage, type Tensor } from '../../tensor/index.ts'
import { anchors, globalSurrogate, lime } from './surrogates.ts'

const rule = (x: Tensor): Tensor => {
  const d = x.data as Float64Array
  const n = x.shape[0] ?? 0
  return fromStorage(
    Float64Array.from({ length: n }, (_, i) => ((d[3 * i] ?? 0) > 0 && (d[3 * i + 1] ?? 0) > 0 ? 1 : 0)),
    [n],
    'float64',
  )
}
const train = Array.from({ length: 200 }, (_, i) => [Math.sin(1.3 * i) * 3, Math.cos(0.7 * i) * 3, ((i * 37) % 11) - 5])

describe('surrogates', () => {
  it('anchors find the conjunction a rule-based model uses, with the requested precision', () => {
    const a = anchors(rule, [2, 2, 0], train, { key: key(1) })
    expect(a.predicates.map((p) => p.feature).sort()).toEqual([0, 1])
    expect(a.precisionLower).toBeGreaterThanOrEqual(0.95)
    expect(a.coverage).toBeGreaterThan(0)
    // Empirical check: points inside the anchor's box get the same prediction.
    const inside = train.filter((r) =>
      a.predicates.every((p) => (r[p.feature] ?? 0) > p.lower && (r[p.feature] ?? 0) <= p.upper),
    )
    const agree = inside.filter(
      (r) => (rule(fromStorage(Float64Array.from(r), [1, 3], 'float64')).data[0] ?? 0) === 1,
    ).length
    expect(agree / inside.length).toBeGreaterThanOrEqual(0.9)
  })

  it('LIME recovers a linear model locally (in standardised units)', () => {
    const linear = (x: Tensor): Tensor => {
      const d = x.data as Float64Array
      const n = x.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => 2 * (d[3 * i] ?? 0) - (d[3 * i + 2] ?? 0)),
        [n],
        'float64',
      )
    }
    const r = lime(linear, [0.5, 0.5, 0.5], train, { nSamples: 500, featureSelection: 'none', alpha: 0, key: key(2) })
    const w = new Map(r.weights)
    const sd = (j: number) => {
      const c = train.map((row) => row[j] ?? 0)
      const m = c.reduce((a, b) => a + b, 0) / c.length
      return Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length)
    }
    expect(w.get(0) ?? 0).toBeCloseTo(2 * sd(0), 6)
    expect(w.get(2) ?? 0).toBeCloseTo(-sd(2), 6)
    expect(r.score).toBeCloseTo(1, 9)
  })

  it('a global surrogate tree of a tree-shaped model is exact', () => {
    const g = globalSurrogate(rule, train, { maxDepth: 2, task: 'classification' })
    expect(g.fidelity).toBe(1)
  })
})

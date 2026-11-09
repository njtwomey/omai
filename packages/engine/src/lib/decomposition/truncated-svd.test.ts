import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { pca } from './pca.ts'
import { truncatedSvd, truncatedSvdAlgorithm } from './truncated-svd.ts'

const x = Array.from({ length: 30 }, (_, i) => [
  10 + Math.sin(i),
  10 + 0.5 * Math.sin(i) + 0.1 * Math.cos(3 * i),
  0.2 * Math.cos(2 * i),
])

describe('truncatedSvd', () => {
  it('does not centre: the first component follows the mean, unlike PCA', () => {
    const t = truncatedSvd.with({ nComponents: 1, solver: 'full' }).fit({ x })
    const p = pca.with({ nComponents: 1 }).fit({ x })
    const c = t.params.components[0] ?? []
    expect(Math.abs(c[0] ?? 0)).toBeGreaterThan(0.6)
    expect(Math.abs(c[1] ?? 0)).toBeGreaterThan(0.6)
    expect(t.params.explainedVarianceRatio[0] ?? 0).toBeGreaterThan(0)
    const cosine = Math.abs(c.reduce((a, v, j) => a + v * (p.params.components[0]?.[j] ?? 0), 0))
    expect(cosine).toBeLessThan(0.9999)
  })

  it('the scores are U Sigma: their norms are the singular values, and the best rank-k approximation error is the discarded tail (Eckart-Young)', () => {
    const full = truncatedSvd.with({ nComponents: 3, solver: 'full' }).fit({ x })
    const two = truncatedSvd.with({ nComponents: 2, solver: 'full' }).fit({ x })
    const scores = toArray(two.transform(x)) as number[][]
    for (let c = 0; c < 2; c++)
      expect(Math.sqrt(scores.reduce((a, r) => a + (r[c] ?? 0) ** 2, 0))).toBeCloseTo(
        two.params.singularValues[c] ?? 0,
        8,
      )
    const back = toArray(two.inverseTransform(two.transform(x))) as number[][]
    const err = back.reduce((a, r, i) => a + r.reduce((b, v, j) => b + (v - (x[i]?.[j] ?? 0)) ** 2, 0), 0)
    expect(Math.sqrt(err)).toBeCloseTo(full.params.singularValues[2] ?? 0, 8)
  })

  it('the randomised solver finds the exact subspace of a low-rank matrix, the same for the same key', () => {
    const low = Array.from({ length: 40 }, (_, i) => [
      1 + (i % 7),
      2 * (1 + (i % 7)) + (i % 3),
      (i % 3) + 0.5,
      3 * (i % 3),
    ])
    const exact = truncatedSvd.with({ nComponents: 2, solver: 'full' }).fit({ x: low })
    const rand = truncatedSvd.with({ nComponents: 2, solver: 'randomized' }).fit({ x: low }, { key: key(3) })
    for (let c = 0; c < 2; c++) {
      expect(rand.params.singularValues[c]).toBeCloseTo(exact.params.singularValues[c] ?? 0, 6)
      for (const [j, v] of (exact.params.components[c] ?? []).entries())
        expect(rand.params.components[c]?.[j]).toBeCloseTo(v, 5)
    }
    expect(truncatedSvd.with({ nComponents: 2 }).fit({ x: low }, { key: key(3) }).params).toEqual(rand.params)
  })

  it('refuses a bad number of components, serialises its state, and a session probes the embedding', () => {
    expect(() => truncatedSvdAlgorithm(x, 4)).toThrow()
    expect(() => truncatedSvdAlgorithm(x, 0)).toThrow()
    const end = runAlgorithm(truncatedSvdAlgorithm(x, 2, { solver: 'full' }), { maxSteps: 2 }).state
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(end))))).toEqual(end)
    const s = truncatedSvd.with({ nComponents: 2, solver: 'full' }).session({ x }, { logs: 'all' })
    s.advance({ steps: 1 })
    expect((s.probe(1, 'model/embedding') as number[][]).length).toBe(30)
  })
})

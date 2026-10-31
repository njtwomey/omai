import { describe, expect, it } from 'vitest'
import { rbfKernel, scaleKernel } from '../kernels/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { logMarginalLikelihood } from './likelihood.ts'
import { sparseBound, sparseGp } from './sparse.ts'

const x = Array.from({ length: 36 }, (_, i) => [i / 6])
const y = x.map(([v], i) => Math.sin(v as number) + 0.1 * Math.sin(23 * i))
const data = { x, y }

describe('sparseBound', () => {
  it('is the exact log marginal likelihood when every training point induces', () => {
    const k = rbfKernel({ lengthScale: 0.8 })
    const exact = logMarginalLikelihood(k, x, y, { alpha: 0.05 }).value
    for (const a of ['sor', 'dtc', 'fitc', 'vfe'] as const)
      expect(sparseBound(a, k, x, y, x, 0.05, 1e-9)).toBeCloseTo(exact, 4)
  })

  it('is a lower bound on the exact likelihood for the variational approximation, for any inducing points', () => {
    const k = rbfKernel({ lengthScale: 0.8 })
    const exact = logMarginalLikelihood(k, x, y, { alpha: 0.05 }).value
    for (const m of [3, 6, 12]) {
      const z = x.filter((_, i) => i % Math.floor(36 / m) === 0)
      expect(sparseBound('vfe', k, x, y, z, 0.05)).toBeLessThanOrEqual(exact + 1e-9)
    }
  })

  it('tightens as inducing points are added', () => {
    const k = rbfKernel({ lengthScale: 0.8 })
    const bound = (m: number): number =>
      sparseBound(
        'vfe',
        k,
        x,
        y,
        x.filter((_, i) => i % Math.floor(36 / m) === 0),
        0.05,
      )
    expect(bound(12)).toBeGreaterThan(bound(6))
    expect(bound(6)).toBeGreaterThan(bound(3))
  })
})

describe('sparseGp', () => {
  it('runs as a session: the bound climbs, the inducing points move, and the probe is a posterior grid', () => {
    const est = sparseGp.with({ inducing: 5, kernel: scaleKernel(rbfKernel({ lengthScale: 1 }), 1) })
    const s = est.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 2 })
    expect(toFlat(s.fittedAt(2).predict([[1]])).length).toBe(1)
    const grid = s.probe(2, 'model/posterior', { resolution: 9 }) as { kind: string; mean: number[] }
    expect(grid.kind).toBe('posteriorGrid')
    expect(grid.mean).toHaveLength(9)
    s.advance()
    expect(s.status).toBe('converged')
    const bound = metricSeries(s.run(), 'gp/bound').values
    expect(bound[bound.length - 1] as number).toBeGreaterThan(bound[0] as number)
    const before = s.fittedAt(0).params.z
    const after = s.fittedAt(s.t).params.z
    expect(after).not.toEqual(before)
  })

  it('keeps the inducing points where they start when told to', () => {
    const fitted = sparseGp.with({ inducing: 5, optimiseInducing: false }).fit(data, { explain: false, key: key(2) })
    const start = sparseGp.with({ inducing: 5, optimise: false }).fit(data, { explain: false, key: key(2) })
    expect(fitted.params.z).toEqual(start.params.z)
  })

  it('fits the curve with few points, with VFE close to the exact process', () => {
    const fitted = sparseGp.with({ inducing: 8, approximation: 'vfe' }).fit(data, { explain: false })
    expect(fitted.score(data)).toBeGreaterThan(0.95)
    const { std } = fitted.predictWithStd([[3], [30]])
    expect(toFlat(std)[1] as number).toBeGreaterThan(toFlat(std)[0] as number)
  })

  it('reverts to the prior away from the data under FITC and DTC, and collapses to zero variance under SoR', () => {
    const base = { inducing: 6, optimise: false, kernel: rbfKernel({ lengthScale: 0.7 }), noise: 0.05 }
    const far = [[40]]
    const std = (a: 'sor' | 'dtc' | 'fitc'): number =>
      toFlat(
        sparseGp
          .with({ ...base, approximation: a })
          .fit(data, { explain: false })
          .predictWithStd(far).std,
      )[0] as number
    expect(std('sor')).toBeLessThan(1e-3)
    expect(std('dtc')).toBeCloseTo(1, 2)
    expect(std('fitc')).toBeCloseTo(1, 2)
  })

  it('revives from JSON and restarts from other inducing subsets', () => {
    const est = sparseGp.with({ inducing: 5 })
    const fitted = est.fit(data, { explain: false })
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predict([[2]]))).toEqual(toFlat(fitted.predict([[2]])))
    const more = est.with({ restarts: 2 }).fit(data, { explain: false, key: key(5) })
    expect(more.params.bound).toBeGreaterThanOrEqual(est.fit(data, { explain: false, key: key(5) }).params.bound - 1e-9)
  })
})

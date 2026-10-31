import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { cca, plsAlgorithm, plsCanonical, plsRegression, plsSvd } from './cross-decomposition.ts'

const noise = (i: number, j: number): number => Math.sin(i * 12.9898 + j * 78.233) * 0.3
const x = Array.from({ length: 40 }, (_, i) => [
  Math.sin(i),
  Math.cos(i * 0.7),
  (i % 5) - 2,
  Math.sin(i * 0.3) + noise(i, 3),
])
const y = x.map((r, i) => [(r[0] ?? 0) * 2 - (r[2] ?? 0) + noise(i, 0), (r[1] ?? 0) + 0.5 * (r[3] ?? 0) + noise(i, 1)])

describe('cross-decomposition', () => {
  it('PLS regression predicts, one latent direction per step, with R^2 rising', () => {
    const f = plsRegression.with({ nComponents: 2 }).fit({ x, y })
    expect(f.score?.({ x, y })).toBeGreaterThan(0.8)
    const s = plsRegression.with({ nComponents: 2 }).session({ x, y }, { key: key(0), logs: 'all' })
    s.advance({ steps: 1 })
    const one = s.fittedAt(1).score?.({ x, y }) ?? 0
    expect(one).toBeLessThan(f.score?.({ x, y }) ?? 0)
    expect(s.fittedAt(1).params.xWeights[0]?.length).toBe(1)
  })

  it('CCA finds correlations in decreasing order; canonical and SVD variants transform', () => {
    const f = cca.with({ nComponents: 2 }).fit({ x, y }, { logs: 'all' })
    expect(f.params.correlations[0] ?? 0).toBeGreaterThan(f.params.correlations[1] ?? 0)
    expect(metricSeries(f.run, 'solver/correlation').values.length).toBe(2)
    expect(plsCanonical.with({ nComponents: 2 }).fit({ x, y }).transform(x).shape).toEqual([40, 2])
    expect(plsSvd({ nComponents: 2 }).fit({ x, y }).transform(x).shape).toEqual([40, 2])
    expect(() => plsSvd({ nComponents: 9 }).fit({ x, y })).toThrow()
  })

  it('states round-trip', () => {
    const state = runAlgorithm(plsAlgorithm(x, y, 'canonical', 2), { maxSteps: 2 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

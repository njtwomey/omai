import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { ardAlgorithm, ardCoefficients, ardRegression } from './ard.ts'
import { designOf } from './design.ts'

function problem(n: number, d: number, noise: number): { x: number[][]; y: number[] } {
  const draws = toFlat(normal(key(11), [n * (d + 1)]))
  const x = Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => draws[i * d + j] ?? 0))
  const y = x.map((r, i) => 2 * (r[0] ?? 0) - 1.5 * (r[1] ?? 0) + 4 + noise * (draws[n * d + i] ?? 0))
  return { x, y }
}

describe('ardRegression', () => {
  const data = problem(80, 6, 0.1)

  it('keeps the relevant features, prunes the others and recovers the weights', () => {
    const f = ardRegression().fit(data)
    expect(f.params.keep).toEqual([true, true, false, false, false, false])
    expect(f.params.coef[0]).toBeCloseTo(2, 1)
    expect(f.params.coef[1]).toBeCloseTo(-1.5, 1)
    expect(f.params.intercept).toBeCloseTo(4, 1)
    for (const [j, k] of f.params.keep.entries()) if (!k) expect(f.params.coef[j]).toBe(0)
    expect(1 / f.params.alpha).toBeCloseTo(0.01, 2)
  })

  it('predicts the linear model and its predictive spread is at least the noise level', () => {
    const f = ardRegression().fit(data)
    const pred = toFlat(f.predict(data.x))
    expect(pred[3]).toBeCloseTo(
      f.params.intercept + (data.x[3] ?? []).reduce((a, v, j) => a + v * (f.params.coef[j] ?? 0), 0),
      10,
    )
    for (const s of toFlat(f.predictStd(data.x)))
      expect(s).toBeGreaterThanOrEqual(Math.sqrt(1 / f.params.alpha) - 1e-12)
    expect(f.score?.(data)).toBeGreaterThan(0.99)
  })

  it('works with more features than rows, and without an intercept', () => {
    const wide = problem(20, 35, 0.05)
    const f = ardRegression().fit(wide)
    expect(f.params.keep.filter(Boolean).length).toBeLessThan(10)
    expect(f.params.keep[0]).toBe(true)
    const through = ardRegression({ fitIntercept: false }).fit({ x: data.x, y: data.x.map((r) => 3 * (r[2] ?? 0)) })
    expect(through.params.intercept).toBe(0)
    expect(through.params.keep[2]).toBe(true)
  })

  it('a session steps the iterations and fittedAt gives the model so far', () => {
    const s = ardRegression().session(data, { logs: 'all' })
    s.advance({ steps: 2 })
    expect(s.t).toBe(2)
    expect(s.fittedAt(2).params.iterations).toBe(2)
    expect(metricSeries(s.run(), 'solver/active').values.length).toBeGreaterThan(1)
  })

  it('the state serialises and the run resumes from it', () => {
    const design = designOf(data, 'test')
    const alg = ardAlgorithm(design)
    const mid = runAlgorithm(alg, { maxSteps: 3 }).state
    expect(stateFromJson(JSON.parse(JSON.stringify(stateToJson(mid))))).toEqual(mid)
    const end = runAlgorithm(alg, { maxSteps: 300 }).state
    const post = ardCoefficients(end, design, true)
    expect(toArray(ardRegression().revive(post).predict(data.x))).toEqual(
      toArray(ardRegression().fit(data).predict(data.x)),
    )
  })

  it('a target that is constant leaves nothing to explain: every weight is pruned or zero', () => {
    const f = ardRegression().fit({ x: data.x, y: data.x.map(() => 5) })
    expect(f.params.coef.every((v) => Math.abs(v) < 1e-6)).toBe(true)
    expect(f.params.intercept).toBeCloseTo(5, 8)
  })
})

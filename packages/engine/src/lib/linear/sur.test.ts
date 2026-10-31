import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { olsSummary } from './inference.ts'
import { sur, surAlgorithm } from './sur.ts'

const n = 150
const z = toFlat(normal(key(21), [n * 6]))
const x = Array.from({ length: n }, (_, i) => [z[i * 6] ?? 0, z[i * 6 + 1] ?? 0, z[i * 6 + 2] ?? 0])
// Errors share a common component (correlation about 0.9); the equations use different features.
const common = Array.from({ length: n }, (_, i) => z[i * 6 + 3] ?? 0)
const y = x.map((r, i) => [
  1 + 2 * (r[0] ?? 0) + (common[i] ?? 0) + 0.3 * (z[i * 6 + 4] ?? 0),
  -1 + 3 * (r[1] ?? 0) - (r[2] ?? 0) + (common[i] ?? 0) + 0.3 * (z[i * 6 + 5] ?? 0),
])

describe('sur', () => {
  it('equals equation-by-equation least squares when the equations share all their regressors', () => {
    const f = sur().fit({ x, y })
    for (let e = 0; e < 2; e++) {
      const ols = olsSummary({ x, y: y.map((r) => r[e] ?? 0) })
      for (let j = 0; j < 3; j++) expect(f.params.coef[j]?.[e]).toBeCloseTo(ols.table.estimates[j] ?? Number.NaN, 8)
      expect(f.params.intercept[e]).toBeCloseTo(ols.table.estimates[3] ?? Number.NaN, 8)
    }
  })

  it('uses the correlated errors: a different-regressor system is estimated more precisely than by least squares', () => {
    const eq = [[0], [1, 2]]
    const f = sur({ equations: eq }).fit({ x, y })
    const ols0 = olsSummary({ x: x.map((r) => [r[0] ?? 0]), y: y.map((r) => r[0] ?? 0) })
    // Planted truth is recovered, and the standard error of the first slope is smaller than OLS's.
    expect(f.params.coef[0]?.[0]).toBeCloseTo(2, 0)
    expect(f.params.coef[1]?.[1]).toBeCloseTo(3, 0)
    expect(f.params.coef[2]?.[1]).toBeCloseTo(-1, 0)
    expect(f.params.coef[1]?.[0]).toBe(0)
    expect(Math.sqrt(f.params.covariance[0]?.[0] ?? Number.NaN)).toBeLessThan(ols0.table.standardErrors?.[0] ?? 0)
    // The cross-equation correlation of the errors is recovered (a shared component of variance ~1.1 against 0.09).
    const s = f.params.sigma
    const corr = (s[0]?.[1] ?? 0) / Math.sqrt((s[0]?.[0] ?? 1) * (s[1]?.[1] ?? 1))
    expect(corr).toBeGreaterThan(0.8)
  })

  it('predicts every equation and scores well', () => {
    const f = sur({ equations: [[0], [1, 2]], maxIter: 1 }).fit({ x, y })
    expect(f.score({ x, y })).toBeGreaterThan(0.8)
    expect(toFlat(f.predict(x)).length).toBe(n * 2)
  })

  it('converges when iterated, and serialises its state', () => {
    const algorithm = surAlgorithm({ x, y }, { equations: [[0], [1, 2]], tol: 1e-9 })
    const run = runAlgorithm(algorithm, { key: key(0), maxSteps: 60 })
    expect(run.status).toBe('converged')
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 2 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, { key: key(0), maxSteps: 58, start: { state: json, step: head.steps, serialised: true } })
    expect(resumed.state.beta).toEqual(run.state.beta)
  })

  it('validates the equations', () => {
    expect(() => sur({ equations: [[0]] }).fit({ x, y })).toThrow(/equations/)
    expect(() => sur({ equations: [[0], [7]] }).fit({ x, y })).toThrow(/column/)
  })
})

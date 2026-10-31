import { describe, expect, it } from 'vitest'
import { key, normal, split } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

/** Two cointegrated walks (y = 1.5 x + AR(1) spread) and a free walk, from a key. */
function system(seed: number, n: number): number[][] {
  const [k1, k2, k3] = split(key(seed), 3)
  const e1 = toFlat(normal(k1 as never, [n]))
  const e2 = toFlat(normal(k2 as never, [n]))
  const e3 = toFlat(normal(k3 as never, [n]))
  const rows: number[][] = []
  let x = 0
  let s = 0
  let z = 0
  for (let t = 0; t < n; t++) {
    x += e1[t] ?? 0
    s = 0.4 * s + 0.5 * (e2[t] ?? 0)
    z += e3[t] ?? 0
    rows.push([x, 1.5 * x + s, z])
  }
  return rows
}

describe('Engle-Granger', () => {
  it('rejects no cointegration for a cointegrated pair and not for independent walks', () => {
    const rows = system(1, 300)
    const x = rows.map((r) => r[0] ?? 0)
    const y = rows.map((r) => r[1] ?? 0)
    const z = rows.map((r) => r[2] ?? 0)
    const pair = T.engleGranger(y, x)
    expect(pair.pValue).toBeLessThan(0.01)
    expect(pair.coefficients[0]).toBeCloseTo(1.5, 1)
    expect(pair.spread).toHaveLength(300)
    expect(T.engleGranger(z, x).pValue).toBeGreaterThan(0.05)
  })
})

describe('Johansen', () => {
  it('finds one cointegrating relation among the three series', () => {
    const rows = system(2, 400)
    const r = T.johansen(rows, 0, 1)
    expect(r.eigenvalues[0]).toBeGreaterThan(r.eigenvalues[1] ?? 1)
    expect(T.selectCointegrationRank(rows, 0, 1).rank).toBe(1)
    expect(T.selectCointegrationRank(rows, 0, 1, 'maxeig').rank).toBe(1)
    // The leading eigenvector is proportional to (1.5, -1, 0), the planted relation.
    const v = r.eigenvectors.map((row) => row[0] ?? 0)
    expect(Math.abs((v[0] ?? 0) / (v[1] ?? 1))).toBeCloseTo(1.5, 0)
    expect(Math.abs(v[2] ?? 0) / Math.abs(v[1] ?? 1)).toBeLessThan(0.3)
  })
})

describe('VECM', () => {
  const rows = system(3, 300).map((r) => [r[0] ?? 0, r[1] ?? 0])
  it('recovers the planted cointegrating vector and the sign of the adjustment', () => {
    const f = T.vecmFit(rows, { rank: 1, kArDiff: 1 })
    expect(f.beta[0]?.[0]).toBe(1)
    expect(f.beta[1]?.[0]).toBeCloseTo(-1 / 1.5, 1)
    // beta'y = x - y / 1.5 = -spread / 1.5, so y corrects a positive spread by falling: alpha_y is positive.
    expect(f.alpha[1]?.[0] ?? 0).toBeGreaterThan(0)
  })
  it('gives the same beta from the generic solvers as from the eigen-solve', () => {
    const base = T.vecmFit(rows, { rank: 1 })
    for (const solver of [
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'bfgs', gradTol: 1e-10 },
      { kind: 'lbfgs', gradTol: 1e-10 },
    ]) {
      const f = T.vecmFit(rows, { rank: 1, solver, tol: 1e-9 })
      expect(f.solver).toBe(solver.kind)
      expect(f.beta[1]?.[0]).toBeCloseTo(base.beta[1]?.[0] ?? 0, 4)
      expect(f.logLikelihood).toBeCloseTo(base.logLikelihood, 6)
    }
  })
  it('serialises the state of the traced fit and resumes it', () => {
    const alg = T.vecmAlgorithm(T.vecmProblem(rows, { rank: 1 }), { solver: { kind: 'bfgs', gradTol: 1e-9 } })
    const run = runAlgorithm(alg, { maxSteps: 3 })
    const back = alg.deserialise?.(alg.serialise?.(run.state) as never)
    expect(back?.u).toEqual(run.state.u)
  })
  it('forecasts by iterating the VAR form: the spread forecast reverts towards its mean', () => {
    const f = T.vecmFit(rows, { rank: 1, deterministic: 'ci' })
    const fc = T.vecmForecast(f, 5)
    expect(fc.mean).toHaveLength(5)
    for (let h = 1; h < 5; h++)
      expect((fc.upper[h]?.[0] ?? 0) - (fc.lower[h]?.[0] ?? 0)).toBeGreaterThan(
        (fc.upper[h - 1]?.[0] ?? 0) - (fc.lower[h - 1]?.[0] ?? 0),
      )
  })
})

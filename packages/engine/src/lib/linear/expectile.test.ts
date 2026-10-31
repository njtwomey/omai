import { describe, expect, it } from 'vitest'
import { ndtr, ndtri, normalPdf } from '../numerics/special/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  expectileAlgorithm,
  expectileObjective,
  expectileProblemOf,
  expectileRegressor,
  expectileSmoothProblem,
} from './expectile.ts'
import { olsSummary } from './inference.ts'

// A deterministic design and noise: x on a golden-ratio sequence, the noise the standard normal quantiles of another.
const n = 400
const frac = (v: number): number => v - Math.floor(v)
const xs = Array.from({ length: n }, (_, i) => frac((i + 1) * 0.7548776662466927) * 4 - 2)
const noise = Array.from({ length: n }, (_, i) => item(ndtri(frac((i + 1) * 0.5698402909980532) * 0.998 + 0.001)))
const x = xs.map((v) => [v])
const y = xs.map((v, i) => 1 + 2 * v + 0.8 * (noise[i] ?? 0))

/** The tau-expectile of a standard normal: tau E[(Z - e)+] = (1 - tau) E[(e - Z)+], by bisection. */
function normalExpectile(tau: number): number {
  const phi = (e: number): number => item(normalPdf(e))
  const cdf = (e: number): number => item(ndtr(e))
  const gap = (e: number): number => tau * (phi(e) - e * (1 - cdf(e))) - (1 - tau) * (phi(e) + e * cdf(e))
  let lo = -10
  let hi = 10
  for (let k = 0; k < 200; k++) {
    const mid = (lo + hi) / 2
    if (gap(mid) > 0) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

describe('expectile objective', () => {
  it('has the gradient and the Hessian of its finite differences', () => {
    const problem = expectileProblemOf({ x, y }, { tau: 0.8, alpha: 0.3 }, 'test')
    const theta = [1.7, 0.4]
    const e = expectileObjective(problem, theta)
    const h = 1e-6
    for (let j = 0; j < 2; j++) {
      const up = [...theta]
      const dn = [...theta]
      up[j] = (up[j] ?? 0) + h
      dn[j] = (dn[j] ?? 0) - h
      const eu = expectileObjective(problem, up)
      const ed = expectileObjective(problem, dn)
      expect(e.gradient[j]).toBeCloseTo((eu.value - ed.value) / (2 * h), 5)
      for (let k = 0; k < 2; k++)
        expect(e.hessian[j * 2 + k]).toBeCloseTo(((eu.gradient[k] ?? 0) - (ed.gradient[k] ?? 0)) / (2 * h), 5)
    }
  })

  it('rejects a level outside (0, 1) and a negative penalty', () => {
    expect(() => expectileProblemOf({ x, y }, { tau: 1 }, 'test')).toThrow(/level/)
    expect(() => expectileProblemOf({ x, y }, { alpha: -1 }, 'test')).toThrow(/penalty/)
  })
})

describe('expectile regression', () => {
  it('is ordinary least squares at tau = 1/2', () => {
    const f = expectileRegressor({ tau: 0.5 }).fit({ x, y })
    const ols = olsSummary({ x, y })
    expect(f.params.coef[0]).toBeCloseTo(ols.table.estimates[0] ?? Number.NaN, 9)
    expect(f.params.intercept).toBeCloseTo(ols.table.estimates[1] ?? Number.NaN, 9)
    expect(f.params.converged).toBe(true)
  })

  it('satisfies the asymmetric normal equations: the weighted residuals are orthogonal to the design', () => {
    const tau = 0.85
    const f = expectileRegressor({ tau }).fit({ x, y })
    const fitted = toFlat(f.predict(x))
    let g0 = 0
    let g1 = 0
    for (let i = 0; i < n; i++) {
      const r = (y[i] ?? 0) - (fitted[i] ?? 0)
      const a = r >= 0 ? tau : 1 - tau
      g0 += a * r * (xs[i] ?? 0)
      g1 += a * r
    }
    expect(Math.abs(g0) / n).toBeLessThan(1e-9)
    expect(Math.abs(g1) / n).toBeLessThan(1e-9)
  })

  it('recovers the slope and the shifted intercept of a planted location-scale model', () => {
    // y = 1 + 2 x + 0.8 z: every conditional expectile has slope 2 and intercept 1 + 0.8 e_tau(Z).
    for (const tau of [0.1, 0.5, 0.9]) {
      const f = expectileRegressor({ tau }).fit({ x, y })
      expect(f.params.coef[0]).toBeCloseTo(2, 1)
      expect(f.params.intercept).toBeCloseTo(1 + 0.8 * normalExpectile(tau), 1)
    }
  })

  it('orders the fitted expectiles by level', () => {
    const at = (tau: number): number =>
      toFlat(
        expectileRegressor({ tau })
          .fit({ x, y })
          .predict([[0.3]]),
      )[0] ?? 0
    expect(at(0.1)).toBeLessThan(at(0.5))
    expect(at(0.5)).toBeLessThan(at(0.9))
  })

  it('is mirrored by negating the targets and the level', () => {
    const low = expectileRegressor({ tau: 0.2 }).fit({ x, y }).params
    const high = expectileRegressor({ tau: 0.8 }).fit({ x, y: y.map((v) => -v) }).params
    expect(high.coef[0]).toBeCloseTo(-(low.coef[0] ?? 0), 9)
    expect(high.intercept).toBeCloseTo(-low.intercept, 9)
  })

  it('weights rows as repetitions', () => {
    const rows = [0, 1, 2, 3, 4, 5]
    const small = { x: rows.map((v) => [v]), y: [0.2, 1.4, 1.7, 3.6, 3.1, 5.5] }
    const weights = [1, 2, 1, 3, 1, 1]
    const repeated = {
      x: rows.flatMap((v, i) => Array.from({ length: weights[i] ?? 1 }, () => [v])),
      y: small.y.flatMap((v, i) => Array.from({ length: weights[i] ?? 1 }, () => v)),
    }
    const a = expectileRegressor({ tau: 0.7 }).fit({ ...small, weights })
    const b = expectileRegressor({ tau: 0.7 }).fit(repeated)
    expect(a.params.coef[0]).toBeCloseTo(b.params.coef[0] ?? Number.NaN, 9)
    expect(a.params.intercept).toBeCloseTo(b.params.intercept, 9)
  })

  it('shrinks the slope as the penalty grows', () => {
    const slope = (alpha: number): number => expectileRegressor({ tau: 0.7, alpha }).fit({ x, y }).params.coef[0] ?? 0
    expect(slope(0.5)).toBeLessThan(slope(0))
    expect(slope(5)).toBeLessThan(slope(0.5))
  })

  it('reaches one optimum with Newton, BFGS and L-BFGS', () => {
    const reference = expectileRegressor({ tau: 0.8 }).fit({ x, y }).params
    for (const solver of [
      smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-10 }),
      smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 }),
    ]) {
      const p = expectileRegressor({ tau: 0.8, solver, tol: 1e-8, maxIter: 2000 }).fit({ x, y }).params
      expect(p.coef[0]).toBeCloseTo(reference.coef[0] ?? Number.NaN, 5)
      expect(p.intercept).toBeCloseTo(reference.intercept, 5)
    }
  })

  it('matches the heteroscedasticity-consistent covariance of least squares at tau = 1/2', () => {
    const f = expectileRegressor({ tau: 0.5 }).fit({ x, y })
    const hc0 = olsSummary({ x, y }, { covType: 'HC0' }).table.standardErrors ?? []
    expect(Math.sqrt(f.params.covariance[0]?.[0] ?? Number.NaN)).toBeCloseTo(hc0[0] ?? Number.NaN, 8)
    expect(Math.sqrt(f.params.covariance[1]?.[1] ?? Number.NaN)).toBeCloseTo(hc0[1] ?? Number.NaN, 8)
  })

  it('serialises the state of the algorithm and resumes from it', () => {
    const problem = expectileProblemOf({ x, y }, { tau: 0.9 }, 'test')
    const algorithm = expectileAlgorithm(problem, { solver: smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-12 }) })
    const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: 6 })
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 3 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(0),
      maxSteps: 3,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.theta).toEqual(straight.state.theta)
    expect(expectileSmoothProblem(problem).x0).toEqual([0, expect.any(Number)])
  })
})

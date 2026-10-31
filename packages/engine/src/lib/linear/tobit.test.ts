import { describe, expect, it } from 'vitest'
import { ndtri } from '../numerics/special/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { olsSummary } from './inference.ts'
import { tobit, tobitAlgorithm, tobitObjective, tobitProblemOf } from './tobit.ts'

const n = 600
const frac = (v: number): number => v - Math.floor(v)
const xs = Array.from({ length: n }, (_, i) => frac((i + 1) * 0.7548776662466927) * 4 - 1)
const noise = Array.from({ length: n }, (_, i) => item(ndtri(frac((i + 1) * 0.5698402909980532) * 0.998 + 0.001)))
const x = xs.map((v) => [v])
// A latent line 0.5 + 1.5 x with noise sd 1, observed clipped below at 0 and above at 4.
const latent = xs.map((v, i) => 0.5 + 1.5 * v + (noise[i] ?? 0))
const clip = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v))

describe('tobit likelihood', () => {
  it('has the gradient and Hessian of its finite differences, with every kind of row', () => {
    const y = latent.map((v) => clip(v, 0, 4))
    const problem = tobitProblemOf({ x, y }, { lower: 0, upper: 4 }, 'test')
    expect(problem.censoring.some((c) => c < 0)).toBe(true)
    expect(problem.censoring.some((c) => c > 0)).toBe(true)
    const theta = [0.9, 0.3, 0.8]
    const e = tobitObjective(problem, theta)
    const h = 1e-6
    for (let j = 0; j < 3; j++) {
      const up = [...theta]
      const dn = [...theta]
      up[j] = (up[j] ?? 0) + h
      dn[j] = (dn[j] ?? 0) - h
      const eu = tobitObjective(problem, up)
      const ed = tobitObjective(problem, dn)
      expect(e.gradient[j]).toBeCloseTo((eu.value - ed.value) / (2 * h), 5)
      for (let k = 0; k < 3; k++)
        expect(e.hessian[j * 3 + k]).toBeCloseTo(((eu.gradient[k] ?? 0) - (ed.gradient[k] ?? 0)) / (2 * h), 5)
    }
  })

  it('rejects limits in the wrong order', () => {
    expect(() => tobitProblemOf({ x, y: latent }, { lower: 1, upper: 1 }, 'test')).toThrow(/below/)
  })
})

describe('tobit', () => {
  it('is least squares with the maximum-likelihood sigma when nothing is censored', () => {
    const f = tobit().fit({ x, y: latent })
    const ols = olsSummary({ x, y: latent })
    expect(f.params.coef[0]).toBeCloseTo(ols.table.estimates[0] ?? Number.NaN, 8)
    expect(f.params.intercept).toBeCloseTo(ols.table.estimates[1] ?? Number.NaN, 8)
    const rss = ols.residuals.reduce((a, r) => a + r * r, 0)
    expect(f.params.sigma).toBeCloseTo(Math.sqrt(rss / n), 8)
    expect(f.params.nLeftCensored + f.params.nRightCensored).toBe(0)
  })

  it('recovers the latent line from data censored on both sides, where least squares cannot', () => {
    const y = latent.map((v) => clip(v, 0, 4))
    const f = tobit({ lower: 0, upper: 4 }).fit({ x, y })
    expect(f.params.nLeftCensored).toBeGreaterThan(50)
    expect(f.params.nRightCensored).toBeGreaterThan(50)
    expect(f.params.coef[0]).toBeCloseTo(1.5, 1)
    expect(f.params.intercept).toBeCloseTo(0.5, 1)
    expect(f.params.sigma).toBeCloseTo(1, 1)
    const naive = olsSummary({ x, y }).table.estimates[0] ?? 0
    expect(Math.abs(naive - 1.5)).toBeGreaterThan(Math.abs((f.params.coef[0] ?? 0) - 1.5) + 0.2)
    expect(f.params.converged).toBe(true)
  })

  it('is mirrored by negating the targets and swapping the limits', () => {
    const y = latent.map((v) => clip(v, -1, 3))
    const a = tobit({ lower: -1, upper: 3 }).fit({ x, y }).params
    const b = tobit({ lower: -3, upper: 1 }).fit({ x, y: y.map((v) => -v) }).params
    expect(b.coef[0]).toBeCloseTo(-(a.coef[0] ?? 0), 7)
    expect(b.intercept).toBeCloseTo(-a.intercept, 7)
    expect(b.sigma).toBeCloseTo(a.sigma, 7)
  })

  it('reaches one optimum with Newton, BFGS and L-BFGS', () => {
    const y = latent.map((v) => clip(v, 0, Number.POSITIVE_INFINITY))
    const reference = tobit({ lower: 0 }).fit({ x, y }).params
    for (const solver of [
      smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-10 }),
      smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 }),
    ]) {
      const p = tobit({ lower: 0, solver, tol: 1e-8, maxIter: 3000 }).fit({ x, y }).params
      expect(p.coef[0]).toBeCloseTo(reference.coef[0] ?? Number.NaN, 4)
      expect(p.intercept).toBeCloseTo(reference.intercept, 4)
      expect(p.sigma).toBeCloseTo(reference.sigma, 4)
    }
  })

  it('predicts the expected observed value of the clipped latent variable', () => {
    const y = latent.map((v) => clip(v, 0, 4))
    const f = tobit({ lower: 0, upper: 4 }).fit({ x, y })
    const { coef, intercept, sigma } = f.params
    const point = 0.7
    const eta = (coef[0] ?? 0) * point + intercept
    // The mean of clip(eta + sigma z) over the standard normal quantile grid.
    const m = 20000
    let total = 0
    for (let i = 0; i < m; i++) total += clip(eta + sigma * item(ndtri((i + 0.5) / m)), 0, 4)
    expect(toFlat(f.predictCensored([[point]]))[0]).toBeCloseTo(total / m, 3)
    expect(toFlat(f.predict([[point]]))[0]).toBeCloseTo(eta, 12)
  })

  it('gives Wald standard errors for the slopes, the intercept and sigma', () => {
    const y = latent.map((v) => clip(v, 0, Number.POSITIVE_INFINITY))
    const f = tobit({ lower: 0 }).fit({ x, y })
    const table = f.exposition[0]
    expect(table?.kind).toBe('coefficientTable')
    const se = (table as { standardErrors?: number[] }).standardErrors ?? []
    expect(se).toHaveLength(3)
    for (const v of se) expect(v).toBeGreaterThan(0)
  })

  it('serialises the state of the algorithm and resumes from it', () => {
    const y = latent.map((v) => clip(v, 0, Number.POSITIVE_INFINITY))
    const algorithm = tobitAlgorithm(tobitProblemOf({ x, y }, { lower: 0 }, 'test'), {
      solver: smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-12 }),
    })
    const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: 6 })
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 3 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(0),
      maxSteps: 3,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.theta).toEqual(straight.state.theta)
  })
})

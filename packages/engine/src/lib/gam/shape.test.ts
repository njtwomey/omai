/**
 * Tests for shape-constrained smooths: the constrained fit has the shape even on data that does not, equals the
 * unconstrained fit when the truth already has the shape, is the optimum of the constrained problem (no feasible
 * direction improves it), and works at selected smoothing parameters and for bounds.
 */

import { describe, expect, it } from 'vitest'
import { glmSolvers, solverConfigOf } from '../optim/index.ts'
import { key, normal } from '../random/index.ts'
import {
  assemblePenalty,
  fitGam,
  GAM_DEFAULTS,
  type GamHyper,
  gamAlgorithm,
  gamProblem,
  gamSolution,
  s,
} from './index.ts'

const noise = (seed: number, n: number): Float64Array => normal(key(seed), [n]).data as Float64Array
const grid = (n: number): number[][] => Array.from({ length: n }, (_, i) => [i / (n - 1)])

/** The partial effect of term 0 on a fine grid. */
function effect(problem: ReturnType<typeof gamProblem>, beta: ArrayLike<number>, points = 400): number[] {
  const t = problem.design.terms[0]
  if (t === undefined) throw new Error('no term')
  const q = { n: points, d: 1, data: Float64Array.from({ length: points }, (_, i) => i / (points - 1)) }
  const block = t.evaluate(q)
  return Array.from({ length: points }, (_, i) => {
    let a = 0
    for (let j = 0; j < t.width; j++) a += (block[i * t.width + j] ?? 0) * (beta[t.offset + j] ?? 0)
    return a
  })
}

const diffs = (v: readonly number[]): number[] => v.slice(1).map((a, i) => a - (v[i] ?? 0))

describe('monotone smooths', () => {
  const n = 80
  const x = grid(n)
  const z = noise(1, n)
  // A decreasing dip in the middle of an increasing trend: the unconstrained fit is not monotone.
  const y = x.map((r, i) => (r[0] ?? 0) * 2 - 3 * Math.exp(-(((r[0] ?? 0) - 0.5) ** 2) / 0.005) + 0.15 * (z[i] ?? 0))
  const hyper = (extra: Partial<GamHyper>): GamHyper => ({ ...GAM_DEFAULTS, method: 'fixed', sp: [0.01], ...extra })

  for (const basis of ['ps', 'cr'] as const) {
    it(`is monotone on a P-spline or cubic regression spline (${basis}) when the data are not`, () => {
      const free = hyper({ terms: [s(0, { basis, k: 12 })] })
      const up = hyper({ terms: [s(0, { basis, k: 12, shape: 'increasing' })] })
      const pFree = gamProblem(x, y, free)
      const pUp = gamProblem(x, y, up)
      const f = effect(pFree, fitGam(pFree, free).state.beta)
      const u = effect(pUp, fitGam(pUp, up).state.beta)
      expect(Math.min(...diffs(f))).toBeLessThan(-1e-3)
      // Exact on a P-spline; for the other bases the constraint holds on an 80-point grid, so between grid points
      // a violation can only be of the order of the curvature times the squared spacing.
      expect(Math.min(...diffs(u))).toBeGreaterThan(basis === 'ps' ? -1e-10 : -2e-4)
      expect(Math.max(...u) - Math.min(...u)).toBeGreaterThan(0.5)
    })
  }

  it('is decreasing when asked, and equals the unconstrained fit when the truth is already monotone', () => {
    const exact = x.map((r) => Math.sqrt(r[0] ?? 0) * 3)
    const free = hyper({ terms: [s(0, { basis: 'ps', k: 10 })], sp: [1e-4] })
    const up = hyper({ terms: [s(0, { basis: 'ps', k: 10, shape: 'increasing' })], sp: [1e-4] })
    const a = fitGam(gamProblem(x, exact, free), free).state
    const b = fitGam(gamProblem(x, exact, up), up).state
    a.beta.forEach((v, i) => {
      expect(b.beta[i]).toBeCloseTo(v, 9)
    })
    const down = hyper({ terms: [s(0, { basis: 'ps', k: 10, shape: 'decreasing' })] })
    const pd = gamProblem(
      x,
      exact.map((v) => -v),
      down,
    )
    expect(Math.max(...diffs(effect(pd, fitGam(pd, down).state.beta)))).toBeLessThan(1e-10)
  })

  it('is the optimum of the constrained problem: no feasible direction lowers the penalised criterion', () => {
    const up = hyper({ terms: [s(0, { basis: 'ps', k: 10, shape: 'increasing' })], sp: [0.003] })
    const problem = gamProblem(x, y, up)
    const { design } = problem
    const state = fitGam(problem, up).state
    const { g } = gamSolution(problem, state)
    const S = assemblePenalty(problem.structure, [Math.log(0.003)])
    const objective = (b: ArrayLike<number>): number => {
      let rss = 0
      for (let i = 0; i < design.n; i++) {
        let e = 0
        for (let j = 0; j < design.p; j++) e += (design.X[i * design.p + j] ?? 0) * (b[j] ?? 0)
        rss += (g.w[i] ?? 0) * ((g.z[i] ?? 0) - e) ** 2
      }
      let pen = 0
      for (let a = 0; a < design.p; a++)
        for (let c = 0; c < design.p; c++) pen += (b[a] ?? 0) * (S[a * design.p + c] ?? 0) * (b[c] ?? 0)
      return rss + pen
    }
    const feasible = (b: ArrayLike<number>): boolean => {
      const c = design.constraints
      if (c === undefined) return true
      for (let r = 0; r < c.count; r++) {
        let a = 0
        for (let j = 0; j < c.width; j++) a += (c.rows[r * c.width + j] ?? 0) * (b[j] ?? 0)
        if (a < (c.rhs[r] ?? 0) - 1e-12) return false
      }
      return true
    }
    const best = objective(state.beta)
    const dirs = noise(9, 300 * design.p)
    let improved = 0
    for (let d = 0; d < 300; d++) {
      for (const step of [1e-1, 1e-2, 1e-3]) {
        const trial = Float64Array.from(state.beta, (v, j) => v + step * (dirs[d * design.p + j] ?? 0))
        if (feasible(trial) && objective(trial) < best - 1e-9) improved++
      }
    }
    expect(improved).toBe(0)
    // And some of those directions are feasible and not trivially so: the unconstrained optimum is infeasible.
    const free = hyper({ terms: [s(0, { basis: 'ps', k: 10 })], sp: [0.003] })
    const pf = gamProblem(x, y, free)
    expect(feasible(fitGam(pf, free).state.beta)).toBe(false)
    expect(objective(fitGam(pf, free).state.beta)).toBeLessThan(best)
  })

  it('selects the smoothing parameter by REML under the constraint, with fewer degrees of freedom than the free fit', () => {
    const free = { ...GAM_DEFAULTS, terms: [s(0, { basis: 'ps', k: 12 })] }
    const up = { ...GAM_DEFAULTS, terms: [s(0, { basis: 'ps', k: 12, shape: 'increasing' as const })] }
    const a = fitGam(gamProblem(x, y, free), free).state
    const pUp = gamProblem(x, y, up)
    const b = fitGam(pUp, up).state
    expect(b.edf).toBeLessThan(a.edf)
    expect(Math.min(...diffs(effect(pUp, b.beta)))).toBeGreaterThan(-1e-10)
  })

  it('refuses constraints on several covariates and a solver of the slot other than the reweighted least squares', () => {
    expect(() =>
      gamProblem(
        [
          [0, 0],
          [1, 1],
          [2, 0],
          [3, 1],
          [4, 2],
          [5, 3],
          [6, 1],
          [7, 5],
          [8, 3],
          [9, 2],
          [10, 4],
          [11, 3],
        ],
        Array.from({ length: 12 }, (_, i) => i),
        { ...GAM_DEFAULTS, terms: [s([0, 1], { shape: 'increasing' })] },
      ),
    ).toThrow(/one covariate/)
    const up = {
      ...GAM_DEFAULTS,
      terms: [s(0, { basis: 'ps', k: 6, shape: 'increasing' as const })],
      solver: solverConfigOf(glmSolvers, { kind: 'newton' }),
    }
    expect(() =>
      gamAlgorithm(
        gamProblem(
          grid(30),
          grid(30).map((r) => r[0] ?? 0),
          up,
        ),
        up,
      ),
    ).toThrow(/irls/)
  })
})

describe('convex, concave and bounded smooths', () => {
  const n = 70
  const x = grid(n)
  const z = noise(5, n)
  const y = x.map((r, i) => Math.sin(5 * (r[0] ?? 0)) + 0.2 * (z[i] ?? 0))
  const fit = (term: ReturnType<typeof s>): { problem: ReturnType<typeof gamProblem>; beta: number[] } => {
    const h = { ...GAM_DEFAULTS, method: 'fixed' as const, sp: [0.002], terms: [term] }
    const problem = gamProblem(x, y, h)
    return { problem, beta: fitGam(problem, h).state.beta }
  }

  it('fits a convex and a concave spline', () => {
    const cvx = fit(s(0, { basis: 'ps', k: 12, shape: 'convex' }))
    const ccv = fit(s(0, { basis: 'ps', k: 12, shape: 'concave' }))
    expect(Math.min(...diffs(diffs(effect(cvx.problem, cvx.beta))))).toBeGreaterThan(-1e-9)
    expect(Math.max(...diffs(diffs(effect(ccv.problem, ccv.beta))))).toBeLessThan(1e-9)
  })

  it('combines shapes: increasing and concave (a saturating curve)', () => {
    const sat = fit(s(0, { basis: 'ps', k: 12, shape: ['increasing', 'concave'] }))
    const e = effect(sat.problem, sat.beta)
    expect(Math.min(...diffs(e))).toBeGreaterThan(-1e-9)
    expect(Math.max(...diffs(diffs(e)))).toBeLessThan(1e-9)
  })

  it('keeps the centred partial effect inside its bounds', () => {
    const free = fit(s(0, { basis: 'cr', k: 10 }))
    const bounded = fit(s(0, { basis: 'cr', k: 10, lower: -0.4, upper: 0.4 }))
    const f = effect(free.problem, free.beta)
    const b = effect(bounded.problem, bounded.beta)
    expect(Math.max(...f)).toBeGreaterThan(0.6)
    expect(Math.max(...b)).toBeLessThan(0.4 + 5e-3)
    expect(Math.min(...b)).toBeGreaterThan(-0.4 - 5e-3)
    // The design carries the constraints as rows over the whole coefficient vector.
    expect(bounded.problem.design.constraints?.width).toBe(bounded.problem.design.p)
  })
})

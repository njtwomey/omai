import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { problem } from '../../../testing/problems.ts'
import { concat, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as O from '../index.ts'

const METHODS: Readonly<
  Record<string, (c: Readonly<Record<string, unknown>>) => O.MinimizeMethod<O.VectorMethodOptions>>
> = {
  trustDogleg: () => O.trustDogleg,
  trustNcg: () => O.trustNcg,
  trustExact: () => O.trustExact,
  newton: () => O.newton,
  newtonCg: (p) => (f, x0, o) => O.newtonCg(f, x0, { ...o, xtol: 1e-8, hessian: p['hessian'] as 'dense' | 'products' }),
}

const GRAD_TOL: Readonly<Record<string, number>> = {
  trustDogleg: 1e-8,
  trustNcg: 1e-8,
  trustExact: 1e-8,
  newton: 1e-10,
}

for (const fixture of ['optim/minimize/trust_region', 'optim/minimize/newton']) {
  describe(`${fixture} against scipy.optimize.minimize`, () => {
    for (const c of loadFixture(fixture).cases) {
      it(c.name, () => {
        const x0 = decode(c.inputs[0]) as Tensor
        const method = (
          METHODS[c.fn] as (p: Readonly<Record<string, unknown>>) => O.MinimizeMethod<O.VectorMethodOptions>
        )(c.params)
        const gradTol = GRAD_TOL[c.fn]
        const r = O.minimize(problem(c.params['problem'] as string), x0, {
          method,
          ...(gradTol === undefined ? {} : { gradTol }),
        })
        if (c.params['success'] !== false) expect(r.converged).toBe(true)
        expectMatches(concat([r.x as Value, tensor([r.fun])]), c.output, c.rtol, c.atol, c.name)
        const nit = c.params['nit'] as number | undefined
        if (nit !== undefined) expect(Math.abs(r.steps - nit)).toBeLessThanOrEqual(1)
      })
    }
  })
}

describe('second-order methods', () => {
  it('Newton converges quadratically: few steps on Rosenbrock', () => {
    const r = O.newton(problem('rosen'), [-1.2, 1], { gradTol: 1e-10 })
    expect(r.converged).toBe(true)
    expect(r.steps).toBeLessThan(30)
    expect(toFlat(r.x as Value).map((v) => Math.round(v * 1e8) / 1e8)).toEqual([1, 1])
  })
  it('eigenvalue modification also reaches the minimum from a saddle region', () => {
    const r = O.newton(problem('himmelblau'), [-0.27, -0.92], { modification: 'eigenvalue' })
    expect(r.converged).toBe(true)
    expect(r.fun).toBeLessThan(1e-12)
  })
  it('modified Cholesky shifts indefinite matrices', () => {
    const m = O.modifiedCholesky([1, 2, 2, 1], 2)
    expect(m.tau).toBeGreaterThan(1)
    const u = m.factor
    // U^T U = H + tau I
    const h00 = (u[0] ?? 0) ** 2
    expect(h00).toBeCloseTo(1 + m.tau, 12)
  })
  it('trust regions trace the radius and the ratio', () => {
    const rec = O.trustExact(problem('beale'), [1, 1], { gradTol: 1e-8 })
    expect(rec.converged).toBe(true)
    expect(rec.nhev).toBeGreaterThan(0)
  })
  it('subproblem solvers respect the radius and find interior Newton steps', () => {
    const H = new Float64Array([2, 0, 0, -1])
    const model = {
      value: 0,
      grad: new Float64Array([1, 1]),
      hessp: (v: Float64Array) => new Float64Array([2 * (v[0] ?? 0), -(v[1] ?? 0)]),
      hess: () => H,
    }
    const steihaug = O.steihaugStep().solve(model, 0.5)
    expect(steihaug.hitsBoundary).toBe(true)
    expect(Math.hypot(steihaug.p[0] ?? 0, steihaug.p[1] ?? 0)).toBeCloseTo(0.5, 12)
    // The exact solver stops within kEasy (10%) of the boundary, as scipy's.
    const exact = O.exactStep().solve(model, 0.5)
    expect(Math.abs(Math.hypot(exact.p[0] ?? 0, exact.p[1] ?? 0) - 0.5)).toBeLessThanOrEqual(0.05)
    const tight = O.exactStep({ kEasy: 1e-10, maxIterations: 100 }).solve(model, 0.5)
    expect(Math.hypot(tight.p[0] ?? 0, tight.p[1] ?? 0)).toBeCloseTo(0.5, 8)
    expect(() => O.doglegStep().solve(model, 0.5)).toThrow(/positive definite/)
    const pd = {
      ...model,
      hessp: (v: Float64Array) => new Float64Array([2 * (v[0] ?? 0), v[1] ?? 0]),
      hess: () => new Float64Array([2, 0, 0, 1]),
    }
    // Steihaug stops early once the residual is below min(0.5, sqrt|g|) |g| (an inexact Newton step), so only the
    // dogleg and exact solvers return the Newton step here.
    expect(O.steihaugStep().solve(pd, 10).hitsBoundary).toBe(false)
    for (const sp of [O.doglegStep(), O.exactStep()]) {
      const step = sp.solve(pd, 10)
      expect(step.hitsBoundary).toBe(false)
      expect(Array.from(step.p).map((v) => Math.round(v * 1e8) / 1e8)).toEqual([-0.5, -1])
    }
  })
  it('the hard case of the exact subproblem', () => {
    // g orthogonal to the eigenvector of the smallest eigenvalue.
    const model = {
      value: 0,
      grad: new Float64Array([1, 0]),
      hessp: (v: Float64Array) => new Float64Array([v[0] ?? 0, -(v[1] ?? 0)]),
      hess: () => new Float64Array([1, 0, 0, -1]),
    }
    const step = O.exactStep().solve(model, 2)
    expect(Math.hypot(step.p[0] ?? 0, step.p[1] ?? 0)).toBeCloseTo(2, 1)
    expect(Math.abs(step.p[1] ?? 0)).toBeGreaterThan(1)
  })
})

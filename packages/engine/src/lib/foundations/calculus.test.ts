import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cos, div, exp, mul, sign, sin, sub, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const cube = (x: Value): Value => mul(x, mul(x, x))

describe('limits and difference quotients', () => {
  it('detects a jump and reports both sides', () => {
    const r = F.limit((x) => sign(x), 0)
    expect(r.exists).toBe(false)
    expect(r.left).toBeCloseTo(-1, 12)
    expect(r.right).toBeCloseTo(1, 12)
    expect(Number.isNaN(r.value)).toBe(true)
    expect(r.steps.length).toBe(12)
    expect(F.limit((x) => sign(x), 0, { side: 'left' }).value).toBeCloseTo(-1, 12)
  })

  it('limits at minus infinity', () => {
    expect(F.limit((x) => exp(x), -Infinity).value).toBeCloseTo(0, 10)
  })

  it('one-sided quotients converge at first order and central at second order', () => {
    const fwd = F.secantTable(sin, 1, { steps: [1e-2, 1e-3] })
    const ctr = F.secantTable(sin, 1, { scheme: 'central', steps: [1e-2, 1e-3] })
    expect((fwd[0]?.error ?? 0) / (fwd[1]?.error ?? 1)).toBeCloseTo(10, 0)
    expect((ctr[0]?.error ?? 0) / (ctr[1]?.error ?? 1)).toBeCloseTo(100, -1)
    expect(F.differenceQuotient((x) => mul(x, x), 3, 0.5, 'backward')).toBe(5.5)
    // Rounding error dominates for tiny steps.
    const tiny = F.secantTable(sin, 1)
    expect(tiny[tiny.length - 1]?.error ?? 0).toBeGreaterThan(tiny[5]?.error ?? 1)
  })
})

describe('Riemann sums and the fundamental theorem', () => {
  it('left and right sums bracket the integral of a monotone function and converge', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 60 }), (n) => {
        const left = F.riemannSum(exp, 0, 1, n, 'left').value
        const right = F.riemannSum(exp, 0, 1, n, 'right').value
        expect(left).toBeLessThan(Math.E - 1)
        expect(right).toBeGreaterThan(Math.E - 1)
        expect(right - left).toBeCloseTo((Math.E - 1) / n, 12)
      }),
    )
    const mid = F.riemannSum(exp, 0, 1, 100, 'midpoint')
    expect(mid.pieces).toHaveLength(100)
    expect(mid.value).toBeCloseTo(Math.E - 1, 4)
    expect(() => F.riemannSum(exp, 0, 1, 0)).toThrow()
  })

  it('the accumulation function differentiates back to the integrand', () => {
    const r = F.fundamentalTheorem((x) => mul(x, exp(mul(x, -1))), 0, [0.2, 1, 3])
    expect(r.maxError).toBeLessThan(1e-7)
    expect(r.accumulated[1]).toBeCloseTo(1 - 2 / Math.E, 10)
    const d = F.definiteIntegral(cos, sin, 0, 2)
    expect(d.difference).toBeLessThan(1e-12)
  })
})

describe('Taylor polynomials', () => {
  it('the Lagrange bound holds over the interval', () => {
    for (const order of [1, 3, 5]) {
      const t = F.taylorApproximation(sin, 0.3, order, [-1, 1.5])
      const xs = toFlat(tensor([-1, -0.4, 0.5, 1.2, 1.5]))
      const err = toFlat(sub(sin(xs), t.evaluate(xs))).map(Math.abs)
      const bound = t.remainderBound(xs)
      for (const [i, e] of err.entries()) expect(e).toBeLessThanOrEqual((bound[i] ?? 0) * (1 + 1e-9) + 1e-15)
    }
    expect(() => F.taylorApproximation(sin, 0, -1)).toThrow()
  })

  it('derivatives come from one Taylor pass', () => {
    const d = F.derivatives(exp, [0, 1], 4)
    for (const row of d) for (const [i, v] of row.entries()) expect(v).toBeCloseTo(Math.exp(i), 12)
    expect(F.derivatives(cube, 2, 1)[1]).toEqual([12])
  })
})

describe("Newton's method and critical points", () => {
  it('converges quadratically near a simple root', () => {
    const r = F.newtonSteps((x) => sub(cube(x), 2), 1)
    expect(r.converged).toBe(true)
    expect(r.root).toBeCloseTo(Math.cbrt(2), 14)
    const errs = r.steps.map((s) => Math.abs(s.x - Math.cbrt(2)))
    // e_{k+1} ~ C e_k^2: the ratio e_{k+1} / e_k^2 settles.
    const ratios = errs.slice(1, 4).map((e, k) => e / (errs[k] ?? 1) ** 2)
    expect(ratios[2] ?? 0).toBeCloseTo(ratios[1] ?? 0, 0)
    // Every next iterate is where the tangent line crosses zero.
    for (const s of r.steps) expect(s.fx + s.slope * (s.next - s.x)).toBeCloseTo(0, 12)
  })

  it('stops at a flat point and reports non-convergence', () => {
    expect(F.newtonSteps((x) => cos(x), 0).converged).toBe(false)
    expect(F.newtonSteps((x) => mul(x, x), 0).root).toBe(0)
    expect(F.newtonSteps((x) => div(1, x), 1, { maxIter: 3 }).converged).toBe(false)
  })

  it('classifies extrema and is inconclusive for flat points', () => {
    expect(F.secondDerivativeTest(cube, 0)).toBe('inconclusive')
    expect(F.secondDerivativeTest((x) => mul(x, mul(x, cube(x))), 0)).toBe('inconclusive')
    expect(F.secondDerivativeTest((x) => mul(x, x), 0)).toBe('minimum')
    const cps = F.criticalPoints(sin, 0, 7)
    expect(cps.map((c) => c.kind)).toEqual(['maximum', 'minimum'])
    expect(cps[0]?.x).toBeCloseTo(Math.PI / 2, 12)
    expect(cps[1]?.value).toBeCloseTo(-1, 14)
  })

  it('finds mean-value points with the secant slope', () => {
    const r = F.meanValuePoints(sin, 0, Math.PI)
    expect(r.slope).toBeCloseTo(0, 15)
    expect(r.points).toHaveLength(1)
    expect(r.points[0]).toBeCloseTo(Math.PI / 2, 12)
    const t = F.tangentLine(exp, 0)
    expect(t).toEqual({ slope: 1, intercept: 1 })
  })
})

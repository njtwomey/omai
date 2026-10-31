/**
 * The exact derivative of a smooth term (the basis's own derivative, centred, scaled by a `by` covariate) equals the
 * Richardson-extrapolated difference to 1e-6, and is the default where the basis has one; a basis without a closed form
 * falls back to differences and refuses `exact`.
 */

import { describe, expect, it } from 'vitest'
import { fitContext, fitGam, GAM_DEFAULTS, gamProblem, s, smoothDerivative } from './index.ts'

const n = 40
const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n, i % 2])
const y = x.map(
  (r, i) => Math.sin(5 * (r[0] ?? 0)) * (r[1] ?? 0) + 0.5 * (r[0] ?? 0) ** 2 + 0.05 * Math.sin(i * 12.9898),
)
// Away from the knots and centres, where the high derivatives of a piecewise or singular kernel jump.
const grid = [0.1, 0.23, 0.45, 0.55, 0.72, 0.88]

const ctxOf = (terms: ReturnType<typeof s>[]) => {
  const hyper = { ...GAM_DEFAULTS, terms }
  const problem = gamProblem(x, y, hyper)
  return fitContext(problem, fitGam(problem, hyper).state)
}

describe('smoothDerivative with exact basis derivatives', () => {
  for (const basis of ['cr', 'ps', 'tp'] as const) {
    for (const order of [1, 2] as const) {
      it(`${basis}, order ${order}: exact equals Richardson to 1e-6`, () => {
        const ctx = ctxOf([s(0, { basis, k: 8 })])
        const exact = smoothDerivative(ctx, 0, grid, { order, method: 'exact' })
        const numeric = smoothDerivative(ctx, 0, grid, { order, method: 'richardson', step: 0.002 })
        exact.fit.forEach((v, i) => {
          expect(Math.abs(v - (numeric.fit[i] ?? 0))).toBeLessThan(1e-6 * (1 + Math.abs(v)))
          expect(Math.abs((exact.upper[i] ?? 0) - (numeric.upper[i] ?? 0))).toBeLessThan(1e-5)
        })
      })
    }
  }

  it('applies to a varying-coefficient term (the by covariate scales the derivative)', () => {
    const ctx = ctxOf([s(0, { basis: 'cr', k: 8, by: 1 })])
    const exact = smoothDerivative(ctx, 0, grid, { method: 'exact' })
    const numeric = smoothDerivative(ctx, 0, grid, { method: 'richardson', step: 0.002 })
    exact.fit.forEach((v, i) => {
      expect(Math.abs(v - (numeric.fit[i] ?? 0))).toBeLessThan(1e-6 * (1 + Math.abs(v)))
    })
  })

  it('is the default where the basis has one, and a basis without one falls back to differences and refuses exact', () => {
    const ctx = ctxOf([s(0, { basis: 'ns', k: 6 })])
    expect(smoothDerivative(ctx, 0, grid).fit.length).toBe(grid.length)
    expect(() => smoothDerivative(ctx, 0, grid, { method: 'exact' })).toThrow(/closed-form/)
    const cr = ctxOf([s(0, { basis: 'cr', k: 8 })])
    const byDefault = smoothDerivative(cr, 0, grid)
    const exact = smoothDerivative(cr, 0, grid, { method: 'exact' })
    byDefault.fit.forEach((v, i) => {
      expect(v).toBe(exact.fit[i])
    })
  })
})

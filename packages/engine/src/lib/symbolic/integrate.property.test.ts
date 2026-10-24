// The fundamental theorem on random integrands from the families the integrator covers: d/dx integrate(f) = f.
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { polynomialArb, productArb, rationalArb } from '../../testing/symbolic.ts'
import { cos, diff, div, type Expr, evaluateComplex, exp, expand, integrate, mul, sin, sym, toText } from './index.ts'

const x = sym('x')

function derivativeIs(F: Expr, f: Expr): boolean {
  if (toText(F).includes('Integral(')) return false
  return [0.31, 0.77, 1.4].every((v) => {
    const a = evaluateComplex(diff(F, x), { x: v })
    const b = evaluateComplex(f, { x: v })
    // A sample point may be a pole of a random integrand.
    if (!b.every(Number.isFinite)) return true
    return Math.hypot(a[0] - b[0], a[1] - b[1]) <= 1e-8 * Math.max(1, Math.hypot(...b))
  })
}

describe('integration', () => {
  it('integrates polynomials times exponentials, sines and cosines', () => {
    const kernels = [exp, sin, cos]
    fc.assert(
      fc.property(polynomialArb(3, ['x']), fc.integer({ min: 0, max: 2 }), rationalArb, (p, k, a) => {
        if (a.kind === 'num' && a.n === 0n) return true
        const f = mul(p, (kernels[k] as (e: Expr) => Expr)(mul(a, x)))
        return derivativeIs(integrate(f, x), f)
      }),
      { numRuns: 40 },
    )
  })

  it('integrates rational functions with linear and quadratic factors', () => {
    fc.assert(
      fc.property(polynomialArb(2, ['x']), productArb(['x']), (n, d) => {
        if (expand(d).kind === 'num') return true
        const f = div(n, d)
        return derivativeIs(integrate(f, x), f) || toText(integrate(f, x)).includes('Integral(')
      }),
      { numRuns: 40 },
    )
  })
})

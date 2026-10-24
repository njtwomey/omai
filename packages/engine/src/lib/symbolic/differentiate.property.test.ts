// Laws of differentiation on random expressions, and agreement with reverse-mode autodiff of the compiled function.
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { exprArb, rationalArb } from '../../testing/symbolic.ts'
import { grad } from '../autodiff/index.ts'
import { tensor, toArray, type Value } from '../tensor/index.ts'
import { add, compile, diff, type Expr, evaluateComplex, mul, sym } from './index.ts'

const x = sym('x')
const POINTS = [0.37, -0.81, 1.23]

function close(a: Expr, b: Expr): boolean {
  return POINTS.every((v) => {
    const pa = evaluateComplex(a, { x: v, y: 0.53 })
    const pb = evaluateComplex(b, { x: v, y: 0.53 })
    return Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= 1e-8 * Math.max(1, Math.hypot(...pa), Math.hypot(...pb))
  })
}

describe('differentiation laws', () => {
  it('is linear', () => {
    fc.assert(
      fc.property(exprArb, exprArb, rationalArb, rationalArb, (f, g, a, b) =>
        close(diff(add(mul(a, f), mul(b, g)), x), add(mul(a, diff(f, x)), mul(b, diff(g, x)))),
      ),
      { numRuns: 50 },
    )
  })

  it('satisfies the product rule', () => {
    fc.assert(
      fc.property(exprArb, exprArb, (f, g) => close(diff(mul(f, g), x), add(mul(diff(f, x), g), mul(f, diff(g, x))))),
      { numRuns: 50 },
    )
  })

  it('mixed partials commute', () => {
    fc.assert(
      fc.property(exprArb, (f) => close(diff(f, 'x', 'y'), diff(f, 'y', 'x'))),
      { numRuns: 40 },
    )
  })

  it('agrees with grad of the compiled function', () => {
    fc.assert(
      fc.property(exprArb, (f) => {
        const F = compile(f, ['x', 'y'])
        const dF = compile(diff(f, x), ['x', 'y'])
        return POINTS.every((v) => {
          const g = toArray(grad((t: Value) => F(t, 0.53))(tensor(v))) as number
          const want = toArray(dF(v, 0.53)) as number
          return !Number.isFinite(want) || Math.abs(g - want) <= 1e-8 * Math.max(1, Math.abs(want))
        })
      }),
      { numRuns: 40 },
    )
  })
})

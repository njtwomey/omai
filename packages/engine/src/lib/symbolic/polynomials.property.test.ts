// Polynomial identities on random polynomials: division, gcd, roots and Sturm counts.
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { polynomialArb, productArb } from '../../testing/symbolic.ts'
import {
  add,
  countRealRoots,
  degree,
  equals,
  evaluateComplex,
  exactRoots,
  expand,
  mul,
  polyDiv,
  polyGcd,
  sub,
  sym,
} from './index.ts'

const x = sym('x')

describe('polynomial identities', () => {
  it('f = q*g + r with deg r < deg g', () => {
    fc.assert(
      fc.property(productArb(['x']), polynomialArb(2, ['x']), (f, g) => {
        if (degree(expand(g), x) < 1) return true
        const { quotient, remainder } = polyDiv(f, g, x)
        return (
          equals(expand(add(mul(quotient, g), remainder)), expand(f)) && degree(remainder, x) < degree(expand(g), x)
        )
      }),
      { numRuns: 60 },
    )
  })

  it('the gcd divides both and contains the common factor', () => {
    fc.assert(
      fc.property(polynomialArb(2, ['x']), polynomialArb(2, ['x']), polynomialArb(1, ['x']), (a, b, c) => {
        const f = expand(mul(a, c))
        const g = expand(mul(b, c))
        if (degree(f, x) < 1 || degree(g, x) < 1) return true
        const h = polyGcd(f, g)
        const divides = (p: typeof f): boolean => equals(polyDiv(p, h, x).remainder, 0)
        // h divides f and g, and the common factor c divides h.
        const common = degree(expand(c), x) < 1 || equals(polyDiv(h, expand(c), x).remainder, 0)
        return divides(f) && divides(g) && common
      }),
      { numRuns: 60 },
    )
  })

  it('exact roots are roots, with multiplicities summing to the degree for small factors', () => {
    fc.assert(
      fc.property(productArb(['x']), (p) => {
        const f = expand(p)
        const n = degree(f, x)
        if (!(n >= 1)) return true
        const rs = exactRoots(f, x)
        const ok = rs.every((r) => {
          const v = evaluateComplex(r.value)
          const fv = evaluateComplex(f, { x: v })
          return Math.hypot(...fv) <= 1e-7 * Math.max(1, Math.hypot(...v) ** n)
        })
        return ok && rs.reduce((s, r) => s + r.multiplicity, 0) === n
      }),
      { numRuns: 60 },
    )
  })

  it('Sturm counts match the real exact roots', () => {
    fc.assert(
      fc.property(productArb(['x']), (p) => {
        const f = expand(p)
        if (!(degree(f, x) >= 1)) return true
        const real = exactRoots(f, x).filter((r) => Math.abs(evaluateComplex(r.value)[1]) < 1e-9)
        return countRealRoots(f, x) === real.length && equals(sub(f, f), 0)
      }),
      { numRuns: 60 },
    )
  })
})

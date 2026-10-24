// Algebraic laws of the rewrites, on random polynomials, products and expressions.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { exprArb, polynomialArb, productArb } from '../../testing/symbolic.ts'
import {
  add,
  apart,
  cancel,
  dependsOn,
  div,
  type Expr,
  equals,
  evaluateComplex,
  expand,
  factor,
  mul,
  simplify,
  sub,
  sym,
  together,
  trigSimplify,
} from './index.ts'

const x = sym('x')

function sameValues(a: Expr, b: Expr): boolean {
  for (const [xv, yv] of [
    [0.37, 0.53],
    [1.21, -0.7],
    [-0.83, 1.4],
  ] as const) {
    const va = evaluateComplex(a, { x: xv, y: yv })
    const vb = evaluateComplex(b, { x: xv, y: yv })
    const scale = Math.max(1, Math.hypot(...va), Math.hypot(...vb))
    if (!(Math.hypot(va[0] - vb[0], va[1] - vb[1]) <= 1e-8 * scale)) return false
  }
  return true
}

describe('rewrite laws', () => {
  it('expand is idempotent', () => {
    fc.assert(
      fc.property(productArb(), (p) => equals(expand(expand(p)), expand(p))),
      { numRuns: 60 },
    )
  })

  it('factor then expand is the identity on polynomials', () => {
    fc.assert(
      fc.property(productArb(), (p) => equals(expand(factor(p)), expand(p))),
      { numRuns: 60 },
    )
  })

  it('factor finds the factors of a product', () => {
    fc.assert(
      fc.property(polynomialArb(2, ['x']), polynomialArb(2, ['x']), (f, g) => {
        // A product of two non-constant polynomials is reducible, so factor must not return a sum.
        if (!dependsOn(expand(f), x) || !dependsOn(expand(g), x)) return true
        const p = expand(mul(f, g))
        const fp = factor(p)
        return equals(expand(fp), p) && fp.kind !== 'add'
      }),
      { numRuns: 60 },
    )
  })

  it('together, cancel and apart preserve values', () => {
    fc.assert(
      fc.property(polynomialArb(2, ['x']), productArb(['x']), (n, d) => {
        const dd = expand(d)
        if (dd.kind === 'num') return true
        const r = add(div(n, dd), div(1, add(x, 3)))
        return sameValues(together(r), r) && sameValues(cancel(r), r) && sameValues(apart(r, x), r)
      }),
      { numRuns: 40 },
    )
  })

  it('cancel gives a canonical form', () => {
    fc.assert(
      fc.property(productArb(['x']), polynomialArb(2, ['x']), (p, q) => {
        if (expand(q).kind === 'num') return true
        const a = div(mul(p, q), q)
        return equals(cancel(a), cancel(expand(p)))
      }),
      { numRuns: 40 },
    )
  })

  it('simplify and trigSimplify preserve values', () => {
    fc.assert(
      fc.property(exprArb, (e) => sameValues(simplify(e), e) && sameValues(trigSimplify(e), e)),
      { numRuns: 40 },
    )
  })

  it('simplify recognises zero', () => {
    fc.assert(
      fc.property(productArb(), (p) => equals(simplify(sub(p, expand(p))), 0)),
      { numRuns: 30 },
    )
    expect(equals(simplify(sub(add(mul(2, x), 2), mul(2, add(x, 1)))), 0)).toBe(true)
  })
})

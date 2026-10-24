// Polynomials against SymPy. Division, gcd, lcm, resultants and discriminants are compared textually (the printed
// canonical form) and by value; Sturm sequences textually; real root counts exactly; roots by value against numpy's
// roots, as sorted (re, im) pairs with multiplicity, real roots first.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import {
  countRealRoots,
  discriminant,
  type Expr,
  evaluateComplex,
  exactRoots,
  parse,
  polyDiv,
  polyGcd,
  polyLcm,
  resultant,
  sturm,
  toText,
} from './index.ts'

const fixture = loadFixture('symbolic/polynomials')
const x = parse('x')
const arg = (inputs: readonly unknown[], i: number): Expr => parse(String(inputs[i]))

const exprOps: Readonly<Record<string, (inputs: readonly unknown[]) => Expr>> = {
  polyDivQuotient: (i) => polyDiv(arg(i, 0), arg(i, 1), x).quotient,
  polyDivRemainder: (i) => polyDiv(arg(i, 0), arg(i, 1), x).remainder,
  polyGcd: (i) => polyGcd(arg(i, 0), arg(i, 1)),
  polyLcm: (i) => polyLcm(arg(i, 0), arg(i, 1)),
  resultant: (i) => resultant(arg(i, 0), arg(i, 1), x),
  discriminant: (i) => discriminant(arg(i, 0), x),
}

describe('polynomials (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const op = exprOps[c.fn]
      if (op !== undefined) {
        const out = symbolicOutput(c)
        const r = op(c.inputs)
        expect(toText(r)).toBe(out.text)
        expectValues(r, out, c.rtol, c.atol, c.name)
        return
      }
      if (c.fn === 'sturm') {
        expect(sturm(arg(c.inputs, 0), x).map(toText)).toEqual(c.output)
        return
      }
      if (c.fn === 'countRealRoots') {
        expect(countRealRoots(arg(c.inputs, 0), x, Number(c.inputs[1]), Number(c.inputs[2]))).toBe(c.output)
        return
      }
      if (c.fn === 'exactRoots') {
        const rs = exactRoots(arg(c.inputs, 0), x)
        expect(rs.reduce((n, r) => n + r.multiplicity, 0)).toBe(c.params['found'])
        const values = rs.flatMap((r) => Array.from({ length: r.multiplicity }, () => evaluateComplex(r.value)))
        const want = c.output as [number, number][]
        expect(values).toHaveLength(want.length)
        values.forEach(([re, im], k) => {
          const [wr, wi] = want[k] as [number, number]
          expect(Math.hypot(re - wr, im - wi)).toBeLessThan(c.atol + c.rtol * Math.hypot(wr, wi) + 1e-6)
        })
        return
      }
      throw new Error(`unknown fn ${c.fn}`)
    })
  }
})

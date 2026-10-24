import { describe, expect, it } from 'vitest'
import { evaluate, evaluateComplex } from './compile.ts'
import { add, mul } from './expr.ts'
import { toText } from './latex.ts'
import { parse } from './parse.ts'
import {
  coefficients,
  countRealRoots,
  degree,
  discriminant,
  exactRoots,
  leadingCoefficient,
  poly,
  polyDiv,
  polyGcd,
  polyLcm,
  polyToExpr,
  realRootIntervals,
  resultant,
  sturm,
} from './polynomials.ts'
import { expand, factor } from './simplify.ts'

const x = parse('x')
const t = (s: string): string => toText(factor(parse(s)))

describe('polynomials', () => {
  it('degree and coefficients', () => {
    expect(degree(parse('x^3*y + y^5'), x)).toBe(3)
    expect(degree(parse('0'))).toBe(Number.NEGATIVE_INFINITY)
    expect(coefficients(parse('a*x^2 + 3*x - 1'), x).map(toText)).toEqual(['a', '3', '-1'])
  })
  it('division, gcd and lcm', () => {
    const { quotient, remainder } = polyDiv(parse('x^3 + 2x + 1'), parse('x - 1'))
    expect([toText(quotient), toText(remainder)]).toEqual(['x**2 + x + 3', '4'])
    expect(toText(polyGcd(parse('x^2 - 1'), parse('x^2 - 3x + 2')))).toBe('x - 1')
    expect(toText(polyGcd(parse('x^2*y - y'), parse('x*y^2 + y^2')))).toBe('x*y + y')
    expect(toText(polyLcm(parse('x^2 - 1'), parse('x^2 - 3x + 2')))).toBe('x**3 - 2*x**2 - x + 2')
  })
  it('factors over the rationals', () => {
    expect(t('x^2 - 1')).toBe('(x - 1)*(x + 1)')
    expect(t('2*x^2 - 2')).toBe('2*(x - 1)*(x + 1)')
    expect(t('x^4 - 1')).toBe('(x - 1)*(x + 1)*(x**2 + 1)')
    expect(t('x^6 - 1')).toBe('(x - 1)*(x + 1)*(x**2 - x + 1)*(x**2 + x + 1)')
    expect(t('x^4 + 4')).toBe('(x**2 - 2*x + 2)*(x**2 + 2*x + 2)')
    expect(t('x^2 - 2')).toBe('x**2 - 2')
    expect(t('x^3 - x^2*y - x*y^2 + y^3')).toBe('(x - y)**2*(x + y)')
    expect(t('x^2 - y^2')).toBe('(x - y)*(x + y)')
    expect(t('1 - x')).toBe('1 - x')
    expect(t('x^8 - 1')).toBe('(x - 1)*(x + 1)*(x**2 + 1)*(x**4 + 1)')
    expect(t('(x^2 + 2x + 1)/(x^2 - 1)')).toBe('(x + 1)/(x - 1)')
  })
  it('finds exact roots', () => {
    const vals = (s: string): string[] => exactRoots(parse(s), x).map((r) => toText(r.value))
    expect(vals('x^2 - 3x + 2')).toEqual(['1', '2'])
    expect(vals('x^3 - x^2 - 2x + 2')).toEqual(['-sqrt(2)', '1', 'sqrt(2)'])
    expect(exactRoots(parse('(x - 1)^3'), x)).toHaveLength(1)
    // Every root of an irreducible cubic and quartic satisfies the polynomial numerically.
    for (const s of [
      'x^3 - 3x + 1',
      'x^3 + x + 1',
      'x^4 + x + 1',
      'x^4 - 10x^2 + 1',
      'x^4 + 2x^3 - x + 3',
      'x^5 - 2',
    ]) {
      const f = parse(s)
      const rs = exactRoots(f, x)
      expect(rs.reduce((n, r) => n + r.multiplicity, 0)).toBe(degree(f, x))
      for (const r of rs) {
        const v = evaluateComplex(r.value)
        const fx = evaluateComplex(f, { x: v })
        expect(Math.hypot(fx[0], fx[1])).toBeLessThan(1e-9)
      }
    }
  })
  it('Sturm sequences and root counts', () => {
    expect(sturm(parse('x^3 - 2x + 1'), x).map(toText)).toEqual(['x**3 - 2*x + 1', '3*x**2 - 2', '4*x/3 - 1', '5/16'])
    expect(countRealRoots(parse('x^3 - 2x + 1'), x)).toBe(3)
    expect(countRealRoots(parse('x^2 + 1'), x)).toBe(0)
    expect(countRealRoots(parse('(x - 1)^2*(x + 2)'), x, 0, 1)).toBe(1)
    const iv = realRootIntervals(parse('x^3 - 2x + 1'), x)
    expect(iv).toHaveLength(3)
  })
  it('converts, divides in several variables and isolates real roots', () => {
    const p = poly(parse('x^2*y + 3*x + sqrt(2)'), x)
    expect(p.gens.map(toText)).toEqual(['x', 'y', 'sqrt(2)'])
    expect(toText(polyToExpr(p))).toBe('x**2*y + 3*x + sqrt(2)')
    expect(toText(leadingCoefficient(parse('a*x^3 + x'), x))).toBe('a')
    expect(() => poly(parse('1/x'), x)).toThrow()
    expect(() => poly(parse('sin(x)'), x)).toThrow()
    const { quotient, remainder } = polyDiv(parse('x^2*y + x*y^2 + y^2'), parse('x*y - 1'))
    expect(toText(expand(add(mul(quotient, parse('x*y - 1')), remainder)))).toBe('x**2*y + x*y**2 + y**2')
    for (const [lo, hi] of realRootIntervals(parse('x^4 - 10x^2 + 1'), x)) {
      const a = evaluate(lo, {})
      const b = evaluate(hi, {})
      const f = (t: number): number => t ** 4 - 10 * t ** 2 + 1
      expect(a <= b && f(a) * f(b) <= 0).toBe(true)
    }
  })
  it('resultants and discriminants', () => {
    expect(toText(resultant(parse('x^2 + y'), parse('x - y'), x))).toBe('y**2 + y')
    expect(toText(discriminant(parse('a*x^2 + b*x + c'), x))).toBe('-4*a*c + b**2')
    expect(toText(discriminant(parse('x^3 + p*x + q'), x))).toBe('-4*p**3 - 27*q**2')
  })
})

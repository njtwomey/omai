import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import {
  add,
  bigO,
  evaluate,
  mul,
  oo,
  pade,
  parse,
  removeO,
  residue,
  series,
  seriesCoefficients,
  seriesSteps,
  sym,
  taylor,
  toText,
} from './index.ts'

const x = sym('x')
const s = (e: string, x0: Parameters<typeof series>[2] = 0, n = 6): string => toText(series(parse(e), x, x0, n))

describe('series', () => {
  it('expands Taylor and Laurent series', () => {
    expect(s('sin(x)')).toBe('x - x**3/6 + x**5/120 + O(x**6)')
    expect(s('exp(x)', 0, 4)).toBe('1 + x + x**2/2 + x**3/6 + O(x**4)')
    expect(s('1/sin(x)', 0, 4)).toBe('1/x + x/6 + 7*x**3/360 + O(x**4)')
    expect(s('1/(x^2*(1 - x))', 0, 2)).toBe('x**(-2) + 1/x + 1 + x + O(x**2)')
    expect(s('tan(x)', 0, 6)).toBe('x + x**3/3 + 2*x**5/15 + O(x**6)')
  })
  it('expands at other points and at infinity', () => {
    expect(s('exp(x)', 1, 3)).toBe('E + E*(x - 1) + E*(x - 1)**2/2 + O((x - 1)**3, (x, 1))')
    expect(s('1/(x + 1)', oo, 4)).toBe('x**(-3) - 1/x**2 + 1/x + O(x**(-4), (x, oo))')
  })
  it('keeps logarithms of the variable', () => {
    expect(s('x*log(x) + x^2', 0, 3)).toBe('x*log(x) + x**2 + O(x**3)')
  })
  it('handles cancellation by computing more terms', () => {
    expect(s('(sin(x) - x)/x^3', 0, 3)).toBe('-1/6 + x**2/120 + O(x**3)')
  })
  it('gives Taylor polynomials, coefficients and residues', () => {
    expect(toText(taylor(parse('cos(x)'), x, 0, 5))).toBe('x**4/24 - x**2/2 + 1')
    expect(seriesCoefficients(parse('exp(x)'), x, 0, 3).coefficients.map(toText)).toEqual(['1', '1', '1/2'])
    expect(seriesCoefficients(parse('1/x + 1'), x, 0, 2).start).toBe(-1)
    expect(toText(residue(parse('cos(x)/x^3'), x))).toBe('-1/2')
    expect(toText(residue(parse('1/(x*(x - 1))'), x, 1))).toBe('1')
    expect(() => taylor(parse('1/x'), x)).toThrow()
  })
  it('computes Pade approximants', () => {
    const r = pade(parse('exp(x)'), x, 2, 2)
    for (const t of [-0.5, 0.1, 0.4])
      expect(Math.abs(evaluate(r, { x: t }) - Math.exp(t))).toBeLessThan(2e-3 * t ** 4 + 1e-12)
    expect(toText(pade(parse('1/(1 - x)'), x, 0, 1))).toBe('-1/(x - 1)')
  })
  it('absorbs terms in big-O', () => {
    expect(toText(add(parse('x^3 + x'), bigO(parse('x^2'), x)))).toBe('x + O(x**2)')
    expect(toText(mul(x, bigO(x, x)))).toBe('O(x**2)')
    expect(toText(bigO(parse('3*x^2 + x^5'), x))).toBe('O(x**2)')
    expect(toText(removeO(series(parse('sin(x)'), x, 0, 4)))).toBe('-x**3/6 + x')
  })
  it('records the expansions', () => {
    const d = seriesSteps(parse('exp(sin(x))'), x, 0, 4)
    expect(d.steps.map((st) => st.rule)).toEqual(['series of sin', 'series of exp'])
    expect(toText(d.result)).toBe('1 + x + x**2/2 + O(x**4)')
  })
  it('refuses essential singularities and branch points', () => {
    for (const e of ['exp(1/x)', 'sqrt(x)']) {
      try {
        series(parse(e), x)
        expect.unreachable()
      } catch (err) {
        expect(isWhiteboxError(err) && err.code === 'not-implemented').toBe(true)
      }
    }
  })
})

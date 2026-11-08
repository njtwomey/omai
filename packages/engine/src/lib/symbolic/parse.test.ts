import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { add, equals, mul, num, parse, parseLatex, sin, sym, toLatex, toText } from './index.ts'

const t = (s: string): string => toText(parse(s))
const l = (s: string): string => toText(parseLatex(s))

describe('parse', () => {
  it('reads SymPy syntax and carets', () => {
    expect(t('x**2 + x^3')).toBe('x**3 + x**2')
    expect(t('-x^2')).toBe('-x**2')
    expect(t('2^3^2')).toBe('512')
    expect(t('x/2/y')).toBe('x/(2*y)')
    expect(t('2x*(x + 1)')).toBe('2*x*(x + 1)')
    // A name followed by a parenthesis is a function application, as in SymPy.
    expect(t('2x(x + 1)')).toBe('2*x(x + 1)')
    expect(t('3!')).toBe('6')
    expect(t('log(8, 2)')).toBe('log(8)/log(2)')
    expect(t('Rational(3, 6) + S(1)')).toBe('3/2')
  })
  it('reads printed forms', () => {
    expect(t('Piecewise((x, x > 0), (-x, True))')).toBe('Piecewise((x, x > 0), (-x, True))')
    expect(t('Eq(x, 1)')).toBe('Eq(x, 1)')
    expect(t('(x > 0) & (x < 1)')).toBe('(x > 0) & (x < 1)')
    expect(t('~(x > 0)')).toBe('x <= 0')
    expect(t('O(x**3 + x**4)')).toBe('O(x**3)')
    expect(t('Integral(exp(x**2), (x, 0, 1))')).toBe('Integral(exp(x**2), (x, 0, 1))')
    expect(t('Derivative(f(x), (x, 2))')).toBe('Derivative(f(x), (x, 2))')
  })
  it('handles floats, exact decimals and bound symbols', () => {
    expect(t('0.5*x')).toBe('0.5*x')
    expect(toText(parse('0.5*x', { exact: true }))).toBe('x/2')
    const p = sym('p', { positive: true })
    expect(toText(parse('sqrt(p^2)', { symbols: { p } }))).toBe('p')
  })
  it('reports syntax errors', () => {
    for (const s of ['x +', '(x', 'x $ y', '']) {
      try {
        parse(s)
        expect.unreachable()
      } catch (e) {
        expect(isOmaiError(e) && e.code === 'value').toBe(true)
      }
    }
  })
})

describe('parseLatex', () => {
  it('reads fractions, roots and scripts', () => {
    expect(l('\\frac{x^{2}}{2} + \\sin{\\left(x \\right)}')).toBe('x**2/2 + sin(x)')
    expect(l('\\sqrt[3]{x + 1}')).toBe('(x + 1)**(1/3)')
    expect(l('x^2 y_1')).toBe('x**2*y_1')
    expect(l('2xy')).toBe('2*x*y')
    expect(l('\\dfrac{1}{x} \\cdot 3')).toBe('3/x')
    expect(l('e^{-x}')).toBe('exp(-x)')
    expect(l('\\alpha \\beta')).toBe('alpha*beta')
    expect(l('x^23')).toBe('3*x**2')
  })
  it('reads functions with and without brackets', () => {
    expect(l('\\sin 2x + \\cos x')).toBe('sin(2*x) + cos(x)')
    expect(l('\\sin^{2}{x}')).toBe('sin(x)**2')
    expect(l('\\sin^{-1} x')).toBe('asin(x)')
    expect(l('\\log_{2} 8')).toBe('log(8)/log(2)')
    expect(l('\\ln(x)')).toBe('log(x)')
    expect(l('\\operatorname{asin}{\\left(x \\right)}')).toBe('asin(x)')
    expect(l('\\arctan x')).toBe('atan(x)')
    expect(l('f{\\left(x,y \\right)}')).toBe('f(x, y)')
    expect(l('\\Gamma\\left(x\\right)')).toBe('gamma(x)')
  })
  it('reads delimiters, absolute values and relations', () => {
    expect(l('|x - 1| + \\left|{y}\\right|')).toBe('Abs(y) + Abs(x - 1)')
    expect(l('\\lfloor x \\rfloor + \\left\\lceil{y}\\right\\rceil')).toBe('ceiling(y) + floor(x)')
    expect(l('x \\leq 2')).toBe('x <= 2')
    expect(l('x = 1')).toBe('Eq(x, 1)')
    expect(l('x!')).toBe('factorial(x)')
    expect(l('\\pi + \\infty')).toBe('oo')
    expect(l('\\left[ x + 1 \\right] 2')).toBe('2*x + 2')
  })
  it('reports unsupported input', () => {
    expect(() => parseLatex('\\int x dx')).toThrow()
    expect(() => parseLatex('\\frac{1}{')).toThrow()
  })
})

describe('product chains', () => {
  it('build once, so a number times a sum times more stays a product (it used to distribute)', () => {
    const x = sym('x')
    const e = mul(num(2), add(x, sym('y')), sin(x))
    expect(equals(parse(toText(e)), e)).toBe(true)
    expect(equals(parseLatex(toLatex(e)), e)).toBe(true)
    expect(toText(parse('2*(x + y)'))).toBe(toText(mul(num(2), add(x, sym('y')))))
  })
})

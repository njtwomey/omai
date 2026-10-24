import { describe, expect, it } from 'vitest'
import { toLatex, toText } from './latex.ts'
import { parse } from './parse.ts'

const t = (s: string): string => toText(parse(s))

describe('canonical construction', () => {
  it('collects terms and coefficients', () => {
    expect(t('x + x')).toBe('2*x')
    expect(t('2x^2 + 3x - x')).toBe('2*x**2 + 2*x')
    expect(t('x*x^2')).toBe('x**3')
    expect(t('x/x')).toBe('1')
    expect(t('2*(x+1)')).toBe('2*x + 2')
    expect(t('-(x+1)')).toBe('-x - 1')
    expect(t('1 - x')).toBe('1 - x')
    expect(t('x - 1')).toBe('x - 1')
  })
  it('evaluates radicals', () => {
    expect(t('sqrt(8)')).toBe('2*sqrt(2)')
    expect(t('sqrt(2)*sqrt(3)')).toBe('sqrt(6)')
    expect(t('1/sqrt(2)')).toBe('sqrt(2)/2')
    expect(t('12^(2/3)')).toBe('2*18**(1/3)')
    expect(t('sqrt(-4)')).toBe('2*I')
    expect(t('(-1)^(1/3)')).toBe('(-1)**(1/3)')
    expect(t('I^2')).toBe('-1')
  })
  it('prints like sympy', () => {
    expect(t('x^2*cos(x) + x*sin(x) + exp(x) + 3')).toBe('x**2*cos(x) + x*sin(x) + exp(x) + 3')
    expect(t('x + exp(x) + log(x) + sin(x) + cos(x)')).toBe('x + exp(x) + log(x) + sin(x) + cos(x)')
    expect(t('x*y/z^2')).toBe('x*y/z**2')
    expect(t('(x+1)/(x-1)')).toBe('(x + 1)/(x - 1)')
    expect(t('-x/(y+1)')).toBe('-x/(y + 1)')
    expect(t('x^-2')).toBe('x**(-2)')
    expect(t('sqrt(2) + 2')).toBe('sqrt(2) + 2')
    expect(t('3 - sqrt(2)')).toBe('3 - sqrt(2)')
    expect(t('pi - x')).toBe('pi - x')
    expect(t('sin(pi/6)')).toBe('1/2')
    expect(t('cos(pi/12)')).toBe('sqrt(2)/4 + sqrt(6)/4')
    expect(t('log(x)/x')).toBe('log(x)/x')
    expect(t('x*exp(-x)')).toBe('x*exp(-x)')
  })
  it('prints latex like sympy', () => {
    const l = (s: string): string => toLatex(parse(s))
    expect(l('x^2 + 2*x + 1')).toBe('x^{2} + 2 x + 1')
    expect(l('x*sin(x)+cos(x)*x^2+exp(x)+3')).toBe(
      'x^{2} \\cos{\\left(x \\right)} + x \\sin{\\left(x \\right)} + e^{x} + 3',
    )
    expect(l('2*x/3')).toBe('\\frac{2 x}{3}')
    expect(l('-x/2')).toBe('- \\frac{x}{2}')
    expect(l('1/(x+1)')).toBe('\\frac{1}{x + 1}')
    expect(l('sin(x)^2')).toBe('\\sin^{2}{\\left(x \\right)}')
    expect(l('Abs(x)')).toBe('\\left|{x}\\right|')
    expect(l('atan(x)')).toBe('\\operatorname{atan}{\\left(x \\right)}')
    expect(l('sqrt(2)/2')).toBe('\\frac{\\sqrt{2}}{2}')
    expect(l('2*3^x')).toBe('2 \\cdot 3^{x}')
    expect(l('(x+1)^(1/3)')).toBe('\\sqrt[3]{x + 1}')
    expect(l('x^(-3/2)')).toBe('\\frac{1}{x^{\\frac{3}{2}}}')
  })
})

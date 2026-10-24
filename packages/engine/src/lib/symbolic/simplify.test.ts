import { describe, expect, it } from 'vitest'
import {
  apart,
  cancel,
  collect,
  compareExprs,
  countOps,
  expand,
  expandSteps,
  factor,
  parse,
  simplify,
  simplifySteps,
  stepsToLatex,
  sym,
  together,
  toText,
  trigSimplify,
  trigSimplifySteps,
} from './index.ts'

const x = sym('x')
const T = (e: ReturnType<typeof parse>): string => toText(e)

describe('expand', () => {
  it('expands products, powers and logs under assumptions', () => {
    expect(T(expand(parse('(x + 1)^3')))).toBe('x**3 + 3*x**2 + 3*x + 1')
    expect(T(expand(parse('exp(x + y)')))).toBe('exp(x)*exp(y)')
    expect(T(expand(parse('x^(y + 2)')))).toBe('x**(y + 2)')
    const p = sym('p', { positive: true })
    const q = sym('q', { positive: true })
    expect(T(expand(parse('log(p*q^2)', { symbols: { p, q } })))).toBe('log(p) + 2*log(q)')
    expect(T(expand(parse('sin(x*(x + 1))')))).toBe('sin(x**2 + x)')
  })
  it('records its steps', () => {
    const d = expandSteps(parse('x*(x + 1)^2'))
    expect(d.steps.map((s) => s.rule)).toEqual(['multinomial', 'distribute'])
    expect(stepsToLatex(d)).toContain('&=')
  })
})

describe('factor and friends', () => {
  it('factors over the rationals in several variables', () => {
    expect(T(factor(parse('x^4 - y^4')))).toBe('(x - y)*(x + y)*(x**2 + y**2)')
    expect(T(factor(parse('x^2*y^2 - 1')))).toBe('(x*y - 1)*(x*y + 1)')
    expect(T(factor(parse('2*x/(4*x^2 - 4)')))).toBe('x/(2*(x - 1)*(x + 1))')
    expect(T(factor(parse('x^5 + x + 1')))).toBe('(x**2 + x + 1)*(x**3 - x**2 + 1)')
  })
  it('collects by symbols and subexpressions', () => {
    expect(T(collect(parse('a*x + b*x + a*y + b*y'), x, sym('y')))).toBe('x*(a + b) + y*(a + b)')
    expect(T(collect(parse('a*sin(t) + b*sin(t) + c'), parse('sin(t)')))).toBe('c + (a + b)*sin(t)')
  })
  it('combines and splits fractions', () => {
    expect(T(together(parse('1/x + 1/x^2')))).toBe('(x + 1)/x**2')
    expect(T(cancel(parse('(x^2 + 2x + 1)/(x^2 - 1)')))).toBe('(x + 1)/(x - 1)')
    expect(T(apart(parse('(x^2 + 1)/(x^2 - 1)'), x))).toBe('1 - 1/(x + 1) + 1/(x - 1)')
    expect(T(apart(parse('1/((x - a)*(x + a))'), x))).toBe('-1/(2*a*(a + x)) + 1/(2*a*(-a + x))')
  })
})

describe('trigSimplify and simplify', () => {
  it('simplifies trigonometric identities', () => {
    expect(T(trigSimplify(parse('sin(x)^2 + cos(x)^2 + 1')))).toBe('2')
    expect(T(trigSimplify(parse('tan(x)*cos(x)')))).toBe('sin(x)')
    expect(T(trigSimplify(parse('cosh(x)^2 - sinh(x)^2')))).toBe('1')
    expect(T(trigSimplify(parse('cos(x)^2 - sin(x)^2')))).toBe('cos(2*x)')
    expect(trigSimplifySteps(parse('2*sin(x)*cos(x)')).steps.length).toBeGreaterThan(0)
    expect(T(trigSimplify(parse('x + 1')))).toBe('x + 1')
  })
  it('chooses the simplest rewrite', () => {
    expect(T(simplify(parse('(x^2 - 1)/(x - 1) + sin(x)^2 + cos(x)^2')))).toBe('x + 2')
    expect(T(simplify(parse('exp(x)*exp(y)')))).toBe('exp(x + y)')
    expect(T(simplify(parse('sin(sin(y)^2 + cos(y)^2)')))).toBe('sin(1)')
    const d = simplifySteps(parse('(x^2 - 1)/(x - 1)'))
    expect(d.steps.map((s) => s.rule)).toEqual(['cancel'])
    expect(countOps(parse('x^2 + 2*x + 1'))).toBe(4)
  })
  it('orders expressions as SymPy', () => {
    const xs = [parse('sin(x)'), parse('2'), parse('y'), parse('x^2'), parse('exp(x)')]
    expect(xs.sort(compareExprs).map(toText)).toEqual(['2', 'x**2', 'y', 'exp(x)', 'sin(x)'])
  })
})

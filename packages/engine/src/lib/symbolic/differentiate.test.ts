import { describe, expect, it } from 'vitest'
import {
  diff,
  diffSteps,
  fn,
  gradient,
  hessian,
  implicitDiff,
  implicitDiffSteps,
  integrate,
  jacobian,
  parse,
  stepsToLatex,
  sym,
  toText,
} from './index.ts'

const x = sym('x')
const t = (e: string, ...v: Parameters<typeof diff>[1][]): string => toText(diff(parse(e), ...v))

describe('diff', () => {
  it('applies the rules', () => {
    expect(t('x^2*sin(x)', 'x')).toBe('x**2*cos(x) + 2*x*sin(x)')
    expect(t('tan(x)', 'x')).toBe('tan(x)**2 + 1')
    expect(t('x^x', 'x')).toBe('x**x*(log(x) + 1)')
    expect(t('2^x', 'x')).toBe('2**x*log(2)')
    expect(t('Abs(x)', 'x')).toBe('sign(x)')
    expect(t('gamma(x)', 'x')).toBe('gamma(x)*polygamma(0, x)')
    expect(t('5', 'x')).toBe('0')
  })
  it('handles orders and mixed partials', () => {
    expect(t('x^5', ['x', 3])).toBe('60*x**2')
    expect(t('x^3*y^2', 'x', 'y')).toBe('6*x**2*y')
    expect(t('x^2', ['x', 0])).toBe('x**2')
    expect(() => diff(parse('x'), ['x', -1])).toThrow()
  })
  it('leaves undefined functions unevaluated', () => {
    const f = fn('f', x)
    expect(toText(diff(f, x))).toBe('Derivative(f(x), x)')
    expect(toText(diff(diff(f, x), x))).toBe('Derivative(f(x), (x, 2))')
    expect(toText(diff(parse('x*f(x)'), x))).toBe('x*Derivative(f(x), x) + f(x)')
  })
  it('differentiates piecewise expressions and integrals', () => {
    expect(t('Piecewise((x^2, x > 0), (-x, True))', 'x')).toBe('Piecewise((2*x, x > 0), (-1, True))')
    const F = integrate(parse('exp(t^2)'), sym('t'))
    expect(toText(F)).toBe('Integral(exp(t**2), t)')
    expect(toText(diff(F, sym('t')))).toBe('exp(t**2)')
    // Leibniz's rule: d/dx of the integral of exp(t^2) from 0 to x^2.
    expect(t('Integral(exp(t**2), (t, 0, x**2))', 'x')).toBe('2*x*exp(x**4)')
  })
  it('records rule steps', () => {
    const d = diffSteps(parse('sin(x^2)*x'), 'x')
    expect(d.steps.map((s) => s.rule)).toEqual(['power rule', 'chain rule', 'product rule'])
    expect(toText(d.result)).toBe('2*x**2*cos(x**2) + sin(x**2)')
    expect(stepsToLatex(d)).toContain('\\text{chain rule}')
  })
})

describe('gradient, jacobian, hessian', () => {
  it('builds them from partial derivatives', () => {
    expect(gradient(parse('x^2*y'), ['x', 'y']).map(toText)).toEqual(['2*x*y', 'x**2'])
    expect(jacobian([parse('x*y'), parse('x + y')], ['x', 'y']).map((r) => r.map(toText))).toEqual([
      ['y', 'x'],
      ['1', '1'],
    ])
    expect(hessian(parse('x^2*y + y^3'), ['x', 'y']).map((r) => r.map(toText))).toEqual([
      ['2*y', '2*x'],
      ['2*x', '6*y'],
    ])
  })
})

describe('implicitDiff', () => {
  it('differentiates curves', () => {
    const y = sym('y')
    expect(toText(implicitDiff(parse('x^2 + y^2 - 4'), y, x))).toBe('-x/y')
    expect(toText(implicitDiff(parse('Eq(x*y, 1)'), y, x))).toBe('-y/x')
    expect(toText(implicitDiff(parse('x^2 + y^2 - 4'), y, x, 2))).toBe('-(x**2 + y**2)/y**3')
    expect(implicitDiffSteps(parse('x^2 + y^2 - 4'), y, x, 2).steps.length).toBeGreaterThan(2)
  })
})

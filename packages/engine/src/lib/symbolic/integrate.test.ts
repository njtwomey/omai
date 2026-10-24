import { describe, expect, it } from 'vitest'
import { NUMERIC_FUNCTIONS } from '../../testing/symbolic.ts'
import { isWhiteboxError } from '../base/index.ts'
import {
  definiteIntegral,
  definiteIntegralSteps,
  diff,
  type Expr,
  evaluateComplex,
  integrate,
  integrateSteps,
  oo,
  parse,
  sym,
  toText,
} from './index.ts'

const x = sym('x')

/** d/dx F equals f at a few points (complex arithmetic, so it also checks off the real domain). */
function checkAntiderivative(f: Expr, F: Expr): void {
  for (const v of [0.31, 0.77, 1.4]) {
    const a = evaluateComplex(diff(F, x), { x: v }, NUMERIC_FUNCTIONS)
    const b = evaluateComplex(f, { x: v }, NUMERIC_FUNCTIONS)
    expect(Math.hypot(a[0] - b[0], a[1] - b[1])).toBeLessThan(1e-9 * Math.max(1, Math.hypot(b[0], b[1])))
  }
}

describe('integrate', () => {
  it('integrates a broad set of elementary integrands', () => {
    const cases = [
      'sec(x)',
      'csc(x)',
      'x^2*log(x)',
      'x*atan(x)',
      'sin(x)^4',
      'sin(x)^2*cos(x)^2',
      'sec(x)^3',
      'tan(x)^4',
      'cot(x)^2',
      '1/(x^3 + 1)',
      'exp(x)/(exp(x) + 1)',
      '1/(x*log(x))',
      'sin(2*x)*cos(3*x)',
      'x^3*exp(x^2)',
      'sqrt(x)',
      '1/sqrt(x)',
      'x/sqrt(1 - x^2)',
      'log(x)^2',
      'x^5*exp(x)',
      'exp(3*x)*sin(2*x)',
      '1/(x^2 + 4*x + 13)',
      '(3*x + 1)/(x^2 + 1)^2',
      'x^2/(x - 1)^3',
      'acos(x)',
      'asinh(x)',
      'sinh(2*x)',
      'tanh(x)',
      'erf(x)',
      '1/sqrt(4 - x^2)',
      'sqrt(x^2 + 9)',
      '2^x',
      'cos(x)^5',
      'x*sin(x)*cos(x)',
      'exp(-2*x^2 + x)',
    ]
    for (const s of cases) {
      const f = parse(s)
      const F = integrate(f, x)
      expect(toText(F), s).not.toContain('Integral')
      checkAntiderivative(f, F)
    }
  })

  it('returns unevaluated integrals where no rule applies', () => {
    expect(toText(integrate(parse('exp(x^2)'), x))).toBe('Integral(exp(x**2), x)')
    expect(toText(integrate(parse('x + sin(x)/x'), x))).toBe('x**2/2 + Integral(sin(x)/x, x)')
    // Irreducible quartic denominators need algebraic extensions (Rothstein-Trager): not attempted.
    expect(toText(integrate(parse('1/(x^4 + 1)'), x))).toBe('Integral(1/(x**4 + 1), x)')
  })

  it('handles symbolic coefficients', () => {
    const F = integrate(parse('a*x^2 + b*x + c'), x)
    expect(toText(F)).toBe('a*x**3/3 + b*x**2/2 + c*x')
    expect(toText(integrate(parse('exp(k*x)'), x))).toBe('exp(k*x)/k')
  })

  it('records the rules used', () => {
    const d = integrateSteps(parse('x*cos(x)'), x)
    expect(d.steps.some((s) => s.rule === 'integration by parts')).toBe(true)
    const sub = integrateSteps(parse('2*x*exp(x^2)'), x)
    expect(sub.steps.some((s) => s.rule.startsWith('substitution'))).toBe(true)
  })
})

describe('definiteIntegral', () => {
  it('evaluates exactly with limits at the ends', () => {
    expect(toText(definiteIntegral(parse('x^2'), x, 0, 3))).toBe('9')
    expect(toText(definiteIntegral(parse('exp(-x)'), x, 0, oo))).toBe('1')
    expect(toText(definiteIntegral(parse('1/(x^2 + 1)'), x, parse('-oo'), oo))).toBe('pi')
    expect(toText(definiteIntegral(parse('log(x)'), x, 1, parse('E')))).toBe('1')
    expect(toText(definiteIntegral(parse('a*x'), x, 0, 2))).toBe('2*a')
  })

  it('falls back to quadrature', () => {
    const v = definiteIntegral(parse('exp(x^2)'), x, 0, 1)
    expect(v.kind).toBe('float')
    expect(Math.abs((v.kind === 'float' ? v.value : 0) - 1.4626517459071817)).toBeLessThan(1e-10)
    expect(definiteIntegralSteps(parse('exp(x^2)'), x, 0, 1).steps.at(-1)?.rule).toContain('numeric fallback')
    const g = definiteIntegral(parse('exp(-x^2)'), x, parse('-oo'), oo, { functions: NUMERIC_FUNCTIONS })
    expect(toText(g)).toBe('sqrt(pi)')
  })

  it('rejects poles inside the interval', () => {
    try {
      definiteIntegral(parse('1/x^2'), x, -1, 1)
      expect.unreachable()
    } catch (e) {
      expect(isWhiteboxError(e) && e.code === 'domain').toBe(true)
    }
  })
})

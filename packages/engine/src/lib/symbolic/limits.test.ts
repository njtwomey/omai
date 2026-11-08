import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { limit, limitSteps, oo, parse, sym, toText } from './index.ts'

const x = sym('x')
const L = (e: string, x0: Parameters<typeof limit>[2], dir: Parameters<typeof limit>[3] = '+'): string =>
  toText(limit(parse(e), x, x0, dir))

describe('limit', () => {
  it('evaluates continuous expressions directly', () => {
    expect(L('x^2 + 1', 2)).toBe('5')
    expect(limitSteps(parse('x^2 + 1'), x, 2).steps[0]?.rule).toBe('direct substitution')
  })
  it('resolves indeterminate forms', () => {
    expect(L('sin(x)/x', 0)).toBe('1')
    expect(L('(1 - cos(x))/x^2', 0)).toBe('1/2')
    expect(L('x*log(x)', 0)).toBe('0')
    expect(L('x^x', 0)).toBe('1')
    expect(L('(1 + 1/x)^x', oo)).toBe('E')
    expect(L('x*exp(-x)', oo)).toBe('0')
    expect(L('log(x)/x', oo)).toBe('0')
    expect(L('sqrt(x^2 + x) - x', oo)).toBe('1/2')
    expect(L('(x^3 - 1)/(x - 1)', 1)).toBe('3')
    expect(L('x^2*exp(x)', parse('-oo'))).toBe('0')
  })
  it('takes one-sided and two-sided limits', () => {
    expect(L('1/x', 0, '+')).toBe('oo')
    expect(L('1/x', 0, '-')).toBe('-oo')
    expect(L('1/x^2', 0, '+-')).toBe('oo')
    expect(L('Abs(x)/x', 0, '-')).toBe('-1')
    expect(L('floor(x)', 1, '-')).toBe('0')
    try {
      limit(parse('1/x'), x, 0, '+-')
      expect.unreachable()
    } catch (e) {
      expect(isOmaiError(e) && e.code === 'value').toBe(true)
    }
  })
  it('keeps symbols', () => {
    expect(L('(exp(a*x) - 1)/x', 0)).toBe('a')
    expect(L('a/x', 0)).toBe('oo*sign(a)')
  })
  it("records l'Hopital's rule and the numeric check", () => {
    const d = limitSteps(parse('x*exp(-x)'), x, oo)
    expect(d.steps.some((s) => s.rule === "l'Hopital's rule")).toBe(true)
    expect(d.steps.at(-1)?.rule.startsWith('numeric check')).toBe(true)
  })
  it('reports oscillation', () => {
    expect(() => limit(parse('sin(x)'), x, oo)).toThrow()
  })
})

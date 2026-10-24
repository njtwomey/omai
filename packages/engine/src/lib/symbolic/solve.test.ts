import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import { evaluateComplex, parse, solve, solveLinearSystem, solveSteps, sym, toText } from './index.ts'

const x = sym('x')
const S = (e: string): string[] => solve(parse(e), x).map(toText)

describe('solve', () => {
  it('solves polynomial and rational equations', () => {
    expect(S('2*x + 3')).toEqual(['-3/2'])
    expect(S('x^2 - 5x + 6')).toEqual(['2', '3'])
    expect(S('x^2 + 1')).toEqual(['-I', 'I'])
    expect(S('x^2 - 2*x - 1')).toEqual(['1 - sqrt(2)', '1 + sqrt(2)'])
    expect(S('(x^2 - 1)/(x - 1)')).toEqual(['-1'])
    expect(S('1/x - 2')).toEqual(['1/2'])
    expect(S('a*x^2 + b*x + c')).toHaveLength(2)
  })
  it('solves equations in a function of x', () => {
    expect(S('exp(2*x) - 3*exp(x) + 2')).toEqual(['0', 'log(2)'])
    expect(S('2^x - 8')).toEqual(['3'])
    expect(S('log(x) - 1')).toEqual(['E'])
    expect(S('sin(x)')).toEqual(['0', 'pi'])
    expect(S('cos(x)')).toEqual(['pi/2', '3*pi/2'])
    expect(S('tan(x) - 1')).toEqual(['pi/4'])
    expect(S('sqrt(x) - 3')).toEqual(['9'])
    expect(S('exp(x)*(x - 1)')).toEqual(['1'])
  })
  it('solves sextics that factor', () => {
    const roots = solve(parse('x^6 - 1'), x).map((r) => evaluateComplex(r))
    expect(roots).toHaveLength(6)
    for (const [re, im] of roots) expect(Math.abs(Math.hypot(re, im) - 1)).toBeLessThan(1e-12)
  })
  it('accepts equations as relations and records steps', () => {
    expect(solve(parse('Eq(x^2, 4)'), x).map(toText)).toEqual(['-2', '2'])
    expect(solveSteps(parse('2^x - 8'), x).steps[0]?.rule).toBe('take a logarithm')
  })
  it('reports identities and unsupported equations', () => {
    try {
      solve(parse('x - x'), x)
      expect.unreachable()
    } catch (e) {
      expect(isWhiteboxError(e)).toBe(true)
    }
    expect(() => solve(parse('x - cos(x)'), x)).toThrow()
  })
})

describe('solveLinearSystem', () => {
  it('solves square, parametric and underdetermined systems', () => {
    const [y, z] = [sym('y'), sym('z')]
    const s = solveLinearSystem([parse('x + y - 3'), parse('Eq(x - y, 1)')], [x, y])
    expect([toText(s['x'] ?? x), toText(s['y'] ?? y)]).toEqual(['2', '1'])
    const p = solveLinearSystem([parse('a*x + y - 1'), parse('x - y')], [x, y])
    expect(toText(p['x'] ?? x)).toBe('1/(a + 1)')
    const u = solveLinearSystem([parse('x + y + z - 1'), parse('x - y')], [x, y, z])
    expect(toText(u['x'] ?? x)).toBe('1/2 - z/2')
    expect(u['z']).toBeUndefined()
  })
  it('rejects inconsistent and nonlinear systems', () => {
    expect(() => solveLinearSystem([parse('x - 1'), parse('x - 2')], [x])).toThrow()
    expect(() => solveLinearSystem([parse('x^2 - 1')], [x])).toThrow()
  })
})

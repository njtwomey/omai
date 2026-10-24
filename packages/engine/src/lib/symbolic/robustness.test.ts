// Inputs that once ran for seconds to minutes (found by the robustness property on random expressions): each now
// returns quickly, within the default work budget, with its result or its documented fallback; and the fallbacks
// themselves, forced by small budgets.
import { describe, expect, it } from 'vitest'
import { DEFAULT_STEPS, lastWork } from './_budget.ts'
import {
  diff,
  type Expr,
  evaluateComplex,
  expand,
  factor,
  integrate,
  integrateSteps,
  parse,
  series,
  simplify,
  simplifySteps,
  sym,
  toText,
  withWorkBudget,
} from './index.ts'

const x = sym('x')

/** Run f, checking that it stays within the default work budget and (generously) within 200 ms. */
function quick<T>(f: () => T): T {
  const t0 = performance.now()
  const out = f()
  expect(performance.now() - t0).toBeLessThan(200)
  expect(lastWork()).toBeLessThanOrEqual(DEFAULT_STEPS)
  return out
}

describe('inputs that used to run away', () => {
  it('integrate(log((x^3 + y + exp(sin(y)/2))^2 + 1)) (was 65 s): unevaluated', () => {
    const f = parse('log((x**3 + y + exp(sin(y)/2))**2 + 1)')
    expect(toText(quick(() => integrate(f, x)))).toBe('Integral(log((x**3 + y + exp(sin(y)/2))**2 + 1), x)')
  })

  it('integrate(log(x^6 y^6 (x + y + log(y^2 + 1))^2 + 1)) (was 2.2 s): unevaluated', () => {
    const f = parse('log(x**6*y**6*(x + y + log(y**2 + 1))**2 + 1)')
    expect(toText(quick(() => integrate(f, x)))).toBe(`Integral(${toText(f)}, x)`)
  })

  it('simplify and factor of x^6 (x + cos(x))^3 + y^2 + y (were 5.5 s each): irreducible', () => {
    const e = parse('x**6*(x + cos(x))**3 + y**2 + y')
    expect(toText(quick(() => simplify(e)))).toBe('x**6*(x + cos(x))**3 + y**2 + y')
    expect(toText(quick(() => factor(e)))).toBe('x**9 + 3*x**8*cos(x) + 3*x**7*cos(x)**2 + x**6*cos(x)**3 + y**2 + y')
  })

  it('series(log(exp(exp(x/2)) + 1)^3, x, 0, 3) (did not finish): the series, coefficients cancelled', () => {
    const s = quick(() => series(parse('log(exp(exp(x/2)) + 1)**3'), x, 0, 3))
    // SymPy prints the same series as log(1 + E)**3 + 3*E*x*log(1 + E)**2/(2 + 2*E) + x**2*(...) + O(x**3).
    expect(toText(s)).toBe(
      'log(1 + E)**3 + 3*E*x*log(1 + E)**2/(2*(1 + E)) + 3*x**2*(2*E*log(1 + E)**2 + exp(2)*log(1 + E)**2 + ' +
        '2*exp(2)*log(1 + E))/(8*(1 + 2*E + exp(2))) + O(x**3)',
    )
  })

  it('integrate and simplify of log((x + y)^12 + 1) (were minutes and 0.8 s): bounded', () => {
    const f = parse('log((x + y)**12 + 1)')
    expect(toText(quick(() => integrate(f, x)))).toBe('Integral(log((x + y)**12 + 1), x)')
    expect(toText(quick(() => simplify(f)))).toBe('log((x + y)**12 + 1)')
    expect(toText(quick(() => simplify(parse('log((x + y)**2*cos(x)**6 + 1)'))))).toBe('log((x + y)**2*cos(x)**6 + 1)')
  })
})

describe('fallbacks when the work runs out', () => {
  it('integrate keeps the terms it finished and leaves the rest unevaluated', () => {
    const f = parse('x + log((x + y)**12 + 1)')
    expect(toText(integrate(f, x))).toBe('x**2/2 + Integral(log((x + y)**12 + 1), x)')
    expect(toText(integrate(f, x, { steps: 2000 }))).toBe('x**2/2 + Integral(log((x + y)**12 + 1), x)')
    expect(toText(integrate(parse('x*sin(x)'), x, { steps: 50 }))).toBe('Integral(x*sin(x), x)')
    expect(toText(integrateSteps(parse('x*sin(x)'), x, { steps: 50 }).result)).toBe('Integral(x*sin(x), x)')
  })

  it('series throws convergence rather than return a series of the wrong order', () => {
    expect(() => series(parse('exp(sin(x))'), x, 0, 8, '+', { steps: 100 })).toThrow(
      expect.objectContaining({ code: 'convergence' }),
    )
  })

  it('simplify and factor return the best form found so far, equal in value', () => {
    const e = parse('(x^2 - 1)/(x - 1) + sin(x)^2 + cos(x)^2')
    const v = (r: Expr): [number, number] => evaluateComplex(r, { x: 0.3 })
    for (const steps of [10, 1000, 3000, 10_000, 100_000]) {
      const r = simplify(e, { steps })
      expect(v(r)[0]).toBeCloseTo(v(e)[0], 12)
      expect(toText(simplifySteps(e, { steps }).result)).toBe(toText(r))
    }
    expect(toText(simplify(e))).toBe('x + 2')
    expect(toText(factor(parse('x^4 - 1'), { steps: 100 }))).toBe('x**4 - 1')
    expect(toText(factor(parse('x^4 - 1')))).toBe('(x - 1)*(x + 1)*(x**2 + 1)')
  })

  it('operations without a fallback throw convergence; withWorkBudget raises the limits', () => {
    const big = parse('(a + b)*(c + d)*(e + f)*(g + h)*(i + j)*(k + l)*(m + n)*(o + p)*(q + r)*(s + t)*(u + v)')
    expect(() => expand(big)).toThrow(expect.objectContaining({ code: 'convergence' }))
    expect(withWorkBudget({ terms: 4096 }, () => expand(big)).kind).toBe('add')
    expect(() => withWorkBudget({ steps: 10 }, () => diff(parse('sin(x)^5'), x))).toThrow(
      expect.objectContaining({ code: 'convergence' }),
    )
    // Inside a budget, an operation with a fallback still takes it.
    expect(toText(withWorkBudget({ steps: 50 }, () => integrate(parse('x*sin(x)'), x)))).toBe('Integral(x*sin(x), x)')
  })
})

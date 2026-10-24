import { describe, expect, it } from 'vitest'
import { toArray } from '../tensor/index.ts'
import {
  and,
  compile,
  eq,
  evaluate,
  ge,
  gt,
  le,
  lt,
  ne,
  neg,
  not,
  or,
  parse,
  piecewise,
  sym,
  toLatex,
  toText,
} from './index.ts'

const x = sym('x')

describe('conditions and piecewise', () => {
  it('evaluates relations between numbers', () => {
    expect(toText(lt(1, 2))).toBe('True')
    expect(toText(ge(parse('pi'), 4))).toBe('False')
    expect(toText(eq(x, x))).toBe('True')
    expect(toText(ne(x, 1))).toBe('Ne(x, 1)')
    expect(toText(le(x, 1))).toBe('x <= 1')
  })
  it('simplifies connectives', () => {
    expect(toText(and(gt(x, 0), lt(1, 2)))).toBe('x > 0')
    expect(toText(or(gt(x, 0), lt(1, 2)))).toBe('True')
    expect(toText(not(not(gt(x, 0))))).toBe('x > 0')
    expect(toText(not(eq(x, 1)))).toBe('Ne(x, 1)')
    expect(toLatex(and(gt(x, 0), lt(x, 1)))).toBe('x > 0 \\wedge x < 1')
  })
  it('builds, prints and evaluates piecewise expressions', () => {
    const p = piecewise([parse('x^2'), lt(x, 1)], [neg(x), gt(x, 5)], 0)
    expect(toText(p)).toBe('Piecewise((x**2, x < 1), (-x, x > 5), (0, True))')
    expect(toLatex(p)).toBe(
      '\\begin{cases} x^{2} & \\text{for}\\: x < 1 \\\\- x & \\text{for}\\: x > 5 \\\\0 & \\text{otherwise} \\end{cases}',
    )
    expect(evaluate(p, { x: 0.5 })).toBe(0.25)
    expect(evaluate(p, { x: 7 })).toBe(-7)
    expect(evaluate(p, { x: 3 })).toBe(0)
    expect(toArray(compile(p, [x])([0.5, 3, 7]))).toEqual([0.25, 0, -7])
    expect(toText(piecewise([x, lt(1, 2)], 0))).toBe('x')
    expect(() => piecewise()).toThrow()
  })
})

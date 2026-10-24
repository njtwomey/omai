// Derivatives against SymPy's diff. Equality is textual (SymPy's printed form) and by value; implicit derivatives are
// compared by value only (SymPy simplifies idiff's result to an equal but different form).
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import {
  type DiffSpec,
  diff,
  type Expr,
  gradient,
  hessian,
  implicitDiff,
  jacobian,
  parse,
  sym,
  toText,
} from './index.ts'

const fixture = loadFixture('symbolic/differentiate')

function compute(fn: string, inputs: readonly unknown[]): Expr {
  const s = inputs.map(String)
  const e = fn === 'jacobian' ? parse('0') : parse(s[0] ?? '0')
  switch (fn) {
    case 'diff':
      return diff(e, s[1] ?? 'x')
    case 'diffMany': {
      const specs: DiffSpec[] = (s[1] ?? '').split(',').map((p) => {
        const [n, k] = p.split(':')
        return [n ?? 'x', Number(k)] as const
      })
      return diff(e, ...specs)
    }
    case 'gradient':
      return gradient(e, ['x', 'y'])[Number(s[1])] as Expr
    case 'hessian': {
      const [i, j] = (s[1] ?? '0,0').split(',').map(Number) as [number, number]
      return hessian(e, ['x', 'y'])[i]?.[j] as Expr
    }
    case 'jacobian': {
      const fs = (s[0] ?? '').split(';').map((f) => parse(f))
      const [i, j] = (s[1] ?? '0,0').split(',').map(Number) as [number, number]
      return jacobian(fs, ['x', 'y'])[i]?.[j] as Expr
    }
    case 'implicitDiff':
      return implicitDiff(e, sym('y'), sym('x'), Number(s[1]))
    default:
      throw new Error(`unknown fn ${fn}`)
  }
}

describe('differentiation (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const out = symbolicOutput(c)
      const r = compute(c.fn, c.inputs)
      if (c.params['exact'] === true) expect(toText(r)).toBe(out.text)
      expectValues(r, out, c.rtol, c.atol, c.name)
    })
  }
})

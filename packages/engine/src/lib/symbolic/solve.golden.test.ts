// Equations against SymPy's solve: equality means the same solution set by value ((re, im) pairs sorted real first);
// linear systems compare each unknown textually and by value.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import { evaluateComplex, parse, solve, solveLinearSystem, sym, toText } from './index.ts'

const fixture = loadFixture('symbolic/solve')

describe('solve (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const s = c.inputs.map(String)
      if (c.fn === 'solve') {
        const sols = solve(parse(s[0] ?? '0'), sym('x')).map((v) => evaluateComplex(v))
        const want = c.output as [number, number][]
        expect(sols.map(([re, im]) => [re.toFixed(9), im.toFixed(9)])).toEqual(
          want.map(([re, im]) => [re.toFixed(9), im.toFixed(9)]),
        )
        return
      }
      const out = symbolicOutput(c)
      const eqs = (s[0] ?? '').split(';').map((e) => parse(e))
      const vars = (s[1] ?? '').split(',').map((n) => sym(n))
      const sol = solveLinearSystem(eqs, vars)
      const v = sol[s[2] ?? 'x']
      if (v === undefined) throw new Error(`no value for ${s[2]}`)
      expect(toText(v)).toBe(out.text)
      expectValues(v, out, c.rtol, c.atol, c.name)
    })
  }
})

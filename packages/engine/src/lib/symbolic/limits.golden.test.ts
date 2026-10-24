// Limits against SymPy's limit: the value prints exactly as SymPy's result.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { symbolicOutput } from '../../testing/symbolic.ts'
import { type LimitDirection, limit, parse, sym, toText } from './index.ts'

const fixture = loadFixture('symbolic/limits')

describe('limits (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const out = symbolicOutput(c)
      const [e, point, dir] = c.inputs.map(String) as [string, string, LimitDirection]
      expect(toText(limit(parse(e), sym('x'), parse(point), dir))).toBe(out.text)
    })
  }
})

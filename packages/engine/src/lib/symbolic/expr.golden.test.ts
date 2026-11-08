// Canonical construction against SymPy's automatic evaluation. Equality here is textual: omai parses SymPy's
// input text and must print exactly what SymPy's str() prints for the evaluated expression; the values at the sample
// points must agree too.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import { parse, toText } from './index.ts'

const fixture = loadFixture('symbolic/expr')

describe('canonical construction (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${String(c.inputs[0])}`, () => {
      const out = symbolicOutput(c)
      const e = parse(String(c.inputs[0]))
      expect(toText(e)).toBe(out.text)
      expectValues(e, out, c.rtol, c.atol, c.name)
    })
  }
})

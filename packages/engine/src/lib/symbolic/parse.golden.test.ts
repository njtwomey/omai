// Text parsing against SymPy's parse_expr with convert_xor and implicit multiplication. Equality is textual (the
// printed canonical form) plus values at the sample points.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import { parse, toText } from './index.ts'

const fixture = loadFixture('symbolic/parse')

describe('parse (sympy parse_expr)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${String(c.inputs[0])}`, () => {
      const out = symbolicOutput(c)
      const e = parse(String(c.inputs[0]))
      expect(toText(e)).toBe(out.text)
      expectValues(e, out, c.rtol, c.atol, c.name)
    })
  }
})

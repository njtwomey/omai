// Symbolic matrices against SymPy's Matrix. Determinants and characteristic polynomials: textual equality with
// SymPy's (expanded) result. Inverses: values entry by entry. Eigenvalues: sorted (re, im) values with multiplicity.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import {
  charpoly,
  det,
  type ExprMatrix,
  eigenvals,
  evaluateComplex,
  inverse,
  matrix,
  parse,
  sym,
  toText,
} from './index.ts'

const fixture = loadFixture('symbolic/matrices')

function parseMatrix(src: string): ExprMatrix {
  return matrix(src.split(';').map((r) => r.split(',').map((v) => parse(v))))
}

describe('matrices (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const s = c.inputs.map(String)
      const m = parseMatrix(s[0] ?? '')
      if (c.fn === 'eigenvals') {
        const vals = eigenvals(m).flatMap((r) => Array.from({ length: r.multiplicity }, () => evaluateComplex(r.value)))
        const want = c.output as [number, number][]
        expect(vals.map(([re, im]) => [re.toFixed(8), (Math.abs(im) < 1e-10 ? 0 : im).toFixed(8)])).toEqual(
          want.map(([re, im]) => [re.toFixed(8), im.toFixed(8)]),
        )
        return
      }
      const out = symbolicOutput(c)
      if (c.fn === 'det') {
        const d = det(m)
        expect(toText(d)).toBe(out.text)
        expectValues(d, out, c.rtol, c.atol, c.name)
        return
      }
      if (c.fn === 'charpoly') {
        expect(toText(charpoly(m, sym('lam')))).toBe(out.text)
        return
      }
      const [i, j] = (s[1] ?? '0,0').split(',').map(Number) as [number, number]
      const v = inverse(m)[i]?.[j]
      if (v === undefined) throw new Error('missing entry')
      expectValues(v, out, c.rtol, c.atol, c.name)
    })
  }
})

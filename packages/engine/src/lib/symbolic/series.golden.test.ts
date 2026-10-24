// Series against SymPy: the truncated series prints exactly as SymPy's (with removeO) and has the same values; the
// order term prints as SymPy's getO(). Residues compare textually. Pade approximants compare coefficients with
// scipy.interpolate.pade (numerator and denominator, highest power first, normalised to q(0) = 1).
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectValues, symbolicOutput } from '../../testing/symbolic.ts'
import { cancel, type ExprOrder, evaluate, pade, parse, removeO, residue, series, sym, toText } from './index.ts'

const fixture = loadFixture('symbolic/series')
const x = sym('x')

describe('series (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const s = c.inputs.map(String)
      const e = parse(s[0] ?? '0')
      if (c.fn === 'series') {
        const out = symbolicOutput(c)
        const full = series(e, x, parse(s[1] ?? '0'), Number(s[2]))
        const body = removeO(full)
        expect(toText(body)).toBe(out.text)
        expectValues(body, out, c.rtol, c.atol, c.name)
        const order = (full.kind === 'add' ? full.args : [full]).find((t) => t.kind === 'order') as
          | ExprOrder
          | undefined
        expect(order === undefined ? 'None' : toText(order)).toBe(c.params['order'])
        return
      }
      if (c.fn === 'residue') {
        const out = symbolicOutput(c)
        expect(toText(residue(e, x))).toBe(out.text)
        return
      }
      if (c.fn === 'pade') {
        const r = cancel(pade(e, x, Number(c.inputs[1]), Number(c.inputs[2])))
        const want = c.output as { p: { data: number[] }; q: { data: number[] } }
        // Our result is p/q in lowest terms with any constant in front; compare the rational function by values.
        for (const t of [0.1, 0.3, -0.2, 0.45]) {
          const pw = want.p.data.reduce((acc, a) => acc * t + a, 0)
          const qw = want.q.data.reduce((acc, a) => acc * t + a, 0)
          expect(evaluate(r, { x: t })).toBeCloseTo(pw / qw, 10)
        }
        return
      }
      throw new Error(`unknown fn ${c.fn}`)
    })
  }
})

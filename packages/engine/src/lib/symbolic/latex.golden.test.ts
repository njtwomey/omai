// LaTeX against SymPy's latex() with default settings: textual equality. Each case also checks that the plain-text
// form equals SymPy's str(), and that parseLatex reads the LaTeX back to the same expression where the subset covers it.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { symbolicOutput } from '../../testing/symbolic.ts'
import { matrix, parse, parseLatex, toLatex, toText } from './index.ts'

const fixture = loadFixture('symbolic/latex')

/** Inputs whose LaTeX uses constructs outside the parseLatex subset (integrals, derivatives, cases, O, i). */
// x1 prints as x_{1}, which reads back as the symbol x_1: LaTeX cannot tell them apart.
const NOT_PARSED =
  /Integral|Derivative|Piecewise|O\(|I\*|Eq\(|>=|theta_hat|re\(|foo|f\(x, y\)|exp\(x\)\*\*y|\(x\*\*2\)\*\*y|x1/

describe('latex (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${String(c.inputs[0])}`, () => {
      const out = symbolicOutput(c)
      if (c.fn === 'latexMatrix') {
        const m = matrix(
          String(c.inputs[0])
            .split(';')
            .map((r) => r.split(',').map((v) => parse(v))),
        )
        expect(toLatex(m)).toBe(c.params['matrix'])
        return
      }
      const e = parse(String(c.inputs[0]))
      expect(toText(e)).toBe(out.text)
      expect(toLatex(e)).toBe(out.latex)
      if (!NOT_PARSED.test(String(c.inputs[0]))) expect(toText(parseLatex(out.latex))).toBe(out.text)
    })
  }
})

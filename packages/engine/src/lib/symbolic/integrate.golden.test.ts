// Integrals against SymPy's integrate. Antiderivatives: equality means "differ by a constant" (values minus SymPy's
// values are the same at every sample point), and differentiating ours gives back the integrand at the sample
// points. Definite integrals: the exact value prints as SymPy's.
import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { expectSameUpToConstant, NUMERIC_FUNCTIONS, symbolicOutput, valuesAt } from '../../testing/symbolic.ts'
import { definiteIntegral, diff, type Expr, integrate, parse, sym, toText } from './index.ts'

const fixture = loadFixture('symbolic/integrate')
const x = sym('x')

function hasIntegral(e: Expr): boolean {
  return toText(e).includes('Integral(')
}

describe('integration (sympy)', () => {
  for (const c of fixture.cases) {
    it(`${c.name}: ${c.inputs.map(String).join(' | ')}`, () => {
      const out = symbolicOutput(c)
      const s = c.inputs.map(String)
      const f = parse(s[0] ?? '0')
      if (c.fn === 'integrate') {
        const F = integrate(f, x)
        expect(hasIntegral(F), toText(F)).toBe(false)
        expectSameUpToConstant(F, out, 1e-8, c.name, f)
        const dF = valuesAt(diff(F, x), out.names)
        const fv = valuesAt(f, out.names)
        dF.forEach(([re, im], k) => {
          const [wr, wi] = fv[k] as [number, number]
          if (!Number.isFinite(wr)) return
          expect(Math.hypot(re - wr, im - wi)).toBeLessThan(1e-8 * Math.max(1, Math.hypot(wr, wi)))
        })
        return
      }
      const v = definiteIntegral(f, x, parse(s[1] ?? '0'), parse(s[2] ?? '0'), { functions: NUMERIC_FUNCTIONS })
      expect(toText(v)).toBe(out.text)
    })
  }
})

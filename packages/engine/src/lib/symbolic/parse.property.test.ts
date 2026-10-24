// Round trips through the printers and parsers: parse(toText(e)) and parseLatex(toLatex(e)) give back e exactly
// (structural equality of the canonical forms).
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { exprArb, productArb } from '../../testing/symbolic.ts'
import { equals, factor, parse, parseLatex, toLatex, toText } from './index.ts'

describe('printing and parsing', () => {
  it('parse inverts toText', () => {
    fc.assert(
      fc.property(exprArb, (e) => equals(parse(toText(e)), e)),
      { numRuns: 80 },
    )
  })

  it('parseLatex inverts toLatex', () => {
    fc.assert(
      fc.property(exprArb, (e) => equals(parseLatex(toLatex(e)), e)),
      { numRuns: 80 },
    )
  })

  it('round trips factored forms', () => {
    fc.assert(
      fc.property(productArb(), (p) => {
        const f = factor(p)
        return (
          equals(parse(toText(f)), parse(toText(parse(toText(f))))) && equals(parseLatex(toLatex(f)), parse(toText(f)))
        )
      }),
      { numRuns: 40 },
    )
  })
})

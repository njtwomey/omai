import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { syntaxTerm } from '../../testing/logic.ts'
import { compareTerms, parseTerm, type Term, termToString } from './index.ts'

describe('reader and writer', () => {
  it('round-trips random terms through the writer and the reader', () => {
    fc.assert(
      fc.property(syntaxTerm(), (t: Term) => {
        const text = termToString(t)
        expect(compareTerms(parseTerm(text), t), text).toBe(0)
        expect(compareTerms(parseTerm(termToString(t, { ignoreOps: true })), t)).toBe(0)
      }),
      { numRuns: 400 },
    )
  })
})

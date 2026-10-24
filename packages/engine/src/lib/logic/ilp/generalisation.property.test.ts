import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { plainTerm } from '../../../testing/logic.ts'
import { compareTerms, isVariant, matchTerm, parseClause, termToString } from '../index.ts'
import { isThetaEquivalent, lgg, lggClauses, reduceClause, thetaSubsumes } from './index.ts'

const ground = plainTerm(3, ['a1', 'b1']).map((t) => t)

describe('generalisation properties', () => {
  it('the lgg generalises both terms and is idempotent and symmetric', () => {
    fc.assert(
      fc.property(ground, ground, (a, b) => {
        const g = lgg(a, b)
        expect(matchTerm(g, a)).not.toBeNull()
        expect(matchTerm(g, b)).not.toBeNull()
        expect(compareTerms(lgg(a, a), a)).toBe(0)
        expect(isVariant(lgg(b, a), g)).toBe(true)
      }),
      { numRuns: 300 },
    )
  })

  it('is least general: any common generalisation found by anti-unifying with a third term subsumes it', () => {
    fc.assert(
      fc.property(ground, ground, ground, (a, b, c) => {
        // lgg(lgg(a, b), c) generalises lgg(a, b), and lgg is associative up to variants.
        const ab = lgg(a, b)
        const abc = lgg(ab, c)
        expect(matchTerm(abc, ab)).not.toBeNull()
        expect(isVariant(abc, lgg(a, lgg(b, c)))).toBe(true)
      }),
      { numRuns: 200 },
    )
  })

  it('clause lggs subsume both clauses and reduction preserves equivalence', () => {
    const lits = ['p(X, Y)', 'q(Y, a)', 'q(X, X)', 'r(Y)', 'p(Y, b)', 'r(a)']
    const clause = fc
      .tuple(fc.subarray(lits, { minLength: 1 }), fc.constantFrom('h(X)', 'h(Y)', 'h(a)'))
      .map(([body, head]) => parseClause(`${head} :- ${body.join(', ')}.`))
    fc.assert(
      fc.property(clause, clause, (c, d) => {
        const g = lggClauses(c, d)
        expect(thetaSubsumes(g, c)).not.toBeNull()
        expect(thetaSubsumes(g, d)).not.toBeNull()
        const r = reduceClause(g)
        expect(isThetaEquivalent(r, g)).toBe(true)
        for (let i = 0; i < r.body.length; i++) {
          const smaller = { head: r.head, body: r.body.filter((_, j) => j !== i) }
          expect(thetaSubsumes(r, smaller), termToString(r.head)).toBeNull()
        }
      }),
      { numRuns: 200 },
    )
  })
})

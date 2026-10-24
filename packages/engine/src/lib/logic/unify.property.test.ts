import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { plainTerm } from '../../testing/logic.ts'
import {
  applySubstitution,
  compareTerms,
  isVariant,
  martelliMontanari,
  mgu,
  type Substitution,
  type Term,
  termVariables,
} from './index.ts'

const same = (a: Term, b: Term): boolean => compareTerms(a, b) === 0

/** A random instance of a term: some of its variables replaced by random terms over other variables. */
const instancePair = fc
  .tuple(plainTerm(3, ['X', 'Y', 'Z']), plainTerm(2, ['U', 'V']), plainTerm(2, ['U', 'V']), plainTerm(2, ['U', 'V']))
  .map(([a, x, y, z]) => {
    const theta: Substitution = new Map([
      ['X', x],
      ['Y', y],
      ['Z', z],
    ])
    return { a, theta }
  })

describe('unification properties', () => {
  it('a unifier makes both sides identical, and agrees with Martelli-Montanari', () => {
    fc.assert(
      fc.property(plainTerm(3), plainTerm(3, ['X', 'Y', 'W']), (a, b) => {
        const s = mgu(a, b)
        const mm = martelliMontanari(a, b)
        expect(s === null).toBe(mm.unifier === null)
        expect(mgu(b, a) === null).toBe(s === null)
        if (s !== null && mm.unifier !== null) {
          expect(same(applySubstitution(a, s), applySubstitution(b, s))).toBe(true)
          expect(isVariant(applySubstitution(a, s), applySubstitution(a, mm.unifier))).toBe(true)
          // Idempotent: no bound variable occurs in a binding.
          for (const v of s.values()) for (const w of termVariables(v)) expect(s.has(w.name)).toBe(false)
        }
      }),
      { numRuns: 300 },
    )
  })

  it('is most general: every unifier factors through the mgu', () => {
    fc.assert(
      fc.property(instancePair, ({ a, theta }) => {
        const target = applySubstitution(a, theta)
        const s = mgu(a, target)
        expect(s).not.toBeNull()
        if (s === null) return
        // theta unifies a and target; it must equal s followed by theta on a's variables.
        for (const v of termVariables(a)) {
          expect(same(applySubstitution(applySubstitution(v, s), theta), applySubstitution(v, theta))).toBe(true)
        }
      }),
      { numRuns: 300 },
    )
  })
})

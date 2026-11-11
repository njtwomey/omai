// The catalogue of reasons: every reason is a refusal with a code, the sentence names each of its facts, and the code decides the kind.
import { describe, expect, it } from 'vitest'
import * as reasons from './_reasons.ts'
import { CODE_KIND, Refusal } from './_refusal.ts'

const catalogue = Object.entries(reasons) as [string, (...facts: string[]) => Refusal][]

describe('the reasons catalogue', () => {
  it('has a function per reason, each returning a Refusal that names every fact it was given', () => {
    expect(catalogue.length).toBeGreaterThan(150)
    for (const [name, reason] of catalogue) {
      const facts = Array.from({ length: reason.length }, (_, k) => `<fact${k}>`)
      const refusal = reason(...facts)
      expect(refusal, name).toBeInstanceOf(Refusal)
      for (const fact of facts) expect(refusal.message, name).toContain(fact)
      expect(Object.values(refusal.facts), name).toEqual(facts)
    }
  })

  it('gives the kind of an issue from the code, and the node a refusal names from its facts', () => {
    const missing = reasons.concentrationMissing('beta')
    expect([missing.code, missing.kind, missing.node]).toEqual(['missing', 'structure', 'beta'])
    expect(reasons.scaleNotPositive('y', -1).message).toBe("the scale of 'y' is -1: it must be positive")
    expect(reasons.randomWhereKnown('x').kind).toBe('node')
    expect(reasons.randomWhereKnown('x').node).toBeUndefined()
    expect(reasons.customOpaque('f').kind).toBe('opaque')
  })

  it('uses every code, and every code has a kind', () => {
    const used = new Set(
      catalogue.map(([, reason]) => reason(...Array.from({ length: reason.length }, () => 'x')).code),
    )
    expect([...used].sort()).toEqual(Object.keys(CODE_KIND).sort())
  })
})

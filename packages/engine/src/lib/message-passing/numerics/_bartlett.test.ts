// The Bartlett rule's cache of gamma rules: bounded over a long run (the degrees of freedom change at every update), and a rule that
// was dropped is made again with the same bits.
import { describe, expect, it } from 'vitest'
import { GAMMA_RULES_KEPT, gammaRulesHeld, wishartRule } from './_bartlett.ts'

const chol = Float64Array.from([1, 0, 0.3, 0.9])

describe('wishartRule', () => {
  it('holds at most GAMMA_RULES_KEPT gamma rules however many degrees of freedom are asked for', () => {
    for (let i = 0; i < 3 * GAMMA_RULES_KEPT; i++) wishartRule(4 + i * 0.013, chol, 2, 4)
    expect(gammaRulesHeld()).toBeLessThanOrEqual(GAMMA_RULES_KEPT)
    expect(gammaRulesHeld()).toBeGreaterThan(0)
  })

  it('gives the same bits for a rule that was dropped and made again as for the first time', () => {
    const first = wishartRule(7.25, chol, 2, 4)
    // enough other shapes to push the first one out
    for (let i = 0; i < 2 * GAMMA_RULES_KEPT; i++) wishartRule(10 + i * 0.031, chol, 2, 4)
    const again = wishartRule(7.25, chol, 2, 4)
    expect(Array.from(again.weight)).toEqual(Array.from(first.weight))
    expect(Array.from(again.lambda)).toEqual(Array.from(first.lambda))
    expect(again.logDetBias).toBe(first.logDetBias)
  })
})

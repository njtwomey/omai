import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { checkProof, proveFol } from './index.ts'

describe('first-order resolution against Pelletier and AIMA problems', () => {
  for (const c of loadFixture('logic/first-order').cases) {
    it(c.name, () => {
      const p = c.params
      const r = proveFol(p['axioms'] as string[], p['conjecture'] as string, { maxSteps: 3000 })
      expect(r.status).toBe(p['status'])
      if (r.status === 'theorem') expect(checkProof(r.proof)).toBe(true)
    })
  }
})

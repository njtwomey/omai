import { describe, expect, it } from 'vitest'
import { MAX_SEED, rerollSeed } from './seed.ts'

describe('rerollSeed', () => {
  it('gives a different seed in range, the same every time', () => {
    for (const s of [0, 1, 42, MAX_SEED]) {
      const next = rerollSeed(s)
      expect(next).not.toBe(s)
      expect(next).toBeGreaterThanOrEqual(0)
      expect(next).toBeLessThanOrEqual(MAX_SEED)
      expect(rerollSeed(s)).toBe(next)
    }
  })
})

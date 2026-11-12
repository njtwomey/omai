import { describe, expect, it } from 'vitest'
import { flipCoin } from './compute.ts'
import { FLIPS, STAGES } from './data.ts'

describe('the is-this-coin-fair computation', () => {
  it('returns plain data of the right shapes and survives a structured clone', () => {
    const said: string[] = []
    const r = flipCoin(1, (s) => said.push(s))
    expect(r.flips).toHaveLength(FLIPS)
    expect(r.grid).toHaveLength(101)
    expect(r.stages.map((s) => s.flips)).toEqual([...STAGES])
    for (const s of r.stages) expect(s.density).toHaveLength(101)
    expect(r.truth).toBeGreaterThanOrEqual(0.35)
    expect(r.truth).toBeLessThanOrEqual(0.8)
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
  })

  it('starts flat and sharpens as flips arrive', () => {
    const r = flipCoin(2)
    const first = r.stages[0]
    expect(first?.heads).toBe(0)
    expect(first?.mean).toBeCloseTo(0.5, 10)
    // a flat prior puts probability on 0.45..0.55 equal to its width
    expect(first?.fair).toBeCloseTo(0.1, 10)
    const widths = r.stages.map((s) => s.hi - s.lo)
    expect(widths[0]).toBeCloseTo(0.9, 10)
    expect(widths.at(-1) as number).toBeLessThan(0.25)
  })

  it('has heads counts that add up and posterior means that follow them', () => {
    const r = flipCoin(3)
    for (const s of r.stages) {
      expect(s.heads).toBe(r.flips.slice(0, s.flips).reduce((a, b) => a + b, 0))
      // the mean of a Beta(1 + heads, 1 + tails)
      expect(s.mean).toBeCloseTo((1 + s.heads) / (2 + s.flips), 10)
      expect(s.lo).toBeLessThan(s.mean)
      expect(s.hi).toBeGreaterThan(s.mean)
    }
  })

  it('flips a different coin for a different seed and the same coin for the same one', () => {
    expect(flipCoin(1).flips).not.toEqual(flipCoin(2).flips)
    expect(flipCoin(1)).toEqual(flipCoin(1))
  })
})

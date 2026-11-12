import { describe, expect, it } from 'vitest'
import { compare, type Size } from './compute.ts'
import { FIRST_SEED, simulate, tally, VISITORS } from './data.ts'

const SMALL: Size = { draws: 2000, checkpoints: [50, 200, VISITORS], bins: 20 }

describe('the which-button-wins computation', () => {
  it('simulates the same visitors for the same seed, and the counts the page shows before play', () => {
    const a = tally(simulate(FIRST_SEED))
    expect(a.visitors).toBe(VISITORS)
    expect(tally(simulate(FIRST_SEED))).toEqual(a)
    expect(compare(FIRST_SEED, undefined, SMALL).tally).toEqual(a)
  })

  it('returns plain data of the right shapes on a small run and survives a structured clone', () => {
    const said: string[] = []
    const r = compare(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(160)
    expect(r.densityA).toHaveLength(160)
    expect(r.densityB).toHaveLength(160)
    expect(r.difference.centres).toHaveLength(20)
    expect(r.difference.density).toHaveLength(20)
    expect(r.strip.visitors).toEqual([50, 200, VISITORS])
    expect(r.strip.better).toHaveLength(3)
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
  })

  it('has posteriors centred on the clicks, intervals around them and probabilities in [0, 1]', () => {
    const r = compare(2, undefined, SMALL)
    // the mean of a Beta(1 + clicks, 1 + misses)
    expect(r.rateA.mean).toBeCloseTo((1 + r.tally.clicksA) / (2 + r.tally.visitors), 10)
    expect(r.rateB.mean).toBeCloseTo((1 + r.tally.clicksB) / (2 + r.tally.visitors), 10)
    for (const i of [r.rateA, r.rateB, r.difference]) {
      expect(i.lo).toBeLessThan(i.mean)
      expect(i.hi).toBeGreaterThan(i.mean)
    }
    for (const p of [r.better, ...r.strip.better]) {
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
    }
    // the last point of the strip is the answer with all the visitors
    expect(r.strip.better.at(-1)).toBe(r.better)
  })

  it('says B is likely better when B really converts more, and the difference points the same way as the clicks', () => {
    const r = compare(3, undefined, SMALL)
    const higherB = r.tally.clicksB > r.tally.clicksA
    expect(r.better > 0.5).toBe(higherB)
    expect(r.difference.mean > 0).toBe(higherB)
  })
})

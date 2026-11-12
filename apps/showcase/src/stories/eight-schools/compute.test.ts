import { describe, expect, it } from 'vitest'
import { fit, type Size } from './compute.ts'
import { EFFECTS, ERRORS, NAMES } from './data.ts'

const SMALL: Size = { chains: 2, warmup: 150, draws: 100 }

describe('the eight-schools computation', () => {
  it('has the classic eight schools', () => {
    expect(NAMES).toHaveLength(8)
    expect(EFFECTS).toEqual([28, 8, -3, 7, -1, 1, 18, 12])
    expect(ERRORS).toEqual([15, 10, 16, 11, 9, 11, 10, 18])
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.schools).toHaveLength(8)
    expect(r.draws).toBe(200)
    expect(r.tauDraws).toHaveLength(200)
    r.schools.forEach((s, j) => {
      expect(s.estimate).toBe(EFFECTS[j])
      expect(s.pooled.lo).toBeLessThan(s.pooled.hi)
      expect(s.pull).toBeGreaterThan(0)
      expect(s.pull).toBeLessThan(1)
    })
    expect(r.mu.lo).toBeLessThan(r.mu.hi)
    expect(r.tau.lo).toBeGreaterThanOrEqual(0)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  })

  it('pulls the noisier schools harder and shrinks every estimate towards the common mean', () => {
    const r = fit(1, undefined, SMALL)
    const byError = [...r.schools].sort((a, b) => a.error - b.error)
    expect((byError.at(-1) as (typeof byError)[number]).pull).toBeGreaterThan(
      (byError[0] as (typeof byError)[number]).pull,
    )
    // The most extreme raw estimate is pulled in.
    expect(r.schools[0]?.pooled.mean).toBeLessThan(28)
    expect(r.schools[0]?.pooled.mean).toBeGreaterThan(r.mu.mean - 1)
  })

  it('draws a different sample for a different seed but fits the same data', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(a.tauDraws).not.toEqual(b.tauDraws)
    expect(a.schools.map((s) => s.estimate)).toEqual(b.schools.map((s) => s.estimate))
  })
})

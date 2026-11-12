import { describe, expect, it } from 'vitest'
import { fit } from './compute.ts'
import { ROWS, SHOWN_SWEEP } from './data.ts'

describe('the how-expectation-propagation-works computation', () => {
  it('returns plain data of the right shapes, and it survives a structured clone', () => {
    const said: string[] = []
    const r = fit(37, (s) => said.push(s))
    expect(r.rows).toBe(ROWS)
    expect(r.points).toHaveLength(ROWS)
    expect(r.clutter).toHaveLength(ROWS)
    // the first sweep has one update for the prior and one for each measurement
    expect(r.first).toHaveLength(ROWS + 1)
    expect(r.first[0]?.cavity).toBeNull()
    expect(r.beliefs.sweep).toHaveLength(r.sweeps)
    expect(r.beliefs.mean).toHaveLength(r.sweeps)
    expect(r.beliefs.sd).toHaveLength(r.sweeps)
    expect(r.beliefs.change).toHaveLength(r.sweeps)
    expect(r.standIns.length).toBeGreaterThanOrEqual(1)
    for (const s of r.standIns) expect(s.messages).toHaveLength(ROWS + 1)
    expect(r.exact.grid).toHaveLength(r.exact.density.length)
    expect(r.exact.grid).toHaveLength(r.exact.epDensity.length)
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('settles on the first seed, and the EP answer is close to the exact posterior', () => {
    const r = fit(37)
    expect(r.settled).toBe(true)
    expect(r.ep.sd).toBeGreaterThan(0)
    expect(Math.abs(r.ep.mean - r.exact.mean)).toBeLessThan(0.5 * r.exact.sd + 0.1)
    expect(Math.abs((r.logEvidence ?? Number.NaN) - r.exact.logEvidence)).toBeLessThan(0.5)
  }, 30000)

  it('works one update out by hand and gets the message the engine sent, on several seeds', () => {
    for (const seed of [37, 2, 3, 5, 8]) {
      const r = fit(seed)
      const t = r.taken
      if (t === undefined) continue
      expect(t.sweep).toBe(SHOWN_SWEEP)
      expect(t.factor).toBeGreaterThanOrEqual(1)
      expect(t.factor).toBeLessThanOrEqual(ROWS)
      // our new stand-in is the engine's, up to the quadrature the engine uses
      expect(t.engineGap).toBeLessThan(1e-4)
      // the cavity plus the new stand-in is the matched bell, which has the mean and sd of the tilted curve
      expect(t.cavity.precision + t.standIn.precision).toBeCloseTo(t.matched.precision, 8)
      expect(t.cavity.shift + t.standIn.shift).toBeCloseTo(t.matched.shift, 8)
      expect(t.matched.shift / t.matched.precision).toBeCloseTo(t.tilted.mean, 8)
      expect(1 / Math.sqrt(t.matched.precision)).toBeCloseTo(t.tilted.sd, 8)
      expect(t.divergence).toBeGreaterThanOrEqual(-1e-9)
    }
  }, 60000)

  it('is a pure function of the seed, and a different seed gives different measurements', () => {
    expect(fit(4).points).toEqual(fit(4).points)
    expect(fit(4).points).not.toEqual(fit(5).points)
  }, 30000)
})

import { describe, expect, it } from 'vitest'
import { fit, lastPassStart, type Size } from './compute.ts'
import { waits } from './data.ts'

const SMALL: Size = { rows: 120, grid: 40 }

describe('the step the last pass begins at', () => {
  it('is 0 without a restart, the step after the restart with one, and never past the last step', () => {
    expect(lastPassStart(0, 12)).toBe(0)
    expect(lastPassStart(5, 12)).toBe(6)
    // a restart on the very last sweep would leave no step to read: the pass begins at the last step
    expect(lastPassStart(12, 12)).toBe(12)
    expect(lastPassStart(30, 12)).toBe(12)
  })
})

describe('the two-humps computation', () => {
  it('has the 272 waiting times of Old Faithful', () => {
    const w = waits()
    expect(w).toHaveLength(272)
    expect(Math.min(...w)).toBe(43)
    expect(Math.max(...w)).toBe(96)
  })

  it('returns plain data of the right shapes on a small run, and finds two groups', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(40)
    expect(r.lower).toHaveLength(40)
    expect(r.upper).toHaveLength(40)
    expect(r.total).toHaveLength(40)
    expect(r.chance).toHaveLength(120)
    expect(r.chance.every((c) => c >= 0 && c <= 1)).toBe(true)
    expect(r.trace.sweep[0]).toBe(0)
    expect(r.trace.sweep.at(-1)).toBe(r.sweeps)
    expect(r.trace.lower).toHaveLength(r.trace.sweep.length)
    expect(r.trace.upper).toHaveLength(r.trace.sweep.length)
    expect(r.settled).toBe(true)
    // the lower group sits near 54 minutes and the upper near 80
    expect(r.means[0]).toBeGreaterThan(48)
    expect(r.means[0]).toBeLessThan(60)
    expect(r.means[1]).toBeGreaterThan(75)
    expect(r.means[1]).toBeLessThan(85)
    expect(r.shares[0] + r.shares[1]).toBeCloseTo(1, 5)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('reaches the same two groups from a different order of the eruptions', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(Math.abs(a.means[0] - b.means[0])).toBeLessThan(1)
    expect(Math.abs(a.means[1] - b.means[1])).toBeLessThan(1)
    // the chances are given in the order of the data, whatever the order the method saw
    expect(a.chance[0]).toBeCloseTo(b.chance[0] as number, 1)
  }, 30000)

  it('on all 272 waits, two orders of the eruptions give centres within a quarter of a minute of each other', () => {
    const a = fit(1)
    const b = fit(2)
    expect(a.chance).toHaveLength(272)
    expect(Math.abs(a.means[0] - b.means[0])).toBeLessThan(0.25)
    expect(Math.abs(a.means[1] - b.means[1])).toBeLessThan(0.25)
    expect(a.trace.sweep[0]).toBe(0)
  }, 60000)
})

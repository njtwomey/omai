import { describe, expect, it } from 'vitest'
import { fit, run } from './compute.ts'
import { FAILURE_RATE, HORIZON, lifetimes, TRUE_SCALE, TRUE_SHAPE, UNITS } from './data.ts'
import { medianLife, quantityOf, readData, readResult } from './reading.ts'

describe('the how-long-until-it-fails computation', () => {
  it('has forty units that are the same every time, some of them censored, and a true survival curve', () => {
    const a = lifetimes()
    expect(a.time).toHaveLength(UNITS)
    expect(a.event).toHaveLength(UNITS)
    expect(lifetimes().time).toEqual(a.time)
    const read = readData(a.time, a.event)
    expect(read.failed + read.censored).toBe(UNITS)
    expect(read.censored).toBeGreaterThanOrEqual(8)
    expect(read.failed).toBeGreaterThanOrEqual(8)
    // the generator's truth is the Weibull survival function with the stated rate and shape
    for (const t of [0.5, 3, 9]) expect(a.survival(t)).toBeCloseTo(Math.exp(-FAILURE_RATE * t ** TRUE_SHAPE), 10)
    expect(TRUE_SCALE).toBeCloseTo(FAILURE_RATE ** (-1 / TRUE_SHAPE), 10)
  })

  it('passes messages on a small run, agrees with the exact grid, and uses the censored units', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), { units: 24, draws: 400, times: 40, grid: 61 })
    expect(r.time).toHaveLength(24)
    expect(r.all.settled).toBe(true)
    expect(r.dropped.settled).toBe(true)
    // EP and the exact posterior agree on both unknowns
    for (const key of ['scale', 'shape'] as const) {
      expect(Math.abs(r.all[key].mean - r.exact[key].mean)).toBeLessThan(0.3 * r.exact[key].sd)
      expect(r.all[key].sd / r.exact[key].sd).toBeGreaterThan(0.8)
      expect(r.all[key].sd / r.exact[key].sd).toBeLessThan(1.25)
    }
    expect(Math.abs(r.exact.correlation)).toBeLessThan(1)
    // dropping the censored units makes the lifetimes look shorter
    expect(r.dropped.scale.mean).toBeLessThan(r.all.scale.mean)
    expect(readResult(r).median.dropped).toBeLessThan(readResult(r).median.fitted)
    // the belief about the scale sharpens as units arrive
    expect(r.sharpening.scale[0]).toBeCloseTo(2, 10)
    for (let i = 1; i < r.sharpening.scale.length; i++)
      expect(r.sharpening.scale[i] as number).toBeLessThan(r.sharpening.scale[i - 1] as number)
    // the curves are over the same times, in order, and the band holds the middle
    const c = r.curves
    expect(c.time).toHaveLength(40)
    expect(c.time.at(-1)).toBeCloseTo(HORIZON, 10)
    expect(c.middle[0]).toBeCloseTo(1, 10)
    for (let i = 0; i < c.time.length; i++) {
      expect(c.lower[i] as number).toBeLessThanOrEqual(c.upper[i] as number)
      expect(c.middle[i] as number).toBeGreaterThanOrEqual((c.lower[i] as number) - 1e-9)
      expect(c.middle[i] as number).toBeLessThanOrEqual((c.upper[i] as number) + 1e-9)
    }
    expect(said.length).toBeGreaterThanOrEqual(4)
    // the result crosses the worker boundary, so it must survive a structured clone
    expect(structuredClone(r)).toEqual(r)
  }, 60000)

  it('takes one update apart for a failure and for a censoring, and matches the engine for the real one', () => {
    const r = fit(1, undefined, { units: 24, draws: 200, times: 20, grid: 81 })
    const u = r.update
    expect(u.failed.scale).toHaveLength(81)
    for (const c of [u.failed, u.censored]) {
      expect(c.cavity).toHaveLength(81)
      expect(c.tilted).toHaveLength(81)
      expect(c.matched).toHaveLength(81)
    }
    const real = u.actual === 'failed' ? u.failed : u.censored
    const other = u.actual === 'failed' ? u.censored : u.failed
    expect(real.engineGap).toBeLessThan(1e-3)
    expect(other.engineGap).toBeUndefined()
    // a censoring moves the belief about the scale up and adds less precision than a failure at the same time
    expect(u.censored.after.mean).toBeGreaterThan(u.censored.before.mean)
    expect(u.censored.precisionScale).toBeLessThan(u.failed.precisionScale)
    // the tilted curves are densities over the scale: they integrate to about the cavity's mass of one
    const mass = (xs: readonly number[], ys: readonly number[]): number =>
      xs
        .slice(1)
        .reduce((s, x, i) => s + (x - (xs[i] as number)) * (((ys[i] as number) + (ys[i + 1] as number)) / 2), 0)
    expect(mass(u.failed.scale, u.failed.tilted)).toBeGreaterThan(0.95)
    expect(mass(u.failed.scale, u.failed.tilted)).toBeLessThan(1.05)
  }, 60000)

  it('runs the story at full size, recovers the truth within the band and reads it', () => {
    const r = run(2)
    expect(r.time).toHaveLength(UNITS)
    expect(r.all.settled).toBe(true)
    // the units were made with these, and the belief is within a few spreads of them
    expect(Math.abs(r.all.scale.mean - Math.log(TRUE_SCALE))).toBeLessThan(3 * r.all.scale.sd)
    expect(Math.abs(r.all.shape.mean - Math.log(TRUE_SHAPE))).toBeLessThan(3 * r.all.shape.sd)
    expect(r.curves.covered).toBeGreaterThan(0.8)
    const read = readResult(r)
    expect(read.meanOff).toBeLessThan(0.5)
    expect(read.scale.low).toBeLessThan(read.scale.value)
    expect(read.scale.high).toBeGreaterThan(read.scale.value)
    expect(read.median.truth).toBeCloseTo(medianLife(TRUE_SCALE, TRUE_SHAPE), 10)
    expect(quantityOf({ mean: 0, sd: 0 }).value).toBe(1)
    expect(structuredClone(r)).toEqual(r)
  }, 120000)
})

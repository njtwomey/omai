import { describe, expect, it } from 'vitest'
import { fit, type Size } from './compute.ts'
import { simulate } from './data.ts'

const SMALL: Size = { chains: 2, warmup: 200, draws: 100, lines: 10, grid: 20, cloud: 30 }

describe('the bayesian-linear-regression computation', () => {
  it('simulates the same twelve points every time', () => {
    const a = simulate()
    expect(a.x).toHaveLength(12)
    expect(a.y).toHaveLength(12)
    expect(simulate()).toEqual(a)
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(20)
    expect(r.mean).toHaveLength(20)
    expect(r.lower).toHaveLength(20)
    expect(r.upper).toHaveLength(20)
    expect(r.draws).toBe(200)
    expect(r.fan).toHaveLength(10)
    expect(r.fan.every((line) => line.length === 20)).toBe(true)
    expect(r.cloud.slope).toHaveLength(30)
    expect(r.cloud.intercept).toHaveLength(30)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  })

  it('has a band that contains the mean and a credible interval around the slope', () => {
    const r = fit(1, undefined, SMALL)
    r.grid.forEach((_, i) => {
      expect(r.lower[i]).toBeLessThanOrEqual(r.mean[i] as number)
      expect(r.upper[i]).toBeGreaterThanOrEqual(r.mean[i] as number)
    })
    expect(r.summary.slope.lo).toBeLessThan(r.summary.slope.hi)
    expect(r.summary.slope.mean).toBeGreaterThan(0)
  })

  it('draws different samples for a different seed but fits the same data', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(a.cloud.slope).not.toEqual(b.cloud.slope)
    expect(Math.abs(a.summary.slope.mean - b.summary.slope.mean)).toBeLessThan(0.3)
  })
})

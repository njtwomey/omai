import { describe, expect, it } from 'vitest'
import { draw, OBSERVED, type Size } from './compute.ts'

const SMALL: Size = { functions: 3, grid: 31 }

const nearest = (grid: readonly number[], x: number): number =>
  grid.reduce((best, g, k) => (Math.abs(g - x) < Math.abs((grid[best] as number) - x) ? k : best), 0)

describe('the functions-from-a-process computation', () => {
  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = draw(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(31)
    expect(r.data.x).toHaveLength(OBSERVED)
    expect(r.data.y).toHaveLength(OBSERVED)
    for (const crowd of [r.prior, r.posterior]) {
      expect(crowd.draws).toHaveLength(3)
      expect(crowd.draws.every((f) => f.length === 31)).toBe(true)
      expect(crowd.mean).toHaveLength(31)
      expect(crowd.lower).toHaveLength(31)
      expect(crowd.upper).toHaveLength(31)
      expect(crowd.sd).toHaveLength(31)
    }
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  })

  it('has a posterior that is tight at the points and never wider than the prior', () => {
    const r = draw(1, undefined, { functions: 3, grid: 121 })
    r.grid.forEach((_, i) => {
      expect(r.posterior.sd[i] as number).toBeLessThanOrEqual((r.prior.sd[i] as number) + 1e-9)
    })
    r.data.x.forEach((x) => {
      expect(r.posterior.sd[nearest(r.grid, x)] as number).toBeLessThan(0.2)
    })
    expect(Math.max(...r.posterior.sd)).toBeGreaterThan(0.5)
  })

  it('draws different functions and points for a different seed', () => {
    const a = draw(1, undefined, SMALL)
    const b = draw(2, undefined, SMALL)
    expect(a.data.x).not.toEqual(b.data.x)
    expect(a.prior.draws).not.toEqual(b.prior.draws)
  })
})

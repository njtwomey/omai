import { describe, expect, it } from 'vitest'
import { type Build, engineOf, fit, type Groups, piecesOf, type Shape, type Size } from './compute.ts'
import { eruptions } from './data.ts'
import { correlation, readResult } from './reading.ts'

const SMALL: Size = { rows: 70, warmup: 6, draws: 8, ellipse: 12 }
const SHAPES: readonly Shape[] = ['round', 'diagonal', 'full']
const GROUPS: readonly Groups[] = ['two', 'three', 'open6', 'open8']
const BUILDS: readonly Build[] = SHAPES.flatMap((shape) => GROUPS.map((groups): Build => ({ shape, groups })))

describe('the data', () => {
  it('has the 272 eruptions of Old Faithful: a length and a wait each', () => {
    const e = eruptions()
    expect(e.length).toHaveLength(272)
    expect(e.wait).toHaveLength(272)
    expect(Math.min(...e.wait)).toBe(43)
    expect(Math.max(...e.length)).toBeCloseTo(5.1, 1)
  })
})

describe('which method runs a choice', () => {
  it('is expectation propagation where it settles quickly (round pairs, diagonal fixed numbers of groups), and sampling for the rest', () => {
    const ep = BUILDS.filter((b) => engineOf(b) === 'ep')
    expect(ep).toEqual([
      { shape: 'round', groups: 'two' },
      { shape: 'diagonal', groups: 'two' },
      { shape: 'diagonal', groups: 'three' },
    ])
  })
})

describe('the mixture computation, for every choice', () => {
  for (const build of BUILDS) {
    it(`${build.shape} groups, ${build.groups}: right shapes, groups by weight, survives a structured clone`, () => {
      const said: string[] = []
      const r = fit(1, (s) => said.push(s), build, SMALL)
      const n = piecesOf(build.groups)
      expect(r.build).toEqual(build)
      expect(r.engine).toBe(engineOf(build))
      expect(r.pieces).toBe(n)
      expect(r.groups).toHaveLength(n)
      expect(r.group).toHaveLength(SMALL.rows)
      expect(r.sure).toHaveLength(SMALL.rows)
      // every eruption names one of the groups, and is that likely to be in it
      expect(r.group.every((g) => Number.isInteger(g) && g >= 0 && g < n)).toBe(true)
      expect(r.sure.every((s) => s >= 1 / n - 1e-9 && s <= 1 + 1e-9)).toBe(true)
      // the groups come heaviest first, and together hold every eruption
      const weights = r.groups.map((g) => g.weight)
      expect(weights).toEqual([...weights].sort((a, b) => b - a))
      expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
      for (const g of r.groups) {
        expect(g.ellipse.x).toHaveLength(SMALL.ellipse)
        expect(g.ellipse.y).toHaveLength(SMALL.ellipse)
        expect(g.spread.every((s) => Number.isFinite(s) && s > 0)).toBe(true)
        expect(Math.abs(g.tilt)).toBeLessThanOrEqual(1 + 1e-9)
        // only a full group can tilt
        if (build.shape !== 'full') expect(g.tilt).toBeCloseTo(0, 12)
      }
      expect(said.length).toBeGreaterThanOrEqual(3)
      // The result crosses the worker boundary, so it must survive a structured clone.
      expect(structuredClone(r)).toEqual(r)
    }, 120_000)
  }
})

describe('expectation propagation and the sampler on diagonal groups', () => {
  it('find the same two groups: weights within 0.05, centres within 0.15 minutes of length and 1 of wait, spreads within 15%', () => {
    const build: Build = { shape: 'diagonal', groups: 'two' }
    const size: Size = { rows: 272, warmup: 60, draws: 80, ellipse: 12 }
    const ep = fit(1, undefined, build, size, 'ep')
    const gibbs = fit(1, undefined, build, size, 'gibbs')
    expect([ep.engine, gibbs.engine]).toEqual(['ep', 'gibbs'])
    expect(ep.settled).toBe(true)
    for (const i of [0, 1]) {
      const [a, b] = [ep.groups[i], gibbs.groups[i]] as [(typeof ep.groups)[number], (typeof ep.groups)[number]]
      expect(Math.abs(a.weight - b.weight)).toBeLessThan(0.05)
      expect(Math.abs(a.centre[0] - b.centre[0])).toBeLessThan(0.15)
      expect(Math.abs(a.centre[1] - b.centre[1])).toBeLessThan(1)
      for (const j of [0, 1]) expect(Math.abs((a.spread[j] as number) / (b.spread[j] as number) - 1)).toBeLessThan(0.15)
    }
  }, 120_000)

  it('settle in a handful of sweeps whatever the order of the eruptions (the sweeps did not depend on the seed)', () => {
    const build: Build = { shape: 'diagonal', groups: 'two' }
    const sweeps = [1, 4, 5].map(
      (seed) => fit(seed, undefined, build, { rows: 272, warmup: 6, draws: 8, ellipse: 12 }, 'ep').sweeps,
    )
    expect(Math.max(...sweeps)).toBeLessThanOrEqual(6)
  }, 120_000)
})

describe('reading a result', () => {
  it('counts the groups of real weight and finds the lean of the eruptions in the biggest one', () => {
    const e = eruptions()
    const r = fit(1, undefined, { shape: 'diagonal', groups: 'two' }, { rows: 272, warmup: 6, draws: 8, ellipse: 12 })
    const read = readResult(r, e.length, e.wait)
    expect(read.rows).toBe(272)
    expect(read.real.length + read.small).toBe(2)
    expect(read.real[0]?.count).toBeGreaterThan(100)
    expect(read.real[0]?.lean).toBeGreaterThan(0)
  }, 120_000)

  it('computes a correlation', () => {
    expect(correlation([1, 2, 3, 4], [2, 4, 6, 8])).toBeCloseTo(1, 10)
    expect(correlation([1, 2, 3, 4], [8, 6, 4, 2])).toBeCloseTo(-1, 10)
    expect(correlation([1, 1, 1], [1, 2, 3])).toBe(0)
  })
})

import { describe, expect, it } from 'vitest'
import { DEFAULT_SIZE, fit, type Size } from './compute.ts'
import { simulate } from './data.ts'
import { readResult } from './reading.ts'

const SMALL: Size = { grid: 20, cuts: 12, chains: 2, iterations: 600, warmup: 100, thin: 10, curves: 6 }

describe('the trees-one-split-at-a-time computation', () => {
  it('simulates the same sixty points every time', () => {
    const a = simulate()
    expect(a.x).toHaveLength(60)
    expect(a.y).toHaveLength(60)
    expect(simulate()).toEqual(a)
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(20)
    expect(r.greedy.stages).toHaveLength(r.greedy.cuts.length + 1)
    for (const stage of r.greedy.stages) {
      expect(stage.mean).toHaveLength(20)
      expect(stage.lower).toHaveLength(20)
      expect(stage.leaves).toHaveLength(stage.cuts.length + 1)
    }
    expect(r.posterior.thresholds).toHaveLength(12)
    expect(r.posterior.usage).toHaveLength(12)
    expect(r.posterior.mean).toHaveLength(20)
    // the chances are probabilities: a cut at a candidate, and a cut near a greedy cut (not a sum of expected uses)
    expect(r.posterior.cutChance).toHaveLength(12)
    for (const p of [...r.posterior.cutChance, ...r.greedy.cuts.map((c) => c.chanceNear)]) {
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
    }
    for (const c of r.greedy.cuts) {
      const window = r.posterior.thresholds.reduce(
        (a, t, i) => a + (Math.abs(t - c.threshold) <= 0.3 ? (r.posterior.usage[i] as number) : 0),
        0,
      )
      expect(c.chanceNear).toBeLessThanOrEqual(window + 1e-12)
    }
    expect(r.posterior.splits.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(r.sampled.curves).toHaveLength(6)
    expect(r.sampled.curves.every((c) => c.length === 20)).toBe(true)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
    expect(readResult(r).firstCut).toBeGreaterThan(0)
  })

  it('finds the first step of the data, and the step size is one over the precision', () => {
    const r = fit(2, undefined, SMALL)
    expect(Math.abs((r.greedy.cuts[0]?.threshold ?? 0) - 1.4)).toBeLessThan(0.3)
    const leaves = r.greedy.stages.at(-1)?.leaves ?? []
    for (const leaf of leaves) expect(leaf.stepSize).toBeCloseTo(1 / (1 + leaf.n), 10)
  })

  it('draws different trees for a different seed, with the same exact answer', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(a.sampled.curves).not.toEqual(b.sampled.curves)
    expect(a.posterior).toEqual(b.posterior)
  })

  it('has a sampler that agrees with the exact posterior at the real size', () => {
    const r = fit(3, undefined, DEFAULT_SIZE)
    expect(r.sampled.rootGap).toBeLessThan(0.08)
    expect(r.sampled.curveGap).toBeLessThan(0.1)
    expect(r.posterior.splits[2]).toBeGreaterThan(0.5)
  })
})

import { clutterTruth } from 'omai-engine/probprog/models'
import { describe, expect, it } from 'vitest'
import { run, solve } from './compute.ts'
import { clearReadings, DECOY, DECOY_READINGS, withDecoys } from './data.ts'
import { humpsOf, readCase } from './reading.ts'

describe('the signal-in-the-clutter computation', () => {
  it('has twelve readings that are the same every time, and sixteen with the decoy group', () => {
    const a = clearReadings()
    expect(a.values).toHaveLength(12)
    expect(a.kind.filter((k) => k === 1)).toHaveLength(4)
    expect(clearReadings().values).toEqual(a.values)
    const b = withDecoys(5)
    expect(b.values).toHaveLength(12 + DECOY_READINGS)
    expect(b.values.slice(0, 12)).toEqual(a.values)
    // the extra readings lie near the decoy
    expect(b.values.slice(12).every((v) => Math.abs(v - DECOY) < 4)).toBe(true)
    expect(withDecoys(6).values.slice(12)).not.toEqual(b.values.slice(12))
  })

  it('finds the value among the clutter and agrees with the exact curve on the clear readings', () => {
    const said: string[] = []
    const c = solve(clearReadings().values, (s) => said.push(s), { record: true, grid: 61 })
    expect(c.readings).toHaveLength(12)
    expect(c.clutter).toHaveLength(12)
    expect(c.clutter.every((p) => p >= 0 && p <= 1)).toBe(true)
    expect(c.exact.curve.x).toHaveLength(61)
    expect(c.exact.curve.density).toHaveLength(61)
    expect(c.settled).toBe(true)
    expect(Math.abs(c.bell.mean - 3)).toBeLessThan(1)
    expect(Math.abs(c.bell.mean - c.exact.mean)).toBeLessThan(0.2)
    expect(c.bell.sd / c.exact.sd).toBeGreaterThan(0.8)
    expect(c.bell.sd / c.exact.sd).toBeLessThan(1.2)
    expect(c.record).toBeDefined()
    expect(said.length).toBeGreaterThanOrEqual(2)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(c)).toEqual(c)
  }, 30000)

  it('runs both cases from a seed and reads them', () => {
    const r = run(2)
    expect(r.clear.readings).toHaveLength(12)
    expect(r.ambiguous.readings).toHaveLength(12 + DECOY_READINGS)
    const read = readCase(r.clear)
    expect(read.humps.length).toBeGreaterThanOrEqual(1)
    expect(read.clutter + read.kept).toBe(12)
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('settles on the sixteen readings for a spread of seeds', () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const r = run(seed)
      expect(r.ambiguous.settled).toBe(true)
      expect(Number.isFinite(r.ambiguous.bell.mean)).toBe(true)
    }
  }, 60000)

  it('draws the exact curve at the resolution of the figure, so a narrow hump is not blurred', () => {
    // thirty readings that agree: a narrow hump (spread near 0.2), which a window of a quarter of a unit would flatten
    const readings = Array.from({ length: 30 }, (_, i) => 2 + 0.01 * (i - 15))
    const c = solve(readings, undefined, { record: false, grid: 241 })
    // the reference: the same exact posterior on a grid fifty times finer, read over a window of 0.006
    const fine = clutterTruth(readings, 40001).marginals['theta']
    if (fine === undefined) throw new Error('no exact posterior')
    const reference = c.exact.curve.x.map((x) => (fine.cdf(x + 0.003) - fine.cdf(x - 0.003)) / 0.006)
    const top = Math.max(...reference)
    expect(top).toBeGreaterThan(1.5)
    expect(Math.max(...c.exact.curve.density) / top).toBeGreaterThan(0.99)
    expect(Math.max(...c.exact.curve.density) / top).toBeLessThan(1.01)
    // and the mass is still one
    const step = (c.exact.curve.x[1] as number) - (c.exact.curve.x[0] as number)
    expect(c.exact.curve.density.reduce((a, b) => a + b, 0) * step).toBeCloseTo(1, 2)
  })

  it('counts the humps of a curve and the share each holds', () => {
    const x = Array.from({ length: 101 }, (_, i) => i / 10)
    const bump =
      (c: number, w: number): ((v: number) => number) =>
      (v) =>
        Math.exp(-0.5 * ((v - c) / w) ** 2)
    const two = humpsOf({ x, density: x.map((v) => bump(2, 0.5)(v) + 0.5 * bump(7, 0.5)(v)) })
    expect(two).toHaveLength(2)
    expect(two[0]?.at).toBeCloseTo(2, 1)
    expect(two[1]?.at).toBeCloseTo(7, 1)
    expect((two[0]?.share ?? 0) + (two[1]?.share ?? 0)).toBeCloseTo(1, 6)
    expect(two[0]?.share).toBeGreaterThan(0.6)
    expect(humpsOf({ x, density: x.map(bump(5, 1)) })).toHaveLength(1)
  })
})

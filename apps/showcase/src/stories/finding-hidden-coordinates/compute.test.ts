import { describe, expect, it } from 'vitest'
import { fit } from './compute.ts'
import { FIRST_SEED, measurements, OUTPUTS, POINTS } from './data.ts'

describe('the hidden-coordinates computation', () => {
  it('draws the data of the first seed before play, in the shape the run uses', () => {
    const rows = measurements(FIRST_SEED)
    expect(rows).toHaveLength(POINTS)
    expect(rows.every((r) => r.length === OUTPUTS)).toBe(true)
  })

  it('returns plain data of the right shapes on a small run, and survives a structured clone', () => {
    const said: string[] = []
    const size = { points: 8, outputs: 3, features: 4, sweeps: 12 }
    const r = fit(FIRST_SEED, (s) => said.push(s), size)
    expect(r.truth).toHaveLength(8)
    expect(r.recovered).toHaveLength(8)
    expect(r.spread).toHaveLength(8)
    expect(r.y).toHaveLength(8)
    expect(r.y.every((row) => row.length === 3)).toBe(true)
    expect(r.clean).toHaveLength(8)
    expect(r.grid).toHaveLength(61)
    expect(r.curves).toHaveLength(3)
    expect(r.curves.every((c) => c.length === 61)).toBe(true)
    expect(r.sweeps).toBeLessThanOrEqual(12)
    expect(Number.isFinite(r.residual)).toBe(true)
    expect(Number.isFinite(r.correlation)).toBe(true)
    expect(r.spread.every((s) => s > 0)).toBe(true)
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
  }, 60000)

  it('finds the coordinate and the curves on the first seed, at the full size', () => {
    const r = fit(FIRST_SEED)
    expect(r.correlation).toBeGreaterThan(0.95)
    expect(r.residual).toBeLessThan(0.15)
  }, 120000)
})

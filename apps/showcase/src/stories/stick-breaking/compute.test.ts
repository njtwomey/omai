import { describe, expect, it } from 'vitest'
import { fit, type Size } from './compute.ts'
import { CENTRES, drawNumbers, PIECES, ROWS } from './data.ts'
import { readPrior, readResult, readTruth, trueShares } from './reading.ts'

const SMALL: Size = { rows: 40, priorDraws: 3, grid: 20 }

describe('the stick-breaking computation', () => {
  it('draws the same 80 numbers every time, across the groups', () => {
    const a = drawNumbers()
    expect(a).toHaveLength(ROWS)
    expect(drawNumbers()).toEqual(a)
    expect(Math.min(...a)).toBeLessThan(CENTRES[0] + 2)
    expect(Math.max(...a)).toBeGreaterThan((CENTRES[3] as number) - 2)
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.prior).toHaveLength(3)
    expect(r.prior.every((row) => row.length === PIECES)).toBe(true)
    // each stick's pieces add up to one
    expect(r.prior.every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9)).toBe(true)
    expect(r.weights).toHaveLength(PIECES)
    expect(r.spread).toHaveLength(PIECES)
    expect(r.centres).toHaveLength(PIECES)
    expect(r.centreSpread).toHaveLength(PIECES)
    expect(Math.abs(r.weights.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9)
    expect(r.spread.every((s) => s >= 0)).toBe(true)
    expect(r.grid).toHaveLength(20)
    expect(r.curves).toHaveLength(PIECES)
    expect(r.curves.every((c) => c.length === 20)).toBe(true)
    expect(r.total).toHaveLength(20)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 60000)

  it('on the real run puts the weight on a few pieces, near the centres of the groups', () => {
    const r = fit(1)
    expect(r.settled).toBe(true)
    const read = readResult(r)
    expect(read.real.length).toBeGreaterThanOrEqual(3)
    expect(read.real.length).toBeLessThanOrEqual(6)
    expect(read.held).toBeGreaterThan(0.85)
    expect(read.worst).toBeLessThan(1)
    // the true shares are marked on the pieces that found a group
    const truth = readTruth(r)
    expect(truth.matched).toBeGreaterThanOrEqual(3)
    expect(trueShares(r).filter((t) => t !== undefined)).toHaveLength(truth.matched)
  }, 60000)

  it('has prior sticks whose first piece is larger than their later ones', () => {
    const r = fit(1, undefined, { rows: 20, priorDraws: 200, grid: 5 })
    const prior = readPrior(r.prior)
    expect(prior.first).toBeGreaterThan(prior.fifth)
    expect(prior.tail).toBeLessThan(0.2)
  }, 60000)
})

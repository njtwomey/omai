import { describe, expect, it } from 'vitest'
import { fit } from './compute.ts'
import { AHEAD, READINGS, simulate } from './data.ts'
import { readResult } from './reading.ts'

describe('the where-is-it-going computation', () => {
  it('has forty readings that are the same whatever the seed, and a future that is not', () => {
    const a = simulate(1)
    const b = simulate(2)
    expect(a.readings).toHaveLength(READINGS)
    expect(a.position).toHaveLength(READINGS + AHEAD)
    expect(a.readings).toEqual(b.readings)
    expect(a.position.slice(0, READINGS)).toEqual(b.position.slice(0, READINGS))
    expect(a.position.slice(READINGS)).not.toEqual(b.position.slice(READINGS))
  })

  it('returns plain data of the right shapes on a small run, and the band widens after the last reading', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), { steps: 20, ahead: 5, record: true })
    expect(r.readings).toHaveLength(20)
    expect(r.truth).toHaveLength(25)
    expect(r.beliefs).toHaveLength(25)
    expect(r.settled).toBe(true)
    const half = (i: number): number => ((r.beliefs[i]?.hi ?? 0) - (r.beliefs[i]?.lo ?? 0)) / 2
    expect(half(24)).toBeGreaterThan(half(19))
    expect(half(19)).toBeGreaterThan(half(10) * 0.9)
    expect(r.record).toBeDefined()
    expect(said.length).toBeGreaterThanOrEqual(2)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('on the real run follows the track more closely than the readings do', () => {
    const r = fit(1)
    const read = readResult(r)
    expect(r.beliefs).toHaveLength(READINGS + AHEAD)
    expect(read.pathError).toBeLessThan(read.readingError)
    expect(read.inside).toBeGreaterThan(0.75 * read.steps)
    expect(read.endHalf).toBeGreaterThan(read.lastHalf)
    // the estimate for the steps with readings does not depend on the seed
    const again = fit(2, undefined, { steps: READINGS, ahead: AHEAD, record: false })
    expect(again.beliefs[10]?.mean).toBeCloseTo(r.beliefs[10]?.mean ?? Number.NaN, 8)
  }, 60000)
})

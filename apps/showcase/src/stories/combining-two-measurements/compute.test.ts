import { describe, expect, it } from 'vitest'
import { MAX_REPLAY_STEPS } from '../../kit/replay.ts'
import { combine } from './compute.ts'
import { DIAL_SD, DIGITAL_SD, PRIOR } from './data.ts'

describe('the combining-two-measurements computation', () => {
  it('returns plain data of the right shapes and survives a structured clone', () => {
    const said: string[] = []
    const r = combine(1, (s) => said.push(s))
    expect(r.messages.map((m) => m.from)).toEqual(['prior', 'digital', 'dial'])
    expect(r.settled).toBe(true)
    expect(r.sweeps).toBeGreaterThanOrEqual(1)
    expect(r.changes.length).toBeGreaterThanOrEqual(1)
    expect(r.record.kind).toBe('modelRun')
    expect(r.record.steps.length).toBeGreaterThanOrEqual(2)
    expect(r.record.steps.length).toBeLessThanOrEqual(MAX_REPLAY_STEPS)
    expect(r.record.structure.nodes.map((n) => n.id)).toContain('factor:temperature')
    expect(said.length).toBeGreaterThanOrEqual(3)
    expect(structuredClone(r)).toEqual(r)
    // the whole record is a few kilobytes
    expect(JSON.stringify(r.record).length).toBeLessThan(20_000)
  })

  it('sends the messages the model says: the prior, and each reading with its error', () => {
    const r = combine(2)
    const [prior, digital, dial] = r.messages
    expect(prior?.mean).toBeCloseTo(PRIOR.mean, 9)
    expect(prior?.variance).toBeCloseTo(PRIOR.sd ** 2, 9)
    expect(digital?.mean).toBeCloseTo(r.digital, 9)
    expect(digital?.variance).toBeCloseTo(DIGITAL_SD ** 2, 9)
    expect(dial?.mean).toBeCloseTo(r.dial, 9)
    expect(dial?.variance).toBeCloseTo(DIAL_SD ** 2, 9)
  })

  it('ends with the product of the messages: precisions add, the mean is their weighted average', () => {
    const r = combine(3)
    const total = r.messages.reduce((s, m) => s + m.precision, 0)
    const mean = r.messages.reduce((s, m) => s + m.precision * m.mean, 0) / total
    expect(r.belief.mean).toBeCloseTo(mean, 9)
    expect(r.belief.sd).toBeCloseTo(1 / Math.sqrt(total), 9)
  })

  it('makes a different room for a different seed and the same one for the same seed', () => {
    expect(combine(1).truth).not.toBe(combine(2).truth)
    expect(combine(5)).toEqual(combine(5))
  })
})

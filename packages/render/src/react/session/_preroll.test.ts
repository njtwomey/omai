import { describe, expect, it } from 'vitest'
import { prerollDecision } from './_preroll.ts'

const base = { preroll: 15, maxMs: 250 }

describe('the preroll of a live run', () => {
  it('waits at the start until a quarter of a second of steps stands ahead, then plays', () => {
    expect(prerollDecision({ ...base, buffering: true, ahead: 0, waitedMs: 0 })).toBe('wait')
    expect(prerollDecision({ ...base, buffering: true, ahead: 14, waitedMs: 100 })).toBe('wait')
    expect(prerollDecision({ ...base, buffering: true, ahead: 15, waitedMs: 0 })).toBe('play')
    expect(prerollDecision({ ...base, buffering: true, ahead: 40, waitedMs: 0 })).toBe('play')
  })

  it('plays on what it has after the longest wait, but never on nothing', () => {
    expect(prerollDecision({ ...base, buffering: true, ahead: 2, waitedMs: 249 })).toBe('wait')
    expect(prerollDecision({ ...base, buffering: true, ahead: 2, waitedMs: 250 })).toBe('play')
    expect(prerollDecision({ ...base, buffering: true, ahead: 0, waitedMs: 5000 })).toBe('wait')
  })

  it('starts waiting when it has caught up with the head, and plays while there is a step ahead', () => {
    expect(prerollDecision({ ...base, buffering: false, ahead: 0, waitedMs: 0 })).toBe('starved')
    expect(prerollDecision({ ...base, buffering: false, ahead: 1, waitedMs: 0 })).toBe('play')
    expect(prerollDecision({ ...base, buffering: false, ahead: 30, waitedMs: 0 })).toBe('play')
  })
})

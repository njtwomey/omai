import { describe, expect, it } from 'vitest'
import { RingBuffer } from './ring.ts'
import { DEFAULT_STEPS, progressOf, startOf, statusOf } from './start.ts'

describe('start policies', () => {
  it('map the engine mode onto a policy, and let an explicit policy win', () => {
    expect(startOf(undefined, undefined)).toBe('play')
    expect(startOf(undefined, 'live')).toBe('play')
    expect(startOf(undefined, 'eager')).toBe('end')
    expect(startOf(undefined, { eager: 10 })).toBe('play')
    expect(startOf('first', 'eager')).toBe('first')
    expect(startOf('play', 'eager')).toBe('play')
  })

  it('say preparing, ready or running', () => {
    expect(statusOf(false, false)).toBe('preparing')
    expect(statusOf(false, true)).toBe('ready')
    expect(statusOf(true, false)).toBe('running')
    expect(statusOf(true, true)).toBe('running')
  })

  it('measure progress against the budget, and read 1 once the run has ended', () => {
    expect(progressOf(0, false, 200)).toBe(0)
    expect(progressOf(50, false, 200)).toBe(0.25)
    expect(progressOf(500, false, 200)).toBe(1)
    expect(progressOf(80, true, 200)).toBe(1)
    expect(progressOf(250, false)).toBe(250 / DEFAULT_STEPS)
    expect(progressOf(5, false, 0)).toBe(0)
  })
})

describe('RingBuffer', () => {
  it('holds the latest items, oldest first, and counts what it dropped', () => {
    const r = new RingBuffer<number>(3)
    expect([r.push(1), r.push(2), r.push(3)]).toEqual([false, false, false])
    expect(r.length).toBe(3)
    expect(r.push(4)).toBe(true)
    expect(r.push(5)).toBe(true)
    expect([r.at(0), r.at(1), r.at(2)]).toEqual([3, 4, 5])
    expect(r.last).toBe(5)
    expect(r.dropped).toBe(2)
    expect(r.at(3)).toBeUndefined()
    expect(r.at(-1)).toBeUndefined()
  })

  it('has room for at least one item, and clears', () => {
    const r = new RingBuffer<string>(0)
    r.push('a')
    r.push('b')
    expect(r.length).toBe(1)
    expect(r.at(0)).toBe('b')
    r.clear()
    expect(r.length).toBe(0)
    expect(r.last).toBeUndefined()
    expect(r.dropped).toBe(0)
  })
})

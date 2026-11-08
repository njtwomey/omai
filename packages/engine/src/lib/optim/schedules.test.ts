import { describe, expect, it } from 'vitest'
import * as O from './index.ts'

describe('schedules without a library reference', () => {
  it('inverse-time decay follows init / (1 + rate t / steps)', () => {
    const s = O.inverseTimeDecay(2, 0.5, 4)
    for (const t of [0, 1, 7, 100]) expect(s(t)).toBeCloseTo(2 / (1 + (0.5 * t) / 4), 15)
    const stair = O.inverseTimeDecay(2, 0.5, 4, true)
    expect([stair(3), stair(4), stair(9)]).toEqual([2, 2 / 1.5, 1])
  })

  it('inverse-square-root decay follows init / sqrt(1 + rate t / steps) and equals the root of inverse-time decay', () => {
    const s = O.inverseSqrtDecay(2, 0.5, 4)
    for (const t of [0, 1, 7, 100]) expect(s(t)).toBeCloseTo(2 / Math.sqrt(1 + (0.5 * t) / 4), 15)
    for (const t of [0, 3, 50]) expect(O.inverseSqrtDecay(1)(t)).toBeCloseTo(Math.sqrt(O.inverseTimeDecay(1, 1)(t)), 15)
  })

  it('constant and joined schedules', () => {
    expect(O.constantSchedule(3)(1e9)).toBe(3)
    expect(() => O.joinSchedules([O.constantSchedule(1)], [5])).toThrow(/one more/)
    const s = O.joinSchedules([O.constantSchedule(1), O.linearSchedule(0, 1, 10), O.constantSchedule(7)], [3, 13])
    expect([s(2), s(3), s(8), s(13)]).toEqual([1, 0, 0.5, 7])
  })

  it('reduces on a plateau of a maximised metric with absolute thresholds', () => {
    const p = O.reduceOnPlateau({ mode: 'max', thresholdMode: 'abs', threshold: 0.1, patience: 0, factor: 0.5 })
    let s = p.init(1)
    const rates: number[] = []
    for (const acc of [0.5, 0.65, 0.7, 0.9, 0.95]) {
      s = p.step(s, acc)
      rates.push(s.lr)
    }
    expect(rates).toEqual([1, 1, 0.5, 0.5, 0.25])
  })
})

import { describe, expect, it } from 'vitest'
import { session, tensorSeries } from '../../lib/trace/index.ts'
import { odeSystems } from './odeSystems.ts'

describe('odeSystems', () => {
  it('integrates predator and prey for 300 steps, logging the state, the step and the evaluations', () => {
    const s = session(odeSystems, { mode: 'eager', steps: 300 })
    expect(s.t).toBe(300)
    const y = tensorSeries(s.run(), 'y')
    expect(y.length).toBe(300)
    for (const e of y) {
      expect(e.shape).toEqual([2])
      expect((e.values ?? []).every((v) => v > 0)).toBe(true)
    }
    expect(s.state.evaluations).toBeGreaterThan(300 * 6)
    // Seeking rebuilds a step from a checkpoint with the same numbers.
    expect(s.stateAt(150).y).toEqual(tensorSeries(s.run(), 'y')[149]?.values)
  })

  it('runs every system with every method', () => {
    for (const system of ['lotka-volterra', 'lorenz', 'van-der-pol', 'damped-pendulum', 'sir'] as const)
      for (const method of ['RK45', 'RK23', 'DOP853'] as const) {
        const s = session(odeSystems, { mode: 'eager', steps: 20, config: { system, method } })
        expect(s.state.failed).toBe(false)
        expect(s.state.t).toBeGreaterThan(0)
      }
  })
})

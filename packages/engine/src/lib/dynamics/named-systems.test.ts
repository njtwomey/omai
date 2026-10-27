import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from '../tensor/index.ts'
import { fieldDivergence, fixedPoints } from './fields.ts'
import {
  dampedPendulumField,
  lorenzField,
  lotkaVolterraField,
  namedSystems,
  sirField,
  vanDerPolField,
} from './named-systems.ts'
import { solveIvp } from './ode/index.ts'

describe('named systems', () => {
  it('Lotka-Volterra conserves its first integral along a tight solution', () => {
    const f = lotkaVolterraField()
    const sol = solveIvp(f, [0, 30], [10, 5], { rtol: 1e-10, atol: 1e-12 })
    const v = (y: number[]): number =>
      0.075 * (y[0] ?? 0) - 1.5 * Math.log(y[0] ?? 1) + 0.1 * (y[1] ?? 0) - Math.log(y[1] ?? 1)
    const rows = sol.ys.map((y) => Array.from(toFlat(y)))
    const first = v(rows[0] ?? [])
    for (const r of rows) expect(Math.abs(v(r) - first)).toBeLessThan(1e-6)
    expect(Math.min(...rows.flat())).toBeGreaterThan(0)
  })

  it('Lorenz contracts volume at the rate -(sigma + 1 + beta) and rests at the origin', () => {
    expect(fieldDivergence(lorenzField(), [3, -2, 20])).toBeCloseTo(-(10 + 1 + 8 / 3), 12)
    expect(toFlat(lorenzField()(0, tensor([0, 0, 0])))).toEqual([0, 0, 0])
  })

  it('the pendulum rests at the bottom and the SIR shares always sum to one', () => {
    const fp = fixedPoints(dampedPendulumField(), { seeds: [[0.5, 0.2]] })
    expect(fp[0]?.type).toBe('stable-spiral')
    const sir = toFlat(sirField({ beta: 0.7, gamma: 0.2 })(0, tensor([0.6, 0.3, 0.1])))
    expect(sir.reduce((a, v) => a + v, 0)).toBeCloseTo(0, 15)
  })

  it('every named system is a field of its own dimension', () => {
    for (const s of Object.values(namedSystems)) {
      expect(toFlat(s.field(0, tensor([...s.y0]))).length).toBe(s.y0.length)
      expect(s.names.length).toBe(s.y0.length)
      expect(s.span[1]).toBeGreaterThan(s.span[0])
    }
    expect(toFlat(vanDerPolField(1)(0, tensor([1, 3])))).toEqual([3, -1])
  })
})

import { describe, expect, it } from 'vitest'
import { mul, neg, tensor, toFlat } from '../../tensor/index.ts'
import { denseSolution, event, fieldOf, hermiteSegment, odeProblem } from './problem.ts'
import { solveIvp } from './solve-ivp.ts'

describe('ODE problems and dense solutions', () => {
  it('binds parameters and validates spans', () => {
    const p = odeProblem((_t, y, k) => neg(mul(k, y)), [2], [0, 1], { params: 3 })
    expect(toFlat(fieldOf(p)(0, tensor([2])))).toEqual([-6])
    expect(toFlat(fieldOf(p, tensor(1))(0, tensor([2])))).toEqual([-2])
    expect(() => odeProblem((_t, y) => y, [1], [0, Number.POSITIVE_INFINITY])).toThrow()
  })

  it('looks up segments in both directions', () => {
    const fwd = solveIvp((_t, y) => neg(y), [0, 2], [1], { denseOutput: true, rtol: 1e-8, atol: 1e-10 })
    const back = solveIvp((_t, y) => neg(y), [2, 0], [Math.exp(-2)], { denseOutput: true, rtol: 1e-8, atol: 1e-10 })
    for (const t of [0, 0.3, 1.1, 2]) {
      expect(Math.abs((toFlat(fwd.sol?.at(t) ?? tensor([0]))[0] ?? 0) - Math.exp(-t))).toBeLessThan(1e-7)
      expect(Math.abs((toFlat(back.sol?.at(t) ?? tensor([0]))[0] ?? 0) - Math.exp(-t))).toBeLessThan(1e-7)
    }
    expect(fwd.sol?.many([0, 1]).shape).toEqual([2, 1])
    expect(back.sol?.tMin).toBe(0)
    expect(() => denseSolution([])).toThrow()
  })

  it('Hermite segments reproduce cubics', () => {
    // y = t^3 on [1, 2]: y' = 3 t^2.
    const s = hermiteSegment(1, tensor([1]), tensor([3]), 2, tensor([8]), tensor([12]))
    expect(toFlat(s.at(1.5))[0]).toBeCloseTo(3.375, 14)
  })

  it('events count up to their terminal number', () => {
    const sol = solveIvp((t) => tensor([Math.cos(t)]), [0, 20], [0], {
      rtol: 1e-9,
      atol: 1e-11,
      events: event((_t, y) => toFlat(y)[0] ?? 0, { terminal: 3 }),
    })
    expect(sol.status).toBe('event')
    expect(sol.tEvents[0]?.length).toBe(3)
  })
})

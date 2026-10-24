import { describe, expect, it } from 'vitest'
import { grad, jvp } from '../../autodiff/index.ts'
import { add, item, mul, neg, sliceAxis, stack, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { recorder } from '../../trace/index.ts'
import { event } from './problem.ts'
import { diffeqsolve, type OdeMethodName, odeint, odeSolver, solveIvp } from './solve-ivp.ts'

const decay = (_t: number, y: Value): Value => neg(y)
const oscillator = (_t: number, y: Value): Value => {
  const x = sliceAxis(y, 0, 0, 1)
  const v = sliceAxis(y, 0, 1, 2)
  return stack([sum(v), neg(sum(x))])
}

describe('solveIvp', () => {
  const adaptive: OdeMethodName[] = [
    'RK23',
    'RK45',
    'DOP853',
    'Tsit5',
    'HeunEuler',
    'Fehlberg45',
    'CashKarp',
    'Radau',
    'BDF',
    'LSODA',
    'Rosenbrock23',
    'Adams',
  ]
  for (const method of adaptive) {
    it(`${method} integrates exponential decay to tolerance`, () => {
      const sol = solveIvp(decay, [0, 2], [1, 2], { method, rtol: 1e-8, atol: 1e-10, tEval: [0.5, 2] })
      expect(sol.success).toBe(true)
      const y = toFlat(sol.y)
      expect(Math.abs((y[2] ?? 0) - Math.exp(-2))).toBeLessThan(1e-5)
      expect(Math.abs((y[3] ?? 0) - 2 * Math.exp(-2))).toBeLessThan(1e-5)
    })
  }

  const fixed: OdeMethodName[] = [
    'Euler',
    'Heun',
    'Midpoint',
    'Ralston',
    'Kutta3',
    'RK4',
    'RK38',
    'BackwardEuler',
    'Trapezoid',
    'ImplicitMidpoint',
    'SDIRK2',
    'SDIRK3',
    'AdamsBashforth',
    'AdamsMoulton',
    'PECE',
  ]
  for (const method of fixed) {
    it(`${method} runs with a fixed step`, () => {
      const sol = solveIvp(decay, [0, 1], [1], { method, dt: 0.01 })
      expect(sol.success).toBe(true)
      expect(Math.abs((toFlat(sol.y).at(-1) ?? 0) - Math.exp(-1))).toBeLessThan(1e-2)
    })
  }

  it('locates terminal events with a direction', () => {
    const sol = solveIvp(oscillator, [0, 10], [1, 0], {
      rtol: 1e-10,
      atol: 1e-12,
      events: [event((_t, y) => toFlat(y)[0] ?? 0, { terminal: true, direction: -1 })],
    })
    expect(sol.status).toBe('event')
    expect(Math.abs((sol.tEvents[0]?.[0] ?? 0) - Math.PI / 2)).toBeLessThan(1e-8)
  })

  it('differentiates through the steps', () => {
    const g = grad((k: Value) =>
      sum(solveIvp((_t, y) => neg(mul(k, y)), [0, 1], [1], { rtol: 1e-10, atol: 1e-12, tEval: [1] }).y),
    )
    // d/dk exp(-k) at k = 0.5
    expect(Math.abs(item(g(tensor(0.5))) + Math.exp(-0.5))).toBeLessThan(1e-7)
    const { tangent } = jvp(
      (k: Value) =>
        sum(
          solveIvp((_t, y) => neg(mul(k, y)), [0, 1], [1], { method: 'Radau', rtol: 1e-10, atol: 1e-12, tEval: [1] }).y,
        ),
      tensor(0.5),
      tensor(1),
    )
    expect(Math.abs(item(tangent) + Math.exp(-0.5))).toBeLessThan(1e-7)
  })

  it('fails cleanly on blow-up', () => {
    const sol = solveIvp((_t, y) => mul(y, y), [0, 2], [1], { maxSteps: 2000 })
    expect(sol.success).toBe(false)
    expect(sol.message.length).toBeGreaterThan(0)
  })

  it('records rejected steps in the trace', () => {
    const rec = recorder()
    solveIvp((_t, y) => add(neg(mul(y, 50)), 1), [0, 1], [5], { rtol: 1e-6, tracer: rec })
    expect(rec.run().events.length).toBeGreaterThan(0)
  })
})

describe('odeint, odeSolver and diffeqsolve', () => {
  it('odeint returns states at the times', () => {
    const y = odeint(decay, [1], [0, 1, 2])
    expect(Math.abs((toFlat(y)[2] ?? 0) - Math.exp(-2))).toBeLessThan(1e-6)
  })
  it('odeSolver fits the nn contract', () => {
    expect(odeSolver()(decay, tensor([1, 1]), [0, 1]).length).toBe(2)
  })
  it('diffeqsolve with a constant step and saveat', () => {
    const s = diffeqsolve(decay, [1], { t0: 0, t1: 1, dt0: 0.01, solver: 'Tsit5', saveat: { ts: [0.5, 1] } })
    expect(s.ts).toEqual([0.5, 1])
    expect(Math.abs((toFlat(s.ys)[1] ?? 0) - Math.exp(-1))).toBeLessThan(1e-10)
  })
})

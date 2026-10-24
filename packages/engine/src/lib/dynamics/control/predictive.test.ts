import { describe, expect, it } from 'vitest'
import { add, mul, sin, sum } from '../../tensor/index.ts'
import { recorder } from '../../trace/index.ts'
import { riccatiRecursion } from './optimal.ts'
import { linearMpc, nonlinearMpc, recedingHorizon } from './predictive.ts'
import type { DiscreteModel } from './trajectory.ts'

const A = [
  [1, 1],
  [0, 1],
]
const B = [[0.5], [1]]
const Q = [
  [1, 0],
  [0, 1],
]
const plant = (x: readonly number[], u: readonly number[]): number[] => [
  (x[0] ?? 0) + (x[1] ?? 0) + 0.5 * (u[0] ?? 0),
  (x[1] ?? 0) + (u[0] ?? 0),
]

describe('model predictive control', () => {
  it('unconstrained MPC applies the finite-horizon LQR feedback', () => {
    const mpc = linearMpc({ A, B, Q, R: [[0.1]], horizon: 8 })
    const K0 = Array.from((riccatiRecursion(A, B, Q, [[0.1]], Q, 8).gains[0] as { data: ArrayLike<number> }).data)
    const x = [2, -1]
    const u = -((K0[0] ?? 0) * 2 + (K0[1] ?? 0) * -1)
    expect(Math.abs((mpc.solve(x).u[0] ?? 0) - u)).toBeLessThan(1e-8)
  })

  it('respects input and state constraints over the receding horizon', () => {
    const mpc = linearMpc({ A, B, Q, R: [[0.1]], horizon: 10, uMin: -0.5, uMax: 0.5, xMin: [-10, -1], xMax: [10, 1] })
    const rec = recorder()
    const run = recedingHorizon(mpc, plant, [5, 0], { steps: 40, tracer: rec })
    expect(run.inputs.every((u) => Math.abs(u[0] ?? 0) <= 0.5 + 1e-9)).toBe(true)
    expect(run.states.every((x) => Math.abs(x[1] ?? 0) <= 1 + 1e-6)).toBe(true)
    expect(Math.abs(run.states.at(-1)?.[0] ?? 1)).toBeLessThan(0.05)
    expect(rec.run().events.length).toBeGreaterThan(0)
    const ip = linearMpc({ A, B, Q, R: [[0.1]], horizon: 10, uMin: -0.5, uMax: 0.5, method: 'interior-point' })
    expect(Math.abs((ip.solve([5, 0]).u[0] ?? 0) - (mpc.solve([5, 0]).u[0] ?? 0))).toBeLessThan(1e-5)
    expect(() => linearMpc({ A, B, Q, R: [[0.1]], horizon: 0 })).toThrow()
  })

  it('nonlinear MPC stabilises a nonlinear system', () => {
    const model: DiscreteModel = {
      stateSize: 1,
      actionSize: 1,
      dynamics: (x, u) => add(add(x, mul(sin(x), 0.3)), u),
      cost: (x, u) => add(sum(mul(x, x), -1), mul(sum(mul(u, u), -1), 0.1)),
    }
    const run = recedingHorizon(
      nonlinearMpc(model, { horizon: 6 }),
      (x, u) => [(x[0] ?? 0) + 0.3 * Math.sin(x[0] ?? 0) + (u[0] ?? 0)],
      [2],
      {
        steps: 12,
      },
    )
    expect(Math.abs(run.states.at(-1)?.[0] ?? 1)).toBeLessThan(1e-3)
  })
})

import { describe, expect, it } from 'vitest'
import { add, concat, matmul, mul, sin, sliceAxis, sum, tensor, transpose, type Value } from '../../tensor/index.ts'
import { riccatiRecursion } from './optimal.ts'
import { type DiscreteModel, ddp, directCollocation, ilqr, multipleShooting, singleShooting } from './trajectory.ts'

const Ad = [
  [1, 0.1],
  [0, 1],
]
const Bd = [[0.005], [0.1]]
// A linear-quadratic model in batched form: x [B, 2], u [B, 1].
const lq: DiscreteModel = {
  stateSize: 2,
  actionSize: 1,
  dynamics: (x, u) => add(matmul(x, transpose(tensor(Ad))), matmul(u, transpose(tensor(Bd)))),
  cost: (x, u) => add(sum(mul(x, x), -1), mul(sum(mul(u, u), -1), 0.1)),
}

describe('trajectory optimisation', () => {
  it('iLQR on a linear-quadratic problem reproduces the finite-horizon LQR gains', () => {
    const T = 20
    const r = ilqr(lq, [1, 0], { horizon: T, squash: false, tolerance: 1e-12 })
    const rec = riccatiRecursion(
      Ad,
      Bd,
      [
        [1, 0],
        [0, 1],
      ],
      [[0.1]],
      [
        [0, 0],
        [0, 0],
      ],
      T,
    )
    // iLQR gains K_t (u = u_bar + K dx) are the negatives of the LQR gains (u = -K x).
    const k = rec.gains.map((g) => Array.from((g as { data: ArrayLike<number> }).data))
    r.gains.forEach((g, t) => {
      g[0]?.forEach((v, j) => {
        expect(Math.abs(v + (k[t]?.[j] ?? 0))).toBeLessThan(1e-6)
      })
    })
    // The optimal controls are the closed-loop LQR inputs.
    let x = [1, 0]
    for (let t = 0; t < T; t++) {
      const u = -((k[t]?.[0] ?? 0) * (x[0] ?? 0) + (k[t]?.[1] ?? 0) * (x[1] ?? 0))
      expect(Math.abs((r.controls[t]?.[0] ?? 0) - u)).toBeLessThan(1e-6)
      x = [(x[0] ?? 0) + 0.1 * (x[1] ?? 0) + 0.005 * u, (x[1] ?? 0) + 0.1 * u]
    }
  })

  const pendulum: DiscreteModel = {
    stateSize: 2,
    actionSize: 1,
    dynamics: (x, u) => {
      const th = sliceAxis(x, -1, 0, 1)
      const w = sliceAxis(x, -1, 1, 2)
      const w2 = add(w, mul(add(mul(sin(th), -10), u), 0.05))
      return concatLast([add(th, mul(w2, 0.05)), w2])
    },
    cost: (x, u) => add(sum(mul(x, x), -1), mul(sum(mul(u, u), -1), 0.01)),
    actionLow: [-3],
    actionHigh: [3],
  }

  it('DDP and iLQR agree on a nonlinear problem, and the transcriptions find similar costs', () => {
    const a = ilqr(pendulum, [1, 0], { horizon: 15 })
    const b = ddp(pendulum, [1, 0], { horizon: 15 })
    expect(Math.abs(a.cost - b.cost) / a.cost).toBeLessThan(1e-3)
    expect(a.controls.every((u) => Math.abs(u[0] ?? 0) <= 3 + 1e-9)).toBe(true)
    const s = singleShooting(pendulum, [1, 0], { horizon: 15 })
    const m = multipleShooting(pendulum, [1, 0], { horizon: 15 })
    expect(Math.abs(s.cost - a.cost) / a.cost).toBeLessThan(2e-2)
    expect(Math.abs(m.cost - a.cost) / a.cost).toBeLessThan(2e-2)
    expect(m.defect).toBeLessThan(1e-6)
  })

  it('direct collocation respects input bounds and the final state', () => {
    const problem = {
      stateSize: 2,
      actionSize: 1,
      dynamics: (x: Value, u: Value): Value => concatLast([sliceAxis(x, 0, 1, 2), u]),
      cost: (_x: Value, u: Value): Value => sum(mul(u, u)),
    }
    const r = directCollocation(problem, [0, 0], {
      duration: 2,
      nodes: 20,
      final: [1, 0],
      actionLow: [-2],
      actionHigh: [2],
    })
    expect(r.defect).toBeLessThan(1e-7)
    expect(r.controls.every((u) => Math.abs(u[0] ?? 0) <= 2 + 1e-8)).toBe(true)
    expect(() => directCollocation(problem, [0, 0], { duration: 1, nodes: 1 })).toThrow()
  })
})

function concatLast(xs: Value[]): Value {
  return concat(xs, -1)
}

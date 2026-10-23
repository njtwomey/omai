import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { foldIn, type Key, key, normal } from '../../random/index.ts'
import { add, item, mul, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { Linear } from '../module/layers.ts'
import type { Params } from '../module/module.ts'
import { AugmentedODEBlock, CDEBlock, ODEBlock, SDEBlock, SecondOrderODEBlock } from './continuous.ts'

/** Classical RK4 with a few sub-steps per interval (a test solver; real solvers live in dynamics/ode). */
const rk4 = (f: (t: number, y: Value) => Value, y0: Value, times: readonly number[], sub = 8): Value[] => {
  const ys: Value[] = [y0]
  for (let i = 1; i < times.length; i++) {
    let y = ys[i - 1] as Value
    const t0 = times[i - 1] ?? 0
    const h = ((times[i] ?? 0) - t0) / sub
    for (let s = 0; s < sub; s++) {
      const t = t0 + s * h
      const k1 = f(t, y)
      const k2 = f(t + h / 2, add(y, mul(k1, h / 2)))
      const k3 = f(t + h / 2, add(y, mul(k2, h / 2)))
      const k4 = f(t + h, add(y, mul(k3, h)))
      y = add(y, mul(add(add(k1, mul(k2, 2)), add(mul(k3, 2), k4)), h / 6))
    }
    ys.push(y)
  }
  return ys
}

const eulerMaruyama = (
  f: (t: number, y: Value) => Value,
  g: (t: number, y: Value) => Value,
  y0: Value,
  times: readonly number[],
  k: Key,
): Value[] => {
  const ys: Value[] = [y0]
  for (let i = 1; i < times.length; i++) {
    const y = ys[i - 1] as Value
    const dt = (times[i] ?? 0) - (times[i - 1] ?? 0)
    const dw = mul(normal(foldIn(k, i), y.shape), Math.sqrt(dt))
    ys.push(add(add(y, mul(f(times[i - 1] ?? 0, y), dt)), mul(g(times[i - 1] ?? 0, y), dw)))
  }
  return ys
}

describe('continuous-depth blocks', () => {
  it('a linear field integrates to the matrix exponential', () => {
    const field = new Linear(1, 1, { bias: false, key: key(0) })
    field.setParams({ weight: tensor([[-0.5]]) })
    const block = new ODEBlock(field, { solver: (f, y, t) => rk4(f, y, t), times: [0, 2] })
    expect(item(block.forward([[1]]))).toBeCloseTo(Math.exp(-1), 5)
    const traj = new ODEBlock(field, { solver: (f, y, t) => rk4(f, y, t), times: [0, 1, 2], trajectory: true })
    expect(traj.forward([[1]]).shape).toEqual([3, 1, 1])
  })

  it('adjoint gradients match backpropagation through the solver', () => {
    const field = new Linear(2, 2, { key: key(1) })
    const solver = (f: (t: number, y: Value) => Value, y: Value, t: readonly number[]): Value[] => rk4(f, y, t, 16)
    const back = new ODEBlock(field, { solver, times: [0, 0.5, 1] })
    const adj = new ODEBlock(field, { solver, times: [0, 0.5, 1], gradient: 'adjoint', timeDependent: false })
    const x = normal(key(2), [3, 2])
    const loss =
      (m: ODEBlock) =>
      (p: Params): Value =>
        sum(mul(m.apply(p, x), m.apply(p, x)))
    const g1 = grad(loss(back))(back.params) as Params
    const g2 = grad(loss(adj))(adj.params) as Params
    const w1 = toFlat((g1['field'] as Params)['weight'] as Value)
    const w2 = toFlat((g2['field'] as Params)['weight'] as Value)
    for (const [i, v] of w1.entries()) expect(w2[i]).toBeCloseTo(v, 6)
    const gx1 = toFlat(grad((v: Value) => sum(back.apply(back.params, v)))(x))
    const gx2 = toFlat(grad((v: Value) => sum(adj.apply(adj.params, v)))(x))
    for (const [i, v] of gx1.entries()) expect(gx2[i]).toBeCloseTo(v, 6)
  })

  it('augmented, second-order, time-dependent and validated times', () => {
    const solver = (f: (t: number, y: Value) => Value, y: Value, t: readonly number[]): Value[] => rk4(f, y, t, 4)
    const aug = new AugmentedODEBlock(new Linear(3, 3, { key: key(3) }), 1, { solver })
    expect(aug.forward([[1, 2]]).shape).toEqual([1, 3])
    expect(
      new AugmentedODEBlock(new Linear(3, 3, { key: key(3) }), 1, { solver, strip: true }).forward([[1, 2]]).shape,
    ).toEqual([1, 2])
    // x'' = -x from x = 1, x' = 0 gives cos t.
    const spring = new Linear(2, 1, { bias: false, key: key(4) })
    spring.setParams({ weight: tensor([[-1, 0]]) })
    const osc = new SecondOrderODEBlock(spring, { solver: (f, y, t) => rk4(f, y, t, 32), times: [0, 1] })
    expect(item(osc.forward([[1]]))).toBeCloseTo(Math.cos(1), 7)
    const td = new ODEBlock(new Linear(3, 2, { key: key(5) }), { solver, timeDependent: true })
    expect(td.forward([[1, 2]]).shape).toEqual([1, 2])
    expect(() => new ODEBlock(new Linear(2, 2), { solver, times: [1, 0] })).toThrow()
  })

  it('a CDE follows its control path and an SDE needs a key', () => {
    const solver = (f: (t: number, y: Value) => Value, y: Value, t: readonly number[]): Value[] => rk4(f, y, t, 4)
    const cde = new CDEBlock(new Linear(2, 3, { key: key(6) }), new Linear(3, 6, { key: key(7) }), {
      solver,
      times: [0, 0.5, 2],
      trajectory: true,
    })
    const path = normal(key(8), [2, 3, 2])
    expect(cde.forward(path).shape).toEqual([2, 3, 3])
    expect(cde.outputShape([2, 3, 2])).toEqual([2, 3, 3])
    const sde = new SDEBlock(new Linear(2, 2, { key: key(9) }), new Linear(2, 2, { key: key(10) }), {
      solver: eulerMaruyama,
      times: [0, 0.1, 0.2],
    })
    expect(() => sde.forward([[1, 2]])).toThrow()
    const a = toFlat(sde.forward([[1, 2]], { key: key(11) }))
    expect(toFlat(sde.forward([[1, 2]], { key: key(11) }))).toEqual(a)
  })
})

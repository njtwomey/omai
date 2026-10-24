import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { reshape, sliceAxis, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { policies, recorder } from '../trace/index.ts'
import * as S from './index.ts'

const a2: Tensor = tensor([
  [-1, 0.5],
  [-0.3, -2],
])
const b2: Tensor = tensor([[1], [0.4]])
const c2: Tensor = tensor([[1, -1]])
const d1: Tensor = tensor([[0.2]])

describe('time responses', () => {
  it('step responses settle at the dc gain', () => {
    const g = S.tf([2, 3], [1, 2, 3])
    const r = S.step(g, { t: S.defaultResponseTimes(S.toSs(g).a, 400) })
    const y = toFlat(r.y as Tensor)
    expect(y[y.length - 1]).toBeCloseTo(1, 2)
    const gd = S.cont2discrete(g, 0.05)
    const yd = toFlat(S.step(gd, { n: 400 }).y as Tensor)
    expect(yd[yd.length - 1]).toBeCloseTo(1, 4)
  })

  it('agree between the zero-order-hold discretisation and a continuous held-input simulation', () => {
    const s = S.ss(a2, b2, c2, d1)
    const t = Array.from({ length: 20 }, (_, k) => 0.1 * k)
    const u = t.map((v) => Math.sin(3 * v))
    const yc = toFlat(S.lsim(s, u, t, { interp: false }).y as Tensor)
    const yd = toFlat(S.lsim(S.cont2discrete(s, 0.1), u).y as Tensor)
    for (const [i, v] of yc.entries()) expect(v).toBeCloseTo(yd[i] ?? Number.NaN, 12)
  })

  it('are traced, one tensor event per sample', () => {
    const rec = recorder()
    const r = S.lsim(S.tf([1], [1, 1]), [1, 1, 1, 1], [0, 1, 2, 3], { tracer: rec, policy: policies.debug })
    expect(r.y.shape).toEqual([4])
    expect(rec.events.filter((e) => e.type === 'tensor').length).toBe(8)
    const y = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'y0' ? [e.value] : []))
    expect(y).toEqual(toFlat(r.y as Tensor))
  })

  it('reject unequal steps and missing times', () => {
    expect(() => S.lsim(S.tf([1], [1, 1]), [1, 1, 1], [0, 1, 3])).toThrow()
    expect(() => S.lsim(S.tf([1], [1, 1]), [1, 1, 1])).toThrow()
    expect(() => S.lsim(S.tf([1], [1, 1]), [[1, 2]], [0])).toThrow()
    expect(() => S.step(S.tf([1], [1, 1]), { input: 2 })).toThrow()
  })

  it('simulate free responses with no input', () => {
    const y = toFlat(S.initial(S.ss([[-1]], [[1]], [[1]]), [2], { t: [0, 1] }).y as Tensor)
    expect(y[1]).toBeCloseTo(2 * Math.exp(-1), 12)
    expect(toFlat(S.lsim(S.ss([[-1]], [[1]], [[1]]), 0, [0, 1], { x0: [1] }).y as Tensor)[1]).toBeCloseTo(
      Math.exp(-1),
      12,
    )
  })
})

describe('time responses are differentiable', () => {
  const t = Array.from({ length: 8 }, (_, k) => 0.25 * k)
  it('in the matrices, the input and the initial state', () => {
    const f = (a: Value, b: Value, u: Value, x0: Value): Value => S.lsim(S.ss(a, b, c2, d1), u, t, { x0 }).y
    const u = tensor(t.map((v) => Math.cos(v)))
    expect(checkPrimitive(f, [a2, b2, u, tensor([0.3, -0.1])], { vmap: false }).failures).toEqual([])
  })

  it('in the coefficients of a transfer function (step and impulse), continuous and discrete', () => {
    const stepOf = (num: Value, den: Value): Value => S.step(S.tf(num, den), { t }).y
    expect(checkPrimitive(stepOf, [tensor([1, 2]), tensor([1, 1.5, 2])], { vmap: false }).failures).toEqual([])
    const impulseOf = (den: Value): Value => S.impulse(S.tf([1, 0.5], den, { dt: 0.5 }), { n: 6 }).y
    expect(checkPrimitive(impulseOf, [tensor([1, -0.4, 0.1])], { vmap: false }).failures).toEqual([])
    const lastState = (a: Value): Value =>
      reshape(sliceAxis(S.initial(S.ss(a, b2, c2), [1, 1], { t }).x, 0, 7, 8), [-1])
    expect(checkPrimitive(lastState, [a2], { vmap: false }).failures).toEqual([])
  })
})

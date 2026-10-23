import { describe, expect, it } from 'vitest'
import { valueAndGrad } from '../autodiff/index.ts'
import { add, mul, square, sub, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as O from './index.ts'

type P = { readonly w: Value; readonly b: Value; readonly e: Value }

const params: P = { w: tensor([1, -2]), b: tensor([0.5]), e: tensor([[3, 4]]) }
const loss = (p: unknown): Value => {
  const q = p as P
  return add(add(sum(square(sub(q.w, 3))), sum(mul(q.b, q.b))), mul(sum(square(q.e)), 0.1))
}

function run(opt: O.GradientTransformation, steps: number, start: unknown = params): P {
  const vg = valueAndGrad(loss)
  let p = start
  let s = opt.init(p)
  for (let t = 0; t < steps; t++) {
    const r = opt.update(vg(p).grad, s, p)
    p = O.applyUpdates(p, r.updates)
    s = r.state
  }
  return p as P
}

describe('masked', () => {
  it('applies the inner transformation to the selected leaves only', () => {
    const opt = O.chain(O.masked(O.scale(0), { w: false, b: true, e: true }), O.scaleByLearningRate(0.1))
    const p = run(opt, 3)
    expect(toFlat(p.b)).toEqual([0.5])
    expect(toFlat(p.e)).toEqual([3, 4])
    expect(toFlat(p.w)[0]).not.toBe(1)
  })

  it('runs global operations over the selection and threads state', () => {
    const g = { w: tensor([3, 4]), b: tensor([12]), e: tensor([[0, 0]]) }
    const t = O.masked(O.clipByGlobalNorm(1), (tree: unknown) => ({
      w: true,
      b: false,
      e: Object.keys(tree as object).length > 0,
    }))
    const u = t.update(g, t.init(g), g).updates as typeof g
    expect(toFlat(u.w).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([0.6, 0.8])
    expect(toFlat(u.b)).toEqual([12])
    // Adam inside a mask equals Adam on the selected leaf alone.
    const inside = run(O.masked(O.adam({ learningRate: 0.1 }), { w: true, b: false, e: false }), 5)
    const alone = run(
      O.multiTransform({ a: O.adam({ learningRate: 0.1 }), z: O.identity() }, { w: 'a', b: 'z', e: 'z' }),
      5,
    )
    expect(toFlat(inside.w)).toEqual(toFlat(alone.w))
  })
})

describe('multiTransform', () => {
  it('updates each group with its own transformation', () => {
    const opt = O.multiTransform(
      { fast: O.sgd({ learningRate: 0.2 }), slow: O.adam({ learningRate: 0.01 }), frozen: O.setToZero() },
      { w: 'fast', b: 'slow', e: 'frozen' },
    )
    const p = run(opt, 4)
    const sgdOnly = run(O.sgd({ learningRate: 0.2 }), 4)
    const adamOnly = run(O.adam({ learningRate: 0.01 }), 4)
    expect(toFlat(p.w)).toEqual(toFlat(sgdOnly.w))
    expect(toFlat(p.b)).toEqual(toFlat(adamOnly.b))
    expect(toFlat(p.e)).toEqual([3, 4])
  })

  it('rejects labels without a transformation and mismatched label trees', () => {
    expect(() => O.multiTransform({ a: O.identity() }, { w: 'a', b: 'x', e: 'a' }).init(params)).toThrow(/label 'x'/)
    expect(() => O.multiTransform({ a: O.identity() }, { w: 'a' }).init(params)).toThrow(/leaves/)
  })
})

describe('small transformations', () => {
  it('scale, identity, trust ratio and dampened momentum', () => {
    expect(toFlat(O.scale(3).update(tensor([1]), undefined).updates as Value)).toEqual([3])
    const tr = O.scaleByTrustRatio()
    expect(toFlat(tr.update(tensor([0, 0]), undefined, tensor([1, 1])).updates as Value)).toEqual([0, 0])
    expect(() => tr.update(tensor([1]), undefined)).toThrow()
    const m = O.traceMomentum(0.5, false, 0.5)
    const a = m.update(tensor([2]), m.init(tensor([0])))
    // second step: 0.5 * 2 + (1 - 0.5) * 2
    expect(toFlat(m.update(tensor([2]), a.state).updates as Value)).toEqual([2])
  })

  it('centred RMSprop divides by the standard deviation of the gradient', () => {
    const t = O.scaleByRms(0.5, 0, true)
    const s1 = t.update(tensor([2]), t.init(tensor([0])))
    // v = 2, m = 1, sqrt(v - m^2) = 1
    expect(toFlat(s1.updates as Value)).toEqual([2])
  })
})

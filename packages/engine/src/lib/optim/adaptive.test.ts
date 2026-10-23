import { describe, expect, it } from 'vitest'
import { valueAndGrad } from '../autodiff/index.ts'
import { maxOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { matmul, mul, reshape, sub, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as O from './index.ts'

const A = tensor([
  [3, 1, 0],
  [1, 2, 0.5],
  [0, 0.5, 1],
])
const quadratic = (x: unknown): Value => {
  const v = reshape(x as Value, [3, 1])
  return mul(sum(mul(v, matmul(A, v))), 0.5)
}

describe('Sophia', () => {
  it('Hutchinson probes average to the Hessian diagonal', () => {
    const est = O.hutchinsonDiagonal(quadratic, key(3))
    const n = 400
    const total = [0, 0, 0]
    for (let t = 0; t < n; t++) {
      for (const [i, v] of toFlat(est(undefined, tensor([1, 1, 1]), t) as Value).entries())
        total[i] = (total[i] ?? 0) + v
    }
    const avg = total.map((v) => v / n)
    for (const [i, d] of [3, 2, 1].entries()) expect(Math.abs((avg[i] ?? 0) - d)).toBeLessThan(0.15)
  })

  it('takes clipped steps no longer than the learning rate and converges', () => {
    const lr = 0.05
    const opt = O.sophiaLite({
      learningRate: lr,
      weightDecay: 0,
      hessianEvery: 1,
      rho: 1,
      hessian: O.hutchinsonDiagonal(quadratic, key(1)),
    })
    const vg = valueAndGrad(quadratic)
    let x: Value = tensor([2, -1, 3])
    let s = opt.init(x)
    for (let t = 0; t < 300; t++) {
      const r = opt.update(vg(x).grad, s, x)
      for (const u of toFlat(r.updates as Value)) expect(Math.abs(u)).toBeLessThanOrEqual(lr + 1e-15)
      x = O.applyUpdates(x, r.updates) as Value
      s = r.state
    }
    expect(maxOf(toFlat(x).map(Math.abs))).toBeLessThan(0.1)
  })
})

describe('Adafactor', () => {
  it('factors matrix statistics into rows and columns', () => {
    const opt = O.adafactor()
    const p = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    const s = opt.update(sub(p, 1), opt.init(p), p).state
    const [row, col] = (s.stats as Value[][])[0] as [Value, Value]
    expect(row.shape).toEqual([2, 1])
    expect(col.shape).toEqual([1, 3])
  })
})

describe('defaults', () => {
  it('RAdam takes plain momentum steps until the SMA length exceeds 5', () => {
    const t = O.scaleByRadam()
    let s = t.init(tensor([0]))
    const seen: number[] = []
    for (let i = 0; i < 7; i++) {
      const r = t.update(tensor([1]), s)
      seen.push(toFlat(r.updates as Value)[0] ?? 0)
      s = r.state
    }
    // Bias-corrected momentum of a constant gradient is exactly 1 for the first steps.
    expect(seen.slice(0, 3).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([1, 1, 1])
    expect(seen[6]).toBeLessThan(0.5)
  })
})

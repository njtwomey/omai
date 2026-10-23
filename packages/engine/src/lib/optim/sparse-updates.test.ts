import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { key, normal } from '../random/index.ts'
import { matmul, mul, reshape, square, sub, sum, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as O from './index.ts'

// A sparse regression: y = X w* with w* = [2, 0, 0, -1, 0], 40 samples.
const X = normal(key(7), [40, 5])
const wStar = tensor([2, 0, 0, -1, 0])
const y = matmul(X, reshape(wStar, [5, 1]))
const mse = (w: unknown): Value => mul(sum(square(sub(matmul(X, reshape(w as Value, [5, 1])), y))), 1 / 40)
const g = grad(mse)

function fit(opt: O.GradientTransformation, steps: number, w0: Value = tensor([0, 0, 0, 0, 0])): number[] {
  let w: Value = w0
  let s = opt.init(w)
  for (let t = 0; t < steps; t++) {
    const r = opt.update(g(w), s, w)
    w = O.applyUpdates(w, r.updates) as Value
    s = r.state
  }
  return toFlat(w)
}

describe('active sets', () => {
  it('are element-wise or row-wise', () => {
    expect(
      toArray(
        O.activeMask(
          tensor([
            [0, 1, 0],
            [0, 0, 0],
            [3, 0, 0],
          ]),
          true,
        ),
      ),
    ).toEqual([
      [1, 1, 1],
      [0, 0, 0],
      [1, 1, 1],
    ])
  })
})

describe('L1 by cumulative penalty', () => {
  it('matches the paper on two hand-computed steps', () => {
    const opt = O.sgdCumulativeL1({ learningRate: 0.5, l1: 0.2, lazy: false })
    let w: Value = tensor([0.3, -0.05])
    let s = opt.init(w)
    // step 1: w' = w - 0.5 g = [0.25, -0.05]; u = 0.1; w'' = [0.15, 0]; q = [-0.1, 0.05]
    let r = opt.update(tensor([0.1, 0]), s, w)
    w = O.applyUpdates(w, r.updates) as Value
    s = r.state
    expect(toFlat(w).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([0.15, 0])
    // step 2: w' = [0.15, 0]; u = 0.2; w'' = max(0, 0.15 - (0.2 - 0.1)) = 0.05
    r = opt.update(tensor([0, 0]), s, w)
    w = O.applyUpdates(w, r.updates) as Value
    expect(toFlat(w).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([0.05, 0])
    expect(r.state.u).toBeCloseTo(0.2, 15)
  })

  it('zeroes irrelevant features exactly and keeps the relevant ones', () => {
    const w = fit(O.sgdCumulativeL1({ learningRate: 0.1, l1: 0.02 }), 400)
    expect([w[1], w[2], w[4]]).toEqual([0, 0, 0])
    expect(w[0]).toBeGreaterThan(1.8)
    expect(w[3]).toBeLessThan(-0.8)
  })
})

describe('truncated gradient', () => {
  it('is soft-thresholded SGD with K = 1 and theta = infinity', () => {
    const w0 = tensor([0.5, -0.02, 1])
    const gr = tensor([0.2, 0.1, -0.4])
    const opt = O.truncatedGradient({ learningRate: 0.5, gravity: 0.1 })
    const u = toFlat(opt.update(gr, opt.init(w0), w0).updates as Value)
    const expected = [-0.15, 0, 0.15]
    for (const [i, v] of u.entries()) expect(v).toBeCloseTo(expected[i] ?? 0, 14)
  })

  it('truncates every K steps and leaves weights beyond theta alone', () => {
    const opt = O.truncatedGradient({ learningRate: 1, gravity: 0.1, every: 2, theta: 0.5 })
    let w: Value = tensor([0.3, 0.9])
    let s = opt.init(w)
    const seen: number[][] = []
    for (let t = 0; t < 2; t++) {
      const r = opt.update(tensor([0, 0]), s, w)
      w = O.applyUpdates(w, r.updates) as Value
      s = r.state
      seen.push(toFlat(w).map((v) => Math.round(v * 1e12) / 1e12))
    }
    expect(seen).toEqual([
      [0.3, 0.9],
      [0.1, 0.9],
    ])
  })

  it('produces a sparse model', () => {
    const w = fit(O.truncatedGradient({ learningRate: 0.1, gravity: 0.05 }), 400)
    expect([w[1], w[2], w[4]]).toEqual([0, 0, 0])
  })
})

describe('FTRL-Proximal', () => {
  it('matches the closed form without regularisation', () => {
    // alpha = 1, beta = 0: w_{t+1} = -z / sqrt(n); constant gradient 2 from w = 0.
    const opt = O.ftrlProximal({ alpha: 1, beta: 0 })
    let w: Value = tensor([0])
    let s = opt.init(w)
    let r = opt.update(tensor([2]), s, w)
    w = O.applyUpdates(w, r.updates) as Value
    s = r.state
    expect(toFlat(w)).toEqual([-1])
    // sigma = (sqrt(8) - 2), z = 2 + 2 - sigma * (-1), n = 8.
    r = opt.update(tensor([2]), s, w)
    w = O.applyUpdates(w, r.updates) as Value
    const sigma = Math.sqrt(8) - 2
    expect(toFlat(w)[0]).toBeCloseTo(-(4 + sigma) / Math.sqrt(8), 14)
  })

  it('converges on a sparse regression with weights exactly zero where |z| <= l1', () => {
    const opt = O.ftrlProximal({ alpha: 0.5, beta: 1, l1: 3, l2: 0.01 })
    let w: Value = tensor([0, 0, 0, 0, 0])
    let s = opt.init(w)
    let zeros = 0
    for (let t = 0; t < 300; t++) {
      const r = opt.update(g(w), s, w)
      w = O.applyUpdates(w, r.updates) as Value
      s = r.state
      const support = toFlat(O.ftrlSupport(s, 3) as Value)
      for (const [i, v] of toFlat(w).entries()) {
        if (support[i] === 0) {
          expect(v).toBe(0)
          zeros += 1
        }
      }
    }
    expect(zeros).toBeGreaterThan(0)
    const final = toFlat(w)
    expect(final[0]).toBeCloseTo(2, 6)
    expect(final[3]).toBeCloseTo(-1, 6)
    for (const i of [1, 2, 4]) expect(Math.abs(final[i] ?? 1)).toBeLessThan(1e-9)
  })

  it('leaves inactive coordinates where they are', () => {
    const opt = O.ftrlProximal()
    const w = tensor([0.7, -0.3])
    const r = opt.update(tensor([1, 0]), opt.init(w), w)
    expect(toFlat(r.updates as Value)[1]).toBe(0)
    expect(toFlat(O.ftrlSupport(r.state, 0) as Value)).toEqual([1, 0])
  })
})

describe('lazy Adagrad', () => {
  it('applies weight decay only to active entries', () => {
    const opt = O.lazyAdagrad({ learningRate: 0.1, weightDecay: 0.5 })
    const w = tensor([1, 1])
    const u = toFlat(opt.update(tensor([0.5, 0]), opt.init(w), w).updates as Value)
    expect(u[1]).toBe(0)
    expect(u[0]).toBeCloseTo(-0.1, 9)
  })
})

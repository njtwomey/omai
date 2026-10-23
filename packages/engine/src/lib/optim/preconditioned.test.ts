import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { lstsq, solve } from '../numerics/linalg/index.ts'
import { key, normal } from '../random/index.ts'
import {
  add,
  concat,
  eye,
  kron,
  matmul,
  mul,
  ones,
  reshape,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import * as O from './index.ts'

const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] ?? Number.NaN))).toBeLessThan(tol)
}

describe('K-FAC', () => {
  it('equals the Kronecker-factored solve without damping', () => {
    const a = O.kfacFactors(normal(key(1), [20, 3]), normal(key(2), [20, 2]))
    const dW = normal(key(3), [2, 3])
    const pre = O.kfacPrecondition(dW, a, 0)
    // vec(G^-1 dW A^-1) (column-major) = (A kron G)^-1 vec(dW); vec in row-major order is (G kron A)^-1.
    const direct = solve(kron(a.g, a.a), reshape(dW, [-1]))
    close(toFlat(pre), toFlat(direct), 1e-10)
  })

  it('splits damping between the factors by pi', () => {
    const f = {
      a: tensor([
        [4, 0],
        [0, 4],
      ]),
      g: tensor([[1]]),
    }
    // pi = sqrt(4 / 1) = 2, so A + 2 sqrt(l) I and G + sqrt(l) / 2 I with l = 1.
    const pre = O.kfacPrecondition([[6, 6]], f, 1)
    close(toFlat(pre), [6 / 6 / 1.5, 6 / 6 / 1.5], 1e-14)
  })

  it('preconditions a layer with its bias jointly and passes other leaves through', () => {
    const x = normal(key(4), [30, 2])
    const dy = normal(key(5), [30, 3])
    const opt = O.kfac({ learningRate: 1, damping: 1e-6, decay: 0 })
    const params = { layer: { w: normal(key(6), [3, 2]), b: tensor([0, 0, 0]) }, scale: tensor([1]) }
    const s = opt.observe(opt.init(params), [{ weight: 'layer.w', bias: 'layer.b', activations: x, outputGrads: dy }])
    const g = { layer: { w: normal(key(7), [3, 2]), b: tensor([1, -1, 2]) }, scale: tensor([5]) }
    const u = opt.update(g, s).updates as typeof g
    const f = O.kfacFactors(x, dy, true)
    const joint = O.kfacPrecondition(concat([g.layer.w, reshape(g.layer.b, [3, 1])], 1), f, 1e-6)
    close(
      toFlat(u.layer.w),
      toFlat(mul(joint, -1)).filter((_, i) => i % 3 !== 2),
      1e-12,
    )
    close(
      toFlat(u.layer.b),
      toFlat(mul(joint, -1)).filter((_, i) => i % 3 === 2),
      1e-12,
    )
    expect(toFlat(u.scale)).toEqual([-5])
  })

  it('rejects mismatched factors', () => {
    const opt = O.kfac({ learningRate: 1 })
    const s = opt.observe(opt.init({ w: tensor([[0, 0]]) }), [
      { weight: 'w', activations: [[1, 2, 3]], outputGrads: [[1]] },
    ])
    expect(() => opt.update({ w: tensor([[1, 1]]) }, s)).toThrow(/do not match/)
  })
})

describe('natural gradient', () => {
  it('reaches the least-squares solution in one step with the Gauss-Newton Fisher', () => {
    const X = normal(key(8), [25, 3]) as Tensor
    const y = normal(key(9), [25])
    const loss = (w: unknown): Value =>
      mul(sum(square(sub(matmul(X, reshape(w as Value, [3, 1])), reshape(y, [25, 1])))), 0.5 / 25)
    const opt = O.naturalGradient({ learningRate: 1, damping: 0 })
    const w0 = tensor([0, 0, 0])
    const s = opt.observe(opt.init(w0), { fisher: mul(matmul(transpose(X), X), 1 / 25) })
    const w1 = O.applyUpdates(w0, opt.update(grad(loss)(w0), s).updates) as Value
    close(toFlat(w1), toFlat(lstsq(X, y)), 1e-10)
  })

  it('averages empirical Fishers and falls back to gradient descent before any observation', () => {
    const opt = O.naturalGradient({ learningRate: 0.5, decay: 0.5, damping: 0 })
    expect(toFlat(opt.update(tensor([2, 4]), opt.init(tensor([0, 0]))).updates as Value)).toEqual([-1, -2])
    let s = opt.observe(opt.init(tensor([0, 0])), {
      perExampleGrads: [
        [2, 0],
        [0, 2],
      ],
    })
    s = opt.observe(s, {
      fisher: [
        [4, 0],
        [0, 4],
      ],
    })
    expect(toArray(s.fisher as Value)).toEqual([
      [3, 0],
      [0, 3],
    ])
    expect(() => opt.observe(s, {})).toThrow()
  })
})

describe('Shampoo', () => {
  it('leaves long modes unpreconditioned and reuses roots between refreshes', () => {
    const opt = O.shampoo({ learningRate: 1, eps: 1, maxDim: 2 })
    const g = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    const s0 = opt.init(g)
    const r = opt.update(g, s0)
    // only the first mode (length 2) is preconditioned, by (I + G G^T)^(-1/4)
    const root = O.matrixInversePthRoot(add(eye(2), matmul(g, transpose(g))), 4)
    close(toFlat(r.updates as Value), toFlat(mul(matmul(root, g), -1)), 1e-12)
    const lazy = O.shampoo({ learningRate: 1, preconditionEvery: 3 })
    const a = lazy.update(g, lazy.init(g))
    const b = lazy.update(g, a.state)
    expect(b.state.roots[0]?.[0]).toBe(a.state.roots[0]?.[0])
  })
})

describe('preconditioning helpers', () => {
  it('apply fixed matrices, diagonals and functions', () => {
    expect(
      toFlat(
        O.preconditionBy([
          [0, 1],
          [1, 0],
        ]).update(tensor([1, 2]), undefined).updates as Value,
      ),
    ).toEqual([2, 1])
    expect(
      toFlat(O.preconditionBy((g) => g.map((v) => v * 3)).update(tensor([1, 2]), undefined).updates as Value),
    ).toEqual([3, 6])
    expect(toFlat(O.diagonalPrecondition(ones([2]), tensor([4, 1]), 0) as Value)).toEqual([0.5, 1])
    expect(toArray(O.empiricalFisher([[1, 1]]))).toEqual([
      [1, 1],
      [1, 1],
    ])
    close(toFlat(O.naturalGradientDirection(eye(2), [1, 1], 1)), [0.5, 0.5], 1e-15)
  })
})

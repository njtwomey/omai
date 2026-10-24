import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad, numericalGradient } from '../autodiff/index.ts'
import { isWhiteboxError } from '../base/index.ts'
import { matmul, mul, sum, swapaxes, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import {
  ard,
  asRows,
  constantKernel,
  crossGram,
  defineKernel,
  fixHyperparameters,
  formatKernel,
  gram,
  gramGradient,
  hyperparameters,
  type Kernel,
  type KernelParams,
  linearKernel,
  maternKernel,
  onDimensions,
  periodicKernel,
  powerKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  scaleKernel,
  sumKernel,
  theta,
  whiteNoiseKernel,
  withTheta,
} from './index.ts'

const X = tensor([
  [0.1, -0.4],
  [0.7, 0.2],
  [-0.5, 0.9],
  [1.2, -1.1],
])
const Y = tensor([
  [0.3, 0.3],
  [-1, 0.5],
])

function throwsCode(f: () => unknown, code: string): void {
  try {
    f()
  } catch (e) {
    expect(isWhiteboxError(e, code as never)).toBe(true)
    return
  }
  throw new Error('expected a throw')
}

describe('defineKernel and point shapes', () => {
  const k = rbfKernel({ lengthScale: 0.8 })
  it('evaluates rows, single points and scalars', () => {
    expect(k.k(X).shape).toEqual([4, 4])
    expect(k.k(X, Y).shape).toEqual([4, 2])
    expect(k.k([0.1, -0.4], Y).shape).toEqual([2])
    expect(k.k(X, [0.3, 0.3]).shape).toEqual([4])
    expect(k.k([0.1, -0.4], [0.1, -0.4]).shape).toEqual([])
    expect(k.k(0.5, 0.5).shape).toEqual([])
    expect(k.diag(X).shape).toEqual([4])
    expect(k.diag([1, 2]).shape).toEqual([])
    expect(asRows([1, 2, 3], 'test').rows.shape).toEqual([1, 3])
  })
  it('rejects bad shapes and negative hyperparameters', () => {
    throwsCode(() => k.k(X, [[1, 2, 3]]), 'shape')
    throwsCode(() => k.k([[[1]]]), 'shape')
    throwsCode(() => rbfKernel({ lengthScale: -1 }), 'domain')
    throwsCode(() => k.withParams({ lengthScale: tensor(1), extra: tensor(2) }), 'value')
  })
  it('defaults the diagonal to the matrix of each row with itself', () => {
    const custom = defineKernel({
      name: 'quadratic',
      params: { c: tensor(2) },
      matrix: (p, x, y) => mul(p.c, matmul(x, swapaxes(y ?? x, 0, 1))),
    })
    const want = [0.34, 1.06, 2.12, 5.3]
    toFlat(custom.diag(X)).forEach((v, i) => {
      expect(v).toBeCloseTo(want[i] ?? 0, 14)
    })
    expect(custom.diag(tensor([[0, 0]])).shape).toEqual([1])
  })
  it('holds a positive hyperparameter that is exactly zero fixed', () => {
    const k0 = linearKernel({ sigma0: 0 })
    expect(k0.constraints).toEqual({ sigma0: 'fixed' })
    expect(theta(k0).shape).toEqual([0])
  })
})

describe('hyperparameters, theta and their gradients', () => {
  const k = sumKernel(scaleKernel(rbfKernel({ lengthScale: [0.5, 2] }), 1.5), whiteNoiseKernel({ noiseLevel: 0.2 }))
  it('lists hyperparameters in theta order with log coordinates', () => {
    const hs = hyperparameters(k)
    expect(hs.map((h) => h.name)).toEqual(['terms[0].base.lengthScale', 'terms[0].outputScale', 'terms[1].noiseLevel'])
    expect(hs.map((h) => h.offset)).toEqual([0, 2, 3])
    expect(toArray(theta(k))).toEqual([Math.log(0.5), Math.log(2), Math.log(1.5), Math.log(0.2)])
  })
  it('round-trips theta through withTheta', () => {
    const t = [0.3, -0.2, 0.1, -1.5]
    const back = toFlat(theta(withTheta(k, t)))
    back.forEach((v, i) => {
      expect(v).toBeCloseTo(t[i] ?? 0, 14)
    })
    throwsCode(() => withTheta(k, [1, 2]), 'shape')
    throwsCode(() => withTheta(k, [1, 2, 3, 4, 5]), 'shape')
  })
  it('differentiates the Gram matrix in theta (the conformance check, including vmap)', () => {
    const t = theta(k)
    expectConformant((v: Value) => gram(withTheta(k, v), X), [t])
    const r = rationalQuadraticKernel({ alpha: 0.7, lengthScale: 1.3 })
    expectConformant((v: Value) => crossGram(withTheta(r, v), X, Y), [theta(r)])
  })
  it('differentiates through the hyperparameter pytree with grad', () => {
    const base = maternKernel({ nu: 2.5, lengthScale: 0.9 })
    const f = (p: KernelParams): Value => sum(base.withParams(p).k(X, Y))
    const g = grad(f)(base.params) as Record<string, Value>
    const fd = numericalGradient((l: Value) => f({ lengthScale: l }), tensor(0.9))
    expect(toFlat(g['lengthScale'] as Value)[0]).toBeCloseTo(toFlat(fd)[0] ?? 0, 7)
  })
  it('gives scikit-learn shaped eval_gradient output, and an empty gradient without hyperparameters', () => {
    const { gram: K, gradient } = gramGradient(k, X)
    expect(K.shape).toEqual([4, 4])
    expect(gradient.shape).toEqual([4, 4, 4])
    expect(gramGradient(linearKernel(), X).gradient.shape).toEqual([4, 4, 0])
  })
  it('fixes hyperparameters, also after withParams', () => {
    const fixed = fixHyperparameters(rationalQuadraticKernel(), ['alpha'])
    expect(theta(fixed).shape).toEqual([1])
    expect(theta(fixed.withParams({ alpha: tensor(3), lengthScale: tensor(2) })).shape).toEqual([1])
    expect(toArray(theta(withTheta(fixed, [Math.log(4)])))).toEqual([Math.log(4)])
    throwsCode(() => fixHyperparameters(rbfKernel(), ['nope']), 'value')
  })
})

describe('kernel algebra', () => {
  const a = rbfKernel({ lengthScale: 0.7 })
  const b = periodicKernel({ period: 1.3 })
  it('sums, multiplies, scales and exponentiates entrywise', () => {
    const ka = toFlat(a.k(X, Y))
    const kb = toFlat(b.k(X, Y))
    const expectEach = (k: Kernel, f: (u: number, v: number) => number): void => {
      toFlat(k.k(X, Y)).forEach((v, i) => {
        expect(v).toBeCloseTo(f(ka[i] ?? 0, kb[i] ?? 0), 13)
      })
    }
    expectEach(sumKernel(a, b), (u, v) => u + v)
    expectEach(productKernel(a, b), (u, v) => u * v)
    expectEach(scaleKernel(a, 2.5), (u) => 2.5 * u)
    expectEach(powerKernel(a, 3), (u) => u ** 3)
    expect(toArray(sumKernel(a, b).diag(X))).toEqual([2, 2, 2, 2])
    expect(toArray(productKernel(a, scaleKernel(b, 3)).diag(X))).toEqual([3, 3, 3, 3])
    expect(sumKernel(a, b).stationary).toBe(true)
    expect(sumKernel(a, linearKernel()).stationary).toBe(false)
    throwsCode(() => sumKernel(), 'value')
  })
  it('restricts to dimensions and widens length scales for ARD', () => {
    const first = onDimensions(a, [0])
    const direct = a.k(
      (toArray(X) as number[][]).map((r) => [r[0] ?? 0]),
      (toArray(Y) as number[][]).map((r) => [r[0] ?? 0]),
    )
    expect(toArray(first.k(X, Y))).toEqual(toArray(direct))
    expect(toArray(first.diag(X))).toEqual([1, 1, 1, 1])
    const wide = ard(sumKernel(a, scaleKernel(rbfKernel({ lengthScale: 2 }))), 2)
    expect(hyperparameters(wide).map((h) => h.shape)).toEqual([[2], [2], []])
    expect(toArray(wide.k(X, Y))).toEqual(toArray(sumKernel(a, scaleKernel(rbfKernel({ lengthScale: 2 }))).k(X, Y)))
  })
  it('ARD follows restrictions and leaves isotropic kernels alone', () => {
    const additive = sumKernel(onDimensions(a, [0]), onDimensions(b, [1]), onDimensions(maternKernel(), [0, 1]))
    const wide = ard(additive, 2)
    expect(hyperparameters(wide).map((h) => [h.name, h.shape])).toEqual([
      ['terms[0].lengthScale', [1]],
      ['terms[1].lengthScale', []],
      ['terms[1].period', []],
      ['terms[2].lengthScale', [2]],
    ])
    expect(toArray(wide.k(X, Y))).toEqual(toArray(additive.k(X, Y)))
    expect(formatKernel(onDimensions(a, [1]))).toBe('rbf(lengthScale=0.7)@[1]')
    expect(ard(linearKernel(), 3).params).toEqual({})
  })
  it('formats compositions', () => {
    expect(formatKernel(productKernel(sumKernel(a, b), constantKernel({ value: 2 })))).toBe(
      '(rbf(lengthScale=0.7) + periodic(lengthScale=1, period=1.3)) * constant(value=2)',
    )
    expect(formatKernel(powerKernel(sumKernel(a, a), 2))).toBe('(rbf(lengthScale=0.7) + rbf(lengthScale=0.7)) ** 2')
    expect(formatKernel(scaleKernel(sumKernel(a, a), 0.5))).toBe('0.5 * (rbf(lengthScale=0.7) + rbf(lengthScale=0.7))')
    expect(formatKernel(maternKernel({ nu: 2.5 }))).toBe('matern(lengthScale=1, nu=2.5)')
    expect(formatKernel(ard(a, 2))).toBe('rbf(lengthScale=[0.7, 0.7])')
  })
  it('adds jitter to a Gram matrix', () => {
    const k = toArray(gram(a, X, { jitter: 0.25 })) as number[][]
    expect(k[0]?.[0]).toBe(1.25)
  })
})

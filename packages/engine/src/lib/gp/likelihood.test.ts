import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import {
  constantKernel,
  gram,
  maternKernel,
  productKernel,
  rbfKernel,
  sumKernel,
  theta,
  whiteNoiseKernel,
  withTheta,
} from '../kernels/index.ts'
import { cholesky, choSolve } from '../numerics/linalg/index.ts'
import { add, diagonal, eye, log, mul, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { hyperparameterCount, logMarginalLikelihood, negativeLogMarginalLikelihood } from './likelihood.ts'

const x = [[0], [0.7], [1.5], [2.2], [3.1]]
const y = [0.1, 0.8, 1.0, 0.3, -0.5]

describe('logMarginalLikelihood', () => {
  it('has a gradient that matches finite differences in theta', () => {
    const kernel = sumKernel(
      productKernel(constantKernel({ value: 1.5 }), maternKernel({ lengthScale: 1.2, nu: 2.5 })),
      whiteNoiseKernel({ noiseLevel: 0.2 }),
    )
    const t = Array.from(toFlat(theta(kernel)))
    const { value, gradient } = logMarginalLikelihood(kernel, x, y, { alpha: 1e-6 })
    expect(gradient).toHaveLength(t.length)
    t.forEach((_, j) => {
      const h = 1e-6
      const up = t.map((v, i) => (i === j ? v + h : v))
      const down = t.map((v, i) => (i === j ? v - h : v))
      const f = (th: number[]): number => logMarginalLikelihood(kernel, x, y, { alpha: 1e-6, theta: th }).value
      expect(gradient[j]).toBeCloseTo((f(up) - f(down)) / (2 * h), 5)
    })
    expect(Number.isFinite(value)).toBe(true)
  })

  it('is the gradient of a plainly written likelihood through grad (the example in the README), the library gradient and a finite difference', () => {
    const xs = tensor([[0], [0.7], [1.5], [2.2]])
    const ys = tensor([0.1, 0.9, 0.8, -0.2])
    const kernel = rbfKernel({ lengthScale: 1 })
    // The negative log marginal likelihood, up to a constant, written as an ordinary function of theta.
    const nll = (t: Value): Value => {
      const L = cholesky(add(gram(withTheta(kernel, t), xs), mul(eye(4), 0.1)))
      const a = choSolve(L, ys)
      return add(mul(sum(mul(ys, a)), 0.5), sum(log(diagonal(L))))
    }
    const viaGrad = toFlat(grad(nll)(tensor([0])))[0] ?? Number.NaN
    const library = logMarginalLikelihood(kernel, xs, ys, { alpha: 0.1 }).gradient[0] ?? Number.NaN
    const at = (t: number): number => logMarginalLikelihood(kernel, xs, ys, { alpha: 0.1, theta: [t] }).value
    const h = 1e-5
    expect(viaGrad).toBeCloseTo(0.1209288, 6)
    expect(viaGrad).toBeCloseTo(-library, 10)
    expect(viaGrad).toBeCloseTo(-(at(h) - at(-h)) / (2 * h), 8)
    const dedicated = toFlat(grad(negativeLogMarginalLikelihood(kernel, xs, ys, { alpha: 0.1 }))(tensor([0])))[0]
    expect(dedicated).toBeCloseTo(viaGrad, 10)
  })

  it('is the Gaussian log density of y under K + alpha I', () => {
    const k = rbfKernel({ lengthScale: 1 })
    const alpha = 0.3
    const { value } = logMarginalLikelihood(k, [[0], [1]], [0.5, -0.2], { alpha })
    // The 2 x 2 case by hand: K = [[1 + a, r], [r, 1 + a]] with r = exp(-1/2).
    const r = Math.exp(-0.5)
    const a = 1 + alpha
    const det = a * a - r * r
    const quad = (a * (0.5 ** 2 + 0.2 ** 2) - 2 * r * 0.5 * -0.2) / det
    expect(value).toBeCloseTo(-0.5 * quad - 0.5 * Math.log(det) - Math.log(2 * Math.PI), 10)
  })

  it('accepts a noise per observation, and reports -infinity (not an error) when K is not positive definite', () => {
    const k = rbfKernel()
    const varied = logMarginalLikelihood(k, x, y, { alpha: tensor([0.1, 0.2, 0.3, 0.4, 0.5]) })
    expect(Number.isFinite(varied.value)).toBe(true)
    const singular = logMarginalLikelihood(k, [[0], [0]], [1, 2], { alpha: 0 })
    expect(singular.value).toBe(-Infinity)
    expect(Array.from(singular.gradient)).toEqual([0])
  })

  it('rejects targets that are not a vector', () => {
    expect(() => logMarginalLikelihood(rbfKernel(), x, [[1], [2], [3], [4], [5]])).toThrow(/y must be/)
  })
})

describe('negativeLogMarginalLikelihood', () => {
  it('is differentiable with the analytic gradient', () => {
    const k = rbfKernel({ lengthScale: 1.1 })
    const f = negativeLogMarginalLikelihood(k, x, y, { alpha: 0.05 })
    const g = toFlat(grad(f)(tensor([Math.log(1.1)])))
    const direct = logMarginalLikelihood(k, x, y, { alpha: 0.05 })
    expect(g[0]).toBeCloseTo(-(direct.gradient[0] ?? 0), 10)
    expect(hyperparameterCount(sumKernel(rbfKernel(), whiteNoiseKernel()))).toBe(2)
  })
})

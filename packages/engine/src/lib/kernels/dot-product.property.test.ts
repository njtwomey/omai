import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { awayFromZero, RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { expectKernelLaws, points } from '../../testing/kernels.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { add, div, matmul, mul, relu, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  arcCosineKernel,
  cosineSimilarityKernel,
  empiricalNtk,
  type Kernel,
  linearKernel,
  neuralTangentKernel,
  polynomialKernel,
  sigmoidKernel,
  theta,
  withTheta,
} from './index.ts'

const KERNELS: readonly (readonly [string, Kernel])[] = [
  ['linear', linearKernel()],
  ['linear with variance and offset', linearKernel({ variance: 1.7, sigma0: 0.6 })],
  ['polynomial', polynomialKernel({ degree: 3, gamma: 0.5, coef0: 0.8 })],
  ['polynomial degree 1', polynomialKernel({ degree: 1 })],
  ['cosine similarity', cosineSimilarityKernel()],
  ['arc-cosine 0', arcCosineKernel({ order: 0 })],
  ['arc-cosine 1', arcCosineKernel({ order: 1 })],
  ['arc-cosine 2', arcCosineKernel({ order: 2 })],
  ['arc-cosine 1, depth 3', arcCosineKernel({ order: 1, depth: 3 })],
  ['ntk relu', neuralTangentKernel({ depth: 2, weightVariance: 2, biasVariance: 0.1 })],
  ['nngp relu', neuralTangentKernel({ depth: 2, weightVariance: 1.5, biasVariance: 0.2, kind: 'nngp' })],
  ['ntk erf', neuralTangentKernel({ depth: 2, activation: 'erf', weightVariance: 1.3, biasVariance: 0.4 })],
]

const X = tensor([
  [0.1, -0.4, 0.5],
  [0.7, 0.2, -0.3],
  [-0.5, 0.9, 0.2],
  [1.2, -1.1, 0.8],
])
const Y = tensor([
  [0.3, 0.35, -0.6],
  [-1, 0.5, 0.1],
])

describe('dot-product kernels (properties)', () => {
  for (const [name, kernel] of KERNELS) {
    it(`${name}: PSD, symmetric and consistent on random points`, () => {
      // Angular kernels are defined for non-zero points; arccos near cos = 1 costs ~sqrt(eps) off the Gram diagonal.
      fc.assert(
        fc.property(awayFromZero([6, 3]), awayFromZero([3, 3]), (x, y) => {
          expectKernelLaws(kernel, x, y, { tol: 1e-8, crossTol: 1e-6 })
        }),
        { numRuns: RUNS },
      )
    })
    it(`${name}: differentiable in the points and in theta`, () => {
      expectConformant((x: Value, y: Value) => kernel.k(x, y), [X, Y])
      expectConformant((x: Value) => kernel.k(x), [X])
      if ((theta(kernel).shape[0] ?? 0) > 0)
        expectConformant((t: Value) => withTheta(kernel, t).k(X, Y), [theta(kernel)])
    })
  }
  it('sigmoid: symmetric and differentiable (not PSD in general)', () => {
    const k = sigmoidKernel({ gamma: 0.4, coef0: -0.3 })
    fc.assert(
      fc.property(points(5, 3), points(2, 3), (x, y) => {
        expectKernelLaws(k, x, y, { psd: false })
      }),
      { numRuns: RUNS },
    )
    expectConformant((t: Value) => withTheta(k, t).k(X, Y), [theta(k)])
  })

  it('arc-cosine kernels are 2 E[step(w.x) step(w.y) (w.x)^n (w.y)^n] for Gaussian w (Monte Carlo)', () => {
    const samples = 200000
    const w = normal(key(3), [3, samples])
    const zx = matmul(X, w)
    const zy = matmul(Y, w)
    for (const order of [0, 1, 2] as const) {
      const feature = (z: Value): number[] => toFlat(z).map((v) => (v > 0 ? v ** order : 0))
      const fx = feature(zx)
      const fy = feature(zy)
      const exact = toFlat(arcCosineKernel({ order }).k(X, Y))
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 2; j++) {
          let s = 0
          for (let t = 0; t < samples; t++) s += (fx[i * samples + t] ?? 0) * (fy[j * samples + t] ?? 0)
          const mc = (2 * s) / samples
          const want = exact[i * 2 + j] ?? 0
          expect(Math.abs(mc - want)).toBeLessThan(0.02 * Math.max(1, Math.abs(want)) + 0.01)
        }
      }
    }
  })

  it('the empirical NTK of a linear model is X X^T, and works on a pytree of parameters', () => {
    const f = (w: Value, x: Value): Value => matmul(x, w)
    const k = toFlat(empiricalNtk(f, tensor([0.2, -1, 0.5]), X, Y))
    const want = toFlat(linearKernel().k(X, Y))
    k.forEach((v, i) => {
      expect(v).toBeCloseTo(want[i] ?? 0, 12)
    })
    // An affine model with a bias: Theta = x . y + 1, summed over two outputs.
    const g = (p: { readonly w: Value; readonly b: Value }, x: Value): Value => add(matmul(x, p.w), p.b)
    const two = toFlat(
      empiricalNtk(
        g,
        {
          w: tensor([
            [1, 0],
            [0, 1],
            [1, 1],
          ]),
          b: tensor([0, 0]),
        },
        X,
      ),
    )
    const xx = toFlat(linearKernel({ sigma0: 1 }).k(X))
    two.forEach((v, i) => {
      expect(v).toBeCloseTo(2 * (xx[i] ?? 0), 12)
    })
  })

  it('the empirical NTK of a wide random ReLU network approaches the infinite-width NTK', () => {
    const wv = 2
    const bv = 0.1
    const analytic = toFlat(neuralTangentKernel({ depth: 1, weightVariance: wv, biasVariance: bv }).k(X, Y))
    const error = (width: number, seed: number): number => {
      const k = key(seed)
      const params = {
        w1: normal(foldIn(k, 0), [3, width]),
        b1: normal(foldIn(k, 1), [width]),
        w2: normal(foldIn(k, 2), [width]),
        b2: normal(foldIn(k, 3), []),
      }
      // The NTK parameterisation: W_std / sqrt(fan_in) and b_std, so that the kernel has a width-independent limit.
      const net = (p: typeof params, x: Value): Value => {
        const h = relu(add(mul(div(matmul(x, p.w1), Math.sqrt(3)), Math.sqrt(wv)), mul(p.b1, Math.sqrt(bv))))
        return add(mul(div(matmul(h, p.w2), Math.sqrt(width)), Math.sqrt(wv)), mul(p.b2, Math.sqrt(bv)))
      }
      const emp = toFlat(empiricalNtk(net, params, X, Y))
      let worst = 0
      emp.forEach((v, i) => {
        const a = analytic[i] ?? 0
        worst = Math.max(worst, Math.abs(v - a) / Math.max(1, Math.abs(a)))
      })
      return worst
    }
    const narrow = [1, 2, 3].map((s) => error(16, s))
    const wide = [1, 2, 3].map((s) => error(4096, s))
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(wide)).toBeLessThan(mean(narrow))
    expect(mean(wide)).toBeLessThan(0.1)
  })

  it('rejects invalid settings', () => {
    expect(() => polynomialKernel({ degree: 1.5 })).toThrow()
    expect(() => arcCosineKernel({ order: 3 as 0 })).toThrow()
    expect(() => arcCosineKernel({ depth: 0 })).toThrow()
    expect(() => neuralTangentKernel({ depth: 0 })).toThrow()
  })
})

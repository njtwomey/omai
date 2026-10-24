import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { expectKernelLaws, points } from '../../testing/kernels.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  cosineKernel,
  exponentialKernel,
  type Kernel,
  laplacianKernel,
  locallyPeriodicKernel,
  maternKernel,
  periodicKernel,
  rationalQuadraticKernel,
  rbfKernel,
  spectralMixtureKernel,
  theta,
  withTheta,
} from './index.ts'

const KERNELS: readonly (readonly [string, Kernel])[] = [
  ['rbf', rbfKernel({ lengthScale: 0.8 })],
  ['rbf ard', rbfKernel({ lengthScale: [0.6, 1.5] })],
  ['matern 1/2', maternKernel({ nu: 0.5, lengthScale: 0.9 })],
  ['matern 3/2', maternKernel({ nu: 1.5, lengthScale: [0.7, 1.3] })],
  ['matern 5/2', maternKernel({ nu: 2.5 })],
  ['matern inf', maternKernel({ nu: Number.POSITIVE_INFINITY, lengthScale: 1.1 })],
  ['matern 0.8', maternKernel({ nu: 0.8, lengthScale: 1.2 })],
  ['matern 3.5', maternKernel({ nu: 3.5, lengthScale: 0.6 })],
  ['rational quadratic', rationalQuadraticKernel({ alpha: 0.6, lengthScale: 1.2 })],
  ['gamma-exponential', exponentialKernel({ power: 1.5, lengthScale: [0.7, 1.1] })],
  ['laplacian', laplacianKernel({ lengthScale: [0.5, 1.5] })],
  ['periodic', periodicKernel({ lengthScale: 0.9, period: 1.7 })],
  ['locally periodic', locallyPeriodicKernel({ lengthScale: 0.8, period: 1.4, decay: 1.9 })],
  [
    'spectral mixture',
    spectralMixtureKernel({
      weights: [0.5, 1.2],
      means: [
        [0.2, 0.4],
        [0.9, 0.1],
      ],
      scales: [
        [0.3, 0.2],
        [0.1, 0.4],
      ],
    }),
  ],
]

const X = tensor([
  [0.1, -0.4],
  [0.7, 0.2],
  [-0.5, 0.9],
  [1.2, -1.1],
])
const Y = tensor([
  [0.3, 0.35],
  [-1, 0.5],
  [0.2, -0.9],
])

describe('stationary kernels (properties)', () => {
  for (const [name, kernel] of KERNELS) {
    it(`${name}: PSD, symmetric, diagonal and shift-invariant on random points`, () => {
      fc.assert(
        fc.property(points(7, 2), points(3, 2), (x, y) => {
          expectKernelLaws(kernel, x, y)
        }),
        { numRuns: RUNS },
      )
    })
    it(`${name}: differentiable in the points and in theta (conformance, vmap included)`, () => {
      expectConformant((x: Value) => kernel.k(x), [X])
      expectConformant((x: Value, y: Value) => kernel.k(x, y), [X, Y])
      expectConformant((t: Value) => withTheta(kernel, t).k(X, Y), [theta(kernel)])
    })
  }
  it('cosine: PSD in one dimension, and conformant', () => {
    const k = cosineKernel({ period: 1.3 })
    fc.assert(
      fc.property(points(6, 1), points(3, 1), (x, y) => {
        expectKernelLaws(k, x, y)
      }),
      { numRuns: RUNS },
    )
    expectConformant((x: Value, t: Value) => withTheta(k, t).k(x, [[0.4], [-1.2]]), [[[0.1], [0.9]], theta(k)])
  })
  it('the Bessel form of Matérn approaches the closed forms', () => {
    for (const nu of [0.5, 1.5, 2.5]) {
      const closed = toFlat(maternKernel({ nu, lengthScale: 0.7 }).k(X, Y))
      const general = toFlat(maternKernel({ nu: nu + 1e-9, lengthScale: 0.7 }).k(X, Y))
      closed.forEach((v, i) => {
        expect(general[i]).toBeCloseTo(v, 7)
      })
    }
    // Large nu tends to the RBF kernel.
    const rbf = toFlat(rbfKernel({ lengthScale: 0.7 }).k(X, Y))
    toFlat(maternKernel({ nu: 60, lengthScale: 0.7 }).k(X, Y)).forEach((v, i) => {
      expect(Math.abs(v - (rbf[i] ?? 0))).toBeLessThan(5e-3)
    })
  })
  it('the gamma-exponential kernel with exponent 2 is an RBF kernel', () => {
    const a = toFlat(exponentialKernel({ power: 2, lengthScale: 1.4 }).k(X, Y))
    const b = toFlat(rbfKernel({ lengthScale: 1.4 / Math.SQRT2 }).k(X, Y))
    a.forEach((v, i) => {
      expect(v).toBeCloseTo(b[i] ?? 0, 13)
    })
  })
  it('a long rational-quadratic alpha tends to the RBF kernel', () => {
    const a = toFlat(rationalQuadraticKernel({ alpha: 1e7, lengthScale: 0.9 }).k(X, Y))
    const b = toFlat(rbfKernel({ lengthScale: 0.9 }).k(X, Y))
    a.forEach((v, i) => {
      expect(v).toBeCloseTo(b[i] ?? 0, 6)
    })
  })
  it('rejects invalid settings', () => {
    expect(() => maternKernel({ nu: 0 })).toThrow()
    expect(() => exponentialKernel({ power: 2.5 })).toThrow()
    expect(() => periodicKernel({ lengthScale: [1, 2] })).toThrow()
    expect(() => spectralMixtureKernel({ weights: [1, 2], means: [[1]], scales: [[1]] })).toThrow()
    expect(() => spectralMixtureKernel({ weights: [1], means: [[1, 2]], scales: [[1, 2]] }).k([[1]])).toThrow()
  })
})

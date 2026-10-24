import fc from 'fast-check'
import { describe, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { expectKernelLaws, points } from '../../testing/kernels.ts'
import { tensor, type Value } from '../tensor/index.ts'
import {
  ard,
  constantKernel,
  type Kernel,
  linearKernel,
  maternKernel,
  onDimensions,
  periodicKernel,
  polynomialKernel,
  powerKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  scaleKernel,
  sumKernel,
  theta,
  withTheta,
} from './index.ts'

const BASE: readonly Kernel[] = [
  rbfKernel({ lengthScale: 0.7 }),
  maternKernel({ nu: 1.5, lengthScale: 1.3 }),
  rationalQuadraticKernel({ alpha: 0.5 }),
  periodicKernel({ period: 1.6 }),
  linearKernel({ sigma0: 0.5 }),
  polynomialKernel({ degree: 2, gamma: 0.5 }),
  constantKernel({ value: 0.4 }),
]

/** Random kernel expressions of depth two over PSD base kernels. */
const expressions: fc.Arbitrary<Kernel> = fc
  .tuple(
    fc.constantFrom(...BASE),
    fc.constantFrom(...BASE),
    fc.constantFrom(...BASE),
    fc.double({ min: 0.1, max: 3, noNaN: true }),
    fc.integer({ min: 0, max: 6 }),
  )
  .map(([a, b, c, s, op]) => {
    switch (op) {
      case 0:
        return sumKernel(a, b, c)
      case 1:
        return productKernel(a, b)
      case 2:
        return scaleKernel(productKernel(a, sumKernel(b, c)), s)
      case 3:
        return powerKernel(sumKernel(a, b), 2)
      case 4:
        return sumKernel(onDimensions(a, [0]), onDimensions(b, [1]))
      case 5:
        return ard(productKernel(a, c), 2)
      default:
        return productKernel(scaleKernel(a, s), powerKernel(b, 3))
    }
  })

describe('kernel algebra (properties)', () => {
  it('sums, products, scalings, integer powers, restrictions and ARD of PSD kernels are PSD kernels', () => {
    fc.assert(
      fc.property(expressions, points(7, 2), points(3, 2), (k, x, y) => {
        expectKernelLaws(k, x, y, { tol: 1e-8 })
      }),
      { numRuns: 3 * RUNS },
    )
  })
  it('compositions are differentiable in theta and in the points', () => {
    const X = tensor([
      [0.1, -0.4],
      [0.7, 0.2],
      [-0.5, 0.9],
    ])
    const Y = tensor([
      [0.3, 0.35],
      [-1, 0.5],
    ])
    const kernels = [
      sumKernel(rbfKernel({ lengthScale: [0.6, 1.4] }), linearKernel({ sigma0: 0.3, variance: 1.2 })),
      productKernel(periodicKernel({ period: 1.2 }), scaleKernel(maternKernel({ nu: 2.5 }), 1.4)),
      powerKernel(rationalQuadraticKernel({ alpha: 1.3 }), 2),
      sumKernel(onDimensions(rbfKernel(), [1]), onDimensions(maternKernel({ nu: 1.5 }), [0])),
    ]
    for (const k of kernels) {
      expectConformant((t: Value) => withTheta(k, t).k(X, Y), [theta(k)])
      expectConformant((t: Value) => withTheta(k, t).k(X), [theta(k)])
      expectConformant((x: Value) => k.k(x, Y), [X])
    }
  })
})

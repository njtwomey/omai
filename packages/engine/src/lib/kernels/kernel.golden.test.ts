import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { expectGpCase } from '../../testing/kernels.ts'
import type { Value } from '../tensor/index.ts'
import {
  constantKernel,
  fixHyperparameters,
  type Kernel,
  linearKernel,
  powerKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  scaleKernel,
  sumKernel,
  whiteNoiseKernel,
} from './index.ts'

const KERNELS: Record<string, () => Kernel> = {
  'constant*rbf+white': () =>
    sumKernel(
      productKernel(constantKernel({ value: 2 }), rbfKernel({ lengthScale: 1.5 })),
      whiteNoiseKernel({ noiseLevel: 0.1 }),
    ),
  'rbf-ard*dot-product': () => productKernel(rbfKernel({ lengthScale: [0.7, 1.9] }), linearKernel({ sigma0: 0.5 })),
  'rational-quadratic**2': () => powerKernel(rationalQuadraticKernel({ lengthScale: 1.2, alpha: 0.8 }), 2),
  'fixed-rbf+white': () =>
    sumKernel(
      fixHyperparameters(rbfKernel({ lengthScale: 1 }), ['lengthScale']),
      whiteNoiseKernel({ noiseLevel: 0.3 }),
    ),
  '(rbf+dot-product)*constant': () =>
    productKernel(
      sumKernel(rbfKernel({ lengthScale: 0.8 }), linearKernel({ sigma0: 1.1 })),
      constantKernel({ value: 0.6 }),
    ),
}

describe('kernel algebra against scikit-learn and GPyTorch', () => {
  for (const c of loadFixture('kernels/kernel').cases) {
    it(c.name, () => {
      if (c.fn === 'gp') {
        const make = KERNELS[c.params['kernel'] as string]
        if (make === undefined) throw new Error(`no kernel for ${c.name}`)
        return expectGpCase(c, make())
      }
      const [x, y] = c.inputs.map(decode) as [Value, Value]
      expectMatches(scaleKernel(rbfKernel({ lengthScale: [0.6, 1.4] }), 1.7).k(x, y), c.output, c.rtol, c.atol, c.name)
    })
  }
})

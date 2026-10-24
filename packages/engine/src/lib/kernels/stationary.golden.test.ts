import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { expectGpCase } from '../../testing/kernels.ts'
import type { Value } from '../tensor/index.ts'
import {
  cosineKernel,
  exponentialKernel,
  type Kernel,
  laplacianKernel,
  locallyPeriodicKernel,
  maternKernel,
  periodicKernel,
  productKernel,
  rationalQuadraticKernel,
  rbfKernel,
  spectralMixtureKernel,
} from './index.ts'

const nuOf = (s: unknown): number => (s === 'inf' ? Number.POSITIVE_INFINITY : Number(s))

function gpKernel(params: Readonly<Record<string, unknown>>): Kernel {
  switch (params['kernel']) {
    case 'rbf':
      return rbfKernel({ lengthScale: params['lengthScale'] as number | number[] })
    case 'rq':
      return rationalQuadraticKernel({ lengthScale: 0.9, alpha: 1.7 })
    case 'periodic':
      return periodicKernel({ lengthScale: 1.2, period: 2.5 })
    case 'periodic-1d':
      return periodicKernel({ lengthScale: 0.7, period: 1.3 })
    case 'locally-periodic':
      return productKernel(periodicKernel({ lengthScale: 1.1, period: 3 }), rbfKernel({ lengthScale: 2 }))
    case 'matern':
      return maternKernel({ lengthScale: 0.8, nu: nuOf(params['nu']) })
    case 'matern-ard':
      return maternKernel({ lengthScale: [0.6, 1.7], nu: nuOf(params['nu']) })
    case 'matern-general':
      return maternKernel({ lengthScale: 1.1, nu: nuOf(params['nu']) })
    default:
      throw new Error(`unknown kernel ${String(params['kernel'])}`)
  }
}

describe('stationary kernels against scikit-learn, GPyTorch and scipy', () => {
  for (const c of loadFixture('kernels/stationary').cases) {
    it(c.name, () => {
      if (c.fn === 'gp') {
        expectGpCase(c, gpKernel(c.params))
        if (c.params['kernel'] === 'locally-periodic') {
          // The same kernel in one piece: theta [decay, lengthScale, period] against scikit-learn's
          // [k1__length_scale, k1__periodicity, k2__length_scale].
          expectGpCase(c, locallyPeriodicKernel({ lengthScale: 1.1, period: 3, decay: 2 }), [1, 2, 0])
        }
        if (c.params['kernel'] === 'matern' && c.params['nu'] === '0.5') {
          expectGpCase(c, exponentialKernel({ lengthScale: 0.8 }))
        }
        return
      }
      const inputs = c.inputs.map(decode) as Value[]
      const [x, y] = inputs as [Value, Value]
      const match = (got: Value): void => expectMatches(got, c.output, c.rtol, c.atol, c.name)
      switch (c.fn) {
        case 'matern':
          return match(
            maternKernel({ nu: c.params['nu'] as number, lengthScale: c.params['lengthScale'] as number }).k(x, y),
          )
        case 'laplacian':
          return match(laplacianKernel({ lengthScale: 1 / (c.params['gamma'] as number) }).k(x, y))
        case 'exponential':
          return match(exponentialKernel({ lengthScale: [0.9, 1.6], power: c.params['power'] as number }).k(x, y))
        case 'spectralMixture': {
          const [, , weights, means, scales] = inputs as [Value, Value, Value, Value, Value]
          return match(spectralMixtureKernel({ weights, means, scales }).k(x, y))
        }
        case 'cosine':
          return match(cosineKernel({ period: c.params['period'] as number }).k(x, y))
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

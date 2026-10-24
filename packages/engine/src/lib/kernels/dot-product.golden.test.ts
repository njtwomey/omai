import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { expectGpCase } from '../../testing/kernels.ts'
import type { Value } from '../tensor/index.ts'
import {
  arcCosineKernel,
  cosineSimilarityKernel,
  linearKernel,
  neuralTangentKernel,
  polynomialKernel,
  sigmoidKernel,
} from './index.ts'

describe('dot-product kernels against scikit-learn, GPyTorch and the closed forms', () => {
  for (const c of loadFixture('kernels/dot_product').cases) {
    it(c.name, () => {
      const [x, y] = c.inputs.map(decode) as [Value, Value]
      const out = c.output as Record<string, unknown>
      const p = c.params
      const match = (got: Value, expected: unknown = c.output, label = c.name): void =>
        expectMatches(got, expected, c.rtol, c.atol, label)
      switch (c.fn) {
        case 'gp':
          return expectGpCase(c, linearKernel({ sigma0: 0.8 }))
        case 'linear':
          return match(linearKernel().k(x, y))
        case 'linearVariance':
          return match(linearKernel({ variance: p['variance'] as number }).k(x, y))
        case 'polynomial':
          return match(
            polynomialKernel({
              degree: p['degree'] as number,
              gamma: p['gamma'] as number,
              coef0: p['coef0'] as number,
            }).k(x, y),
          )
        case 'sigmoid':
          return match(sigmoidKernel({ gamma: p['gamma'] as number, coef0: p['coef0'] as number }).k(x, y))
        case 'cosineSimilarity':
          return match(cosineSimilarityKernel().k(x, y))
        case 'cosineSimilarityGram':
          return match(cosineSimilarityKernel().k(x))
        case 'arcCosine': {
          const k = arcCosineKernel({ order: p['order'] as 0 | 1 | 2, depth: p['depth'] as number })
          match(k.k(x, y), out['cross'], `${c.name} cross`)
          return match(k.k(x), out['gram'], `${c.name} gram`)
        }
        case 'ntk': {
          const options = {
            activation: p['activation'] as 'relu' | 'erf',
            depth: p['depth'] as number,
            weightVariance: p['weightVariance'] as number,
            biasVariance: p['biasVariance'] as number,
          }
          const ntk = neuralTangentKernel(options)
          const nngp = neuralTangentKernel({ ...options, kind: 'nngp' })
          match(ntk.k(x, y), out['ntk'], `${c.name} ntk`)
          match(nngp.k(x, y), out['nngp'], `${c.name} nngp`)
          match(ntk.k(x), out['ntkGram'], `${c.name} ntk gram`)
          return match(nngp.k(x), out['nngpGram'], `${c.name} nngp gram`)
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

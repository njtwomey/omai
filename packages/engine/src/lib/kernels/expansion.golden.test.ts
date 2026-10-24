import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Value } from '../tensor/index.ts'
import {
  additiveChi2Sampler,
  nystroem,
  polynomialCountSketch,
  polynomialKernel,
  randomFourierFeatures,
  rbfKernel,
  skewedChi2Sampler,
} from './index.ts'

describe('feature maps against scikit-learn kernel_approximation, given the same draws', () => {
  for (const c of loadFixture('kernels/expansion').cases) {
    it(c.name, () => {
      const [x, a, b] = c.inputs.map(decode) as [Value, Value, Value]
      const out = c.output as Record<string, unknown>
      const p = c.params
      const match = (got: Value, expected: unknown = c.output, label = c.name): void =>
        expectMatches(got, expected, c.rtol, c.atol, label)
      switch (c.fn) {
        case 'rbfSampler':
          return match(randomFourierFeatures({ weights: a, offset: b }).transform(x))
        case 'nystroem': {
          const gamma = p['gamma'] as number
          const kernel =
            p['kernel'] === 'rbf'
              ? rbfKernel({ lengthScale: 1 / Math.sqrt(2 * gamma) })
              : polynomialKernel({ gamma, degree: p['degree'] as number, coef0: p['coef0'] as number })
          const map = nystroem(kernel, x, { indices: p['indices'] as number[] })
          match(map.normalization, out['normalization'], `${c.name} normalization`)
          return match(map.transform(x), out['transform'], `${c.name} transform`)
        }
        case 'additiveChi2': {
          const interval = p['interval'] as number | undefined
          const map = additiveChi2Sampler({
            sampleSteps: p['steps'] as number,
            ...(interval === undefined ? {} : { sampleInterval: interval }),
          })
          return match(map.transform(x))
        }
        case 'skewedChi2':
          return match(skewedChi2Sampler({ weights: a, offset: b, skewedness: p['skewedness'] as number }).transform(x))
        case 'polynomialCountSketch':
          return match(
            polynomialCountSketch({
              degree: p['degree'] as number,
              coef0: p['coef0'] as number,
              gamma: p['gamma'] as number,
              nComponents: 16,
              indexHash: p['indexHash'] as number[][],
              bitHash: p['bitHash'] as number[][],
            }).transform(x),
          )
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

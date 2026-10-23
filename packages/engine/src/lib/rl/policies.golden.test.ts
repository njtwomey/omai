import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { boltzmannProbabilities, categoricalPolicy, gaussianPolicy } from './policies.ts'

describe('policy heads against torch.distributions', () => {
  for (const c of loadFixture('rl/policies').cases) {
    it(c.name, () => {
      const [a, b, d] = c.inputs.map((v) => decode(v) as Tensor)
      let got: Tensor | ReturnType<typeof tensor>
      switch (c.fn) {
        case 'categoricalLogProb':
          got = categoricalPolicy(a as Tensor).logProb(b as Tensor) as Tensor
          break
        case 'categoricalEntropy':
          got = categoricalPolicy(a as Tensor).entropy() as Tensor
          break
        case 'boltzmannProbabilities':
          got = tensor(boltzmannProbabilities(toFlat(a as Tensor), c.params['temperature'] as number))
          break
        case 'gaussianLogProb':
          got = gaussianPolicy(a as Tensor, b as Tensor).logProb(d as Tensor) as Tensor
          break
        case 'gaussianEntropy':
          got = gaussianPolicy(a as Tensor, b as Tensor).entropy() as Tensor
          break
        case 'squashedLogProb':
          got = gaussianPolicy(a as Tensor, b as Tensor, { squash: true }).logProb(d as Tensor) as Tensor
          break
        default:
          got = gaussianPolicy(a as Tensor, b as Tensor, {
            squash: true,
            low: c.params['low'] as number[],
            high: c.params['high'] as number[],
          }).logProb(d as Tensor) as Tensor
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

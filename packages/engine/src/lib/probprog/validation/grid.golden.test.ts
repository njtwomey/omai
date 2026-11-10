import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { bernoulli, normal } from '../../dists/index.ts'
import { add, mul, type Tensor, tensor } from '../../tensor/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import { gridPosterior } from './grid.ts'

describe('probprog/validation/grid against scipy quadrature', () => {
  for (const c of loadFixture('probprog/validation/grid').cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const model: Model = () => {
        const b0 = sample('b0', normal(0, 2))
        const b1 = sample('b1', normal(0, 2))
        plate('n', 16, () => observe('y', bernoulli({ logits: add(b0, mul(b1, x)) }), y))
      }
      const g = gridPosterior(model, undefined, { points: 61 })
      const m0 = g.marginal('b0')
      const m1 = g.marginal('b1')
      expectMatches(tensor([g.logEvidence, m0.mean, m0.sd, m1.mean, m1.sd]), c.output, c.rtol, c.atol, c.name)
    })
  }
})

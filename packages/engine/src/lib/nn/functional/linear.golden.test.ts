import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { bilinear, embedding, embeddingBag } from './linear.ts'

describe('linear maps and embeddings against torch', () => {
  for (const c of loadFixture('nn/functional/linear').cases) {
    it(c.name, () => {
      const [a, b, x, w] = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      let y: Value
      if (c.fn === 'bilinear') y = bilinear(a as Tensor, b as Tensor, x as Tensor, w as Tensor)
      else if (c.fn === 'embedding')
        y = embedding(a as Tensor, b as Tensor, p['maxNorm'] === undefined ? {} : { maxNorm: p['maxNorm'] as number })
      else
        y = embeddingBag(a as Tensor, b as Tensor, {
          mode: p['mode'] as 'sum' | 'mean' | 'max',
          ...(p['offsets'] === undefined ? {} : { offsets: p['offsets'] as number[] }),
          ...(x === undefined ? {} : { perSampleWeights: x }),
          ...(p['paddingIdx'] === undefined ? {} : { paddingIdx: p['paddingIdx'] as number }),
          ...(p['includeLastOffset'] === undefined ? {} : { includeLastOffset: true }),
        })
      expectMatches(y, c.output, 1e-12, 1e-12, c.name)
    })
  }
})

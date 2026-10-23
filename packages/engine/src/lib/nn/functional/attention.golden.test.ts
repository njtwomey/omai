import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor } from '../../tensor/index.ts'
import { scaledDotProductAttention } from './attention.ts'

describe('attention against torch', () => {
  for (const c of loadFixture('nn/functional/attention').cases) {
    it(c.name, () => {
      const [q, k, v, mask] = c.inputs.map((x) => decode(x) as Tensor)
      const p = c.params
      const y = scaledDotProductAttention(q as Tensor, k as Tensor, v as Tensor, {
        ...(mask === undefined ? {} : { mask }),
        ...(p['scale'] === undefined ? {} : { scale: p['scale'] as number }),
        ...(p['causal'] === undefined ? {} : { causal: true }),
        ...(p['enableGqa'] === undefined ? {} : { enableGqa: true }),
      })
      expectMatches(y, c.output, 1e-12, 1e-13, c.name)
    })
  }
})

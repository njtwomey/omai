import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor } from '../../tensor/index.ts'
import { CausalConv1d, MaskedConv2d, type PixelCnnMaskType } from './masked.ts'

type Params = Readonly<Record<string, unknown>>

const stateOf = (p: Params): Record<string, Tensor> =>
  Object.fromEntries(Object.entries(p['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))

describe('causal and masked convolutions against torch', () => {
  for (const c of loadFixture('nn/blocks/masked').cases) {
    it(c.name, () => {
      const cfg = c.params['config'] as Params
      const x = decode(c.inputs[0]) as Tensor
      if (c.fn === 'causalConv1d') {
        const m = new CausalConv1d(cfg['in'] as number, cfg['out'] as number, cfg['kernel'] as number, {
          dilation: cfg['dilation'] as number,
          groups: cfg['groups'] as number,
        }).loadStateDict(stateOf(c.params))
        expectMatches(m.forward(x), c.output, 1e-11, 1e-12, c.name)
        return
      }
      const m = new MaskedConv2d(
        cfg['in'] as number,
        cfg['out'] as number,
        [cfg['kh'] as number, cfg['kw'] as number],
        cfg['type'] as PixelCnnMaskType,
        { colours: cfg['colours'] as number, dilation: cfg['dilation'] as number },
      ).loadStateDict(stateOf(c.params))
      // The mask itself is part of the contract: it matches the reference's.
      expectMatches(m.mask, c.params['mask'], 0, 0, `${c.name} mask`)
      expectMatches(m.forward(x), c.output, 1e-11, 1e-12, c.name)
    })
  }
})

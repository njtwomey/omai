import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { RGCNConv } from './graph.ts'

type Params = Readonly<Record<string, unknown>>

describe('relational graph convolution against torch', () => {
  for (const c of loadFixture('nn/blocks/graph').cases) {
    it(c.name, () => {
      const cfg = c.params['config'] as Params
      const [x, edgeIndex, types] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor, Tensor]
      const state = Object.fromEntries(
        Object.entries(c.params['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]),
      )
      const bases = cfg['bases'] as number
      const m = new RGCNConv(cfg['in'] as number, cfg['out'] as number, cfg['relations'] as number, {
        aggregate: cfg['aggregate'] as 'mean' | 'sum',
        ...(bases > 0 ? { numBases: bases } : {}),
      }).loadStateDict(state)
      expectMatches(m.forward({ x, edgeIndex, edgeType: toFlat(types) }), c.output, 1e-11, 1e-12, c.name)
    })
  }
})

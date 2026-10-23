import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, TensorLike, Value } from '../../tensor/index.ts'
import { MultiheadAttention, type MultiheadAttentionOptions } from './attention.ts'
import { Transformer, TransformerDecoderLayer, TransformerEncoderLayer } from './transformer.ts'

type Params = Readonly<Record<string, unknown>>

const stateOf = (p: Params): Record<string, Tensor> =>
  Object.fromEntries(Object.entries(p['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))

describe('nn.MultiheadAttention against torch', () => {
  for (const c of loadFixture('nn/module/attention').cases) {
    it(c.name, () => {
      const cfg = c.params['config'] as Params
      const m = new MultiheadAttention(
        cfg['embedDim'] as number,
        cfg['numHeads'] as number,
        cfg as MultiheadAttentionOptions,
      )
      m.loadStateDict(stateOf(c.params))
      const xs = c.inputs.map((x) => decode(x) as Tensor)
      const count = c.params['inputCount'] as number
      const masks = c.params['masks'] as string[]
      const maskOf = (name: string): TensorLike | undefined => {
        const i = masks.indexOf(name)
        return i < 0 ? undefined : xs[count + i]
      }
      const padding = maskOf('keyPaddingMask')
      const attnMask = maskOf('attnMask')
      const r = m.forward({
        query: xs[0] as Tensor,
        ...(count > 1 ? { key: xs[1] as Tensor } : {}),
        ...(count > 2 ? { value: xs[2] as Tensor } : {}),
        ...(padding === undefined ? {} : { keyPaddingMask: padding }),
        ...(attnMask === undefined ? {} : { attnMask }),
        averageAttnWeights: c.params['average'] as boolean,
      })
      const got = c.params['part'] === 'output' ? r.output : (r.weights as Value)
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('transformer layers against torch', () => {
  for (const c of loadFixture('nn/module/transformer').cases) {
    it(c.name, () => {
      const cfg = c.params['config'] as Params
      const options = {
        dimFeedforward: cfg['ff'] as number,
        dropout: 0,
        batchFirst: true,
        activation: (cfg['activation'] as 'relu' | 'gelu' | undefined) ?? 'relu',
        normFirst: (cfg['normFirst'] as boolean | undefined) ?? false,
      }
      const d = cfg['dModel'] as number
      const h = cfg['nhead'] as number
      const xs = c.inputs.map((x) => decode(x) as Tensor)
      const state = stateOf(c.params)
      let y: Value
      if (c.fn === 'encoderLayer' || c.fn === 'encoderLayerMasked') {
        const layer = new TransformerEncoderLayer(d, h, options).loadStateDict(state)
        y =
          c.fn === 'encoderLayer'
            ? layer.forward(xs[0] as Tensor)
            : layer.forward({ src: xs[0] as Tensor, mask: xs[1] as Tensor, keyPaddingMask: xs[2] as Tensor })
      } else if (c.fn === 'decoderLayer') {
        const layer = new TransformerDecoderLayer(d, h, options).loadStateDict(state)
        y = layer.forward({ tgt: xs[0] as Tensor, memory: xs[1] as Tensor, tgtMask: xs[2] as Tensor })
      } else {
        const n = cfg['layers'] as number
        const model = new Transformer(d, h, { ...options, numEncoderLayers: n, numDecoderLayers: n }).loadStateDict(
          state,
        )
        y = model.forward({ src: xs[0] as Tensor, tgt: xs[1] as Tensor, tgtMask: xs[2] as Tensor })
      }
      expectMatches(y, c.output, Math.max(c.rtol, 1e-11), Math.max(c.atol, 1e-12), c.name)
    })
  }
})

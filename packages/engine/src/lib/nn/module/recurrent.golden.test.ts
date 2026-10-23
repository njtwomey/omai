import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { type PackedSequence, packPaddedSequence, padPackedSequence } from '../functional/recurrent.ts'
import type { Module } from './module.ts'
import { GRU, GRUCell, LSTM, LSTMCell, RNN, type RNNBase, RNNCell } from './recurrent.ts'

type Params = Readonly<Record<string, unknown>>

const stateOf = (p: Params): Record<string, Tensor> =>
  Object.fromEntries(Object.entries(p['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))

function build(cfg: Params): RNNBase {
  const options = {
    numLayers: (cfg['layers'] as number | undefined) ?? 1,
    bidirectional: (cfg['bidirectional'] as boolean | undefined) ?? false,
    batchFirst: (cfg['batchFirst'] as boolean | undefined) ?? false,
  }
  const i = cfg['input'] as number
  const h = cfg['hidden'] as number
  if (cfg['kind'] === 'LSTM') return new LSTM(i, h, { ...options, projSize: (cfg['proj'] as number | undefined) ?? 0 })
  if (cfg['kind'] === 'GRU') return new GRU(i, h, options)
  return new RNN(i, h, { ...options, nonlinearity: cfg['relu'] === true ? 'relu' : 'tanh' })
}

describe('recurrent modules against torch', () => {
  for (const c of loadFixture('nn/module/recurrent').cases) {
    it(c.name, () => {
      const cfg = c.params['config'] as Params
      const xs = c.inputs.map((x) => decode(x) as Tensor)
      const part = c.params['part'] as string | undefined
      let got: Value
      if (c.fn === 'cell') {
        const kind = cfg['kind'] as string
        const [x, h, cc] = xs as [Tensor, Tensor, Tensor | undefined]
        if (kind === 'LSTMCell') {
          const r = (new LSTMCell(4, 5).loadStateDict(stateOf(c.params)) as LSTMCell).forward({
            input: x,
            h,
            c: cc as Tensor,
          })
          got = part === 'c' ? r[1] : r[0]
        } else {
          const m: Module<never, unknown> = kind === 'RNNCell' ? new RNNCell(4, 5) : new GRUCell(4, 5)
          m.loadStateDict(stateOf(c.params))
          got = (m as RNNCell).forward({ input: x, h })
        }
      } else {
        const m = build(cfg).loadStateDict(stateOf(c.params))
        if (c.fn === 'packed') {
          const lengths = cfg['lengths'] as number[]
          const r = m.forward(packPaddedSequence(xs[0] as Tensor, lengths))
          got = part === 'h' ? r.h : padPackedSequence(r.output as PackedSequence).padded
        } else {
          const r = m.forward({
            input: xs[0] as Tensor,
            ...(xs.length > 1 ? { h0: xs[1] as Tensor, c0: xs[2] as Tensor } : {}),
          })
          got = (part === 'output' ? r.output : part === 'h' ? r.h : r.c) as Value
        }
      }
      expectMatches(got, c.output, 1e-11, 1e-12, c.name)
    })
  }
})

import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { concat, sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import { DeltaNetMixer, GatedLinearAttentionMixer, LinearAttentionMixer, RetNetMixer } from './linear-attention.ts'
import type { MixerState, SequenceMixer } from './mixer.ts'
import { sequenceMixers } from './mixers.ts'
import { Hyena, MinGRU, MinLSTM, RWKVMixer } from './recurrent-mixers.ts'
import { MixerBlock, SequenceModel } from './sequence-model.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value, digits = 9): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}

const generate = (m: SequenceMixer, x: Value): Value => {
  let state: MixerState | undefined
  const outs: Value[] = []
  for (let t = 0; t < (x.shape[1] ?? 0); t++) {
    const r = m.forward({ x: sliceAxis(x, 1, t, t + 1), ...(state === undefined ? {} : { state }) })
    outs.push(r.output)
    state = r.state
  }
  return concat(outs, 1)
}

describe('every mixer of the table', () => {
  const dim = 8
  const x = n(1, [2, 9, dim])
  for (const name of sequenceMixers.names) {
    describe(name, () => {
      const make = (): SequenceMixer => sequenceMixers.make({ kind: name, stateSize: 4 })(dim, key(2)) as SequenceMixer
      const m = make()

      it('maps [B, L, D] to [B, L, D] causally', () => {
        const y = m.forward(x)
        expect(y.output.shape).toEqual([2, 9, dim])
        const x2 = concat([sliceAxis(x, 1, 0, 5), n(3, [2, 4, dim])], 1)
        close(sliceAxis(m.forward(x2).output, 1, 0, 5), sliceAxis(y.output, 1, 0, 5), 10)
      })

      it('generates token by token with the outputs of the parallel pass, and resumes from a state', () => {
        const par = m.forward(x).output
        close(generate(m, x), par, 8)
        const head = m.forward(sliceAxis(x, 1, 0, 4))
        const tail = m.forward({ x: sliceAxis(x, 1, 4, 9), state: head.state })
        close(concat([head.output, tail.output], 1), par, 8)
      })

      it('sits in a residual block of a sequence model that trains under grad', () => {
        const model = new SequenceModel(dim, 1, (d, k) => sequenceMixers.make({ kind: name, stateSize: 4 })(d, k), {
          inputSize: 7,
          key: key(4),
        })
        expect(model.forward([[1, 2, 3, 4]]).shape).toEqual([1, 4, 7])
        const w = (model.params['embed'] as Record<string, Value>)['weight'] as Value
        const f = (e: Value): Value => model.apply({ ...model.params, embed: { weight: e } }, [[1, 2, 3]])
        expect(checkPrimitive(f, [w], { rtol: 1e-4, atol: 1e-6 }).failures).toEqual([])
      })
    })
  }

  it('lists the mixers of the paper family and a baseline', () => {
    expect(sequenceMixers.names).toEqual(
      expect.arrayContaining([
        'mamba',
        'mamba2',
        's4d',
        's4',
        's5',
        'lru',
        'linear-attention',
        'retnet',
        'gla',
        'deltanet',
        'rwkv',
        'mingru',
        'minlstm',
        'hyena',
        'attention',
      ]),
    )
    expect(sequenceMixers.where((p) => !p.constantState)).toEqual(['hyena', 'attention'])
  })
})

describe('forms of one layer', () => {
  const x = n(11, [2, 10, 8])
  const forms = ['parallel', 'recurrent', 'chunked'] as const

  it('RetNet: the parallel, recurrent and chunkwise forms agree', () => {
    const out = forms.map((form) => new RetNetMixer(8, { heads: 2, form, chunk: 3, key: key(12) }).forward(x))
    close((out[1] as { output: Value }).output, (out[0] as { output: Value }).output, 10)
    close((out[2] as { output: Value }).output, (out[0] as { output: Value }).output, 10)
    close(
      (out[2] as { state: MixerState }).state['s'] as Value,
      (out[1] as { state: MixerState }).state['s'] as Value,
      10,
    )
  })

  it('linear attention and gated linear attention: the three forms agree', () => {
    for (const make of [
      (form: (typeof forms)[number]) =>
        new LinearAttentionMixer(8, { form, chunk: 4, featureMap: 'elu', key: key(13) }),
      (form: (typeof forms)[number]) => new GatedLinearAttentionMixer(8, { form, chunk: 4, key: key(14) }),
    ]) {
      const ref = make('recurrent').forward(x).output
      close(make('parallel').forward(x).output, ref, 10)
      close(make('chunked').forward(x).output, ref, 10)
    }
  })

  it('minGRU and minLSTM: the parallel scan equals the recurrence', () => {
    for (const M of [MinGRU, MinLSTM]) {
      const a = new M(8, { form: 'scan', key: key(15) }).forward(x).output
      const b = new M(8, { form: 'recurrent', key: key(15) }).forward(x).output
      close(a, b, 10)
    }
  })

  it('a decay that is never forgotten makes retention a plain causal sum (per head gamma -> 1 is the long-memory head)', () => {
    const m = new RetNetMixer(8, { heads: 2, key: key(16) })
    expect(m.gamma[0]).toBeCloseTo(1 - 2 ** -5, 15)
    expect(m.gamma[1]).toBeCloseTo(1 - 2 ** -6, 15)
  })

  it('DeltaNet overwrites: writing the same key twice replaces the value instead of adding', () => {
    const m = new DeltaNetMixer(4, { heads: 1, key: key(17) })
    const x1 = n(18, [1, 1, 4])
    const once = m.forward(x1).state['s'] as Value // [1, 1, dv, dk]
    const twice = m.forward(concat([x1, x1], 1)).state['s'] as Value
    // The second write of the same key moves the memory toward the value (beta < 1), it does not double it.
    const a = toFlat(once)
    const b = toFlat(twice)
    const ratio = b.map((v, i) => v / (a[i] ?? 1))
    expect(Math.max(...ratio.map((r) => Math.abs(r)))).toBeLessThan(2)
  })

  it('RWKV, Hyena and DeltaNet pass the gradient check', () => {
    for (const m of [
      new RWKVMixer(3, { key: key(19) }),
      new Hyena(3, { key: key(20) }),
      new DeltaNetMixer(4, { heads: 2, key: key(21) }),
    ]) {
      const d = m.dim
      expect(
        checkPrimitive((v) => m.apply(m.params, v).output, [n(22, [1, 3, d])], { rtol: 1e-4, atol: 1e-6 }).failures,
      ).toEqual([])
    }
  })

  it('a block adds the residual and keeps the shapes', () => {
    const b = new MixerBlock(new RetNetMixer(8, { key: key(23) }), { ffn: 'gated', key: key(24) })
    expect(b.forward(x).output.shape).toEqual([2, 10, 8])
  })
})

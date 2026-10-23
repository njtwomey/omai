import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { foldIn, key, normal } from '../../random/index.ts'
import { concat, sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import { tapContext } from '../../trace/index.ts'
import * as NN from '../index.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value, digits = 11): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}

describe('grouped-query attention', () => {
  it('decoding step by step with a KV cache equals one causal pass', () => {
    const gqa = new NN.GroupedQueryAttention(8, 4, { numKvHeads: 2, key: key(0) })
    const x = n(1, [2, 5, 8])
    const full = gqa.forward({ x }).output
    let cache = NN.emptyKVCache(2, 2, 2)
    const steps: Value[] = []
    for (let t = 0; t < 5; t++) {
      const r = gqa.forward({ x: sliceAxis(x, 1, t, t + 1), cache })
      cache = r.cache
      steps.push(r.output)
    }
    close(concat(steps, 1), full)
    expect(cache.keys.shape).toEqual([2, 2, 5, 2])
  })

  it('multi-query attention, sliding windows, cross-attention and taps', () => {
    const mqa = new NN.GroupedQueryAttention(8, 4, { numKvHeads: 1, window: 2, rope: false, key: key(2) })
    const taps = tapContext()
    const r = mqa.forward({ x: n(3, [1, 4, 8]) }, { taps, path: 'attn' })
    expect(r.output.shape).toEqual([1, 4, 8])
    const w = taps.entries.get('attn.weights')?.value as Value
    expect(w.shape).toEqual([1, 4, 4, 4])
    expect(toFlat(w)[2]).toBe(0)
    const cross = new NN.GroupedQueryAttention(8, 2, { key: key(4) }).forward({
      x: n(5, [1, 3, 8]),
      context: n(6, [1, 7, 8]),
    })
    expect(cross.cache.keys.shape).toEqual([1, 2, 7, 4])
    expect(() => new NN.GroupedQueryAttention(8, 4, { numKvHeads: 3 })).toThrow()
  })
})

describe('multi-head attention module', () => {
  it('taps its weights, accepts unbatched input and needs a key for dropout in training', () => {
    const mha = new NN.MultiheadAttention(4, 2, { dropout: 0.5, key: key(0) })
    const x = n(1, [3, 4])
    const taps = tapContext()
    const r = mha.eval().forward(x, { taps })
    expect(r.output.shape).toEqual([3, 4])
    expect(r.weights?.shape).toEqual([3, 3])
    expect(taps.entries.has('weights')).toBe(true)
    expect(() => mha.train().forward(x)).toThrow()
    expect(mha.train().forward(x, { key: key(2) }).output.shape).toEqual([3, 4])
  })
})

describe('transformer modules', () => {
  it('state dicts follow PyTorch names', () => {
    const t = new NN.Transformer(8, 2, { numEncoderLayers: 1, numDecoderLayers: 1, dimFeedforward: 16, key: key(0) })
    const names = Object.keys(t.stateDict())
    for (const k of [
      'encoder.layers.0.self_attn.in_proj_weight',
      'encoder.layers.0.self_attn.out_proj.weight',
      'encoder.layers.0.linear1.weight',
      'encoder.norm.weight',
      'decoder.layers.0.multihead_attn.in_proj_bias',
      'decoder.layers.0.norm3.bias',
    ])
      expect(names).toContain(k)
    expect(t.parameterCount()).toBeGreaterThan(0)
  })

  it('an encoder layer passes the gradient conformance check', () => {
    const layer = new NN.TransformerEncoderLayer(4, 2, { dimFeedforward: 6, dropout: 0, batchFirst: true, key: key(1) })
    const p = layer.params
    const f = (x: Value): Value => layer.apply(p, x, { train: false })
    expect(checkPrimitive(f, [n(2, [1, 3, 4])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
  })

  it('pre-norm stacks with a final norm and causal masks', () => {
    const enc = new NN.TransformerEncoder(
      (i) => new NN.TransformerEncoderLayer(4, 1, { dimFeedforward: 8, normFirst: true, key: foldIn(key(3), i) }),
      2,
      { norm: new NN.LayerNorm(4) },
    )
    const x = n(4, [5, 1, 4])
    const a = enc.eval().forward({ src: x, isCausal: true })
    const b = enc.forward({ src: x, mask: NN.squareSubsequentMask(5) })
    close(a, b)
    // A causal encoder's first output depends on the first input only.
    const y = enc.forward({ src: concat([sliceAxis(x, 0, 0, 1), n(5, [4, 1, 4])], 0), isCausal: true })
    close(sliceAxis(y, 0, 0, 1), sliceAxis(a, 0, 0, 1))
  })

  it('positional modules', () => {
    const x = n(6, [2, 3, 4])
    expect(new NN.SinusoidalPositionalEncoding(4).forward(x).shape).toEqual([2, 3, 4])
    expect(new NN.LearnedPositionalEmbedding(8, 4, { key: key(7) }).forward(x).shape).toEqual([2, 3, 4])
    expect(new NN.RotaryEmbedding(4, { interleaved: true }).forward(x).shape).toEqual([2, 3, 4])
  })
})

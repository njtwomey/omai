import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { concat, sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import type { KVCache } from '../module/attention.ts'
import {
  encoderDecoder,
  FeedForward,
  GatedFeedForward,
  TransformerBlock,
  TransformerLM,
  TransformerStack,
} from './transformer.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value, digits = 11): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}

describe('transformer blocks', () => {
  it('feed-forward blocks keep the width and pass the gradient check', () => {
    const ff = new FeedForward(4, 6, { key: key(0) })
    const gff = new GatedFeedForward(4, 6, { key: key(1) })
    const x = n(2, [2, 4])
    expect(ff.forward(x).shape).toEqual([2, 4])
    expect(Object.keys(gff.stateDict())).toEqual(['down_proj.weight', 'gate_proj.weight', 'up_proj.weight'])
    expect(checkPrimitive((v) => gff.apply(gff.params, v), [x], { rtol: 1e-5 }).failures).toEqual([])
  })

  it('a stack decodes step by step with caches exactly as in one pass', () => {
    const stack = new TransformerStack(2, (i) => new TransformerBlock(8, 4, { numKvHeads: 2, key: key(10 + i) }))
    const x = n(3, [1, 5, 8])
    const full = stack.forward({ x }).output
    let caches: readonly KVCache[] = stack.emptyCaches(1)
    const outs: Value[] = []
    for (let t = 0; t < 5; t++) {
      const r = stack.forward({ x: sliceAxis(x, 1, t, t + 1), caches })
      caches = r.caches
      outs.push(r.output)
    }
    close(concat(outs, 1), full)
  })

  it('post-norm, layer norm, plain feed-forward and sliding windows', () => {
    const b = new TransformerBlock(8, 2, { preNorm: false, norm: 'layer', gated: false, window: 2, key: key(4) })
    expect(b.forward(n(5, [2, 3, 8])).output.shape).toEqual([2, 3, 8])
    expect(() => new TransformerBlock(8, 2, { crossAttention: true }).forward(n(6, [1, 2, 8]))).toThrow()
  })

  it('encoder-decoder and language model', () => {
    const ed = encoderDecoder(8, 2, { encoderLayers: 1, decoderLayers: 2, key: key(7) })
    const r = ed.forward({ source: n(8, [2, 4, 8]), target: n(9, [2, 3, 8]) })
    expect(r.output.shape).toEqual([2, 3, 8])
    expect(r.caches.length).toBe(2)
    const lm = new TransformerLM(11, 8, 2, 2, { key: key(10) })
    const tokens = [[1, 4, 2, 7]]
    const logits = lm.forward(tokens).output
    expect(logits.shape).toEqual([1, 4, 11])
    // Incremental logits for the last token equal the full pass.
    const first = lm.forward({ tokens: [[1, 4, 2]], caches: lm.emptyCaches(1) })
    const last = lm.forward({ tokens: [[7]], caches: first.caches }).output
    close(last, sliceAxis(logits, 1, 3, 4))
    const untied = new TransformerLM(11, 8, 1, 2, { tieEmbeddings: false, key: key(11) })
    expect(Object.keys(untied.stateDict())).toContain('lm_head.weight')
  })
})

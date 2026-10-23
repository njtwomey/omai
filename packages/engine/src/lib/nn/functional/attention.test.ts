import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { matmul, softmax, sum, swapaxes, tanh, tensor, toArray, toFlat, type Value } from '../../tensor/index.ts'
import {
  attention,
  causalMask,
  linearAttention,
  localMask,
  maskedSoftmax,
  mergeHeads,
  multiHeadAttention,
  scaledDotProductAttention,
  splitHeads,
} from './attention.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value, digits = 12): void => {
  const x = toFlat(a)
  const y = toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('attention', () => {
  it('soft-capping is cap tanh(scores / cap) before the softmax', () => {
    const q = n(0, [3, 4])
    const k = n(1, [5, 4])
    const v = n(2, [5, 2])
    const scores = toFlat(matmul(q, swapaxes(k, 0, 1))).map((s) => 2 * Math.tanh(s / 2 / 2))
    const w = softmax(tensor(Array.from({ length: 3 }, (_, i) => scores.slice(i * 5, i * 5 + 5))), -1)
    close(scaledDotProductAttention(q, k, v, { softCap: 2 }), matmul(w, v))
    expect(toFlat(tanh(0)).length).toBe(1)
  })

  it('a sliding window attends to the most recent positions only', () => {
    const q = n(3, [6, 4])
    const r = attention(q, q, q, { window: 2 })
    const w = toArray(r.weights) as number[][]
    for (const [i, row] of w.entries()) {
      for (const [j, x] of row.entries()) if (j > i || i - j >= 2) expect(x).toBe(0)
    }
    expect(toArray(localMask(4, 4, 1))).toEqual([
      [1, 1, 0, 0],
      [1, 1, 1, 0],
      [0, 1, 1, 1],
      [0, 0, 1, 1],
    ])
    expect(toArray(causalMask(2, 3))).toEqual([
      [1, 0, 0],
      [1, 1, 0],
    ])
  })

  it('a fully masked row gives zeros, not NaN', () => {
    const y = maskedSoftmax([[1, 2]], tensor([[false, false]], { dtype: 'bool' }))
    expect(toArray(y)).toEqual([[0, 0]])
  })

  it('an additive bias shifts the scores', () => {
    const q = n(4, [2, 3])
    const bias = n(5, [2, 2])
    const plain = attention(q, q, q, { mask: bias }).output
    close(attention(q, q, q, { bias }).output, plain)
  })

  it('linear attention matches its quadratic form', () => {
    const q = n(6, [4, 3])
    const k = n(7, [4, 3])
    const v = n(8, [4, 2])
    const phi = (x: number): number => (x > 0 ? x + 1 : Math.exp(x))
    const fq = (toArray(q) as number[][]).map((r) => r.map(phi))
    const fk = (toArray(k) as number[][]).map((r) => r.map(phi))
    const vv = toArray(v) as number[][]
    for (const causal of [false, true]) {
      const want = fq.map((qi, i) => {
        const ws = fk.map((kj, j) => (causal && j > i ? 0 : qi.reduce((s, x, d) => s + x * (kj[d] ?? 0), 0)))
        const z = ws.reduce((a, b) => a + b, 0) + 1e-6
        return [0, 1].map((c) => ws.reduce((s, w, j) => s + w * (vv[j]?.[c] ?? 0), 0) / z)
      })
      close(linearAttention(q, k, v, { causal }), tensor(want), 10)
    }
  })

  it('heads split and merge back', () => {
    const x = n(9, [2, 3, 6])
    close(mergeHeads(splitHeads(x, 3)), x)
    expect(() => splitHeads(x, 4)).toThrow()
  })

  it('attention and multi-head attention pass the gradient conformance check', () => {
    ok(
      (q, k, v) => scaledDotProductAttention(q, k, v, { causal: true, softCap: 3 }),
      [n(10, [1, 3, 2]), n(11, [1, 3, 2]), n(12, [1, 3, 2])],
    )
    ok((q, k, v) => linearAttention(q, k, v, { causal: true }), [n(13, [3, 2]), n(14, [3, 2]), n(15, [3, 2])])
    const e = 4
    ok(
      (x, w, b, o) =>
        multiHeadAttention(x, x, x, { inProjWeight: w, inProjBias: b, outProjWeight: o }, { numHeads: 2, causal: true })
          .output,
      [n(16, [1, 3, e]), n(17, [3 * e, e]), n(18, [3 * e]), n(19, [e, e])],
    )
  })

  it('needs a key for dropout in training', () => {
    const q = n(20, [2, 2])
    expect(() => attention(q, q, q, { dropout: 0.5 })).toThrow()
    expect(sum(attention(q, q, q, { dropout: 0.5, training: false }).weights).shape).toEqual([])
  })
})

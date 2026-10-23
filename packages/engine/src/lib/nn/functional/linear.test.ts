import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { sum, toArray, type Value } from '../../tensor/index.ts'
import { bilinear, embedding, embeddingBag, linear } from './linear.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const ok = (f: (...xs: Value[]) => Value, xs: Value[]): void => {
  expect(checkPrimitive(f, xs, { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
}

describe('linear and embedding gradients', () => {
  it('linear and bilinear maps', () => {
    ok((x, w, b) => linear(x, w, b), [n(0, [2, 3]), n(1, [4, 3]), n(2, [4])])
    ok((a, b, w, c) => bilinear(a, b, w, c), [n(3, [2, 3]), n(4, [2, 2]), n(5, [4, 3, 2]), n(6, [4])])
  })
  it('embeddings and bags', () => {
    ok(
      (w) =>
        embedding(
          [
            [1, 2],
            [0, 1],
          ],
          w,
        ),
      [n(7, [3, 2])],
    )
    for (const mode of ['sum', 'mean', 'max'] as const) {
      ok((w) => embeddingBag([0, 2, 1, 1, 2], w, { offsets: [0, 2], mode }), [n(9, [3, 2])])
    }
    ok(
      (w, s) => embeddingBag([0, 2, 1], w, { offsets: [0, 1], mode: 'sum', perSampleWeights: s }),
      [n(10, [3, 2]), n(11, [3])],
    )
  })
})

describe('embedding behaviour', () => {
  it('the padding row gets no gradient', () => {
    const g = grad((w: Value) => sum(embedding([0, 1, 1], w, { paddingIdx: 1 })))(n(0, [2, 2]))
    expect(toArray(g)).toEqual([
      [1, 1],
      [0, 0],
    ])
  })
  it('rescales rows longer than maxNorm (as a constant factor, like PyTorch renormalising under no_grad)', () => {
    const rows = toArray(
      embedding(
        [0, 1],
        [
          [3, 4],
          [0.3, 0.4],
        ],
        { maxNorm: 1 },
      ),
    ) as number[][]
    expect(rows.map((r) => Math.round(Math.hypot(...r) * 1e6) / 1e6)).toEqual([1, 0.5])
  })
  it('empty bags are zeros and out-of-range indices are rejected', () => {
    expect(
      toArray(
        embeddingBag(
          [1],
          [
            [1, 1],
            [2, 2],
          ],
          { offsets: [0, 0], mode: 'mean' },
        ),
      ),
    ).toEqual([
      [0, 0],
      [2, 2],
    ])
    expect(() => embedding([3], [[1], [2]])).toThrow()
    expect(() => embeddingBag([1], [[1]], {})).toThrow()
  })
})

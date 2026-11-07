import { describe, expect, it } from 'vitest'
import { tensor, treeFlatten } from './index.ts'

describe('treeFlatten', () => {
  it('flattens tensors in arrays and records in key order', () => {
    const { leaves } = treeFlatten({ b: tensor([1]), a: [tensor([2, 3])] })
    expect(leaves.map((l) => l.shape)).toEqual([[2], [1]])
  })

  it('names the path of a leaf that is not a tensor, and says to wrap it', () => {
    expect(() => treeFlatten([tensor([1]), { w: [1, 2, 3] }])).toThrow(
      /got number at \[1\]\.w\[0\]: wrap numbers and arrays of numbers with tensor/,
    )
    expect(() => treeFlatten(3)).toThrow(/got number at the root: wrap/)
  })
})

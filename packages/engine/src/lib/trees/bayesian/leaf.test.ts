// The leaf models: what they refuse. A label outside 0..K-1 has no class to count, and a one-hot row of zeros would drop the row
// from the counts, the evidence and the minimum leaf size without a word.
import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { dirichletLeaf } from './leaf.ts'
import { bayesData } from './tree.ts'

describe('dirichletLeaf labels', () => {
  const xs = [[0], [1], [2], [3]]

  it('counts the rows of labels 0..K-1', () => {
    expect(toFlat(dirichletLeaf(3).statistics([0, 1, 2, 2]))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
  })

  it('refuses 1-based labels, naming the label and the classes', () => {
    expect(() => bayesData(xs, [1, 2, 3, 2], dirichletLeaf(3))).toThrow(/label 3.*0 to 2/)
  })

  it('refuses a negative, a fractional and a missing label', () => {
    for (const bad of [-1, 0.5, Number.NaN])
      expect(() => bayesData(xs, [0, 1, 0, bad], dirichletLeaf(2)), String(bad)).toThrow(/dirichletLeaf: label/)
  })
})

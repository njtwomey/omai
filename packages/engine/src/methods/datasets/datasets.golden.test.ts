import { describe, expect, it } from 'vitest'
import { bayesPredict, generate } from '../../lib/data/index.ts'
import { svdvals } from '../../lib/numerics/linalg/index.ts'
import { key } from '../../lib/random/index.ts'
import { type Tensor, tensor } from '../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import {
  circlePoints,
  gaussianQuantileLabels,
  hastie10_2,
  makeLowRankMatrix,
  moonPoints,
  sCurveEmbedding,
  swissRollEmbedding,
} from './index.ts'

describe('classification generators against sklearn.datasets', () => {
  for (const c of loadFixture('methods/datasets/classification').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'moonPoints' || c.fn === 'circlePoints') {
        const pts =
          c.fn === 'moonPoints' ? moonPoints(p['n'] as number) : circlePoints(p['n'] as number, p['factor'] as number)
        const got = p['part'] === 'x' ? pts.x : tensor(pts.y, { dtype: 'int32' })
        expectMatches(got, c.output, 1e-12, 1e-15, c.name)
        return
      }
      const x = decode(c.inputs[0]) as Tensor
      if (c.fn === 'gaussianQuantileLabels') {
        expectMatches(tensor(gaussianQuantileLabels(x, p['k'] as number), { dtype: 'int32' }), c.output, 0, 0, c.name)
        return
      }
      const { truth } = generate(hastie10_2, { key: key(0) })
      expectMatches(bayesPredict(truth, x), c.output, 0, 0, c.name)
    })
  }
})

describe('manifold embeddings against sklearn.datasets', () => {
  for (const c of loadFixture('methods/datasets/manifolds').cases) {
    it(c.name, () => {
      const [t, h] = [decode(c.inputs[0]) as Tensor, decode(c.inputs[1]) as Tensor]
      const f = c.fn === 'sCurveEmbedding' ? sCurveEmbedding : swissRollEmbedding
      expectMatches(f(t, h), c.output, 1e-12, 1e-12, c.name)
    })
  }
})

describe('matrix generators against sklearn.datasets', () => {
  for (const c of loadFixture('methods/datasets/matrices').cases) {
    it(c.name, () => {
      const p = c.params
      const { x, singularValues } = makeLowRankMatrix(key(3), {
        nSamples: p['n'] as number,
        nFeatures: p['d'] as number,
        effectiveRank: p['rank'] as number,
        tailStrength: p['tail'] as number,
      })
      expectMatches(svdvals(x), c.output, 1e-10, 1e-12, c.name)
      expectMatches(tensor(singularValues as number[]), c.output, 1e-10, 1e-12, c.name)
    })
  }
  it('has cases', () => {
    expect(loadFixture('methods/datasets/matrices').cases.length).toBe(3)
  })
})

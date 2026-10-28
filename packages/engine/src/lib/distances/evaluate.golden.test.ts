import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Value } from '../tensor/index.ts'
import {
  type DistanceName,
  distance,
  pairwiseDistances,
  pairwiseSimilarities,
  type SimilarityName,
  type SimilarityOptions,
  similarity,
} from './index.ts'

describe('named distances and similarities against scipy.spatial.distance and scikit-learn', () => {
  for (const c of loadFixture('distances/evaluate').cases) {
    it(c.name, () => {
      const [a, b] = c.inputs.map(decode) as [Value, Value]
      const { metric: name, ...rest } = c.params
      // Tensor parameters (weights, variances, covariances) arrive encoded; scalars and names pass through.
      const options = Object.fromEntries(Object.entries(rest).map(([k, v]) => [k, decode(v)])) as SimilarityOptions
      const got =
        c.fn === 'pairwise'
          ? pairwiseDistances(a, b, name as DistanceName, options)
          : c.fn === 'elementwise'
            ? distance(a, b, name as DistanceName, options)
            : c.fn === 'pairwiseSimilarity'
              ? pairwiseSimilarities(a, b, name as SimilarityName, options)
              : similarity(a, b, name as SimilarityName, options)
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

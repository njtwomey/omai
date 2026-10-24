import { describe, expect, it } from 'vitest'
import * as M from './index.ts'

describe('beyond-accuracy measures', () => {
  const recs = [
    [0, 1, 2],
    [1, 2, 3],
    [1, 4, 5],
  ]

  it('measures coverage, novelty and serendipity', () => {
    expect(M.catalogCoverage(recs, 10).estimate.value).toBe(0.6)
    // Uniform popularity gives log2(n) bits for every item.
    expect(
      M.novelty(
        recs,
        Array.from({ length: 8 }, () => 5),
      ).estimate.value,
    ).toBeCloseTo(3, 14)
    expect(M.serendipity(recs, [[0, 1], [3], []], [[1], [], []]).estimate.value).toBeCloseTo((1 / 3 + 1 / 3) / 3, 15)
    expect(M.catalogCoverage.spec.inputs).toBe('sets')
  })

  it('measures diversity of item features', () => {
    const features = [
      [1, 0],
      [0, 1],
      [1, 1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ]
    const ild = M.intraListDiversity([[0, 3]], features)
    expect(ild.estimate.value).toBeCloseTo(0, 15)
    expect(M.intraListDiversity([[0, 5]], features).estimate.value).toBe(2)
    expect(M.intraListDiversity([[0, 1]], features, { distance: 'euclidean' }).estimate.value).toBeCloseTo(
      Math.SQRT2,
      15,
    )
  })

  it('computes the Gini of exposure', () => {
    expect(M.exposureGini([[0], [0], [0]], 3).estimate.value).toBeCloseTo(2 / 3, 15)
    expect(M.exposureGini([[0, 1, 2]], 3).estimate.value).toBe(0)
    // Position weighting makes the first item more exposed.
    expect(M.exposureGini([[0, 1, 2]], 3, { position: 'log' }).estimate.value).toBeGreaterThan(0)
    expect(M.exposureGini.spec.direction).toBe('lower')
  })
})

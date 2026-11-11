// Relabelling the components of a fixed point: the permutations, the move of every axis of a site that is indexed by the
// components, and the canonical form (the components sorted by one site's mean).
import { describe, expect, it } from 'vitest'
import { canonicalSites, moveAxis, orderingSite, permutations, relabelRecord, sortingPermutation } from './_relabel.ts'
import type { SiteRecord } from './_site-mixture.ts'

const vector = (mean: number[], variance: number[]): SiteRecord => ({
  kind: 'gamma',
  shape: [mean.length],
  mean,
  variance,
})

describe('permutations and moves', () => {
  it('lists the k! permutations, the identity first, each once', () => {
    const p = permutations(3)
    expect(p).toHaveLength(6)
    expect(p[0]).toEqual([0, 1, 2])
    expect(new Set(p.map((q) => q.join(''))).size).toBe(6)
  })

  it('moves entry q to perm[q]', () => {
    expect(moveAxis(['a', 'b', 'c'], [2, 0, 1])).toEqual(['b', 'c', 'a'])
  })
})

describe('relabelling a record', () => {
  it('moves a vector of means and variances, and both axes of a covariance', () => {
    const r: SiteRecord = {
      kind: 'multivariateNormal',
      shape: [3],
      mean: [1, 2, 3],
      variance: [4, 5, 6],
      covariance: [4, 0.1, 0.2, 0.1, 5, 0.3, 0.2, 0.3, 6],
    }
    const out = relabelRecord(r, [2, 0, 1], 3)
    expect(out.mean).toEqual([2, 3, 1])
    expect(out.variance).toEqual([5, 6, 4])
    // entry (a, b) of the covariance goes to (perm a, perm b)
    expect(out.covariance?.[0 * 3 + 1]).toBeCloseTo(0.3)
    expect(out.covariance?.[2 * 3 + 0]).toBeCloseTo(0.1)
  })

  it('moves both axes of a matrix of components and leaves a site of another size alone', () => {
    const matrix: SiteRecord = { kind: 'beta', shape: [2, 2], mean: [0.9, 0.1, 0.2, 0.8], variance: [1, 2, 3, 4] }
    expect(relabelRecord(matrix, [1, 0], 2).mean).toEqual([0.8, 0.2, 0.1, 0.9])
    const other = vector([1, 2, 3, 4], [1, 1, 1, 1])
    expect(relabelRecord(other, [2, 0, 1], 3)).toEqual(other)
  })

  it('moves the probabilities of a categorical site and recomputes the index mean it reports', () => {
    const r: SiteRecord = { kind: 'categorical', shape: [], mean: [0.3], variance: [0.2], probs: [0.7, 0.3, 0] }
    const out = relabelRecord(r, [1, 2, 0], 3)
    expect(out.probs).toEqual([0, 0.7, 0.3])
    expect(out.mean[0]).toBeCloseTo(0.7 + 0.6)
  })
})

describe('the canonical form', () => {
  it('sorts the components by the key site and relabels every site the same way', () => {
    const rate = vector([9, 3, 22], [1, 1, 1])
    const z: SiteRecord = { kind: 'categorical', shape: [], mean: [1], variance: [0], probs: [0, 1, 0] }
    const out = canonicalSites({ rate, z }, 'rate', 3)
    expect(out['rate']?.mean).toEqual([3, 9, 22])
    // the step was in component 1, which was the middle value 3, and now sits at index 0
    expect(out['z']?.probs).toEqual([1, 0, 0])
    expect(sortingPermutation(rate, 3)).toEqual([1, 0, 2])
  })

  it('the key site is the first continuous vector of the number of components', () => {
    const sites = {
      z: { kind: 'categorical', shape: [], mean: [0], variance: [0], probs: [1, 0, 0] } as SiteRecord,
      w: vector([1, 2], [1, 1]),
      rate: vector([1, 2, 3], [1, 1, 1]),
    }
    expect(orderingSite(sites, 3)).toBe('rate')
    expect(orderingSite(sites, 4)).toBeUndefined()
  })
})

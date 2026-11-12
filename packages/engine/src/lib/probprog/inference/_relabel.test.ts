// Relabelling the components of a fixed point: the permutations, the move of every axis of a site that is indexed by the
// components, and the canonical form (the components sorted by one site's mean).
import { describe, expect, it } from 'vitest'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import {
  canonicalSites,
  componentAxesOf,
  moveAxis,
  orderingSite,
  permutations,
  relabelRecord,
  sortingPermutation,
} from './_relabel.ts'
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

describe('which axes are the components', () => {
  // two components over two-dimensional data: the number of components is also the number of dimensions, so a site with a
  // component axis and a data axis is a [2, 2] one that only its first axis moves
  const spec = defineModel('two-by-two', (m) => {
    const row = m.range('row')
    const k = m.range('component', { size: 2 })
    const dim = m.range('dim', { size: 2 })
    const y = m.data('y', { dims: [row, dim], role: 'response' })
    const w = m.latent('w', d.dirichlet([1, 1]))
    const mu = m.latent('mu', d.normal(0, 3), { dims: [k, dim] })
    const centre = m.latent('centre', d.normal(0, 1), { dims: [dim] })
    const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] })
    m.observe('y', d.normal(mu.at(z).add(centre), 1), y)
  })

  it('reads the component axes from how the discrete latent is used: the looked-up dimension and the class probabilities', () => {
    const axesOf = componentAxesOf(spec.spec)
    expect(axesOf?.('mu')).toEqual([0])
    expect(axesOf?.('centre')).toEqual([])
    expect(axesOf?.('w')).toEqual([0])
    expect(axesOf?.('z')).toEqual([])
    // a name that is not a node of the model is not known
    expect(axesOf?.('nowhere')).toBeUndefined()
  })

  it('reads the component dimension from a pick lookup, the last dimension of its first operand', () => {
    // a mixture of regressions with a slope per class, the prediction of a row's class picked from [row, component]
    const regression = defineModel('per-class-slope', (m) => {
      const row = m.range('row')
      const k = m.range('component', { size: 2 })
      const x = m.data('x', { dims: [row] })
      const y = m.data('y', { dims: [row] })
      const w = m.latent('w', d.dirichlet([1, 1]))
      const slope = m.latent('slope', d.normal(0, 3), { dims: [k] })
      const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] })
      m.observe('y', d.normal(ops.pick(ops.mul(x, slope), z), 1), y)
    })
    const axesOf = componentAxesOf(regression.spec)
    expect(axesOf?.('slope')).toEqual([0])
    expect(axesOf?.('w')).toEqual([0])
    expect(axesOf?.('z')).toEqual([])
  })

  it('does not call a site a non-component one when the model has no lookup it can read: unknown, every axis of length k', () => {
    // a mixture written with where has no component dimension to find: the class probabilities are the categories, and any
    // other site is unknown (the old reading), not "no component axes" (which would leave a vector of rates unsorted)
    const gated = defineModel('where-lookup', (m) => {
      const row = m.range('row')
      const y = m.data('y', { dims: [row] })
      const w = m.latent('w', d.dirichlet([1, 1]))
      const lo = m.latent('lo', d.normal(0, 3))
      const hi = m.latent('hi', d.normal(0, 3))
      const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] })
      m.observe('y', d.normal(ops.where(ops.eq(z, 0), lo, hi), 1), y)
    })
    const axesOf = componentAxesOf(gated.spec)
    expect(axesOf?.('w')).toEqual([0])
    expect(axesOf?.('lo')).toBeUndefined()
    expect(axesOf?.('nowhere')).toBeUndefined()
  })

  it('has none for a model with no discrete latent', () => {
    const plain = defineModel('plain', (m) => {
      const y = m.data('y', { dims: [m.range('row')] })
      m.observe('y', d.normal(m.latent('mu', d.normal(0, 1)), 1), y)
    })
    expect(componentAxesOf(plain.spec)).toBeUndefined()
  })

  it('moves only the component axis of a [2, 2] site that is per component and per dimension', () => {
    // rows are components, columns are dimensions: swapping the components swaps the rows and not the columns
    const mu: SiteRecord = { kind: 'normal', shape: [2, 2], mean: [1, 2, 3, 4], variance: [5, 6, 7, 8] }
    expect(relabelRecord(mu, [1, 0], 2, [0]).mean).toEqual([3, 4, 1, 2])
    expect(relabelRecord(mu, [1, 0], 2, [0]).variance).toEqual([7, 8, 5, 6])
    // without the axes every axis of length k moves (the old reading): rows and columns
    expect(relabelRecord(mu, [1, 0], 2).mean).toEqual([4, 3, 2, 1])
  })

  it('leaves alone a vector of the length of the components that is not per component, with its covariance', () => {
    const centre: SiteRecord = {
      kind: 'multivariateNormal',
      shape: [2],
      mean: [1, 2],
      variance: [3, 4],
      covariance: [3, 0.5, 0.5, 4],
    }
    expect(relabelRecord(centre, [1, 0], 2, [])).toEqual(centre)
    expect(relabelRecord(centre, [1, 0], 2, [0]).mean).toEqual([2, 1])
  })

  it('sorts by a site that is per component and not by a vector of the right length that is not, and relabels the rest correctly', () => {
    const centre = vector([9, 1], [1, 1])
    const mu: SiteRecord = { kind: 'normal', shape: [2, 2], mean: [5, 6, 1, 2], variance: [1, 1, 1, 1] }
    const spread = vector([3, 2], [1, 1])
    const sites = { centre, mu, spread }
    const axesOf = (site: string): readonly number[] | undefined => (site === 'centre' ? [] : [0])
    // `centre` has the length of the components and comes first, but it is over the dimensions
    expect(orderingSite(sites, 2)).toBe('centre')
    expect(orderingSite(sites, 2, axesOf)).toBe('spread')
    const out = canonicalSites(sites, 'spread', 2, axesOf)
    expect(out['spread']?.mean).toEqual([2, 3])
    expect(out['mu']?.mean).toEqual([1, 2, 5, 6])
    expect(out['centre']?.mean).toEqual([9, 1])
  })

  it('does not move the probabilities of a categorical site over another number of categories', () => {
    const z: SiteRecord = { kind: 'categorical', shape: [], mean: [1], variance: [0.2], probs: [0.2, 0.3, 0.5] }
    expect(relabelRecord(z, [1, 0], 2)).toEqual(z)
  })
})

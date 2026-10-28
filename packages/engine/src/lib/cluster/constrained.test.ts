import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { metricSeries } from '../trace/index.ts'
import { brokenConstraints, constrainedAssignment, copKMeans } from './constrained.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 60, centers: 3, clusterStd: 0.5 } })

describe('COP-k-means', () => {
  it('honours must-link and cannot-link constraints', () => {
    const f = copKMeans.with({ nClusters: 3, mustLink: [[0, 1]], cannotLink: [[2, 3]] }).fit(data, { key: key(7) })
    expect(f.params.labels[0]).toBe(f.params.labels[1])
    expect(f.params.labels[2]).not.toBe(f.params.labels[3])
    expect(f.params.brokenMustLink).toEqual([])
    expect(f.params.brokenCannotLink).toEqual([])
  })

  it('relaxes infeasible constraints and reports them, or fails when strict', () => {
    // Three rows pairwise cannot-linked can't fit two clusters: one pair has to share.
    const cannotLink: [number, number][] = [
      [0, 1],
      [1, 2],
      [0, 2],
    ]
    const f = copKMeans.with({ nClusters: 2, cannotLink }).fit(data, { key: key(7), logs: 'all' })
    expect(f.params.brokenCannotLink.length).toBe(1)
    expect(metricSeries(f.run, 'solver/relaxed').values.at(-1)).toBeGreaterThanOrEqual(1)
    expect(() => copKMeans.with({ nClusters: 2, cannotLink, strict: true }).fit(data, { key: key(7) })).toThrow(
      /no cluster satisfying/,
    )
  })

  it('assigns around a constraint and lists broken pairs', () => {
    expect(constrainedAssignment([[0], [0.1]], [[0], [5]], { cannotLink: [[0, 1]] }).labels).toEqual([0, 1])
    expect(brokenConstraints([0, 0, 1], { mustLink: [[0, 2]], cannotLink: [[0, 1]] })).toEqual({
      mustLink: [0],
      cannotLink: [0],
    })
  })
})

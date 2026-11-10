// The probit page's compute: three runs, each read as a posterior at a snapshot, and the comparison with a grid.
import { gridTruth, probitRegression } from 'omai-engine/probprog/models'
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../shared/smoke.ts'
import { BOUND, COORDINATES, compare, probitRuns } from './compute.ts'

describe('the probit runs', () => {
  const specs = probitRuns()
  const kept = specs.map((s) => ({
    name: s.label,
    kept: snapshotOf(s, { numWarmup: 30, numDraws: 30 }, 200),
  }))
  // A coarse grid: enough to compare with, quick to make.
  const truth = gridTruth(probitRegression.program(), { points: 9, maxRefinements: 0, width: 5 })

  it('runs NUTS, EP and Laplace on the benchmark’s probit regression', () => {
    expect(specs.map((s) => s.id)).toEqual(['nuts', 'ep', 'laplace'])
    expect(BOUND.sizes).toEqual({ row: 50, feature: 3 })
    for (const k of kept) expect(k.kept?.posterior.sites).toContain('w')
  })

  it('keeps the messages of EP as a record of the run', () => {
    expect(kept[1]?.kept?.record?.steps.length).toBeGreaterThan(1)
    expect(kept[0]?.kept?.record).toBeUndefined()
  })

  it('compares the methods with the grid on the three weights', () => {
    const c = compare(kept, truth.marginals)
    expect(c?.methods).toEqual(['NUTS', 'EP', 'Laplace'])
    expect(c?.quantities.map((q) => q.name)).toEqual([...COORDINATES])
    expect(c?.agreement).toHaveLength(9)
  })
})

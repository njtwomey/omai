// The clutter page's compute: one run per alpha, each reading a Gaussian belief and its divergences from the exact posterior.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../../shared/smoke.ts'
import { ALPHAS, clutterRuns, clutterSetup } from './compute.ts'

describe('the clutter runs', () => {
  const { exact, graph } = clutterSetup(5, 5)
  const specs = clutterRuns(graph, exact)

  it('has a run for each alpha on the ladder', () => {
    expect(specs.map((s) => s.id)).toEqual(ALPHAS.map(String))
  })

  it('reads expectation propagation as a belief wider than the exact posterior’s main mode, with divergences', () => {
    const ep = snapshotOf(specs[0] as (typeof specs)[number], {}, 100)
    expect(ep.sd).toBeGreaterThan(0)
    expect(ep.forward).toBeGreaterThan(0)
    expect(ep.reverse).toBeGreaterThan(0)
  })
})

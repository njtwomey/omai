// The guides page's compute: one run per guide, each read as a record of the variational run and draws of the guide.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../../shared/smoke.ts'
import { GUIDES, guideRuns, targetDraws } from './compute.ts'

describe('the guide runs', () => {
  const specs = guideRuns()
  it('has the four guides, and exact draws of the banana to set beside them', () => {
    expect(specs.map((s) => s.id)).toEqual(GUIDES.map((g) => g.id))
    expect(targetDraws(10).x0).toHaveLength(10)
  })

  it('reads a mean-field run as an ELBO record with beliefs and 500 draws of the guide', () => {
    const k = snapshotOf(specs[0] as (typeof specs)[number], {}, 150)
    expect(k.record.elbo.steps.length).toBeGreaterThan(10)
    expect(k.x0).toHaveLength(500)
    expect(k.x1).toHaveLength(500)
    expect(k.snapshots).toHaveLength(1)
  })
})

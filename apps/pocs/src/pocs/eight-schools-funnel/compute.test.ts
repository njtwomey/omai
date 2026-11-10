// The eight schools page's compute: three forms of the model, each read as the diagnostics of its chains.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../shared/smoke.ts'
import { FORMS, schoolsRuns } from './compute.ts'

describe('the eight schools runs', () => {
  const specs = schoolsRuns()
  it('has the centred, non-centred and collapsed forms', () => {
    expect(specs.map((s) => s.id)).toEqual([...FORMS])
  })

  it('reads the chains of the collapsed form: mu, log tau and their diagnostics, no school effects', () => {
    const k = snapshotOf(specs[2] as (typeof specs)[number], { numWarmup: 40, numDraws: 60 }, 200)
    expect(k.draws).toBe(60)
    expect(k.record?.quantities.map((q) => q.name)).toEqual(['mu', 'log tau'])
    expect(k.divergences).toBe(0)
    expect(k.essTau).toBeGreaterThan(1)
  })

  it('reads the first school’s effect as well in the non-centred form', () => {
    const k = snapshotOf(specs[1] as (typeof specs)[number], { numWarmup: 40, numDraws: 60 }, 200)
    expect(k.record?.quantities.map((q) => q.name)).toEqual(['mu', 'log tau', 'theta[0]'])
  })
})

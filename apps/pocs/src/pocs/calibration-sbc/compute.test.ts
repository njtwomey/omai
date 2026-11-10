// The calibration page's compute: a calibration run per method, a replication a step, read as the record of the ranks so far.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../shared/smoke.ts'
import { calibrationRuns, METHODS } from './compute.ts'

describe('the calibration runs', () => {
  const specs = calibrationRuns()
  it('has a run for each method', () => {
    expect(specs.map((s) => s.id)).toEqual(METHODS.map((m) => m.id))
  })

  it('has no record before the tenth replication, and the ranks of the intercept and the slope from then on', () => {
    const early = snapshotOf(specs[2] as (typeof specs)[number], {}, 9)
    expect(early.record).toBeUndefined()
    const k = snapshotOf(specs[2] as (typeof specs)[number], {}, 12)
    expect(k.done).toBe(12)
    expect(k.record?.quantities.map((q) => q.name)).toEqual(['a', 'b'])
  })
})

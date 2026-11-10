// The evidence page's compute: six estimators on a conjugate model, each read as the log evidence so far, against the closed form.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../shared/smoke.ts'
import { ESTIMATORS, evidenceRuns, exactLogEvidence } from './compute.ts'

describe('the evidence runs', () => {
  const exact = exactLogEvidence('normal-normal')
  const specs = evidenceRuns('normal-normal', 1)
  it('has the closed form and a run for each estimator', () => {
    expect(exact).toBeCloseTo(-23.9485, 3)
    expect(specs.map((s) => s.id)).toEqual(ESTIMATORS.map(([id]) => id))
  })

  it('reads importance sampling and bridge sampling within a tenth of a nat of the exact value', () => {
    for (const id of ['importance-evidence', 'bridge-sampling']) {
      const spec = specs.find((s) => s.id === id) as (typeof specs)[number]
      const e = snapshotOf(spec, {}, 2000)
      expect(Math.abs((e?.value ?? Number.NaN) - exact)).toBeLessThan(0.1)
    }
  })
})

// The two-modes page's compute: NUTS and parallel tempering, each read as chain diagnostics and the share of each chain in
// the upper mode, tempering with its swap rates.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../shared/smoke.ts'
import { modeRuns } from './compute.ts'

describe('the two-mode runs', () => {
  const specs = modeRuns()
  it('runs NUTS and parallel tempering with four chains each', () => {
    expect(specs.map((s) => s.id)).toEqual(['nuts', 'tempering'])
    const k = snapshotOf(specs[0] as (typeof specs)[number], { numWarmup: 30, numDraws: 40 }, 100)
    expect(k.upper).toHaveLength(4)
    expect(k.record?.quantities.map((q) => q.name)).toEqual(['x[0]', 'x[1]'])
  })

  it('reads the swap rate of each neighbouring pair of temperatures and the round trips', () => {
    const k = snapshotOf(specs[1] as (typeof specs)[number], { numWarmup: 40, numDraws: 80 }, 200)
    expect(k.swaps).toHaveLength(5)
    for (const r of k.swaps ?? []) expect(r >= 0 && r <= 1).toBe(true)
    expect(k.roundTrips).toBeGreaterThanOrEqual(0)
  })
})

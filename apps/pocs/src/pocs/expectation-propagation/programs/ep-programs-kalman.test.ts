// The vector-state tracker: EP on the scan agrees with the exact Kalman smoother to rounding, and over many seeds its 90% intervals
// cover the true position and the true velocity (which is never read) at about the rate they claim.

import { describe, expect, it } from 'vitest'
import { agreement, recovery } from '../../../shared/ep/recovery.ts'
import { defaultSetup, infer, simulate } from './ep-program-kalman/program.ts'
import { kalmanReference } from './ep-program-kalman/reference.ts'

describe('ep-program-kalman: a position and a velocity read through the position', () => {
  it('agrees with the exact smoother on the position and the velocity, and converges', async () => {
    for (const seed of [1, 2, 3]) {
      const setup = { ...defaultSetup, seed }
      const sim = simulate(setup)
      const { position, velocity, run } = await infer(sim.y, setup)
      expect(run.status, `seed ${seed}`).toBe('converged')
      const exact = kalmanReference(sim.y, setup)
      for (const [rows, ref] of [
        [position, exact.position],
        [velocity, exact.velocity],
      ] as const) {
        const a = agreement(rows, ref)
        expect(a.meanZ, `seed ${seed}`).toBeLessThan(1e-6)
        expect(a.sdRatioRange[0], `seed ${seed}`).toBeGreaterThan(1 - 1e-6)
        expect(a.sdRatioRange[1], `seed ${seed}`).toBeLessThan(1 + 1e-6)
      }
    }
  }, 60_000)

  it('is calibrated over 12 seeds: the truth inside the 90% intervals at 80 to 98% of the steps, errors near one posterior sd', async () => {
    const pooled = { position: [0, 0, 0], velocity: [0, 0, 0] }
    for (let seed = 1; seed <= 12; seed++) {
      const setup = { ...defaultSetup, seed, steps: 50 }
      const sim = simulate(setup)
      const { position, velocity } = await infer(sim.y, setup)
      for (const [name, rows, truth] of [
        ['position', position, sim.position],
        ['velocity', velocity, sim.velocity],
      ] as const) {
        const r = recovery(rows, truth)
        pooled[name][0] = (pooled[name][0] as number) + r.coverage / 12
        pooled[name][1] = (pooled[name][1] as number) + r.zrms ** 2 / 12
      }
    }
    for (const name of ['position', 'velocity'] as const) {
      expect(pooled[name][0], name).toBeGreaterThan(0.8)
      expect(pooled[name][0], name).toBeLessThan(0.98)
      expect(Math.sqrt(pooled[name][1] as number), name).toBeGreaterThan(0.7)
      expect(Math.sqrt(pooled[name][1] as number), name).toBeLessThan(1.4)
    }
  }, 120_000)
})

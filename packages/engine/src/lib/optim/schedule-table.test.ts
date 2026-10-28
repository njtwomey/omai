import { describe, expect, it } from 'vitest'
import { defaults, grid } from '../space/index.ts'
import { schedules } from './schedule-table.ts'

describe('schedules', () => {
  it('every shape starts at the peak (or warms up to it) and a decaying one never rises again', () => {
    for (const config of grid(schedules.space, 1)) {
      const name = JSON.stringify(config)
      const s = schedules.make(config)(0.1)
      const { warmup, decays, restarts } = schedules.properties(config)
      const warm = warmup ? (config as { warmupSteps: number }).warmupSteps : 0
      expect(s(warm), name).toBeCloseTo(0.1, 12)
      if (warmup) expect(s(0), name).toBeLessThan(0.1)
      if (decays && !restarts) {
        let last = s(warm)
        for (let t = warm + 1; t < warm + 300; t++) {
          const v = s(t)
          expect(v, `${name} at ${t}`).toBeLessThanOrEqual(last + 1e-12)
          last = v
        }
      }
      if (!decays) expect(s(500), name).toBeCloseTo(0.1, 12)
    }
  })

  it('is one slot with the constant schedule as the default, and restarts restart', () => {
    expect(defaults(schedules.space)).toEqual({ kind: 'constant' })
    expect(schedules.where((p) => p.warmup)).toEqual(['warmupCosine', 'warmupExponential'])
    const restarts = schedules.make({ kind: 'cosineRestarts', period: 10, growth: 1, endRatio: 0 })(1)
    expect(restarts(0)).toBeCloseTo(1, 12)
    expect(restarts(10)).toBeCloseTo(1, 12)
    expect(restarts(5)).toBeLessThan(1)
  })
})

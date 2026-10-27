import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { mcmc } from './mcmc.ts'

describe('mcmc', () => {
  it('runs a chain with every position on the tape and the target as a probe', () => {
    const s = session(mcmc, { mode: 'eager', steps: 40, config: { target: 'mixture', kernel: 'hmc', stepSize: 0.3 } })
    const run = s.run()
    // The start and every transition.
    expect(tensorSeries(run, 'chain/position')).toHaveLength(41)
    const acceptance = metricSeries(run, 'chain/acceptance').values
    expect(acceptance.at(-1)).toBeGreaterThan(0)
    expect(acceptance.at(-1)).toBeLessThanOrEqual(1)
    const surface = s.probe(20, 'target/density') as { readonly kind: string; readonly values: number[][] }
    expect(surface.kind).toBe('decisionSurface')
    expect(surface.values).toHaveLength(70)
    expect(Math.max(...surface.values.flat())).toBeCloseTo(1, 6)
  })
})

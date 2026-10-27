import { describe, expect, it } from 'vitest'
import { metricSeries, session } from '../../lib/trace/index.ts'
import { changeOfVariables } from './changeOfVariables.ts'

describe('changeOfVariables', () => {
  it('deforms to the full map with a density and a grid to probe at any step', () => {
    const s = session(changeOfVariables, { mode: 'eager', steps: 100, config: { map: 'squash', notches: 20 } })
    expect(s.t).toBe(20)
    expect(s.status).toBe('converged')
    const ss = metricSeries(s.run(), 'map/s').values
    expect(ss.at(-1)).toBe(1)
    // The squash shrinks the support, so the pushed entropy falls.
    const entropy = metricSeries(s.run(), 'map/entropy').values
    expect(entropy.at(-1)).toBeLessThan(entropy[0] ?? 0)
    const density = s.probe(10, 'map/density') as { readonly values: number[][]; readonly rows: { points: number[][] } }
    expect(density.values).toHaveLength(60)
    expect(density.values.flat().every((v) => v >= 0 && Number.isFinite(v))).toBe(true)
    expect(density.rows.points).toHaveLength(300)
    const grid = s.probe(10, 'map/grid') as { readonly series: unknown[] }
    expect(grid.series).toHaveLength(26)
  })
})

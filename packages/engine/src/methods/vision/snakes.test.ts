import type { DecisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { snakes } from './snakes.ts'

describe('snakes', () => {
  it('shrinks from the frame onto the shape, with the picture as a probe', () => {
    const s = session(snakes, { mode: 'eager', steps: 150, config: { size: 40, points: 48 } })
    const run = s.run()
    const pts = tensorSeries(run, 'contour/points')
    expect(pts.length).toBeGreaterThan(10)
    expect(pts[0]?.shape).toEqual([48, 2])
    const area = metricSeries(run, 'contour/area').values
    expect(area.at(-1)).toBeLessThan((area[0] ?? 0) * 0.8)
    const overlap = metricSeries(run, 'contour/overlap').values
    expect(overlap.at(-1)).toBeGreaterThan(overlap[0] ?? 1)
    const p = s.probe(5, 'contour/picture') as DecisionSurface
    expect(p.values?.length).toBe(40)
    expect(p.rows?.points.length).toBe(48)
  })
})

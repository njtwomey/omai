import type { CurveSet, DecisionSurface } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session } from '../../lib/trace/index.ts'
import { haarBoosting } from './haarBoosting.ts'

describe('haarBoosting', () => {
  it('drives the training error down stump by stump, with the stump and template as probes', () => {
    const s = session(haarBoosting, { mode: 'eager', steps: 8, config: { windows: 30, stride: 4 } })
    const run = s.run()
    const error = metricSeries(run, 'boost/trainingError').values
    expect(error.length).toBeGreaterThan(0)
    expect(error.at(-1)).toBeLessThan(0.35)
    const stump = s.probe(1, 'boost/stump') as DecisionSurface
    expect(stump.values?.length).toBe(16)
    expect(stump.values?.flat().some((v) => v !== 0)).toBe(true)
    const t = s.probe(error.length, 'boost/template') as DecisionSurface
    expect(t.values?.length).toBe(16)
    const w = s.probe(2, 'boost/weights') as CurveSet
    expect(w.series.map((q) => q.name)).toEqual(['circles (positive)', 'squares and triangles (negative)'])
  })
})

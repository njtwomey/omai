import type { DecisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { peronaMalik } from './peronaMalik.ts'

describe('peronaMalik', () => {
  it('smooths a noisy scene towards the clean one, with the conduction as a probe', () => {
    const s = session(peronaMalik, { mode: 'eager', steps: 30, config: { size: 32 } })
    const run = s.run()
    const pixels = tensorSeries(run, 'image/pixels')
    expect(pixels).toHaveLength(30)
    expect(pixels[0]?.shape).toEqual([32, 32])
    const psnr = metricSeries(run, 'quality/psnr').values
    const start = metricSeries(run, 'quality/psnrStart').values
    expect(psnr.at(-1)).toBeGreaterThan(start.at(-1) ?? Number.POSITIVE_INFINITY)
    const g = s.probe(10, 'image/conduction') as DecisionSurface
    expect(g.kind).toBe('decisionSurface')
    expect(g.values?.length).toBe(32)
    expect(g.values?.flat().every((v) => v >= 0 && v <= 1)).toBe(true)
  })
})

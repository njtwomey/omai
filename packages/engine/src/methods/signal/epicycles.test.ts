import { describe, expect, it } from 'vitest'
import { shapeOutline } from '../../lib/datasets/outlines/index.ts'
import { fourierDrawing } from './epicycles.ts'

describe('fourierDrawing', () => {
  it('records the chain, the trace and the spectrum, keeping more energy with more terms', () => {
    const pts = shapeOutline('cat', 512)
    const few = fourierDrawing(pts, { keep: { count: 3 } })
    const many = fourierDrawing(pts, { keep: { energy: 0.999 }, order: 'speed', zoom: 50, source: 'cat' })
    const [r] = many.exposition
    expect(r?.kind).toBe('epicycles')
    if (r?.kind !== 'epicycles') return
    expect(r.chain.k.length).toBe(many.chain.length)
    expect(r.trace.x.at(-1)).toBeCloseTo(r.trace.x[0] ?? 0, 12)
    expect(r.spectrum.kept.filter(Boolean)).toHaveLength(many.chain.length)
    expect(r.zoom).toBe(50)
    expect(many.energyKept).toBeGreaterThanOrEqual(0.999)
    expect(few.energyKept).toBeLessThan(many.energyKept)
  })
})

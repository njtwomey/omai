import type { BoardSteps, DecisionSurface } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { grabCutSegmentation } from './grabCutSegmentation.ts'

describe('grabCutSegmentation', () => {
  it('cuts the polygon out of its rectangle, with a board and a picture', () => {
    const s = session(grabCutSegmentation, { mode: 'eager', steps: 6, config: { size: 24 } })
    const run = s.run()
    const masks = tensorSeries(run, 'cut/mask')
    expect(masks.length).toBeGreaterThan(0)
    expect(masks[0]?.shape).toEqual([24, 24])
    const overlap = metricSeries(run, 'cut/overlap').values
    expect(overlap.at(-1)).toBeGreaterThan(0.5)
    const board = s.project('board') as BoardSteps
    expect(board.categories).toEqual(['background', 'foreground'])
    expect(board.steps.length).toBe(masks.length)
    const p = s.probe(1, 'cut/picture') as DecisionSurface
    expect(p.values?.length).toBe(24)
    expect(p.rows?.classes).toEqual(['rectangle'])
  })
})

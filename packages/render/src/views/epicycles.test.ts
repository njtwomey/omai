import { shapeOutline } from 'mdml-engine/datasets/outlines'
import { fourierDrawing } from 'mdml-engine/methods/signal'
import { describe, expect, it } from 'vitest'
import { epicycleFrames } from './epicycles.ts'
import { viewFor } from './registry.ts'

describe('epicycleFrames', () => {
  it('plays one cycle, with and without the zoom on the pen', () => {
    const pts = shapeOutline('cat', 256).map((p): [number, number] => [p[0], p[1]])
    for (const zoom of [undefined, 50]) {
      const [record] = fourierDrawing(pts, {
        keep: { count: 20 },
        steps: 24,
        ...(zoom === undefined ? {} : { zoom }),
      }).exposition
      if (record?.kind !== 'epicycles') throw new Error('expected an epicycles record')
      const frames = epicycleFrames(record)
      expect(frames.count).toBe(25)
      for (const f of [0, 12, 24]) expect(frames.option(f).series).toBeDefined()
      expect(frames.label(12)).toContain('t 0.500')
    }
    expect(viewFor('epicycles').component).toBe('EpicyclePlayer')
  })
})

import { describe, expect, it } from 'vitest'
import { cornellBox, pathTrace } from '../graphics/index.ts'
import { key } from '../random/index.ts'
import { progressiveRender } from './rendering.ts'

describe('progressiveRender', () => {
  it('records the colour image as the samples double, the error falling, and ends on the one-shot render', () => {
    const { scene, camera } = cornellBox({ contents: 'blocks' })
    const r = progressiveRender(scene, camera, 16, 12, { key: key(0), samples: 64, maxBounces: 2 })
    const [record] = r.exposition
    expect(record.kind).toBe('imageSteps')
    expect(record.channels).toBe(3)
    expect(record.steps.map((s) => s.iteration)).toEqual([1, 2, 4, 8, 16, 32, 64])
    expect(record.steps[0]?.images[0]?.[0]).toHaveLength(16 * 12 * 3)
    const errors = record.steps.map((s) => s.objective ?? 0)
    // The error falls like one over the square root of the samples: 4 to 64 samples should more than halve it.
    expect(errors[6]).toBeLessThan((errors[2] ?? 0) / 2)
    // Slicing the run changes nothing: it is the same keyed run as rendering in one go.
    const once = pathTrace(scene, camera, 16, 12, { key: key(0), samples: 64, maxBounces: 2 })
    expect(Array.from(r.image.rgb)).toEqual(Array.from(once.image.rgb))
  })
})

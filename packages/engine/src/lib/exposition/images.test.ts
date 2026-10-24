import { describe, expect, it } from 'vitest'
import { imageSteps } from './images.ts'
import { isExposition } from './record.ts'

describe('imageSteps', () => {
  it('keeps groups of images per step', () => {
    const r = imageSteps({
      method: 'vae',
      height: 2,
      width: 2,
      groups: ['data', 'samples'],
      steps: [
        { iteration: 0, images: [[[0, 1, 1, 0]], [[0, 0, 0, 0]]] },
        { iteration: 5, images: [[[0, 1, 1, 0]], [[0, 1, 1, 0]]], objective: 1.5 },
      ],
    })
    expect(r.kind).toBe('imageSteps')
    expect(isExposition(r)).toBe(true)
  })

  it('rejects images of the wrong size and missing groups', () => {
    const base = { method: 'm', height: 2, width: 2, groups: ['a'] }
    expect(() => imageSteps({ ...base, steps: [{ iteration: 0, images: [[[0, 1, 2]]] }] })).toThrow(/images\[0\]\[0\]/)
    expect(() => imageSteps({ ...base, steps: [{ iteration: 0, images: [] }] })).toThrow(/images/)
    expect(() => imageSteps({ ...base, height: 0, steps: [] })).toThrow(/size/)
  })
})

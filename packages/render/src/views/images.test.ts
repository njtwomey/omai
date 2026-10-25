import { imageSteps } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { imageFrames } from './images.ts'
import { stillOf, viewFor } from './registry.ts'

const seriesOf = (o: object): { name?: string }[] => (o as { series: { name?: string }[] }).series

describe('image view', () => {
  it('plays a mosaic of groups of images with the objective', () => {
    const r = imageSteps({
      method: 'vae',
      height: 2,
      width: 2,
      groups: ['data', 'samples'],
      steps: [
        { iteration: 0, images: [[[0, 1, 1, 0]], [[0, 0, 0, 0]]], objective: 3 },
        { iteration: 9, images: [[[0, 1, 1, 0]], [[0, 1, 1, 0]]], objective: 1 },
      ],
      stepName: 'step',
    })
    const frames = imageFrames(r)
    expect(frames.count).toBe(2)
    expect(frames.label(1)).toBe('step 9/9 · 1')
    const names = seriesOf(frames.option(0)).map((s) => s.name)
    expect(names).toContain('__images')
    expect(names).toContain('__groups')
    expect(viewFor('imageSteps').component).toBe('ImagePlayer')
    expect(stillOf(r)).toBeTruthy()
  })

  it('draws a single step without an objective panel', () => {
    const r = imageSteps({
      method: 'm',
      height: 1,
      width: 1,
      groups: ['a'],
      steps: [{ iteration: 0, images: [[[1]]] }],
    })
    expect(seriesOf(imageFrames(r).option(0)).length).toBe(2)
  })
})

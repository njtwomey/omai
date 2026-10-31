import { sampleGrid } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { sampleGridScene } from './sample-grid.ts'

const images = sampleGrid({
  modality: 'image',
  height: 2,
  width: 2,
  stepName: 'epoch',
  columns: 2,
  steps: [0, 1, 2, 4, 8].map((step) => ({
    step,
    samples: [
      [0, 1, 1, 0],
      [1, 0, 0, 1],
      [0.5, 0.5, 0.5, 0.5],
    ],
    labels: ['a', 'b', 'c'],
  })),
})
const nodes = (kind: string, step: number, filmstrip = 0) =>
  sampleGridScene(images, step, { filmstrip }).children.filter((n) => n.kind === kind)

describe('the sample grid scene', () => {
  it('draws one image per sample of the viewed step, with a caption each, and names the step', () => {
    const s = sampleGridScene(images, 4)
    expect(nodes('image', 4).length).toBe(3)
    expect(s.title).toBe('3 image samples after epoch 4')
    const texts = s.children.flatMap((n) => (n.kind === 'text' ? [n.text] : []))
    expect(texts).toEqual(['epoch 4', 'a', 'b', 'c'])
  })

  it('adds a filmstrip row of small images per earlier step, with the step written beside it', () => {
    const s = sampleGridScene(images, 8, { filmstrip: 2 })
    expect(s.children.filter((n) => n.kind === 'image').length).toBe(3 + 2 * 3)
    expect(s.title).toContain('with 2 earlier epochs in a filmstrip')
    const texts = s.children.flatMap((n) => (n.kind === 'text' ? [n.text] : []))
    expect(texts).toContain('earlier epochs')
  })

  it('draws text samples in cells and says so when nothing is recorded', () => {
    const t = sampleGrid({ modality: 'text', columns: 1, steps: [{ step: 0, samples: ['hello world', 'abc'] }] })
    const s = sampleGridScene(t, 0)
    expect(s.children.filter((n) => n.kind === 'rect').length).toBe(2)
    expect(s.children.some((n) => n.kind === 'text' && n.text === 'hello world')).toBe(true)
    expect(sampleGridScene(sampleGrid({ modality: 'text', steps: [] })).children.length).toBe(1)
  })
})

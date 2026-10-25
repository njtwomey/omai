import { counterfactualSet, prototypeSet } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { counterfactualRows, counterfactualTableView, prototypeView } from './examples.ts'
import { stillOf, viewFor } from './registry.ts'

type Opt = { grid?: unknown[]; series: { name?: string }[]; graphic?: { type: string; style?: { text?: string } }[] }

const cf = counterfactualSet({
  method: 'growing spheres',
  features: ['income', 'debt', 'age'],
  output: 'P(approve)',
  original: [30, 10, 40],
  originalPrediction: 0.3,
  counterfactuals: [
    { x: [36, 10, 40], prediction: 0.6, distance: 6, valid: true, changed: [0] },
    { x: [30, 4, 40], prediction: 0.45, distance: 6, valid: false, changed: [1] },
  ],
  immutable: [2],
})

describe('example-based views', () => {
  it('tabulates the changed and immutable features with each column’s outcome', () => {
    expect(counterfactualRows(cf)).toEqual([0, 1, 2])
    const texts = ((counterfactualTableView(cf) as unknown as Opt).graphic ?? []).map((g) => g.style?.text)
    expect(texts).toContain('age (fixed)')
    expect(texts).toContain('no')
    expect(viewFor('counterfactuals').component).toBe('CounterfactualView')
    expect(stillOf(cf)).toBeDefined()
  })

  it('marks prototypes and criticisms on a scatter, or draws them as images', () => {
    const scatter = prototypeSet({
      method: 'MMD-critic',
      features: ['a', 'b'],
      prototypes: [0],
      criticisms: [2],
      prototypeData: [[0, 0]],
      criticismData: [[5, 5]],
      witness: [-0.1, -0.1, 0.4],
      points: [
        [0, 0],
        [0.1, 0],
        [5, 5],
      ],
      labels: ['x', 'x', 'y'],
    })
    expect((prototypeView(scatter) as unknown as Opt).series.map((s) => s.name)).toEqual([
      '__data',
      'prototypes',
      'criticisms',
    ])
    const images = prototypeSet({
      method: 'MMD-critic',
      features: ['p0', 'p1', 'p2', 'p3'],
      prototypes: [0, 1],
      criticisms: [2],
      prototypeData: [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
      ],
      criticismData: [[1, 1, 1, 1]],
      witness: [0, 0, 0.2],
      imageShape: [2, 2],
    })
    expect((prototypeView(images) as unknown as Opt).grid).toHaveLength(3)
  })
})

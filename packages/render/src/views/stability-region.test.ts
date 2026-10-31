import { sampleStabilityRegion } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { stabilityRegionLayers, stabilityRegionView, stabilityVerdict } from './stability-region.ts'

const euler = (marks: [number, number][]) =>
  sampleStabilityRegion(
    'forward Euler',
    { numerator: [1, 1] },
    { real: [-3, 1], imaginary: [-2, 2] },
    {
      n: 81,
      order: 1,
      stepSize: 0.5,
      marks: marks.map((at) => ({ at })),
    },
  )

describe('the stability region view', () => {
  it('counts the marks inside the disc of forward Euler, outside it and off the plot', () => {
    expect(
      stabilityVerdict(
        euler([
          [-1, 0],
          [-0.5, 0.5],
          [-2.5, 0],
          [0.5, 0.5],
          [-8, 0],
        ]),
      ),
    ).toEqual({
      stable: 2,
      unstable: 2,
      off: 1,
    })
    expect(stabilityVerdict(euler([]))).toEqual({ stable: 0, unstable: 0, off: 0 })
  })

  it('draws the region, its boundary, the axis and a series for each kind of mark', () => {
    // Region, boundary, imaginary axis: three layers; the marks add one layer with a series per verdict present.
    expect(stabilityRegionLayers(euler([])).length).toBe(3)
    const layers = stabilityRegionLayers(
      euler([
        [-1, 0],
        [-2.5, 0],
      ]),
    )
    expect(layers.length).toBe(4)
    expect(layers[3]?.series.length).toBe(2)
    const o = stabilityRegionView(
      euler([
        [-1, 0],
        [-2.5, 0],
      ]),
    ) as { aria: { label: { description: string } } }
    expect(o.aria.label.description).toContain('forward Euler, order 1')
    expect(o.aria.label.description).toContain('1 of the eigenvalues times the step size are inside it, 1 outside')
  })
})

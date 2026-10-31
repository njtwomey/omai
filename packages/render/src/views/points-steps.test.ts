import { pointsSteps } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import {
  CAMERA_PRESETS,
  cameraPreset,
  pointsStepsScene3d,
  pointsStepsValueRange,
  pointsStepsView,
  trailOpacity,
} from './points-steps.ts'

const r = pointsSteps({
  dims: 3,
  valueName: 'speed',
  range: [
    [0, 1],
    [0, 1],
    [0, 1],
  ],
  steps: [
    {
      step: 0,
      points: [
        [0, 0, 0],
        [1, 1, 1],
      ],
      values: [0, 10],
    },
    { step: 10, points: [[0.5, 0.5, 0.5]], values: [5] },
  ],
})

describe('the look of accumulating points', () => {
  it('fades a trail linearly to a tenth over the given number of steps', () => {
    expect(trailOpacity([0, 5, 10], 10)).toEqual([0.1, 0.55, 1])
    expect(trailOpacity([0, 100], 10)).toEqual([0.1, 1])
    expect(trailOpacity([0, 5])).toEqual([1, 1])
  })

  it('finds the extent of the values over the whole run, or the record says', () => {
    expect(pointsStepsValueRange(r)).toEqual([0, 10])
    expect(pointsStepsValueRange({ ...r, valueRange: [-1, 2] })).toEqual([-1, 2])
  })

  it('colours a 3-D scene by value along the ramp and fades the older points', () => {
    const plain = pointsStepsScene3d(r, 10, { colour: 'none' })
    const valued = pointsStepsScene3d(r, 10, { colour: 'values' })
    const fills = (s: typeof plain): unknown[] =>
      s.items.filter((i) => i.kind === 'point').map((i) => (i as { fill?: string }).fill)
    expect(new Set(fills(plain)).size).toBeLessThanOrEqual(2)
    const v = fills(valued)
    expect(new Set(v).size).toBe(3)
    expect(v.every((f) => typeof f === 'string' && f.startsWith('#'))).toBe(true)
    const faded = pointsStepsScene3d(r, 10, { colour: 'values', fade: 10 })
    const opacities = faded.items.filter((i) => i.kind === 'point').map((i) => (i as { opacity?: number }).opacity)
    expect(opacities).toEqual([0.1, 0.1, 1])
    expect(valued.title).toContain('coloured by speed from 0 to 10')
  })

  it('draws a 2-D figure in bands of value and age, and says so in its description', () => {
    const flat = pointsSteps({
      dims: 2,
      valueName: 'speed',
      steps: [
        {
          step: 0,
          points: [
            [0, 0],
            [1, 1],
          ],
          values: [0, 10],
        },
        { step: 10, points: [[0.5, 0.5]], values: [5] },
      ],
    })
    const o = pointsStepsView(flat, 10, { colour: 'values', fade: 10 }) as {
      series: unknown[]
      aria: { label: { description: string } }
    }
    expect(o.series.length).toBe(3)
    expect(o.aria.label.description).toContain('coloured by speed from 0 to 10')
    expect(o.aria.label.description).toContain('fading')
    // With no colour channel, the newest points are set apart in their own series, as before.
    expect((pointsStepsView(flat, 10, { colour: 'none' }) as { series: unknown[] }).series.length).toBe(2)
  })

  it('names four viewpoints, three of them flat and orthographic', () => {
    expect(CAMERA_PRESETS).toEqual(['iso', 'front', 'side', 'top'])
    expect(cameraPreset('iso')).toEqual({ projection: 'perspective', azimuth: -60, elevation: 30 })
    expect(cameraPreset('top')).toEqual({ projection: 'orthographic', azimuth: -90, elevation: 90 })
    expect(cameraPreset('front').elevation).toBe(0)
  })
})

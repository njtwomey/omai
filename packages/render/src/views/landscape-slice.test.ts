import { landscapeSlice, sampleLandscape } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import {
  contourLevels,
  curveLossAt,
  landscapeSliceLayers,
  landscapeSliceScene3d,
  landscapeSliceView,
  sliceMinimum,
} from './landscape-slice.ts'

const x = [-2, -1, 0, 1, 2]
const bowl = sampleLandscape((a) => (a - 1) ** 2, x, undefined, {
  axes: ['step'],
  paths: [{ name: 'gd', coords: [[-2], [0], [1]] }],
  guides: [{ name: 'tangent', from: [-2, 9], to: [2, -3] }],
})
const plane = sampleLandscape((a, b) => a * a + 4 * b * b, x, [-1, 0, 1], {
  axes: ['u', 'v'],
  paths: [
    {
      coords: [
        [-2, 1],
        [0, 0],
      ],
    },
  ],
})

describe('the landscape slice view', () => {
  it('reads the loss between grid points and clamps beyond them', () => {
    expect(curveLossAt(bowl, -1.5)).toBe(6.5)
    expect(curveLossAt(bowl, 10)).toBe(1)
    expect(curveLossAt(bowl, -10)).toBe(9)
  })

  it('finds the lowest point of a curve and of a plane', () => {
    expect(sliceMinimum(bowl)).toEqual({ at: [1], loss: 0 })
    expect(sliceMinimum(plane)).toEqual({ at: [0, 0], loss: 0 })
  })

  it('spaces contour levels evenly, or by decades when the loss spans them', () => {
    expect(contourLevels([0, 4], 3)).toEqual([0, 2, 4])
    expect(contourLevels([1, 100], 3)).toEqual([1, 10, 100])
  })

  it('draws a curve with guides and iterates, and contours with the path over them', () => {
    // The curve, the guide, and the path's line and its points; the plane adds the contours and the lowest point.
    expect(landscapeSliceLayers(bowl).length).toBe(4)
    expect(landscapeSliceLayers(plane).length).toBe(4)
    const o = landscapeSliceView(bowl) as { aria: { label: { description: string } } }
    expect(o.aria.label.description).toContain('lowest at 1 (0)')
    expect(o.aria.label.description).toContain('3 optimiser iterates')
  })

  it('draws a plane as a surface of one face per cell, with the path lying on it', () => {
    const s = landscapeSliceScene3d(plane)
    expect(s.items.filter((i) => i.kind === 'face').length).toBe(4 * 2)
    expect(s.items.some((i) => i.kind === 'polyline')).toBe(true)
    expect(s.title).toContain('u and v')
  })

  it('a slice with no paths still draws', () => {
    expect(() => landscapeSliceView(landscapeSlice({ dims: 1, axes: ['a'], x: [0, 1], curve: [1, 2] }))).not.toThrow()
  })
})

import { decisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import {
  cellAt,
  clampViewport,
  dataToPixel,
  minimapRect,
  panViewport,
  pixelToData,
  sameViewport,
  surfaceRaster,
  surfaceViewport,
  zoomViewport,
} from './zoom.ts'

const view = { x: [0, 4], y: [0, 2] } as const

describe('zooming and panning a viewport', () => {
  it('zoom about a point keeps that point where it is on screen', () => {
    const size = [400, 200] as const
    const focus = pixelToData(view, size, 300, 40)
    const z = zoomViewport(view, focus, 3)
    const [px, py] = dataToPixel(z, size, focus[0], focus[1])
    expect(px).toBeCloseTo(300, 9)
    expect(py).toBeCloseTo(40, 9)
    expect(z.x[1] - z.x[0]).toBeCloseTo(4 / 3, 12)
    expect(z.y[1] - z.y[0]).toBeCloseTo(2 / 3, 12)
  })

  it('zooming in then out by the same factor returns the viewport', () => {
    const back = zoomViewport(zoomViewport(view, [1, 1.5], 5), [1, 1.5], 1 / 5)
    expect(back.x[0]).toBeCloseTo(0, 12)
    expect(back.x[1]).toBeCloseTo(4, 12)
    expect(back.y[1]).toBeCloseTo(2, 12)
  })

  it('refuses to collapse below the floating-point resolution', () => {
    const deep = { x: [-0.743644786, -0.7436447859], y: [0.1318259042, 0.1318259043] } as const
    const z = zoomViewport(deep, [-0.74364478595, 0.13182590425], 1e9)
    expect(sameViewport(z, deep)).toBe(true)
  })

  it('pan moves by fractions of the span and clamp keeps the viewport in the domain', () => {
    expect(panViewport(view, 0.5, -1)).toEqual({ x: [2, 6], y: [-2, 0] })
    const domain = { x: [-1, 5], y: [-1, 3] } as const
    const p = panViewport(view, 1, 1, domain)
    expect(p.x).toEqual([1, 5])
    expect(p.y).toEqual([1, 3])
    // Zooming out past the domain shrinks to it.
    const out = zoomViewport(view, [2, 1], 0.1, domain)
    expect(out.x).toEqual([-1, 5])
    expect(clampViewport({ x: [-3, -1], y: [0, 1] }, { x: [-2, 1], y: [-1, 1] }).x).toEqual([-2, 0])
  })

  it('maps pixels to data and back with row 0 at the top', () => {
    expect(pixelToData(view, [400, 200], 100, 50)).toEqual([1, 1.5])
    expect(dataToPixel(view, [400, 200], 1, 1.5)).toEqual([100, 50])
    expect(pixelToData(view, [400, 200], 0, 0)).toEqual([0, 2])
  })

  it('draws the viewport in the minimap of its domain', () => {
    expect(minimapRect({ x: [0, 1], y: [0, 1] }, { x: [-1, 1], y: [-1, 1] })).toEqual({
      left: 0.5,
      top: 0,
      width: 0.5,
      height: 0.5,
    })
    const whole = minimapRect(view, view)
    expect([whole.left, whole.top, whole.width, whole.height]).toEqual([0, 0, 1, 1])
  })
})

describe('reading and colouring a grid', () => {
  const s = decisionSurface({
    features: ['a', 'b'],
    x: [0, 1, 2],
    y: [10, 20],
    values: [
      [1, 2, 3],
      [4, 5, 6],
    ],
  })
  it('finds the nearest cell, with half a cell of slack, and nothing outside', () => {
    expect(cellAt(s, 1.2, 19)).toEqual({ i: 1, j: 1, value: 5 })
    expect(cellAt(s, -0.4, 10)?.value).toBe(1)
    expect(cellAt(s, -0.6, 10)).toBeUndefined()
    expect(cellAt(s, 1, 26)).toBeUndefined()
    expect(cellAt(decisionSurface({ features: ['a', 'b'], x: [0], y: [0] }), 0, 0)).toBeUndefined()
    expect(surfaceViewport(s)).toEqual({ x: [0, 2], y: [10, 20] })
  })

  it('rasters the last row first and maps the extremes to the ends of the ramp', () => {
    const r = surfaceRaster(s, { ramp: ['#000000', '#ffffff'], range: [1, 6] })
    expect([r.width, r.height]).toEqual([3, 2])
    // Top row of the picture is the last row of the grid: values 4, 5, 6.
    expect(Array.from(r.rgba.slice(0, 4))).toEqual(
      [Math.round(255 * (3 / 5)), Math.round(255 * (3 / 5)), Math.round(255 * (3 / 5)), 255].map((v, k) =>
        k === 3 ? 255 : v,
      ),
    )
    // The bottom-left cell holds the minimum: black.
    expect(Array.from(r.rgba.slice(4 * 3, 4 * 3 + 3))).toEqual([0, 0, 0])
    const nan = surfaceRaster(decisionSurface({ features: ['a', 'b'], x: [0], y: [0], values: [[Number.NaN]] }))
    expect(Array.from(nan.rgba)).toEqual([128, 128, 128, 255])
  })
})

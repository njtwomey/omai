import { describe, expect, it } from 'vitest'
import { circle, group, line, polygon, rect, scene } from './build.ts'
import { paintColour, rasterise } from './raster.ts'

const pixel = (img: ReturnType<typeof rasterise>, x: number, y: number): number[] =>
  Array.from(img.rgba.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4))

describe('rasterise', () => {
  it('fills rectangles, circles and polygons over the background', () => {
    const s = scene(10, 10, [
      rect(0, 0, 5, 10, { fill: '#ff0000' }),
      circle(7.5, 2.5, 1.5, { fill: '#00ff00' }),
      polygon(
        [
          [6, 6],
          [10, 6],
          [10, 10],
        ],
        { fill: '#0000ff' },
      ),
    ])
    const img = rasterise(s)
    expect(pixel(img, 1, 1)).toEqual([255, 0, 0, 255])
    expect(pixel(img, 7, 2)).toEqual([0, 255, 0, 255])
    expect(pixel(img, 9, 7)).toEqual([0, 0, 255, 255])
    expect(pixel(img, 6, 9)).toEqual([255, 255, 255, 255])
  })

  it('applies group transforms, scaling and line strokes', () => {
    const s = scene(8, 8, [
      group([rect(0, 0, 2, 2, { fill: '#000000' })], { translate: [4, 4] }),
      line(0, 0, 0, 8, { stroke: '#ff0000', strokeWidth: 2 }),
    ])
    const img = rasterise(s, { width: 16, height: 16 })
    expect(pixel(img, 9, 9)).toEqual([0, 0, 0, 255])
    expect(pixel(img, 0, 8)).toEqual([255, 0, 0, 255])
    expect(pixel(img, 4, 4)).toEqual([255, 255, 255, 255])
  })

  it('blends opacity and resolves theme tokens and tab colours', () => {
    const img = rasterise(scene(2, 2, [rect(0, 0, 2, 2, { fill: '#000000', opacity: 0.5 })]))
    expect(pixel(img, 0, 0)).toEqual([128, 128, 128, 255])
    expect(paintColour('tab0')).toEqual([31, 119, 180, 1])
    expect(paintColour('none')).toBeUndefined()
    expect(paintColour('rgba(1, 2, 3, 0.5)')).toEqual([1, 2, 3, 0.5])
    expect(paintColour('#zz' as never)).toEqual(paintColour('fg'))
  })
})

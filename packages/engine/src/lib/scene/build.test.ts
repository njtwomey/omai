import { describe, expect, it } from 'vitest'
import { tensor } from '../tensor/index.ts'
import { group, image, isScene, polygon, scene, toPixels } from './build.ts'

describe('build', () => {
  it('converts grey tensors with their range, and NaN to transparent', () => {
    const px = toPixels(tensor([[0, 5, 10, Number.NaN]]))
    expect([px.width, px.height]).toEqual([4, 1])
    expect(Array.from(px.rgba)).toEqual([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 0, 0, 0, 0])
    expect(Array.from(toPixels([[5]], { range: [0, 10] }).rgba)).toEqual([128, 128, 128, 255])
    expect(Array.from(toPixels([[3, 3]]).rgba.slice(0, 4))).toEqual([0, 0, 0, 255])
  })

  it('converts RGB and RGBA arrays in 0-1 or 0-255', () => {
    expect(Array.from(toPixels([[[0, 128, 255]]]).rgba)).toEqual([0, 128, 255, 255])
    expect(Array.from(toPixels([[[1, 1, 1, 0.5]]]).rgba)).toEqual([255, 255, 255, 128])
    expect(Array.from(toPixels(tensor([[[0.5], [1]]])).rgba.slice(4))).toEqual([255, 255, 255, 255])
  })

  it('rejects bad shapes', () => {
    expect(() => toPixels(tensor([1, 2, 3]))).toThrow()
    expect(() => toPixels([[[1, 2]]])).toThrow()
    expect(() => toPixels([[1, 2], [3]])).toThrow()
    expect(() =>
      toPixels([
        [
          [1, 2, 3],
          [1, 2],
        ],
      ]),
    ).toThrow()
    expect(() => toPixels({ shape: [2, 2], data: [1, 2] })).toThrow()
  })

  it('builds image nodes from pixels or arrays, and groups and scenes', () => {
    const node = image(
      0,
      0,
      4,
      4,
      tensor([
        [0, 1],
        [1, 0],
      ]),
      { smoothing: true, id: 'img' },
    )
    expect(node.pixels.height).toBe(2)
    expect(node.smoothing).toBe(true)
    expect(
      group(
        [
          polygon([
            [0, 0],
            [1, 0],
            [0, 1],
          ]),
        ],
        { rotate: 45, opacity: 0.5 },
      ).rotate,
    ).toBe(45)
    expect(isScene(scene(1, 1, [], { background: 'bg' }))).toBe(true)
    expect(isScene(null)).toBe(false)
  })
})

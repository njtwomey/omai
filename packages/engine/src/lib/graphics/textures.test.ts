import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  bakeTexture,
  bricks,
  checkerboard,
  cubeUv,
  floatImage,
  imageFromPixels,
  imageToPixels,
  mipLevel,
  mipmaps,
  noiseTexture,
  sampleBilinear,
  sampleMipmap,
  sampleNearest,
  sphereUv,
  wrapIndex,
} from './index.ts'

const ramp = floatImage(8, 4, (x, y) => [x + 10 * y, 2 * x, y])

describe('textures', () => {
  it('samples texel centres exactly and blends linearly between them', () => {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 8; x++) {
        const u = (x + 0.5) / 8
        const v = (y + 0.5) / 4
        expect(sampleNearest(ramp, u, v)[0]).toBe(x + 10 * y)
        expect(sampleBilinear(ramp, u, v)[0]).toBeCloseTo(x + 10 * y, 12)
      }
    }
    // Inside the texture an affine image is reproduced exactly by bilinear filtering.
    fc.assert(
      fc.property(
        fc.double({ min: 0.0625, max: 0.9375, noNaN: true }),
        fc.double({ min: 0.125, max: 0.875, noNaN: true }),
        (u, v) => {
          const s = sampleBilinear(ramp, u, v, 'clamp')
          expect(s[0]).toBeCloseTo(u * 8 - 0.5 + 10 * (v * 4 - 0.5), 9)
          expect(s[3]).toBe(1)
        },
      ),
    )
  })

  it('wraps by tiling, clamping and mirroring', () => {
    expect([-5, -1, 0, 3, 4, 9].map((i) => wrapIndex(i, 4, 'repeat'))).toEqual([3, 3, 0, 3, 0, 1])
    expect([-5, -1, 0, 3, 4, 9].map((i) => wrapIndex(i, 4, 'clamp'))).toEqual([0, 0, 0, 3, 3, 3])
    expect([-5, -1, 0, 3, 4, 9].map((i) => wrapIndex(i, 4, 'mirror'))).toEqual([3, 0, 0, 3, 3, 1])
    expect(sampleBilinear(ramp, 0, 0.125, 'repeat')[1]).toBeCloseTo(7, 12)
    expect(sampleBilinear(ramp, 0, 0.125, 'clamp')[1]).toBeCloseTo(0, 12)
  })

  it('keeps the mean through the mipmap chain and blends levels trilinearly', () => {
    const chain = mipmaps(ramp)
    expect(chain.levels.map((l) => [l.width, l.height])).toEqual([
      [8, 4],
      [4, 2],
      [2, 1],
      [1, 1],
    ])
    const mean = (k: number): number => {
      const l = chain.levels[k]
      if (l === undefined) return Number.NaN
      let s = 0
      for (let i = 0; i < l.width * l.height; i++) s += l.rgb[3 * i] ?? 0
      return s / (l.width * l.height)
    }
    for (let k = 1; k < 4; k++) expect(mean(k)).toBeCloseTo(mean(0), 12)
    const odd = mipmaps(floatImage(3, 1, (x) => [x, 0, 0]))
    expect(odd.levels[1]?.rgb[0]).toBe(1)
    const a = sampleMipmap(chain, 0.3, 0.6, 1)
    const b = sampleMipmap(chain, 0.3, 0.6, 2)
    const mid = sampleMipmap(chain, 0.3, 0.6, 1.25)
    expect(mid[0]).toBeCloseTo(0.75 * a[0] + 0.25 * b[0], 12)
    expect(sampleMipmap(chain, 0.3, 0.6, 1.4, { filter: 'nearest' })).toEqual(a)
    expect(sampleMipmap(chain, 0.3, 0.6, 99)).toEqual(sampleMipmap(chain, 0.3, 0.6, 3))
    expect(mipLevel(0.5 / 8, 0, 0, 0.5 / 4, 8, 4)).toBe(0)
    expect(mipLevel(0, 1, 0, 0, 8, 4)).toBe(2)
  })

  it('maps directions to sphere and cube coordinates consistently', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        (u, v) => {
          const phi = 2 * Math.PI * (u - 0.5)
          const lat = Math.PI * (0.5 - v)
          const d: [number, number, number] = [
            Math.cos(lat) * Math.cos(phi),
            Math.sin(lat),
            Math.cos(lat) * Math.sin(phi),
          ]
          const [uu, vv] = sphereUv(d)
          expect(uu).toBeCloseTo(u, 9)
          expect(vv).toBeCloseTo(v, 9)
          const c = cubeUv(d)
          expect(c.uv[0]).toBeGreaterThanOrEqual(0)
          expect(c.uv[1]).toBeLessThanOrEqual(1)
        },
      ),
    )
    expect(
      [0, 1, 2, 3, 4, 5].map(
        (f) =>
          cubeUv(
            [
              [1, 0, 0],
              [-1, 0, 0],
              [0, 1, 0],
              [0, -1, 0],
              [0, 0, 1],
              [0, 0, -1],
            ][f] as [number, number, number],
          ).face,
      ),
    ).toEqual([0, 1, 2, 3, 4, 5])
    expect(() => cubeUv([0, 0, 0])).toThrow()
  })

  it('converts between float images and display pixels', () => {
    const img = floatImage(3, 1, (x) => [x / 2, x / 2, x / 2])
    const px = imageToPixels(img)
    expect(Array.from(px.rgba.slice(0, 4))).toEqual([0, 0, 0, 255])
    const back = imageFromPixels(px)
    for (let i = 0; i < 9; i++) expect(Math.abs((back.rgb[i] ?? 0) - (img.rgb[i] ?? 0))).toBeLessThan(0.005)
    expect(
      imageToPixels(
        floatImage(1, 1, () => [100, 1, 0]),
        { toneMap: 'reinhard', encoding: 'linear' },
      ).rgba[1],
    ).toBe(128)
    expect(
      imageToPixels(
        floatImage(1, 1, () => [0.25, 0, 0]),
        { exposure: 2, encoding: 2.2 },
      ).rgba[0],
    ).toBe(255)
    expect(
      imageToPixels(
        floatImage(1, 1, () => [10, 0, 0]),
        { toneMap: 'aces' },
      ).rgba[0],
    ).toBeGreaterThan(250)
    const withAlpha = imageFromPixels(
      { width: 1, height: 1, rgba: new Uint8ClampedArray([10, 20, 30, 40]) },
      { srgb: false },
    )
    expect(Array.from(imageToPixels(withAlpha, { encoding: 'linear' }).rgba)).toEqual([10, 20, 30, 40])
    expect(() => floatImage(0, 1)).toThrow()
  })

  it('draws procedural checkerboards, bricks and keyed noise', () => {
    const t = bakeTexture(8, 8, (u, v) => checkerboard(u, v, { scale: 4 }))
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        expect(t.rgb[3 * (8 * y + x)]).toBe((Math.floor(x / 2) + Math.floor(y / 2)) % 2 === 0 ? 1 : 0.1)
      }
    }
    // Odd rows are offset by half a brick, so a vertical mortar line in one row is brick in the next.
    expect(bricks(0.001, 0.125)[0]).toBe(0.8)
    expect(bricks(0.001, 0.375)[0]).toBe(0.6)
    expect(bricks(0.25, 0.375)[0]).toBe(0.8)
    const n = noiseTexture(key(1), 16, 8, { scale: 4, octaves: 2 })
    expect(noiseTexture(key(1), 16, 8, { scale: 4, octaves: 2 })).toEqual(n)
    expect(noiseTexture(key(2), 16, 8, { scale: 4, octaves: 2 })).not.toEqual(n)
    const v = noiseTexture(key(1), 4, 4, { kind: 'value', a: [1, 0, 0], b: [0, 0, 1] })
    expect(v.rgb.every((x) => x >= 0 && x <= 1)).toBe(true)
  })
})

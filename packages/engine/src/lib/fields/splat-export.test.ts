import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { gaussianEllipsoids, gaussiansFromPoints, gaussiansToPly, randomGaussians } from './index.ts'

describe('gaussianEllipsoids', () => {
  it('gives the centre, colour, opacity and axes (longest first) of each Gaussian', () => {
    // Scales 0.1, 0.4, 0.2 along the axes, rotated a quarter turn about z (x goes to y, y to -x).
    const h = Math.SQRT1_2
    const base = gaussiansFromPoints([[1, 2, 3]], [[0.2, 0.4, 0.6]], { opacity: 0.25 })
    const rotated = {
      ...base,
      logScales: fromStorage(Float64Array.from([Math.log(0.1), Math.log(0.4), Math.log(0.2)]), [1, 3], 'float64'),
      rotations: fromStorage(Float64Array.from([h, 0, 0, h]), [1, 4], 'float64'),
    }
    const [e] = gaussianEllipsoids(rotated)
    expect(e?.centre).toEqual([1, 2, 3])
    expect(e?.opacity).toBeCloseTo(0.25, 12)
    expect(e?.rgb.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([0.2, 0.4, 0.6])
    const lengths = e?.axes.map((a) => Math.hypot(...a)) as number[]
    expect(lengths[0]).toBeCloseTo(0.4, 12)
    expect(lengths[1]).toBeCloseTo(0.2, 12)
    expect(lengths[2]).toBeCloseTo(0.1, 12)
    // The longest axis was the y axis of the Gaussian, which the quarter turn sends to -x.
    expect(e?.axes[0][0]).toBeCloseTo(-0.4, 9)
    expect(Math.abs(e?.axes[0][1] as number)).toBeLessThan(1e-9)
  })
})

describe('gaussiansToPly', () => {
  it('writes the header and one 17-float record per Gaussian, with the parameters as stored', () => {
    const set = randomGaussians(key(0), 4, { opacity: 0.3, scale: 0.2 })
    const bytes = gaussiansToPly(set)
    const text = new TextDecoder('latin1').decode(bytes)
    const end = text.indexOf('end_header\n') + 'end_header\n'.length
    expect(text.startsWith('ply\nformat binary_little_endian 1.0\nelement vertex 4\n')).toBe(true)
    expect(text.match(/property float/g)?.length).toBe(17)
    expect(bytes.length).toBe(end + 4 * 17 * 4)
    const view = new DataView(bytes.buffer)
    const at = (i: number, k: number): number => view.getFloat32(end + (i * 17 + k) * 4, true)
    const mean = toFlat(set.means)
    const colour = toFlat(set.colours)
    expect(at(2, 0)).toBeCloseTo(mean[6] as number, 6)
    expect(at(2, 2)).toBeCloseTo(mean[8] as number, 6)
    expect(at(2, 6)).toBeCloseTo(colour[6] as number, 6)
    expect(at(2, 9)).toBeCloseTo(Math.log(0.3 / 0.7), 6)
    expect(at(2, 10)).toBeCloseTo(Math.log(0.2), 6)
    // The identity rotation (w, x, y, z) = (1, 0, 0, 0).
    expect([at(2, 13), at(2, 14), at(2, 15), at(2, 16)]).toEqual([1, 0, 0, 0])
  })
})

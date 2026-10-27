import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const image = (h: number, w: number, c: number): fc.Arbitrary<number[][][]> =>
  fc.array(
    fc.array(fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: c, maxLength: c }), {
      minLength: w,
      maxLength: w,
    }),
    { minLength: h, maxLength: h },
  )

const close = (a: readonly number[], b: readonly number[], tol: number): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] ?? 0))).toBeLessThanOrEqual(tol)
}

describe('colour spaces', () => {
  it('round-trip through HSV, XYZ, Lab and YCrCb', () => {
    fc.assert(
      fc.property(image(3, 4, 3), (x) => {
        const flat = x.flat(2)
        close(toFlat(V.hsvToRgb(V.rgbToHsv(x))), flat, 1e-12)
        // OpenCV's XYZ matrices are rounded inverses of each other (to six digits).
        close(toFlat(V.xyzToRgb(V.rgbToXyz(x))), flat, 1e-4)
        close(toFlat(V.labToRgb(V.rgbToLab(x))), flat, 1e-4)
        close(toFlat(V.ycrcbToRgb(V.rgbToYcrcb(x))), flat, 2e-3)
      }),
      { numRuns: RUNS },
    )
  })

  it('keeps grey grey and orders the channels of HSV', () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (v) => {
        const g = V.greyToRgb([[v]])
        expect(toFlat(V.rgbToGrey(g))[0]).toBeCloseTo(v, 12)
        expect(toFlat(V.rgbToGrey(g, { standard: '709' }))[0]).toBeCloseTo(v, 12)
        expect(toFlat(V.rgbToHsv(g))).toEqual([0, 0, v])
        const lab = toFlat(V.rgbToLab(g))
        expect(Math.abs(lab[1] ?? 1)).toBeLessThan(1e-3)
        expect(Math.abs(lab[2] ?? 1)).toBeLessThan(1e-3)
      }),
      { numRuns: RUNS },
    )
  })

  it('dispatches cvtColor by name and rejects unknown codes', () => {
    const x = [[[0.1, 0.5, 0.9]]]
    expect(toFlat(V.cvtColor(x, 'RGB2HSV'))).toEqual(toFlat(V.rgbToHsv(x)))
    expect(() => V.cvtColor(x, 'BGR2LUV' as V.ColourConversion)).toThrow()
    expect(() => V.rgbToGrey([[0.5]])).toThrow()
  })
})

describe('colour balance', () => {
  it('grey world equalises the channel means; white patch maps the brightest value to 1', () => {
    fc.assert(
      fc.property(image(4, 4, 3), (x) => {
        const flat = x.flat(2)
        if (flat.some((v) => v < 0.05)) return
        const g = toFlat(V.whiteBalance(x, { method: 'grey-world' }))
        const means = [0, 1, 2].map((k) => g.filter((_, i) => i % 3 === k).reduce((a, b) => a + b, 0) / 16)
        // Clipping at 1 can only lower a mean.
        const target = [0, 1, 2].map((k) => flat.filter((_, i) => i % 3 === k).reduce((a, b) => a + b, 0) / 16)
        const grey = target.reduce((a, b) => a + b, 0) / 3
        for (const m of means) expect(m).toBeLessThanOrEqual(grey + 1e-12)
        const w = toFlat(V.whiteBalance(x))
        for (let k = 0; k < 3; k++) expect(Math.max(...w.filter((_, i) => i % 3 === k))).toBeCloseTo(1, 12)
      }),
      { numRuns: RUNS },
    )
  })

  it('splits and merges channels, and applies the transfer curves', () => {
    const x = [
      [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
      ],
    ]
    expect(toFlat(V.mergeChannels(V.splitChannels(x)))).toEqual(x.flat(2))
    close(toFlat(V.linearToSrgb(V.srgbToLinear(x))), x.flat(2), 1e-12)
    expect(toFlat(V.adjustGamma([0, 0.25, 1], 2))).toEqual([0, 0.0625, 1])
    expect(() => V.adjustGamma([2], 1)).toThrow()
  })
})

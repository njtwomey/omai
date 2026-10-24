import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, toFlat, type Value } from '../tensor/index.ts'
import {
  acesFilm,
  exposure,
  gammaDecode,
  gammaEncode,
  linearToSrgb,
  luminance,
  reinhard,
  srgbToLinear,
} from './index.ts'

describe('tone mapping and gamma', () => {
  const xs = Array.from({ length: 41 }, (_, i) => i / 40)

  it('round-trips sRGB and power-law gamma, and joins the sRGB pieces continuously', () => {
    const back = toFlat(srgbToLinear(linearToSrgb(xs)))
    for (const [i, x] of xs.entries()) expect(back[i]).toBeCloseTo(x, 14)
    const g = toFlat(gammaDecode(gammaEncode(xs, 2.4), 2.4))
    for (const [i, x] of xs.entries()) expect(g[i]).toBeCloseTo(x, 14)
    // The standard's rounded constants leave a step of about 3e-8 where the pieces meet.
    expect(item(linearToSrgb(0.0031308 + 1e-12))).toBeCloseTo(item(linearToSrgb(0.0031308)), 7)
    expect(item(srgbToLinear(0.04045 + 1e-12))).toBeCloseTo(0.04045 / 12.92, 7)
  })

  it('maps radiance monotonically into [0, 1]', () => {
    const big = xs.map((x) => 20 * x)
    for (const op of [(v: number[]) => reinhard(v), (v: number[]) => acesFilm(v), (v: number[]) => reinhard(v, 20)]) {
      const y = toFlat(op(big))
      for (let i = 1; i < y.length; i++) expect(y[i] ?? 0).toBeGreaterThanOrEqual(y[i - 1] ?? 0)
      expect(y.reduce((m, x) => Math.min(m, x), Infinity)).toBeGreaterThanOrEqual(0)
      expect(y.reduce((m, x) => Math.max(m, x), -Infinity)).toBeLessThanOrEqual(1 + 1e-15)
    }
    expect(item(reinhard(20, 20))).toBeCloseTo(1, 15)
    expect(item(exposure(3, -1))).toBe(1.5)
    expect(
      toFlat(
        luminance([
          [1, 0, 0],
          [0, 0, 1],
        ]),
      ),
    ).toEqual([0.2126, 0.0722])
  })

  for (const [name, f, input] of [
    ['reinhard', (x: Value) => reinhard(x), [0.2, 1.5, 4]],
    ['reinhard extended', (x: Value) => reinhard(x, 3), [0.2, 1.5, 4]],
    ['acesFilm', (x: Value) => acesFilm(x), [0.05, 0.3, 0.6]],
    ['linearToSrgb', (x: Value) => linearToSrgb(x), [0.001, 0.2, 0.9]],
    ['srgbToLinear', (x: Value) => srgbToLinear(x), [0.01, 0.3, 0.9]],
    ['gammaEncode', (x: Value) => gammaEncode(x), [0.1, 0.5, 0.9]],
    ['exposure', (x: Value) => exposure(x, 0.5), [0.1, 0.5, 0.9]],
  ] as const) {
    it(`${name} passes checkPrimitive`, () => {
      expect(checkPrimitive(f, [input]).failures).toEqual([])
    })
  }
})

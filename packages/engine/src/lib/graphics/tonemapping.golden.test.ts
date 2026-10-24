import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
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

describe('tone mapping and gamma against their closed forms', () => {
  for (const c of loadFixture('graphics/tonemapping').cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      const p = c.params
      const ops: Record<string, () => Value> = {
        linearToSrgb: () => linearToSrgb(x),
        srgbToLinear: () => srgbToLinear(x),
        gammaEncode: () => gammaEncode(x, p['gamma'] as number),
        gammaDecode: () => gammaDecode(x, p['gamma'] as number),
        reinhard: () => reinhard(x),
        reinhardExtended: () => reinhard(x, p['white'] as number),
        acesFilm: () => acesFilm(x),
        exposure: () => exposure(x, p['ev'] as number),
        luminance: () => luminance(x),
      }
      const op = ops[c.fn]
      if (op === undefined) throw new Error(`unknown case ${c.fn}`)
      expectMatches(op(), c.output, c.rtol, c.atol, c.name)
    })
  }
})

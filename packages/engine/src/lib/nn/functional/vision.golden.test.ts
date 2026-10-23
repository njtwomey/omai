import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import { affineGrid, gridSample } from './sampling.ts'
import { type InterpolationMode, interpolate, type PadModeTorch, pad, pixelShuffle, pixelUnshuffle } from './vision.ts'

describe('vision ops against torch', () => {
  for (const c of loadFixture('nn/functional/vision').cases) {
    it(c.name, () => {
      const x = decode(c.inputs[0]) as Tensor
      const p = c.params
      let y: Value
      if (c.fn === 'interpolate') {
        y = interpolate(x, {
          mode: p['mode'] as InterpolationMode,
          ...(p['size'] === null ? {} : { size: p['size'] as number[] }),
          ...(p['scaleFactor'] === null ? {} : { scaleFactor: p['scaleFactor'] as number }),
          ...(p['alignCorners'] === null ? {} : { alignCorners: p['alignCorners'] as boolean }),
        })
      } else if (c.fn === 'pixelShuffle') y = pixelShuffle(x, p['factor'] as number)
      else if (c.fn === 'pixelUnshuffle') y = pixelUnshuffle(x, p['factor'] as number)
      else y = pad(x, p['widths'] as number[], p['mode'] as PadModeTorch, p['value'] as number)
      expectMatches(y, c.output, 1e-12, 1e-13, c.name)
    })
  }
})

describe('spatial transformer ops against torch', () => {
  for (const c of loadFixture('nn/functional/sampling').cases) {
    it(c.name, () => {
      const [a, b] = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const y =
        c.fn === 'affineGrid'
          ? affineGrid(a as Tensor, p['size'] as number[], p['alignCorners'] as boolean)
          : gridSample(a as Tensor, b as Tensor, {
              mode: p['mode'] as 'bilinear' | 'nearest' | 'bicubic',
              paddingMode: p['paddingMode'] as 'zeros' | 'border' | 'reflection',
              alignCorners: p['alignCorners'] as boolean,
            })
      expectMatches(y, c.output, 1e-12, 1e-12, c.name)
    })
  }
})

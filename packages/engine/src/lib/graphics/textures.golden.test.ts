import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import { bricks, checkerboard, type FloatImage, mipmaps, sampleBilinear, sampleNearest, type Wrap } from './index.ts'

const image = (x: unknown): FloatImage => {
  const t = decode(x) as Tensor
  const [h = 0, w = 0] = t.shape
  return { width: w, height: h, rgb: Float64Array.from(t.data) }
}
const flat = (data: readonly number[], shape: readonly number[]): Tensor =>
  decode({ dtype: 'float64', shape, data }) as Tensor

describe('textures against scipy.ndimage', () => {
  for (const c of loadFixture('graphics/textures').cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'nearest':
        case 'bilinear': {
          const tex = image(c.inputs[0])
          const uv = toArray(decode(c.inputs[1]) as Tensor) as number[][]
          const f = c.fn === 'nearest' ? sampleNearest : sampleBilinear
          const rows = uv.map(([u = 0, v = 0]) => f(tex, u, v, c.params['wrap'] as Wrap).slice(0, 3))
          expectMatches(flat(rows.flat(), [rows.length, 3]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'mipmaps': {
          const levels = mipmaps(image(c.inputs[0])).levels.flatMap((l) => Array.from(l.rgb))
          expectMatches(flat(levels, [levels.length]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'checkerboard':
        case 'bricks': {
          const uv = toArray(decode(c.inputs[0]) as Tensor) as number[][]
          const values = uv.map(([u = 0, v = 0]) => (c.fn === 'bricks' ? bricks(u, v) : checkerboard(u, v))[0])
          expectMatches(flat(values, [values.length]), c.output, c.rtol, c.atol, c.name)
          return
        }
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})

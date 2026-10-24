import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import { fromAscii } from '../worlds/index.ts'
import { type FloatImage, renderRaycast } from './index.ts'

const texture = (x: unknown): FloatImage => {
  const t = decode(x) as Tensor
  return { width: t.shape[1] ?? 0, height: t.shape[0] ?? 0, rgb: Float64Array.from(t.data) }
}

describe('ray-cast frames against an independent brute-force reference', () => {
  for (const c of loadFixture('graphics/raycasting').cases) {
    it(c.name, () => {
      const p = c.params
      const w = fromAscii(p['rows'] as string[])
      const pose = { x: p['x'] as number, y: p['y'] as number, heading: p['heading'] as number }
      const size = { width: p['width'] as number, height: p['height'] as number, fov: p['fov'] as number }
      const common = { ...size, floor: [0.3, 0.3, 0.3] as const, ceiling: [0.5, 0.6, 0.9] as const }
      const flat = { wall: [0.8, 0.7, 0.6] as const, forest: [0.1, 0.6, 0.2] as const }
      const frame =
        c.name === 'textured'
          ? renderRaycast(w, pose, {
              ...common,
              walls: { wall: texture(c.inputs[0]), forest: texture(c.inputs[1]) },
              floor: texture(c.inputs[2]),
              fog: { colour: [1, 1, 1], density: 0.1 },
            })
          : renderRaycast(w, pose, {
              ...common,
              walls: flat,
              ...(c.name === 'sprite'
                ? {
                    sprites: [
                      {
                        x: (p['sprite'] as number[])[0] ?? 0,
                        y: (p['sprite'] as number[])[1] ?? 0,
                        colour: [1, 0.8, 0.2] as const,
                      },
                    ],
                  }
                : {}),
            })
      const got = decode({
        dtype: 'float64',
        shape: [size.height, size.width, 3],
        data: Array.from(frame.image.rgb),
      }) as Tensor
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
      const depth = p['depth'] as number[] | undefined
      if (depth !== undefined) for (const [i, d] of depth.entries()) expect(frame.depth[i]).toBeCloseTo(d, 10)
      if (c.name === 'sprite') expect(frame.sprites[0]?.visible).toBeGreaterThan(0)
    })
  }
})

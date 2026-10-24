import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Vec3 } from '../scene/index.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import { mesh, pinholeCamera, rasteriseMesh } from './index.ts'

describe('rasterisation against an independent per-pixel reference', () => {
  for (const c of loadFixture('graphics/rasterise').cases) {
    it(c.name, () => {
      const positions = toArray(decode(c.inputs[0]) as Tensor) as unknown as Vec3[]
      const colours = toArray(decode(c.inputs[1]) as Tensor) as unknown as Vec3[]
      const triangles = Array.from({ length: positions.length / 3 }, (_, k) => [3 * k, 3 * k + 1, 3 * k + 2] as const)
      const cam = pinholeCamera({ eye: [0.5, 1, 4], target: [0, 0, 0], fov: 50, near: 0.1, far: 50 })
      const r = rasteriseMesh(mesh(positions, triangles, { colours }), cam, { width: 16, height: 12 })
      expect(Array.from(r.triangleIds)).toEqual(c.params['ids'])
      for (const [i, d] of (c.params['depth'] as (number | null)[]).entries()) {
        if (d === null) expect(r.depth[i]).toBe(Number.POSITIVE_INFINITY)
        else expect(r.depth[i]).toBeCloseTo(d, 10)
      }
      expect(r.triangles[3]?.culled).toBe(true)
      const got = decode({ dtype: 'float64', shape: [12, 16, 3], data: Array.from(r.image.rgb) }) as Tensor
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

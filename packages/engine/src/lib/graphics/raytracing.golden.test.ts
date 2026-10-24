import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  directionalLight,
  material,
  pinholeCamera,
  plane,
  pointLight,
  rayScene,
  renderWhitted,
  sphere,
} from './index.ts'

// The scene of oracle/cases/graphics/raytracing.py.
const scene = rayScene(
  [
    { shape: plane([0, 0, 0], [0, 1, 0]), material: material({ albedo: [0.6, 0.6, 0.6] }) },
    {
      shape: sphere([0, 1, 0], 1),
      material: material({
        model: 'blinn-phong',
        albedo: [0.7, 0.2, 0.2],
        specular: [0.5, 0.5, 0.5],
        shininess: 20,
        mirror: 0.3,
      }),
    },
    { shape: sphere([2.2, 0.7, -0.5], 0.7), material: material({ albedo: [0.2, 0.3, 0.8] }) },
  ],
  {
    lights: [pointLight([3, 5, 4], [60, 60, 60]), directionalLight([-1, -2, -1], [0.5, 0.5, 0.5])],
    background: [0.2, 0.3, 0.5],
    ambient: [0.05, 0.05, 0.05],
  },
)

describe('Whitted ray tracing against an independent reference render', () => {
  for (const c of loadFixture('graphics/raytracing').cases) {
    it(c.name, () => {
      const cam = pinholeCamera({ eye: [0, 2, 6], target: [0.5, 0.8, 0], fov: 50, near: 0.1, far: 100 })
      const { image } = renderWhitted(scene, cam, 16, 12, { maxDepth: 3 })
      const got = decode({ dtype: 'float64', shape: [12, 16, 3], data: Array.from(image.rgb) }) as Tensor
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})

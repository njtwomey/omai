/**
 * Classic test scenes for the ray and path tracers, with the cameras that frame them: the Cornell box (a white room with
 * a red wall on the left, a green wall on the right and a square lamp in the ceiling) holding either a mirror sphere
 * and a glass sphere, as in smallpt, or two white blocks, as in the original photographs.
 */

import { MdmlError } from '../base/index.ts'
import type { Vec3 } from '../scene/index.ts'
import { type PinholeCamera, pinholeCamera } from './camera.ts'
import { box, sphere } from './intersect.ts'
import { areaLight } from './lights.ts'
import { type RayScene, rayScene, type SceneObject } from './raytracing.ts'
import { material } from './shading.ts'

/** Options of {@link cornellBox}. */
export interface CornellBoxOptions {
  /** What stands on the floor (default `'spheres'`: a mirror and a glass sphere; `'blocks'`: two white blocks). */
  readonly contents?: 'spheres' | 'blocks'
  /** The lamp's radiance (default 12). */
  readonly lamp?: number
  /** Light samples per side of the lamp for each shading point (default 1: one jittered sample). */
  readonly lightSamples?: number
}

// The room spans x in [-1, 1], y in [0, 2], z in [-1, 1]; walls are thin boxes so light cannot leak round them, and the
// front stays open to the camera.
const WALL = 0.02
const LAMP = 0.25

/**
 * The Cornell box: a closed white room, open at the front, with a red left wall, a green right wall and a square lamp
 * in the ceiling (an area light for next-event estimation, with an emissive panel that the camera and specular paths
 * see), holding a mirror sphere and a glass sphere or two white blocks; and a pinhole camera that frames the opening.
 *
 * @param options - The contents, the lamp's radiance and the light samples.
 * @returns The scene and its camera.
 * @throws MdmlError with code `value` for a non-positive lamp radiance.
 * @concept path-tracing
 * @example
 * ```ts
 * import { cornellBox } from 'mdml-engine/graphics'
 * const { scene, camera } = cornellBox()
 * [scene.objects.length, scene.lights.length, camera.eye] // => [8, 1, [0, 1, 3.4]]
 * ```
 */
export function cornellBox(options: CornellBoxOptions = {}): {
  readonly scene: RayScene
  readonly camera: PinholeCamera
} {
  const lamp = options.lamp ?? 12
  if (!(lamp > 0 && Number.isFinite(lamp)))
    throw new MdmlError('value', 'cornellBox: the lamp radiance must be positive')
  const white = material({ albedo: [0.75, 0.75, 0.75] })
  const red = material({ albedo: [0.75, 0.2, 0.2] })
  const green = material({ albedo: [0.2, 0.75, 0.2] })
  const radiance: Vec3 = [lamp, lamp, lamp]
  const objects: SceneObject[] = [
    { name: 'floor', shape: box([-1, -WALL, -1 - WALL], [1, 0, 1]), material: white },
    { name: 'ceiling', shape: box([-1, 2, -1 - WALL], [1, 2 + WALL, 1]), material: white },
    { name: 'back wall', shape: box([-1, 0, -1 - WALL], [1, 2, -1]), material: white },
    { name: 'left wall', shape: box([-1 - WALL, -WALL, -1 - WALL], [-1, 2 + WALL, 1]), material: red },
    { name: 'right wall', shape: box([1, -WALL, -1 - WALL], [1 + WALL, 2 + WALL, 1]), material: green },
    {
      name: 'lamp',
      shape: box([-LAMP, 2 - WALL, -LAMP], [LAMP, 2, LAMP]),
      material: material({ albedo: [0, 0, 0], emission: radiance }),
      lightSurface: true,
    },
  ]
  if ((options.contents ?? 'spheres') === 'spheres') {
    objects.push(
      { name: 'mirror sphere', shape: sphere([-0.45, 0.35, -0.4], 0.35), material: material({ mirror: 1 }) },
      {
        name: 'glass sphere',
        shape: sphere([0.45, 0.35, 0.3], 0.35),
        material: material({ albedo: [1, 1, 1], transparency: 1, ior: 1.5 }),
      },
    )
  } else {
    objects.push(
      { name: 'tall block', shape: box([-0.65, 0, -0.65], [-0.05, 1.2, -0.05]), material: white },
      { name: 'short block', shape: box([0.1, 0, -0.05], [0.7, 0.6, 0.55]), material: white },
    )
  }
  // The area light sits just below the lamp panel, facing down (edgeU x edgeV points to -y).
  const light = areaLight([-LAMP, 2 - WALL - 1e-3, -LAMP], [2 * LAMP, 0, 0], [0, 0, 2 * LAMP], radiance, {
    samples: options.lightSamples ?? 1,
  })
  const scene = rayScene(objects, { lights: [light] })
  const camera = pinholeCamera({ eye: [0, 1, 3.4], target: [0, 1, 0], up: [0, 1, 0], fov: 40 })
  return { scene, camera }
}

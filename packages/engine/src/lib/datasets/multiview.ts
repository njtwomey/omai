/**
 * Multi-view images of a small scene, the data of novel view synthesis (NeRF, Gaussian splatting): a few coloured
 * spheres and a box, lit by one directional light with shadows, rendered from cameras placed on a hemisphere around
 * them at a low resolution. The truth is the scene itself, so any view, including the held-out ones that come with the
 * data, can be rendered exactly (`renderMultiView`) and a fitted model judged on views it never saw.
 */

import type { Vec3 } from '../base/index.ts'
import { type DatasetGenerator, type GeneratorKnobs, type ImageData, images, type Truth } from '../data/index.ts'
import {
  box,
  material,
  type PinholeCamera,
  pinholeCamera,
  pixelRay,
  type RayScene,
  rayScene,
  type SceneObject,
  sceneHit,
  sceneOccluded,
  sphere,
} from '../graphics/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { stamp } from './_util.ts'

/** A coloured sphere of a multi-view scene. */
export interface SceneSphere {
  readonly centre: Vec3
  readonly radius: number
  readonly albedo: Vec3
}

/** A coloured axis-aligned box of a multi-view scene. */
export interface SceneBox {
  readonly min: Vec3
  readonly max: Vec3
  readonly albedo: Vec3
}

/** A multi-view scene: its objects, the direction towards the one light, the ambient share and the background colour. */
export interface MultiViewScene {
  readonly spheres: readonly SceneSphere[]
  readonly boxes: readonly SceneBox[]
  /** The unit direction from the scene towards the light. */
  readonly light: Vec3
  /** The share of the albedo that is seen with no light (0 to 1). */
  readonly ambient: number
  readonly background: Vec3
}

/** The objects of the scene, in the order the `objects` knob takes them. */
const OBJECTS: readonly { readonly sphere?: SceneSphere; readonly box?: SceneBox }[] = [
  { sphere: { centre: [-0.45, 0, 0], radius: 0.45, albedo: [0.9, 0.2, 0.15] } },
  { sphere: { centre: [0.45, 0.1, 0.2], radius: 0.4, albedo: [0.2, 0.75, 0.3] } },
  { box: { min: [-0.3, -0.75, -0.7], max: [0.35, -0.2, -0.15], albedo: [0.2, 0.35, 0.9] } },
  { sphere: { centre: [0, 0.55, -0.45], radius: 0.28, albedo: [0.95, 0.85, 0.2] } },
]

const norm = (v: Vec3): Vec3 => {
  const n = Math.hypot(...v)
  return [v[0] / n, v[1] / n, v[2] / n]
}

/**
 * The scene of the first `objects` objects (red sphere, green sphere, blue box, yellow sphere), lit from above and the
 * side, on a white background.
 *
 * @param objects - How many objects, 1 to 4 (default 3).
 * @returns The scene.
 * @concept novel-view-synthesis
 * @example
 * ```ts
 * import { multiViewScene } from 'omai-engine/datasets'
 * [multiViewScene(3).spheres.length, multiViewScene(3).boxes.length] // => [2, 1]
 * ```
 */
export function multiViewScene(objects = 3): MultiViewScene {
  const chosen = OBJECTS.slice(0, Math.max(1, Math.min(OBJECTS.length, objects)))
  return {
    spheres: chosen.flatMap((o) => (o.sphere === undefined ? [] : [o.sphere])),
    boxes: chosen.flatMap((o) => (o.box === undefined ? [] : [o.box])),
    light: norm([0.5, 0.8, 0.4]),
    ambient: 0.35,
    background: [1, 1, 1],
  }
}

/** The scene as a ray-traced scene of the graphics library (its objects in a bounding-volume hierarchy). */
function traced(scene: MultiViewScene): RayScene {
  const objects: SceneObject[] = [
    ...scene.spheres.map((s) => ({ shape: sphere(s.centre, s.radius), material: material({ albedo: s.albedo }) })),
    ...scene.boxes.map((b) => ({ shape: box(b.min, b.max), material: material({ albedo: b.albedo }) })),
  ]
  return rayScene(objects, { background: scene.background })
}

/**
 * Render a view of a multi-view scene exactly: one ray through every pixel centre, the nearest hit shaded by Lambert's
 * law with one shadow ray towards the light, the background where nothing is hit. Pixel `(i, j)` has its centre at
 * `(i + 0.5, j + 0.5)`, as in `cameraRays`, which is also where a model renders its rays.
 *
 * @param scene - The scene.
 * @param camera - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @returns The image `[height, width, 3]` with values in [0, 1].
 * @concept novel-view-synthesis
 * @example
 * ```ts
 * import { renderMultiView, multiViewScene } from 'omai-engine/datasets'
 * import { pinholeCamera } from 'omai-engine/graphics'
 * renderMultiView(multiViewScene(3), pinholeCamera({ eye: [0, 0, 4] }), 8, 6).shape // => [6, 8, 3]
 * ```
 */
export function renderMultiView(scene: MultiViewScene, camera: PinholeCamera, width: number, height: number): Tensor {
  const world = traced(scene)
  const out = new Float64Array(width * height * 3)
  for (let j = 0; j < height; j++)
    for (let i = 0; i < width; i++) {
      const ray = pixelRay(camera, width, height, i + 0.5, j + 0.5)
      const hit = sceneHit(world, ray)
      let rgb: Vec3 = scene.background
      if (hit !== null) {
        const albedo = world.objects[hit.object]?.material.albedo ?? scene.background
        const lit = Math.max(
          0,
          hit.normal[0] * scene.light[0] + hit.normal[1] * scene.light[1] + hit.normal[2] * scene.light[2],
        )
        const origin: Vec3 = [
          hit.point[0] + 1e-6 * hit.normal[0],
          hit.point[1] + 1e-6 * hit.normal[1],
          hit.point[2] + 1e-6 * hit.normal[2],
        ]
        const shadowed = lit > 0 && sceneOccluded(world, { origin, direction: scene.light }, 100)
        const shade = scene.ambient + (1 - scene.ambient) * (shadowed ? 0 : lit)
        rgb = [albedo[0] * shade, albedo[1] * shade, albedo[2] * shade]
      }
      out.set(rgb, 3 * (j * width + i))
    }
  return fromStorage(out, [height, width, 3], 'float64')
}

/**
 * A camera on a sphere of the given radius around the origin, looking at it, with y up: the eye is at
 * `radius (cos(el) cos(az), sin(el), cos(el) sin(az))` for `elevation` and `azimuth` in degrees. The scene's cameras are
 * placed this way, so a camera made here with another angle is a novel view of the same scene.
 *
 * @param radius - The distance from the origin.
 * @param elevation - Degrees above the horizontal plane.
 * @param azimuth - Degrees about the vertical axis, from the x axis towards z.
 * @param fov - Vertical field of view in degrees.
 * @param near - Near clipping distance.
 * @param far - Far clipping distance.
 * @returns The camera.
 * @concept novel-view-synthesis
 * @example
 * ```ts
 * import { orbitCamera } from 'omai-engine/datasets'
 * orbitCamera(4, 0, 90, 40, 0.1, 100).eye.map((v) => Math.round(v * 1e9) / 1e9 + 0) // => [0, 0, 4]
 * ```
 */
export function orbitCamera(
  radius: number,
  elevation: number,
  azimuth: number,
  fov: number,
  near: number,
  far: number,
): PinholeCamera {
  const el = (elevation * Math.PI) / 180
  const az = (azimuth * Math.PI) / 180
  return pinholeCamera({
    eye: [radius * Math.cos(el) * Math.cos(az), radius * Math.sin(el), radius * Math.cos(el) * Math.sin(az)],
    target: [0, 0, 0],
    up: [0, 1, 0],
    fov,
    near,
    far,
  })
}

/**
 * Cameras on a hemisphere around the origin, all looking at it: azimuths evenly spaced around the circle (with a small
 * jitter from the key) and elevations drawn between `minElevation` and `maxElevation` degrees.
 *
 * @param key - The key.
 * @param count - The number of cameras.
 * @param options - Orbit radius, field of view (degrees), clipping distances, the elevation range and the first azimuth.
 * @returns The cameras.
 * @concept novel-view-synthesis
 * @example
 * ```ts
 * import { hemisphereCameras } from 'omai-engine/datasets'
 * import { key } from 'omai-engine/random'
 * const cams = hemisphereCameras(key(0), 6)
 * [cams.length, cams.every((c) => Math.abs(Math.hypot(...c.eye) - 4) < 1e-9)] // => [6, true]
 * ```
 */
export function hemisphereCameras(
  key: Key,
  count: number,
  options: {
    readonly radius?: number
    readonly fov?: number
    readonly near?: number
    readonly far?: number
    readonly minElevation?: number
    readonly maxElevation?: number
    readonly azimuthOffset?: number
  } = {},
): PinholeCamera[] {
  const radius = options.radius ?? 4
  const [lo, hi] = [options.minElevation ?? 20, options.maxElevation ?? 65]
  const u = uniform(key, [2 * count]).data
  return Array.from({ length: count }, (_, i) => {
    const azimuth = (options.azimuthOffset ?? 0) + (360 * (i + 0.2 * ((u[2 * i] as number) - 0.5))) / count
    const elevation = lo + (hi - lo) * (u[2 * i + 1] as number)
    return orbitCamera(radius, elevation, azimuth, options.fov ?? 40, options.near ?? 0.1, options.far ?? 100)
  })
}

/**
 * Views of a small scene from cameras on a hemisphere, for novel view synthesis: `views` training images and `heldOut`
 * further views at the azimuths between them, each `resolution` pixels square, all rendered exactly. The data are the
 * training images (`[views, h, w, 3]`); the truth carries the scene, the cameras, the sampling interval `near` to `far`
 * a model should use, and the held-out cameras with their images, in `truth.extra`.
 *
 * @concept novel-view-synthesis
 * @index dataset Posed multi-view images of a small scene, with held-out views.
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { multiViewImages } from 'omai-engine/datasets'
 * const g = generate(multiViewImages, { knobs: { views: 4, resolution: 8, heldOut: 2 } })
 * [g.data.images.shape, (g.truth.extra?.['cameras'] as unknown[]).length] // => [[4, 8, 8, 3], 4]
 * ```
 */
export const multiViewImages: DatasetGenerator<ImageData> = {
  name: 'multiViewImages',
  kind: 'image',
  description:
    'Posed views of a small lit scene from a hemisphere of cameras, with the scene as truth, for novel views.',
  knobs: dict({
    views: int(1, 200, { label: 'views', description: 'The number of training views.', default: 12 }),
    heldOut: int(0, 50, {
      label: 'held-out views',
      description: 'Further views, between the training ones.',
      default: 3,
    }),
    resolution: int(2, 128, {
      label: 'resolution',
      description: 'The side of each square image in pixels.',
      default: 16,
    }),
    objects: int(1, 4, { label: 'objects', description: 'The number of objects in the scene.', default: 3 }),
    fov: real(10, 120, { label: 'field of view', description: 'The vertical field of view in degrees.', default: 40 }),
    radius: real(2.5, 20, {
      label: 'orbit radius',
      description: 'The distance of the cameras from the scene’s centre.',
      default: 4,
    }),
  }),
  generate: (k: Key, knobs: GeneratorKnobs) => {
    const views = knobs['views'] as number
    const heldOut = knobs['heldOut'] as number
    const side = knobs['resolution'] as number
    const radius = knobs['radius'] as number
    const fov = knobs['fov'] as number
    const scene = multiViewScene(knobs['objects'] as number)
    const near = Math.max(0.1, radius - 2)
    const far = radius + 2
    const cameras = hemisphereCameras(foldIn(k, 0), views, { radius, fov, near: 0.1 })
    const held = hemisphereCameras(foldIn(k, 1), heldOut, {
      radius,
      fov,
      near: 0.1,
      azimuthOffset: 180 / Math.max(1, views),
    })
    const render = (cams: readonly PinholeCamera[]): Float64Array => {
      const all = new Float64Array(cams.length * side * side * 3)
      for (const [v, c] of cams.entries()) all.set(renderMultiView(scene, c, side, side).data, v * side * side * 3)
      return all
    }
    const data = images(fromStorage(render(cameras), [views, side, side, 3], 'float64'), {
      meta: stamp('multiViewImages', multiViewImages.description, knobs),
    })
    const truth: Truth = {
      extra: {
        scene,
        cameras,
        near,
        far,
        background: scene.background,
        resolution: side,
        heldOut: { cameras: held, images: fromStorage(render(held), [heldOut, side, side, 3], 'float64') },
        renderView: (camera: PinholeCamera, width: number, height: number): Tensor =>
          renderMultiView(scene, camera, width, height),
      },
    }
    return { data, truth }
  },
}

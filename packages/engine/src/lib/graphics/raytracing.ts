/**
 * Whitted ray tracing: following rays from the camera into a scene and spawning reflection, refraction and shadow rays
 * at each hit. The ray-traced scene (objects in a bounding-volume hierarchy, lights and background), Snell's law and
 * Schlick's Fresnel term, a renderer as a traced algorithm with one step per ray generation, and the full ray tree of a
 * single ray for figures.
 */

import { type Json, OmaiError, type Vec3 } from '../base/index.ts'
import { dotProduct } from '../foundations/index.ts'
import type { Key } from '../random/index.ts'
import {
  add,
  div,
  expandDims,
  greater,
  maximum,
  mul,
  neg,
  promoteFloat,
  sqrt,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { check3, dot3, madd3, mul3, normalise3, reflect3, refract3, scale3 } from './_vec3.ts'
import { type Bvh, type BvhOptions, buildBvh, intersectBvh, occludedBvh } from './bvh.ts'
import { cameraRays, type Ray, type RenderCamera } from './camera.ts'
import { type Hit, intersect, RAY_EPSILON, type RayShape } from './intersect.ts'
import { type Light, type LightSample, shadePoints } from './lights.ts'
import { fresnelSchlick, type Material } from './shading.ts'
import { type FloatImage, sampleBilinear } from './textures.ts'

/**
 * An object of a ray-traced scene: a shape, its material, and optionally an albedo map (a texture sampled at the hit's
 * surface coordinates, or a procedural function of them) that replaces the material's albedo.
 *
 * @concept ray-tracing
 */
export interface SceneObject {
  readonly shape: RayShape
  readonly material: Material
  readonly albedoMap?: FloatImage | ((u: number, v: number) => Vec3)
  readonly name?: string
  /**
   * The object is the visible surface of one of the scene's lights (a lamp panel over an area light). The path tracer
   * then counts its emission only on camera and specular paths: after a diffuse bounce the light's next-event estimate
   * has already counted it.
   */
  readonly lightSurface?: boolean
}

/**
 * A scene for ray and path tracing: objects (bounded ones in a bounding-volume hierarchy, planes tested separately),
 * lights, the background radiance seen by rays that escape, and an ambient term.
 *
 * @concept ray-tracing
 */
export interface RayScene {
  readonly objects: readonly SceneObject[]
  readonly lights: readonly Light[]
  readonly background: Vec3
  readonly ambient: Vec3
  readonly bvh: Bvh | null
  /** The object of each shape of the hierarchy. */
  readonly bvhObjects: readonly number[]
  /** Objects that are planes (unbounded). */
  readonly planes: readonly number[]
}

/**
 * Assemble a scene: build the hierarchy over the bounded objects. Defaults: no lights, a black background and no
 * ambient light.
 *
 * @param objects - The objects.
 * @param options - Lights, background, ambient radiance and hierarchy options.
 * @returns The scene.
 * @concept ray-tracing
 * @example
 * ```ts
 * import { material, plane, rayScene, sphere } from 'omai-engine/graphics'
 * const s = rayScene([{ shape: sphere([0, 1, 0], 1), material: material() }, { shape: plane([0, 0, 0], [0, 1, 0]), material: material() }])
 * [s.bvhObjects, s.planes] // => [[0], [1]]
 * ```
 */
export function rayScene(
  objects: readonly SceneObject[],
  options: {
    readonly lights?: readonly Light[]
    readonly background?: Vec3
    readonly ambient?: Vec3
    readonly bvh?: BvhOptions
  } = {},
): RayScene {
  const bvhObjects: number[] = []
  const planes: number[] = []
  for (const [i, o] of objects.entries()) (o.shape.kind === 'plane' ? planes : bvhObjects).push(i)
  const shapes = bvhObjects.map((i) => (objects[i] as SceneObject).shape)
  return {
    objects,
    lights: options.lights ?? [],
    background: options.background ?? [0, 0, 0],
    ambient: options.ambient ?? [0, 0, 0],
    bvh: shapes.length > 0 ? buildBvh(shapes, options.bvh) : null,
    bvhObjects,
    planes,
  }
}

/**
 * The nearest object a ray hits in a scene (the hierarchy for bounded objects, then every plane).
 *
 * @param s - The scene.
 * @param ray - The ray.
 * @param tMin - Smallest accepted parameter (default {@link RAY_EPSILON}).
 * @param tMax - Largest accepted parameter.
 * @returns The hit and its object index, or null.
 * @concept ray-tracing
 * @example
 * ```ts
 * import { material, rayScene, sceneHit, sphere } from 'omai-engine/graphics'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material() }])
 * sceneHit(s, { origin: [0, 0, 3], direction: [0, 0, -1] })?.t // => 2
 * ```
 */
export function sceneHit(
  s: RayScene,
  ray: Ray,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): (Hit & { readonly object: number }) | null {
  let best: (Hit & { readonly object: number }) | null = null
  let limit = tMax
  if (s.bvh !== null) {
    const { hit } = intersectBvh(s.bvh, ray, { tMin, tMax: limit })
    if (hit !== null) {
      best = { ...hit, object: s.bvhObjects[hit.index] ?? -1 }
      limit = hit.t
    }
  }
  for (const i of s.planes) {
    const o = s.objects[i]
    if (o === undefined) continue
    const h = intersect(ray, o.shape, tMin, limit)
    if (h !== null) {
      best = { ...h, object: i }
      limit = h.t
    }
  }
  return best
}

/**
 * Whether anything in a scene blocks a ray before `tMax` (a shadow ray).
 *
 * @param s - The scene.
 * @param ray - The ray.
 * @param tMax - The far end.
 * @param tMin - The near end.
 * @returns Whether it is blocked.
 * @concept shadows
 * @example
 * ```ts
 * import { material, rayScene, sceneOccluded, sphere } from 'omai-engine/graphics'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material() }])
 * sceneOccluded(s, { origin: [0, 0, 3], direction: [0, 0, -1] }, 10) // => true
 * ```
 */
export function sceneOccluded(s: RayScene, ray: Ray, tMax: number, tMin: number = RAY_EPSILON): boolean {
  if (s.bvh !== null && occludedBvh(s.bvh, ray, tMax, tMin)) return true
  for (const i of s.planes) {
    const o = s.objects[i]
    if (o !== undefined && intersect(ray, o.shape, tMin, tMax) !== null) return true
  }
  return false
}

/**
 * The albedo of an object at a hit: its albedo map sampled at the hit's surface coordinates (bilinear, tiled), else
 * the material's.
 *
 * @param o - The object.
 * @param hit - The hit.
 * @returns The albedo.
 * @concept texture-mapping
 * @example
 * ```ts
 * import { albedoAt, material, sphere } from 'omai-engine/graphics'
 * const o = { shape: sphere([0, 0, 0], 1), material: material(), albedoMap: (u, v) => [u, v, 0] }
 * albedoAt(o, { t: 1, point: [0, 0, 1], normal: [0, 0, 1], frontFace: true, uv: [0.25, 0.5] }) // => [0.25, 0.5, 0]
 * ```
 */
export function albedoAt(o: SceneObject, hit: Hit): Vec3 {
  const map = o.albedoMap
  if (map === undefined) return o.material.albedo
  if (typeof map === 'function') return map(hit.uv[0], hit.uv[1])
  const c = sampleBilinear(map, hit.uv[0], hit.uv[1])
  return [c[0], c[1], c[2]]
}

// ----------------------------------------------------------------------------------------- reflection and refraction

/**
 * Snell's law in vector form: the refracted direction of unit directions `d` through unit normals `n` (facing against
 * `d`) with relative index `eta = n1 / n2`, `eta d + (eta cos_i - sqrt(1 - eta^2 (1 - cos_i^2))) n`; zero vectors
 * under total internal reflection. Differentiable.
 *
 * @param d - Unit incident directions `[..., 3]`.
 * @param n - Unit normals on the incident side `[..., 3]`.
 * @param eta - Ratios of refractive indices `n1 / n2`.
 * @returns Unit refracted directions `[..., 3]` (zero under total internal reflection).
 * @concept snells-law
 * @example
 * ```ts
 * import { refract } from 'omai-engine/graphics'
 * import { toFlat } from 'omai-engine/tensor'
 * toFlat(refract([0, -1, 0], [0, 1, 0], 1 / 1.5)) // => [0, -1, 0]
 * toFlat(refract([0.8, -0.6, 0], [0, 1, 0], 1.5)) // => [0, 0, 0]
 * ```
 */
export function refract(d: TensorLike, n: TensorLike, eta: TensorLike): Value {
  const [dv, nv, e] = promoteFloat(d, n, eta) as [Value, Value, Value]
  const cosi = neg(dotProduct(dv, nv))
  const k = sub(1, mul(square(e), sub(1, square(cosi))))
  const ok = greater(k, 0)
  const root = sqrt(maximum(k, 1e-300))
  const t = add(mul(dv, expandDims(e, -1)), mul(nv, expandDims(sub(mul(e, cosi), root), -1)))
  return where(expandDims(ok, -1), t, 0)
}

/**
 * Schlick's Fresnel reflectance at a dielectric boundary from index `n1` into `n2`:
 * `R0 + (1 - R0)(1 - cos)^5` with `R0 = ((n1 - n2) / (n1 + n2))^2`, using the cosine on the less dense side (the
 * transmitted angle when leaving a denser medium), and 1 under total internal reflection. The transmitted fraction is
 * `1 - R`. Differentiable.
 *
 * @param cosI - Cosines of the angle of incidence `[...]`.
 * @param n1 - Index on the incident side.
 * @param n2 - Index on the far side.
 * @returns Reflectances `[...]` in [0, 1].
 * @concept fresnel-equations
 * @example
 * ```ts
 * import { schlick } from 'omai-engine/graphics'
 * import { item } from 'omai-engine/tensor'
 * Math.round(item(schlick(1, 1, 1.5)) * 1e12) / 1e12 // => 0.04
 * item(schlick(0.2, 1.5, 1)) // => 1
 * ```
 */
export function schlick(cosI: TensorLike, n1: TensorLike, n2: TensorLike): Value {
  const [c, a, b] = promoteFloat(cosI, n1, n2) as [Value, Value, Value]
  const r0 = square(div(sub(a, b), add(a, b)))
  const sin2t = mul(square(div(a, b)), sub(1, square(c)))
  const tir = greater(sin2t, 1)
  const cosT = sqrt(maximum(sub(1, where(tir, 0, sin2t)), 1e-300))
  const cos = where(greater(a, b), cosT, c)
  return where(tir, 1, fresnelSchlick(cos, r0))
}

// Scalar Schlick for the renderers' per-ray loops (the same formula as `schlick`).
function schlickScalar(cosI: number, n1: number, n2: number): number {
  const r0 = ((n1 - n2) / (n1 + n2)) ** 2
  let c = cosI
  if (n1 > n2) {
    const sin2t = (n1 / n2) ** 2 * (1 - cosI * cosI)
    if (sin2t > 1) return 1
    c = Math.sqrt(1 - sin2t)
  }
  return r0 + (1 - r0) * (1 - c) ** 5
}

/**
 * How a hit splits a ray's energy (Whitted's model): the local (shaded) weight `1 - mirror - transparency`, the mirror
 * weight `mirror + transparency F`, and the refraction weight `transparency (1 - F)`, with `F` Schlick's reflectance
 * (all reflected under total internal reflection). The three always sum to 1.
 *
 * @param m - The material.
 * @param cosI - Cosine between the reversed ray and the normal on the ray's side.
 * @param entering - Whether the ray enters the object (from outside).
 * @returns The weights and the refractive indices on each side.
 * @concept ray-tracing
 * @example
 * ```ts
 * import { material, splitWeights } from 'omai-engine/graphics'
 * const w = splitWeights(material({ transparency: 1 }), 1, true)
 * [w.local, Math.round(w.reflect * 1e12) / 1e12, Math.round(w.refract * 1e12) / 1e12] // => [0, 0.04, 0.96]
 * ```
 */
export function splitWeights(
  m: Material,
  cosI: number,
  entering: boolean,
): {
  readonly local: number
  readonly reflect: number
  readonly refract: number
  readonly n1: number
  readonly n2: number
} {
  const [n1, n2] = entering ? [1, m.ior] : [m.ior, 1]
  const f = m.transparency > 0 ? schlickScalar(Math.max(0, Math.min(1, cosI)), n1, n2) : 0
  return {
    local: 1 - m.mirror - m.transparency,
    reflect: m.mirror + m.transparency * f,
    refract: m.transparency * (1 - f),
    n1,
    n2,
  }
}

// --------------------------------------------------------------------------------------------- Whitted ray tracing

/** A shadow ray towards one light sample. */
export interface ShadowRecord {
  readonly direction: Vec3
  readonly distance: number
  readonly blocked: boolean
}

/**
 * One ray of a Whitted trace, for drawing the ray tree: where it came from, what it hit, the shadow rays from the hit,
 * its weight (the product of the mirror and refraction weights above it) and the radiance it added to its pixel.
 */
export interface RayRecord {
  readonly id: number
  /** The ray this one was spawned from (-1 for camera rays). */
  readonly parent: number
  readonly pixel: number
  readonly depth: number
  readonly kind: 'camera' | 'reflection' | 'refraction'
  readonly origin: Vec3
  readonly direction: Vec3
  readonly weight: Vec3
  /** The hit distance (Infinity on a miss). */
  readonly t: number
  /** The object hit (-1 on a miss). */
  readonly object: number
  readonly point: Vec3 | null
  /** The normal on the side the ray arrived from. */
  readonly normal: Vec3 | null
  readonly shadows: readonly ShadowRecord[]
  /** Radiance added to the pixel by this ray (already weighted). */
  readonly added: Vec3
}

interface Task {
  readonly origin: Vec3
  readonly direction: Vec3
  readonly weight: Vec3
  readonly pixel: number
  readonly depth: number
  readonly kind: RayRecord['kind']
  readonly parent: number
}

/** The state of {@link whittedAlgorithm}: the rays of the current generation, the image so far, and the records. */
export interface WhittedState {
  readonly generation: number
  readonly tasks: readonly Task[]
  readonly image: Float64Array
  readonly records: readonly RayRecord[]
  readonly rays: number
  readonly shadowRays: number
}

/** Options of {@link whittedAlgorithm}, {@link renderWhitted} and {@link traceRay}. */
export interface WhittedOptions {
  /** The deepest secondary ray (default 5; camera rays are depth 0). */
  readonly maxDepth?: number
  /** Rays whose weight falls below this in every channel are not spawned (default 1e-4). */
  readonly minWeight?: number
  /** Record every ray (`true`) or the rays of some pixels (default none). */
  readonly record?: boolean | readonly number[]
  /** A key for jittered area-light samples (default: stratified cell centres). */
  readonly key?: Key
}

const offset = (p: Vec3, n: Vec3, side: number): Vec3 =>
  madd3(p, n, side * 1e-7 * Math.max(1, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])))

// One generation: shade every ray's hit (batched per object), add the radiance to its pixel, and spawn the mirror and
// refraction rays.
function generation(s: RayScene, state: WhittedState, options: WhittedOptions): WhittedState {
  const image = Float64Array.from(state.image)
  const records = [...state.records]
  const maxDepth = options.maxDepth ?? 5
  const minWeight = options.minWeight ?? 1e-4
  const recordAll = options.record === true
  const recordSet = new Set(Array.isArray(options.record) ? options.record : [])
  const wants = (t: Task): boolean => recordAll || recordSet.has(t.pixel)
  const next: Task[] = []
  let shadowRays = 0
  const hits = state.tasks.map((t) => sceneHit(s, { origin: t.origin, direction: t.direction }))
  const add3To = (pixel: number, c: Vec3): void => {
    for (let k = 0; k < 3; k++) image[3 * pixel + k] = (image[3 * pixel + k] ?? 0) + (c[k] ?? 0)
  }
  const ids = new Map<number, number>()
  const byObject = new Map<number, number[]>()
  for (const [i, t] of state.tasks.entries()) {
    const h = hits[i]
    if (h === null || h === undefined) {
      const added = mul3(t.weight, s.background)
      add3To(t.pixel, added)
      if (wants(t)) {
        records.push({
          ...t,
          id: records.length,
          t: Number.POSITIVE_INFINITY,
          object: -1,
          point: null,
          normal: null,
          shadows: [],
          added,
        })
      }
      continue
    }
    const list = byObject.get(h.object) ?? []
    list.push(i)
    byObject.set(h.object, list)
  }
  for (const [object, list] of byObject) {
    const o = s.objects[object]
    if (o === undefined) continue
    const m = o.material
    const points = new Float64Array(3 * list.length)
    const normals = new Float64Array(3 * list.length)
    const views = new Float64Array(3 * list.length)
    const albedo = new Float64Array(3 * list.length)
    const shadows: ShadowRecord[][] = list.map(() => [])
    for (const [r, i] of list.entries()) {
      const h = hits[i]
      const t = state.tasks[i]
      if (h === null || h === undefined || t === undefined) continue
      const nf = h.frontFace ? h.normal : scale3(h.normal, -1)
      points.set(h.point, 3 * r)
      normals.set(nf, 3 * r)
      views.set(scale3(t.direction, -1), 3 * r)
      albedo.set(albedoAt(o, h), 3 * r)
    }
    const visible = (r: number, sample: LightSample): boolean => {
      shadowRays++
      const p: Vec3 = [points[3 * r] ?? 0, points[3 * r + 1] ?? 0, points[3 * r + 2] ?? 0]
      const n: Vec3 = [normals[3 * r] ?? 0, normals[3 * r + 1] ?? 0, normals[3 * r + 2] ?? 0]
      const blocked =
        dot3(n, sample.direction) <= 0 ||
        sceneOccluded(s, { origin: offset(p, n, 1), direction: sample.direction }, sample.distance * (1 - 1e-9))
      const t = state.tasks[list[r] ?? 0]
      if (t !== undefined && wants(t))
        shadows[r]?.push({ direction: sample.direction, distance: sample.distance, blocked })
      return !blocked
    }
    const local =
      m.mirror + m.transparency >= 1
        ? new Float64Array(3 * list.length)
        : shadePoints({ ...m, emission: [0, 0, 0] }, points, normals, views, s.lights, {
            visible,
            ambient: s.ambient,
            albedo,
            ...(options.key === undefined ? {} : { key: options.key }),
          })
    for (const [r, i] of list.entries()) {
      const h = hits[i]
      const t = state.tasks[i]
      if (h === null || h === undefined || t === undefined) continue
      const n: Vec3 = [normals[3 * r] ?? 0, normals[3 * r + 1] ?? 0, normals[3 * r + 2] ?? 0]
      const cosI = -dot3(t.direction, n)
      const w = splitWeights(m, cosI, h.frontFace)
      const radiance: Vec3 = [
        m.emission[0] + w.local * (local[3 * r] ?? 0),
        m.emission[1] + w.local * (local[3 * r + 1] ?? 0),
        m.emission[2] + w.local * (local[3 * r + 2] ?? 0),
      ]
      const added = mul3(t.weight, radiance)
      add3To(t.pixel, added)
      let id = -1
      if (wants(t)) {
        id = records.length
        records.push({ ...t, id, t: h.t, object, point: h.point, normal: n, shadows: shadows[r] ?? [], added })
      }
      ids.set(i, id)
      if (t.depth >= maxDepth) continue
      const refracted = w.refract > 0 ? refract3(t.direction, n, w.n1 / w.n2) : undefined
      const reflectWeight = w.reflect + (refracted === undefined ? w.refract : 0)
      const spawn = (kind: Task['kind'], direction: Vec3, k: number, side: number): void => {
        const weight = scale3(t.weight, k)
        if (Math.max(weight[0], weight[1], weight[2]) <= minWeight) return
        next.push({
          origin: offset(h.point, n, side),
          direction,
          weight,
          pixel: t.pixel,
          depth: t.depth + 1,
          kind,
          parent: id,
        })
      }
      if (reflectWeight > 0) spawn('reflection', reflect3(t.direction, n), reflectWeight, 1)
      if (refracted !== undefined) spawn('refraction', refracted, w.refract, -1)
    }
  }
  return {
    generation: state.generation + 1,
    tasks: next,
    image,
    records,
    rays: state.rays + state.tasks.length,
    shadowRays: state.shadowRays + shadowRays,
  }
}

function initialState(tasks: Task[], pixels: number): WhittedState {
  return { generation: 0, tasks, image: new Float64Array(3 * pixels), records: [], rays: 0, shadowRays: 0 }
}

/**
 * Whitted ray tracing as a traced {@link Algorithm}, one step per generation of rays: every ray of the generation is
 * intersected with the scene, its hit shaded by the material's BRDF under every light sample whose shadow ray is
 * clear (all hits on one object shaded in one batched call), and the mirror and refraction rays it spawns form the
 * next generation, weighted by {@link splitWeights}. The run ends when no ray is left; depth and weight caps bound it
 * on any scene (including mirrors facing each other).
 *
 * @param s - The scene.
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - Depth and weight caps, recording and area-light jitter.
 * @returns The algorithm.
 * @concept ray-tracing
 * @index algorithm Whitted ray tracing, a batch of rays per step.
 * @example
 * ```ts
 * import { material, pinholeCamera, pointLight, rayScene, sphere, whittedAlgorithm } from 'omai-engine/graphics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ mirror: 0.5 }) }], { lights: [pointLight([0, 5, 5], [50, 50, 50])] })
 * runAlgorithm(whittedAlgorithm(s, pinholeCamera(), 8, 6), { maxSteps: 10 }).status // => 'converged'
 * ```
 */
export function whittedAlgorithm(
  s: RayScene,
  cam: RenderCamera,
  width: number,
  height: number,
  options: WhittedOptions = {},
): Algorithm<WhittedState> {
  return {
    name: 'graphics/whitted',
    config: { width, height, maxDepth: options.maxDepth ?? 5, minWeight: options.minWeight ?? 1e-4 } as Json,
    schema: channels({
      metrics: {
        rays: { description: 'rays traced in this generation' },
        shadowRays: { description: 'shadow rays traced so far' },
        meanRadiance: { description: 'mean pixel radiance so far (all channels)' },
      },
    }),
    init: () => {
      const bundle = cameraRays(cam, width, height)
      const tasks: Task[] = []
      for (let k = 0; k < width * height; k++) {
        tasks.push({
          origin: [bundle.origins[3 * k] ?? 0, bundle.origins[3 * k + 1] ?? 0, bundle.origins[3 * k + 2] ?? 0],
          direction: [
            bundle.directions[3 * k] ?? 0,
            bundle.directions[3 * k + 1] ?? 0,
            bundle.directions[3 * k + 2] ?? 0,
          ],
          weight: [1, 1, 1],
          pixel: k,
          depth: 0,
          kind: 'camera',
          parent: -1,
        })
      }
      return initialState(tasks, width * height)
    },
    step: (state) => generation(s, state, options),
    done: (state) => (state.tasks.length === 0 ? 'converged' : undefined),
    observe: (state) => {
      let total = 0
      for (const v of state.image) total += v
      return {
        metrics: { rays: state.tasks.length, shadowRays: state.shadowRays, meanRadiance: total / state.image.length },
      }
    },
  }
}

/**
 * Render a scene by Whitted ray tracing (see {@link whittedAlgorithm}).
 *
 * @param s - The scene.
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - Depth and weight caps, recording, area-light jitter, and a tracer and policy for the run.
 * @returns The linear image, the recorded rays and the ray counts.
 * @concept ray-tracing
 * @index function Whitted ray tracing with reflection, refraction and shadows.
 * @example
 * ```ts
 * import { material, pinholeCamera, rayScene, renderWhitted, sphere } from 'omai-engine/graphics'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ emission: [1, 0, 0] }) }])
 * const r = renderWhitted(s, pinholeCamera(), 4, 4)
 * [r.image.rgb[3 * 5], r.image.rgb[0], r.stats.rays] // => [1, 0, 16]
 * ```
 */
export function renderWhitted(
  s: RayScene,
  cam: RenderCamera,
  width: number,
  height: number,
  options: WhittedOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): {
  readonly image: FloatImage
  readonly records: readonly RayRecord[]
  readonly stats: { readonly rays: number; readonly shadowRays: number; readonly generations: number }
} {
  const { tracer, policy, parent, ...rest } = options
  const run = runAlgorithm(whittedAlgorithm(s, cam, width, height, rest), {
    maxSteps: (rest.maxDepth ?? 5) + 2,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
  })
  const st = run.state
  return {
    image: { width, height, rgb: st.image },
    records: st.records,
    stats: { rays: st.rays, shadowRays: st.shadowRays, generations: st.generation },
  }
}

/**
 * Trace one ray through a scene the Whitted way and return the radiance it carries back with its whole ray tree
 * (reflection and refraction rays, and shadow rays at every hit), for figures that follow one pixel.
 *
 * @param s - The scene.
 * @param ray - The ray (its direction is normalised here).
 * @param options - Depth and weight caps and area-light jitter.
 * @returns The radiance and the records of every ray.
 * @throws OmaiError with code `value` for a zero or non-finite direction, or a negative depth cap.
 * @concept ray-tracing
 * @example
 * ```ts
 * import { material, plane, rayScene, sphere, traceRay } from 'omai-engine/graphics'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ mirror: 1 }) }], { background: [0.5, 0.5, 0.5] })
 * const r = traceRay(s, { origin: [0, 0, 3], direction: [0, 0, -1] })
 * [r.radiance, r.records.map((x) => x.kind)] // => [[0.5, 0.5, 0.5], ['camera', 'reflection']]
 * ```
 */
export function traceRay(
  s: RayScene,
  ray: Ray,
  options: Omit<WhittedOptions, 'record'> = {},
): { readonly radiance: Vec3; readonly records: readonly RayRecord[] } {
  const maxDepth = options.maxDepth ?? 5
  if (!(Number.isInteger(maxDepth) && maxDepth >= 0)) throw new OmaiError('value', 'traceRay: maxDepth must be >= 0')
  const direction = normalise3(ray.direction, 'the ray direction')
  let state = initialState(
    [
      {
        origin: check3(ray.origin, 'the ray origin'),
        direction,
        weight: [1, 1, 1],
        pixel: 0,
        depth: 0,
        kind: 'camera',
        parent: -1,
      },
    ],
    1,
  )
  for (let g = 0; g <= maxDepth && state.tasks.length > 0; g++)
    state = generation(s, state, { ...options, record: true })
  return { radiance: [state.image[0] ?? 0, state.image[1] ?? 0, state.image[2] ?? 0], records: state.records }
}

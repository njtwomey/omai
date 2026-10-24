import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import type { Vec3 } from '../scene/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { basis3, dot3, madd3, mul3, normalise3, reflect3, refract3, scale3 } from './_vec3.ts'
import { pixelRay, type Ray, type RenderCamera } from './camera.ts'
import { lightSamples } from './lights.ts'
import { albedoAt, type RayScene, sceneHit, sceneOccluded, splitWeights } from './raytracing.ts'
import type { FloatImage } from './textures.ts'

/**
 * Cosine-weighted sampling of the hemisphere about `+z` (Malley's method: a uniform point on the unit disc lifted to
 * the hemisphere), with density `cos(theta) / pi`.
 *
 * @param u1 - A uniform in [0, 1) (the squared radius on the disc).
 * @param u2 - A uniform in [0, 1) (the angle around).
 * @returns A unit direction with non-negative z.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { cosineHemisphere } from 'whitebox-engine/graphics'
 * cosineHemisphere(0, 0) // => [0, 0, 1]
 * ```
 */
export function cosineHemisphere(u1: number, u2: number): Vec3 {
  const r = Math.sqrt(u1)
  const phi = 2 * Math.PI * u2
  return [r * Math.cos(phi), r * Math.sin(phi), Math.sqrt(Math.max(0, 1 - u1))]
}

/**
 * Keyed cosine-weighted directions about a unit normal (each from {@link cosineHemisphere}, rotated into a frame
 * around the normal).
 *
 * @param key - The key.
 * @param normal - The unit normal.
 * @param n - How many directions.
 * @returns The directions, all with non-negative cosine to the normal.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { sampleCosineHemisphere } from 'whitebox-engine/graphics'
 * import { key } from 'whitebox-engine/random'
 * sampleCosineHemisphere(key(0), [0, 1, 0], 100).every((d) => d[1] >= 0) // => true
 * ```
 */
export function sampleCosineHemisphere(key: Key, normal: Vec3, n: number): Vec3[] {
  const u = uniform(key, [2 * n]).data
  const nn = normalise3(normal, 'normal')
  const [t, b] = basis3(nn)
  const out: Vec3[] = []
  for (let i = 0; i < n; i++) {
    const [x, y, z] = cosineHemisphere(u[2 * i] ?? 0, u[2 * i + 1] ?? 0)
    out.push(madd3(madd3(scale3(t, x), b, y), nn, z))
  }
  return out
}

/** A vertex of a traced path, for drawing it: what happened there and the path's state after it. */
export interface PathVertex {
  readonly point: Vec3 | null
  readonly normal: Vec3 | null
  readonly object: number
  /** `'diffuse'`, `'mirror'`, `'reflect'` and `'refract'` scatter; `'miss'` escapes; `'absorbed'` ends at the cap. */
  readonly event: 'diffuse' | 'mirror' | 'reflect' | 'refract' | 'miss' | 'absorbed'
  /** The throughput after the vertex. */
  readonly throughput: Vec3
  /** The radiance gathered so far. */
  readonly radiance: Vec3
}

/** Uniforms a bounce consumes: the event choice, two for the new direction, then two per area-light sample. */
function uniformsPerBounce(s: RayScene): number {
  let n = 3
  for (const l of s.lights) if (l.kind === 'area') n += 2 * l.samples * l.samples
  return n
}

interface PathState {
  ray: Ray
  throughput: Vec3
  radiance: Vec3
  alive: boolean
}

// One bounce of one path, driven by explicit uniforms `u[offset ...]`: add emission and next-event estimates of the
// lights at the hit, then scatter (diffuse by cosine sampling, ideal mirror, or Fresnel-chosen reflection or refraction).
function bounce(s: RayScene, p: PathState, u: ArrayLike<number>, offset: number, last: boolean): PathVertex {
  const hit = sceneHit(s, p.ray)
  if (hit === null) {
    p.radiance = madd3(p.radiance, mul3(p.throughput, s.background), 1)
    p.alive = false
    return { point: null, normal: null, object: -1, event: 'miss', throughput: p.throughput, radiance: p.radiance }
  }
  const o = s.objects[hit.object]
  if (o === undefined) throw new WhiteboxError('value', 'pathTrace: hit an unknown object')
  const m = o.material
  const n = hit.frontFace ? hit.normal : scale3(hit.normal, -1)
  const d = p.ray.direction
  p.radiance = madd3(p.radiance, mul3(p.throughput, m.emission), 1)
  const w = splitWeights(m, -dot3(d, n), hit.frontFace)
  const choice = u[offset] ?? 0
  const lift = (side: number): Vec3 =>
    madd3(
      hit.point,
      n,
      side * 1e-7 * Math.max(1, Math.abs(hit.point[0]), Math.abs(hit.point[1]), Math.abs(hit.point[2])),
    )
  if (choice < w.local) {
    const albedo = albedoAt(o, hit)
    // Next-event estimation of the lights (which rays never hit): Lambertian BRDF albedo / pi times E cos.
    let k = offset + 3
    for (const light of s.lights) {
      const count = light.kind === 'area' ? 2 * light.samples * light.samples : 0
      const jitter = Array.from({ length: count }, (_, i) => u[k + i] ?? 0.5)
      k += count
      for (const sample of lightSamples(light, hit.point, count > 0 ? jitter : undefined)) {
        const cos = dot3(n, sample.direction)
        if (cos <= 0) continue
        if (sceneOccluded(s, { origin: lift(1), direction: sample.direction }, sample.distance * (1 - 1e-9))) continue
        p.radiance = madd3(p.radiance, mul3(mul3(p.throughput, albedo), sample.irradiance), cos / Math.PI)
      }
    }
    // Choosing the diffuse event with probability `local` and weighting by 1 / local cancels; the sampled
    // direction's BRDF cos / pdf is the albedo.
    p.throughput = mul3(p.throughput, albedo)
    const [t, b] = basis3(n)
    const [x, y, z] = cosineHemisphere(u[offset + 1] ?? 0, u[offset + 2] ?? 0)
    p.ray = { origin: lift(1), direction: normalise3(madd3(madd3(scale3(t, x), b, y), n, z)) }
    if (last) p.alive = false
    return {
      point: hit.point,
      normal: n,
      object: hit.object,
      event: last ? 'absorbed' : 'diffuse',
      throughput: p.throughput,
      radiance: p.radiance,
    }
  }
  const refracted = w.refract > 0 ? refract3(d, n, w.n1 / w.n2) : undefined
  const reflectShare = w.reflect + (refracted === undefined ? w.refract : 0)
  let event: PathVertex['event']
  if (choice < w.local + reflectShare || refracted === undefined) {
    p.ray = { origin: lift(1), direction: reflect3(d, n) }
    event = m.transparency > 0 ? 'reflect' : 'mirror'
  } else {
    p.ray = { origin: lift(-1), direction: refracted }
    event = 'refract'
  }
  if (last) p.alive = false
  return {
    point: hit.point,
    normal: n,
    object: hit.object,
    event: last ? 'absorbed' : event,
    throughput: p.throughput,
    radiance: p.radiance,
  }
}

/** Options of {@link pathTracerAlgorithm} and {@link pathTrace}. */
export interface PathTraceOptions {
  /** Samples per pixel (default 16). */
  readonly samples?: number
  /** The most scattering events per path (default 4). */
  readonly maxBounces?: number
  /** Jitter each sample inside its pixel (default true); off, every sample goes through the pixel centre. */
  readonly jitter?: boolean
}

/** The state of {@link pathTracerAlgorithm}: samples taken and the per-pixel sums of radiance and its square. */
export interface PathTracerState {
  readonly samples: number
  readonly sum: Float64Array
  readonly sumSquares: Float64Array
}

/**
 * A small path tracer as a traced {@link Algorithm}, one step per sample of every pixel: each path starts at the camera
 * (jittered within its pixel), adds the emission of what it hits and next-event estimates of the scene's lights at
 * diffuse hits (shadow rays), and scatters by cosine-weighted sampling (diffuse), ideal reflection (mirror) or
 * Fresnel-weighted choice between reflection and refraction (glass), for at most `maxBounces` events. Non-Lambertian
 * shading models are treated as Lambertian with their albedo. Every random number comes from the step key, so a run
 * is a pure function of its key; the mean image converges to the solution of the rendering equation for this model.
 *
 * @param s - The scene.
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - Samples, bounces and jitter.
 * @returns The algorithm.
 * @concept path-tracing
 * @example
 * ```ts
 * import { material, pathTracerAlgorithm, pinholeCamera, rayScene, sphere } from 'whitebox-engine/graphics'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material() }], { background: [1, 1, 1] })
 * runAlgorithm(pathTracerAlgorithm(s, pinholeCamera(), 4, 4, { samples: 3 }), { seed: 0, maxSteps: 10 }).steps // => 3
 * ```
 */
export function pathTracerAlgorithm(
  s: RayScene,
  cam: RenderCamera,
  width: number,
  height: number,
  options: PathTraceOptions = {},
): Algorithm<PathTracerState> {
  const samples = options.samples ?? 16
  const maxBounces = options.maxBounces ?? 4
  if (!(Number.isInteger(samples) && samples >= 1 && Number.isInteger(maxBounces) && maxBounces >= 0)) {
    throw new WhiteboxError('value', 'pathTrace: samples must be >= 1 and maxBounces >= 0')
  }
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new WhiteboxError('value', 'pathTrace: the image size must be positive integers')
  }
  const n = width * height
  const per = uniformsPerBounce(s)
  return {
    name: 'graphics/path-tracer',
    config: { width, height, samples, maxBounces, jitter: options.jitter !== false } as Json,
    schema: channels({
      metrics: {
        samples: { description: 'samples per pixel so far' },
        meanRadiance: { description: 'mean radiance over pixels and channels' },
        standardError: { description: 'mean standard error of the pixel means (all channels)' },
      },
      tensors: { image: { role: 'state', description: 'the mean radiance per pixel', shape: [height, width, 3] } },
    }),
    init: () => ({ samples: 0, sum: new Float64Array(3 * n), sumSquares: new Float64Array(3 * n) }),
    step: (state, ctx) => {
      const sum = Float64Array.from(state.sum)
      const sumSquares = Float64Array.from(state.sumSquares)
      const jitter = options.jitter === false ? undefined : uniform(foldIn(ctx.key, 0), [2 * n]).data
      const paths: PathState[] = []
      for (let k = 0; k < n; k++) {
        const x = (k % width) + (jitter?.[2 * k] ?? 0.5)
        const y = Math.floor(k / width) + (jitter?.[2 * k + 1] ?? 0.5)
        paths.push({ ray: pixelRay(cam, width, height, x, y), throughput: [1, 1, 1], radiance: [0, 0, 0], alive: true })
      }
      for (let b = 0; b <= maxBounces; b++) {
        const u = uniform(foldIn(ctx.key, 1 + b), [per * n]).data
        let alive = 0
        for (const [k, p] of paths.entries()) {
          if (!p.alive) continue
          bounce(s, p, u, per * k, b === maxBounces)
          if (p.alive) alive++
        }
        if (alive === 0) break
      }
      for (const [k, p] of paths.entries()) {
        for (let c = 0; c < 3; c++) {
          const v = p.radiance[c] ?? 0
          sum[3 * k + c] = (sum[3 * k + c] ?? 0) + v
          sumSquares[3 * k + c] = (sumSquares[3 * k + c] ?? 0) + v * v
        }
      }
      return { samples: state.samples + 1, sum, sumSquares }
    },
    done: (state) => (state.samples >= samples ? 'converged' : undefined),
    observe: (state) => {
      const m = Math.max(1, state.samples)
      let total = 0
      let se = 0
      const mean = new Float64Array(3 * n)
      for (let i = 0; i < 3 * n; i++) {
        const mu = (state.sum[i] ?? 0) / m
        mean[i] = mu
        total += mu
        se += m > 1 ? Math.sqrt(Math.max(0, (state.sumSquares[i] ?? 0) / m - mu * mu) / (m - 1)) : 0
      }
      return {
        metrics: { samples: state.samples, meanRadiance: total / (3 * n), standardError: se / (3 * n) },
        tensors: { image: { role: 'state', value: mean, shape: [height, width, 3] } },
      }
    },
  }
}

/**
 * Render a scene with the path tracer of {@link pathTracerAlgorithm}: the mean of `samples` paths per pixel.
 *
 * @param s - The scene.
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - Samples, bounces and jitter, the key, and a tracer and policy for the run.
 * @returns The mean image and the samples taken.
 * @concept path-tracing
 * @example
 * ```ts
 * import { material, pathTrace, pinholeCamera, rayScene, sphere } from 'whitebox-engine/graphics'
 * import { key } from 'whitebox-engine/random'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ albedo: [0.5, 0.5, 0.5] }) }], { background: [1, 1, 1] })
 * const r = pathTrace(s, pinholeCamera(), 4, 4, { key: key(0), samples: 4, jitter: false })
 * [r.image.rgb[3 * 5], r.image.rgb[0]] // => [0.5, 1]
 * ```
 */
export function pathTrace(
  s: RayScene,
  cam: RenderCamera,
  width: number,
  height: number,
  options: PathTraceOptions & { readonly key: Key } & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>>,
): { readonly image: FloatImage; readonly samples: number } {
  const { key, tracer, policy, parent, ...rest } = options
  const run = runAlgorithm(pathTracerAlgorithm(s, cam, width, height, rest), {
    key,
    maxSteps: rest.samples ?? 16,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
  })
  const m = run.state.samples
  return { image: { width, height, rgb: Float64Array.from(run.state.sum, (v) => v / m) }, samples: m }
}

/**
 * Follow one path through a scene with an explicit key, returning every vertex (for figures of a single light path).
 *
 * @param s - The scene.
 * @param ray - The starting ray.
 * @param key - The key; bounce `b` draws its uniforms from `foldIn(key, b)`.
 * @param options - The most scattering events (default 4).
 * @returns The radiance carried back and the vertices.
 * @concept path-tracing
 * @example
 * ```ts
 * import { material, rayScene, sphere, tracePath } from 'whitebox-engine/graphics'
 * import { key } from 'whitebox-engine/random'
 * const s = rayScene([{ shape: sphere([0, 0, 0], 1), material: material({ albedo: [0.5, 0.5, 0.5] }) }], { background: [1, 1, 1] })
 * const p = tracePath(s, { origin: [0, 0, 3], direction: [0, 0, -1] }, key(0))
 * [p.radiance, p.vertices.map((v) => v.event)] // => [[0.5, 0.5, 0.5], ['diffuse', 'miss']]
 * ```
 */
export function tracePath(
  s: RayScene,
  ray: Ray,
  key: Key,
  options: { readonly maxBounces?: number } = {},
): { readonly radiance: Vec3; readonly vertices: readonly PathVertex[] } {
  const maxBounces = options.maxBounces ?? 4
  if (!(Number.isInteger(maxBounces) && maxBounces >= 0)) throw new WhiteboxError('value', 'tracePath: maxBounces >= 0')
  const per = uniformsPerBounce(s)
  const p: PathState = { ray, throughput: [1, 1, 1], radiance: [0, 0, 0], alive: true }
  const vertices: PathVertex[] = []
  for (let b = 0; b <= maxBounces && p.alive; b++) {
    vertices.push(bounce(s, p, uniform(foldIn(key, b), [per]).data, 0, b === maxBounces))
  }
  return { radiance: p.radiance, vertices }
}

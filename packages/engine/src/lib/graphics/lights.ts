/**
 * Light sources for shading: point lights with inverse-square falloff, directional lights, and rectangular area lights
 * sampled with stratified or jittered samples (which give soft shadows), plus shading of a batch of surface points
 * under these lights with a shadow-ray visibility test per sample.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, isKey, type Key, uniform } from '../random/index.ts'
import type { Vec3 } from '../scene/index.ts'
import { concreteOrThrow, fromStorage } from '../tensor/index.ts'
import { check3, cross3, dot3, length3, madd3, normalise3, scale3, sub3 } from './_vec3.ts'
import { type Material, shade } from './shading.ts'

/**
 * A light: a point light of radiant intensity `intensity` (irradiance falls off as `1 / d^2`), a directional light
 * (the sun: parallel rays travelling along `direction` with irradiance `irradiance`), or a one-sided rectangular area
 * light at `corner` spanned by `edgeU` and `edgeV`, emitting radiance `radiance` towards `edgeU x edgeV`, sampled
 * on a `samples x samples` grid (soft shadows).
 *
 * @concept light-sources
 */
export type Light =
  | { readonly kind: 'point'; readonly position: Vec3; readonly intensity: Vec3 }
  | { readonly kind: 'directional'; readonly direction: Vec3; readonly irradiance: Vec3 }
  | {
      readonly kind: 'area'
      readonly corner: Vec3
      readonly edgeU: Vec3
      readonly edgeV: Vec3
      readonly radiance: Vec3
      readonly samples: number
    }

/**
 * A point light.
 *
 * @param position - Where it is.
 * @param intensity - Radiant intensity per channel.
 * @returns The light.
 * @concept light-sources
 * @example
 * ```ts
 * import { pointLight } from 'mdml-engine/graphics'
 * pointLight([0, 2, 0], [10, 10, 10]).kind // => 'point'
 * ```
 */
export function pointLight(position: Vec3, intensity: Vec3): Light {
  return { kind: 'point', position: check3(position, 'position'), intensity: check3(intensity, 'intensity') }
}

/**
 * A directional light.
 *
 * @param direction - The direction the light travels (normalised here).
 * @param irradiance - Irradiance on a surface facing the light, per channel.
 * @returns The light.
 * @concept light-sources
 * @example
 * ```ts
 * import { directionalLight } from 'mdml-engine/graphics'
 * directionalLight([0, 0, -2], [1, 1, 1]).direction // => [0, 0, -1]
 * ```
 */
export function directionalLight(direction: Vec3, irradiance: Vec3): Light {
  return {
    kind: 'directional',
    direction: normalise3(check3(direction, 'direction'), 'direction'),
    irradiance: check3(irradiance, 'irradiance'),
  }
}

/**
 * A one-sided rectangular area light.
 *
 * @param corner - One corner.
 * @param edgeU - The first edge from the corner.
 * @param edgeV - The second edge (the light faces `edgeU x edgeV`).
 * @param radiance - Emitted radiance per channel.
 * @param options - Samples per side (default 4, so 16 shadow rays).
 * @returns The light.
 * @throws MdmlError with code `value` for parallel edges or a sample count below 1.
 * @concept light-sources
 * @example
 * ```ts
 * import { areaLight } from 'mdml-engine/graphics'
 * areaLight([0, 2, 0], [1, 0, 0], [0, 0, 1], [5, 5, 5]).samples // => 4
 * ```
 */
export function areaLight(
  corner: Vec3,
  edgeU: Vec3,
  edgeV: Vec3,
  radiance: Vec3,
  options: { readonly samples?: number } = {},
): Light {
  const samples = options.samples ?? 4
  if (!(Number.isInteger(samples) && samples >= 1)) throw new MdmlError('value', 'areaLight: samples must be >= 1')
  const u = check3(edgeU, 'edgeU')
  const v = check3(edgeV, 'edgeV')
  normalise3(cross3(u, v), 'edgeU x edgeV')
  return {
    kind: 'area',
    corner: check3(corner, 'corner'),
    edgeU: u,
    edgeV: v,
    radiance: check3(radiance, 'radiance'),
    samples,
  }
}

/** One sample of a light seen from a point: the unit direction towards it, the distance, and the irradiance it delivers. */
export interface LightSample {
  readonly direction: Vec3
  /** Distance to the sample point (Infinity for directional lights). */
  readonly distance: number
  /** Irradiance on a surface facing the sample (the receiver's cosine is applied by {@link shade}). */
  readonly irradiance: Vec3
}

/**
 * Sample a light from a point: one sample for point and directional lights, `samples^2` stratified samples for area
 * lights (cell centres, or jittered within each cell by a key, or by `2 samples^2` given uniforms in [0, 1), as a
 * path tracer supplies them). An area sample carries
 * `L cos(theta_light) A / (N d^2)`, so the samples sum to the light's irradiance; samples from behind the light carry
 * zero.
 *
 * @param light - The light.
 * @param point - The receiving point.
 * @param jitter - Optional key, or uniforms, for jittered area samples.
 * @returns The samples.
 * @concept light-sources
 * @example
 * ```ts
 * import { lightSamples, pointLight } from 'mdml-engine/graphics'
 * lightSamples(pointLight([0, 0, 2], [4, 4, 4]), [0, 0, 0]) // => [{ direction: [0, 0, 1], distance: 2, irradiance: [1, 1, 1] }]
 * ```
 */
export function lightSamples(light: Light, point: Vec3, jitter?: Key | ArrayLike<number>): LightSample[] {
  switch (light.kind) {
    case 'point': {
      const w = sub3(light.position, point)
      const d = length3(w)
      if (!(d > 0)) return []
      return [{ direction: scale3(w, 1 / d), distance: d, irradiance: scale3(light.intensity, 1 / (d * d)) }]
    }
    case 'directional':
      return [
        { direction: scale3(light.direction, -1), distance: Number.POSITIVE_INFINITY, irradiance: light.irradiance },
      ]
    case 'area': {
      const k = light.samples
      const normal = normalise3(cross3(light.edgeU, light.edgeV))
      const area = length3(cross3(light.edgeU, light.edgeV))
      const offsets =
        jitter === undefined ? undefined : isKey(jitter) ? uniform(foldIn(jitter, 0), [2 * k * k]).data : jitter
      const out: LightSample[] = []
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          const s = i * k + j
          const a = (i + (offsets?.[2 * s] ?? 0.5)) / k
          const b = (j + (offsets?.[2 * s + 1] ?? 0.5)) / k
          const p = madd3(madd3(light.corner, light.edgeU, a), light.edgeV, b)
          const w = sub3(p, point)
          const d = length3(w)
          if (!(d > 0)) continue
          const dir = scale3(w, 1 / d)
          const cosLight = Math.max(0, -dot3(dir, normal))
          out.push({
            direction: dir,
            distance: d,
            irradiance: scale3(light.radiance, (cosLight * area) / (k * k * d * d)),
          })
        }
      }
      return out
    }
  }
}

/**
 * Shade a batch of surface points lit by lights, with a visibility test per light sample (shadow rays): the sum over
 * samples of {@link shade}, plus the material's emission. One call shades every point, so the BRDF runs once per batch.
 *
 * @param m - The material.
 * @param points - Positions, flat `[x, y, z, ...]`.
 * @param normals - Unit normals, flat.
 * @param views - Unit directions towards the viewer, flat.
 * @param lights - The lights.
 * @param options - `visible(point index, sample)` for shadows (default: everything visible), a key for jittered area
 *   samples, an ambient radiance added as `albedo * ambient`, and per-point albedo (flat RGB) replacing the
 *   material's, for textured surfaces.
 * @returns Radiance per point, flat RGB.
 * @concept rendering-equation
 * @example
 * ```ts
 * import { material, pointLight, shadePoints } from 'mdml-engine/graphics'
 * const m = material({ albedo: [Math.PI, Math.PI, Math.PI] })
 * Array.from(shadePoints(m, [0, 0, 0], [0, 0, 1], [0, 0, 1], [pointLight([0, 0, 1], [1, 2, 3])])) // => [1, 2, 3]
 * ```
 */
export function shadePoints(
  m: Material,
  points: ArrayLike<number>,
  normals: ArrayLike<number>,
  views: ArrayLike<number>,
  lights: readonly Light[],
  options: {
    readonly visible?: (index: number, sample: LightSample) => boolean
    readonly key?: Key
    readonly ambient?: Vec3
    readonly albedo?: ArrayLike<number>
  } = {},
): Float64Array {
  const count = Math.floor(points.length / 3)
  const out = new Float64Array(3 * count)
  const ambient = options.ambient ?? [0, 0, 0]
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++)
      out[3 * i + c] = (m.emission[c] ?? 0) + (options.albedo?.[3 * i + c] ?? m.albedo[c] ?? 0) * (ambient[c] ?? 0)
  }
  const rows: number[] = []
  const nRows: number[] = []
  const lRows: number[] = []
  const vRows: number[] = []
  const eRows: number[] = []
  const aRows: number[] = []
  for (let i = 0; i < count; i++) {
    const p: Vec3 = [points[3 * i] ?? 0, points[3 * i + 1] ?? 0, points[3 * i + 2] ?? 0]
    for (const [li, light] of lights.entries()) {
      const samples = lightSamples(light, p, options.key === undefined ? undefined : foldIn(foldIn(options.key, i), li))
      for (const s of samples) {
        if (s.irradiance[0] === 0 && s.irradiance[1] === 0 && s.irradiance[2] === 0) continue
        if (options.visible !== undefined && !options.visible(i, s)) continue
        rows.push(i)
        for (let c = 0; c < 3; c++) {
          nRows.push(normals[3 * i + c] ?? 0)
          vRows.push(views[3 * i + c] ?? 0)
          lRows.push(s.direction[c] ?? 0)
          eRows.push(s.irradiance[c] ?? 0)
          aRows.push(options.albedo?.[3 * i + c] ?? m.albedo[c] ?? 0)
        }
      }
    }
  }
  if (rows.length === 0) return out
  const shape = [rows.length, 3]
  const radiance = concreteOrThrow(
    shade(
      m,
      fromStorage(Float64Array.from(nRows), shape, 'float64'),
      fromStorage(Float64Array.from(lRows), shape, 'float64'),
      fromStorage(Float64Array.from(vRows), shape, 'float64'),
      fromStorage(Float64Array.from(eRows), shape, 'float64'),
      fromStorage(Float64Array.from(aRows), shape, 'float64'),
    ),
    'shadePoints',
  ).data
  for (const [r, i] of rows.entries()) {
    for (let c = 0; c < 3; c++) out[3 * i + c] = (out[3 * i + c] ?? 0) + (radiance[3 * r + c] ?? 0)
  }
  return out
}

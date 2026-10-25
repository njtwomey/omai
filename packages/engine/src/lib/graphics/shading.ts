/**
 * Surface shading models: how much light a surface reflects towards the viewer. The classical terms (Lambert, Phong,
 * Blinn-Phong), their normalised BRDFs, the microfacet pieces (GGX distribution, Smith masking, Schlick and exact
 * Fresnel) assembled into Cook-Torrance and glTF's metallic-roughness model, and materials with `brdf` and `shade`
 * evaluating them on batches.
 */

import { MdmlError } from '../base/index.ts'
import { dotProduct, unitVector } from '../foundations/index.ts'
import type { Vec3 } from '../scene/index.ts'
import {
  add,
  broadcastShapes,
  broadcastTo,
  div,
  expandDims,
  greater,
  maximum,
  mul,
  pow,
  promoteFloat,
  sqrt,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { positivePow } from './_ops.ts'
import { check3 } from './_vec3.ts'

// Shading terms are tensor compositions: they broadcast over leading axes (vectors in the last axis, of length 3),
// so a renderer shades a whole batch of hits in one call, and every term is differentiable in its parameters.

const cosine = (a: Value, b: Value): Value => dotProduct(a, b)
const clamp0 = (x: Value): Value => maximum(x, 0)

/**
 * The mirror reflection of directions `d` about unit normals `n`: `d - 2 (d . n) n`. Differentiable.
 *
 * @param d - Directions `[..., 3]`.
 * @param n - Unit normals `[..., 3]`.
 * @returns Reflected directions `[..., 3]`.
 * @concept reflection
 * @example
 * ```ts
 * import { reflect } from 'mdml-engine/graphics'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(reflect([1, -1, 0], [0, 1, 0])) // => [1, 1, 0]
 * ```
 */
export function reflect(d: TensorLike, n: TensorLike): Value {
  const [dv, nv] = promoteFloat(d, n) as [Value, Value]
  return sub(dv, mul(nv, expandDims(mul(cosine(dv, nv), 2), -1)))
}

/**
 * The half vector between a light direction and a view direction, `(l + v) / |l + v|`. Differentiable.
 *
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @returns Unit half vectors `[..., 3]`.
 * @concept blinn-phong
 * @example
 * ```ts
 * import { halfVector } from 'mdml-engine/graphics'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(halfVector([1, 0, 0], [0, 1, 0])).map((x) => Math.round(x * 1e6) / 1e6) // => [0.707107, 0.707107, 0]
 * ```
 */
export function halfVector(l: TensorLike, v: TensorLike): Value {
  const [lv, vv] = promoteFloat(l, v) as [Value, Value]
  return unitVector(add(lv, vv))
}

/**
 * Lambert's cosine term `max(0, n . l)`: how much light a matte surface receives at an angle. Differentiable.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @returns The clamped cosines `[...]`.
 * @concept lambertian-reflectance
 * @example
 * ```ts
 * import { lambert } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(lambert([0, 0, 1], [0.6, 0, 0.8])) // => 0.8
 * item(lambert([0, 0, 1], [0, 0, -1])) // => 0
 * ```
 */
export function lambert(n: TensorLike, l: TensorLike): Value {
  const [nv, lv] = promoteFloat(n, l) as [Value, Value]
  return clamp0(cosine(nv, lv))
}

/**
 * Phong's specular term `max(0, r . v)^s`, with `r` the light direction mirrored about the normal. Differentiable
 * (also in the shininess, where the term is positive).
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param shininess - The exponent `s`.
 * @returns The specular terms `[...]`.
 * @concept phong-shading
 * @example
 * ```ts
 * import { phong } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(phong([0, 0, 1], [0, 0, 1], [0, 0, 1], 20)) // => 1
 * item(phong([0, 0, 1], [0.6, 0, 0.8], [-0.6, 0, 0.8], 10)) // => 1
 * ```
 */
export function phong(n: TensorLike, l: TensorLike, v: TensorLike, shininess: TensorLike): Value {
  const [nv, lv, vv, s] = promoteFloat(n, l, v, shininess) as [Value, Value, Value, Value]
  const r = reflect(mul(lv, -1), nv)
  return positivePow(cosine(r, vv), s)
}

/**
 * Blinn's modification of Phong's term, `max(0, n . h)^s` with the half vector `h`. Differentiable.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param shininess - The exponent `s`.
 * @returns The specular terms `[...]`.
 * @concept blinn-phong
 * @example
 * ```ts
 * import { blinnPhong } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(blinnPhong([0, 0, 1], [0.6, 0, 0.8], [-0.6, 0, 0.8], 50)) // => 1
 * ```
 */
export function blinnPhong(n: TensorLike, l: TensorLike, v: TensorLike, shininess: TensorLike): Value {
  const [nv, lv, vv, s] = promoteFloat(n, l, v, shininess) as [Value, Value, Value, Value]
  return positivePow(cosine(nv, halfVector(lv, vv)), s)
}

/**
 * The Lambertian BRDF `albedo / pi`: integrated against the cosine over the hemisphere it reflects exactly the albedo.
 *
 * @param albedo - Reflectances (any shape, often `[..., 3]`).
 * @returns The BRDF values, same shape.
 * @concept brdf
 * @example
 * ```ts
 * import { lambertBrdf } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(lambertBrdf(Math.PI)) // => 1
 * ```
 */
export function lambertBrdf(albedo: TensorLike): Value {
  const [a] = promoteFloat(albedo) as [Value]
  return div(a, Math.PI)
}

/**
 * The normalised (energy-conserving) Phong BRDF `(s + 2) / (2 pi) max(0, r . v)^s` (Lafortune and Willems): at normal
 * incidence it reflects exactly its weight over the hemisphere, and less at grazing angles.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param shininess - The exponent `s`.
 * @returns The BRDF values `[...]`.
 * @concept brdf
 * @example
 * ```ts
 * import { phongBrdf } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(phongBrdf([0, 0, 1], [0, 0, 1], [0, 0, 1], 0)) * Math.PI // => 1
 * ```
 */
export function phongBrdf(n: TensorLike, l: TensorLike, v: TensorLike, shininess: TensorLike): Value {
  const [s] = promoteFloat(shininess) as [Value]
  return mul(div(add(s, 2), 2 * Math.PI), phong(n, l, v, s))
}

/**
 * The normalised Blinn-Phong distribution of microfacet normals `D(h) = (s + 2) / (2 pi) cos^s theta_h`, which
 * integrates to one against `cos theta_h` over the hemisphere.
 *
 * @param cosH - Cosines between the normal and the half vector `[...]`.
 * @param shininess - The exponent `s`.
 * @returns The densities `[...]`.
 * @concept microfacet
 * @example
 * ```ts
 * import { blinnPhongDistribution } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(blinnPhongDistribution(1, 0)) * Math.PI // => 1
 * ```
 */
export function blinnPhongDistribution(cosH: TensorLike, shininess: TensorLike): Value {
  const [c, s] = promoteFloat(cosH, shininess) as [Value, Value]
  return mul(div(add(s, 2), 2 * Math.PI), positivePow(c, s))
}

/**
 * The normalised Blinn-Phong BRDF `(s + 8) / (8 pi) max(0, n . h)^s` (the approximation in Akenine-Moller, Haines
 * and Hoffman's Real-Time Rendering), which keeps highlights about equally bright as the shininess changes.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param shininess - The exponent `s`.
 * @returns The BRDF values `[...]`.
 * @concept brdf
 * @example
 * ```ts
 * import { blinnPhongBrdf } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(blinnPhongBrdf([0, 0, 1], [0, 0, 1], [0, 0, 1], 0)) * Math.PI // => 1
 * ```
 */
export function blinnPhongBrdf(n: TensorLike, l: TensorLike, v: TensorLike, shininess: TensorLike): Value {
  const [s] = promoteFloat(shininess) as [Value]
  return mul(div(add(s, 8), 8 * Math.PI), blinnPhong(n, l, v, s))
}

/**
 * The GGX (Trowbridge-Reitz) distribution of microfacet normals,
 * `D(h) = alpha^2 / (pi ((n . h)^2 (alpha^2 - 1) + 1)^2)` for `n . h > 0` (zero otherwise), with `alpha` the square of
 * the perceptual roughness. It integrates to one against `cos theta_h`. Differentiable.
 *
 * @param cosH - Cosines between the normal and the half vector `[...]`.
 * @param alpha - The width `alpha` (roughness squared), positive.
 * @returns The densities `[...]`.
 * @concept microfacet
 * @example
 * ```ts
 * import { ggxDistribution } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(ggxDistribution(1, 1)) * Math.PI // => 1
 * item(ggxDistribution(1, 0.5)) * Math.PI // => 4
 * ```
 */
export function ggxDistribution(cosH: TensorLike, alpha: TensorLike): Value {
  const [c, a] = promoteFloat(cosH, alpha) as [Value, Value]
  const a2 = square(a)
  const d = add(mul(square(c), sub(a2, 1)), 1)
  return where(greater(c, 0), div(a2, mul(square(d), Math.PI)), 0)
}

/**
 * Smith's masking function for GGX, `G1(v) = 2 (n . v) / (n . v + sqrt(alpha^2 + (1 - alpha^2) (n . v)^2))`: the
 * fraction of microfacets facing `v` that are not hidden by others. One at normal incidence. Differentiable.
 *
 * @param cosV - Cosines between the normal and the direction `[...]`, positive.
 * @param alpha - The width `alpha` (roughness squared).
 * @returns The masking fractions `[...]`, in [0, 1].
 * @concept microfacet
 * @example
 * ```ts
 * import { smithG1 } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(smithG1(1, 0.5)) // => 1
 * ```
 */
export function smithG1(cosV: TensorLike, alpha: TensorLike): Value {
  const [c, a] = promoteFloat(cosV, alpha) as [Value, Value]
  const a2 = square(a)
  return div(mul(c, 2), add(c, sqrt(add(a2, mul(sub(1, a2), square(c))))))
}

/**
 * The separable Smith shadowing-masking term `G(l, v) = G1(l) G1(v)` for GGX.
 *
 * @param cosL - Cosines between the normal and the light direction `[...]`.
 * @param cosV - Cosines between the normal and the view direction `[...]`.
 * @param alpha - The width `alpha` (roughness squared).
 * @returns The terms `[...]`.
 * @concept microfacet
 * @example
 * ```ts
 * import { smithG } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(smithG(1, 1, 0.3)) // => 1
 * ```
 */
export function smithG(cosL: TensorLike, cosV: TensorLike, alpha: TensorLike): Value {
  return mul(smithG1(cosL, alpha), smithG1(cosV, alpha))
}

/**
 * Schlick's approximation of Fresnel reflectance, `F = f0 + (1 - f0) (1 - cos theta)^5`, with `f0` the reflectance
 * at normal incidence (a scalar for dielectrics, an RGB colour for metals). Differentiable.
 *
 * @param cosTheta - Cosines of the angle of incidence `[...]` (give a trailing axis of 1 to broadcast against RGB).
 * @param f0 - Reflectances at normal incidence.
 * @returns The reflectances, broadcast.
 * @concept fresnel-equations
 * @example
 * ```ts
 * import { fresnelSchlick } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * item(fresnelSchlick(1, 0.04)) // => 0.04
 * item(fresnelSchlick(0, 0.04)) // => 1
 * ```
 */
export function fresnelSchlick(cosTheta: TensorLike, f0: TensorLike): Value {
  const [c, f] = promoteFloat(cosTheta, f0) as [Value, Value]
  return add(f, mul(sub(1, f), pow(clamp0(sub(1, c)), 5)))
}

/**
 * The exact Fresnel reflectance of unpolarised light at a smooth dielectric boundary from index `n1` into `n2`: the
 * mean of the s and p reflectances, and 1 under total internal reflection. Differentiable away from the critical angle.
 *
 * @param cosI - Cosines of the angle of incidence `[...]`, in [0, 1].
 * @param n1 - Refractive index on the incident side.
 * @param n2 - Refractive index on the far side.
 * @returns The reflectances `[...]`.
 * @concept fresnel-equations
 * @example
 * ```ts
 * import { fresnelDielectric } from 'mdml-engine/graphics'
 * import { item } from 'mdml-engine/tensor'
 * Math.round(item(fresnelDielectric(1, 1, 1.5)) * 1e12) / 1e12 // => 0.04
 * item(fresnelDielectric(0.1, 1.5, 1)) // => 1
 * ```
 */
export function fresnelDielectric(cosI: TensorLike, n1: TensorLike, n2: TensorLike): Value {
  const [ci, a, b] = promoteFloat(cosI, n1, n2) as [Value, Value, Value]
  const eta = div(a, b)
  const sin2t = mul(square(eta), sub(1, square(ci)))
  const tir = greater(sin2t, 1)
  // The unused branch is evaluated at a safe point, so total internal reflection has a zero (not NaN) derivative.
  const ct = sqrt(maximum(sub(1, where(tir, 0, sin2t)), 1e-300))
  const rs = div(sub(mul(a, ci), mul(b, ct)), add(mul(a, ci), mul(b, ct)))
  const rp = div(sub(mul(a, ct), mul(b, ci)), add(mul(a, ct), mul(b, ci)))
  return where(tir, 1, div(add(square(rs), square(rp)), 2))
}

/**
 * The Cook-Torrance microfacet specular BRDF `D G F / (4 (n . l) (n . v))` with the GGX distribution, Smith-GGX
 * shadowing and Schlick's Fresnel term; `alpha = roughness^2`. Zero where the light or viewer is below the surface.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param roughness - Perceptual roughness in (0, 1].
 * @param f0 - Reflectance at normal incidence: a scalar, or RGB `[..., 3]`.
 * @returns The BRDF values `[..., 1]` for scalar `f0` (or `[..., 3]` for RGB).
 * @concept microfacet
 * @example
 * ```ts
 * import { cookTorrance, ggxDistribution } from 'mdml-engine/graphics'
 * import { toFlat, item } from 'mdml-engine/tensor'
 * const f = toFlat(cookTorrance([0, 0, 1], [0, 0, 1], [0, 0, 1], 0.5, 1))[0]
 * Math.abs(f - item(ggxDistribution(1, 0.25)) / 4) < 1e-12 // => true
 * ```
 */
export function cookTorrance(
  n: TensorLike,
  l: TensorLike,
  v: TensorLike,
  roughness: TensorLike,
  f0: TensorLike,
): Value {
  const [nv, lv, vv, r, f] = promoteFloat(n, l, v, roughness, f0) as [Value, Value, Value, Value, Value]
  const alpha = square(r)
  const h = halfVector(lv, vv)
  const nl = cosine(nv, lv)
  const nvv = cosine(nv, vv)
  const visible = greater(mul(clamp0(nl), clamp0(nvv)), 0)
  const safeL = maximum(nl, 1e-12)
  const safeV = maximum(nvv, 1e-12)
  const dg = div(mul(ggxDistribution(cosine(nv, h), alpha), smithG(safeL, safeV, alpha)), mul(mul(safeL, safeV), 4))
  const fr = fresnelSchlick(expandDims(clamp0(cosine(h, vv)), -1), f)
  return where(expandDims(visible, -1), mul(expandDims(dg, -1), fr), 0)
}

/**
 * The metallic-roughness BRDF of glTF and most game engines: a Cook-Torrance specular lobe with
 * `f0 = mix(0.04, albedo, metallic)` plus a Lambertian diffuse lobe `(1 - F)(1 - metallic) albedo / pi`.
 *
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param albedo - Base colours `[..., 3]`.
 * @param roughness - Perceptual roughness in (0, 1].
 * @param metallic - Metalness in [0, 1].
 * @returns The BRDF values `[..., 3]`.
 * @concept microfacet
 * @example
 * ```ts
 * import { metallicRoughness } from 'mdml-engine/graphics'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(metallicRoughness([0, 0, 1], [0, 0, -1], [0, 0, 1], [1, 1, 1], 0.5, 0)) // => [0, 0, 0]
 * ```
 */
export function metallicRoughness(
  n: TensorLike,
  l: TensorLike,
  v: TensorLike,
  albedo: TensorLike,
  roughness: TensorLike,
  metallic: TensorLike,
): Value {
  const [nv, lv, vv, a, r, m] = promoteFloat(n, l, v, albedo, roughness, metallic) as Value[] as [
    Value,
    Value,
    Value,
    Value,
    Value,
    Value,
  ]
  const mk = expandDims(m, -1)
  const f0 = add(mul(sub(1, mk), 0.04), mul(mk, a))
  const spec = cookTorrance(nv, lv, vv, r, f0)
  const h = halfVector(lv, vv)
  const fr = fresnelSchlick(expandDims(clamp0(cosine(h, vv)), -1), f0)
  const diffuse = mul(mul(sub(1, fr), sub(1, mk)), lambertBrdf(a))
  const lit = expandDims(greater(mul(clamp0(cosine(nv, lv)), clamp0(cosine(nv, vv))), 0), -1)
  return add(spec, where(lit, diffuse, 0))
}

// ------------------------------------------------------------------------------------------------------- materials

/** The shading models a {@link Material} can use. */
export type ShadingModel = 'lambert' | 'phong' | 'blinn-phong' | 'cook-torrance'

/**
 * How a surface reflects light: a shading model with its parameters, plus the ideal mirror, transparent and emissive
 * parts that ray and path tracers use. Colours are linear RGB.
 */
export interface Material {
  readonly model: ShadingModel
  /** Diffuse (and, for Cook-Torrance, base) colour. */
  readonly albedo: Vec3
  /** Specular weight of the Phong and Blinn-Phong lobes. */
  readonly specular: Vec3
  readonly shininess: number
  /** Cook-Torrance perceptual roughness in (0, 1]. */
  readonly roughness: number
  /** Cook-Torrance metalness in [0, 1]. */
  readonly metallic: number
  /** Emitted radiance. */
  readonly emission: Vec3
  /** Weight of the ideal mirror reflection (0 to 1). */
  readonly mirror: number
  /** Weight of the ideal refraction (0 to 1); refracting surfaces use their Fresnel reflectance as the mirror weight. */
  readonly transparency: number
  /** Refractive index of the inside. */
  readonly ior: number
}

/**
 * A material: defaults are a grey Lambertian surface (albedo 0.8), specular 0.04 with shininess 32, roughness 0.5, no
 * metal, emission, mirror or transparency, and the refractive index of glass (1.5).
 *
 * @param options - Any material fields.
 * @returns The material.
 * @throws MdmlError with code `value` for weights outside [0, 1] or a non-positive index or roughness.
 * @example
 * ```ts
 * import { material } from 'mdml-engine/graphics'
 * material({ model: 'phong', albedo: [1, 0, 0] }).shininess // => 32
 * ```
 */
export function material(options: Partial<Material> = {}): Material {
  const m: Material = {
    model: options.model ?? 'lambert',
    albedo: check3(options.albedo ?? [0.8, 0.8, 0.8], 'albedo'),
    specular: check3(options.specular ?? [0.04, 0.04, 0.04], 'specular'),
    shininess: options.shininess ?? 32,
    roughness: options.roughness ?? 0.5,
    metallic: options.metallic ?? 0,
    emission: check3(options.emission ?? [0, 0, 0], 'emission'),
    mirror: options.mirror ?? 0,
    transparency: options.transparency ?? 0,
    ior: options.ior ?? 1.5,
  }
  const unit = (x: number): boolean => x >= 0 && x <= 1
  if (!(unit(m.metallic) && unit(m.mirror) && unit(m.transparency) && m.mirror + m.transparency <= 1)) {
    throw new MdmlError(
      'value',
      'material: metallic, mirror and transparency must be in [0, 1], mirror + transparency <= 1',
    )
  }
  if (!(m.ior > 0 && m.roughness > 0 && m.roughness <= 1 && m.shininess >= 0)) {
    throw new MdmlError('value', 'material: need ior > 0, roughness in (0, 1] and shininess >= 0')
  }
  return m
}

/**
 * The BRDF of a material for batches of normals, light directions and view directions (Lambert: `albedo / pi`;
 * Phong and Blinn-Phong: Lambert plus the normalised specular lobe weighted by `specular`; Cook-Torrance: the
 * metallic-roughness BRDF). Differentiable in the directions.
 *
 * @param m - The material.
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param albedo - Per-row albedo `[..., 3]` replacing the material's (texture-mapped surfaces).
 * @returns The BRDF values `[..., 3]`.
 * @concept brdf
 * @example
 * ```ts
 * import { brdf, material } from 'mdml-engine/graphics'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(brdf(material({ albedo: [Math.PI, 0, 0] }), [0, 0, 1], [0, 0, 1], [0, 0, 1])) // => [1, 0, 0]
 * ```
 */
export function brdf(m: Material, n: TensorLike, l: TensorLike, v: TensorLike, albedo?: TensorLike): Value {
  const [nv, lv, vv, a] = promoteFloat(n, l, v, albedo ?? m.albedo) as [Value, Value, Value, Value]
  const shape = broadcastShapes(broadcastShapes(broadcastShapes(nv.shape, lv.shape), vv.shape), a.shape)
  const diffuse = (): Value => broadcastTo(lambertBrdf(a), shape)
  switch (m.model) {
    case 'lambert':
      return diffuse()
    case 'phong':
      return add(diffuse(), mul(expandDims(phongBrdf(nv, lv, vv, m.shininess), -1), m.specular))
    case 'blinn-phong':
      return add(diffuse(), mul(expandDims(blinnPhongBrdf(nv, lv, vv, m.shininess), -1), m.specular))
    case 'cook-torrance':
      return metallicRoughness(nv, lv, vv, a, m.roughness, m.metallic)
  }
}

/**
 * Outgoing radiance towards the viewer from one light sample per row: `brdf(n, l, v) * E * max(0, n . l)`, with `E`
 * the irradiance the light delivers at normal incidence (from {@link lightSamples}).
 *
 * @param m - The material.
 * @param n - Unit normals `[..., 3]`.
 * @param l - Unit directions towards the light `[..., 3]`.
 * @param v - Unit directions towards the viewer `[..., 3]`.
 * @param irradiance - Normal-incidence irradiance per row `[..., 3]`.
 * @param albedo - Optional per-row albedo `[..., 3]` replacing the material's.
 * @returns Radiance `[..., 3]`.
 * @concept rendering-equation
 * @example
 * ```ts
 * import { material, shade } from 'mdml-engine/graphics'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(shade(material({ albedo: [1, 1, 1] }), [0, 0, 1], [0, 0, 1], [0, 0, 1], [Math.PI, 0, 0])) // => [1, 0, 0]
 * ```
 */
export function shade(
  m: Material,
  n: TensorLike,
  l: TensorLike,
  v: TensorLike,
  irradiance: TensorLike,
  albedo?: TensorLike,
): Value {
  const [nv, lv, vv, e] = promoteFloat(n, l, v, irradiance) as [Value, Value, Value, Value]
  return mul(mul(brdf(m, nv, lv, vv, albedo), e), expandDims(lambert(nv, lv), -1))
}

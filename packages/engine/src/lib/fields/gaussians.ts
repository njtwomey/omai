/**
 * The set of 3-D Gaussians of Gaussian splatting (Kerbl et al. 2023): each has a mean, an anisotropic extent given by
 * log-scales and a rotation (a quaternion), an opacity (as a logit, so it stays in (0, 1)), and a colour, the zeroth
 * spherical-harmonic coefficient with optionally the three degree-1 coefficients that make it depend on the viewing
 * direction. A set is a plain record of tensors, so it is a pytree the library's optimisers update, and its covariance
 * `Sigma = R S S^T R^T` is built from the rotation and scales so that it stays positive definite. Initialisation is from
 * a point cloud (scales from the distance to the nearest points) or at random in a box, from a key.
 */

import { MdmlError } from '../base/index.ts'
import { kNearest } from '../numerics/spatial/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  einsum,
  exp,
  expandDims,
  fromStorage,
  max,
  maximum,
  mul,
  reshape,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/** The zeroth spherical-harmonic constant `1 / (2 sqrt(pi))`. */
export const SH_C0 = 0.28209479177387814
/** The degree-1 spherical-harmonic constant `sqrt(3 / (4 pi))`. */
export const SH_C1 = 0.4886025119029199

/**
 * A set of `n` 3-D Gaussians as a pytree of tensors.
 *
 * @index structure A set of 3-D Gaussians as a pytree of tensors.
 */
export interface GaussianSet {
  /** Centres `[n, 3]`. */
  readonly means: Value
  /** Natural logarithms of the standard deviations along the Gaussian's own axes `[n, 3]`. */
  readonly logScales: Value
  /** Rotations as quaternions `(w, x, y, z)` `[n, 4]`, normalised when used. */
  readonly rotations: Value
  /** Opacities as logits `[n]`: the opacity is the sigmoid. */
  readonly opacityLogits: Value
  /** The zeroth spherical-harmonic coefficients of the colour `[n, 3]`: the colour is `0.5 + C0 * coefficient`. */
  readonly colours: Value
  /** The degree-1 coefficients `[n, 3, 3]` (the y, z and x terms, each an RGB triple), for colour that depends on the view. */
  readonly sh?: Value
}

const rows = (v: Value): number => v.shape[0] ?? 0

/**
 * The number of Gaussians in a set.
 *
 * @param set - The set.
 * @returns The count.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussianCount, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * gaussianCount(randomGaussians(key(0), 7)) // => 7
 * ```
 */
export function gaussianCount(set: GaussianSet): number {
  return rows(set.means)
}

/**
 * The coefficient that gives a colour: the zeroth spherical-harmonic coefficient of an RGB colour, `(rgb - 0.5) / C0`.
 *
 * @param rgb - Colours `[n, 3]` in [0, 1].
 * @returns The coefficients `[n, 3]`.
 * @example
 * ```ts
 * import { colourToCoefficient } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(colourToCoefficient([[0.5, 0.5, 0.5]])) // => [[0, 0, 0]]
 * ```
 */
export function colourToCoefficient(rgb: TensorLike): Value {
  return div(sub(asTensor(rgb) as Value, 0.5), SH_C0)
}

/** Options for creating Gaussians. */
export interface GaussianInit {
  /** The initial standard deviation (a number, the same for all) when not set from neighbours (default 0.1). */
  readonly scale?: number
  /** The initial opacity in (0, 1) (default 0.1). */
  readonly opacity?: number
  /** Degree of the spherical harmonics: 0 (a colour per Gaussian, default) or 1 (view-dependent). */
  readonly shDegree?: 0 | 1
}

const logit = (p: number): number => Math.log(p / (1 - p))

function assemble(means: Value, logScales: Value, colours: Value, options: GaussianInit): GaussianSet {
  const n = rows(means)
  const q = new Float64Array(n * 4)
  for (let i = 0; i < n; i++) q[4 * i] = 1
  return {
    means,
    logScales,
    rotations: fromStorage(q, [n, 4], 'float64'),
    opacityLogits: fromStorage(new Float64Array(n).fill(logit(options.opacity ?? 0.1)), [n], 'float64'),
    colours,
    ...(options.shDegree === 1 ? { sh: fromStorage(new Float64Array(n * 9), [n, 3, 3], 'float64') } : {}),
  }
}

/**
 * Gaussians scattered uniformly in a box, with grey-ish random colours: a start for fitting a scene when there is no
 * point cloud. Rotations are the identity.
 *
 * @param key - The key.
 * @param count - The number of Gaussians.
 * @param options - The box half-width (default 1), the scale, opacity and spherical-harmonic degree.
 * @returns The set.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const g = randomGaussians(key(0), 5, { boxHalfWidth: 0.5 })
 * [g.means.shape, g.colours.shape, g.rotations.shape] // => [[5, 3], [5, 3], [5, 4]]
 * ```
 */
export function randomGaussians(
  key: Key,
  count: number,
  options: GaussianInit & { readonly boxHalfWidth?: number } = {},
): GaussianSet {
  const half = options.boxHalfWidth ?? 1
  const means = uniform(foldIn(key, 0), [count, 3], { low: -half, high: half })
  const rgb = uniform(foldIn(key, 1), [count, 3], { low: 0.3, high: 0.7 })
  const logScales = fromStorage(new Float64Array(count * 3).fill(Math.log(options.scale ?? 0.1)), [count, 3], 'float64')
  return assemble(means, logScales, colourToCoefficient(rgb), options)
}

/**
 * Gaussians at the points of a cloud, as 3DGS initialises from a structure-from-motion cloud: each takes its point's
 * colour, and its scale is the mean distance to its `neighbours` nearest points (isotropic), unless `scale` is given.
 *
 * @param points - The points `[n, 3]`.
 * @param colours - Their colours `[n, 3]` in [0, 1] (default grey).
 * @param options - Neighbours for the scale (default 3), a fixed scale, the opacity and the harmonic degree.
 * @returns The set.
 * @throws MdmlError with code `shape` for points that are not `[n, 3]` or colours of another count.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussiansFromPoints } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * // Three collinear points 1 apart: the middle one's two nearest neighbours are at distance 1.
 * const g = gaussiansFromPoints([[0, 0, 0], [1, 0, 0], [2, 0, 0]], undefined, { neighbours: 2 })
 * toArray(g.logScales)[1] // => [0, 0, 0]
 * ```
 */
export function gaussiansFromPoints(
  points: TensorLike,
  colours?: TensorLike,
  options: GaussianInit & { readonly neighbours?: number } = {},
): GaussianSet {
  const p = asTensor(points) as Tensor
  if (p.shape.length !== 2 || p.shape[1] !== 3)
    throw new MdmlError('shape', `gaussiansFromPoints: points must be [n, 3], got [${p.shape.join(', ')}]`)
  const n = rows(p)
  const rgb =
    colours === undefined
      ? fromStorage(new Float64Array(n * 3).fill(0.5), [n, 3], 'float64')
      : (asTensor(colours) as Tensor)
  if (rgb.shape[0] !== n) throw new MdmlError('shape', 'gaussiansFromPoints: one colour per point')
  let scales: Float64Array
  if (options.scale !== undefined || n < 2) scales = new Float64Array(n).fill(options.scale ?? 0.1)
  else {
    const k = Math.min(options.neighbours ?? 3, n - 1)
    const d = toFlat(kNearest(p, undefined, k, { excludeSelf: true }).distances)
    scales = Float64Array.from({ length: n }, (_, i) => {
      let s = 0
      for (let j = 0; j < k; j++) s += d[i * k + j] as number
      return Math.max(s / k, 1e-7)
    })
  }
  const logScales = new Float64Array(n * 3)
  for (let i = 0; i < n; i++) logScales.fill(Math.log(scales[i] as number), 3 * i, 3 * i + 3)
  return assemble(p, fromStorage(logScales, [n, 3], 'float64'), colourToCoefficient(rgb), options)
}

/**
 * Rotation matrices of quaternions `(w, x, y, z)` (normalised first), `[n, 3, 3]`. Differentiable.
 *
 * @param q - Quaternions `[n, 4]`.
 * @returns The rotation matrices.
 * @concept rotations
 * @example
 * ```ts
 * import { quaternionToMatrix } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * // A quarter turn about z: the x axis goes to the y axis.
 * const h = Math.SQRT1_2
 * toArray(quaternionToMatrix([[h, 0, 0, h]]))[0].map((row) => row.map((v) => Math.round(v * 1e9) / 1e9 + 0)) // => [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
 * ```
 */
export function quaternionToMatrix(q: TensorLike): Value {
  const v = asTensor(q) as Value
  const n = rows(v)
  const unit = div(v, sqrt(sum(square(v), -1, true)))
  const c = (k: number): Value => squeeze(sliceAxis(unit, -1, k, k + 1), -1)
  const [w, x, y, z] = [c(0), c(1), c(2), c(3)] as [Value, Value, Value, Value]
  const two = (a: Value, b: Value): Value => mul(mul(a, b), 2)
  const entries = [
    sub(1, mul(add(square(y), square(z)), 2)),
    sub(two(x, y), two(w, z)),
    add(two(x, z), two(w, y)),
    add(two(x, y), two(w, z)),
    sub(1, mul(add(square(x), square(z)), 2)),
    sub(two(y, z), two(w, x)),
    sub(two(x, z), two(w, y)),
    add(two(y, z), two(w, x)),
    sub(1, mul(add(square(x), square(y)), 2)),
  ]
  return reshape(stack(entries, -1), [n, 3, 3])
}

/**
 * The factor `M = R S` of each Gaussian's covariance (`R` the rotation, `S` the diagonal of standard deviations), so
 * that `Sigma = M M^T`: `[n, 3, 3]`.
 *
 * @param set - The Gaussians.
 * @returns The factors.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { covarianceFactor, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * covarianceFactor(randomGaussians(key(0), 4)).shape // => [4, 3, 3]
 * ```
 */
export function covarianceFactor(set: GaussianSet): Value {
  return mul(quaternionToMatrix(set.rotations), expandDims(exp(set.logScales), 1))
}

/**
 * The 3-D covariance matrices `Sigma = R S S^T R^T` of the Gaussians `[n, 3, 3]` (symmetric positive definite).
 *
 * @param set - The Gaussians.
 * @returns The covariances.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussiansFromPoints, covariance3d } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * // An isotropic Gaussian of standard deviation 0.5 has covariance 0.25 I.
 * toArray(covariance3d(gaussiansFromPoints([[0, 0, 0]], undefined, { scale: 0.5 })))[0][0] // => [0.25, 0, 0]
 * ```
 */
export function covariance3d(set: GaussianSet): Value {
  const m = covarianceFactor(set)
  return einsum('nij,nkj->nik', m, m)
}

/**
 * The colour of each Gaussian seen along a direction: `max(0, 0.5 + C0 f0 + C1 (-y f1 + z f2 - x f3))` with the
 * degree-1 terms only when the set has them (`direction` the unit vector from the camera to the Gaussian, `[n, 3]`).
 *
 * @param set - The Gaussians.
 * @param direction - The viewing directions `[n, 3]` (needed for view-dependent colour).
 * @returns The colours `[n, 3]`.
 * @concept spherical-harmonics
 * @example
 * ```ts
 * import { gaussianColour, gaussiansFromPoints } from 'mdml-engine/fields'
 * import { toArray } from 'mdml-engine/tensor'
 * const g = gaussiansFromPoints([[0, 0, 0]], [[0.2, 0.4, 0.6]])
 * toArray(gaussianColour(g))[0].map((v) => Math.round(v * 1e9) / 1e9) // => [0.2, 0.4, 0.6]
 * ```
 */
export function gaussianColour(set: GaussianSet, direction?: TensorLike): Value {
  let c = add(mul(set.colours, SH_C0), 0.5)
  if (set.sh !== undefined && direction !== undefined) {
    const d = asTensor(direction) as Value
    const comp = (k: number): Value => expandDims(squeeze(sliceAxis(d, -1, k, k + 1), -1), -1)
    const term = (k: number): Value => squeeze(sliceAxis(set.sh as Value, 1, k, k + 1), 1)
    const harmonic = add(
      add(mul(mul(comp(1), term(0)), -SH_C1), mul(mul(comp(2), term(1)), SH_C1)),
      mul(mul(comp(0), term(2)), -SH_C1),
    )
    c = add(c, harmonic)
  }
  return maximum(c, 0)
}

/**
 * The opacities of the Gaussians `[n]`, in (0, 1).
 *
 * @param set - The Gaussians.
 * @returns The sigmoid of the logits.
 * @example
 * ```ts
 * import { gaussianOpacity, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * gaussianOpacity(randomGaussians(key(0), 1, { opacity: 0.25 })).data[0] // => 0.25
 * ```
 */
export function gaussianOpacity(set: GaussianSet): Value {
  return div(1, add(1, exp(mul(set.opacityLogits, -1))))
}

/**
 * A set with the Gaussians of the given indices, in that order (a Gaussian may be listed twice: cloning).
 *
 * @param set - The set.
 * @param indices - Indices into it.
 * @returns The selection.
 * @example
 * ```ts
 * import { selectGaussians, gaussianCount, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * gaussianCount(selectGaussians(randomGaussians(key(0), 5), [0, 0, 3])) // => 3
 * ```
 */
export function selectGaussians(set: GaussianSet, indices: readonly number[]): GaussianSet {
  const idx = fromStorage(Int32Array.from(indices), [indices.length], 'int32')
  return mapGaussians(set, (v) => takeRows(v, idx))
}

/**
 * Two sets joined: all of the first, then all of the second.
 *
 * @param a - The first set.
 * @param b - The second set (with the same harmonic degree).
 * @returns The union.
 * @example
 * ```ts
 * import { concatGaussians, gaussianCount, randomGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * gaussianCount(concatGaussians(randomGaussians(key(0), 2), randomGaussians(key(1), 3))) // => 5
 * ```
 */
export function concatGaussians(a: GaussianSet, b: GaussianSet): GaussianSet {
  return {
    means: concat([a.means, b.means], 0),
    logScales: concat([a.logScales, b.logScales], 0),
    rotations: concat([a.rotations, b.rotations], 0),
    opacityLogits: concat([a.opacityLogits, b.opacityLogits], 0),
    colours: concat([a.colours, b.colours], 0),
    ...(a.sh !== undefined && b.sh !== undefined ? { sh: concat([a.sh, b.sh], 0) } : {}),
  }
}

/** Apply a function to every tensor of a set. */
export function mapGaussians(set: GaussianSet, f: (v: Value) => Value): GaussianSet {
  return {
    means: f(set.means),
    logScales: f(set.logScales),
    rotations: f(set.rotations),
    opacityLogits: f(set.opacityLogits),
    colours: f(set.colours),
    ...(set.sh === undefined ? {} : { sh: f(set.sh) }),
  }
}

import { take } from '../tensor/index.ts'

const takeRows = (v: Value, idx: Value): Value => take(v, idx, 0)

/**
 * The largest standard deviation of each Gaussian `[n]` (world units): the largest of its scales.
 *
 * @param set - The set.
 * @returns The largest scales.
 * @example
 * ```ts
 * import { gaussiansFromPoints, largestScale } from 'mdml-engine/fields'
 * largestScale(gaussiansFromPoints([[0, 0, 0]], undefined, { scale: 0.3 })).data[0] // => 0.3
 * ```
 */
export function largestScale(set: GaussianSet): Value {
  return exp(max(set.logScales, -1))
}

/**
 * A sample from each Gaussian's distribution `[n, 3]`, as 3DGS splits a large Gaussian into two drawn from it:
 * `mean + M z` with `z` standard normal.
 *
 * @param key - The key.
 * @param set - The set.
 * @returns A point per Gaussian.
 * @example
 * ```ts
 * import { randomGaussians, sampleGaussians } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * sampleGaussians(key(0), randomGaussians(key(1), 4)).shape // => [4, 3]
 * ```
 */
export function sampleGaussians(key: Key, set: GaussianSet): Value {
  const z = normal(key, [gaussianCount(set), 3])
  return add(set.means, einsum('nij,nj->ni', covarianceFactor(set), z))
}

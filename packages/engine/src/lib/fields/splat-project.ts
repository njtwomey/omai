/**
 * Projecting 3-D Gaussians onto an image, as in 3D Gaussian splatting: the covariance `Sigma = R S S^T R^T` is moved to
 * camera space (`W Sigma W^T`) and projected through the Jacobian `J` of the perspective map at the Gaussian's centre
 * (the first-order, EWA splatting approximation), giving the 2-D covariance `Sigma' = J W Sigma W^T J^T` of an ellipse on
 * the image; 0.3 pixel squared is added to its diagonal (the paper's low-pass dilation, so a Gaussian never covers less
 * than about a pixel). Each Gaussian also gets its depth, its view-dependent colour and the radius (3 standard
 * deviations of the ellipse's longest axis) that culls it when it lies outside the image or behind the camera. All of
 * it is differentiable in the means, scales, rotations, opacities and colours.
 */

import { MdmlError } from '../base/index.ts'
import { cameraBasis, type PinholeCamera } from '../graphics/index.ts'
import {
  add,
  clip,
  concat,
  div,
  einsum,
  expandDims,
  fromStorage,
  matmul,
  mul,
  primalOf,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  take,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { covarianceFactor, type GaussianSet, gaussianColour, gaussianCount, gaussianOpacity } from './gaussians.ts'

/** The pixel-squared variance added to the diagonal of every projected covariance (the paper's anti-aliasing dilation). */
export const SPLAT_DILATION = 0.3

/** The Gaussians that survive culling, projected into an image. */
export interface Projected {
  /** The image size. */
  readonly width: number
  readonly height: number
  /** For each projected Gaussian, its index in the set it came from. */
  readonly index: Int32Array
  /** Centres in pixels `[m, 2]` (`x` to the right, `y` down). */
  readonly mean2d: Value
  /** The 2-D covariance entries `[m, 3]`: `(a, b, c)` for the matrix `[[a, b], [b, c]]`. */
  readonly cov2d: Value
  /** Depth along the view direction `[m]`. */
  readonly depth: Value
  /** The opacity `[m]` and the colour `[m, 3]` as seen from the camera. */
  readonly opacity: Value
  readonly colour: Value
  /** The extent of each ellipse in pixels, 3 standard deviations of its longest axis (concrete). */
  readonly radius: Float64Array
}

/** Options of {@link projectGaussians}. */
export interface ProjectOptions {
  /** Added to the projected centres `[n, 2]` (zeros, to read the gradient with respect to the screen position). */
  readonly screenOffsets?: Value
  /** Keep Gaussians that are outside the image or behind the camera (default false: they are culled). */
  readonly keepAll?: boolean
}

/** The pinhole's camera-to-pixel numbers and its rotation (rows: right, down, forward) for an image size. */
export function cameraFrame(
  camera: PinholeCamera,
  width: number,
  height: number,
): {
  readonly rotation: readonly (readonly number[])[]
  readonly focal: number
  readonly tanX: number
  readonly tanY: number
} {
  const { right, up, forward } = cameraBasis(camera)
  const tanY = Math.tan((camera.fov * Math.PI) / 360)
  return {
    rotation: [right, [-up[0], -up[1], -up[2]], forward],
    focal: height / (2 * tanY),
    tanX: tanY * (width / height),
    tanY,
  }
}

/**
 * Project Gaussians into an image: centres, 2-D covariances (with the dilation), depths, opacities and colours as seen from
 * the camera, culling Gaussians behind the near plane or whose ellipse (3 standard deviations) lies outside the image.
 * The Jacobian is evaluated at the centre with its ratios `x/z` and `y/z` clamped to 1.3 times the field of view, as the
 * reference implementation does, so a Gaussian near the edge does not shear without bound.
 *
 * @param set - The Gaussians.
 * @param camera - The pinhole camera.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param options - Screen offsets for reading view-space gradients, and `keepAll`.
 * @returns The projected Gaussians.
 * @throws MdmlError with code `value` for a camera whose near plane is not positive.
 * @concept gaussian-splatting
 * @reference Kerbl, Kopanas, Leimkuehler and Drettakis (2023), 3D Gaussian splatting for real-time radiance field
 *   rendering; Zwicker, Pfister, van Baar and Gross (2001), EWA splatting.
 * @example
 * ```ts
 * import { gaussiansFromPoints, projectGaussians } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * import { toArray } from 'mdml-engine/tensor'
 * // A Gaussian of standard deviation 0.5 at the origin, seen from 4 away with a 90 degree field of view in a 64 pixel
 * // image: its 2-D variance is (32 * 0.5 / 4)^2 = 16, plus the dilation 0.3.
 * const g = gaussiansFromPoints([[0, 0, 0]], undefined, { scale: 0.5 })
 * const p = projectGaussians(g, pinholeCamera({ eye: [0, 0, 4], fov: 90 }), 64, 64)
 * [toArray(p.mean2d), toArray(p.cov2d)[0].map((v) => Math.round(v * 1e9) / 1e9)] // => [[[32, 32]], [16.3, 0, 16.3]]
 * ```
 */
export function projectGaussians(
  set: GaussianSet,
  camera: PinholeCamera,
  width: number,
  height: number,
  options: ProjectOptions = {},
): Projected {
  if (!(camera.near > 0)) throw new MdmlError('value', 'projectGaussians: the camera’s near plane must be positive')
  const { rotation, focal, tanX, tanY } = cameraFrame(camera, width, height)
  const R = fromStorage(Float64Array.from(rotation.flat()), [3, 3], 'float64')
  const eye = fromStorage(Float64Array.from(camera.eye), [3], 'float64')
  const col = (v: Value, k: number): Value => squeeze(sliceAxis(v, -1, k, k + 1), -1)
  // Camera-space centres: x to the right, y down, z forward.
  const cam = matmul(sub(set.means, eye), transpose(R))
  const tz = col(cam, 2)
  const limX = 1.3 * tanX
  const limY = 1.3 * tanY
  const tx = mul(clip(div(col(cam, 0), tz), -limX, limX), tz)
  const ty = mul(clip(div(col(cam, 1), tz), -limY, limY), tz)
  // The rows of J W (2 x 3): J = [[f/z, 0, -f x/z^2], [0, f/z, -f y/z^2]] and W the rotation.
  const invZ = div(focal, tz)
  const slope = (t: Value): Value => div(mul(t, -focal), square(tz))
  const w = (r: number): Tensor => fromStorage(Float64Array.from(rotation[r] as number[]), [1, 3], 'float64')
  const a1 = add(mul(expandDims(invZ, -1), w(0)), mul(expandDims(slope(tx), -1), w(2)))
  const a2 = add(mul(expandDims(invZ, -1), w(1)), mul(expandDims(slope(ty), -1), w(2)))
  const m = covarianceFactor(set)
  const u1 = einsum('nj,nji->ni', a1, m)
  const u2 = einsum('nj,nji->ni', a2, m)
  const a = add(sum(square(u1), -1), SPLAT_DILATION)
  const b = sum(mul(u1, u2), -1)
  const c = add(sum(square(u2), -1), SPLAT_DILATION)
  let mean2d = concat(
    [
      expandDims(add(div(mul(col(cam, 0), focal), tz), width / 2), -1),
      expandDims(add(div(mul(col(cam, 1), focal), tz), height / 2), -1),
    ],
    -1,
  )
  if (options.screenOffsets !== undefined) mean2d = add(mean2d, options.screenOffsets)
  const cov2d = concat([expandDims(a, -1), expandDims(b, -1), expandDims(c, -1)], -1)
  const direction = (() => {
    const d = sub(set.means, eye)
    return div(d, sqrt(sum(square(d), -1, true)))
  })()
  const colour = gaussianColour(set, direction)
  const opacity = gaussianOpacity(set)
  // Culling, decided on the values (a Gaussian that is culled has no effect on the image, so no gradient).
  const n = gaussianCount(set)
  const depth = primalOf(tz).data
  const centre = primalOf(mean2d).data
  const ca = primalOf(a).data
  const cb = primalOf(b).data
  const cc = primalOf(c).data
  const radius = new Float64Array(n)
  const keep: number[] = []
  for (let i = 0; i < n; i++) {
    const mid = 0.5 * ((ca[i] as number) + (cc[i] as number))
    const det = (ca[i] as number) * (cc[i] as number) - (cb[i] as number) ** 2
    const lambda = mid + Math.sqrt(Math.max(0.1, mid * mid - det))
    radius[i] = Math.ceil(3 * Math.sqrt(lambda))
    const x = centre[2 * i] as number
    const y = centre[2 * i + 1] as number
    const inFront = (depth[i] as number) > camera.near
    const inside =
      x + (radius[i] as number) >= 0 &&
      x - (radius[i] as number) <= width &&
      y + (radius[i] as number) >= 0 &&
      y - (radius[i] as number) <= height
    if (options.keepAll === true || (inFront && inside)) keep.push(i)
  }
  const index = Int32Array.from(keep)
  const pick = (v: Value): Value => (keep.length === n ? v : take(v, fromStorage(index, [index.length], 'int32'), 0))
  return {
    width,
    height,
    index,
    mean2d: pick(mean2d),
    cov2d: pick(cov2d),
    depth: pick(tz),
    opacity: pick(opacity),
    colour: pick(colour),
    radius: Float64Array.from(keep, (i) => radius[i] as number),
  }
}

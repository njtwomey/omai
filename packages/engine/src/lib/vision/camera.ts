/**
 * The pinhole camera with lens distortion, as OpenCV models it: rotation vectors (`Rodrigues`, over `numerics/geometry`
 * rotations), projection of 3-D points with radial, tangential and rational distortion, iterative undistortion of
 * points, undistortion maps and undistorted images (resampled by `signal/image`'s `mapCoordinates`), the pose of a
 * known object from its image (`solvePnP`: a linear start refined by Levenberg-Marquardt, and a RANSAC wrapper) and
 * the reprojection error.
 *
 * @index function Pinhole camera with lens distortion: project, undistort, Rodrigues.
 */

import { MdmlError, type Vec3 } from '../base/index.ts'
import { rotationFromMatrix, rotationFromRotvec } from '../numerics/geometry/index.ts'
import type { Key } from '../random/index.ts'
import { image } from '../signal/index.ts'
import { ransac } from '../stats/index.ts'
import {
  add,
  concat,
  div,
  fromStorage,
  matmul,
  mul,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { type PointsLike, pointsOf, rasterOf, valuesOf } from './_image.ts'
import { det3, type MutableRows, matMul, nullVector, points3Of, refine, svdOf, transposeOf } from './_refine.ts'
import { getPerspectiveTransform } from './geometry.ts'

/** Distortion coefficients in OpenCV's order: k1, k2, p1, p2, then optionally k3, k4, k5, k6 (missing are 0). */
export type Distortion = readonly number[]

/** A camera pose: the rotation vector (axis times angle) and translation taking object to camera coordinates. */
export interface Pose {
  readonly rvec: [number, number, number]
  readonly tvec: [number, number, number]
}

/** Distortion coefficients padded to OpenCV's eight. */
const coeffs = (d: Distortion | undefined): number[] => Array.from({ length: 8 }, (_, i) => d?.[i] ?? 0)

/**
 * The rotation matrix of a rotation vector (cv2.Rodrigues): the rotation by |r| radians about r / |r|.
 *
 * @param rvec - The rotation vector.
 * @returns The 3 x 3 rotation matrix.
 * @concept rotations
 * @example
 * ```ts
 * import { rodrigues } from 'mdml-engine/vision'
 * rodrigues([0, 0, Math.PI / 2]).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
 * ```
 */
export function rodrigues(rvec: readonly number[]): MutableRows {
  return rotationFromRotvec(rvec).asMatrix()
}

/**
 * The rotation vector of a rotation matrix (cv2.Rodrigues, matrix to vector): axis times angle, the angle in [0, pi].
 *
 * @param r - A 3 x 3 rotation matrix.
 * @returns The rotation vector.
 * @concept rotations
 * @example
 * ```ts
 * import { rodriguesVector } from 'mdml-engine/vision'
 * rodriguesVector([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0, 1.570796326795]
 * ```
 */
export function rodriguesVector(r: readonly (readonly number[])[]): [number, number, number] {
  return rotationFromMatrix(r.map((row) => [...row])).asRotvec()
}

/** Normalised camera coordinates to distorted ones (OpenCV's model). */
function distort(x: number, y: number, k: readonly number[]): [number, number] {
  const r2 = x * x + y * y
  const r4 = r2 * r2
  const r6 = r4 * r2
  const radial =
    (1 + (k[0] ?? 0) * r2 + (k[1] ?? 0) * r4 + (k[4] ?? 0) * r6) /
    (1 + (k[5] ?? 0) * r2 + (k[6] ?? 0) * r4 + (k[7] ?? 0) * r6)
  const p1 = k[2] ?? 0
  const p2 = k[3] ?? 0
  return [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y]
}

/**
 * Project 3-D points into an image (cv2.projectPoints): x = R X + t, perspective division, OpenCV's distortion
 * (k1, k2, k3 radial over k4, k5, k6, and p1, p2 tangential), then the camera matrix.
 *
 * @param objectPoints - Points `[n, 3]` in object coordinates.
 * @param pose - The rotation vector and translation.
 * @param cameraMatrix - The 3 x 3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
 * @param distortion - Distortion coefficients (default none).
 * @returns Image points `[n, 2]`.
 * @concept camera-projection
 * @example
 * ```ts
 * import { projectPoints } from 'mdml-engine/vision'
 * projectPoints([[0, 0, 5], [1, 0, 5]], { rvec: [0, 0, 0], tvec: [0, 0, 0] }, [[100, 0, 50], [0, 100, 40], [0, 0, 1]]) // => [[50, 40], [70, 40]]
 * ```
 */
export function projectPoints(
  objectPoints: TensorLike | readonly (readonly number[])[],
  pose: Pose,
  cameraMatrix: readonly (readonly number[])[],
  distortion?: Distortion,
): [number, number][] {
  const X = points3Of(objectPoints, 'projectPoints')
  const R = rodrigues(pose.rvec)
  const t = pose.tvec
  const k = coeffs(distortion)
  const fx = cameraMatrix[0]?.[0] ?? 1
  const fy = cameraMatrix[1]?.[1] ?? 1
  const cx = cameraMatrix[0]?.[2] ?? 0
  const cy = cameraMatrix[1]?.[2] ?? 0
  return X.map(([a, b, c]) => {
    const xc = (R[0]?.[0] ?? 0) * a + (R[0]?.[1] ?? 0) * b + (R[0]?.[2] ?? 0) * c + t[0]
    const yc = (R[1]?.[0] ?? 0) * a + (R[1]?.[1] ?? 0) * b + (R[1]?.[2] ?? 0) * c + t[1]
    const zc = (R[2]?.[0] ?? 0) * a + (R[2]?.[1] ?? 0) * b + (R[2]?.[2] ?? 0) * c + t[2]
    const iz = zc !== 0 ? 1 / zc : 1
    const [xd, yd] = distort(xc * iz, yc * iz, k)
    return [fx * xd + cx, fy * yd + cy]
  })
}

/**
 * The rotation matrix of a rotation vector as a differentiable value: R = I + a K + b K^2 with K the cross-product
 * matrix of r, a = sin(t) / t and b = 2 sin^2(t / 2) / t^2 (accurate near zero), for refining poses by autodiff.
 *
 * @param r - The rotation vector `[3]`.
 * @returns The rotation matrix `[3, 3]`.
 * @concept rotations
 * @example
 * ```ts
 * import { rodriguesValue } from 'mdml-engine/vision'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(rodriguesValue(tensor([0, 0, 0]))) // => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * ```
 */
export function rodriguesValue(r: Value): Value {
  const x = reshape(sliceAxis(r, 0, 0, 1), [])
  const y = reshape(sliceAxis(r, 0, 1, 2), [])
  const z = reshape(sliceAxis(r, 0, 2, 3), [])
  const t2 = add(sum(mul(r, r)), 1e-30)
  const t = sqrt(t2)
  const a = div(sin(t), t)
  const s = sin(mul(t, 0.5))
  const b = div(mul(mul(s, s), 2), t2)
  const zero = mul(x, 0)
  const K = reshape(stack([zero, mul(z, -1), y, z, zero, mul(x, -1), mul(y, -1), x, zero]), [3, 3])
  const K2 = matmul(K, K)
  return add(
    add(
      tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
      mul(K, a),
    ),
    mul(K2, b),
  )
}

/**
 * Projected points as a differentiable function of the pose, intrinsics and distortion: the residual core of pose
 * refinement and calibration.
 *
 * @param objectPoints - Points `[n, 3]` (constant).
 * @param rvec - Rotation vector `[3]`.
 * @param tvec - Translation `[3]`.
 * @param intrinsics - [fx, fy, cx, cy].
 * @param distortion - Up to eight coefficients `[k]` (k1, k2, p1, p2, k3, k4, k5, k6).
 * @returns Image points `[n, 2]` (differentiable in every argument but the object points).
 * @concept camera-projection
 * @example
 * ```ts
 * import { projectValue } from 'mdml-engine/vision'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(projectValue(tensor([[1, 0, 5]]), tensor([0, 0, 0]), tensor([0, 0, 0]), tensor([100, 100, 50, 40]), tensor([0, 0, 0, 0]))) // => [[70, 40]]
 * ```
 */
export function projectValue(
  objectPoints: Value,
  rvec: Value,
  tvec: Value,
  intrinsics: Value,
  distortion: Value,
): Value {
  const R = rodriguesValue(rvec)
  const xc = add(matmul(objectPoints, transpose(R)), reshape(tvec, [1, 3]))
  const z = sliceAxis(xc, 1, 2, 3)
  const x = div(sliceAxis(xc, 1, 0, 1), z)
  const y = div(sliceAxis(xc, 1, 1, 2), z)
  const k = (i: number): Value | number =>
    i < (distortion.shape[0] ?? 0) ? reshape(sliceAxis(distortion, 0, i, i + 1), []) : 0
  const r2 = add(mul(x, x), mul(y, y))
  const r4 = mul(r2, r2)
  const r6 = mul(r4, r2)
  const num = add(add(add(mul(r2, k(0)), mul(r4, k(1))), mul(r6, k(4))), 1)
  const den = add(add(add(mul(r2, k(5)), mul(r4, k(6))), mul(r6, k(7))), 1)
  const radial = div(num, den)
  const xy = mul(x, y)
  const xd = add(add(mul(x, radial), mul(xy, mul(k(2), 2))), mul(add(r2, mul(mul(x, x), 2)), k(3)))
  const yd = add(add(mul(y, radial), mul(add(r2, mul(mul(y, y), 2)), k(2))), mul(xy, mul(k(3), 2)))
  const f = (i: number): Value => reshape(sliceAxis(intrinsics, 0, i, i + 1), [])
  return concat([add(mul(xd, f(0)), f(2)), add(mul(yd, f(1)), f(3))], 1)
}

/**
 * Undistort image points (cv2.undistortPoints): the normalised coordinates (x - cx) / fx, (y - cy) / fy, the
 * distortion removed by OpenCV's fixed-point iteration (five steps by default, as OpenCV), then optionally the
 * rectifying rotation R and a new camera matrix P (without P the result stays in normalised coordinates).
 *
 * @param points - Distorted image points `[n, 2]`.
 * @param cameraMatrix - The 3 x 3 intrinsic matrix.
 * @param distortion - Distortion coefficients.
 * @param options - R (3 x 3), P (3 x 3 or 3 x 4) and the iteration count.
 * @returns Undistorted points `[n, 2]`.
 * @concept lens-distortion
 * @example
 * ```ts
 * import { undistortPoints } from 'mdml-engine/vision'
 * undistortPoints([[70, 40]], [[100, 0, 50], [0, 100, 40], [0, 0, 1]], []) // => [[0.2, 0]]
 * ```
 */
export function undistortPoints(
  points: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  distortion: Distortion,
  options: {
    readonly R?: readonly (readonly number[])[]
    readonly P?: readonly (readonly number[])[]
    readonly iterations?: number
  } = {},
): [number, number][] {
  const pts = pointsOf(points, 'undistortPoints')
  const k = coeffs(distortion)
  const any = distortion.some((v) => v !== 0)
  const fx = cameraMatrix[0]?.[0] ?? 1
  const fy = cameraMatrix[1]?.[1] ?? 1
  const cx = cameraMatrix[0]?.[2] ?? 0
  const cy = cameraMatrix[1]?.[2] ?? 0
  let RR: MutableRows | undefined = options.R?.map((r) => [...r])
  if (options.P !== undefined) {
    const Vec3 = options.P.map((r) => [r[0] ?? 0, r[1] ?? 0, r[2] ?? 0])
    RR = matMul(
      Vec3,
      RR ?? [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    )
  }
  const iterations = options.iterations ?? 5
  return pts.map(([u, v]) => {
    let x = (u - cx) / fx
    let y = (v - cy) / fy
    if (any) {
      const x0 = x
      const y0 = y
      for (let j = 0; j < iterations; j++) {
        const r2 = x * x + y * y
        const icdist =
          (1 + (((k[7] ?? 0) * r2 + (k[6] ?? 0)) * r2 + (k[5] ?? 0)) * r2) /
          (1 + (((k[4] ?? 0) * r2 + (k[1] ?? 0)) * r2 + (k[0] ?? 0)) * r2)
        if (icdist < 0) {
          x = (u - cx) / fx
          y = (v - cy) / fy
          break
        }
        const dx = 2 * (k[2] ?? 0) * x * y + (k[3] ?? 0) * (r2 + 2 * x * x)
        const dy = (k[2] ?? 0) * (r2 + 2 * y * y) + 2 * (k[3] ?? 0) * x * y
        x = (x0 - dx) * icdist
        y = (y0 - dy) * icdist
      }
    }
    if (RR !== undefined) {
      const xx = (RR[0]?.[0] ?? 0) * x + (RR[0]?.[1] ?? 0) * y + (RR[0]?.[2] ?? 0)
      const yy = (RR[1]?.[0] ?? 0) * x + (RR[1]?.[1] ?? 0) * y + (RR[1]?.[2] ?? 0)
      const ww = 1 / ((RR[2]?.[0] ?? 0) * x + (RR[2]?.[1] ?? 0) * y + (RR[2]?.[2] ?? 0))
      return [xx * ww, yy * ww]
    }
    return [x, y]
  })
}

/**
 * The undistortion and rectification maps (cv2.initUndistortRectifyMap): for every pixel of the output image, the
 * position in the distorted input that it samples: back-project through the new camera matrix and the inverse
 * rectification, distort, project with the original camera matrix.
 *
 * @param cameraMatrix - The original 3 x 3 intrinsics.
 * @param distortion - Distortion coefficients.
 * @param size - The output size [width, height].
 * @param options - R (default identity) and the new camera matrix (default the original).
 * @returns mapX and mapY `[height, width]`.
 * @concept lens-distortion
 * @example
 * ```ts
 * import { initUndistortRectifyMap } from 'mdml-engine/vision'
 * initUndistortRectifyMap([[10, 0, 2], [0, 10, 1], [0, 0, 1]], [0.1], [4, 3]).mapX.shape // => [3, 4]
 * ```
 */
export function initUndistortRectifyMap(
  cameraMatrix: readonly (readonly number[])[],
  distortion: Distortion,
  size: readonly [number, number],
  options: {
    readonly R?: readonly (readonly number[])[]
    readonly newCameraMatrix?: readonly (readonly number[])[]
  } = {},
): { readonly mapX: Tensor; readonly mapY: Tensor } {
  const [w, h] = size
  const k = coeffs(distortion)
  const A = cameraMatrix
  const Ar = (options.newCameraMatrix ?? cameraMatrix).map((r) => [r[0] ?? 0, r[1] ?? 0, r[2] ?? 0])
  const R = options.R?.map((r) => [...r]) ?? [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
  const iR = invertTo(matMul(Ar, R))
  const fx = A[0]?.[0] ?? 1
  const fy = A[1]?.[1] ?? 1
  const cx = A[0]?.[2] ?? 0
  const cy = A[1]?.[2] ?? 0
  const mx = new Float64Array(w * h)
  const my = new Float64Array(w * h)
  for (let i = 0; i < h; i++)
    for (let j = 0; j < w; j++) {
      const X = (iR[0]?.[0] ?? 0) * j + (iR[0]?.[1] ?? 0) * i + (iR[0]?.[2] ?? 0)
      const Y = (iR[1]?.[0] ?? 0) * j + (iR[1]?.[1] ?? 0) * i + (iR[1]?.[2] ?? 0)
      const W = (iR[2]?.[0] ?? 0) * j + (iR[2]?.[1] ?? 0) * i + (iR[2]?.[2] ?? 0)
      const [xd, yd] = distort(X / W, Y / W, k)
      mx[i * w + j] = fx * xd + cx
      my[i * w + j] = fy * yd + cy
    }
  return { mapX: fromStorage(mx, [h, w], 'float64'), mapY: fromStorage(my, [h, w], 'float64') }
}

/** The inverse of a 3 x 3 matrix by its adjugate. */
function invertTo(m: MutableRows): MutableRows {
  const d = det3(m)
  if (d === 0) throw new MdmlError('value', 'singular 3 x 3 matrix')
  const a = (i: number, j: number): number => m[i]?.[j] ?? 0
  return [
    [
      a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1),
      a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2),
      a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1),
    ],
    [
      a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2),
      a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0),
      a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2),
    ],
    [
      a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0),
      a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1),
      a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0),
    ],
  ].map((r) => r.map((v) => v / d))
}

/**
 * Remap an image through coordinate maps (cv2.remap with bilinear interpolation, by `signal/image`'s
 * `mapCoordinates`): out(y, x) = in(mapY(y, x), mapX(y, x)), 0 outside the image; channels alike. OpenCV quantises
 * the sub-pixel position to 1/32 pixel; this interpolates exactly.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param mapX - Source columns `[h, w]`.
 * @param mapY - Source rows `[h, w]`.
 * @returns The remapped image `[h, w]` (or `[h, w, C]`).
 * @concept geometric-transform
 * @example
 * ```ts
 * import { remap } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(remap([[0, 10]], [[0.5]], [[0]])) // => [[5]]
 * ```
 */
export function remap(x: TensorLike, mapX: TensorLike, mapY: TensorLike): Tensor {
  const r = rasterOf(x, 'remap')
  const mx = valuesOf(mapX)
  const my = valuesOf(mapY)
  const [h = 0, w = 0] = mx.shape
  const n = h * w
  const out = new Float64Array(n * r.c)
  for (let k = 0; k < r.c; k++) {
    const plane = fromStorage(
      Float64Array.from({ length: r.h * r.w }, (_, i) => r.data[i * r.c + k] ?? 0),
      [r.h, r.w],
      'float64',
    )
    const coords = fromStorage(Float64Array.from([...my.data, ...mx.data]), [2, n], 'float64')
    const v = toFlat(image.mapCoordinates(plane, coords, { order: 1, mode: 'constant' }))
    for (let i = 0; i < n; i++) out[i * r.c + k] = v[i] ?? 0
  }
  return fromStorage(out, r.rank === 3 ? [h, w, r.c] : [h, w], 'float64')
}

/**
 * Undistort an image (cv2.undistort): the undistortion maps of the camera, then bilinear resampling.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param cameraMatrix - The 3 x 3 intrinsics.
 * @param distortion - Distortion coefficients.
 * @param options - The new camera matrix (default the original).
 * @returns The undistorted image, the input's shape.
 * @concept lens-distortion
 * @example
 * ```ts
 * import { undistortImage } from 'mdml-engine/vision'
 * undistortImage([[0, 1], [1, 0]], [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], [0.05]).shape // => [2, 2]
 * ```
 */
export function undistortImage(
  x: TensorLike,
  cameraMatrix: readonly (readonly number[])[],
  distortion: Distortion,
  options: { readonly newCameraMatrix?: readonly (readonly number[])[] } = {},
): Tensor {
  const r = rasterOf(x, 'undistortImage')
  const { mapX, mapY } = initUndistortRectifyMap(cameraMatrix, distortion, [r.w, r.h], options)
  return remap(x, mapX, mapY)
}

/**
 * The root-mean-square reprojection error of a pose: the distance between observed image points and the projections
 * of their object points.
 *
 * @param objectPoints - Points `[n, 3]`.
 * @param imagePoints - Observed points `[n, 2]`.
 * @param pose - The pose.
 * @param cameraMatrix - The intrinsics.
 * @param distortion - Distortion coefficients.
 * @returns sqrt(mean of squared distances).
 * @concept camera-projection
 * @example
 * ```ts
 * import { reprojectionError } from 'mdml-engine/vision'
 * reprojectionError([[0, 0, 5]], [[53, 44]], { rvec: [0, 0, 0], tvec: [0, 0, 0] }, [[100, 0, 50], [0, 100, 40], [0, 0, 1]]) // => 5
 * ```
 */
export function reprojectionError(
  objectPoints: TensorLike | readonly (readonly number[])[],
  imagePoints: PointsLike,
  pose: Pose,
  cameraMatrix: readonly (readonly number[])[],
  distortion?: Distortion,
): number {
  const proj = projectPoints(objectPoints, pose, cameraMatrix, distortion)
  const obs = pointsOf(imagePoints, 'reprojectionError')
  if (obs.length !== proj.length) throw new MdmlError('shape', 'reprojectionError: one image point per object point')
  let s = 0
  proj.forEach(([u, v], i) => {
    const [a, b] = obs[i] as [number, number]
    s += (u - a) ** 2 + (v - b) ** 2
  })
  return Math.sqrt(s / Math.max(1, proj.length))
}

/** The nearest rotation to a matrix (by its SVD, with the determinant made positive) and the scale it removed. */
function nearestRotation(m: MutableRows): { R: MutableRows; scale: number } {
  const { u, s, v } = svdOf(m)
  let R = matMul(u, transposeOf(v))
  if (det3(R) < 0) {
    const uf = u.map((r) => [r[0] ?? 0, r[1] ?? 0, -(r[2] ?? 0)])
    R = matMul(uf, transposeOf(v))
  }
  return { R, scale: (s[0] ?? 0) + (s[1] ?? 0) + (s[2] ?? 0) }
}

/** A linear pose from correspondences in normalised camera coordinates: the planar homography or the 3-D DLT. */
function linearPose(X: readonly Vec3[], x: readonly [number, number][]): Pose {
  const planar = X.every((p) => Math.abs(p[2]) < 1e-12)
  let R: MutableRows
  let t: number[]
  if (planar) {
    const src = X.map(([a, b]) => [a, b] as [number, number])
    const H =
      X.length === 4
        ? getPerspectiveTransform(src, x)
        : (() => {
            const rows: number[][] = []
            src.forEach(([a, b], i) => {
              const [u, v] = x[i] as [number, number]
              rows.push([a, b, 1, 0, 0, 0, -u * a, -u * b, -u], [0, 0, 0, a, b, 1, -v * a, -v * b, -v])
            })
            const h = nullVector(rows)
            return [h.slice(0, 3), h.slice(3, 6), h.slice(6, 9)]
          })()
    const c0 = H.map((r) => r[0] ?? 0)
    const c1 = H.map((r) => r[1] ?? 0)
    const c2 = H.map((r) => r[2] ?? 0)
    const n0 = Math.hypot(...c0)
    const n1 = Math.hypot(...c1)
    let lambda = 2 / (n0 + n1)
    if ((c2[2] ?? 0) * lambda < 0) lambda = -lambda
    const r0 = c0.map((v) => v * lambda)
    const r1 = c1.map((v) => v * lambda)
    const r2 = [
      (r0[1] ?? 0) * (r1[2] ?? 0) - (r0[2] ?? 0) * (r1[1] ?? 0),
      (r0[2] ?? 0) * (r1[0] ?? 0) - (r0[0] ?? 0) * (r1[2] ?? 0),
      (r0[0] ?? 0) * (r1[1] ?? 0) - (r0[1] ?? 0) * (r1[0] ?? 0),
    ]
    R = nearestRotation([0, 1, 2].map((i) => [r0[i] ?? 0, r1[i] ?? 0, r2[i] ?? 0])).R
    t = c2.map((v) => v * lambda)
  } else {
    if (X.length < 6) throw new MdmlError('value', 'solvePnP: six non-coplanar points are needed')
    const rows: number[][] = []
    X.forEach(([a, b, c], i) => {
      const [u, v] = x[i] as [number, number]
      rows.push([a, b, c, 1, 0, 0, 0, 0, -u * a, -u * b, -u * c, -u])
      rows.push([0, 0, 0, 0, a, b, c, 1, -v * a, -v * b, -v * c, -v])
    })
    let p = nullVector(rows)
    // P is s [R | t]: choose the sign that makes the left 3 x 3 block a proper rotation (s > 0).
    if (det3([p.slice(0, 3), p.slice(4, 7), p.slice(8, 11)]) < 0) p = p.map((v) => -v)
    const { R: R0, scale } = nearestRotation([p.slice(0, 3), p.slice(4, 7), p.slice(8, 11)])
    R = R0
    t = [p[3] ?? 0, p[7] ?? 0, p[11] ?? 0].map((v) => (v * 3) / scale)
  }
  return { rvec: rodriguesVector(R), tvec: [t[0] ?? 0, t[1] ?? 0, t[2] ?? 0] }
}

/**
 * The pose of an object from its image (cv2.solvePnP with SOLVEPNP_ITERATIVE): a linear start (the plane-to-image
 * homography for planar objects, the direct linear transform otherwise) on undistorted points, refined by
 * Levenberg-Marquardt on the reprojection error.
 *
 * @param objectPoints - Points `[n, 3]` (n >= 4 coplanar, or n >= 6).
 * @param imagePoints - Observed points `[n, 2]`.
 * @param cameraMatrix - The intrinsics.
 * @param distortion - Distortion coefficients (default none).
 * @param options - An initial pose to refine from instead of the linear start.
 * @returns The pose.
 * @concept pose-estimation
 * @example
 * ```ts
 * import { projectPoints, solvePnP } from 'mdml-engine/vision'
 * const K = [[200, 0, 64], [0, 200, 48], [0, 0, 1]]
 * const X = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.2, 0]]
 * const x = projectPoints(X, { rvec: [0.1, -0.2, 0.05], tvec: [-0.5, -0.4, 4] }, K)
 * solvePnP(X, x, K).tvec.map((v) => Math.round(v * 1e6) / 1e6) // => [-0.5, -0.4, 4]
 * ```
 */
export function solvePnP(
  objectPoints: TensorLike | readonly (readonly number[])[],
  imagePoints: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  distortion?: Distortion,
  options: { readonly initial?: Pose } = {},
): Pose {
  const X = points3Of(objectPoints, 'solvePnP')
  const x = pointsOf(imagePoints, 'solvePnP')
  if (X.length !== x.length) throw new MdmlError('shape', 'solvePnP: one image point per object point')
  if (X.length < 4) throw new MdmlError('value', 'solvePnP: at least four points are needed')
  const init = options.initial ?? linearPose(X, undistortPoints(x, cameraMatrix, distortion ?? []))
  const obj = tensor(X.map((p) => [...p]))
  const obs = tensor(x.map((p) => [...p]))
  const K = tensor([
    cameraMatrix[0]?.[0] ?? 1,
    cameraMatrix[1]?.[1] ?? 1,
    cameraMatrix[0]?.[2] ?? 0,
    cameraMatrix[1]?.[2] ?? 0,
  ])
  const d = tensor(coeffs(distortion))
  const res = (p: Value): Value =>
    reshape(sub(projectValue(obj, sliceAxis(p, 0, 0, 3), sliceAxis(p, 0, 3, 6), K, d), obs), [-1])
  const sol = refine(res, [...init.rvec, ...init.tvec])
  return { rvec: [sol[0] ?? 0, sol[1] ?? 0, sol[2] ?? 0], tvec: [sol[3] ?? 0, sol[4] ?? 0, sol[5] ?? 0] }
}

/**
 * Robust pose estimation (cv2.solvePnPRansac): {@link solvePnP} on random minimal samples (four points for a planar
 * object, six otherwise), scored by reprojection error, then refitted on the inliers of the best.
 *
 * @param objectPoints - Points `[n, 3]`.
 * @param imagePoints - Observed points `[n, 2]`.
 * @param cameraMatrix - The intrinsics.
 * @param distortion - Distortion coefficients.
 * @param options - Threshold (default 8 pixels), trials (default 100), confidence (0.99) and key.
 * @returns The pose and the inlier mask.
 * @concept pose-estimation
 * @example
 * ```ts
 * import { projectPoints, solvePnPRansac } from 'mdml-engine/vision'
 * const K = [[200, 0, 64], [0, 200, 48], [0, 0, 1]]
 * const X = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.2, 0], [0.2, 0.7, 0]]
 * const x = projectPoints(X, { rvec: [0, 0, 0], tvec: [0, 0, 4] }, K)
 * x[5] = [0, 0]
 * solvePnPRansac(X, x, K).mask // => [1, 1, 1, 1, 1, 0]
 * ```
 */
export function solvePnPRansac(
  objectPoints: TensorLike | readonly (readonly number[])[],
  imagePoints: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  distortion?: Distortion,
  options: {
    readonly threshold?: number
    readonly maxIters?: number
    readonly confidence?: number
    readonly key?: Key
  } = {},
): Pose & { readonly mask: number[] } {
  const X = points3Of(objectPoints, 'solvePnPRansac')
  const x = pointsOf(imagePoints, 'solvePnPRansac')
  const planar = X.every((p) => Math.abs(p[2]) < 1e-12)
  const idx = X.map((_, i) => i)
  // Minimal samples take the linear pose alone; the final refit on the inliers is refined.
  const minimal = planar ? 4 : 6
  const fit = (sample: readonly number[]): Pose | undefined => {
    try {
      if (sample.length === minimal)
        return linearPose(
          sample.map((i) => X[i] as Vec3),
          undistortPoints(
            sample.map((i) => x[i] as [number, number]),
            cameraMatrix,
            distortion ?? [],
          ),
        )
      return solvePnP(
        sample.map((i) => X[i] as Vec3),
        sample.map((i) => x[i] as [number, number]),
        cameraMatrix,
        distortion,
      )
    } catch {
      return undefined
    }
  }
  const residual = (pose: Pose | undefined, i: number): number => {
    if (pose === undefined) return Number.POSITIVE_INFINITY
    const [u, v] = (projectPoints([X[i] as Vec3], pose, cameraMatrix, distortion)[0] ?? [0, 0]) as [number, number]
    const [a, b] = x[i] as [number, number]
    return Math.hypot(u - a, v - b)
  }
  const r = ransac(
    idx,
    { fit, residual },
    {
      minSamples: minimal,
      residualThreshold: options.threshold ?? 8,
      maxTrials: options.maxIters ?? 100,
      stopProbability: options.confidence ?? 0.99,
      ...(options.key === undefined ? { seed: 0 } : { key: options.key }),
    },
  )
  const pose = r.model
  if (pose === undefined) throw new MdmlError('value', 'solvePnPRansac: no pose fits the inliers')
  const mask = new Array<number>(X.length).fill(0)
  for (const i of r.inliers) mask[i] = 1
  return { ...pose, mask }
}

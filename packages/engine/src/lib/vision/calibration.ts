/**
 * Camera calibration and stereo rectification, OpenCV style: Zhang's method for planar targets (a homography per view,
 * intrinsics from the image of the absolute conic, extrinsics from each homography) refined jointly with OpenCV's five
 * distortion coefficients by Levenberg-Marquardt (`calibrateCamera`), and Bouguet's rectifying rotations for a
 * calibrated stereo pair (`stereoRectify`).
 *
 * @index function Zhang's camera calibration and stereo rectification.
 */

// TODO: stereoRectify's alpha-dependent new camera matrices (OpenCV fits them to the undistorted image corners);
// here both views share the smaller focal length and the mean principal point.

import { MdmlError } from '../base/index.ts'
import { homography } from '../numerics/geometry/index.ts'
import { concat, reshape, sliceAxis, sub, tensor, toArray, type Value } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { type PointsLike, pointsOf } from './_image.ts'
import { invOf, type Matrix, matMul, nullVector, points3Of, refine, transposeOf } from './_refine.ts'
import { type Distortion, type Pose, projectValue, reprojectionError, rodrigues, rodriguesVector } from './camera.ts'
import { type RefinementPath, type RefinementState, steppedRefinement } from './refinement.ts'

/** The output of {@link calibrateCamera}. */
export interface Calibration {
  /** The root-mean-square reprojection error over every point of every view. */
  readonly rms: number
  readonly cameraMatrix: Matrix
  /** k1, k2, p1, p2, k3. */
  readonly distortion: number[]
  /** One pose per view. */
  readonly poses: Pose[]
}

/** The intrinsics of planar homographies by Zhang's closed form (zero skew imposed). */
function zhangIntrinsics(hs: readonly Matrix[]): Matrix {
  const v = (h: Matrix, i: number, j: number): number[] => {
    const a = (r: number, c: number): number => h[r]?.[c] ?? 0
    return [
      a(0, i) * a(0, j),
      a(0, i) * a(1, j) + a(1, i) * a(0, j),
      a(1, i) * a(1, j),
      a(2, i) * a(0, j) + a(0, i) * a(2, j),
      a(2, i) * a(1, j) + a(1, i) * a(2, j),
      a(2, i) * a(2, j),
    ]
  }
  // Zero skew: B12 = 0, so its column is dropped from the system.
  const drop = (r: number[]): number[] => r.filter((_, k) => k !== 1)
  const rows: number[][] = []
  for (const h of hs) {
    rows.push(drop(v(h, 0, 1)))
    const v11 = v(h, 0, 0)
    const v22 = v(h, 1, 1)
    rows.push(drop(v11.map((x, k) => x - (v22[k] ?? 0))))
  }
  const b5 = nullVector(rows)
  let b = [b5[0] ?? 0, 0, ...b5.slice(1)]
  if ((b[0] ?? 0) < 0) b = b.map((x) => -x)
  const [B11 = 0, B12 = 0, B22 = 0, B13 = 0, B23 = 0, B33 = 0] = b
  const v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
  const lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
  const alpha = Math.sqrt(lambda / B11)
  const beta = Math.sqrt((lambda * B11) / (B11 * B22 - B12 * B12))
  const u0 = (-B13 * alpha * alpha) / lambda
  if (!(Number.isFinite(alpha) && Number.isFinite(beta)))
    throw new MdmlError('value', 'calibrateCamera: the views do not determine the intrinsics')
  return [
    [alpha, 0, u0],
    [0, beta, v0],
    [0, 0, 1],
  ]
}

/** The pose of a planar target from its homography and the intrinsics. */
function zhangPose(h: Matrix, k: Matrix): Pose {
  const ki = invOf(k)
  const col = (j: number): number[] => [0, 1, 2].map((i) => h[i]?.[j] ?? 0)
  const kh = (j: number): number[] =>
    matMul(
      ki,
      col(j).map((x) => [x]),
    ).map((r) => r[0] ?? 0)
  const h1 = kh(0)
  const h2 = kh(1)
  const h3 = kh(2)
  let lambda = 1 / Math.hypot(...h1)
  if ((h3[2] ?? 0) * lambda < 0) lambda = -lambda
  const r1 = h1.map((x) => x * lambda)
  const r2 = h2.map((x) => x * lambda)
  const r3 = [
    (r1[1] ?? 0) * (r2[2] ?? 0) - (r1[2] ?? 0) * (r2[1] ?? 0),
    (r1[2] ?? 0) * (r2[0] ?? 0) - (r1[0] ?? 0) * (r2[2] ?? 0),
    (r1[0] ?? 0) * (r2[1] ?? 0) - (r1[1] ?? 0) * (r2[0] ?? 0),
  ]
  const R = [0, 1, 2].map((i) => [r1[i] ?? 0, r2[i] ?? 0, r3[i] ?? 0])
  // The nearest rotation, through the rotation vector of the (nearly orthogonal) matrix.
  const rvec = rodriguesVector(R)
  const t = h3.map((x) => x * lambda)
  return { rvec, tvec: [t[0] ?? 0, t[1] ?? 0, t[2] ?? 0] }
}

/**
 * Calibrate a camera from views of a planar target (cv2.calibrateCamera): Zhang's closed form gives the intrinsics
 * (zero skew) and each view's pose from the target-to-image homographies; then the focal lengths, principal point,
 * the distortion k1, k2, p1, p2, k3 and every pose are refined together by Levenberg-Marquardt on the reprojection
 * error, as OpenCV does (it starts from the image centre instead; both reach the same optimum from a good start).
 *
 * @param objectPoints - Per view, target points `[n, 3]` with z = 0.
 * @param imagePoints - Per view, their image points `[n, 2]`.
 * @param options - The number of refinement iterations (default 100) and whether to fix k3 at 0 (default false).
 * @returns The RMS error, camera matrix, distortion and poses.
 * @throws MdmlError with code `value` for fewer than two views or a non-planar target.
 * @concept camera-calibration
 * @example
 * ```ts
 * import { calibrateCamera, projectPoints } from 'mdml-engine/vision'
 * const K = [[300, 0, 160], [0, 310, 120], [0, 0, 1]]
 * const board = Array.from({ length: 20 }, (_, i) => [i % 5, Math.floor(i / 5), 0])
 * const poses = [{ rvec: [0.2, 0.1, 0], tvec: [-2, -1.5, 8] }, { rvec: [-0.1, 0.3, 0.1], tvec: [-2, -2, 9] }, { rvec: [0.3, -0.2, 0.2], tvec: [-1.5, -1, 7] }]
 * const c = calibrateCamera(poses.map(() => board), poses.map((p) => projectPoints(board, p, K)), { fixK3: true })
 * c.cameraMatrix.map((r) => r.map((v) => Math.round(v * 1e4) / 1e4)) // => [[300, 0, 160], [0, 310, 120], [0, 0, 1]]
 * ```
 */
export function calibrateCamera(
  objectPoints: readonly (readonly (readonly number[])[])[],
  imagePoints: readonly PointsLike[],
  options: { readonly iterations?: number; readonly fixK3?: boolean } = {},
): Calibration {
  const problem = calibrationProblem(objectPoints, imagePoints, options.fixK3 === true)
  const s = refine(problem.residuals, problem.start, options.iterations ?? 100)
  return problem.calibrationAt(Float64Array.from(s))
}

/** Everything a calibration refinement needs: the start (Zhang's closed form), the residuals, and the reading of a parameter vector. */
interface CalibrationProblem {
  readonly residuals: (p: Value) => Value
  readonly start: number[]
  readonly closedForm: Calibration
  /** The calibration a parameter vector describes, with its RMS reprojection error. */
  calibrationAt(s: ArrayLike<number>): Calibration
}

function calibrationProblem(
  objectPoints: readonly (readonly (readonly number[])[])[],
  imagePoints: readonly PointsLike[],
  fixK3: boolean,
): CalibrationProblem {
  const views = objectPoints.length
  if (views < 2 || imagePoints.length !== views)
    throw new MdmlError('value', 'calibrateCamera: at least two views, one image point set per view')
  const X = objectPoints.map((o) => points3Of(o, 'calibrateCamera'))
  const x = imagePoints.map((p) => pointsOf(p, 'calibrateCamera'))
  for (const [v, o] of X.entries()) {
    if (o.some((p) => Math.abs(p[2]) > 1e-12))
      throw new MdmlError('value', 'calibrateCamera: the target must be planar (z = 0)')
    if (o.length !== x[v]?.length) throw new MdmlError('shape', 'calibrateCamera: one image point per target point')
  }
  const hs = X.map(
    (o, v) =>
      toArray(
        homography(
          o.map(([a, b]) => [a, b] as [number, number]),
          x[v] as [number, number][],
        ),
      ) as Matrix,
  )
  const K0 = zhangIntrinsics(hs)
  const poses0 = hs.map((h) => zhangPose(h, K0))
  const nd = fixK3 ? 4 : 5
  const objs = X.map((o) => tensor(o.map((p) => [...p])))
  const obs = x.map((p) => tensor(p.map((q) => [...q])))
  const residuals = (p: Value): Value => {
    const intr = sliceAxis(p, 0, 0, 4)
    const dist = sliceAxis(p, 0, 4, 4 + nd)
    const parts = objs.map((o, v) => {
      const base = 4 + nd + 6 * v
      const proj = projectValue(o, sliceAxis(p, 0, base, base + 3), sliceAxis(p, 0, base + 3, base + 6), intr, dist)
      return reshape(sub(proj, obs[v] as Value), [-1])
    })
    return concat(parts)
  }
  const start = [
    K0[0]?.[0] ?? 1,
    K0[1]?.[1] ?? 1,
    K0[0]?.[2] ?? 0,
    K0[1]?.[2] ?? 0,
    ...new Array<number>(nd).fill(0),
    ...poses0.flatMap((q) => [...q.rvec, ...q.tvec]),
  ]
  const calibrationAt = (s: ArrayLike<number>): Calibration => {
    const cameraMatrix = [
      [s[0] ?? 1, 0, s[2] ?? 0],
      [0, s[1] ?? 1, s[3] ?? 0],
      [0, 0, 1],
    ]
    const distortion = Array.from({ length: 5 }, (_, i) => (i < nd ? (s[4 + i] ?? 0) : 0))
    const poses: Pose[] = X.map((_, v) => {
      const b = 4 + nd + 6 * v
      return { rvec: [s[b] ?? 0, s[b + 1] ?? 0, s[b + 2] ?? 0], tvec: [s[b + 3] ?? 0, s[b + 4] ?? 0, s[b + 5] ?? 0] }
    })
    let total = 0
    let count = 0
    X.forEach((o, v) => {
      const e = reprojectionError(o, x[v] as [number, number][], poses[v] as Pose, cameraMatrix, distortion)
      total += e * e * o.length
      count += o.length
    })
    return { rms: Math.sqrt(total / Math.max(1, count)), cameraMatrix, distortion, poses }
  }
  return { residuals, start, closedForm: calibrationAt(start), calibrationAt }
}

/** The state of {@link calibrationAlgorithm}: the refinement's state, whose decoded part is the calibration at the iterate. */
export type CalibrationState = RefinementState<Calibration>

/**
 * Zhang's calibration with its Levenberg-Marquardt refinement one iteration per step: the first state is the closed-form
 * calibration (intrinsics from the homographies, zero distortion), and every step is one iteration of the joint
 * refinement of the intrinsics, the distortion and every pose, with the reprojection error and the calibration at the
 * iterate in the state. Run to convergence it reaches {@link calibrateCamera}'s optimum.
 *
 * @param objectPoints - Per view, target points `[n, 3]` with z = 0.
 * @param imagePoints - Per view, their image points `[n, 2]`.
 * @param options - The most iterations (default 100) and whether to fix k3 at 0 (default false).
 * @returns The algorithm; read `state.decoded` for the camera matrix, distortion and poses, `state.rms` for the error.
 * @throws MdmlError with code `value` for fewer than two views or a non-planar target.
 * @concept camera-calibration
 * @example
 * ```ts
 * import { calibrationAlgorithm, projectPoints } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const K = [[300, 0, 160], [0, 310, 120], [0, 0, 1]]
 * const board = Array.from({ length: 20 }, (_, i) => [i % 5, Math.floor(i / 5), 0])
 * const poses = [{ rvec: [0.2, 0.1, 0], tvec: [-2, -1.5, 8] }, { rvec: [-0.1, 0.3, 0.1], tvec: [-2, -2, 9] }, { rvec: [0.3, -0.2, 0.2], tvec: [-1.5, -1, 7] }]
 * const alg = calibrationAlgorithm(poses.map(() => board), poses.map((p) => projectPoints(board, p, K)), { fixK3: true })
 * runAlgorithm(alg, { maxSteps: 100 }).state.rms < 1e-6 // => true
 * ```
 */
export function calibrationAlgorithm(
  objectPoints: readonly (readonly (readonly number[])[])[],
  imagePoints: readonly PointsLike[],
  options: { readonly iterations?: number; readonly fixK3?: boolean } = {},
): Algorithm<CalibrationState, RefinementPath> {
  const problem = calibrationProblem(objectPoints, imagePoints, options.fixK3 === true)
  return steppedRefinement(
    'cameraCalibration',
    { views: objectPoints.length, fixK3: options.fixK3 === true },
    problem.residuals,
    problem.start,
    (x) => problem.calibrationAt(x),
    options.iterations ?? 100,
  )
}

/** The output of {@link stereoRectify}. */
export interface StereoRectification {
  /** The rectifying rotations of the first and second camera. */
  readonly R1: Matrix
  readonly R2: Matrix
  /** The 3 x 4 projection matrices of the rectified cameras. */
  readonly P1: Matrix
  readonly P2: Matrix
  /** The 4 x 4 disparity-to-depth matrix. */
  readonly Q: Matrix
}

/**
 * Rectify a calibrated stereo pair (Bouguet's method, cv2.stereoRectify): each camera is turned by half the relative
 * rotation, then both by the rotation that makes the baseline parallel to the image rows (or columns, for a vertical
 * pair), so corresponding points share a row. The rotations match OpenCV's; the projection matrices use the smaller
 * focal length (as OpenCV, with its barrel-distortion correction) and the mean principal point (see the TODO).
 *
 * @param k1 - The first camera's intrinsics.
 * @param d1 - Its distortion.
 * @param k2 - The second camera's intrinsics.
 * @param d2 - Its distortion.
 * @param size - The image size [width, height].
 * @param R - The rotation from the first camera's frame to the second's.
 * @param T - The translation, x2 = R x1 + T.
 * @returns R1, R2, P1, P2 and Q.
 * @concept stereo-rectification
 * @example
 * ```ts
 * import { stereoRectify } from 'mdml-engine/vision'
 * const K = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
 * const r = stereoRectify(K, [], K, [], [100, 80], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [-1, 0, 0])
 * r.R1.map((row) => row.map((v) => Math.round(v * 1e12) / 1e12)) // => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * ```
 */
export function stereoRectify(
  k1: readonly (readonly number[])[],
  d1: Distortion,
  k2: readonly (readonly number[])[],
  d2: Distortion,
  size: readonly [number, number],
  R: readonly (readonly number[])[],
  T: readonly [number, number, number],
): StereoRectification {
  const om = rodriguesVector(R).map((v) => -0.5 * v)
  const rr = rodrigues(om)
  const t = matMul(
    rr,
    T.map((v) => [v]),
  ).map((r) => r[0] ?? 0)
  const idx = Math.abs(t[0] ?? 0) > Math.abs(t[1] ?? 0) ? 0 : 1
  const c = t[idx] ?? 0
  const nt = Math.hypot(...t)
  const uu = [0, 0, 0]
  uu[idx] = c > 0 ? 1 : -1
  let ww = [
    (t[1] ?? 0) * (uu[2] ?? 0) - (t[2] ?? 0) * (uu[1] ?? 0),
    (t[2] ?? 0) * (uu[0] ?? 0) - (t[0] ?? 0) * (uu[2] ?? 0),
    (t[0] ?? 0) * (uu[1] ?? 0) - (t[1] ?? 0) * (uu[0] ?? 0),
  ]
  const nw = Math.hypot(...ww)
  if (nw > 0) ww = ww.map((v) => (v * Math.acos(Math.abs(c) / nt)) / nw)
  const wR = rodrigues(ww)
  const R1 = matMul(wR, transposeOf(rr))
  const R2 = matMul(wR, rr)
  const tn = matMul(
    R2,
    T.map((v) => [v]),
  ).map((r) => r[0] ?? 0)
  const [w, h] = size
  let fc = Number.MAX_VALUE
  for (const [K, D] of [
    [k1, d1],
    [k2, d2],
  ] as const) {
    let f = K[idx ^ 1]?.[idx ^ 1] ?? 1
    const dk1 = D[0] ?? 0
    if (dk1 < 0) f *= 1 + (dk1 * (w * w + h * h)) / (4 * f * f)
    fc = Math.min(fc, f)
  }
  const cx = ((k1[0]?.[2] ?? 0) + (k2[0]?.[2] ?? 0)) / 2
  const cy = ((k1[1]?.[2] ?? 0) + (k2[1]?.[2] ?? 0)) / 2
  const P1 = [
    [fc, 0, cx, 0],
    [0, fc, cy, 0],
    [0, 0, 1, 0],
  ]
  const P2 = P1.map((r) => [...r])
  ;(P2[idx] as number[])[3] = (tn[idx] ?? 0) * fc
  const tx = tn[idx] ?? 1
  const Q = [
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0, fc],
    [0, 0, -1 / tx, 0],
  ]
  return { R1, R2, P1, P2, Q }
}

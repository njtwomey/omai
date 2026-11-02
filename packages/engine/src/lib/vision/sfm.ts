/**
 * Structure from motion in steps. The two-view start: the essential matrix from correspondences by RANSAC as an
 * algorithm (one hypothesis per step, the inliers of the best so far in the state), then the pose recovered by the
 * cheirality check and the inliers triangulated (`twoViewReconstruction`). Then bundle adjustment: the cameras' poses and
 * the 3-D points refined together by Levenberg-Marquardt on the reprojection error, one iteration per step, with the
 * first camera held fixed. New views join with `solvePnP` against the points already built.
 *
 * @index function Structure from motion in steps: RANSAC two-view start, bundle adjustment.
 */
import type { Json } from '../base/index.ts'
import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { type RansacState, ransacAlgorithm } from '../stats/index.ts'
import { concat, reshape, sliceAxis, sub, type TensorLike, take, tensor, type Value } from '../tensor/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { type PointsLike, pointsOf } from './_image.ts'
import type { Matrix } from './_refine.ts'
import { type Pose, projectPoints, projectValue, solvePnP } from './camera.ts'
import { findEssentialMat, recoverPose, triangulatePoints } from './epipolar.ts'
import { type RefinementPath, type RefinementState, steppedRefinement } from './refinement.ts'

type P = readonly [number, number]

/** Pixels to normalised camera coordinates. */
function normalise(points: readonly P[], k: readonly (readonly number[])[]): P[] {
  const fx = k[0]?.[0] ?? 1
  const fy = k[1]?.[1] ?? 1
  const cx = k[0]?.[2] ?? 0
  const cy = k[1]?.[2] ?? 0
  return points.map(([u, v]) => [(u - cx) / fx, (v - cy) / fy] as const)
}

/** The Sampson distance of a normalised correspondence to an essential matrix (OpenCV's error for it). */
function sampson(e: Matrix, a: P, b: P): number {
  const E = (r: number, c: number): number => e[r]?.[c] ?? 0
  const l = [0, 1, 2].map((r) => E(r, 0) * a[0] + E(r, 1) * a[1] + E(r, 2))
  const lt = [0, 1, 2].map((c) => E(0, c) * b[0] + E(1, c) * b[1] + E(2, c))
  const num = b[0] * (l[0] ?? 0) + b[1] * (l[1] ?? 0) + (l[2] ?? 0)
  return Math.sqrt((num * num) / ((l[0] ?? 0) ** 2 + (l[1] ?? 0) ** 2 + (lt[0] ?? 0) ** 2 + (lt[1] ?? 0) ** 2))
}

/** Options of {@link essentialRansacAlgorithm}. */
export interface EssentialRansacOptions {
  /** The inlier threshold in pixels on the Sampson distance (default 1). */
  readonly threshold?: number
  /** The most hypotheses (default 200). */
  readonly maxTrials?: number
  /** The probability of an outlier-free sample at which to stop early (default 0.999). */
  readonly confidence?: number
  readonly key?: Key
}

/**
 * RANSAC for the essential matrix as an algorithm: each step draws eight correspondences, fits the normalised
 * eight-point essential matrix to them, and keeps the hypothesis with the most inliers under the Sampson distance. The
 * state has the trials so far, the best model, and its inliers (indices of correspondences), which a session draws
 * growing as better hypotheses are found.
 *
 * @param points1 - Points in the first image `[n, 2]` (pixels).
 * @param points2 - The matching points in the second image.
 * @param cameraMatrix - The intrinsics shared by both views.
 * @param options - Threshold, trials, stopping probability and key.
 * @returns The algorithm (state: `bestModel` is the essential matrix, `bestInliers` the indices).
 * @concept essential-matrix
 * @example
 * ```ts
 * import { essentialRansacAlgorithm } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const K = [[100, 0, 0], [0, 100, 0], [0, 0, 1]]
 * const X = Array.from({ length: 12 }, (_, i) => [(i % 4) - 1.5, Math.floor(i / 4) - 1, 4 + (i % 3)])
 * const p1 = X.map(([x, y, z]) => [(100 * (x as number)) / (z as number), (100 * (y as number)) / (z as number)])
 * const p2 = X.map(([x, y, z]) => [(100 * ((x as number) - 1)) / (z as number), (100 * (y as number)) / (z as number)])
 * runAlgorithm(essentialRansacAlgorithm(p1, p2, K), { maxSteps: 30 }).state.bestInliers.length // => 12
 * ```
 */
export function essentialRansacAlgorithm(
  points1: PointsLike,
  points2: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  options: EssentialRansacOptions = {},
): Algorithm<RansacState<Matrix>> {
  const p1 = pointsOf(points1, 'essentialRansacAlgorithm')
  const p2 = pointsOf(points2, 'essentialRansacAlgorithm')
  if (p1.length !== p2.length) throw new MdmlError('shape', 'essentialRansacAlgorithm: the point sets differ in size')
  const a = normalise(p1, cameraMatrix)
  const b = normalise(p2, cameraMatrix)
  const focal = ((cameraMatrix[0]?.[0] ?? 1) + (cameraMatrix[1]?.[1] ?? 1)) / 2
  const identity: Matrix = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]
  return ransacAlgorithm<number, Matrix>(
    p1.map((_, i) => i),
    {
      // A model is fitted on normalised points, so the intrinsics of the fit are the identity.
      fit: (s) =>
        findEssentialMat(
          s.map((i) => a[i] as P),
          s.map((i) => b[i] as P),
          identity,
          { method: 'all' },
        ).model,
      residual: (e, i) => sampson(e, a[i] as P, b[i] as P),
    },
    {
      minSamples: 8,
      residualThreshold: (options.threshold ?? 1) / focal,
      maxTrials: options.maxTrials ?? 200,
      stopProbability: options.confidence ?? 0.999,
      ...(options.key === undefined ? { seed: 0 } : { key: options.key }),
    },
  )
}

/** Two views reconstructed: the pose of the second camera, the triangulated inliers. */
export interface TwoViewReconstruction {
  /** The essential matrix refit to the inliers. */
  readonly essential: Matrix
  /** The rotation of the second camera (the first is the identity). */
  readonly R: Matrix
  /** The unit translation of the second camera: the scale of a two-view reconstruction is free. */
  readonly t: [number, number, number]
  /** The indices of the correspondences that were triangulated (RANSAC inliers in front of both cameras). */
  readonly indices: readonly number[]
  /** Their 3-D points in the first camera's frame. */
  readonly points: readonly [number, number, number][]
}

/**
 * The two-view start of structure from motion from a RANSAC state: the essential matrix refit to the inliers, the
 * relative pose recovered by the cheirality check, and the inliers in front of both cameras triangulated. The first
 * camera sits at the origin; the translation is a unit vector (scale is not determined by two views).
 *
 * @param ransac - A state of {@link essentialRansacAlgorithm}.
 * @param points1 - Points in the first image.
 * @param points2 - The matching points in the second image.
 * @param cameraMatrix - The intrinsics.
 * @returns The pose, the points and which correspondences they are.
 * @throws MdmlError with code `value` if RANSAC found fewer than eight inliers.
 * @concept structure-from-motion
 * @example
 * ```ts
 * import { essentialRansacAlgorithm, twoViewReconstruction } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const K = [[100, 0, 0], [0, 100, 0], [0, 0, 1]]
 * const X = Array.from({ length: 12 }, (_, i) => [(i % 4) - 1.5, Math.floor(i / 4) - 1, 4 + (i % 3)])
 * const p1 = X.map(([x, y, z]) => [(100 * (x as number)) / (z as number), (100 * (y as number)) / (z as number)])
 * const p2 = X.map(([x, y, z]) => [(100 * ((x as number) - 1)) / (z as number), (100 * (y as number)) / (z as number)])
 * const r = runAlgorithm(essentialRansacAlgorithm(p1, p2, K), { maxSteps: 30 }).state
 * twoViewReconstruction(r, p1, p2, K).points.length // => 12
 * ```
 */
export function twoViewReconstruction(
  ransac: RansacState<Matrix>,
  points1: PointsLike,
  points2: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
): TwoViewReconstruction {
  const p1 = pointsOf(points1, 'twoViewReconstruction')
  const p2 = pointsOf(points2, 'twoViewReconstruction')
  const inl = ransac.bestInliers
  if (inl.length < 8) throw new MdmlError('value', 'twoViewReconstruction: fewer than eight inliers')
  const s1 = inl.map((i) => p1[i] as P)
  const s2 = inl.map((i) => p2[i] as P)
  const essential = findEssentialMat(s1, s2, cameraMatrix, { method: 'all' }).model
  const pose = recoverPose(essential, s1, s2, cameraMatrix)
  const keep = inl.filter((_, k) => pose.mask[k] === 1)
  const n1 = normalise(
    keep.map((i) => p1[i] as P),
    cameraMatrix,
  )
  const n2 = normalise(
    keep.map((i) => p2[i] as P),
    cameraMatrix,
  )
  const P0 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ]
  const P1 = pose.R.map((row, i) => [...row, pose.t[i] ?? 0])
  return { essential, R: pose.R, t: pose.t, indices: keep, points: triangulatePoints(P0, P1, n1, n2) }
}

/** A scene for bundle adjustment: the poses of the cameras, the 3-D points, and what each camera saw. */
export interface BundleScene {
  /** One pose per camera (object to camera coordinates); camera 0 is held fixed. */
  readonly poses: readonly Pose[]
  /** The points `[n][3]`. */
  readonly points: readonly (readonly number[])[]
  /** The observations of each camera: one `[n, 2]` array, row j the pixel of point j (ignored where not visible). */
  readonly observations: readonly PointsLike[]
  /** Which points each camera saw (default all). */
  readonly visible?: readonly (readonly boolean[])[]
}

/** The geometry a bundle adjustment state describes. */
export interface BundleGeometry {
  readonly poses: readonly Pose[]
  readonly points: readonly [number, number, number][]
}

/** The state of {@link bundleAdjustmentAlgorithm}. */
export type BundleState = RefinementState<BundleGeometry>

/**
 * Bundle adjustment as an algorithm: the poses of every camera but the first and all the 3-D points refined together
 * by Levenberg-Marquardt on the reprojection error of every observation, one iteration per step. The first camera is
 * fixed, which leaves the overall scale free (the reconstruction is metric only up to scale); compare structure to the
 * truth after fitting a scale. Intrinsics are known and fixed.
 *
 * @param scene - The cameras, points and observations.
 * @param cameraMatrix - The intrinsics shared by the cameras.
 * @param options - The most iterations (default 50).
 * @returns The algorithm; read `state.decoded` for the poses and points, `state.rms` for the error in pixels.
 * @throws MdmlError with code `value` if the observations do not match the cameras and points.
 * @concept bundle-adjustment
 * @example
 * ```ts
 * import { bundleAdjustmentAlgorithm, projectPoints } from 'mdml-engine/vision'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const K = [[200, 0, 0], [0, 200, 0], [0, 0, 1]]
 * const X = Array.from({ length: 12 }, (_, i) => [(i % 4) - 1.5, Math.floor(i / 4) - 1, 5 + (i % 3)])
 * const poses = [{ rvec: [0, 0, 0], tvec: [0, 0, 0] }, { rvec: [0, 0.1, 0], tvec: [-1, 0, 0] }, { rvec: [0.05, -0.1, 0], tvec: [0.5, 0.2, 0.1] }] as const
 * const obs = poses.map((p) => projectPoints(X, { rvec: [...p.rvec], tvec: [...p.tvec] } as never, K))
 * const start = X.map(([x, y, z]) => [(x as number) + 0.1, (y as number) - 0.1, (z as number) + 0.2])
 * const alg = bundleAdjustmentAlgorithm({ poses: poses as never, points: start, observations: obs }, K)
 * runAlgorithm(alg, { maxSteps: 40 }).state.rms < 1e-3 // => true
 * ```
 */
export function bundleAdjustmentAlgorithm(
  scene: BundleScene,
  cameraMatrix: readonly (readonly number[])[],
  options: { readonly iterations?: number } = {},
): Algorithm<BundleState, RefinementPath> {
  const V = scene.poses.length
  const N = scene.points.length
  if (V < 2 || scene.observations.length !== V)
    throw new MdmlError('value', 'bundleAdjustmentAlgorithm: at least two cameras, one observation set each')
  const obs = scene.observations.map((o) => pointsOf(o, 'bundleAdjustmentAlgorithm'))
  const seen = scene.poses.map((_, c) => Array.from({ length: N }, (_, j) => scene.visible?.[c]?.[j] ?? true))
  for (const [c, o] of obs.entries())
    if (o.length !== N)
      throw new MdmlError('value', `bundleAdjustmentAlgorithm: camera ${c} needs one observation row per point`)
  const index = seen.map((s) => s.flatMap((v, j) => (v ? [j] : [])))
  const observed = obs.map((o, c) => tensor((index[c] as number[]).map((j) => [...(o[j] as P)])))
  const intr = tensor([
    cameraMatrix[0]?.[0] ?? 1,
    cameraMatrix[1]?.[1] ?? 1,
    cameraMatrix[0]?.[2] ?? 0,
    cameraMatrix[1]?.[2] ?? 0,
  ])
  const noDistortion = tensor([0, 0, 0, 0])
  const poseStart = scene.poses.slice(1).flatMap((p) => [...p.rvec, ...p.tvec])
  const start = [...poseStart, ...scene.points.flatMap((p) => [p[0] ?? 0, p[1] ?? 0, p[2] ?? 0])]
  const nPose = 6 * (V - 1)
  const poseOf = (x: Value | ArrayLike<number>, c: number): { r: Value; t: Value } | Pose => {
    if (c === 0) return { r: tensor([0, 0, 0]), t: tensor([0, 0, 0]) } as { r: Value; t: Value }
    const b = 6 * (c - 1)
    return { r: sliceAxis(x as Value, 0, b, b + 3), t: sliceAxis(x as Value, 0, b + 3, b + 6) } as {
      r: Value
      t: Value
    }
  }
  const residuals = (x: Value): Value => {
    const pts = reshape(sliceAxis(x, 0, nPose, nPose + 3 * N), [N, 3])
    const parts = scene.poses.map((_, c) => {
      const { r, t } = poseOf(x, c) as { r: Value; t: Value }
      const visiblePoints = index[c]?.length === N ? pts : take(pts, index[c] as number[], 0)
      return reshape(sub(projectValue(visiblePoints, r, t, intr, noDistortion), observed[c] as Value), [-1])
    })
    return concat(parts)
  }
  const decode = (x: Float64Array): BundleGeometry => ({
    poses: scene.poses.map((_, c) =>
      c === 0
        ? (scene.poses[0] as Pose)
        : {
            rvec: [x[6 * (c - 1)] ?? 0, x[6 * (c - 1) + 1] ?? 0, x[6 * (c - 1) + 2] ?? 0],
            tvec: [x[6 * (c - 1) + 3] ?? 0, x[6 * (c - 1) + 4] ?? 0, x[6 * (c - 1) + 5] ?? 0],
          },
    ),
    points: Array.from(
      { length: N },
      (_, j) =>
        [x[nPose + 3 * j] ?? 0, x[nPose + 3 * j + 1] ?? 0, x[nPose + 3 * j + 2] ?? 0] as [number, number, number],
    ),
  })
  return steppedRefinement(
    'bundleAdjustment',
    { cameras: V, points: N, observations: index.reduce((a, s) => a + s.length, 0) } as Json,
    residuals,
    start,
    decode,
    options.iterations ?? 50,
  )
}

/**
 * The pose of a camera joining a reconstruction, from the 3-D points already built and where it sees them
 * (`solvePnP` wrapped for the incremental step).
 *
 * @param points - The known 3-D points, at least six.
 * @param pixels - Where the new camera sees them.
 * @param cameraMatrix - The intrinsics.
 * @returns The pose.
 * @concept structure-from-motion
 * @example
 * ```ts
 * import { addView, projectPoints } from 'mdml-engine/vision'
 * const K = [[200, 0, 0], [0, 200, 0], [0, 0, 1]]
 * const X = Array.from({ length: 10 }, (_, i) => [(i % 4) - 1.5, Math.floor(i / 4) - 1, 5 + (i % 3)])
 * const truth = { rvec: [0, 0.1, 0], tvec: [-0.5, 0.1, 0] } as never
 * addView(X, projectPoints(X, truth, K), K).tvec.map((v) => Math.round(v * 1e4) / 1e4) // => [-0.5, 0.1, 0]
 * ```
 */
export function addView(
  points: TensorLike | readonly (readonly number[])[],
  pixels: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
): Pose {
  return solvePnP(points, pixels, cameraMatrix)
}

/**
 * The reprojection error of a bundle: the root-mean-square distance in pixels between every visible observation and the
 * projection of its point (the quantity {@link bundleAdjustmentAlgorithm} lowers).
 *
 * @param scene - The cameras, points and observations.
 * @param cameraMatrix - The intrinsics.
 * @returns The RMS error in pixels.
 * @concept bundle-adjustment
 * @example
 * ```ts
 * import { bundleReprojectionError, projectPoints } from 'mdml-engine/vision'
 * const K = [[200, 0, 0], [0, 200, 0], [0, 0, 1]]
 * const X = [[0, 0, 5], [1, 0, 6], [0, 1, 5]]
 * const poses = [{ rvec: [0, 0, 0], tvec: [0, 0, 0] }, { rvec: [0, 0, 0], tvec: [-1, 0, 0] }] as never
 * bundleReprojectionError({ poses, points: X, observations: (poses as Parameters<typeof projectPoints>[1][]).map((p) => projectPoints(X, p, K)) }, K) // => 0
 * ```
 */
export function bundleReprojectionError(scene: BundleScene, cameraMatrix: readonly (readonly number[])[]): number {
  let total = 0
  let count = 0
  scene.poses.forEach((pose, c) => {
    const projected = projectPoints(scene.points as number[][], pose, cameraMatrix)
    const seen = pointsOf(scene.observations[c] as PointsLike, 'bundleReprojectionError')
    projected.forEach(([u, v], j) => {
      if (scene.visible?.[c]?.[j] === false) return
      const [a, b] = seen[j] as P
      total += (u - a) ** 2 + (v - b) ** 2
      count++
    })
  })
  return Math.sqrt(total / Math.max(1, count))
}

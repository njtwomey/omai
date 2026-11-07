/**
 * Calibrated two-view geometry, OpenCV style: the essential matrix of normalised correspondences (the eight-point
 * algorithm projected onto the essential manifold, with RANSAC; OpenCV uses Nister's five-point solver), its
 * decomposition into two rotations and a translation direction, the relative pose recovered by the cheirality check,
 * and linear (DLT) triangulation of points seen by two projection matrices.
 */

import { MdmlError } from '../base/index.ts'
import type { Point2 } from '../numerics/geometry/index.ts'
import { fundamentalMatrix } from '../numerics/geometry/index.ts'
import type { Key } from '../random/index.ts'
import { ransac } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { type PointsLike, pointsOf } from './_image.ts'
import { det3, invOf, type MutableRows, matMul, nullVector, svdOf, transposeOf } from './_refine.ts'
import type { RobustFit } from './geometry.ts'

/** Pixel points to normalised camera coordinates K^-1 [u, v, 1]. */
function normalised(points: readonly Point2[], k: readonly (readonly number[])[]): Point2[] {
  const fx = k[0]?.[0] ?? 1
  const fy = k[1]?.[1] ?? 1
  const s = k[0]?.[1] ?? 0
  const cx = k[0]?.[2] ?? 0
  const cy = k[1]?.[2] ?? 0
  return points.map(([u, v]) => {
    const y = (v - cy) / fy
    return [(u - cx - s * y) / fx, y]
  })
}

/** The nearest essential matrix: singular values made (1, 1, 0). */
function toEssential(e: MutableRows): MutableRows {
  const { u, v } = svdOf(e)
  const d = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ]
  return matMul(matMul(u, d), transposeOf(v))
}

/** The essential matrix of normalised correspondences: the eight-point fundamental matrix, made essential. */
function fitEssential(a: readonly Point2[], b: readonly Point2[]): MutableRows {
  const f = toArray(fundamentalMatrix(a as Point2[], b as Point2[])) as MutableRows
  const e = toEssential(f)
  // Unit Frobenius norm (sqrt 2 for singular values 1, 1), the sign with the largest entry positive.
  let big = 0
  for (const r of e) for (const v of r) if (Math.abs(v) > Math.abs(big)) big = v
  const s = Math.sign(big) / Math.SQRT2
  return e.map((r) => r.map((v) => v * s))
}

/**
 * The essential matrix from pixel correspondences and the camera matrix (cv2.findEssentialMat): the points are
 * normalised by K^-1, E is found by the normalised eight-point algorithm with its singular values forced to (1, 1, 0)
 * (inside RANSAC with the threshold scaled by the mean focal length, as OpenCV does), and returned with unit Frobenius
 * norm. OpenCV uses Nister's five-point solver; on noise-free data both give the same E up to sign.
 *
 * @param points1 - Points in the first image `[n, 2]`, n >= 8.
 * @param points2 - Matching points in the second image.
 * @param cameraMatrix - The intrinsics (shared by both views).
 * @param options - Method ('ransac' or 'all'), threshold (default 1 pixel), trials, confidence and key.
 * @returns E with x2^T E x1 = 0 in normalised coordinates, and the inlier mask.
 * @concept essential-matrix
 * @example
 * ```ts
 * import { findEssentialMat } from 'mdml-engine/vision'
 * const X = [[0, 0, 4], [1, 0, 5], [0, 1, 6], [1, 1, 4], [-1, 0.5, 5], [0.5, -1, 7], [2, 1, 6], [-1, -1, 5], [0.3, 0.8, 3]]
 * const p1 = X.map(([x, y, z]) => [x / z, y / z])
 * const p2 = X.map(([x, y, z]) => [(x - 1) / z, y / z])
 * const r = findEssentialMat(p1, p2, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], { method: 'all' })
 * r.model.map((row) => row.map((v) => Math.round(Math.abs(v) * 1e9) / 1e9)) // => [[0, 0, 0], [0, 0, 0.707106781], [0, 0.707106781, 0]]
 * ```
 */
export function findEssentialMat(
  points1: PointsLike,
  points2: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  options: {
    readonly method?: 'ransac' | 'all'
    readonly threshold?: number
    readonly maxIters?: number
    readonly confidence?: number
    readonly key?: Key
  } = {},
): RobustFit {
  const p1 = pointsOf(points1, 'findEssentialMat')
  const p2 = pointsOf(points2, 'findEssentialMat')
  if (p1.length !== p2.length) throw new MdmlError('shape', 'findEssentialMat: the point sets differ in size')
  if (p1.length < 8) throw new MdmlError('value', 'findEssentialMat: at least eight correspondences are needed')
  const a = normalised(p1, cameraMatrix)
  const b = normalised(p2, cameraMatrix)
  const n = a.length
  if ((options.method ?? 'ransac') === 'all') return { model: fitEssential(a, b), mask: new Array<number>(n).fill(1) }
  const focal = ((cameraMatrix[0]?.[0] ?? 1) + (cameraMatrix[1]?.[1] ?? 1)) / 2
  const error = (e: MutableRows, i: number): number => {
    const [x1, y1] = a[i] as Point2
    const [x2, y2] = b[i] as Point2
    const E = (r: number, c: number): number => e[r]?.[c] ?? 0
    const l = [0, 1, 2].map((r) => E(r, 0) * x1 + E(r, 1) * y1 + E(r, 2))
    const lt = [0, 1, 2].map((c) => E(0, c) * x2 + E(1, c) * y2 + E(2, c))
    const num = x2 * (l[0] ?? 0) + y2 * (l[1] ?? 0) + (l[2] ?? 0)
    // The Sampson distance, OpenCV's error for the essential matrix.
    return Math.sqrt((num * num) / ((l[0] ?? 0) ** 2 + (l[1] ?? 0) ** 2 + (lt[0] ?? 0) ** 2 + (lt[1] ?? 0) ** 2))
  }
  const r = ransac(
    a.map((_, i) => i),
    {
      fit: (s: readonly number[]): MutableRows | undefined => {
        try {
          return fitEssential(
            s.map((i) => a[i] as Point2),
            s.map((i) => b[i] as Point2),
          )
        } catch {
          return undefined
        }
      },
      residual: (e: MutableRows | undefined, i: number) => (e === undefined ? Number.POSITIVE_INFINITY : error(e, i)),
    },
    {
      minSamples: 8,
      residualThreshold: (options.threshold ?? 1) / focal,
      maxTrials: options.maxIters ?? 1000,
      stopProbability: options.confidence ?? 0.999,
      ...(options.key === undefined ? { seed: 0 } : { key: options.key }),
    },
  )
  if (r.model === undefined) throw new MdmlError('value', 'findEssentialMat: the inliers are degenerate')
  const mask = new Array<number>(n).fill(0)
  for (const i of r.inliers) mask[i] = 1
  return { model: r.model, mask }
}

/**
 * Decompose an essential matrix (cv2.decomposeEssentialMat): with E = U diag(1, 1, 0) V^T (U and V made proper), the
 * candidate rotations U W V^T and U W^T V^T and the translation direction +-u3, W the quarter turn about z.
 *
 * @param e - The essential matrix.
 * @returns The two rotations and the unit translation (up to sign).
 * @concept essential-matrix
 * @example
 * ```ts
 * import { decomposeEssentialMat } from 'mdml-engine/vision'
 * const { t } = decomposeEssentialMat([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
 * t.map((v) => Math.abs(v)) // => [1, 0, 0]
 * ```
 */
export function decomposeEssentialMat(e: readonly (readonly number[])[]): {
  readonly R1: MutableRows
  readonly R2: MutableRows
  readonly t: [number, number, number]
} {
  let { u, v } = svdOf(e.map((r) => [...r]))
  if (det3(u) < 0) u = u.map((r) => r.map((x) => -x))
  if (det3(v) < 0) v = v.map((r) => r.map((x) => -x))
  const W = [
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
  ]
  const R1 = matMul(matMul(u, W), transposeOf(v))
  const R2 = matMul(matMul(u, transposeOf(W)), transposeOf(v))
  return { R1, R2, t: [u[0]?.[2] ?? 0, u[1]?.[2] ?? 0, u[2]?.[2] ?? 0] }
}

/**
 * Triangulate points seen by two cameras (cv2.triangulatePoints, the linear DLT): for each pair, the null vector of
 * the four equations x P_3 - P_1, y P_3 - P_2 of both views, dehomogenised.
 *
 * @param proj1 - The first 3 x 4 projection matrix.
 * @param proj2 - The second.
 * @param points1 - Points in the first image `[n, 2]`.
 * @param points2 - Matching points in the second image.
 * @returns The 3-D points `[n, 3]`.
 * @concept triangulation
 * @example
 * ```ts
 * import { triangulatePoints } from 'mdml-engine/vision'
 * const P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
 * const P2 = [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]]
 * triangulatePoints(P1, P2, [[0.2, 0.1]], [[0, 0.1]]).map((p) => p.map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, 0.5, 5]]
 * ```
 */
export function triangulatePoints(
  proj1: readonly (readonly number[])[],
  proj2: readonly (readonly number[])[],
  points1: PointsLike,
  points2: PointsLike,
): [number, number, number][] {
  const a = pointsOf(points1, 'triangulatePoints')
  const b = pointsOf(points2, 'triangulatePoints')
  if (a.length !== b.length) throw new MdmlError('shape', 'triangulatePoints: the point sets differ in size')
  const row = (Point2: readonly (readonly number[])[], r: number): number[] =>
    [0, 1, 2, 3].map((c) => Point2[r]?.[c] ?? 0)
  return a.map(([x1, y1], i) => {
    const [x2, y2] = b[i] as Point2
    const eq = (Point2: readonly (readonly number[])[], x: number, r: number): number[] =>
      row(Point2, 2).map((v, c) => x * v - (row(Point2, r)[c] ?? 0))
    const X = nullVector([eq(proj1, x1, 0), eq(proj1, y1, 1), eq(proj2, x2, 0), eq(proj2, y2, 1)])
    const w = X[3] ?? 1
    return [(X[0] ?? 0) / w, (X[1] ?? 0) / w, (X[2] ?? 0) / w]
  })
}

/**
 * Recover the relative pose from an essential matrix (cv2.recoverPose): triangulate the correspondences under each of
 * the four (R, +-t) candidates and keep the one with the most points in front of both cameras (and nearer than a
 * distance limit), as OpenCV's cheirality check does.
 *
 * @param e - The essential matrix.
 * @param points1 - Points in the first image `[n, 2]` (pixels).
 * @param points2 - Matching points in the second image.
 * @param cameraMatrix - The intrinsics.
 * @param options - The distance limit (default 50, in units of the baseline) and an inlier mask to restrict to.
 * @returns R and the unit t of the second camera, the count of points passing the check and their mask.
 * @concept relative-pose
 * @example
 * ```ts
 * import { recoverPose } from 'mdml-engine/vision'
 * const K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * const X = [[0, 0, 4], [1, 0, 5], [0, 1, 6], [1, 1, 4], [-1, 0.5, 5]]
 * const p1 = X.map(([x, y, z]) => [x / z, y / z])
 * const p2 = X.map(([x, y, z]) => [(x - 1) / z, y / z])
 * const r = recoverPose([[0, 0, 0], [0, 0, -1], [0, 1, 0]], p1, p2, K)
 * [r.t.map((v) => Math.round(v * 1e9) / 1e9), r.good] // => [[-1, 0, 0], 5]
 * ```
 */
export function recoverPose(
  e: readonly (readonly number[])[],
  points1: PointsLike,
  points2: PointsLike,
  cameraMatrix: readonly (readonly number[])[],
  options: { readonly distanceThreshold?: number; readonly mask?: readonly number[] } = {},
): { readonly R: MutableRows; readonly t: [number, number, number]; readonly good: number; readonly mask: number[] } {
  const a = normalised(pointsOf(points1, 'recoverPose'), cameraMatrix)
  const b = normalised(pointsOf(points2, 'recoverPose'), cameraMatrix)
  const dist = options.distanceThreshold ?? 50
  const { R1, R2, t } = decomposeEssentialMat(e)
  const P0 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ]
  const candidates: { R: MutableRows; t: [number, number, number] }[] = [
    { R: R1, t },
    { R: R2, t },
    { R: R1, t: [-t[0], -t[1], -t[2]] },
    { R: R2, t: [-t[0], -t[1], -t[2]] },
  ]
  let best = { R: R1, t, good: -1, mask: [] as number[] }
  for (const c of candidates) {
    const P1 = c.R.map((row, i) => [...row, c.t[i] ?? 0])
    const X = triangulatePoints(P0, P1, a, b)
    const mask = X.map(([x, y, z], i): number => {
      if (options.mask !== undefined && (options.mask[i] ?? 0) === 0) return 0
      if (!(z > 0 && z < dist)) return 0
      const z2 = (c.R[2]?.[0] ?? 0) * x + (c.R[2]?.[1] ?? 0) * y + (c.R[2]?.[2] ?? 0) * z + c.t[2]
      return z2 > 0 && z2 < dist ? 1 : 0
    })
    const good = mask.reduce((s, v) => s + v, 0)
    if (good > best.good) best = { R: c.R, t: c.t, good, mask }
  }
  return best
}

/**
 * The essential matrix of a fundamental matrix and the cameras' intrinsics: E = K2^T F K1.
 *
 * @param f - The fundamental matrix.
 * @param k1 - The first camera's intrinsics.
 * @param k2 - The second camera's (default k1).
 * @returns E.
 * @concept essential-matrix
 * @example
 * ```ts
 * import { essentialFromFundamental } from 'mdml-engine/vision'
 * essentialFromFundamental([[0, 0, 0], [0, 0, -1], [0, 1, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) // => [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
 * ```
 */
export function essentialFromFundamental(
  f: readonly (readonly number[])[],
  k1: readonly (readonly number[])[],
  k2: readonly (readonly number[])[] = k1,
): MutableRows {
  return matMul(matMul(transposeOf(k2), f), k1)
}

/**
 * The fundamental matrix of an essential matrix and the intrinsics: F = K2^-T E K1^-1.
 *
 * @param e - The essential matrix.
 * @param k1 - The first camera's intrinsics.
 * @param k2 - The second camera's (default k1).
 * @returns F.
 * @concept epipolar-geometry
 * @example
 * ```ts
 * import { fundamentalFromEssential } from 'mdml-engine/vision'
 * fundamentalFromEssential([[0, 0, 0], [0, 0, -1], [0, 1, 0]], [[2, 0, 0], [0, 2, 0], [0, 0, 1]]) // => [[0, 0, 0], [0, 0, -0.5], [0, 0.5, 0]]
 * ```
 */
export function fundamentalFromEssential(
  e: readonly (readonly number[])[],
  k1: readonly (readonly number[])[],
  k2: readonly (readonly number[])[] = k1,
): MutableRows {
  return matMul(matMul(transposeOf(invOf(k2)), e), invOf(k1))
}

/**
 * Planar and two-view geometry from point correspondences, OpenCV style: the exact perspective and affine transforms
 * of four and three point pairs, mapping points through a homography, robust homography and affine estimation
 * (normalised DLT or least squares inside RANSAC, then refined on the inliers by Levenberg-Marquardt), the fundamental
 * matrix by the normalised eight-point algorithm with RANSAC, epipolar lines and the Sampson distance. Points are
 * (x, y) pixels; RANSAC draws its samples from an explicit key.
 */

import { MdmlError } from '../base/index.ts'
import { applyHomography, fundamentalMatrix, homography } from '../numerics/geometry/index.ts'
import { dgesv, lstsq } from '../numerics/linalg/index.ts'
import type { Key } from '../random/index.ts'
import { ransac } from '../stats/index.ts'
import {
  concat,
  div,
  matmul,
  reshape,
  sliceAxis,
  sub,
  type Tensor,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { type PointsLike, pointsOf } from './_image.ts'
import { type Matrix, refine } from './_refine.ts'

type P = readonly [number, number]

/** Options of the RANSAC estimators. */
export interface RansacEstimateOptions {
  /** 'ransac' (default) or 'all' (every point is an inlier). */
  readonly method?: 'ransac' | 'all'
  /** The largest reprojection error, in pixels, of an inlier (default 3). */
  readonly threshold?: number
  /** The most RANSAC trials (default 2000). */
  readonly maxIters?: number
  /** Stop once an outlier-free sample is this likely (default 0.995). */
  readonly confidence?: number
  /** The key of the RANSAC samples (default the key of seed 0). */
  readonly key?: Key
}

/** A robust fit: the model and which correspondences are its inliers. */
export interface RobustFit {
  readonly model: Matrix
  /** 1 for an inlier, 0 for an outlier, per correspondence. */
  readonly mask: number[]
}

function pairsOf(src: PointsLike, dst: PointsLike, name: string, min: number): { a: P[]; b: P[] } {
  const a = pointsOf(src, name)
  const b = pointsOf(dst, name)
  if (a.length !== b.length) throw new MdmlError('shape', `${name}: the point sets differ in size`)
  if (a.length < min) throw new MdmlError('value', `${name}: at least ${min} correspondences are needed`)
  return { a, b }
}

/**
 * The perspective transform taking four points exactly onto four others (cv2.getPerspectiveTransform): the 8 x 8
 * linear system for the homography with H[2][2] = 1, solved by LU.
 *
 * @param src - Four source points.
 * @param dst - Four destination points.
 * @returns The 3 x 3 homography.
 * @throws MdmlError when three of the points are collinear (a singular system).
 * @concept homography
 * @example
 * ```ts
 * import { getPerspectiveTransform } from 'mdml-engine/vision'
 * getPerspectiveTransform([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [2, 0], [2, 2], [0, 2]]) // => [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
 * ```
 */
export function getPerspectiveTransform(src: PointsLike, dst: PointsLike): Matrix {
  const { a, b } = pairsOf(src, dst, 'getPerspectiveTransform', 4)
  if (a.length !== 4) throw new MdmlError('value', 'getPerspectiveTransform: exactly four correspondences')
  const A = new Float64Array(64)
  const rhs = new Float64Array(8)
  for (let i = 0; i < 4; i++) {
    const [x, y] = a[i] as P
    const [u, v] = b[i] as P
    A.set([x, y, 1, 0, 0, 0, -x * u, -y * u], i * 8)
    A.set([0, 0, 0, x, y, 1, -x * v, -y * v], (i + 4) * 8)
    rhs[i] = u
    rhs[i + 4] = v
  }
  const h = dgesv(A, rhs, 8)
  return [
    [h[0] ?? 0, h[1] ?? 0, h[2] ?? 0],
    [h[3] ?? 0, h[4] ?? 0, h[5] ?? 0],
    [h[6] ?? 0, h[7] ?? 0, 1],
  ]
}

/**
 * The affine transform taking three points exactly onto three others (cv2.getAffineTransform).
 *
 * @param src - Three source points.
 * @param dst - Three destination points.
 * @returns The 2 x 3 affine matrix.
 * @concept affine-transform
 * @example
 * ```ts
 * import { getAffineTransform } from 'mdml-engine/vision'
 * getAffineTransform([[0, 0], [1, 0], [0, 1]], [[1, 1], [3, 1], [1, 4]]) // => [[2, 0, 1], [0, 3, 1]]
 * ```
 */
export function getAffineTransform(src: PointsLike, dst: PointsLike): Matrix {
  const { a, b } = pairsOf(src, dst, 'getAffineTransform', 3)
  if (a.length !== 3) throw new MdmlError('value', 'getAffineTransform: exactly three correspondences')
  const A = new Float64Array(36)
  const rhs = new Float64Array(6)
  for (let i = 0; i < 3; i++) {
    const [x, y] = a[i] as P
    const [u, v] = b[i] as P
    A.set([x, y, 1, 0, 0, 0], i * 12)
    A.set([0, 0, 0, x, y, 1], i * 12 + 6)
    rhs[2 * i] = u
    rhs[2 * i + 1] = v
  }
  const m = dgesv(A, rhs, 6)
  return [
    [m[0] ?? 0, m[1] ?? 0, m[2] ?? 0],
    [m[3] ?? 0, m[4] ?? 0, m[5] ?? 0],
  ]
}

/**
 * Map points through a homography (cv2.perspectiveTransform for 2-D points), with the perspective division.
 *
 * @param points - Points `[n, 2]`.
 * @param h - The 3 x 3 matrix.
 * @returns The mapped points.
 * @concept homography
 * @example
 * ```ts
 * import { perspectiveTransform } from 'mdml-engine/vision'
 * perspectiveTransform([[2, 1]], [[1, 0, 0], [0, 1, 0], [0, 1, 1]]) // => [[1, 0.5]]
 * ```
 */
export function perspectiveTransform(points: PointsLike, h: readonly (readonly number[])[]): [number, number][] {
  const p = pointsOf(points, 'perspectiveTransform')
  if (p.length === 0) return []
  return toArray(applyHomography(h, p) as Tensor) as [number, number][]
}

/** Reprojection residuals (dst - H src) of a homography with H[2][2] = 1, as a differentiable function of H[0..7]. */
function homographyResiduals(a: readonly P[], b: readonly P[]): (p: Value) => Value {
  const src = tensor(a.map(([x, y]) => [x, y, 1]))
  const dst = tensor(b.map(([x, y]) => [x, y]))
  return (p) => {
    const H = reshape(concat([p, tensor([1])]), [3, 3])
    const q = matmul(src, transpose(H))
    const xy = div(sliceAxis(q, 1, 0, 2), sliceAxis(q, 1, 2, 3))
    return reshape(sub(xy, dst), [-1])
  }
}

/** The homography of correspondences: the normalised DLT, refined by minimising the reprojection error. */
function fitHomography(a: readonly P[], b: readonly P[], polish: boolean): Matrix {
  const h0 = homography(a as P[], b as P[])
  if (!polish || a.length <= 4) return h0
  const x = refine(homographyResiduals(a, b), h0.flat().slice(0, 8), 50)
  return [
    [x[0] ?? 0, x[1] ?? 0, x[2] ?? 0],
    [x[3] ?? 0, x[4] ?? 0, x[5] ?? 0],
    [x[6] ?? 0, x[7] ?? 0, 1],
  ]
}

/** The reprojection error of a correspondence under a homography. */
const homographyError = (h: Matrix, p: P, q: P): number => {
  const w = (h[2]?.[0] ?? 0) * p[0] + (h[2]?.[1] ?? 0) * p[1] + (h[2]?.[2] ?? 0)
  const x = ((h[0]?.[0] ?? 0) * p[0] + (h[0]?.[1] ?? 0) * p[1] + (h[0]?.[2] ?? 0)) / w
  const y = ((h[1]?.[0] ?? 0) * p[0] + (h[1]?.[1] ?? 0) * p[1] + (h[1]?.[2] ?? 0)) / w
  return Math.hypot(x - q[0], y - q[1])
}

/** Run RANSAC over correspondences with a model that may fail on degenerate samples. */
function robust(
  a: readonly P[],
  b: readonly P[],
  minSamples: number,
  fit: (a: readonly P[], b: readonly P[], final: boolean) => Matrix,
  error: (m: Matrix, p: P, q: P) => number,
  options: RansacEstimateOptions,
  name: string,
): RobustFit {
  const n = a.length
  if ((options.method ?? 'ransac') === 'all') return { model: fit(a, b, true), mask: new Array<number>(n).fill(1) }
  const idx = Array.from({ length: n }, (_, i) => i)
  let final = false
  const model = {
    fit: (sample: readonly number[]): Matrix | undefined => {
      try {
        return fit(
          sample.map((i) => a[i] as P),
          sample.map((i) => b[i] as P),
          final,
        )
      } catch {
        return undefined
      }
    },
    residual: (m: Matrix | undefined, i: number): number =>
      m === undefined ? Number.POSITIVE_INFINITY : error(m, a[i] as P, b[i] as P),
  }
  const r = ransac(idx, model, {
    minSamples,
    residualThreshold: options.threshold ?? 3,
    maxTrials: options.maxIters ?? 2000,
    stopProbability: options.confidence ?? 0.995,
    ...(options.key === undefined ? { seed: 0 } : { key: options.key }),
  })
  final = true
  // Local optimisation (as LO-RANSAC and OpenCV's USAC): refit on the inliers and re-score until the set is stable.
  const threshold = options.threshold ?? 3
  let inliers = [...r.inliers]
  let refit: Matrix | undefined
  for (let it = 0; it < 10; it++) {
    refit = model.fit(inliers)
    if (refit === undefined) throw new MdmlError('value', `${name}: the inliers are degenerate`)
    const fitted = refit
    const next = idx.filter((i) => error(fitted, a[i] as P, b[i] as P) <= threshold)
    if (next.length < minSamples || (next.length === inliers.length && next.every((v, k) => v === inliers[k]))) break
    inliers = next
  }
  if (refit === undefined) throw new MdmlError('value', `${name}: the inliers are degenerate`)
  const mask = new Array<number>(n).fill(0)
  for (const i of inliers) mask[i] = 1
  return { model: refit, mask }
}

/**
 * Estimate the homography between two point sets (cv2.findHomography): with method 'all', the normalised DLT over
 * every pair; with 'ransac', the DLT of random minimal samples scored by reprojection error, its inliers refitted. The
 * final fit is refined by Levenberg-Marquardt on the reprojection error, as OpenCV does, so on clean data the two
 * agree to rounding (RANSAC draws different samples, so on noisy data with outliers only the inlier set and the fit to
 * it are comparable).
 *
 * @param src - Source points `[n, 2]`, n >= 4.
 * @param dst - Destination points `[n, 2]`.
 * @param options - Method, threshold, trials, confidence and key.
 * @returns The homography (H[2][2] = 1) and the inlier mask.
 * @concept homography
 * @index function Robust homography from point correspondences.
 * @example
 * ```ts
 * import { findHomography } from 'mdml-engine/vision'
 * const src = [[0, 0], [4, 0], [4, 4], [0, 4], [2, 1]]
 * const { model } = findHomography(src, src.map(([x, y]) => [2 * x + 1, 2 * y - 3]), { method: 'all' })
 * model.map((r) => r.map((v) => Math.round(v * 1e9) / 1e9)) // => [[2, 0, 1], [0, 2, -3], [0, 0, 1]]
 * ```
 */
export function findHomography(src: PointsLike, dst: PointsLike, options: RansacEstimateOptions = {}): RobustFit {
  const { a, b } = pairsOf(src, dst, 'findHomography', 4)
  return robust(a, b, 4, fitHomography, homographyError, options, 'findHomography')
}

/** The least-squares affine map of correspondences (exact for three). */
function fitAffine(a: readonly P[], b: readonly P[]): Matrix {
  const A = a.map(([x, y]) => [x, y, 1])
  const sol = toFlat(
    lstsq(
      A,
      b.map(([u, v]) => [u, v]),
    ) as Tensor,
  )
  return [
    [sol[0] ?? 0, sol[2] ?? 0, sol[4] ?? 0],
    [sol[1] ?? 0, sol[3] ?? 0, sol[5] ?? 0],
  ]
}

/**
 * Estimate the affine map between two point sets (cv2.estimateAffine2D): least squares over every pair, or RANSAC
 * over minimal three-point samples with the inliers refitted (the least-squares fit is the optimum OpenCV's
 * Levenberg-Marquardt refinement reaches, the residuals being linear).
 *
 * @param src - Source points `[n, 2]`, n >= 3.
 * @param dst - Destination points `[n, 2]`.
 * @param options - Method, threshold, trials, confidence and key.
 * @returns The 2 x 3 affine matrix and the inlier mask.
 * @concept affine-transform
 * @example
 * ```ts
 * import { estimateAffine2D } from 'mdml-engine/vision'
 * const src = [[0, 0], [1, 0], [0, 1], [3, 2]]
 * estimateAffine2D(src, src.map(([x, y]) => [x - y + 2, x + y]), { method: 'all' }).model.map((r) => r.map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, -1, 2], [1, 1, 0]]
 * ```
 */
export function estimateAffine2D(src: PointsLike, dst: PointsLike, options: RansacEstimateOptions = {}): RobustFit {
  const { a, b } = pairsOf(src, dst, 'estimateAffine2D', 3)
  const error = (m: Matrix, p: P, q: P): number =>
    Math.hypot(
      (m[0]?.[0] ?? 0) * p[0] + (m[0]?.[1] ?? 0) * p[1] + (m[0]?.[2] ?? 0) - q[0],
      (m[1]?.[0] ?? 0) * p[0] + (m[1]?.[1] ?? 0) * p[1] + (m[1]?.[2] ?? 0) - q[1],
    )
  return robust(a, b, 3, fitAffine, error, options, 'estimateAffine2D')
}

/** The fundamental matrix of correspondences by the normalised eight-point algorithm, scaled to F[2][2] = 1. */
function fitFundamental(a: readonly P[], b: readonly P[]): Matrix {
  const f = fundamentalMatrix(a as P[], b as P[])
  const s = f[2]?.[2] ?? 0
  return Math.abs(s) > 1.1920928955078125e-7 ? f.map((r) => r.map((v) => v / s)) : f
}

/** The larger distance of a correspondence from its two epipolar lines (OpenCV's fundamental-matrix error). */
function epipolarError(f: Matrix, p: P, q: P): number {
  const F = (i: number, j: number): number => f[i]?.[j] ?? 0
  let a = F(0, 0) * p[0] + F(0, 1) * p[1] + F(0, 2)
  let b = F(1, 0) * p[0] + F(1, 1) * p[1] + F(1, 2)
  let c = F(2, 0) * p[0] + F(2, 1) * p[1] + F(2, 2)
  const d2 = (q[0] * a + q[1] * b + c) ** 2 / (a * a + b * b)
  a = F(0, 0) * q[0] + F(1, 0) * q[1] + F(2, 0)
  b = F(0, 1) * q[0] + F(1, 1) * q[1] + F(2, 1)
  c = F(0, 2) * q[0] + F(1, 2) * q[1] + F(2, 2)
  const d1 = (p[0] * a + p[1] * b + c) ** 2 / (a * a + b * b)
  return Math.sqrt(Math.max(d1, d2))
}

/**
 * Estimate the fundamental matrix (cv2.findFundamentalMat): the normalised eight-point algorithm with rank two
 * enforced (method 'all', OpenCV's FM_8POINT), or inside RANSAC over eight-point samples scored by the distance to the
 * epipolar lines (OpenCV samples seven points); scaled so F[2][2] = 1, as OpenCV does.
 *
 * @param points1 - Points in the first image `[n, 2]`, n >= 8.
 * @param points2 - Matching points in the second image.
 * @param options - Method, threshold (default 3 pixels), trials, confidence and key.
 * @returns F with x2^T F x1 = 0 and the inlier mask.
 * @concept epipolar-geometry
 * @example
 * ```ts
 * import { findFundamentalMat } from 'mdml-engine/vision'
 * const p1 = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1], [2, 3], [4, 2]]
 * findFundamentalMat(p1, p1.map(([x, y]) => [x + 1, y]), { method: 'all' }).model.length // => 3
 * ```
 */
export function findFundamentalMat(
  points1: PointsLike,
  points2: PointsLike,
  options: RansacEstimateOptions = {},
): RobustFit {
  const { a, b } = pairsOf(points1, points2, 'findFundamentalMat', 8)
  return robust(a, b, 8, fitFundamental, epipolarError, options, 'findFundamentalMat')
}

/**
 * Epipolar lines of points (cv2.computeCorrespondEpilines): for points in image 1 the lines F x in image 2, for
 * points in image 2 the lines F^T x in image 1, as [a, b, c] with a x + b y + c = 0 and a^2 + b^2 = 1.
 *
 * @param points - Points `[n, 2]`.
 * @param whichImage - The image the points are in (1 or 2).
 * @param f - The fundamental matrix.
 * @returns One line per point.
 * @concept epipolar-geometry
 * @example
 * ```ts
 * import { computeCorrespondEpilines } from 'mdml-engine/vision'
 * computeCorrespondEpilines([[1, 2]], 1, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]) // => [[0, -1, 2]]
 * ```
 */
export function computeCorrespondEpilines(
  points: PointsLike,
  whichImage: 1 | 2,
  f: readonly (readonly number[])[],
): [number, number, number][] {
  const p = pointsOf(points, 'computeCorrespondEpilines')
  const F = (i: number, j: number): number => (whichImage === 1 ? f[i]?.[j] : f[j]?.[i]) ?? 0
  return p.map(([x, y]) => {
    const a = F(0, 0) * x + F(0, 1) * y + F(0, 2)
    const b = F(1, 0) * x + F(1, 1) * y + F(1, 2)
    const c = F(2, 0) * x + F(2, 1) * y + F(2, 2)
    const s = Math.hypot(a, b)
    return s > 0 ? [a / s, b / s, c / s] : [a, b, c]
  })
}

/**
 * The Sampson distance of a correspondence under a fundamental matrix (cv2.sampsonDistance): the first-order
 * geometric error (x2^T F x1)^2 / ((F x1)_1^2 + (F x1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2).
 *
 * @param p1 - The point in image 1.
 * @param p2 - The point in image 2.
 * @param f - The fundamental matrix.
 * @returns The squared distance.
 * @concept epipolar-geometry
 * @example
 * ```ts
 * import { sampsonDistance } from 'mdml-engine/vision'
 * sampsonDistance([1, 2], [3, 2], [[0, 0, 0], [0, 0, -1], [0, 1, 0]]) // => 0
 * ```
 */
export function sampsonDistance(
  p1: readonly [number, number],
  p2: readonly [number, number],
  f: readonly (readonly number[])[],
): number {
  const F = (i: number, j: number): number => f[i]?.[j] ?? 0
  const fx = [0, 1, 2].map((i) => F(i, 0) * p1[0] + F(i, 1) * p1[1] + F(i, 2))
  const ftx = [0, 1, 2].map((j) => F(0, j) * p2[0] + F(1, j) * p2[1] + F(2, j))
  const num = p2[0] * (fx[0] ?? 0) + p2[1] * (fx[1] ?? 0) + (fx[2] ?? 0)
  const den = (fx[0] ?? 0) ** 2 + (fx[1] ?? 0) ** 2 + (ftx[0] ?? 0) ** 2 + (ftx[1] ?? 0) ** 2
  return (num * num) / den
}

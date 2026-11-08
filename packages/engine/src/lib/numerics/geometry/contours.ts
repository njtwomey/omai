/**
 * Contours of gridded data and of Gaussians: iso-lines of a 2-D array by marching squares (as skimage's
 * `find_contours`), nice contour levels (as matplotlib's `MaxNLocator`), and the confidence ellipse of a 2-D Gaussian
 * at a number of standard deviations or a coverage probability.
 */

import { maxOf, minOf, OmaiError } from '../../base/index.ts'
import {
  add,
  concreteOrThrow,
  cos,
  flip,
  fromStorage,
  matmul,
  mul,
  promoteFloat,
  reshape,
  sin,
  sqrt,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../../tensor/index.ts'
import { vectorOf } from '../_vector.ts'
import { eigh } from '../linalg/index.ts'
import type { Point2 } from './triangulation.ts'

/** Options of {@link contourLines}. */
export interface ContourOptions {
  /** At saddles, connect the low-valued corners (default, as skimage) or the high-valued ones. */
  readonly fullyConnected?: 'low' | 'high'
  /** Wind contours counter-clockwise around low values (default) or high values. */
  readonly positiveOrientation?: 'low' | 'high'
  /** Column coordinates (length = columns): return [x, y] points instead of [row, col] indices. */
  readonly x?: readonly number[]
  /** Row coordinates (length = rows). */
  readonly y?: readonly number[]
}

/**
 * Iso-value contours of a 2-D array by marching squares, as skimage.measure.find_contours: each grid square crossing the
 * level contributes segments between linearly interpolated edge points, and segments are joined into paths (closed
 * paths repeat their first point). Points are (row, col), or (x, y) when coordinates are given.
 *
 * @param z - Values `[rows, cols]` (NaN squares are skipped).
 * @param level - The iso-value.
 * @param options - Saddle connectivity, orientation and coordinates.
 * @returns The paths.
 * @concept marching-squares
 * @example
 * ```ts
 * import { contourLines } from 'omai-engine/numerics/geometry'
 * contourLines([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 0.5)[0].length // => 5
 * ```
 */
export function contourLines(z: TensorLike, level: number, options: ContourOptions = {}): [number, number][][] {
  const t = concreteOrThrow(promoteFloat(z)[0] as Tensor, 'contourLines')
  if (t.shape.length !== 2) throw new OmaiError('shape', 'contourLines: z must be 2-D')
  const [rows = 0, cols = 0] = t.shape
  const v = toFlat(t)
  const at = (r: number, c: number): number => v[r * cols + c] ?? Number.NaN
  const fraction = (from: number, to: number): number => (to === from ? 0 : (level - from) / (to - from))
  const high = options.fullyConnected === 'high'
  const segments: [Point2, Point2][] = []
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const ul = at(r, c)
      const ur = at(r, c + 1)
      const ll = at(r + 1, c)
      const lr = at(r + 1, c + 1)
      if ([ul, ur, ll, lr].some(Number.isNaN)) continue
      let square = 0
      if (ul > level) square |= 1
      if (ur > level) square |= 2
      if (ll > level) square |= 4
      if (lr > level) square |= 8
      if (square === 0 || square === 15) continue
      const top: Point2 = [r, c + fraction(ul, ur)]
      const bottom: Point2 = [r + 1, c + fraction(ll, lr)]
      const left: Point2 = [r + fraction(ul, ll), c]
      const right: Point2 = [r + fraction(ur, lr), c + 1]
      switch (square) {
        case 1:
          segments.push([top, left])
          break
        case 2:
          segments.push([right, top])
          break
        case 3:
          segments.push([right, left])
          break
        case 4:
          segments.push([left, bottom])
          break
        case 5:
          segments.push([top, bottom])
          break
        case 6:
          if (high) segments.push([left, top], [right, bottom])
          else segments.push([right, top], [left, bottom])
          break
        case 7:
          segments.push([right, bottom])
          break
        case 8:
          segments.push([bottom, right])
          break
        case 9:
          if (high) segments.push([top, right], [bottom, left])
          else segments.push([top, left], [bottom, right])
          break
        case 10:
          segments.push([bottom, top])
          break
        case 11:
          segments.push([bottom, left])
          break
        case 12:
          segments.push([left, right])
          break
        case 13:
          segments.push([top, right])
          break
        case 14:
          segments.push([left, top])
          break
      }
    }
  }
  let paths = assemble(segments)
  if (options.positiveOrientation === 'high') paths = paths.map((p) => [...p].reverse())
  if (options.x !== undefined || options.y !== undefined) {
    const xs = options.x ?? Array.from({ length: cols }, (_, i) => i)
    const ys = options.y ?? Array.from({ length: rows }, (_, i) => i)
    const lerp = (axis: readonly number[], u: number): number => {
      const i = Math.min(Math.floor(u), axis.length - 2)
      const f = u - i
      return (axis[i] ?? 0) * (1 - f) + (axis[i + 1] ?? 0) * f
    }
    return paths.map((p) => p.map(([r, c]): [number, number] => [lerp(xs, c), lerp(ys, r)]))
  }
  return paths.map((p) => p.map(([r, c]): [number, number] => [r, c]))
}

/** Join directed segments into paths (skimage's _assemble_contours, keeping its ordering). */
function assemble(segments: readonly [Point2, Point2][]): Point2[][] {
  const key = (p: Point2): string => `${p[0]},${p[1]}`
  type Entry = { path: Point2[]; id: number }
  const contours = new Map<number, Point2[]>()
  const starts = new Map<string, Entry>()
  const ends = new Map<string, Entry>()
  let next = 0
  for (const [from, to] of segments) {
    if (from[0] === to[0] && from[1] === to[1]) continue
    const tail = starts.get(key(to))
    if (tail !== undefined) starts.delete(key(to))
    const head = ends.get(key(from))
    if (head !== undefined) ends.delete(key(from))
    if (tail !== undefined && head !== undefined) {
      if (tail.path === head.path) {
        head.path.push(to)
      } else if (tail.id > head.id) {
        head.path.push(...tail.path)
        contours.delete(tail.id)
        starts.set(key(head.path[0] as Point2), head)
        ends.set(key(head.path[head.path.length - 1] as Point2), head)
      } else {
        tail.path.unshift(...head.path)
        starts.delete(key(head.path[0] as Point2))
        contours.delete(head.id)
        starts.set(key(tail.path[0] as Point2), tail)
        ends.set(key(tail.path[tail.path.length - 1] as Point2), tail)
      }
    } else if (tail === undefined && head === undefined) {
      const path: Point2[] = [from, to]
      const entry = { path, id: next }
      contours.set(next, path)
      starts.set(key(from), entry)
      ends.set(key(to), entry)
      next++
    } else if (head === undefined && tail !== undefined) {
      tail.path.unshift(from)
      starts.set(key(from), tail)
    } else if (head !== undefined) {
      head.path.push(to)
      ends.set(key(to), head)
    }
  }
  return [...contours.entries()].sort((a, b) => a[0] - b[0]).map(([, p]) => p)
}

/**
 * Nice contour levels inside the range of the data (as matplotlib's MaxNLocator): steps of 1, 2, 2.5 or 5 times a power
 * of ten, at most n intervals, keeping only levels strictly between the minimum and maximum. Returns a tensor
 * (migration, D-009: was `number[]`).
 *
 * @param z - Values (any shape; NaNs ignored).
 * @param n - Target number of intervals (default 7).
 * @returns Increasing levels `[k]`.
 * @example
 * ```ts
 * import { contourLevels } from 'omai-engine/numerics/geometry'
 * contourLevels([0, 1], 4) // => [0.25, 0.5, 0.75]
 * ```
 */
export function contourLevels(z: TensorLike, n = 7): Tensor<'float64', [number]> {
  return vectorOf(levelsOf(z, n))
}

function levelsOf(z: TensorLike, n: number): number[] {
  const values = toFlat(concreteOrThrow(promoteFloat(z)[0] as Tensor, 'contourLevels')).filter((v) => !Number.isNaN(v))
  const lo = minOf(values)
  const hi = maxOf(values)
  if (!(hi > lo)) return []
  const raw = (hi - lo) / n
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw * (1 - 1e-12)) ?? 10 * power
  const levels: number[] = []
  for (let k = Math.ceil(lo / step); k * step <= hi + 1e-12 * step; k++) {
    const level = Number((k * step).toPrecision(12))
    if (level > lo && level < hi) levels.push(level)
  }
  return levels
}

/**
 * Contours at several levels. Takes the levels as a tensor-like (migration, D-009: was `readonly number[]`); the paths
 * are ragged lists of points and stay lists.
 *
 * @param z - Values `[rows, cols]`.
 * @param levels - The levels `[k]` (default {@link contourLevels}).
 * @param options - As {@link contourLines}.
 * @returns One entry per level with its paths.
 * @throws OmaiError with code `shape` unless `levels` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { isolines } from 'omai-engine/numerics/geometry'
 * isolines([[0, 1], [2, 3]], [1.5]).length // => 1
 * ```
 */
export function isolines(
  z: TensorLike,
  levels?: TensorLike,
  options: ContourOptions = {},
): { level: number; paths: [number, number][][] }[] {
  return (levels === undefined ? levelsOf(z, 7) : toVector(levels, 'isolines')).map((level) => ({
    level,
    paths: contourLines(z, level, options),
  }))
}

/** A confidence ellipse of a 2-D Gaussian. */
export interface CovarianceEllipse {
  /** Points on the ellipse `[n, 2]`. */
  readonly points: Value
  /** Semi-axes, major first. */
  readonly semiAxes: Value
  /** Angle of the major axis from the x axis (radians). */
  readonly angle: number
}

/**
 * The ellipse {x : (x - mu)^T Sigma^-1 (x - mu) = k^2} of a 2-D Gaussian: k standard deviations, or the radius holding
 * a given probability (k^2 = -2 log(1 - p), the chi-squared quantile with two degrees of freedom). Built from the
 * eigendecomposition with operations, so differentiable in the mean and covariance.
 *
 * @param mu - Mean `[2]`.
 * @param sigma - Covariance `[2, 2]`.
 * @param options - k (default 1) or a confidence level, and the number of points (default 100).
 * @returns Points, semi-axes and orientation.
 * @example
 * ```ts
 * import { covarianceEllipse } from 'omai-engine/numerics/geometry'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(covarianceEllipse([0, 0], [[4, 0], [0, 1]]).semiAxes) // => [2, 1]
 * ```
 */
export function covarianceEllipse(
  mu: TensorLike,
  sigma: TensorLike,
  options: { readonly nStd?: number; readonly confidence?: number; readonly points?: number } = {},
): CovarianceEllipse {
  const [m, s] = promoteFloat(mu, sigma) as [Value, Value]
  const k = options.confidence !== undefined ? Math.sqrt(-2 * Math.log(1 - options.confidence)) : (options.nStd ?? 1)
  const { eigenvalues, eigenvectors } = eigh(s)
  // eigh sorts ascending; the major axis is the last.
  const radii = mul(sqrt(eigenvalues), k)
  const n = options.points ?? 100
  const theta = fromStorage(
    Float64Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n),
    [n, 1],
    'float64',
  )
  const circle = add(
    mul(cos(theta), fromStorage(Float64Array.from([1, 0]), [1, 2], 'float64')),
    mul(sin(theta), fromStorage(Float64Array.from([0, 1]), [1, 2], 'float64')),
  )
  // Columns of the circle scaled by the radii (ascending), then rotated by the eigenvectors.
  const scaled = mul(circle, reshape(radii, [1, 2]))
  const points = add(matmul(scaled, swapaxes(eigenvectors, 0, 1)), reshape(m, [1, 2]))
  const vec = toFlat(eigenvectors as Tensor)
  return {
    points,
    semiAxes: flip(radii, 0),
    angle: Math.atan2(vec[3] ?? 0, vec[1] ?? 0),
  }
}

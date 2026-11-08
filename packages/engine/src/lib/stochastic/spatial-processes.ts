/**
 * Spatial point processes in a rectangle: homogeneous and inhomogeneous spatial Poisson processes, the Thomas and
 * Matern cluster processes (Poisson parents with scattered children), and Ripley's K and Besag's L functions with edge
 * corrections (as astropy), which measure clustering or regularity against complete spatial randomness.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, normal, poisson, uniform } from '../random/index.ts'
import {
  add,
  exp,
  fromStorage,
  mul,
  promoteFloat,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { primalNumbers } from './_shared.ts'

/** A rectangular observation window. */
export interface Window {
  readonly xmin: number
  readonly xmax: number
  readonly ymin: number
  readonly ymax: number
}

const UNIT: Window = { xmin: 0, xmax: 1, ymin: 0, ymax: 1 }

function areaOf(w: Window): number {
  const a = (w.xmax - w.xmin) * (w.ymax - w.ymin)
  if (!(a > 0)) throw new OmaiError('value', 'window: needs xmax > xmin and ymax > ymin')
  return a
}

function uniformPoints(key: Key, n: number, w: Window): number[] {
  const u = uniform(key, [n, 2]).data
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    out.push(w.xmin + (u[2 * i] ?? 0) * (w.xmax - w.xmin), w.ymin + (u[2 * i + 1] ?? 0) * (w.ymax - w.ymin))
  }
  return out
}

function asPoints(flat: readonly number[]): Tensor<'float64'> {
  return fromStorage(Float64Array.from(flat), [flat.length / 2, 2], 'float64')
}

/**
 * A spatial Poisson process on a rectangle: a Poisson(lambda |W|) number of uniform points, thinned by
 * lambda(x, y) / bound for an inhomogeneous intensity.
 *
 * @param key - The key.
 * @param intensity - A constant intensity, or a function lambda(x, y).
 * @param window - The rectangle (default the unit square).
 * @param maxIntensity - A bound on an intensity function (required for one).
 * @returns The points `[n, 2]`.
 * @throws OmaiError with code `value` when a function exceeds its bound.
 * @concept spatial-point-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { spatialPoisson } from 'omai-engine/stochastic'
 * spatialPoisson(key(0), 50).shape[1] // => 2
 * ```
 */
export function spatialPoisson(
  key: Key,
  intensity: number | ((x: number, y: number) => number),
  window: Window = UNIT,
  maxIntensity?: number,
): Tensor<'float64'> {
  const area = areaOf(window)
  const bound = typeof intensity === 'number' ? intensity : maxIntensity
  if (bound === undefined || !(bound >= 0)) {
    throw new OmaiError('value', 'spatialPoisson: a non-negative intensity (or bound for a function) is needed')
  }
  const n = poisson(foldIn(key, 0), bound * area, [1]).data[0] ?? 0
  const pts = uniformPoints(foldIn(key, 1), n, window)
  if (typeof intensity === 'number') return asPoints(pts)
  const u = uniform(foldIn(key, 2), [n]).data
  const kept: number[] = []
  for (let i = 0; i < n; i++) {
    const x = pts[2 * i] ?? 0
    const y = pts[2 * i + 1] ?? 0
    const lam = intensity(x, y)
    if (!(lam >= 0) || lam > bound * (1 + 1e-12)) throw new OmaiError('value', 'spatialPoisson: bound exceeded')
    if ((u[i] ?? 1) * bound < lam) kept.push(x, y)
  }
  return asPoints(kept)
}

/** Parameters of the Neyman-Scott cluster processes. */
export interface ClusterParams {
  /** The intensity kappa of the parent points. */
  readonly parentRate: number
  /** The mean number of children per parent. */
  readonly meanChildren: number
}

function clusterProcess(
  key: Key,
  params: ClusterParams,
  window: Window,
  margin: number,
  offset: (key: Key, n: number) => number[],
): Tensor<'float64'> {
  areaOf(window)
  if (!(params.parentRate >= 0 && params.meanChildren >= 0)) throw new OmaiError('value', 'cluster: bad rates')
  // parents in a window enlarged by the margin, so clusters straddling the edge are not lost
  const big = {
    xmin: window.xmin - margin,
    xmax: window.xmax + margin,
    ymin: window.ymin - margin,
    ymax: window.ymax + margin,
  }
  const parents = primalNumbers(spatialPoisson(foldIn(key, 0), params.parentRate, big))
  const kept: number[] = []
  for (let p = 0; p < parents.length / 2; p++) {
    const kids = poisson(foldIn(key, 1 + 2 * p), params.meanChildren, [1]).data[0] ?? 0
    const d = offset(foldIn(key, 2 + 2 * p), kids)
    for (let c = 0; c < kids; c++) {
      const x = (parents[2 * p] ?? 0) + (d[2 * c] ?? 0)
      const y = (parents[2 * p + 1] ?? 0) + (d[2 * c + 1] ?? 0)
      if (x >= window.xmin && x <= window.xmax && y >= window.ymin && y <= window.ymax) kept.push(x, y)
    }
  }
  return asPoints(kept)
}

/**
 * The Thomas cluster process: Poisson parents with intensity kappa, each with a Poisson(mu) number of children displaced
 * by N(0, sigma^2 I). Its K function is pi r^2 + (1 - exp(-r^2 / (4 sigma^2))) / kappa ({@link thomasK}).
 *
 * @param key - The key.
 * @param params - Parent intensity, mean children and the displacement scale sigma.
 * @param window - The rectangle (default the unit square).
 * @returns The points `[n, 2]`.
 * @concept cluster-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { thomasProcess } from 'omai-engine/stochastic'
 * thomasProcess(key(0), { parentRate: 10, meanChildren: 5, sigma: 0.02 }).shape[1] // => 2
 * ```
 */
export function thomasProcess(
  key: Key,
  params: ClusterParams & { readonly sigma: number },
  window: Window = UNIT,
): Tensor<'float64'> {
  if (!(params.sigma > 0)) throw new OmaiError('value', 'thomasProcess: sigma must be positive')
  return clusterProcess(key, params, window, 4 * params.sigma, (k, n) =>
    Array.from(normal(k, [n, 2]).data, (z) => z * params.sigma),
  )
}

/**
 * The Matern cluster process: Poisson parents with intensity kappa, each with a Poisson(mu) number of children uniform
 * in the disc of radius R around it.
 *
 * @param key - The key.
 * @param params - Parent intensity, mean children and the radius R.
 * @param window - The rectangle (default the unit square).
 * @returns The points `[n, 2]`.
 * @concept cluster-process
 * @example
 * ```ts
 * import { key } from 'omai-engine/random'
 * import { maternClusterProcess } from 'omai-engine/stochastic'
 * maternClusterProcess(key(0), { parentRate: 10, meanChildren: 5, radius: 0.05 }).shape[1] // => 2
 * ```
 */
export function maternClusterProcess(
  key: Key,
  params: ClusterParams & { readonly radius: number },
  window: Window = UNIT,
): Tensor<'float64'> {
  if (!(params.radius > 0)) throw new OmaiError('value', 'maternClusterProcess: the radius must be positive')
  return clusterProcess(key, params, window, params.radius, (k, n) => {
    const u = uniform(k, [n, 2]).data
    const out: number[] = []
    for (let i = 0; i < n; i++) {
      const r = params.radius * Math.sqrt(u[2 * i] ?? 0)
      const a = 2 * Math.PI * (u[2 * i + 1] ?? 0)
      out.push(r * Math.cos(a), r * Math.sin(a))
    }
    return out
  })
}

/**
 * The K function of the Thomas process, pi r^2 + (1 - exp(-r^2 / (4 sigma^2))) / kappa (Poisson's pi r^2 plus the
 * clustering excess).
 *
 * @param r - Distances.
 * @param params - Parent intensity kappa and displacement scale sigma.
 * @returns K(r).
 * @concept ripleys-k
 * @example
 * ```ts
 * import { thomasK } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(thomasK([0], { parentRate: 10, sigma: 0.1 })) // => [0]
 * ```
 */
export function thomasK(r: TensorLike, params: { readonly parentRate: number; readonly sigma: number }): Value {
  const [rr] = promoteFloat(r) as [Value]
  const r2 = mul(rr, rr)
  return add(mul(r2, Math.PI), mul(sub(1, exp(mul(r2, -1 / (4 * params.sigma ** 2)))), 1 / params.parentRate))
}

/** The edge corrections of {@link ripleysK} (astropy.stats.RipleysKEstimator's modes). */
export type RipleyCorrection = 'none' | 'translation' | 'ohser' | 'var-width' | 'ripley'

/**
 * Ripley's K function of a planar point pattern in a rectangle, K(r) = |W| / (n (n - 1)) sum_{i != j} w_ij 1(d_ij < r),
 * with the edge corrections of astropy.stats.RipleysKEstimator: none, translation (Stoyan's 1 / |W cap (W + d_ij)|
 * weights), Ohser's isotropised set covariance, variable width (only points farther than r from the boundary) and
 * Ripley's isotropic correction. Under complete spatial randomness K(r) = pi r^2; clustering raises it.
 *
 * @param points - The points `[n, 2]` inside the window.
 * @param radii - The distances r.
 * @param options - The window (default the unit square) and the correction (default `none`).
 * @returns K at each radius.
 * @concept ripleys-k
 * @reference B. D. Ripley, The second-order analysis of stationary point processes, J. Appl. Prob. 13 (1976).
 * @example
 * ```ts
 * import { ripleysK } from 'omai-engine/stochastic'
 * Array.from(ripleysK([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [0.05, 0.2])) // => [0, 0.3333333333333333]
 * ```
 */
export function ripleysK(
  points: TensorLike,
  radii: ArrayLike<number>,
  options: { readonly window?: Window; readonly correction?: RipleyCorrection } = {},
): Float64Array {
  const w = options.window ?? UNIT
  const area = areaOf(w)
  const p = primalNumbers(points)
  const n = p.length / 2
  if (n < 2) throw new OmaiError('value', 'ripleysK: needs at least two points')
  const width = w.xmax - w.xmin
  const height = w.ymax - w.ymin
  const out = new Float64Array(radii.length)
  const mode = options.correction ?? 'none'
  if (mode === 'var-width') {
    for (let r = 0; r < radii.length; r++) {
      const rad = radii[r] ?? 0
      let total = 0
      let interior = 0
      for (let i = 0; i < n; i++) {
        const xi = p[2 * i] ?? 0
        const yi = p[2 * i + 1] ?? 0
        const edge = Math.min(Math.min(w.xmax - xi, w.ymax - yi), Math.min(xi - w.xmin, yi - w.ymin))
        if (!(edge > rad)) continue
        interior += 1
        for (let j = 0; j < n; j++)
          if (j !== i && Math.hypot(xi - (p[2 * j] ?? 0), yi - (p[2 * j + 1] ?? 0)) < rad) total += 1
      }
      out[r] = (area * (interior === 0 ? total : total / interior)) / n
    }
    return out
  }
  const ratio = Math.max(height / width, width / height)
  for (let i = 0; i < n - 1; i++) {
    const xi = p[2 * i] ?? 0
    const yi = p[2 * i + 1] ?? 0
    const hor = Math.min(w.xmax - xi, xi - w.xmin)
    const ver = Math.min(w.ymax - yi, yi - w.ymin)
    for (let j = i + 1; j < n; j++) {
      const dx = Math.abs(xi - (p[2 * j] ?? 0))
      const dy = Math.abs(yi - (p[2 * j + 1] ?? 0))
      const d = Math.hypot(dx, dy)
      let weight = 1
      if (mode === 'translation') weight = 1 / ((width - dx) * (height - dy))
      else if (mode === 'ohser') {
        const x = d / Math.sqrt(area / ratio)
        const u = x > 1 ? Math.sqrt(x * x - 1) : 0
        const v = x > ratio && x < Math.sqrt(ratio * ratio + 1) ? Math.sqrt(x * x - ratio * ratio) : 0
        let c = 0
        if (x >= 0 && x <= 1) c = Math.PI - 2 * x * (1 + 1 / ratio) + (x * x) / ratio
        else if (x > 1 && x <= ratio) c = 2 * Math.asin(1 / x) - 1 / ratio - 2 * (x - u)
        else if (x > ratio && x < Math.sqrt(ratio * ratio + 1)) {
          c = 2 * Math.asin((ratio - u * v) / (x * x)) + 2 * u + (2 * v) / ratio - ratio - (1 + x * x) / ratio
        }
        weight = 1 / ((area / Math.PI) * c)
      } else if (mode === 'ripley') {
        const inside = d <= Math.hypot(hor, ver)
        const w1 = 1 - (Math.acos(Math.min(ver, d) / d) + Math.acos(Math.min(hor, d) / d)) / Math.PI
        const w2 = 0.75 - (0.5 * (Math.acos(ver / d) + Math.acos(hor / d))) / Math.PI
        weight = 1 / (inside ? w1 : w2)
      }
      for (let r = 0; r < radii.length; r++) if (d < (radii[r] ?? 0)) out[r] = (out[r] ?? 0) + weight
    }
  }
  const scale = mode === 'none' || mode === 'ripley' ? (2 * area) / (n * (n - 1)) : (2 * area * area) / (n * (n - 1))
  for (let r = 0; r < radii.length; r++) out[r] = (out[r] ?? 0) * scale
  return out
}

/**
 * Besag's L function L(r) = sqrt(K(r) / pi), which is r under complete spatial randomness.
 *
 * @param points - The points `[n, 2]`.
 * @param radii - The distances.
 * @param options - Window and edge correction, as for {@link ripleysK}.
 * @returns L at each radius.
 * @concept ripleys-k
 * @example
 * ```ts
 * import { ripleysL } from 'omai-engine/stochastic'
 * Array.from(ripleysL([[0.1, 0.1], [0.2, 0.1], [0.9, 0.9]], [0.05])) // => [0]
 * ```
 */
export function ripleysL(
  points: TensorLike,
  radii: ArrayLike<number>,
  options: { readonly window?: Window; readonly correction?: RipleyCorrection } = {},
): Float64Array {
  return Float64Array.from(ripleysK(points, radii, options), (k) => Math.sqrt(k / Math.PI))
}

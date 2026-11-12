/**
 * The grid posterior, the most reusable oracle of the validation kernel: the unconstrained density of a program
 * (Jacobians included, so its integral is the evidence) evaluated on a tensor-product grid around the Laplace mode,
 * widened until the edges carry no mass and refined until the evidence stops changing. For models of up to about four
 * latent coordinates it gives the evidence, exact-to-quadrature marginals, moments and CDFs, against which an
 * approximate method is scored. The trapezoid rule on a smooth, rapidly decaying density converges exponentially, so
 * the evidence and the moments are exact to quadrature accuracy with a few dozen points per axis; the marginal CDFs are
 * cumulative trapezoids, accurate to second order in the spacing (about 1e-4 at the default resolution).
 */

import { OmaiError } from '../../base/index.ts'
import { bijectors, type Support } from '../../dists/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import {
  add,
  arange,
  concat,
  cumsum,
  diagonal,
  div,
  full,
  mul,
  slice,
  sqrt,
  stack,
  sub,
  tensor,
  toArray,
  toFlat,
  zeros,
} from '../../tensor/index.ts'
import { findInitialParams, type LatentLayout, potentialEnergy } from '../density.ts'
import { laplaceFit } from '../engines/laplace.ts'
import type { Model } from '../primitives.ts'
import type { MarginalOracle } from './oracle.ts'

/** Options of {@link gridPosterior}. */
export interface GridOptions {
  /** The key of the search for a starting point (default key(0)). */
  readonly key?: Key
  /** Points per axis of the first grid (default 161, 61, 33, 17 for one to four coordinates). */
  readonly points?: number
  /** Half-width of the box in Laplace standard deviations (default 7). */
  readonly width?: number
  /** Relative tolerance of the evidence between refinements (default 1e-7). */
  readonly tol?: number
  /** Refinements by doubling the points per axis (default 2). */
  readonly maxRefinements?: number
  /** An explicit box on the unconstrained scale, one `[low, high]` per coordinate (skips the Laplace fit). */
  readonly box?: readonly (readonly [number, number])[]
}

/** A grid posterior. */
export interface GridPosterior {
  /** The number of unconstrained coordinates. */
  readonly dim: number
  /** The coordinate names (`mu`, `theta[1]`), as the unconstrained model names them. */
  readonly names: readonly string[]
  /** log p(data), the integral of the unconstrained density. */
  readonly logEvidence: number
  /** The change of the log evidence at the last refinement (an error estimate). */
  readonly evidenceError: number
  /** The points per axis and the box of the final grid. */
  readonly points: number
  readonly box: readonly (readonly [number, number])[]
  /** Density evaluations made in all. */
  readonly evaluations: number
  /** The marginal of one coordinate on the constrained scale (sites with an elementwise support only). */
  marginal(name: string): MarginalOracle
  /** Every coordinate's marginal on the constrained scale. */
  marginals(): MarginalOracle[]
  /** The mean and sd of a coordinate on the unconstrained scale. */
  unconstrained(name: string): { readonly mean: number; readonly sd: number }
  /** E[f(u)] over the posterior for a function of the unconstrained vector. */
  expectation(f: (u: Float64Array) => number): number
  /**
   * The marginal density of one coordinate on the constrained scale, at the points of the grid axis (so a curve to draw: `x`
   * increasing, `p` integrating to one over `x`).
   */
  density(name: string): { readonly x: readonly number[]; readonly p: readonly number[] }
  /**
   * The joint density of two coordinates on the constrained scale (the others integrated out), on the product of their grid
   * axes: `p[i][j]` is the density at `(x[i], y[j])`, for contours.
   */
  joint(
    a: string,
    b: string,
  ): { readonly x: readonly number[]; readonly y: readonly number[]; readonly p: readonly (readonly number[])[] }
}

const ELEMENTWISE: ReadonlySet<Support['kind']> = new Set([
  'real',
  'positive',
  'nonNegative',
  'unitInterval',
  'interval',
  'greaterThan',
  'lessThan',
])

const trapezoid = (n: number, h: number): number[] =>
  toFlat(concat([tensor([h / 2]), full([n - 2], h), tensor([h / 2])]))

interface Evaluated {
  readonly logZ: number
  /** The normalised density marginal per axis on the axis points (integrates to one with the trapezoid weights). */
  readonly marginals: number[][]
  readonly edge: readonly (readonly [number, number])[]
  readonly logf: Float64Array
  readonly axes: number[][]
  readonly weights: number[][]
}

function evaluate(
  logDensity: (u: Float64Array) => number,
  box: readonly (readonly [number, number])[],
  n: number,
): Evaluated {
  const d = box.length
  const axes = box.map(([lo, hi]) => toFlat(add(div(mul(arange(n), hi - lo), n - 1), lo)))
  const weights = box.map(([lo, hi]) => trapezoid(n, (hi - lo) / (n - 1)))
  const total = n ** d
  // The density table and the point handed to the callback are the working buffers of the grid sweep (the callback takes a
  // Float64Array and is called points^dim times), written in place.
  const logf = new Float64Array(total)
  const u = new Float64Array(d)
  let max = Number.NEGATIVE_INFINITY
  for (let flat = 0; flat < total; flat++) {
    let r = flat
    for (let k = d - 1; k >= 0; k--) {
      u[k] = axes[k]?.[r % n] ?? 0
      r = Math.floor(r / n)
    }
    const v = logDensity(u)
    logf[flat] = Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
    if ((logf[flat] ?? Number.NEGATIVE_INFINITY) > max) max = logf[flat] ?? max
  }
  if (!Number.isFinite(max))
    throw new OmaiError('value', 'gridPosterior: the density is not finite anywhere on the grid')
  const marg = axes.map(() => toFlat(zeros([n])))
  let z = 0
  for (let flat = 0; flat < total; flat++) {
    let w = 1
    let r = flat
    const idx = new Array<number>(d)
    for (let k = d - 1; k >= 0; k--) {
      idx[k] = r % n
      w *= weights[k]?.[idx[k] ?? 0] ?? 0
      r = Math.floor(r / n)
    }
    const f = Math.exp((logf[flat] ?? Number.NEGATIVE_INFINITY) - max)
    z += f * w
    // The marginal on axis k holds f times the weights of the other axes.
    for (let k = 0; k < d; k++) {
      const m = marg[k] as number[]
      m[idx[k] ?? 0] = (m[idx[k] ?? 0] ?? 0) + (f * w) / (weights[k]?.[idx[k] ?? 0] ?? 1)
    }
  }
  const peak = marg.map((m) => m.reduce((a, b) => Math.max(a, b), 0))
  const edge = marg.map((m, k) => [(m[0] ?? 0) / (peak[k] ?? 1), (m[n - 1] ?? 0) / (peak[k] ?? 1)] as const)
  for (const m of marg) for (let j = 0; j < n; j++) m[j] = (m[j] ?? 0) / z
  return { logZ: max + Math.log(z), marginals: marg, edge, logf, axes, weights }
}

/**
 * The grid posterior of a program: the unconstrained density on a tensor-product grid around the Laplace mode, widened
 * where the edges still carry mass and refined until the evidence converges, with the evidence, the marginals and
 * the expectations of the result. It is the oracle for models of up to about four latent coordinates (the cost is
 * points^dim density evaluations); sites with discrete latent variables are summed out by enumeration as usual.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Starting key, points per axis, box width, tolerance and refinements, or an explicit box.
 * @returns The grid posterior.
 * @throws OmaiError with code `value` for more than four coordinates, or when the Laplace fit fails and no box is given.
 * @concept posterior-validation
 * @index function Grid posterior of a small program: evidence, marginals and moments to quadrature accuracy.
 * @example
 * ```ts
 * import { observe, sample } from 'omai-engine/probprog'
 * import { gridPosterior } from 'omai-engine/probprog/validation'
 * import { normal } from 'omai-engine/dists'
 * // m ~ N(0, 1), y ~ N(m, 1) at y = 1.2: the evidence is the N(0, 2) density at 1.2 and the posterior N(0.6, 0.5).
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }
 * const g = gridPosterior(model, undefined)
 * const exact = -0.5 * Math.log(4 * Math.PI) - 1.2 ** 2 / 4
 * [Math.abs(g.logEvidence - exact) < 1e-9, Math.abs((g.marginal('m').mean) - 0.6) < 1e-9] // => [true, true]
 * ```
 */
export function gridPosterior<D = void>(model: Model<D>, data: D, options: GridOptions = {}): GridPosterior {
  const um = potentialEnergy(model, data)
  const d = um.dim
  if (d > 4) throw new OmaiError('value', `gridPosterior: ${d} coordinates is too many for a grid (at most four)`)
  let n = options.points ?? [161, 61, 33, 17][d - 1] ?? 17
  const tol = options.tol ?? 1e-7
  let box: [number, number][]
  if (options.box !== undefined) {
    if (options.box.length !== d) throw new OmaiError('shape', `gridPosterior: the box needs ${d} intervals`)
    box = options.box.map(([lo, hi]) => [lo, hi])
  } else {
    const start = findInitialParams(um, model, data, { key: options.key ?? makeKey(0) })
    if (!start.valid) throw new OmaiError('value', 'gridPosterior: no valid starting point')
    const fit = laplaceFit(um, start.unconstrained)
    const width = options.width ?? 7
    const spread = mul(sqrt(diagonal(fit.covariance)), width)
    box = toArray(stack([sub(fit.mode, spread), add(fit.mode, spread)], 1)) as [number, number][]
  }
  const f = (u: Float64Array): number => um.logDensity.logDensity(u)
  let evaluations = 0
  const run = (grid: number): Evaluated => {
    evaluations += grid ** d
    return evaluate(f, box, grid)
  }
  // Widen the box on the sides whose marginal still has mass at the edge.
  let e = run(n)
  for (let round = 0; round < 6; round++) {
    let widened = false
    e.edge.forEach(([lo, hi], k) => {
      const range = (box[k]?.[1] ?? 0) - (box[k]?.[0] ?? 0)
      if (lo > 1e-9) {
        ;(box[k] as [number, number])[0] -= 0.5 * range
        widened = true
      }
      if (hi > 1e-9) {
        ;(box[k] as [number, number])[1] += 0.5 * range
        widened = true
      }
    })
    if (!widened) break
    e = run(n)
  }
  // Refine by doubling until the evidence is stable.
  let error = Number.POSITIVE_INFINITY
  for (let level = 0; level < (options.maxRefinements ?? 2); level++) {
    const finer = run(2 * n - 1)
    error = Math.abs(finer.logZ - e.logZ)
    n = 2 * n - 1
    e = finer
    if (error < tol) break
  }
  if (!Number.isFinite(error)) error = Number.NaN
  const names = um.logDensity.names
  const final = e
  const owner = (k: number): LatentLayout | undefined => um.layout.find((l) => k >= l.offset && k < l.offset + l.size)
  const marginalOf = (name: string): MarginalOracle => {
    const k = names.indexOf(name)
    if (k < 0) throw new OmaiError('value', `gridPosterior: no coordinate ${name}`)
    const layout = owner(k)
    if (layout === undefined || !ELEMENTWISE.has(layout.support.kind))
      throw new OmaiError('not-implemented', `gridPosterior: the support of ${name} has no coordinate-wise marginal`)
    const u = final.axes[k] as number[]
    const p = final.marginals[k] as number[]
    const w = final.weights[k] as number[]
    const b = bijectors.bijectorForSupport(layout.support)
    const x = toFlat(b.forward(tensor(u)))
    const increasing = (x[n - 1] ?? 0) >= (x[0] ?? 0)
    // Cumulative trapezoid of the marginal in u, which carries over to x unchanged (a monotone map).
    const [pt, ut] = [tensor(p), tensor(u)]
    const step = mul(
      div(add(slice(pt, [1], [n]), slice(pt, [0], [n - 1])), 2),
      sub(slice(ut, [1], [n]), slice(ut, [0], [n - 1])),
    )
    const running = concat([zeros([1]), cumsum(step)])
    const F = toFlat(div(running, slice(running, [n - 1], [n])))
    let mean = 0
    let second = 0
    for (let j = 0; j < n; j++) {
      const m = (p[j] ?? 0) * (w[j] ?? 0)
      mean += m * (x[j] ?? 0)
      second += m * (x[j] ?? 0) ** 2
    }
    const xs = increasing ? x : [...x].reverse()
    const Fs = increasing ? F : F.map((v) => 1 - v).reverse()
    const locate = (grid: ArrayLike<number>, target: number): number => {
      let lo = 0
      let hi = n - 1
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1
        if ((grid[mid] ?? 0) <= target) lo = mid
        else hi = mid
      }
      return lo
    }
    const lerp = (a: number, b: number, t: number): number => a + t * (b - a)
    return {
      name,
      mean,
      sd: Math.sqrt(Math.max(second - mean * mean, 0)),
      cdf: (v) => {
        if (v <= (xs[0] ?? 0)) return 0
        if (v >= (xs[n - 1] ?? 0)) return 1
        const j = locate(xs, v)
        const t = (v - (xs[j] ?? 0)) / ((xs[j + 1] ?? 0) - (xs[j] ?? 0)) || 0
        return lerp(Fs[j] ?? 0, Fs[j + 1] ?? 0, t)
      },
      quantile: (q) => {
        if (q <= 0) return xs[0] ?? 0
        if (q >= 1) return xs[n - 1] ?? 0
        const j = locate(Fs, q)
        const t = (q - (Fs[j] ?? 0)) / ((Fs[j + 1] ?? 0) - (Fs[j] ?? 0)) || 0
        return lerp(xs[j] ?? 0, xs[j + 1] ?? 0, t)
      },
    }
  }
  // The constrained values of an axis and the derivative of the map there (the Jacobian that turns a density in u into one in x).
  const axisOf = (name: string): { readonly x: number[]; readonly slope: number[]; readonly k: number } => {
    const k = names.indexOf(name)
    if (k < 0) throw new OmaiError('value', `gridPosterior: no coordinate ${name}`)
    const layout = owner(k)
    if (layout === undefined || !ELEMENTWISE.has(layout.support.kind))
      throw new OmaiError('not-implemented', `gridPosterior: the support of ${name} has no coordinate-wise density`)
    const u = final.axes[k] as number[]
    const b = bijectors.bijectorForSupport(layout.support)
    const x = toFlat(b.forward(tensor(u)))
    const slope = x.map((_, j) => {
      const a = Math.max(0, j - 1)
      const c = Math.min(n - 1, j + 1)
      return Math.abs(((x[c] ?? 0) - (x[a] ?? 0)) / ((u[c] ?? 0) - (u[a] ?? 0)))
    })
    return { x, slope, k }
  }
  const densityOf = (name: string): { readonly x: readonly number[]; readonly p: readonly number[] } => {
    const { x, slope, k } = axisOf(name)
    const p = final.marginals[k] as number[]
    const order = x.map((_, j) => j).sort((i, j) => (x[i] ?? 0) - (x[j] ?? 0))
    return {
      x: order.map((j) => x[j] ?? 0),
      p: order.map((j) => (p[j] ?? 0) / Math.max(slope[j] ?? 1, 1e-300)),
    }
  }
  const jointOf = (
    a: string,
    b: string,
  ): { readonly x: readonly number[]; readonly y: readonly number[]; readonly p: readonly (readonly number[])[] } => {
    const A = axisOf(a)
    const B = axisOf(b)
    if (A.k === B.k) throw new OmaiError('value', 'gridPosterior: joint needs two different coordinates')
    const acc = toArray(zeros([n, n])) as number[][]
    const total = n ** d
    const peak = final.logf.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY)
    for (let flat = 0; flat < total; flat++) {
      let r = flat
      let w = 1
      let ia = 0
      let ib = 0
      for (let q = d - 1; q >= 0; q--) {
        const j = r % n
        if (q === A.k) ia = j
        else if (q === B.k) ib = j
        else w *= final.weights[q]?.[j] ?? 0
        r = Math.floor(r / n)
      }
      const row = acc[ia] as number[]
      row[ib] = (row[ib] ?? 0) + Math.exp((final.logf[flat] ?? Number.NEGATIVE_INFINITY) - peak) * w
    }
    const scale = Math.exp(peak - final.logZ)
    const p = acc.map((row, i) => row.map((v, j) => (v * scale) / ((A.slope[i] ?? 1) * (B.slope[j] ?? 1))))
    return { x: A.x, y: B.x, p }
  }
  const unconstrainedOf = (name: string): { readonly mean: number; readonly sd: number } => {
    const k = names.indexOf(name)
    if (k < 0) throw new OmaiError('value', `gridPosterior: no coordinate ${name}`)
    const u = final.axes[k] as number[]
    const p = final.marginals[k] as number[]
    const w = final.weights[k] as number[]
    let mean = 0
    let second = 0
    for (let j = 0; j < n; j++) {
      const m = (p[j] ?? 0) * (w[j] ?? 0)
      mean += m * (u[j] ?? 0)
      second += m * (u[j] ?? 0) ** 2
    }
    return { mean, sd: Math.sqrt(Math.max(second - mean * mean, 0)) }
  }
  return {
    dim: d,
    names,
    logEvidence: final.logZ,
    evidenceError: error,
    points: n,
    box,
    evaluations,
    marginal: marginalOf,
    marginals: () => names.map(marginalOf),
    unconstrained: unconstrainedOf,
    density: densityOf,
    joint: jointOf,
    expectation: (g) => {
      const total = n ** d
      const u = new Float64Array(d)
      let acc = 0
      for (let flat = 0; flat < total; flat++) {
        let r = flat
        let w = 1
        for (let k = d - 1; k >= 0; k--) {
          const j = r % n
          u[k] = final.axes[k]?.[j] ?? 0
          w *= final.weights[k]?.[j] ?? 0
          r = Math.floor(r / n)
        }
        acc += Math.exp((final.logf[flat] ?? Number.NEGATIVE_INFINITY) - final.logZ) * w * g(u)
      }
      return acc
    },
  }
}

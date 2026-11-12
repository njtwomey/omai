/**
 * The references by grid: the marginal of a one-dimensional density known up to a constant (midpoint rule), of a positive
 * hyperparameter on a log-spaced grid with what hangs below it mixed over its posterior, and of a density on a tensor grid in a
 * box (marginals and pairwise joints). Each takes a log density as a callback, so the model it is the reference of is written
 * once, by hand, independent of the rule under test.
 */

import { OmaiError } from '../../../base/index.ts'
import { add, arange, div, exp, mul, toArray, toFlat, zeros } from '../../../tensor/index.ts'
import { type ExactMarginal, gaussianMixtureMarginal } from './marginal.ts'

/**
 * The marginal of a one-dimensional posterior known up to a constant (its log density), by the midpoint rule on `points`
 * midpoints of `[lo, hi]`: the moments, the normalised density and the range where it has mass.
 *
 * @param logDensity - The log of the unnormalised density.
 * @param lo - The left end.
 * @param hi - The right end.
 * @param points - The number of midpoints (default 8000).
 * @returns The marginal.
 * @example
 * ```ts
 * import { quadratureMarginal } from 'omai-engine/probprog/validation'
 * Math.abs(quadratureMarginal((x) => -0.5 * x * x, -10, 10).sd - 1) < 1e-6 // => true
 * ```
 */
export function quadratureMarginal(
  logDensity: (x: number) => number,
  lo: number,
  hi: number,
  points = 8000,
): ExactMarginal {
  const h = (hi - lo) / points
  const xs = toFlat(add(mul(add(arange(points), 0.5), h), lo))
  const logs = xs.map(logDensity)
  const { weights: w, top } = logWeights(logs)
  const z = w.reduce((a, b) => a + b, 0)
  const mean = xs.reduce((a, x, i) => a + (x * (w[i] ?? 0)) / z, 0)
  const sd = Math.sqrt(xs.reduce((a, x, i) => a + ((x - mean) ** 2 * (w[i] ?? 0)) / z, 0))
  const keep = xs.filter((_, i) => (w[i] ?? 0) > 1e-6)
  return {
    mean,
    sd,
    pdf: (x) => (x < lo || x > hi ? 0 : Math.exp(logDensity(x) - top) / (z * h)),
    range: [Math.min(...keep) - 2 * h, Math.max(...keep) + 2 * h],
  }
}

/** The marginals of a density on a tensor grid, and the joint of any two of its coordinates. */
export interface GridMarginals {
  readonly marginals: readonly ExactMarginal[]
  /** The marginal of coordinate `k`. */
  readonly marginal: (k: number) => ExactMarginal
  /** The log of the integral of the density over the box (the sum over the grid times the cell's volume). */
  readonly logEvidence: number
  /** The posterior expectation of a function of the coordinates (the sum over the grid, weighted by the normalised density). */
  readonly expect: (f: (x: readonly number[]) => number) => number
  /** The points of each coordinate's axis and the normalised mass of the marginal at each (a sum of the grid's cells, so a weight, not a density). */
  readonly axes: readonly (readonly number[])[]
  readonly mass: (k: number) => readonly number[]
  /**
   * The marginal of a quantity that is Gaussian given the coordinates (the function values of a Gaussian process given its
   * hyperparameters): the mixture of the conditional normals over the grid, cells with under 1e-6 of the top mass left out.
   */
  readonly mixture: (
    given: (x: readonly number[]) => { readonly mean: number; readonly variance: number },
  ) => ExactMarginal
  /** The joint of coordinates `a` and `b`, on their axes. */
  readonly joint: (
    a: number,
    b: number,
  ) => { readonly x: readonly number[]; readonly y: readonly number[]; readonly p: readonly (readonly number[])[] }
}

/**
 * The marginals and pairwise joints of a density known up to a constant on a box, by summing a tensor grid (`n` points per
 * coordinate): for a model whose program form has too many coordinates for the library's grid (a dense vector latent).
 *
 * @param logDensity - The log of the unnormalised density of the coordinates.
 * @param box - The interval of each coordinate.
 * @param n - Points per coordinate (default 41).
 * @returns The marginals (piecewise-linear densities), the joint, the log of the integral and the expectation of a function.
 * @example
 * ```ts
 * import { gridMarginals } from 'omai-engine/probprog/validation'
 * const g = gridMarginals((x) => -0.5 * (x[0] ?? 0) ** 2 - 0.5 * (x[1] ?? 0) ** 2, [[-6, 6], [-6, 6]], 41)
 * Math.abs((g.marginals[0]?.sd ?? 0) - 1) < 1e-3 // => true
 * ```
 */
export function gridMarginals(
  logDensity: (x: readonly number[]) => number,
  box: readonly (readonly [number, number])[],
  n = 41,
): GridMarginals {
  const d = box.length
  const axes = box.map(([lo, hi]) => toFlat(add(div(mul(arange(n), hi - lo), n - 1), lo)))
  const total = n ** d
  // The density table and the point handed to the callback are the working buffers of the grid sweep (points^dim steps),
  // written in place.
  // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
  const logs = new Float64Array(total)
  const x = toFlat(zeros([d]))
  let top = Number.NEGATIVE_INFINITY
  for (let flat = 0; flat < total; flat++) {
    let r = flat
    for (let k = d - 1; k >= 0; k--) {
      x[k] = axes[k]?.[r % n] ?? 0
      r = Math.floor(r / n)
    }
    const v = logDensity(x)
    logs[flat] = Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
    if ((logs[flat] ?? Number.NEGATIVE_INFINITY) > top) top = logs[flat] ?? top
  }
  const index = (flat: number, k: number): number => Math.floor(flat / n ** (d - 1 - k)) % n
  const h = box.map(([lo, hi]) => (hi - lo) / (n - 1))
  // the normalised mass of every cell, a table of points^dim entries filled in place with the running total
  // banned-ok: reference: an exact answer written out in plain arithmetic to check the library against
  const mass = new Float64Array(total)
  let z = 0
  for (let flat = 0; flat < total; flat++) {
    mass[flat] = Math.exp((logs[flat] ?? Number.NEGATIVE_INFINITY) - top)
    z += mass[flat] ?? 0
  }
  const marginals = axes.map((axis, k): ExactMarginal => {
    const m = toFlat(zeros([n]))
    for (let flat = 0; flat < total; flat++) m[index(flat, k)] = (m[index(flat, k)] ?? 0) + (mass[flat] ?? 0) / z
    const mean = axis.reduce((a, v, i) => a + v * (m[i] ?? 0), 0)
    const sd = Math.sqrt(axis.reduce((a, v, i) => a + (v - mean) ** 2 * (m[i] ?? 0), 0))
    const step = h[k] ?? 1
    return {
      mean,
      sd,
      pdf: (v) => {
        const lo = axis[0] ?? 0
        const t = (v - lo) / step
        if (t < 0 || t > n - 1) return 0
        const i = Math.min(n - 2, Math.floor(t))
        return ((m[i] ?? 0) * (1 - (t - i)) + (m[i + 1] ?? 0) * (t - i)) / step
      },
      range: [axis[0] ?? 0, axis[n - 1] ?? 0],
    }
  })
  const cell = h.reduce((a, v) => a * v, 1)
  return {
    marginals,
    axes,
    mass: (k) => {
      const m = toFlat(zeros([n]))
      for (let flat = 0; flat < total; flat++) m[index(flat, k)] = (m[index(flat, k)] ?? 0) + (mass[flat] ?? 0) / z
      return m
    },
    mixture: (given) => {
      const point = toFlat(zeros([d]))
      const parts: { weight: number; mean: number; variance: number }[] = []
      for (let flat = 0; flat < total; flat++) {
        if (!((mass[flat] ?? 0) > 1e-6)) continue
        for (let k = 0; k < d; k++) point[k] = axes[k]?.[index(flat, k)] ?? 0
        parts.push({ weight: (mass[flat] ?? 0) / z, ...given(point) })
      }
      const kept = parts.reduce((a, p) => a + p.weight, 0)
      return gaussianMixtureMarginal(
        parts.map((p) => ({ ...p, weight: p.weight / kept })),
        (m, sd) => [m - 6 * sd, m + 6 * sd],
      )
    },
    logEvidence: top + Math.log(z * cell),
    expect: (f) => {
      const point = toFlat(zeros([d]))
      let sum = 0
      for (let flat = 0; flat < total; flat++) {
        if (!((mass[flat] ?? 0) > 0)) continue
        for (let k = 0; k < d; k++) point[k] = axes[k]?.[index(flat, k)] ?? 0
        sum += ((mass[flat] ?? 0) / z) * f(point)
      }
      return sum
    },
    marginal: (k) => {
      const m = marginals[k]
      if (m === undefined) throw new OmaiError('value', `gridMarginals: there is no coordinate ${k}`)
      return m
    },
    joint: (a, b) => {
      const p = toArray(zeros([n, n])) as number[][]
      for (let flat = 0; flat < total; flat++) {
        const row = p[index(flat, a)] as number[]
        row[index(flat, b)] = (row[index(flat, b)] ?? 0) + (mass[flat] ?? 0) / z
      }
      const ha = h[a] ?? 1
      const hb = h[b] ?? 1
      return { x: axes[a] ?? [], y: axes[b] ?? [], p: p.map((row) => row.map((v) => v / (ha * hb))) }
    },
  }
}

/** What a hyperparameter's value makes of one quantity: its moments and density given that value. */
export interface ConditionalMarginal {
  readonly mean: number
  readonly variance: number
  readonly pdf: (x: number) => number
}

/** The posterior of a positive hyperparameter on a log-spaced grid, and the marginals of what hangs below it. */
export interface HyperPosterior {
  /** The hyperparameter's own marginal. */
  readonly hyper: ExactMarginal
  /** The marginal of a quantity that is, given the hyperparameter, a known family: the mixture over the posterior of it. */
  readonly mix: (given: (h: number) => ConditionalMarginal) => ExactMarginal
}

/**
 * The posterior of a positive hyperparameter by quadrature on a log-spaced grid, and a way to mix what depends on it: in a
 * hierarchy whose members are independent given the hyperparameter (a population of probabilities with a shared strength, of
 * rates with a shared prior rate), each member's exact marginal is the mixture of its conditional distributions over this
 * posterior.
 *
 * @param logDensity - The log of the hyperparameter's unnormalised marginal posterior density (with respect to its own value).
 * @param lo - The smallest value on the grid (positive).
 * @param hi - The largest.
 * @param points - Grid points (default 1500).
 * @returns The hyperparameter's marginal, and the mixing function.
 * @example
 * ```ts
 * import { hyperPosterior } from 'omai-engine/probprog/validation'
 * const post = hyperPosterior((h) => 2 * Math.log(h) - 3 * h, 1e-3, 40)
 * Math.abs(post.hyper.mean - 3 / 3) < 1e-3 // => true
 * ```
 */
export function hyperPosterior(
  logDensity: (h: number) => number,
  lo: number,
  hi: number,
  points = 1500,
): HyperPosterior {
  const step = (Math.log(hi) - Math.log(lo)) / points
  const hs = toFlat(exp(add(mul(add(arange(points), 0.5), step), Math.log(lo))))
  const logs = hs.map((h) => logDensity(h) + Math.log(h))
  const { weights: w0 } = logWeights(logs)
  const z = w0.reduce((a, b) => a + b, 0)
  const w = w0.map((v) => v / z)
  const mean = hs.reduce((a, h, i) => a + h * (w[i] ?? 0), 0)
  const sd = Math.sqrt(hs.reduce((a, h, i) => a + (h - mean) ** 2 * (w[i] ?? 0), 0))
  const keep = w.map((_, i) => i).filter((i) => (w[i] ?? 0) > 1e-9 * Math.max(...w))
  const kept = keep.map((i) => hs[i] ?? 0)
  const hyper: ExactMarginal = {
    mean,
    sd,
    pdf: (x) => {
      if (x <= 0) return 0
      const i = Math.round((Math.log(x) - Math.log(lo)) / step - 0.5)
      return i < 0 || i >= points ? 0 : (w[i] ?? 0) / step / x
    },
    range: [Math.max(0, Math.min(...kept) * 0.8), Math.max(...kept) * 1.2],
  }
  return {
    hyper,
    mix: (given) => {
      const parts = keep.map((i) => ({ weight: w[i] ?? 0, ...given(hs[i] ?? 1) }))
      const total = parts.reduce((a, p) => a + p.weight, 0)
      return gaussianMixtureMarginal(
        parts.map((p) => ({ ...p, weight: p.weight / total })),
        (m, sdm) => [m - 6 * sdm, m + 8 * sdm],
      )
    },
  }
}

/**
 * The weights of points from the log of their unnormalised density, as the exponential of the log minus its largest (so the
 * largest weight is one and nothing overflows): the first step of every reference on a grid.
 *
 * @param logs - The log density at each point.
 * @returns The weights, and the largest log that was taken off.
 * @example
 * ```ts
 * import { logWeights } from 'omai-engine/probprog/validation'
 * logWeights([0, Math.log(2)]).weights // => [0.5, 1]
 * ```
 */
export function logWeights(logs: readonly number[]): { readonly weights: number[]; readonly top: number } {
  const top = Math.max(...logs)
  return { weights: logs.map((l) => Math.exp(l - top)), top }
}

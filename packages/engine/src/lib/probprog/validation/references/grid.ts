/**
 * The references by grid: the marginal of a one-dimensional density known up to a constant (midpoint rule), of a positive
 * hyperparameter on a log-spaced grid with what hangs below it mixed over its posterior, and of a density on a tensor grid in a
 * box (marginals and pairwise joints). Each takes a log density as a callback, so the model it is the reference of is written
 * once, by hand, independent of the rule under test.
 */

import { OmaiError } from '../../../base/index.ts'
import type { ExactMarginal } from './marginal.ts'

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
  const xs = Array.from({ length: points }, (_, i) => lo + (i + 0.5) * h)
  const logs = xs.map(logDensity)
  const top = Math.max(...logs)
  const w = logs.map((l) => Math.exp(l - top))
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
 * @returns The marginals (piecewise-linear densities) and the joint.
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
  const axes = box.map(([lo, hi]) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1)))
  const total = n ** d
  const logs = new Float64Array(total)
  const x = new Array<number>(d).fill(0)
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
  const mass = new Float64Array(total)
  let z = 0
  for (let flat = 0; flat < total; flat++) {
    mass[flat] = Math.exp((logs[flat] ?? Number.NEGATIVE_INFINITY) - top)
    z += mass[flat] ?? 0
  }
  const marginals = axes.map((axis, k): ExactMarginal => {
    const m = new Float64Array(n)
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
  return {
    marginals,
    marginal: (k) => {
      const m = marginals[k]
      if (m === undefined) throw new OmaiError('value', `gridMarginals: there is no coordinate ${k}`)
      return m
    },
    joint: (a, b) => {
      const p = Array.from({ length: n }, () => new Array<number>(n).fill(0))
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
  const hs = Array.from({ length: points }, (_, i) => Math.exp(Math.log(lo) + (i + 0.5) * step))
  const logs = hs.map((h) => logDensity(h) + Math.log(h))
  const top = Math.max(...logs)
  const w0 = logs.map((l) => Math.exp(l - top))
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
      const m1 = parts.reduce((a, p) => a + (p.weight / total) * p.mean, 0)
      const m2 = parts.reduce((a, p) => a + (p.weight / total) * (p.variance + p.mean ** 2), 0)
      const sdm = Math.sqrt(Math.max(m2 - m1 * m1, 0))
      return {
        mean: m1,
        sd: sdm,
        pdf: (x) => parts.reduce((a, p) => a + (p.weight / total) * p.pdf(x), 0),
        range: [m1 - 6 * sdm, m1 + 8 * sdm],
      }
    },
  }
}

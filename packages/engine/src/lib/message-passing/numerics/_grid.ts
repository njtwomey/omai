// A two-dimensional integral of a log density against the cavities of two variables, by a panelled Gauss-Legendre grid that
// is zoomed onto where the mass is: what the two-variable numeric sites (a likelihood over a predictor and a second random
// scalar, a beta with two random shapes) integrate with. A variable is a Gaussian (its coordinate is the value) or a gamma
// (its coordinate is the log of the value, the density in that coordinate carrying the Jacobian), and the integrand is a
// function of the two coordinates.

import { OmaiError } from '../../base/index.ts'
import type { Arg } from '../core/types.ts'
import { legendreRule } from './_quadrature.ts'
import { digammaOf, lgammaOf, trigammaOf } from './_scalar.ts'

/** One coordinate of the integral: where the cavity lives, its normalised log density, and the kinks the integrand has. */
export interface GridAxis {
  /** True for an observed value: a single node of weight one. */
  readonly point: boolean
  /** The window the cavity lives in, and the centre and spread the first pass starts from. */
  readonly lo: number
  readonly hi: number
  /** The log of the normalised density of the cavity in the coordinate. */
  readonly logWeight: (x: number) => number
  /** Coordinates where the integrand is not smooth: panels never straddle them. */
  readonly kinks: readonly number[]
}

/** The result of an integral: the log mass, and per coordinate the mean, the variance and the mean of `exp(x)`. */
export interface GridResult {
  readonly logMass: number
  readonly mean: readonly [number, number]
  readonly variance: readonly [number, number]
  readonly meanExp: readonly [number, number]
  /** The final grid, for a caller that needs another expectation under the same weights: the nodes of each axis and `weights[i * n1 + j]`, summing to one. */
  readonly nodes: readonly [Float64Array, Float64Array]
  readonly weights: Float64Array
}

/** The offsets 0, 2^-(n-1), ..., 1/4, 1/2 of a stretch of n panels, each half the one after it (the last boundary, 1, is the stretch's end). */
const graded = (n: number): number[] => [0, ...Array.from({ length: n - 1 }, (_, i) => 2 ** -(n - 1 - i))]

/** The panel boundaries of [a, b], `count` panels: uniform, graded geometrically towards an end that is a kink. */
function boundaries(a: number, b: number, count: number, kinkLeft: boolean, kinkRight: boolean): number[] {
  const len = b - a
  if (kinkLeft && kinkRight) {
    const n = Math.max(2, Math.ceil(count / 2))
    const left = graded(n).map((v) => a + (len / 2) * v)
    const right = graded(n)
      .map((v) => b - (len / 2) * v)
      .reverse()
    return [...left, ...right]
  }
  if (kinkLeft) return [...graded(count).map((v) => a + len * v), b]
  if (kinkRight)
    return [
      a,
      ...graded(count)
        .map((v) => b - len * v)
        .reverse(),
    ].slice(0, count + 1)
  return Array.from({ length: count + 1 }, (_, i) => a + (len * i) / count)
}

/** The nodes and weights of a panelled rule on [lo, hi], with the panels split at the kinks that fall inside and graded towards them. */
function panels(
  lo: number,
  hi: number,
  kinks: readonly number[],
  count: number,
  order: number,
): { readonly x: number[]; readonly w: number[] } {
  const cuts = [lo, ...kinks.filter((k) => k > lo && k < hi).sort((p, q) => p - q), hi]
  const rule = legendreRule(order)
  const x: number[] = []
  const w: number[] = []
  for (let s = 0; s + 1 < cuts.length; s++) {
    const a = cuts[s] as number
    const b = cuts[s + 1] as number
    const n = Math.max(6, Math.round((count * (b - a)) / (hi - lo)))
    const edges = boundaries(a, b, n, s > 0, s + 2 < cuts.length)
    for (let p = 0; p + 1 < edges.length; p++) {
      const l = edges[p] as number
      const h = (edges[p + 1] as number) - l
      for (let i = 0; i < order; i++) {
        x.push(l + ((rule.x[i] as number) + 1) * 0.5 * h)
        w.push((rule.w[i] as number) * 0.5 * h)
      }
    }
  }
  return { x, w }
}

/** The grid of the passes that only find the window (coarse) and of the last pass, which gives the answer (fine): panels per axis and nodes per panel. */
const COARSE = { panels: 8, order: 5 }
const FINE = { panels: 10, order: 6 }
const PASSES = 6

/**
 * The integral of `exp(logf(x0, x1))` against the two cavities, found on a panelled grid over each axis, then again on a window
 * around where the mass was (up to four passes, while the window shrinks), so a cavity much wider than the integrand, as in the
 * first sweeps of a run, is integrated as well as a narrow one.
 *
 * @param axes - The two coordinates.
 * @param column - The log of the integrand as a function of the second coordinate that returns a function of the first, so what
 * depends on the second coordinate alone (a gamma function of a shape) is computed once per node of it, not once per point.
 * @returns The log mass and the moments of each coordinate under the normalised integrand.
 * @throws OmaiError with code `value` when the integrand has no mass under the cavities.
 */
export function integratePair(
  axes: readonly [GridAxis, GridAxis],
  column: (x1: number) => (x0: number) => number,
): GridResult {
  let window: [number, number][] = axes.map((a) => [a.lo, a.hi])
  let result: GridResult | undefined
  let fine = false
  for (let pass = 0; pass < PASSES; pass++) {
    const grid = fine || pass === PASSES - 1 ? FINE : COARSE
    const grids = axes.map((a, k) => {
      if (a.point) return { x: [a.lo], w: [1], logw: [0] }
      const [lo, hi] = window[k] as [number, number]
      const g = panels(lo, hi, a.kinks, grid.panels, grid.order)
      return { x: g.x, w: g.w, logw: g.x.map((xi, i) => Math.log(g.w[i] as number) + a.logWeight(xi)) }
    }) as [{ x: number[]; w: number[]; logw: number[] }, { x: number[]; w: number[]; logw: number[] }]
    const n0 = grids[0].x.length
    const n1 = grids[1].x.length
    const lw = new Float64Array(n0 * n1)
    let top = Number.NEGATIVE_INFINITY
    for (let j = 0; j < n1; j++) {
      const f = column(grids[1].x[j] as number)
      const wj = grids[1].logw[j] as number
      for (let i = 0; i < n0; i++) {
        const v = (grids[0].logw[i] as number) + wj + f(grids[0].x[i] as number)
        lw[i * n1 + j] = v
        if (v > top) top = v
      }
    }
    if (!Number.isFinite(top))
      throw new OmaiError('value', 'numeric rule: the factor has no mass under the incoming messages')
    let total = 0
    for (let k = 0; k < lw.length; k++) {
      const v = Math.exp((lw[k] as number) - top)
      lw[k] = v
      total += v
    }
    // The marginal weights of the nodes of each axis give every moment without visiting the points again.
    const p0 = new Float64Array(n0)
    const p1 = new Float64Array(n1)
    for (let i = 0; i < n0; i++)
      for (let j = 0; j < n1; j++) {
        const w = (lw[i * n1 + j] as number) / total
        lw[i * n1 + j] = w
        p0[i] = (p0[i] as number) + w
        p1[j] = (p1[j] as number) + w
      }
    const margins = [p0, p1] as const
    const m = [0, 0]
    const m2 = [0, 0]
    const me = [0, 0]
    for (let k = 0; k < 2; k++) {
      const xs = (grids[k] as { x: number[] }).x
      const pk = margins[k] as Float64Array
      for (let i = 0; i < xs.length; i++) {
        const x = xs[i] as number
        const w = pk[i] as number
        m[k] = (m[k] as number) + w * x
        m2[k] = (m2[k] as number) + w * x * x
        me[k] = (me[k] as number) + w * Math.exp(x)
      }
    }
    const variance = [
      Math.max((m2[0] as number) - (m[0] as number) ** 2, 0),
      Math.max((m2[1] as number) - (m[1] as number) ** 2, 0),
    ]
    result = {
      logMass: top + Math.log(total),
      mean: [m[0] as number, m[1] as number],
      variance: [variance[0] as number, variance[1] as number],
      meanExp: [me[0] as number, me[1] as number],
      nodes: [Float64Array.from(grids[0].x), Float64Array.from(grids[1].x)],
      weights: lw,
    }
    // Zoom onto the mass: the range of nodes where the marginal weight is not negligible, a node wider on each side, inside the cavity's window.
    let shrunk = false
    const next = axes.map((a, k): [number, number] => {
      const [lo, hi] = window[k] as [number, number]
      if (a.point) return [lo, hi]
      const g = (grids[k] as { x: number[] }).x
      const marginal = margins[k] as Float64Array
      const top = marginal.reduce((p, q) => Math.max(p, q), 0)
      let first = 0
      while (first < g.length && (marginal[first] as number) < 1e-9 * top) first++
      let last = g.length - 1
      while (last > first && (marginal[last] as number) < 1e-9 * top) last--
      const spacing = (hi - lo) / g.length
      const nlo = Math.max(a.lo, (g[Math.max(first - 1, 0)] as number) - spacing)
      const nhi = Math.min(a.hi, (g[Math.min(last + 1, g.length - 1)] as number) + spacing)
      if (!(nhi > nlo)) return [lo, hi]
      if (nhi - nlo < 0.7 * (hi - lo)) shrunk = true
      return [nlo, nhi]
    })
    if (!shrunk) {
      if (fine || grid === FINE) break
      // The window has settled: one fine pass over it is the answer.
      fine = true
      continue
    }
    window = next
  }
  return result as GridResult
}

/** The axis of a Gaussian cavity (an observed value is a point). */
export function gaussianAxis(mean: number, variance: number, kinks: readonly number[] = [], span = 9): GridAxis {
  const sd = Math.sqrt(variance)
  return {
    point: false,
    lo: mean - span * sd,
    hi: mean + span * sd,
    kinks,
    logWeight: (x) => -0.5 * (Math.log(2 * Math.PI * variance) + (x - mean) ** 2 / variance),
  }
}

/** The axis of an observed coordinate. */
export function pointAxis(x: number): GridAxis {
  return { point: true, lo: x, hi: x, kinks: [], logWeight: () => 0 }
}

/**
 * The axis of a gamma cavity `Gamma(shape, rate)` in the coordinate `t = log value`: the density of `t` is
 * `value^shape exp(-rate value)` over the gamma normaliser, the Jacobian included.
 */
export function gammaAxis(shape: number, rate: number, kinks: readonly number[] = []): GridAxis {
  const centre = digammaOf(shape) - Math.log(rate)
  const sd = Math.sqrt(trigammaOf(shape))
  const logNorm = lgammaOf(shape) - shape * Math.log(rate)
  return {
    point: false,
    lo: Math.max(centre - 12 * sd, -45),
    hi: Math.min(centre + 8 * sd, 45),
    kinks,
    logWeight: (t) => shape * t - rate * Math.exp(t) - logNorm,
  }
}

/** The axis of an argument of a family `gaussian` (coordinate the value) or `gamma` (coordinate its log), observed or a message. */
export function axisOf(a: Arg, kinks: readonly number[] = []): GridAxis {
  if (a.value !== undefined) {
    const v = typeof a.value === 'number' ? a.value : Number.NaN
    return pointAxis(a.family.kind === 'gamma' ? Math.log(v) : v)
  }
  const eta = a.eta as readonly number[]
  if (a.family.kind === 'gamma') {
    const shape = (eta[0] ?? 0) + 1
    const rate = -(eta[1] ?? 0)
    if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
      throw new OmaiError('value', 'a gamma message has no density')
    return gammaAxis(shape, rate, kinks)
  }
  const j = eta[1] ?? 0
  if (!(j > 0) || !Number.isFinite(j)) throw new OmaiError('value', 'a Gaussian message has no density')
  return gaussianAxis((eta[0] ?? 0) / j, 1 / j, kinks)
}

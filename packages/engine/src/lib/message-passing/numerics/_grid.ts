// The integral of a log density against the cavities of up to three variables, by a panelled Gauss-Legendre grid that is zoomed
// onto where the mass is: what the numeric sites over two or three random scalars (a likelihood over a predictor and a second
// random scalar, a beta with two random shapes, a bivariate Gaussian with two gamma precisions and a beta correlation) integrate
// with. A coordinate is the value of a Gaussian, the log of a gamma variable or the logit of a beta variable, each with the
// Jacobian in its density; an observed coordinate is a single node. One algorithm (panel each axis, add the log weights, shift
// by the maximum, normalise, find the marginals, shrink the window onto them, again) with the choices that differ between its
// callers as the named `gridProfiles`: how the panels are laid and graded towards a kink, the panels and nodes of the coarse and
// the fine pass, and the order the weights are stored in (the order the sums run in, so a result is the same number as it was).

import { ImproperMessageError, ParameterError } from '../core/errors.ts'
import type { Arg } from '../core/types.ts'
import { centredMoments, shiftedSum } from './_moments.ts'
import { legendreRule } from './_quadrature.ts'
import { lgammaOf, softplusOf } from './_scalar.ts'
import { betaLogitWindow, gammaLogWindow, windowProfiles } from './_windows.ts'

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

/** The panels and nodes per panel of one pass. */
export interface PassShape {
  readonly panels: number
  readonly order: number
}

/**
 * How a grid is laid and searched.
 *
 * - `coarse` and `fine`: the passes that only find the window, and the last, which gives the answer.
 * - `passes`: the most passes.
 * - `panelling`: `graded` splits the window at the kinks and grades the panels geometrically towards a kink; `uniform` has equal
 *   panels and no kinks.
 * - `layout`: where the weight of the node `(i0, i1, i2)` is stored. `first-fastest` is `i0 + n0 (i1 + n1 i2)`; `first-slowest` (two
 *   axes) is `i0 n1 + i1`. The sums of a pass run in storage order, so the layout decides the last digits of a result.
 */
export interface GridProfile {
  readonly coarse: PassShape
  readonly fine: PassShape
  readonly passes: number
  readonly panelling: 'graded' | 'uniform'
  readonly layout: 'first-fastest' | 'first-slowest'
}

/**
 * The profiles of the library's grids: `pair` (the likelihood sites of two scalars, with kinks), `axes` (the covariance sites of up
 * to three) and `line` (one smooth coordinate whose integrand is expensive and whose window starts far wider than the mass: the
 * auxiliary variable of the softmax site).
 */
export const gridProfiles: { readonly pair: GridProfile; readonly axes: GridProfile; readonly line: GridProfile } = {
  pair: {
    coarse: { panels: 8, order: 5 },
    fine: { panels: 10, order: 6 },
    passes: 6,
    panelling: 'graded',
    layout: 'first-slowest',
  },
  axes: {
    coarse: { panels: 4, order: 4 },
    fine: { panels: 6, order: 4 },
    passes: 6,
    panelling: 'uniform',
    layout: 'first-fastest',
  },
  line: {
    coarse: { panels: 12, order: 5 },
    fine: { panels: 12, order: 8 },
    passes: 6,
    panelling: 'uniform',
    layout: 'first-fastest',
  },
}

/** The result of the integral: the log mass, and over the final grid the nodes of each axis, the normalised weights and the marginal weights per axis. */
export interface GridNResult {
  readonly logMass: number
  /** The coordinates of the nodes of each axis. */
  readonly nodes: readonly Float64Array[]
  /** The normalised weight of the node `(i0, i1, i2)`, stored as the profile's layout says (missing axes have one node). */
  readonly weights: Float64Array
  /** The marginal weights of the nodes of each axis. */
  readonly marginals: readonly Float64Array[]
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
function gradedPanels(
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

/** The nodes and weights of equal panels on [lo, hi] (no kinks). */
function uniformPanels(lo: number, hi: number, shape: PassShape): { readonly x: number[]; readonly w: number[] } {
  const r = legendreRule(shape.order)
  const x: number[] = []
  const w: number[] = []
  const h = (hi - lo) / shape.panels
  for (let p = 0; p < shape.panels; p++)
    for (let i = 0; i < shape.order; i++) {
      x.push(lo + p * h + ((r.x[i] as number) + 1) * 0.5 * h)
      w.push((r.w[i] as number) * 0.5 * h)
    }
  return { x, w }
}

/**
 * The integrand of a grid: for the coordinates of the axes after the first, the log of the integrand as a function of the first
 * coordinate. The split lets what depends on the later coordinates alone (a gamma function of a shape) be computed once per node
 * of them, not once per point.
 */
export type GridIntegrand = (rest: readonly number[]) => (x0: number) => number

/**
 * The integral of `exp(logf(t))` against the cavities of one to three axes: a first grid over each axis's window finds where the
 * mass lives (each window is shrunk to the nodes where the marginal weight is not negligible, until it settles), and a fine grid
 * over the settled window gives the answer.
 *
 * @param axes - The coordinates (one to three).
 * @param integrand - The log of the integrand (the cavities' densities are separate).
 * @param profile - How the grid is laid (`gridProfiles`).
 * @returns The log mass and the weights of the final grid.
 * @throws ParameterError for no axis or more than three.
 * @throws ImproperMessageError when the integrand has no mass under the cavities.
 */
function integrateGrid(
  axes: readonly GridAxis[],
  integrand: GridIntegrand,
  profile: GridProfile = gridProfiles.axes,
): GridNResult {
  const k = axes.length
  if (k < 1 || k > 3) throw new ParameterError('integrateAxes: one to three axes')
  let window: [number, number][] = axes.map((a) => [a.lo, a.hi])
  let result: GridNResult | undefined
  let fine = false
  for (let pass = 0; pass < profile.passes; pass++) {
    const shape = fine || pass === profile.passes - 1 ? profile.fine : profile.coarse
    const grids = axes.map((a, i) => {
      if (a.point) return { x: [a.lo], logw: [0] }
      const [lo, hi] = window[i] as [number, number]
      const g =
        profile.panelling === 'graded'
          ? gradedPanels(lo, hi, a.kinks, shape.panels, shape.order)
          : uniformPanels(lo, hi, shape)
      return { x: g.x, logw: g.x.map((xi, q) => Math.log(g.w[q] as number) + a.logWeight(xi)) }
    })
    const sizes = grids.map((g) => g.x.length)
    const n0 = sizes[0] as number
    const n1 = sizes[1] ?? 1
    const n2 = sizes[2] ?? 1
    const first = (grids[0] as { x: number[]; logw: number[] }).x
    const logw0 = (grids[0] as { logw: number[] }).logw
    const slowest = profile.layout === 'first-slowest'
    const at = (a: number, b: number, c: number): number => (slowest ? a * n1 + b : a + n0 * (b + n1 * c))
    const logw = new Float64Array(n0 * n1 * n2)
    const rest = new Array<number>(k - 1)
    for (let c = 0; c < n2; c++)
      for (let b = 0; b < n1; b++) {
        if (k > 1) rest[0] = (grids[1] as { x: number[] }).x[b] as number
        if (k > 2) rest[1] = (grids[2] as { x: number[] }).x[c] as number
        const f = integrand(rest)
        const lw1 = k > 1 ? ((grids[1] as { logw: number[] }).logw[b] as number) : 0
        const lw2 = k > 2 ? ((grids[2] as { logw: number[] }).logw[c] as number) : 0
        for (let a = 0; a < n0; a++) logw[at(a, b, c)] = (logw0[a] as number) + lw1 + lw2 + f(first[a] as number)
      }
    // The weights in storage order, shifted by the largest log weight.
    const { top, weights: lw, total, logTotal } = shiftedSum(logw)
    if (!Number.isFinite(top))
      throw new ImproperMessageError('numeric rule: the factor has no mass under the incoming messages')
    // Normalise the weights, and add each to the marginal of every axis it lies on.
    const marginals = sizes.map((n) => new Float64Array(n))
    const m0 = marginals[0] as Float64Array
    const m1 = marginals[1]
    const m2 = marginals[2]
    const visit = (a: number, b: number, c: number): void => {
      const i = at(a, b, c)
      const w = (lw[i] as number) / total
      lw[i] = w
      m0[a] = (m0[a] as number) + w
      if (m1 !== undefined) m1[b] = (m1[b] as number) + w
      if (m2 !== undefined) m2[c] = (m2[c] as number) + w
    }
    if (slowest) {
      for (let a = 0; a < n0; a++) for (let b = 0; b < n1; b++) visit(a, b, 0)
    } else {
      for (let c = 0; c < n2; c++) for (let b = 0; b < n1; b++) for (let a = 0; a < n0; a++) visit(a, b, c)
    }
    result = { logMass: logTotal, nodes: grids.map((g) => Float64Array.from(g.x)), weights: lw, marginals }
    // Zoom onto the mass: the range of nodes where the marginal weight is not negligible, a node wider on each side, inside the cavity's window.
    let shrunk = false
    const next = axes.map((a, i): [number, number] => {
      const [lo, hi] = window[i] as [number, number]
      if (a.point) return [lo, hi]
      const x = (grids[i] as { x: number[] }).x
      const m = marginals[i] as Float64Array
      const peak = m.reduce((p, q) => Math.max(p, q), 0)
      let from = 0
      while (from < x.length && (m[from] as number) < ZOOM_FLOOR * peak) from++
      let to = x.length - 1
      while (to > from && (m[to] as number) < ZOOM_FLOOR * peak) to--
      const spacing = (hi - lo) / x.length
      const nlo = Math.max(a.lo, (x[Math.max(from - 1, 0)] as number) - spacing)
      const nhi = Math.min(a.hi, (x[Math.min(to + 1, x.length - 1)] as number) + spacing)
      if (!(nhi > nlo)) return [lo, hi]
      if (nhi - nlo < ZOOM_SHRINK * (hi - lo)) shrunk = true
      return [nlo, nhi]
    })
    if (!shrunk) {
      if (fine || shape === profile.fine) break
      // The window has settled: one fine pass over it is the answer.
      fine = true
      continue
    }
    window = next
  }
  return result as GridNResult
}

/** A node whose marginal weight is below this fraction of the largest is outside the mass: the zoom cuts there. */
const ZOOM_FLOOR = 1e-9
/** A window that shrinks to less than this fraction of itself is still moving: another coarse pass. */
const ZOOM_SHRINK = 0.7

/**
 * The integral over one to three axes in the profile `axes` (equal panels, no kinks): the covariance sites' grid.
 *
 * @param axes - The coordinates (one to three).
 * @param logf - The log of the integrand at the coordinates `t[0], t[1], ...` (the cavities' densities are separate).
 * @returns The log mass and the weights of the final grid.
 * @throws ParameterError for no axis or more than three.
 * @throws ImproperMessageError when the integrand has no mass under the cavities.
 */
export function integrateAxes(axes: readonly GridAxis[], logf: (t: Float64Array) => number): GridNResult {
  const t = new Float64Array(axes.length)
  return integrateGrid(
    axes,
    (rest) => {
      for (let i = 0; i < rest.length; i++) t[i + 1] = rest[i] as number
      return (x0) => {
        t[0] = x0
        return logf(t)
      }
    },
    gridProfiles.axes,
  )
}

/**
 * The integral of `exp(logf(x))` over one coordinate in the profile `line` (equal panels, a coarse search for the window that is
 * shrunk onto the mass, then eight nodes a panel): the grid of an integral whose integrand costs a sum over many classes.
 *
 * @param axis - The coordinate (its `logWeight` is part of the integrand; give a flat one when `logf` holds everything).
 * @param logf - The log of the integrand at the coordinate (the axis's density is separate).
 * @returns The log mass, and the nodes and normalised weights of the final grid (`weights[i]` belongs to `nodes[0][i]`).
 * @throws ImproperMessageError when the integrand has no mass in the window.
 */
export function integrateLine(axis: GridAxis, logf: (x: number) => number): GridNResult {
  return integrateGrid([axis], () => logf, gridProfiles.line)
}

/** The result of a two-coordinate integral: the log mass, and per coordinate the mean, the variance and the mean of `exp(x)`. */
export interface GridResult {
  readonly logMass: number
  readonly mean: readonly [number, number]
  readonly variance: readonly [number, number]
  readonly meanExp: readonly [number, number]
  /** The final grid, for a caller that needs another expectation under the same weights: the nodes of each axis and `weights[i * n1 + j]`, summing to one. */
  readonly nodes: readonly [Float64Array, Float64Array]
  readonly weights: Float64Array
}

/**
 * The integral of `exp(logf(x0, x1))` against the two cavities in the profile `pair` (graded panels with kinks, the window
 * zoomed up to four times while it shrinks), with the moments of each coordinate: so a cavity much wider than the integrand, as in
 * the first sweeps of a run, is integrated as well as a narrow one.
 *
 * @param axes - The two coordinates.
 * @param column - The log of the integrand as a function of the second coordinate that returns a function of the first, so what
 * depends on the second coordinate alone (a gamma function of a shape) is computed once per node of it, not once per point.
 * @returns The log mass and the moments of each coordinate under the normalised integrand.
 * @throws ImproperMessageError when the integrand has no mass under the cavities.
 */
export function integratePair(
  axes: readonly [GridAxis, GridAxis],
  column: (x1: number) => (x0: number) => number,
): GridResult {
  const r = integrateGrid(axes, (rest) => column(rest[0] as number), gridProfiles.pair)
  const stats = [0, 1].map((k) => {
    const xs = r.nodes[k] as Float64Array
    const pk = r.marginals[k] as Float64Array
    let meanExp = 0
    for (let i = 0; i < xs.length; i++) meanExp += (pk[i] as number) * Math.exp(xs[i] as number)
    // The marginals are normalised: the mean is their weighted sum, the variance the weighted squares of the deviations from it.
    return { ...centredMoments(pk, 1, xs), meanExp }
  }) as [{ mean: number; variance: number; meanExp: number }, { mean: number; variance: number; meanExp: number }]
  return {
    logMass: r.logMass,
    mean: [stats[0].mean, stats[1].mean],
    variance: [stats[0].variance, stats[1].variance],
    meanExp: [stats[0].meanExp, stats[1].meanExp],
    nodes: [r.nodes[0] as Float64Array, r.nodes[1] as Float64Array],
    weights: r.weights,
  }
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
  const { lo, hi } = gammaLogWindow(shape, rate, windowProfiles.gridGamma)
  const logNorm = lgammaOf(shape) - shape * Math.log(rate)
  return { point: false, lo, hi, kinks, logWeight: (t) => shape * t - rate * Math.exp(t) - logNorm }
}

/** The axis of a beta cavity `Beta(a, b)` in the coordinate `s = logit(u)`: the density of `s` is `u^a (1 - u)^b` over the beta function. */
export function betaAxis(a: number, b: number): GridAxis {
  const { lo, hi } = betaLogitWindow(a, b, windowProfiles.gridBeta)
  const logNorm = lgammaOf(a) + lgammaOf(b) - lgammaOf(a + b)
  return {
    point: false,
    lo,
    hi,
    kinks: [],
    // u = 1 / (1 + e^-s): log u = -softplus(-s), log (1 - u) = -softplus(s)
    logWeight: (s) => -a * softplusOf(-s) - b * softplusOf(s) - logNorm,
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
      throw new ImproperMessageError('a gamma message has no density')
    return gammaAxis(shape, rate, kinks)
  }
  const j = eta[1] ?? 0
  if (!(j > 0) || !Number.isFinite(j)) throw new ImproperMessageError('a Gaussian message has no density')
  return gaussianAxis((eta[0] ?? 0) / j, 1 / j, kinks)
}

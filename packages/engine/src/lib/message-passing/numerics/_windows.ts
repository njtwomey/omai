// Where the mass of a cavity lives, in the coordinate an integral over it is taken in: the one place that says how far a window
// reaches and where it is cut. A gamma variable is integrated in its log, a beta in its logit, a Gaussian in its value, and every
// integral of the numeric sites starts from the window of the cavity in that coordinate, then looks again where the mass is found.
// The reach (in standard deviations of the coordinate) and the clamp (the coordinate beyond which a value is not a number worth
// integrating over) belong to the integral, so they are named profiles here, never constants in the operator that integrates. The
// uniform grid of the tilts that integrate over one such coordinate (`tiltGrid`) is here too, with the zoom that finds the mass inside
// a window that is far wider than a likelihood leaves it.

import { shiftedSum } from './_moments.ts'
import { digammaOf, trigammaOf } from './_scalar.ts'

/** How far a window reaches either side of its centre (in standard deviations of the coordinate) and where it is cut (the coordinate is clamped to `[-clamp, clamp]`). */
export interface WindowProfile {
  readonly below: number
  readonly above: number
  readonly clamp: number
}

/**
 * The windows of the library's integrals over a positive variable in its log (a gamma cavity, a concentration, a precision), over a
 * probability in its logit (a beta cavity) and over a Gaussian on a log, in two kinds of integral.
 *
 * A uniform grid with a zoom (`tiltGrid`: the precision, gamma-scale, concentration and vector-scale tilts) has one window, the same
 * for all of them: twenty standard deviations of the coordinate either side of the centre reach far into the exponential left tail of
 * a log gamma density and past the super-exponential right one, the clamp keeps a value inside `exp(+-60)`, and the zoom finds the
 * mass inside it, so the window does not have to be tuned to the likelihood.
 *
 * The panelled grid of two or three coordinates (`_grid.ts`) has a few nodes on each axis (sixteen to sixty) and zooms by a node at a
 * time, so its first window must not be wider than the mass can be found in: a gamma axis reaches 12 standard deviations below and 8
 * above, a beta axis 10 either side. Widened to the tilts' twenty, the evidence of `bivariate-correlation` (three axes) moves by
 * 1.3e-3 nats.
 */
export const windowProfiles: {
  /** The zoomed uniform grids over a gamma, a beta or a Gaussian on a log: the one profile of every tilt. */
  readonly logTilt: WindowProfile
  /** A gamma axis of the panelled grid, in its log. */
  readonly gridGamma: WindowProfile
  /** A beta axis of the panelled grid, in its logit. */
  readonly gridBeta: WindowProfile
} = {
  logTilt: { below: 20, above: 20, clamp: 60 },
  gridGamma: { below: 12, above: 8, clamp: 45 },
  gridBeta: { below: 10, above: 10, clamp: 40 },
}

/** A window `[lo, hi]` of a coordinate and the centre of the cavity in it. */
export interface Window {
  readonly centre: number
  readonly lo: number
  readonly hi: number
}

/**
 * The window of a gamma cavity `Gamma(shape, rate)` in `t = log value`: centred where the mean of `log value` is, and as wide as
 * the standard deviation of `log value` (the square root of the trigamma function of the shape) times the profile's reach.
 *
 * @param shape - The cavity's shape.
 * @param rate - The cavity's rate.
 * @param profile - The reach and the clamp.
 * @returns The window.
 */
export function gammaLogWindow(shape: number, rate: number, profile: WindowProfile): Window {
  const centre = digammaOf(shape) - Math.log(rate)
  const sd = Math.sqrt(trigammaOf(shape))
  return {
    centre,
    lo: Math.max(centre - profile.below * sd, -profile.clamp),
    hi: Math.min(centre + profile.above * sd, profile.clamp),
  }
}

/**
 * The window of a beta cavity `Beta(a, b)` in `s = logit(value)`.
 *
 * @param a - The cavity's first shape.
 * @param b - The cavity's second shape.
 * @param profile - The reach and the clamp.
 * @returns The window.
 */
export function betaLogitWindow(a: number, b: number, profile: WindowProfile): Window {
  const centre = digammaOf(a) - digammaOf(b)
  const sd = Math.sqrt(trigammaOf(a) + trigammaOf(b))
  return {
    centre,
    lo: Math.max(centre - profile.below * sd, -profile.clamp),
    hi: Math.min(centre + profile.above * sd, profile.clamp),
  }
}

/**
 * The window of a Gaussian cavity in its own coordinate, `mean` plus or minus `span` standard deviations, cut at the clamp when
 * the profile has one (a Gaussian on the log of a positive variable).
 *
 * @param mean - The cavity's mean.
 * @param variance - The cavity's variance.
 * @param profile - The reach (`below` and `above`) and the clamp (`Infinity` for none).
 * @returns The window.
 */
export function gaussianWindow(mean: number, variance: number, profile: WindowProfile): Window {
  const sd = Math.sqrt(variance)
  return {
    centre: mean,
    lo: Math.max(mean - profile.below * sd, -profile.clamp),
    hi: Math.min(mean + profile.above * sd, profile.clamp),
  }
}

/** How the window of a belief in the variational quadrature is laid around its mode: the reach, the floor of the left reach per unit of `1 / shape`, and the caps. */
const variationalWindow: {
  readonly reach: number
  readonly leftPerShape: number
  readonly leftCap: number
  readonly rightCap: number
} = { reach: 9, leftPerShape: 20, leftCap: 45, rightCap: 8 }

/**
 * The window of a gamma belief `Gamma(shape, rate)` in `t = log value` for the variational quadrature: around the mode `log(shape / rate)`,
 * reaching left by at least `20 / shape` (the left tail of the density of `t` is only exponential) and at most 45, right by nine standard
 * deviations of `t` and at most 8.
 *
 * @param shape - The belief's shape.
 * @param rate - The belief's rate.
 * @returns The window.
 */
export function gammaModeWindow(shape: number, rate: number): Window {
  const sd = Math.sqrt(trigammaOf(shape))
  const mode = Math.log(shape / rate)
  const left = Math.min(
    Math.max(variationalWindow.reach * sd, variationalWindow.leftPerShape / shape),
    variationalWindow.leftCap,
  )
  const right = Math.min(variationalWindow.reach * sd, variationalWindow.rightCap)
  return { centre: mode, lo: mode - left, hi: mode + right }
}

/**
 * The window of a beta belief `Beta(p, q)` in `s = logit(value)` for the variational quadrature: around the mode `log(p / q)`, reaching each
 * side by at least `20 / shape` of that side's shape and at most 45.
 *
 * @param p - The belief's first shape.
 * @param q - The belief's second shape.
 * @returns The window.
 */
export function betaModeWindow(p: number, q: number): Window {
  const sd = Math.sqrt(trigammaOf(p) + trigammaOf(q))
  const mode = Math.log(p / q)
  const reach = variationalWindow.reach * sd
  const left = Math.min(Math.max(reach, variationalWindow.leftPerShape / p), variationalWindow.leftCap)
  const right = Math.min(Math.max(reach, variationalWindow.leftPerShape / q), variationalWindow.leftCap)
  return { centre: mode, lo: mode - left, hi: mode + right }
}

/** The points of a uniform grid over a window: `lo + i * step`, `i` from `0` to `points - 1`. */
export interface UniformGrid {
  readonly lo: number
  readonly hi: number
  readonly points: number
  readonly step: number
}

/**
 * A uniform grid over a window.
 *
 * @param window - The window.
 * @param points - The number of points (at least 2).
 * @returns The grid.
 */
export function uniformGrid(window: { readonly lo: number; readonly hi: number }, points: number): UniformGrid {
  return { lo: window.lo, hi: window.hi, points, step: (window.hi - window.lo) / (points - 1) }
}

/**
 * The `points` nodes of a uniform grid over a window, `lo + (hi - lo) i / (points - 1)` (the multiplication first: not the same last bit as
 * `lo + i * step`, which `uniformGrid` gives; the variational quadrature has always laid its nodes this way).
 *
 * @param window - The window.
 * @param points - The number of nodes (at least 2).
 * @returns The nodes.
 */
export function uniformNodes(window: { readonly lo: number; readonly hi: number }, points: number): Float64Array {
  return Float64Array.from({ length: points }, (_, i) => window.lo + ((window.hi - window.lo) * i) / (points - 1))
}

/** The weights of a grid, relative to the largest, below which a node is outside the mass: the zoom cuts there. */
const LIVE = 1e-13
/** A window that shrinks to less than this fraction of itself when zoomed is worth another grid. */
const SHRINK = 0.6
/** The most grids one tilt lays: the first over the cavity's window, then up to two over the window the mass was found in. */
const GRIDS = 3

/** What a one-dimensional grid over a window gives: its nodes, their spacing, and the log weights shifted and exponentiated. */
export interface TiltGrid {
  readonly t: Float64Array
  readonly step: number
  /** The largest log weight. */
  readonly top: number
  /** `exp(logw - top)`: at most one, the largest exactly one. */
  readonly weights: Float64Array
  /** The sum of the weights. */
  readonly total: number
}

/**
 * A uniform grid over the window of a cavity, zoomed onto where the mass is: the log weights are laid on `points` nodes of the window,
 * and when the nodes that carry weight (more than `1e-13` of the largest) are less than six tenths of the window, the same number of
 * nodes is laid over them (a node of margin either side), up to three grids in all. The window of a cavity is far wider than the
 * tilted mass that a likelihood leaves, and one fixed grid over it loses resolution exactly when the mass is narrow; the zoom is what
 * lets the profile of the window be one wide number for every integral.
 *
 * @param window - The window of the cavity in the coordinate (`gammaLogWindow` and the like).
 * @param points - The nodes of each grid (at least 3).
 * @param logw - The log of the integrand (the cavity's density included) at a coordinate.
 * @returns The nodes of the last grid, its spacing, its largest log weight and its shifted weights with their sum.
 * @throws ImproperMessageError when no node of the first grid has a finite weight.
 */
export function tiltGrid(
  window: { readonly lo: number; readonly hi: number },
  points: number,
  logw: (t: number) => number,
): TiltGrid {
  let w = window
  for (let grid = 1; ; grid++) {
    const t = uniformNodes(w, points)
    const { top, weights, total } = shiftedSum(Float64Array.from(t, logw))
    const step = (w.hi - w.lo) / (points - 1)
    if (grid === GRIDS || !Number.isFinite(top)) return { t, step, top, weights, total }
    let from = 0
    while (from < points - 1 && (weights[from] as number) < LIVE) from++
    let to = points - 1
    while (to > from && (weights[to] as number) < LIVE) to--
    const lo = t[Math.max(from - 1, 0)] as number
    const hi = t[Math.min(to + 1, points - 1)] as number
    if (!(hi - lo < SHRINK * (w.hi - w.lo))) return { t, step, top, weights, total }
    w = { lo, hi }
  }
}

/** The nodes of the scan the window finder lays over its current window. */
const SCAN = 801

/**
 * The window of the line holding the mass of a log density: scanned on a grid of `scan` points from `start`, widened while the
 * edges of the window are not negligible (more than 40 nats below the top), then narrowed onto the nodes within 40 nats of the top
 * and scanned again until the mass fills the window (a spike a grid of this size would miss is narrowed onto round by round). The one
 * way the library finds where a density lives, for the density of a prior and for the factor times the cavity of a tilted integral, so
 * a likelihood does not describe its own window.
 *
 * @param logDensity - The log density (up to a constant) on an array of points.
 * @param lo - The lower end of the support (`-Infinity` for none).
 * @param hi - The upper end (`Infinity` for none).
 * @param start - A point near the bulk of the mass, inside `[lo, hi]`, where the search starts.
 * @param scan - The nodes of each scan (default 801).
 * @returns The window `[a, b]`.
 */
export function findWindow(
  logDensity: (x: Float64Array) => Float64Array,
  lo: number,
  hi: number,
  start: number,
  scan: number = SCAN,
): readonly [number, number] {
  let a = Math.max(lo, start - 5)
  let b = Math.min(hi, start + 5)
  for (let round = 0; round < 40; round++) {
    const xs = Float64Array.from({ length: scan }, (_, i) => a + ((b - a) * i) / (scan - 1))
    const lp = logDensity(xs)
    let top = Number.NEGATIVE_INFINITY
    for (const v of lp) if (v > top) top = v
    if (!Number.isFinite(top)) {
      // Nothing finite in the window: look wider.
      const span = b - a
      a = Math.max(lo, a - 2 * span)
      b = Math.min(hi, b + 2 * span)
      continue
    }
    const span = b - a
    let first = 0
    let last = scan - 1
    while (first < scan - 1 && (lp[first] as number) < top - 40) first++
    while (last > 0 && (lp[last] as number) < top - 40) last--
    const wide = (first === 0 && a > lo) || (last === scan - 1 && b < hi)
    if (wide) {
      if (first === 0 && a > lo) a = Math.max(lo, a - 2 * span)
      if (last === scan - 1 && b < hi) b = Math.min(hi, b + 2 * span)
      continue
    }
    const pad = (2 * span) / (scan - 1)
    const na = Math.max(lo, a + (span * first) / (scan - 1) - pad)
    const nb = Math.min(hi, a + (span * last) / (scan - 1) + pad)
    // A window much wider than the mass is narrowed onto it and scanned again (a spike a grid of this size would miss).
    if (nb - na < 0.5 * span) {
      a = na
      b = nb
      continue
    }
    return [a, b]
  }
  return [a, b]
}

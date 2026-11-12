// Where the mass of a cavity lives, in the coordinate an integral over it is taken in: the one place that says how far a window
// reaches and where it is cut. A gamma variable is integrated in its log, a beta in its logit, a Gaussian in its value, and every
// integral of the numeric sites starts from the window of the cavity in that coordinate, then looks again where the mass is found.
// The reach (in standard deviations of the coordinate) and the clamp (the coordinate beyond which a value is not a number worth
// integrating over) belong to the integral, so they are named profiles here, never constants in the operator that integrates.

import { digammaOf, trigammaOf } from './_scalar.ts'

/** How far a window reaches either side of its centre (in standard deviations of the coordinate) and where it is cut (the coordinate is clamped to `[-clamp, clamp]`). */
export interface WindowProfile {
  readonly below: number
  readonly above: number
  readonly clamp: number
}

/**
 * The windows of the library's integrals, by the integral that uses them. The numbers are the ones each integral has always
 * used; making them agree is a change of accuracy, not of structure, and is decided here and nowhere else.
 */
export const windowProfiles: {
  /** The panelled grid of two or three coordinates (`grid.ts`): a gamma cavity in its log. */
  readonly pairGrid: WindowProfile
  /** The beta cavity in its logit, in the panelled grid. */
  readonly betaGrid: WindowProfile
  /** The Gaussian precision `gaussian-gamma-precision` integrates the difference over. */
  readonly precisionTilt: WindowProfile
  /** The gamma-scale hierarchies (`gamma-rate`, `gamma-product`). */
  readonly gammaScale: WindowProfile
  /** The random concentration of a beta or Dirichlet. */
  readonly concentration: WindowProfile
  /** The one multiplier of a vector of precisions and the gamma input of a chain of functions. */
  readonly vectorScale: WindowProfile
} = {
  pairGrid: { below: 12, above: 8, clamp: 45 },
  betaGrid: { below: 10, above: 10, clamp: 40 },
  precisionTilt: { below: 20, above: 20, clamp: 60 },
  gammaScale: { below: 18, above: 18, clamp: 60 },
  concentration: { below: 14, above: 14, clamp: 40 },
  vectorScale: { below: 15, above: 15, clamp: 60 },
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

/**
 * The log weights shifted by their maximum and exponentiated, so the weights of a grid never underflow: `weights[i] = exp(logw[i] -
 * top)` with `top` the largest log weight (`-Infinity` when every weight is zero: the caller decides what that means).
 *
 * @param logw - The log weights.
 * @returns The shift `top` and the shifted weights.
 */
export function shiftedWeights(logw: ArrayLike<number>): { readonly top: number; readonly weights: Float64Array } {
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < logw.length; i++) if ((logw[i] as number) > top) top = logw[i] as number
  return { top, weights: Float64Array.from({ length: logw.length }, (_, i) => Math.exp((logw[i] as number) - top)) }
}

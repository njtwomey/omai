// The one-dimensional tilted integral of the numeric sites: the mass and moments of a Gaussian cavity times a factor of the
// variable, on a panelled Gauss-Legendre grid laid where the factor lives (a window from the factor's own hint, split at its
// kinks) and then zoomed onto where the mass is found to be, so a cavity much wider than the likelihood (the first sweeps of
// a run) and one much narrower both resolve. What the likelihood sites, the unary functions and the numeric priors share.

import { ImproperMessageError } from '../core/errors.ts'
import { shiftedSum } from './_moments.ts'
import { gaussianGridOn } from './_quadrature.ts'

/** Where a factor, as a function of the variable, lives: a centre and a scale, how far (in scales) to look, and the kinks to split panels at. */
export interface WindowHint {
  readonly centre: number
  readonly scale: number
  /** How many scales either side of the centre the factor is integrated over (default 120, for heavy tails). */
  readonly reach?: number
  /** Points where the log factor has a kink or a jump (the outcome of an absolute-value likelihood, the origin of a rectifier). */
  readonly kinks?: readonly number[]
  /** Whether to zoom onto the mass found (default true); a heavy-tailed factor keeps its whole window, as the tails carry variance. */
  readonly zoom?: boolean
}

/** The tilted mass and moments, and those of a function of the variable when one is asked for. */
export interface Tilted {
  readonly logMass: number
  readonly mean: number
  readonly variance: number
  /** The mean and variance of `fn(x)` under the tilted density (equal to `mean` and `variance` without `fn`). */
  readonly fnMean: number
  readonly fnVariance: number
  /** The mean and variance of each of the extra functions asked for, in order. */
  readonly extra: readonly { readonly mean: number; readonly variance: number }[]
}

const PANELS = 80
const ORDER = 8
const ZOOMS = 6

interface Grid {
  readonly lo: number
  readonly hi: number
  readonly xs: number[]
  readonly lw: number[]
}

function evaluate(
  cavity: { readonly mean: number; readonly variance: number },
  lo: number,
  hi: number,
  kinks: readonly number[],
  logf: (x: number) => number,
): Grid {
  const cuts = [lo, ...kinks.filter((k) => k > lo && k < hi).sort((a, b) => a - b), hi]
  const xs: number[] = []
  const lw: number[] = []
  for (let c = 0; c + 1 < cuts.length; c++) {
    const a = cuts[c] as number
    const b = cuts[c + 1] as number
    const panels = Math.max(8, Math.round((PANELS * (b - a)) / (hi - lo)))
    const g = gaussianGridOn(cavity.mean, cavity.variance, a, b, panels, ORDER)
    for (let i = 0; i < g.x.length; i++) {
      const x = g.x[i] as number
      xs.push(x)
      lw.push(Math.log(g.w[i] as number) + logf(x))
    }
  }
  return { lo, hi, xs, lw }
}

function moments(
  g: Grid,
  fn: ((x: number) => number) | undefined,
  more: readonly ((x: number) => number)[],
): {
  logMass: number
  mean: number
  variance: number
  fnMean: number
  fnVariance: number
  top: number
  extra: { mean: number; variance: number }[]
} {
  const { top, weights, total: m0, logTotal } = shiftedSum(g.lw)
  if (!Number.isFinite(top)) throw new ImproperMessageError('the factor has no mass against this cavity')
  let m1 = 0
  for (let i = 0; i < weights.length; i++) m1 += (weights[i] as number) * (g.xs[i] as number)
  const mean = m1 / m0
  let m2 = 0
  let f1 = 0
  const fs = fn === undefined ? undefined : g.xs.map((x) => fn(x))
  for (let i = 0; i < weights.length; i++) {
    const e = weights[i] as number
    m2 += e * ((g.xs[i] as number) - mean) ** 2
    if (fs !== undefined) f1 += e * (fs[i] as number)
  }
  const variance = m2 / m0
  let fnMean = mean
  let fnVariance = variance
  if (fs !== undefined) {
    fnMean = f1 / m0
    let f2 = 0
    for (let i = 0; i < weights.length; i++) f2 += (weights[i] as number) * ((fs[i] as number) - fnMean) ** 2
    fnVariance = f2 / m0
  }
  const extra = more.map((h) => {
    const hs = g.xs.map((x) => h(x))
    let a1 = 0
    for (let i = 0; i < weights.length; i++) a1 += (weights[i] as number) * (hs[i] as number)
    const mu = a1 / m0
    let a2 = 0
    for (let i = 0; i < weights.length; i++) a2 += (weights[i] as number) * ((hs[i] as number) - mu) ** 2
    return { mean: mu, variance: a2 / m0 }
  })
  return { logMass: logTotal, mean, variance, fnMean, fnVariance, top, extra }
}

/**
 * The log mass and the mean and variance of the density proportional to N(mean, variance) exp(logf(x)), by a panelled
 * Gauss-Legendre rule. The window starts at the cavity's mean plus or minus nine standard deviations, narrowed to the
 * factor's own window when a hint is given (a heavy-tailed likelihood far narrower than its cavity), split at the hint's
 * kinks; it is then zoomed onto the mass found, up to six times, so the rule resolves a factor much narrower than the
 * cavity as well as one much wider. With `fn` the mean and variance of `fn(x)` under the same density come back too (the
 * output of a unary function). The log weights are shifted by their maximum, so nothing underflows.
 *
 * @param cavity - The Gaussian message the factor is integrated against.
 * @param logf - The log of the (powered) factor.
 * @param hint - Where the factor lives, when that is known.
 * @param fn - A function of the variable whose moments are wanted too.
 * @param more - Further functions of the variable whose moments are wanted (the conditional moments of another variable).
 * @returns The tilted log mass and moments.
 * @throws ImproperMessageError when the factor has no mass, or the integrand does not decay to the cavity's edge (an improper tilt).
 */
export function tiltedByQuadrature(
  cavity: { readonly mean: number; readonly variance: number },
  logf: (x: number) => number,
  hint?: WindowHint,
  fn?: (x: number) => number,
  more: readonly ((x: number) => number)[] = [],
): Tilted {
  const sd = Math.sqrt(cavity.variance)
  const cavityLo = cavity.mean - 9 * sd
  const cavityHi = cavity.mean + 9 * sd
  let lo = cavityLo
  let hi = cavityHi
  const kinks = hint?.kinks ?? []
  if (hint !== undefined) {
    const reach = hint.reach ?? 120
    const a = Math.max(lo, hint.centre - reach * hint.scale)
    const b = Math.min(hi, hint.centre + reach * hint.scale)
    // A cavity far from where the factor lives keeps its own window (the part of it nearest the factor).
    if (b > a) {
      lo = a
      hi = b
    }
  }
  let grid = evaluate(cavity, lo, hi, kinks, logf)
  let r = moments(grid, fn, more)
  // An integrand that peaks at the edge of the cavity's own window (still growing there) has no proper tilted density.
  if (hint === undefined && grid.lw.length > 2) {
    const n = grid.lw.length
    let at = 0
    for (let i = 1; i < n; i++) if ((grid.lw[i] as number) > (grid.lw[at] as number)) at = i
    if ((at < 0.02 * n && lo === cavityLo) || (at > 0.98 * n && hi === cavityHi))
      throw new ImproperMessageError('the tilted density does not decay inside the cavity window (an improper tilt)')
  }
  for (let z = 0; z < (hint?.zoom === false ? 0 : ZOOMS); z++) {
    const sdT = Math.sqrt(r.variance)
    const nlo = Math.max(grid.lo, r.mean - 10 * sdT)
    const nhi = Math.min(grid.hi, r.mean + 10 * sdT)
    if (!(nhi > nlo) || nhi - nlo > 0.6 * (grid.hi - grid.lo)) break
    grid = evaluate(cavity, nlo, nhi, kinks, logf)
    r = moments(grid, fn, more)
  }
  return {
    logMass: r.logMass,
    mean: r.mean,
    variance: r.variance,
    fnMean: r.fnMean,
    fnVariance: r.fnVariance,
    extra: r.extra,
  }
}

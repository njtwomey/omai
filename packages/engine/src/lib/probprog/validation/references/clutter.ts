/**
 * The exact posterior of the clutter problem with unknown precisions by quadrature over its four continuous unknowns (the
 * switches summed out one by one), and of a sparse sequence of outliers with an unknown share and precision. Each returns what the
 * atoms' pages draw: marginals as curves with their moments, and the probability that each switch is on.
 */

import { add, arange, div, mul, toFlat, zeros } from '../../../tensor/index.ts'
import { type ExactMarginal, weightedMarginal } from './marginal.ts'
import { lg } from './special.ts'

/** The priors of the clutter problem with unknown precisions. */
export interface ClutterPriors {
  /** The sd of the centre's zero-mean normal prior. */
  readonly centre: number
  readonly inlier: { readonly shape: number; readonly rate: number }
  readonly clutter: { readonly shape: number; readonly rate: number }
  readonly weight: { readonly a: number; readonly b: number }
}

/** The exact posterior of the clutter problem with unknown precisions: the continuous unknowns' marginals and the switches'. */
export interface ClutterPosterior {
  readonly centre: ExactMarginal
  readonly inlier: ExactMarginal
  readonly clutter: ExactMarginal
  readonly weight: ExactMarginal
  /** P(observation i is clutter). */
  readonly clutterProbability: readonly number[]
}

const logGammaDensity = (t: number, shape: number, rate: number, constant: number): number =>
  shape * Math.log(rate) - constant + (shape - 1) * Math.log(t) - rate * t

/**
 * The exact posterior of the clutter problem with an unknown centre, inlier precision, clutter precision and clutter weight, with
 * every observation's switch summed out: a quadrature over the four unknowns (the centre around the data's median, the two
 * precisions on a log grid, the weight on the unit interval), and each observation's probability of being clutter as its
 * average over that posterior.
 *
 * @param y - The observations.
 * @param priors - The priors.
 * @param points - Grid points per precision and for the weight (default 20; the centre has 1.4 times as many).
 * @returns The marginals and the probability that each observation is clutter.
 * @concept posterior-validation
 * @index function Exact clutter posterior with unknown centre, precisions and weight, by quadrature.
 * @example
 * ```ts
 * import { clutterPosterior } from 'omai-engine/probprog/validation'
 * const priors = { centre: 5, inlier: { shape: 2, rate: 1 }, clutter: { shape: 2, rate: 20 }, weight: { a: 2, b: 5 } }
 * clutterPosterior([1.4, 1.6, 1.5, 1.3, 9, 1.5], priors, 8).clutterProbability.length // => 6
 * ```
 */
export function clutterPosterior(y: readonly number[], priors: ClutterPriors, points = 20): ClutterPosterior {
  const n = y.length
  const median = [...y].sort((a, b) => a - b)[Math.floor(n / 2)] as number
  const span = (lo: number, hi: number, m: number): number[] =>
    toFlat(add(div(mul(add(arange(m), 0.5), hi - lo), m), lo))
  const centres = span(median - 1.2, median + 1.2, Math.round(points * 1.4))
  const logIn = span(Math.log(0.4), Math.log(40), points)
  const logOut = span(Math.log(0.004), Math.log(2), points)
  const weights = span(0.01, 0.75, points)
  const tIn = logIn.map(Math.exp)
  const tOut = logOut.map(Math.exp)
  const lgIn = lg(priors.inlier.shape)
  const lgOut = lg(priors.clutter.shape)
  const lbeta = lg(priors.weight.a + priors.weight.b) - lg(priors.weight.a) - lg(priors.weight.b)
  const half = 0.5 * Math.log(2 * Math.PI)
  // log N(y_i; 0, 1/tau) for each clutter precision.
  const outLog = tOut.map((t) => y.map((v) => 0.5 * Math.log(t) - half - 0.5 * t * v * v))
  const logw = weights.map(Math.log)
  const log1w = weights.map((w) => Math.log(1 - w))
  const na = centres.length
  // the table of every cell's log posterior, na x points^3 entries filled in place by the sweep below
  const lp = new Float64Array(na * points * points * points)
  const inlier = toFlat(zeros([n]))
  let top = Number.NEGATIVE_INFINITY
  let at = 0
  for (let a = 0; a < na; a++)
    for (let b = 0; b < points; b++) {
      const t = tIn[b] as number
      for (let i = 0; i < n; i++)
        inlier[i] = 0.5 * Math.log(t) - half - 0.5 * t * ((y[i] as number) - (centres[a] as number)) ** 2
      const prior =
        -0.5 * ((centres[a] as number) / priors.centre) ** 2 +
        logGammaDensity(t, priors.inlier.shape, priors.inlier.rate, lgIn) +
        (logIn[b] as number)
      for (let c = 0; c < points; c++) {
        const pc =
          prior +
          logGammaDensity(tOut[c] as number, priors.clutter.shape, priors.clutter.rate, lgOut) +
          (logOut[c] as number) +
          lbeta
        const out = outLog[c] as number[]
        for (let e = 0; e < points; e++) {
          let v = pc + (priors.weight.a - 1) * (logw[e] as number) + (priors.weight.b - 1) * (log1w[e] as number)
          for (let i = 0; i < n; i++) {
            const p = (log1w[e] as number) + (inlier[i] as number)
            const q = (logw[e] as number) + (out[i] as number)
            v += Math.max(p, q) + Math.log1p(Math.exp(-Math.abs(p - q)))
          }
          lp[at++] = v
          if (v > top) top = v
        }
      }
    }
  const margins = {
    centre: toFlat(zeros([na])),
    inlier: toFlat(zeros([points])),
    clutter: toFlat(zeros([points])),
    weight: toFlat(zeros([points])),
  }
  const clutter = toFlat(zeros([n]))
  let norm = 0
  at = 0
  for (let a = 0; a < na; a++)
    for (let b = 0; b < points; b++) {
      const t = tIn[b] as number
      for (let c = 0; c < points; c++)
        for (let e = 0; e < points; e++) {
          const w = Math.exp((lp[at++] as number) - top)
          norm += w
          margins.centre[a] = (margins.centre[a] as number) + w
          margins.inlier[b] = (margins.inlier[b] as number) + w
          margins.clutter[c] = (margins.clutter[c] as number) + w
          margins.weight[e] = (margins.weight[e] as number) + w
          // The switches: only the cells that carry mass matter.
          if (w < 1e-6) continue
          const out = outLog[c] as number[]
          for (let i = 0; i < n; i++) {
            const p =
              (log1w[e] as number) +
              0.5 * Math.log(t) -
              half -
              0.5 * t * ((y[i] as number) - (centres[a] as number)) ** 2
            clutter[i] = (clutter[i] as number) + w / (1 + Math.exp(p - ((logw[e] as number) + (out[i] as number))))
          }
        }
    }
  return {
    centre: weightedMarginal(centres, margins.centre, 'linear'),
    inlier: weightedMarginal(tIn, margins.inlier, 'linear'),
    clutter: weightedMarginal(tOut, margins.clutter, 'linear'),
    weight: weightedMarginal(weights, margins.weight, 'linear'),
    clutterProbability: clutter.map((v) => v / norm),
  }
}

/** The priors of the inlier and outlier mixture (the switches summed out) with an unknown centre, precision and share. */
export interface OutlierMixturePriors {
  readonly centre: number
  readonly precision: { readonly shape: number; readonly rate: number }
  readonly share: { readonly a: number; readonly b: number }
  /** How many times wider than the inliers the outliers are (known). */
  readonly width: number
}

/** The marginals of the inlier and outlier mixture's unknowns. */
export interface OutlierMixturePosterior {
  readonly centre: ExactMarginal
  readonly precision: ExactMarginal
  readonly share: ExactMarginal
}

/**
 * The exact posterior of an unknown centre, inlier precision and outlier share of the inlier and outlier mixture (each
 * observation is `N(mu, 1/tau)` or, with the share, `N(mu, width^2/tau)`; the switches are summed out), by quadrature over the
 * three unknowns: the centre around the data's median, the precision on a log grid, the share on the unit interval.
 *
 * @param y - The observations.
 * @param priors - The priors.
 * @param points - Grid points per unknown (default 40).
 * @returns The three marginals.
 * @concept posterior-validation
 * @index function Exact outlier-mixture posterior with unknown centre, precision and share.
 * @example
 * ```ts
 * import { outlierMixturePosterior } from 'omai-engine/probprog/validation'
 * const priors = { centre: 5, precision: { shape: 2, rate: 1 }, share: { a: 2, b: 5 }, width: 8 }
 * Math.abs(outlierMixturePosterior([1.4, 1.6, 1.5, 1.3, 9, 1.5], priors, 12).centre.mean - 1.5) < 0.6 // => true
 * ```
 */
export function outlierMixturePosterior(
  y: readonly number[],
  priors: OutlierMixturePriors,
  points = 40,
): OutlierMixturePosterior {
  const n = y.length
  const median = [...y].sort((a, b) => a - b)[Math.floor(n / 2)] as number
  const span = (lo: number, hi: number): number[] =>
    toFlat(add(div(mul(add(arange(points), 0.5), hi - lo), points), lo))
  const centres = span(median - 1.2, median + 1.2)
  const logs = span(Math.log(0.4), Math.log(40))
  const tau = logs.map(Math.exp)
  const shares = span(0.01, 0.75)
  const lgConst = lg(priors.precision.shape)
  const lbeta = lg(priors.share.a + priors.share.b) - lg(priors.share.a) - lg(priors.share.b)
  const half = 0.5 * Math.log(2 * Math.PI)
  const lw = Math.log(priors.width)
  // the table of every cell's log posterior, filled in place by the sweep below
  const lp = new Float64Array(points ** 3)
  let top = Number.NEGATIVE_INFINITY
  let at = 0
  for (let a = 0; a < points; a++)
    for (let b = 0; b < points; b++) {
      const t = tau[b] as number
      const inlier = y.map((v) => 0.5 * Math.log(t) - half - 0.5 * t * (v - (centres[a] as number)) ** 2)
      const outlier = y.map(
        (v) => 0.5 * Math.log(t) - lw - half - (0.5 * t * (v - (centres[a] as number)) ** 2) / priors.width ** 2,
      )
      const prior =
        -0.5 * ((centres[a] as number) / priors.centre) ** 2 +
        logGammaDensity(t, priors.precision.shape, priors.precision.rate, lgConst) +
        (logs[b] as number)
      for (let e = 0; e < points; e++) {
        const w = shares[e] as number
        let v = prior + lbeta + (priors.share.a - 1) * Math.log(w) + (priors.share.b - 1) * Math.log(1 - w)
        for (let i = 0; i < n; i++) {
          const p = Math.log(1 - w) + (inlier[i] as number)
          const q = Math.log(w) + (outlier[i] as number)
          v += Math.max(p, q) + Math.log1p(Math.exp(-Math.abs(p - q)))
        }
        lp[at++] = v
        if (v > top) top = v
      }
    }
  const margins = [0, 1, 2].map(() => toFlat(zeros([points]))) as [number[], number[], number[]]
  at = 0
  for (let a = 0; a < points; a++)
    for (let b = 0; b < points; b++)
      for (let e = 0; e < points; e++) {
        const w = Math.exp((lp[at++] as number) - top)
        margins[0][a] = (margins[0][a] as number) + w
        margins[1][b] = (margins[1][b] as number) + w
        margins[2][e] = (margins[2][e] as number) + w
      }
  return {
    centre: weightedMarginal(centres, margins[0], 'linear'),
    precision: weightedMarginal(tau, margins[1], 'linear'),
    share: weightedMarginal(shares, margins[2], 'linear'),
  }
}

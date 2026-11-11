/**
 * The reference answers of the extended gate atoms, for models whose posterior has no closed form and is too wide for a grid
 * over every unknown: a Gibbs sampler for a mixture of Gaussians with a normal prior on each mean, a gamma prior on each
 * precision and a Dirichlet prior on the weights (every conditional is conjugate, the ordered form is the relabelled chain), the
 * exact posterior of the clutter problem with unknown precisions by quadrature over its four continuous unknowns (the switches
 * summed out one by one), and the exact posterior of a sparse sequence with an unknown inclusion probability and slab precision
 * (two dimensions). Each returns what the atoms' pages draw: marginals as curves with their moments, and the probability that
 * each switch is on.
 */

import { OmaiError } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { foldIn, gammaVariates, type Key, key, uniformStream } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import type { ExactMarginal } from './ep-atoms.ts'
import { normalPdf } from './ep-atoms-quad.ts'

/** `log Gamma(x)` of a number. */
const lg = (x: number): number => item(lgamma(x))

/**
 * The marginal of a quantity from posterior draws: the sample mean and sd, and a Gaussian kernel density estimate (Silverman's
 * bandwidth), over the range the draws cover.
 *
 * @param draws - The draws.
 * @returns The marginal.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { samplesMarginal } from 'omai-engine/bayesian-examples'
 * samplesMarginal([0, 1, 2, 3, 4]).mean // => 2
 * ```
 */
export function samplesMarginal(draws: readonly number[]): ExactMarginal {
  const n = draws.length
  const mean = draws.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(draws.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  const h = Math.max(1.06 * sd * n ** -0.2, 1e-9)
  // Thinned, so the density costs a few hundred kernels, not thousands.
  const stride = Math.max(1, Math.floor(n / 400))
  const kept = draws.filter((_, i) => i % stride === 0)
  return {
    mean,
    sd,
    pdf: (x) => kept.reduce((a, c) => a + normalPdf(x, c, h), 0) / kept.length,
    range: [Math.min(...draws) - 2 * h, Math.max(...draws) + 2 * h],
  }
}

/** A seeded stream of standard normals (Box-Muller on the library's counter-based uniforms) and gamma variates. */
function streams(k: Key): {
  readonly uniform: () => number
  readonly normal: () => number
  readonly gamma: (a: number) => number
} {
  const uniform = uniformStream(k)
  let spare: number | undefined
  let calls = 0
  return {
    uniform,
    normal: () => {
      if (spare !== undefined) {
        const s = spare
        spare = undefined
        return s
      }
      const r = Math.sqrt(-2 * Math.log(1 - uniform()))
      const t = 2 * Math.PI * uniform()
      spare = r * Math.sin(t)
      return r * Math.cos(t)
    },
    gamma: (a) => gammaVariates(foldIn(k, 1 + calls++), [a])[0] as number,
  }
}

/** What a Gibbs run of a mixture is told. */
export interface MixtureGibbsSpec {
  readonly x: readonly number[]
  /** The prior of each component's mean (one per component; their number is the number of components). */
  readonly means: readonly { readonly mean: number; readonly sd: number }[]
  /** The gamma prior of every component's precision. */
  readonly precision: { readonly shape: number; readonly rate: number }
  /** The Dirichlet prior of the weights, one entry per component. */
  readonly weights: readonly number[]
  readonly seed: number
  readonly burn: number
  readonly draws: number
  /** Relabel every draw so that the means ascend: the ordered form of the model, whose prior is exchangeable. */
  readonly ordered: boolean
}

/** The draws of a mixture's parameters (by component) and each observation's posterior probability of each component. */
export interface MixtureGibbs {
  readonly mean: readonly (readonly number[])[]
  readonly precision: readonly (readonly number[])[]
  readonly weight: readonly (readonly number[])[]
  /** `responsibility[i][k]`: the average over the draws of P(observation i is from component k | the draw's parameters). */
  readonly responsibility: readonly (readonly number[])[]
}

/**
 * A Gibbs sampler for a mixture of Gaussians with component means, precisions and weights unknown (normal, gamma and Dirichlet
 * priors): the switches, the means, the precisions and the weights in turn, every conditional in closed form. With `ordered`, each
 * kept draw is relabelled so that the means ascend, which is exact for a prior that is the same for every component.
 *
 * @param spec - The data, the priors, the seed and the lengths.
 * @returns The draws and the responsibilities.
 * @throws OmaiError with code `value` when the priors do not have one entry per component.
 * @concept posterior-validation
 * @index function Gibbs sampler for a Gaussian mixture with unknown means, precisions and weights: the reference of the gate atoms.
 * @example
 * ```ts
 * import { mixtureGibbs } from 'omai-engine/bayesian-examples'
 * const x = [-2, -2.2, -1.8, 2, 2.1, 1.9]
 * const g = mixtureGibbs({ x, means: [{ mean: 0, sd: 3 }, { mean: 0, sd: 3 }], precision: { shape: 2, rate: 2 }, weights: [2, 2], seed: 1, burn: 50, draws: 100, ordered: true })
 * g.mean[0]?.length // => 100
 * ```
 */
export function mixtureGibbs(spec: MixtureGibbsSpec): MixtureGibbs {
  const K = spec.means.length
  if (spec.weights.length !== K) throw new OmaiError('value', 'mixtureGibbs: one weight prior per component')
  const rng = streams(key(spec.seed))
  const { x } = spec
  const n = x.length
  // Start from the sorted data cut into K groups.
  const order = [...x.keys()].sort((a, b) => (x[a] as number) - (x[b] as number))
  const z = new Array<number>(n).fill(0)
  order.forEach((i, r) => {
    z[i] = Math.min(K - 1, Math.floor((r * K) / n))
  })
  let mu = Array.from({ length: K }, () => 0)
  let tau = Array.from({ length: K }, () => spec.precision.shape / spec.precision.rate)
  let pi = Array.from({ length: K }, () => 1 / K)
  const out = {
    mean: Array.from({ length: K }, () => [] as number[]),
    precision: Array.from({ length: K }, () => [] as number[]),
    weight: Array.from({ length: K }, () => [] as number[]),
  }
  const resp = Array.from({ length: n }, () => new Array<number>(K).fill(0))
  for (let sweep = 0; sweep < spec.burn + spec.draws; sweep++) {
    // The means and precisions given the switches.
    const count = new Array<number>(K).fill(0)
    const total = new Array<number>(K).fill(0)
    for (let i = 0; i < n; i++) {
      const k = z[i] as number
      count[k] = (count[k] as number) + 1
      total[k] = (total[k] as number) + (x[i] as number)
    }
    mu = mu.map((_, k) => {
      const prior = spec.means[k] as { mean: number; sd: number }
      const precision = 1 / prior.sd ** 2 + (count[k] as number) * (tau[k] as number)
      const centre = (prior.mean / prior.sd ** 2 + (tau[k] as number) * (total[k] as number)) / precision
      return centre + rng.normal() / Math.sqrt(precision)
    })
    tau = tau.map((_, k) => {
      let ss = 0
      for (let i = 0; i < n; i++) if (z[i] === k) ss += ((x[i] as number) - (mu[k] as number)) ** 2
      return rng.gamma(spec.precision.shape + (count[k] as number) / 2) / (spec.precision.rate + ss / 2)
    })
    const g = pi.map((_, k) => rng.gamma((spec.weights[k] as number) + (count[k] as number)))
    const gs = g.reduce((a, b) => a + b, 0)
    pi = g.map((v) => v / gs)
    // The switches given the parameters.
    const rows: number[][] = []
    for (let i = 0; i < n; i++) {
      const lp = pi.map(
        (p, k) =>
          Math.log(p) +
          0.5 * Math.log(tau[k] as number) -
          0.5 * (tau[k] as number) * ((x[i] as number) - (mu[k] as number)) ** 2,
      )
      const top = Math.max(...lp)
      const w = lp.map((v) => Math.exp(v - top))
      const s = w.reduce((a, b) => a + b, 0)
      rows.push(w.map((v) => v / s))
      let u = rng.uniform() * s
      let pick = K - 1
      for (let k = 0; k < K; k++) {
        u -= w[k] as number
        if (u <= 0) {
          pick = k
          break
        }
      }
      z[i] = pick
    }
    if (sweep < spec.burn) continue
    const perm = [...Array(K).keys()]
    if (spec.ordered) perm.sort((a, b) => (mu[a] as number) - (mu[b] as number))
    perm.forEach((from, to) => {
      ;(out.mean[to] as number[]).push(mu[from] as number)
      ;(out.precision[to] as number[]).push(tau[from] as number)
      ;(out.weight[to] as number[]).push(pi[from] as number)
      for (let i = 0; i < n; i++)
        (resp[i] as number[])[to] = ((resp[i] as number[])[to] as number) + ((rows[i] as number[])[from] as number)
    })
  }
  return { ...out, responsibility: resp.map((r) => r.map((v) => v / spec.draws)) }
}

/** The marginal of one axis of a grid posterior: `mass` at `points`, as a density interpolated between them. */
function axisMarginal(points: readonly number[], mass: readonly number[]): ExactMarginal {
  const z = mass.reduce((a, b) => a + b, 0)
  const mean = points.reduce((a, p, i) => a + (p * (mass[i] as number)) / z, 0)
  const sd = Math.sqrt(points.reduce((a, p, i) => a + ((p - mean) ** 2 * (mass[i] as number)) / z, 0))
  // The density at a grid point: its mass over the width of its cell (in the variable's own units).
  const dens = points.map((p, i) => {
    const prev = points[i - 1] ?? p - ((points[1] as number) - p)
    const next = points[i + 1] ?? p + (p - (points[i - 1] as number))
    return (mass[i] as number) / z / ((next - prev) / 2)
  })
  const top = Math.max(...dens)
  const live = points.filter((_, i) => (dens[i] as number) > 1e-5 * top)
  return {
    mean,
    sd,
    pdf: (x) => {
      if (x <= (points[0] as number) || x >= (points[points.length - 1] as number)) return 0
      let lo = 0
      while ((points[lo + 1] as number) < x) lo++
      const t = (x - (points[lo] as number)) / ((points[lo + 1] as number) - (points[lo] as number))
      return (1 - t) * (dens[lo] as number) + t * (dens[lo + 1] as number)
    },
    range: [Math.min(...live), Math.max(...live)],
  }
}

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
 * @index function Exact posterior of the clutter problem with unknown centre, precisions and weight, by quadrature.
 * @example
 * ```ts
 * import { clutterPosterior } from 'omai-engine/bayesian-examples'
 * const priors = { centre: 5, inlier: { shape: 2, rate: 1 }, clutter: { shape: 2, rate: 20 }, weight: { a: 2, b: 5 } }
 * clutterPosterior([1.4, 1.6, 1.5, 1.3, 9, 1.5], priors, 8).clutterProbability.length // => 6
 * ```
 */
export function clutterPosterior(y: readonly number[], priors: ClutterPriors, points = 20): ClutterPosterior {
  const n = y.length
  const median = [...y].sort((a, b) => a - b)[Math.floor(n / 2)] as number
  const span = (lo: number, hi: number, m: number): number[] =>
    Array.from({ length: m }, (_, i) => lo + ((hi - lo) * (i + 0.5)) / m)
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
  const lp = new Float64Array(na * points * points * points)
  const inlier = new Array<number>(n)
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
    centre: new Array<number>(na).fill(0),
    inlier: new Array<number>(points).fill(0),
    clutter: new Array<number>(points).fill(0),
    weight: new Array<number>(points).fill(0),
  }
  const clutter = new Array<number>(n).fill(0)
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
    centre: axisMarginal(centres, margins.centre),
    inlier: axisMarginal(tIn, margins.inlier),
    clutter: axisMarginal(tOut, margins.clutter),
    weight: axisMarginal(weights, margins.weight),
    clutterProbability: clutter.map((v) => v / norm),
  }
}

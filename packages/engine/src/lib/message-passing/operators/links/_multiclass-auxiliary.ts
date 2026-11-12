// The tilted moments of the multiclass site for any number of classes, by a reduction of the integral over the class scores to
// one coordinate. The product grid of `multiclass.ts` has a node for every combination of the free scores and stops at six; this is
// the same integral written so that the scores do not meet, and it costs the number of classes.
//
// Softmax. The normaliser is a gamma integral, `S^-p = (1 / Gamma(p)) int t^(p-1) exp(-t S) dt`, with `S = sum_k exp(z_k)` and `p`
// the power of the factor, so given the auxiliary variable `t` the factor is a product over the classes of `exp(-t exp(z_k))` and
// the scores are independent of each other: the mass is `int dt t^(p-1) prod_k A_k(t)` with `A_k(t) = E_cavity_k[exp(-t exp(z_k))]`
// (the numerator `exp(p z_y)` shifts the cavity of the observed class's score by `p v_y`: a Gaussian times an exponential), the
// tilted marginal of every score is the mixture over `t` of the conditional ones (so its mean and variance are the weighted mean
// of the conditional means and the weighted conditional variance plus the spread of the conditional means), and each `A_k` and its
// conditional moments are one-dimensional integrals. There is no approximation but the quadrature: the line grid outside (the log of
// `t`, a window found by the grid's own zoom), a trapezoid rule in each score inside (spectrally accurate for a Gaussian times a
// smooth step).
//
// Probit. `P(y | z) = int phi(e) prod_(j != y) Phi(z_y - z_j + e) de`, and `c = z_y + e` is Gaussian under the cavity of `z_y`
// (variance plus one), so the mass is `int N(c; m_y, v_y + 1) prod_j Phi((c - m_j) / sqrt(1 + v_j)) dc`: each `Phi` against a
// Gaussian cavity is another `Phi`, and given `c` every score has its conditional moments in closed form (the observed class's is
// Gaussian: the posterior of `z_y` given `c`, the others' the truncated Gaussian of `z_j + xi <= c - ...`). One coordinate, for any
// number of classes. The power of the factor is not representable (a power of the integral over `c` is not an integral), so the
// rule takes the factor whole.

import { logNdtr } from '../../../numerics/special/_kernels.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { type GridAxis, gaussianAxis, integrateLine, pointAxis } from '../../numerics/_grid.ts'
import { centredMoments } from '../../numerics/_moments.ts'
import { digammaOf, lgammaOf, trigammaOf } from '../../numerics/_scalar.ts'

/** The Gaussian cavity of a free score. */
export interface Cavity {
  readonly mean: number
  readonly variance: number
}

/** What the rule needs to know of the factor: the free scores' cavities, which of them is the observed class, and the known scores. */
export interface AuxiliaryProblem {
  /** The cavities of the free scores. */
  readonly cavities: readonly Cavity[]
  /** The position of the observed class among the free scores, or `-1` when its score is a known value. */
  readonly observed: number
  /** The known score of the observed class when `observed` is `-1`. */
  readonly observedScore: number
  /** The log of the sum of `exp(score)` over the known scores, the observed class's included when it is known (softmax; `-Infinity` when none). */
  readonly fixedLogSum: number
  /** The known scores of the classes other than the observed one (probit). */
  readonly fixedOthers: readonly number[]
  /** The power of the factor (fractional EP). */
  readonly power: number
}

/** The log mass of the factor against the normalised cavities, and the mean and the variance of each free score under the tilted distribution. */
export interface AuxiliaryTilt {
  readonly logMass: number
  readonly mean: readonly number[]
  readonly variance: readonly number[]
}

/** How far a score's window reaches either side of its mean, in standard deviations. */
const REACH = 8
/** The widest step of the trapezoid rule in a score (the smooth step `exp(-t exp(z))` has a width of about one), and its step in standard deviations of the cavity. */
const STEP = 0.5
const STEP_SD = 0.8
/** The most nodes in a score's grid (a cavity far wider than any posterior is a first sweep's: the step grows rather than the cost). */
const MOST_NODES = 400

/** A score under its Gaussian cavity: a uniform grid over `mean +- REACH sd`, the log weights of the (normalised) Gaussian, and `exp(z)` at each node. */
export interface ScoreGrid {
  readonly z: Float64Array
  readonly logw: Float64Array
  readonly e: Float64Array
}

export function scoreGrid(mean: number, variance: number): ScoreGrid {
  const sd = Math.sqrt(variance)
  const lo = mean - REACH * sd
  const hi = mean + REACH * sd
  const n = Math.min(MOST_NODES, Math.ceil((hi - lo) / Math.min(STEP, STEP_SD * sd)) + 1)
  const z = Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  const logw = Float64Array.from(z, (v) => (-0.5 * (v - mean) ** 2) / variance)
  const norm = logSum(logw)
  for (let i = 0; i < n; i++) logw[i] = (logw[i] as number) - norm
  return { z, logw, e: Float64Array.from(z, Math.exp) }
}

/** `log(sum exp(v))` of numbers, `-Infinity` for none. */
export function logSum(values: ArrayLike<number>): number {
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < values.length; i++) if ((values[i] as number) > top) top = values[i] as number
  if (!Number.isFinite(top)) return top
  let s = 0
  for (let i = 0; i < values.length; i++) s += Math.exp((values[i] as number) - top)
  return top + Math.log(s)
}

/** `log A(t)` for a score: the log of the sum over its nodes of `w exp(-t e)`, shifted by the largest term. */
export function logTransform(g: ScoreGrid, t: number): number {
  const n = g.z.length
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < n; i++) {
    const v = (g.logw[i] as number) - t * (g.e[i] as number)
    if (v > top) top = v
  }
  if (!Number.isFinite(top)) return Number.NEGATIVE_INFINITY
  let s = 0
  for (let i = 0; i < n; i++) s += Math.exp((g.logw[i] as number) - t * (g.e[i] as number) - top)
  return top + Math.log(s)
}

/** The mean and the centred variance of a score under its cavity times `exp(-t exp(z))`. */
function conditionalMoments(g: ScoreGrid, t: number): { readonly mean: number; readonly variance: number } {
  const n = g.z.length
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) v[i] = (g.logw[i] as number) - t * (g.e[i] as number)
  let top = Number.NEGATIVE_INFINITY
  for (const x of v) if (x > top) top = x
  let total = 0
  for (let i = 0; i < n; i++) {
    const w = Math.exp((v[i] as number) - top)
    v[i] = w
    total += w
  }
  return centredMoments(v, total, g.z)
}

/** The mean and the variance of a score from its conditional moments at the nodes of a mixture (the law of total variance: the conditional variances plus the spread of the conditional means). */
function mixtureMoments(
  weights: ArrayLike<number>,
  at: (i: number) => { readonly mean: number; readonly variance: number },
  nodes: readonly number[],
): { readonly mean: number; readonly variance: number } {
  let sum = 0
  let first = 0
  for (const i of nodes) {
    sum += weights[i] as number
    first += (weights[i] as number) * at(i).mean
  }
  const mean = first / sum
  let second = 0
  for (const i of nodes) {
    const c = at(i)
    second += (weights[i] as number) * (c.variance + (c.mean - mean) ** 2)
  }
  return { mean, variance: second / sum }
}

/** A node of the outer grid whose weight is below this (the weights sum to one) is outside the mass. */
const NEGLIGIBLE = 1e-18

/** The smallest power the window of the auxiliary variable can hold (its left tail is `exp(p s)`: reaching 1e-13 of the mass takes `30 / p` below the centre). */
const SMALLEST_POWER = 0.03

/**
 * The tilted moments of a softmax factor (power `p`) against Gaussian cavities for any number of free scores: one integral over the
 * log of the auxiliary variable, with one-dimensional integrals over each score inside it.
 *
 * @param problem - The cavities of the free scores, the observed class and the known scores.
 * @returns The log mass and the mean and variance of each free score.
 * @throws ParameterError when the power is below the smallest the window holds.
 * @throws ImproperMessageError when the factor has no mass under the cavities.
 */
export function softmaxAuxiliary(problem: AuxiliaryProblem): AuxiliaryTilt {
  const { cavities, observed, power: p } = problem
  if (!(p >= SMALLEST_POWER))
    throw new ParameterError(`multiclass-site: the auxiliary-variable rule takes a power of at least ${SMALLEST_POWER}`)
  const r = cavities.length
  // The numerator exp(p z_y) of the observed class moves its cavity: N(m, v) exp(p z) = exp(p m + p^2 v / 2) N(m + p v, v).
  let constant = observed < 0 ? p * problem.observedScore : 0
  const shifted = cavities.map((c, k) => {
    if (k !== observed) return c
    constant = p * c.mean + 0.5 * p * p * c.variance
    return { mean: c.mean + p * c.variance, variance: c.variance }
  })
  const grids = shifted.map((c) => scoreGrid(c.mean, c.variance))
  const known = problem.fixedLogSum
  const knownSum = Number.isFinite(known) ? Math.exp(known) : 0
  // The window of log t: given the scores t is Gamma(p, S), so log t sits at digamma(p) - log S with the left tail exp(p s) and a
  // right tail that dies at once; S ranges over the cavities' reach (six standard deviations).
  const high = logSum([known, ...shifted.map((c) => c.mean + 6 * Math.sqrt(c.variance))])
  const low = logSum([known, ...shifted.map((c) => c.mean - 6 * Math.sqrt(c.variance))])
  const centre = digammaOf(p)
  const lo = Math.max(centre - high - 30 / p - 2 * Math.sqrt(trigammaOf(p)), -600)
  const hi = Math.min(centre - low + 5 + Math.max(0, Math.log(1 / p)), 600)
  const axis: GridAxis = { point: false, lo, hi, kinks: [], logWeight: () => 0 }
  const integral = integrateLine(axis, (s) => {
    const t = Math.exp(s)
    let total = p * s - t * knownSum
    for (let k = 0; k < r; k++) total += logTransform(grids[k] as ScoreGrid, t)
    return total
  })
  const nodes = integral.nodes[0] as Float64Array
  const kept: number[] = []
  for (let q = 0; q < nodes.length; q++) if ((integral.weights[q] as number) > NEGLIGIBLE) kept.push(q)
  const conditional = new Map<number, { readonly mean: number; readonly variance: number }[]>()
  for (const q of kept) {
    const t = Math.exp(nodes[q] as number)
    conditional.set(
      q,
      grids.map((g) => conditionalMoments(g, t)),
    )
  }
  const mean: number[] = []
  const variance: number[] = []
  for (let k = 0; k < r; k++) {
    const m = mixtureMoments(
      integral.weights,
      (q) => (conditional.get(q) as { mean: number; variance: number }[])[k] as { mean: number; variance: number },
      kept,
    )
    mean.push(m.mean)
    variance.push(m.variance)
  }
  return { logMass: constant - lgammaOf(p) + integral.logMass, mean, variance }
}

/** The log of the standard normal density. */
const logPhiDensity = (x: number): number => -0.5 * x * x - 0.9189385332046727

/**
 * The tilted moments of a multinomial probit factor against Gaussian cavities for any number of free scores: one integral over
 * `c = z_y + e` (the observed class's score plus its noise, Gaussian under the cavity), with every score's moments given `c` in
 * closed form.
 *
 * @param problem - The cavities of the free scores, the observed class and the known scores of the others (the power must be one).
 * @returns The log mass and the mean and variance of each free score.
 * @throws ParameterError when the power is not one.
 * @throws ImproperMessageError when the factor has no mass under the cavities or a tilted variance is not positive.
 */
export function probitAuxiliary(problem: AuxiliaryProblem): AuxiliaryTilt {
  const { cavities, observed } = problem
  if (problem.power !== 1)
    throw new ParameterError('multiclass-site: the multinomial probit of more than six scores takes a power of one')
  const r = cavities.length
  // The other classes, free (with their cavities) and known (variance zero): the factor of each is Phi((c - m) / sqrt(1 + v)).
  const others: { readonly index: number; readonly mean: number; readonly variance: number }[] = []
  for (let k = 0; k < r; k++)
    if (k !== observed)
      others.push({ index: k, mean: (cavities[k] as Cavity).mean, variance: (cavities[k] as Cavity).variance })
  const all = [...others, ...problem.fixedOthers.map((z) => ({ index: -1, mean: z, variance: 0 }))]
  const scale = all.map((o) => 1 / Math.sqrt(1 + o.variance))
  const observedCavity = observed >= 0 ? (cavities[observed] as Cavity) : { mean: problem.observedScore, variance: 0 }
  const axis = gaussianAxis(observedCavity.mean, observedCavity.variance + 1)
  const integral = integrateLine(axis, (c) => {
    let total = 0
    for (let j = 0; j < all.length; j++)
      total += logNdtr((c - (all[j] as { mean: number }).mean) * (scale[j] as number))
    return total
  })
  const nodes = integral.nodes[0] as Float64Array
  const kept: number[] = []
  for (let q = 0; q < nodes.length; q++) if ((integral.weights[q] as number) > NEGLIGIBLE) kept.push(q)
  const mean = new Array<number>(r).fill(0)
  const variance = new Array<number>(r).fill(0)
  if (observed >= 0) {
    // z_y given c is Gaussian: its posterior against the noise, the mean linear in c with the same variance at every node.
    const gain = observedCavity.variance / (observedCavity.variance + 1)
    const spread = centredMoments(integral.weights, 1, nodes)
    mean[observed] = observedCavity.mean + gain * (spread.mean - observedCavity.mean)
    variance[observed] = gain + gain * gain * spread.variance
  }
  // The other free scores: the truncated Gaussian of z_j given z_j + xi <= c, at every node of the mixture over c.
  for (const [j, o] of others.entries()) {
    const m = mixtureMoments(
      integral.weights,
      (q) => {
        const alpha = ((nodes[q] as number) - o.mean) * (scale[j] as number)
        const ratio = Math.exp(logPhiDensity(alpha) - logNdtr(alpha))
        const shrink = o.variance * (scale[j] as number)
        return { mean: o.mean - shrink * ratio, variance: o.variance - shrink * shrink * ratio * (alpha + ratio) }
      },
      kept,
    )
    mean[o.index] = m.mean
    variance[o.index] = m.variance
  }
  if (!variance.every((v) => v > 0 && Number.isFinite(v)))
    throw new ImproperMessageError('multiclass-site: a tilted variance is not positive')
  return { logMass: integral.logMass, mean, variance }
}

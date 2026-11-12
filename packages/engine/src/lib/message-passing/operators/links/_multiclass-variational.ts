// The variational (mean-field) messages of the multiclass site for any number of classes, by the same reduction to one coordinate
// that `_multiclass-auxiliary.ts` gives the tilted moments. The product grid of `multiclass.ts` integrates the expected log over
// every combination of the free scores and stops at six; the mean field has an advantage the cavities of EP do not: the beliefs of
// the scores are independent, so an expectation of a product over the classes is the product of their expectations.
//
// Softmax. `log P(y | z) = z_y - log S` with `S = sum_k exp(z_k)`, and Frullani's integral writes `-log S = int_0^inf (exp(-t S) -
// exp(-t)) dt / t` (a gamma integral over the auxiliary variable `t`, in `s = log t` the measure is `ds`). Under independent
// beliefs `E exp(-t S) = exp(-t K) prod_k A_k(t)` with `A_k(t) = E_k[exp(-t exp(z_k))]` (`K` the sum of the known scores' exponentials),
// so the expected log is `E z_y + int (exp(-t K) prod_k A_k(t) - exp(-t)) ds`, exact up to the quadrature (a trapezoid rule in `s`:
// the integrand is analytic, decays like `exp(s)` to the left and like a Gaussian tail to the right, so the rule is spectrally
// accurate), and the expected log as a function of one score `z_j` with the others averaged is the same integral with `A_j(t)`
// replaced by `exp(-t exp(z_j))`: every message is a vector of those over the nodes of the score's quadrature, found from one
// table of `A_k(t_s)` per factor and sweep. No bound and no approximation but the quadrature: this is the expected log the product
// grid computes for six scores, at a cost that is the number of classes.
//
// Probit. `P(y | z) = int N(c; z_y, 1) prod_(j != y) Phi(c - z_j) dc` with `c = z_y + e` the observed class's score plus its noise, an
// expectation of a product whose log has no separable form, so the rule is a bound. `F(G) = log int N(c; z_y, 1) exp(G(c)) dc` is
// convex in `G`, so Jensen's inequality over the scores of the other classes (not the observed one's) gives `E log P >= E_(z_y) log
// int N(c; z_y, 1) exp(H(c)) dc` with `H(c) = sum_j E log Phi(c - z_j)`, the expectation of `log Phi` of a Gaussian (one Gauss-Hermite
// sum) at every node of a grid of `c`. The observed class's score stays inside the log: the only gap is the other scores' spread,
// and it is zero for known scores. The bound is a closed function of the beliefs, so the message to a score is the exact natural
// gradient of it: to a score `z` of another class the expectation of `log Phi(c - z)` under the density of `c` that the bound's
// optimum puts (`W(c)`, the mixture over the observed class's belief of its posteriors), to the observed class's score the value of
// the bound at each of its nodes. One grid, a trapezoid rule in `c`: the integrand is analytic and decays as a Gaussian.

import { logNdtr } from '../../../numerics/special/_kernels.ts'
import { add, arange, exp, mul, neg } from '../../../tensor/index.ts'
import { ImproperMessageError } from '../../core/errors.ts'
import { float64Of } from '../../numerics/_arrays.ts'
import { gaussianNodes } from '../../numerics/_quadrature.ts'
import { type Cavity, logSum, logTransform, scoreGrid } from './_multiclass-auxiliary.ts'

/** The beliefs of the scores of one factor: a Gaussian each, a known score a belief of variance zero. */
export type ScoreBeliefs = readonly Cavity[]

/** The step of the trapezoid rule in `s = log t` (the integrand is analytic in a strip of half-width `pi / 2`: the error is about `exp(-pi^2 / step)`). */
const STEP = 0.4
/** The most nodes of the outer rule (a window wider than `STEP` times this grows the step instead). */
const MOST_OUTER = 900

/** What the softmax's expected log and messages share for one set of beliefs: the outer nodes and the table of `log A_k(t_s)`. */
export interface SoftmaxTable {
  /** `t = exp(s)` at each node. */
  readonly t: Float64Array
  /** The step in `s`. */
  readonly step: number
  /** `log A_k(t)` at each node for every free class (`undefined` for a known one). */
  readonly logA: readonly (Float64Array | undefined)[]
  /** `-t K` at each node, `K` the sum of the known scores' exponentials. */
  readonly known: Float64Array
  /** `-t K + sum_k log A_k(t)`: the log of `E exp(-t S)`. */
  readonly total: Float64Array
}

/**
 * The table of the softmax's transforms for the beliefs of the scores: the outer grid over `log t`, with a window that holds the
 * integrand (a left end where `t S` is negligible, a right end where `E exp(-t S)` is, both from six standard deviations of the
 * scores) and `log A_k` for every free class at every node.
 *
 * @param beliefs - The score beliefs (variance zero for a known score).
 * @returns The table.
 */
export function softmaxTable(beliefs: ScoreBeliefs): SoftmaxTable {
  const free = beliefs.filter((b) => b.variance > 0)
  const knownLog = logSum(beliefs.filter((b) => !(b.variance > 0)).map((b) => b.mean))
  const knownSum = Number.isFinite(knownLog) ? Math.exp(knownLog) : 0
  const high = logSum([knownLog, ...free.map((b) => b.mean + 6 * Math.sqrt(b.variance))])
  const low = logSum([knownLog, ...free.map((b) => b.mean - 6 * Math.sqrt(b.variance))])
  const lo = -high - 22
  const hi = -low + 6
  const count = Math.min(MOST_OUTER, Math.ceil((hi - lo) / STEP) + 1)
  const step = (hi - lo) / (count - 1)
  const t = float64Of(exp(add(lo, mul(arange(count), step))))
  const grids = beliefs.map((b) => (b.variance > 0 ? scoreGrid(b.mean, b.variance) : undefined))
  const logA = grids.map((g) => (g === undefined ? undefined : Float64Array.from(t, (v) => logTransform(g, v))))
  const known = float64Of(mul(neg(t), knownSum))
  const total = Float64Array.from(t, (_, i) => {
    let s = known[i] as number
    for (const a of logA) if (a !== undefined) s += a[i] as number
    return s
  })
  return { t, step, logA, known, total }
}

/**
 * The expected log of the softmax probability of class `y` under independent Gaussian beliefs of the scores:
 * `E z_y - E log sum_k exp(z_k)`, exact up to the quadrature.
 *
 * @param table - The table of the beliefs.
 * @param beliefs - The score beliefs.
 * @param y - The observed class.
 * @returns The expected log.
 */
export function softmaxExpectedLog(table: SoftmaxTable, beliefs: ScoreBeliefs, y: number): number {
  let sum = 0
  for (let i = 0; i < table.t.length; i++) sum += Math.exp(table.total[i] as number) - Math.exp(-(table.t[i] as number))
  return (beliefs[y] as Cavity).mean + table.step * sum
}

/**
 * The expected log of the softmax probability of class `y` as a function of the score of class `to` (the other scores averaged
 * under their beliefs), at the given values of that score: the function the natural-gradient message of the score needs.
 *
 * @param table - The table of the beliefs.
 * @param beliefs - The score beliefs.
 * @param y - The observed class.
 * @param to - A free class.
 * @param z - The values of the score `to`.
 * @returns `E_{others}[log P(y | z)]` at each value.
 */
export function softmaxScoreExpectation(
  table: SoftmaxTable,
  beliefs: ScoreBeliefs,
  y: number,
  to: number,
  z: ArrayLike<number>,
): Float64Array {
  const n = table.t.length
  // log E_{others} exp(-t S_others) at each node: the known scores' term and the free classes' transforms but the class's own.
  const rest = Float64Array.from(table.known)
  for (let k = 0; k < table.logA.length; k++) {
    const a = table.logA[k]
    if (k === to || a === undefined) continue
    for (let i = 0; i < n; i++) rest[i] = (rest[i] as number) + (a[i] as number)
  }
  const constant = to === y ? Number.NaN : (beliefs[y] as Cavity).mean
  const out = new Float64Array(z.length)
  let base = 0
  for (let i = 0; i < n; i++) base -= Math.exp(-(table.t[i] as number))
  for (let q = 0; q < z.length; q++) {
    const e = Math.exp(z[q] as number)
    let sum = base
    for (let i = 0; i < n; i++) sum += Math.exp((rest[i] as number) - (table.t[i] as number) * e)
    out[q] = (to === y ? (z[q] as number) : constant) + table.step * sum
  }
  return out
}

// ----- the probit -----

/** The nodes of the standard normal for the expectation of `log Phi` of a Gaussian (the other scores' inner expectation). */
const NORMAL = gaussianNodes(0, 1, 16)

/** The nodes of the observed class's score in the bound. */
const OBSERVED_NODES = 48
/** The step of the grid of `c` and the most nodes it has. */
const C_STEP = 0.125
const C_MOST = 1500

/** `E log Phi(x)` for `x ~ N(a, s2)`. */
export function expectedLogPhi(a: number, s2: number): number {
  if (!(s2 > 0)) return logNdtr(a)
  const s = Math.sqrt(s2)
  let total = 0
  for (let i = 0; i < NORMAL.x.length; i++) total += (NORMAL.w[i] as number) * logNdtr(a + s * (NORMAL.x[i] as number))
  return total
}

/** The bound on the expected log of the multinomial probit probability: the grid of `c`, the terms of the other classes and the value. */
export interface ProbitBound {
  /** The nodes of `c = z_y + e`. */
  readonly c: Float64Array
  /** The step of the grid. */
  readonly step: number
  /** `H(c) = sum_j E log Phi(c - z_j)` over the other classes, at each node. */
  readonly h: Float64Array
  /** The density of `c` the bound's optimum puts (summing to one): the weights of the other classes' messages. */
  readonly weights: Float64Array
  /** The lower bound on the expected log probability of the observed class. */
  readonly value: number
}

/** `log int N(c; z, 1) exp(H(c)) dc` on the grid (and the posterior weights of `c` given `z` when asked). */
function logIntegral(
  bound: Pick<ProbitBound, 'c' | 'step' | 'h'>,
  z: number,
  posterior?: Float64Array,
  scale = 1,
): number {
  const n = bound.c.length
  const logw = new Float64Array(n)
  let top = Number.NEGATIVE_INFINITY
  for (let i = 0; i < n; i++) {
    const v = -0.5 * ((bound.c[i] as number) - z) ** 2 + (bound.h[i] as number)
    logw[i] = v
    if (v > top) top = v
  }
  let sum = 0
  for (let i = 0; i < n; i++) {
    const w = Math.exp((logw[i] as number) - top)
    logw[i] = w
    sum += w
  }
  if (posterior !== undefined)
    for (let i = 0; i < n; i++) posterior[i] = (posterior[i] as number) + (scale * (logw[i] as number)) / sum
  return top + Math.log(sum * bound.step) - 0.9189385332046727
}

/**
 * The bound on `E log P(y | z)` of the multinomial probit under independent Gaussian beliefs of the scores:
 * `E_(z_y) log int N(c; z_y, 1) exp(H(c)) dc`, `H(c) = sum_j E log Phi(c - z_j)` over the other classes.
 *
 * @param beliefs - The score beliefs (variance zero for a known score).
 * @param y - The observed class.
 * @returns The grid of `c`, `H`, the optimal density of `c` and the bound.
 * @throws ImproperMessageError when the bound is not finite.
 */
export function probitBound(beliefs: ScoreBeliefs, y: number): ProbitBound {
  const observed = beliefs[y] as Cavity
  const others = beliefs.filter((_, k) => k !== y)
  const sdY = Math.sqrt(observed.variance)
  let lo = observed.mean - 8 * sdY
  let hi = observed.mean + 8 * sdY
  for (const o of others) {
    lo = Math.min(lo, o.mean - 8 * Math.sqrt(o.variance))
    hi = Math.max(hi, o.mean + 8 * Math.sqrt(o.variance))
  }
  lo -= 8
  hi += 8
  const count = Math.min(C_MOST, Math.ceil((hi - lo) / C_STEP) + 1)
  const step = (hi - lo) / (count - 1)
  const c = float64Of(add(lo, mul(arange(count), step)))
  const h = Float64Array.from(c, (v) => {
    let total = 0
    for (const o of others) total += expectedLogPhi(v - o.mean, o.variance)
    return total
  })
  const nodes = gaussianNodes(observed.mean, observed.variance, OBSERVED_NODES)
  const weights = new Float64Array(count)
  let value = 0
  for (let n = 0; n < nodes.x.length; n++)
    value += (nodes.w[n] as number) * logIntegral({ c, step, h }, nodes.x[n] as number, weights, nodes.w[n] as number)
  if (!Number.isFinite(value)) throw new ImproperMessageError('multiclass-site: the probit bound is not finite')
  return { c, step, h, weights, value }
}

/**
 * The bound's expected log as a function of the score of class `to` (the others averaged under their beliefs), at the given values of
 * that score: for the observed class `log int N(c; z, 1) exp(H(c)) dc`, for another class `E_c log Phi(c - z)` under the optimal
 * density of `c`.
 *
 * @param bound - The bound of the beliefs.
 * @param y - The observed class.
 * @param to - A free class.
 * @param z - The values of the score `to`.
 * @returns The expected log at each value.
 */
export function probitScoreExpectation(bound: ProbitBound, y: number, to: number, z: ArrayLike<number>): Float64Array {
  if (to === y) return Float64Array.from(z, (value) => logIntegral(bound, value))
  const kept: number[] = []
  for (let i = 0; i < bound.weights.length; i++) if ((bound.weights[i] as number) > 1e-14) kept.push(i)
  return Float64Array.from(z, (value) => {
    let total = 0
    for (const i of kept) total += (bound.weights[i] as number) * logNdtr((bound.c[i] as number) - value)
    return total
  })
}

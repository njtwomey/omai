/**
 * The multiclass site: a categorical outcome `y` among `K` classes whose probabilities are a softmax (`link: 'logit'`, the
 * multinomial logit) or the multinomial probit (`link: 'probit'`: `y` is the class whose utility `z_k + e_k`, with standard normal
 * `e_k`, is the largest) of `K` Gaussian scores `z_k`, one port per class. The factor is a function of the `K` scores jointly, so
 * the tilted moments are a `K`-dimensional integral against the product of the cavities; it is found by adaptive Gauss-Hermite
 * quadrature with a full-covariance proposal (the cavity first, then the Gaussian of the tilted moments just found, three rounds),
 * which is exact for a smooth factor and well behaved when the cavities are much wider than the factor is, as in the first
 * sweeps. A class observed as a known score is a constant coordinate, and up to six free scores are integrated (the grid is a
 * product; larger problems need a delta-method site, which this is not). The linear multiclass classifier is one such factor per
 * row on the scores `a . w_k` of the rows (the compiler makes each score a Gaussian variable).
 */

import { OmaiError } from '../../../base/index.ts'
import { ndtr } from '../../../numerics/special/_kernels.ts'
import { errorText } from '../../core/_messages.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, numberParam } from '../../core/args.ts'
import type { Arg, FactorOperator, OperatorContext, OperatorResult, Params, Port } from '../../core/types.ts'
import { gaussianNodes } from '../../numerics/_quadrature.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

/** The nodes per free score for a number of free scores (the grid is their product): the cap keeps a site at some tens of thousands of evaluations. */
const NODES: Readonly<Record<number, number>> = { 1: 48, 2: 24, 3: 8, 4: 7, 5: 6, 6: 5 }

/** Rounds of adaptation of the proposal: the first uses the cavity, each next the Gaussian of the tilted moments of the one before. */
const ROUNDS = 3

/** The classes and the outcome from the parameters, checked. */
function classesOf(params: Params): { readonly classes: number; readonly y: number; readonly probit: boolean } {
  const classes = numberParam(params, 'classes')
  const y = numberParam(params, 'y')
  if (!(Number.isInteger(classes) && classes >= 2))
    throw new OmaiError('value', 'multiclass-site: classes is an integer of at least 2')
  if (!(Number.isInteger(y) && y >= 0 && y < classes))
    throw new OmaiError('value', `multiclass-site: the class must be an integer from 0 to ${classes - 1}`)
  const link = params['link'] ?? 'logit'
  if (link !== 'logit' && link !== 'probit')
    throw new OmaiError('value', "multiclass-site: the link is 'logit' or 'probit'")
  return { classes, y, probit: link === 'probit' }
}

/** log P(y | z) under the softmax. */
function logSoftmax(z: ArrayLike<number>, y: number): number {
  let top = Number.NEGATIVE_INFINITY
  for (let k = 0; k < z.length; k++) if ((z[k] as number) > top) top = z[k] as number
  let s = 0
  for (let k = 0; k < z.length; k++) s += Math.exp((z[k] as number) - top)
  return (z[y] as number) - top - Math.log(s)
}

const E_NODES = gaussianNodes(0, 1, 20)

/** log P(y | z) under the multinomial probit: the largest of `z_k + e_k`, `e_k` standard normal; one integral over `e_y`. */
function logProbit(z: ArrayLike<number>, y: number): number {
  let total = 0
  for (let i = 0; i < E_NODES.x.length; i++) {
    let p = 1
    for (let j = 0; j < z.length; j++)
      if (j !== y) p *= ndtr((z[y] as number) - (z[j] as number) + (E_NODES.x[i] as number))
    total += (E_NODES.w[i] as number) * p
  }
  return Math.log(total)
}

/** The Cholesky factor of a small symmetric positive-definite matrix (row-major `n x n`), or `undefined`. */
function chol(a: readonly number[], n: number): number[] | undefined {
  const l = new Array<number>(n * n).fill(0)
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j] as number
      for (let k = 0; k < j; k++) s -= (l[i * n + k] as number) * (l[j * n + k] as number)
      if (i === j) {
        if (!(s > 0)) return undefined
        l[i * n + i] = Math.sqrt(s)
      } else l[i * n + j] = s / (l[j * n + j] as number)
    }
  return l
}

interface Tilt {
  readonly logMass: number
  /** The tilted mean and covariance (row-major) of the free scores, in the order of `free`. */
  readonly mean: readonly number[]
  readonly cov: readonly number[]
  readonly free: readonly number[]
}

/**
 * The last integral, kept: the cavities of the scores of one factor are the same for every port (each port is a different variable),
 * so the messages to the `K` ports and the evidence of one sweep share one integral.
 */
let last: { readonly key: string; readonly value: Tilt } | undefined

/** The tilted mean, covariance and log mass of the free scores against the cavities (the integral over normalised cavities). */
function tilt(c: OperatorContext): Tilt {
  const key = JSON.stringify([c.args.map((a) => (a.value !== undefined ? a.value : a.eta)), c.params, c.power])
  if (last?.key === key) return last.value
  const value = integrate(c)
  last = { key, value }
  return value
}

/** The integral itself. */
function integrate(c: OperatorContext): Tilt {
  const { classes, y, probit } = classesOf(c.params)
  const free: number[] = []
  const fixed = new Array<number>(classes).fill(0)
  const cavity: { mean: number; variance: number }[] = []
  for (let k = 0; k < classes; k++) {
    const a = c.args[k] as Arg
    if (a.value !== undefined) fixed[k] = typeof a.value === 'number' ? a.value : Number.NaN
    else {
      free.push(k)
      cavity.push(gaussianOf(a))
    }
  }
  const r = free.length
  const n = NODES[r]
  if (r === 0) {
    const lp = probit ? logProbit(fixed, y) : logSoftmax(fixed, y)
    return { logMass: c.power * lp, mean: [], cov: [], free }
  }
  if (n === undefined)
    throw new OmaiError('value', `multiclass-site: ${r} free scores are more than the six this rule integrates`)
  const grid = gaussianNodes(0, 1, n)
  const z = fixed.slice()
  const logf = probit ? logProbit : logSoftmax
  // The cavity as a Gaussian: log density at a point of the free scores, its constants made once.
  const cavMean = cavity.map((g) => g.mean)
  const cavHalfPrecision = cavity.map((g) => 0.5 / g.variance)
  const cavConstant = -0.5 * cavity.reduce((a, g) => a + Math.log(2 * Math.PI * g.variance), 0)
  const logCav = (x: readonly number[]): number => {
    let s = cavConstant
    for (let i = 0; i < r; i++) s -= ((x[i] as number) - (cavMean[i] as number)) ** 2 * (cavHalfPrecision[i] as number)
    return s
  }
  let mu = cavity.map((g) => g.mean)
  let cov = new Array<number>(r * r).fill(0)
  for (let i = 0; i < r; i++) cov[i * r + i] = (cavity[i] as { variance: number }).variance
  let result: Tilt | undefined
  const total = n ** r
  for (let round = 0; round < ROUNDS; round++) {
    const l = chol(cov, r)
    if (l === undefined) throw new OmaiError('value', 'multiclass-site: the tilted covariance is not positive definite')
    let logDet = 0
    for (let i = 0; i < r; i++) logDet += 2 * Math.log(l[i * r + i] as number)
    const logw = new Float64Array(total)
    const xs = new Float64Array(total * r)
    const t = new Array<number>(r).fill(0)
    const x = new Array<number>(r).fill(0)
    let top = Number.NEGATIVE_INFINITY
    for (let idx = 0; idx < total; idx++) {
      let rest = idx
      let lw = 0
      for (let i = r - 1; i >= 0; i--) {
        const j = rest % n
        rest = (rest - j) / n
        t[i] = grid.x[j] as number
        lw += Math.log(grid.w[j] as number)
      }
      // x = mu + L t; the proposal's log density at x is -r/2 log 2 pi - logdet/2 - |t|^2 / 2.
      let tt = 0
      for (let i = 0; i < r; i++) {
        let s = mu[i] as number
        for (let k = 0; k <= i; k++) s += (l[i * r + k] as number) * (t[k] as number)
        x[i] = s
        xs[idx * r + i] = s
        tt += (t[i] as number) ** 2
        z[free[i] as number] = s
      }
      // The node's weight is w_t f(x) N_cav(x) / N_proposal(x); the Gaussian weights of the product rule already stand for N_proposal.
      const lq = -0.5 * (r * Math.log(2 * Math.PI) + logDet + tt)
      const v = lw + c.power * logf(z, y) + logCav(x) - lq
      logw[idx] = v
      if (v > top) top = v
    }
    if (!Number.isFinite(top))
      throw new OmaiError('value', 'multiclass-site: the factor has no mass under the cavities')
    let sum = 0
    const m = new Array<number>(r).fill(0)
    for (let idx = 0; idx < total; idx++) {
      const w = Math.exp((logw[idx] as number) - top)
      logw[idx] = w
      sum += w
      for (let i = 0; i < r; i++) m[i] = (m[i] as number) + w * (xs[idx * r + i] as number)
    }
    for (let i = 0; i < r; i++) m[i] = (m[i] as number) / sum
    const v = new Array<number>(r * r).fill(0)
    for (let idx = 0; idx < total; idx++) {
      const w = logw[idx] as number
      for (let i = 0; i < r; i++) {
        const di = (xs[idx * r + i] as number) - (m[i] as number)
        for (let k = 0; k <= i; k++)
          v[i * r + k] = (v[i * r + k] as number) + w * di * ((xs[idx * r + k] as number) - (m[k] as number))
      }
    }
    for (let i = 0; i < r; i++)
      for (let k = 0; k <= i; k++) {
        const e = (v[i * r + k] as number) / sum
        v[i * r + k] = e
        v[k * r + i] = e
      }
    const done =
      result !== undefined &&
      m.every((mi, i) => Math.abs(mi - (result?.mean[i] as number)) <= 1e-3 * Math.sqrt(v[i * r + i] as number)) &&
      m.every((_, i) => Math.abs((v[i * r + i] as number) / (result?.cov[i * r + i] as number) - 1) <= 1e-3)
    result = { logMass: top + Math.log(sum), mean: m, cov: v, free }
    mu = m
    cov = v
    if (done) break
  }
  return result as Tilt
}

/** The log probability of the observed class at the scores: the softmax's or the multinomial probit's. */
function scoreLog(params: Params): (z: readonly number[]) => number {
  const { y, probit } = classesOf(params)
  return probit ? (z) => logProbit(z, y) : (z) => logSoftmax(z, y)
}

/**
 * A categorical outcome with softmax (or multinomial probit) probabilities of Gaussian class scores: one port per class, the
 * parameters `y` (the observed class), `classes` (their number) and `link`. Expectation propagation: the tilted mean and variance of
 * every free score from an adaptive Gauss-Hermite integral over all of them, so the scores of the classes that were not observed
 * hear about each other. It supports a power. Variational message passing is non-conjugate (the expected log of a softmax has no
 * closed form in Gaussian scores): the message to a score is the natural gradient of the expectation by the same product rule.
 *
 * @concept factor-operators
 * @index structure A categorical outcome with softmax or multinomial-probit probabilities of Gaussian class scores (adaptive Gauss-Hermite).
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, multiclassSite } from 'omai-engine/message-passing'
 * const args = [0, 1, 2].map(() => messageArg(gaussian, gaussianEta(0, 1)))
 * const m = multiclassSite.ep?.({ mode: 'ep', to: 0, args, params: { y: 0, classes: 3 }, power: 1 })
 * m?.message.map((v) => v.toFixed(3)) // => ['0.000', '0.000']
 * ```
 */
export const multiclassSite: FactorOperator = {
  name: 'multiclass-site',
  description:
    'y ~ Categorical(softmax(z)) (or the multinomial probit) of Gaussian class scores z_0..z_(K-1): an adaptive Gauss-Hermite EP site',
  ports: (params: Params): readonly Port[] =>
    Array.from({ length: Math.max(2, Math.floor(numberParam(params, 'classes', 2))) }, (_, k) => ({
      name: `z${k}`,
      accepts: ['gaussian'],
      constant: true,
    })),
  support: { ep: 'numeric', vmp: 'numeric' },
  readsTarget: true,
  power: true,
  // The mean field: the non-conjugate message of log P(y | z) to a score under the beliefs of the others (the grid is a product over
  // the free scores, with fewer nodes the more there are). The softmax is log-concave in the scores, so the precisions are positive.
  vmp: (c): OperatorResult => ({ message: numericMessage(quadraturePorts(c.args), c.to, scoreLog(c.params)) }),
  expectedLog: (c) => numericExpectedLog(quadraturePorts(c.args), scoreLog(c.params)),
  check: (params) => {
    try {
      classesOf(params)
      return undefined
    } catch (e) {
      return errorText(e)
    }
  },
  ep: (c): OperatorResult => {
    const t = tilt(c)
    const at = t.free.indexOf(c.to)
    if (at < 0) return { message: [0, 0] }
    const own = etaOf(c.args[c.to] as Arg)
    return { message: gaussianSiteMessage(own, t.mean[at] as number, t.cov[at * t.free.length + at] as number) }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    let z = tilt(c).logMass
    for (const a of c.args) z += massOf(a)
    return z
  },
}

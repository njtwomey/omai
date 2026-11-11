/**
 * Plated regression operators: one factor for a whole plate of independent rows of a regression on a Gaussian weight
 * vector, so a model with a thousand or a hundred thousand rows is one factor on one variable, with no variable and no
 * factor per row. The rows are the factor's parameters (a design matrix `X`, row-major, and the outcomes `y`), or for the
 * Gaussian link their sufficient statistics (`X'X`, `X'y`, `y'y`), and the message to the weights is computed in one
 * call over all the rows.
 *
 * The Gaussian link is exact and closed form: the message is `[tau X'y, tau X'X]`, independent of what the other
 * factors say, so its cost does not depend on the number of rows. A nonlinear link (probit, logistic) runs expectation
 * propagation over the plate inside the call: each row keeps a site (a rank-one natural-parameter message on the weights)
 * for the length of the call, the cavity of a row is obtained from the posterior by a rank-one downdate, and the scalar
 * link's own moment match gives the new site, a sweep costing one `d x d` update per row (`O(N d^2)`) and the sweeps
 * running to convergence. The message returned is the product of the sites. The fixed point is the one the expanded graph
 * (a factor and a predictor variable per row) reaches, so the two give the same posterior; the plate reads the data once
 * and the graph does not grow with it. Under variational message passing the message is stateless (the row's expected
 * log factor under the current belief), so a sweep is one pass.
 */

import { OmaiError } from '../../../base/index.ts'
import {
  allProper,
  etaOf,
  ImproperMessageError,
  messageArg,
  numberParam,
  valueArg,
  vectorParam,
} from '../../core/args.ts'
import { bernoulli, gaussian, gaussianEta } from '../../core/families.ts'
import { gaussianMv, gaussianMvMoments } from '../../core/families-vector.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
import { logistic } from '../links/links.ts'
import { probit } from '../links/truncation.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

/** The rows of a plate: the design matrix (row-major, `n x d`) and the outcomes. */
interface Rows {
  readonly n: number
  readonly d: number
  readonly x: readonly number[]
  readonly y: readonly number[]
}

function rowsOf(params: OperatorContext['params'], d: number): Rows {
  const x = vectorParam(params, 'X')
  const y = vectorParam(params, 'y')
  if (x.length !== y.length * d)
    throw new OmaiError(
      'value',
      `a plate: X must have as many entries as outcomes times the weights, ${y.length} x ${d}`,
    )
  return { n: y.length, d, x, y }
}

/** The mean and covariance of a Gaussian-vector message as a mean array and a row-major covariance. */
function momentsOf(eta: Eta, d: number): { mean: Float64Array; covariance: Float64Array } {
  const { mean, covariance } = gaussianMvMoments(eta, d)
  return { mean: Float64Array.from(mean), covariance: Float64Array.from(covariance) }
}

/** The row `i` of the design matrix. */
const rowAt = (r: Rows, i: number): readonly number[] => r.x.slice(i * r.d, (i + 1) * r.d)

/** Sigma a, and a' Sigma a, for the row `a` (`Sigma` row-major `d x d`). */
function quadratic(sigma: Float64Array, a: readonly number[], out: Float64Array): number {
  const d = a.length
  let v = 0
  for (let r = 0; r < d; r++) {
    let s = 0
    for (let k = 0; k < d; k++) s += (sigma[r * d + k] ?? 0) * (a[k] ?? 0)
    out[r] = s
    v += s * (a[r] ?? 0)
  }
  return v
}

const dot = (a: readonly number[], b: ArrayLike<number>): number => {
  let s = 0
  for (let k = 0; k < a.length; k++) s += (a[k] ?? 0) * (b[k] ?? 0)
  return s
}

/** The message `[sum nu_i a_i, sum lambda_i a_i a_i']` of the sites of the rows. */
function sitesMessage(r: Rows, nu: ArrayLike<number>, lam: ArrayLike<number>): Eta {
  const { d } = r
  const out = new Array<number>(d + d * d).fill(0)
  for (let i = 0; i < r.n; i++) {
    const n = nu[i] ?? 0
    const l = lam[i] ?? 0
    if (n === 0 && l === 0) continue
    const base = i * d
    for (let a = 0; a < d; a++) {
      const xa = r.x[base + a] ?? 0
      out[a] = (out[a] ?? 0) + n * xa
      for (let b = 0; b < d; b++) out[d + a * d + b] = (out[d + a * d + b] ?? 0) + l * xa * (r.x[base + b] ?? 0)
    }
  }
  return out
}

/** The posterior `[h, J]` of the weights as a mean and covariance, from the cavity and the sites. */
function posteriorOf(cavity: Eta, message: Eta, d: number): { mean: Float64Array; covariance: Float64Array; eta: Eta } {
  const eta = cavity.map((v, i) => v + (message[i] ?? 0))
  const factor = spd(eta.slice(d), d)
  if (factor === undefined) throw new ImproperMessageError('a plate: the posterior of the weights is not proper')
  return { mean: factor.solve(eta.slice(0, d)), covariance: factor.inverse(), eta }
}

/** The settings of the inner expectation propagation of a plate. */
const INNER_SWEEPS = 100
const INNER_TOL = 1e-11

interface Plate {
  readonly nu: Float64Array
  readonly lam: Float64Array
  readonly message: Eta
}

/**
 * Expectation propagation over the rows of a plate against a cavity of the weights: the sites of the rows on the
 * predictors `a_i . w`, swept in order until none moves, each from the cavity of its row (the posterior with its own site
 * taken out by a rank-one downdate) and the scalar link's moment match.
 */
function plateEp(link: FactorOperator, c: OperatorContext, r: Rows): Plate {
  const d = r.d
  const cavity = etaOf(arg(c, 0))
  const nu = new Float64Array(r.n)
  const lam = new Float64Array(r.n)
  const sa = new Float64Array(d)
  const siteContext = (y: number, h: number, p: number): OperatorContext => ({
    mode: 'ep',
    to: 1,
    args: [valueArg(bernoulli, y), messageArg(gaussian, [h, p])],
    params: c.params,
    power: 1,
  })
  for (let sweep = 0; sweep < INNER_SWEEPS; sweep++) {
    // The posterior from scratch each sweep, so the rank-one updates of the last one do not accumulate rounding.
    const { mean, covariance } = posteriorOf(cavity, sitesMessage(r, nu, lam), d)
    let change = 0
    for (let i = 0; i < r.n; i++) {
      const a = rowAt(r, i)
      const v = quadratic(covariance, a, sa)
      const m = dot(a, mean)
      const precision = 1 / v - (lam[i] ?? 0)
      if (!(v > 0 && precision > 0)) continue
      let site: Eta
      try {
        site = (link.ep as NonNullable<FactorOperator['ep']>)(
          siteContext(r.y[i] as number, m / v - (nu[i] ?? 0), precision),
        ).message
      } catch (e) {
        if (e instanceof OmaiError && e.code === 'value') continue
        throw e
      }
      const dnu = (site[0] ?? 0) - (nu[i] ?? 0)
      const dlam = (site[1] ?? 0) - (lam[i] ?? 0)
      const denominator = 1 + dlam * v
      if (!(denominator > 0)) continue
      const k = dlam / denominator
      const coefficient = dnu - k * m - k * dnu * v
      for (let p = 0; p < d; p++) {
        mean[p] = (mean[p] ?? 0) + coefficient * (sa[p] ?? 0)
        for (let q = 0; q < d; q++)
          covariance[p * d + q] = (covariance[p * d + q] ?? 0) - k * (sa[p] ?? 0) * (sa[q] ?? 0)
      }
      nu[i] = site[0] ?? 0
      lam[i] = site[1] ?? 0
      change = Math.max(
        change,
        Math.abs(dnu) / (1 + Math.abs(nu[i] ?? 0)),
        Math.abs(dlam) / (1 + Math.abs(lam[i] ?? 0)),
      )
    }
    if (change < INNER_TOL) break
  }
  return { nu, lam, message: sitesMessage(r, nu, lam) }
}

/** The log mass of the plate against the raw cavity, from converged sites (the EP energy of the rows, see the module). */
function plateEvidence(link: FactorOperator, c: OperatorContext, r: Rows): number {
  if (!allProper(c.args) || link.logEvidence === undefined) return Number.NaN
  const d = r.d
  const { nu, lam, message } = plateEp(link, c, r)
  let post: ReturnType<typeof posteriorOf>
  try {
    post = posteriorOf(etaOf(arg(c, 0)), message, d)
  } catch (e) {
    if (e instanceof ImproperMessageError) return Number.NaN
    throw e
  }
  const sa = new Float64Array(d)
  let total = gaussianMv(d).logPartition(post.eta)
  for (let i = 0; i < r.n; i++) {
    const a = rowAt(r, i)
    const v = quadratic(post.covariance, a, sa)
    const m = dot(a, post.mean)
    const precision = 1 / v - (lam[i] ?? 0)
    if (!(v > 0 && precision > 0)) return Number.NaN
    const cavityZ: Eta = [m / v - (nu[i] ?? 0), precision]
    const row = link.logEvidence({
      mode: 'ep',
      to: 1,
      args: [valueArg(bernoulli, r.y[i] as number), messageArg(gaussian, cavityZ)],
      params: c.params,
      power: 1,
    })
    total += row - gaussian.logPartition([m / v, 1 / v])
  }
  return total
}

/** The variational message of the plate: the scalar link's message to each predictor under the belief, summed. */
function plateVariational(link: FactorOperator, c: OperatorContext, r: Rows): Eta {
  const d = r.d
  const { mean, covariance } = momentsOf(etaOf(arg(c, 0)), d)
  const sa = new Float64Array(d)
  const nu = new Float64Array(r.n)
  const lam = new Float64Array(r.n)
  for (let i = 0; i < r.n; i++) {
    const a = rowAt(r, i)
    const v = quadratic(covariance, a, sa)
    const m = dot(a, mean)
    const site = (link.vmp as NonNullable<FactorOperator['vmp']>)({
      mode: 'vmp',
      to: 1,
      args: [valueArg(bernoulli, r.y[i] as number), messageArg(gaussian, gaussianEta(m, v))],
      params: c.params,
      power: 1,
    }).message
    nu[i] = site[0] ?? 0
    lam[i] = site[1] ?? 0
  }
  return sitesMessage(r, nu, lam)
}

/** The expected log factor of the plate under the belief: the sum over the rows of the scalar link's. */
function plateExpectedLog(link: FactorOperator, c: OperatorContext, r: Rows): number {
  const d = r.d
  const { mean, covariance } = momentsOf(etaOf(arg(c, 0)), d)
  const sa = new Float64Array(d)
  let total = 0
  for (let i = 0; i < r.n; i++) {
    const a = rowAt(r, i)
    const v = quadratic(covariance, a, sa)
    const m = dot(a, mean)
    total += (link.expectedLog as NonNullable<FactorOperator['expectedLog']>)({
      mode: 'vmp',
      to: 1,
      args: [valueArg(bernoulli, r.y[i] as number), messageArg(gaussian, gaussianEta(m, v))],
      params: c.params,
      power: 1,
    })
  }
  return total
}

/**
 * The plate of a scalar link on a Gaussian weight vector: `y_i | a_i . w ~ link` for every row of a design matrix, as one
 * factor on `w`. The parameters are `X` (the design matrix, row-major) and `y` (the outcomes, one per row), with the
 * link's own (`noise` of the probit). Under expectation propagation the message is the product of the rows' sites after
 * the inner propagation of the module summary; under variational message passing it is the sum of the link's messages
 * to the predictors under the belief; the evidence and the bound are the sums over the rows.
 *
 * @param name - The operator's name.
 * @param link - The scalar link (ports `[outcome, predictor]`, an observed outcome and a Gaussian predictor).
 * @param description - The operator's description.
 * @returns The operator.
 * @throws OmaiError with code `value` when the link has no EP rule to run over the rows.
 * @concept factor-operators
 * @index function Lift a scalar link to a plate of rows: one factor on the weights, vectorised.
 * @example
 * ```ts
 * import { glmPlate, gaussianMv, gaussianMvEta, messageArg, probit } from 'omai-engine/message-passing'
 * const plate = glmPlate('my-plate', probit, 'a probit plate')
 * const w = gaussianMv(2)
 * const args = [messageArg(w, gaussianMvEta([0, 0], [1, 0, 0, 1]))]
 * const params = { X: [1, 0, 0, 1], y: [1, 0] }
 * plate.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message.length // => 6
 * ```
 */
export function glmPlate(name: string, link: FactorOperator, description: string): FactorOperator {
  if (link.ep === undefined) throw new OmaiError('value', `glmPlate: ${link.name} has no EP rule to lift`)
  return {
    name,
    description,
    ports: [{ name: 'w', accepts: ['gaussian-mv'] }],
    support: {
      ep: link.support.ep ?? 'numeric',
      ...(link.vmp === undefined ? {} : { vmp: link.support.vmp ?? 'exact' }),
    },
    readsTarget: true,
    check: (params, families) => {
      const d = families[0]?.dim ?? 0
      const x = params['X']
      const y = params['y']
      return Array.isArray(x) && Array.isArray(y) && x.length === y.length * d
        ? undefined
        : `${name}: X must have as many entries as outcomes times the weights`
    },
    ep: (c): OperatorResult => ({
      message: plateEp(link, c, rowsOf(c.params, arg(c, 0).family.dim)).message,
    }),
    ...(link.vmp === undefined
      ? {}
      : {
          vmp: (c: OperatorContext): OperatorResult => ({
            message: plateVariational(link, c, rowsOf(c.params, arg(c, 0).family.dim)),
          }),
        }),
    logEvidence: (c) => plateEvidence(link, c, rowsOf(c.params, arg(c, 0).family.dim)),
    ...(link.expectedLog === undefined
      ? {}
      : { expectedLog: (c: OperatorContext) => plateExpectedLog(link, c, rowsOf(c.params, arg(c, 0).family.dim)) }),
  }
}

/**
 * A plate of probit regression rows, `y_i | a_i . w ~ Bernoulli(Phi(a_i . w / noise))`, as one factor on the weights.
 *
 * @concept factor-operators
 * @index structure A plate of probit regression rows on a Gaussian weight vector, one vectorised factor.
 */
export const probitPlate: FactorOperator = glmPlate('probit-plate', probit, 'a plate of probit rows on w')

/**
 * A plate of logistic regression rows, `y_i | a_i . w ~ Bernoulli(sigmoid(a_i . w))`, as one factor on the weights.
 *
 * @concept factor-operators
 * @index structure A plate of logistic regression rows on a Gaussian weight vector, one vectorised factor.
 */
export const logisticPlate: FactorOperator = glmPlate('logistic-plate', logistic, 'a plate of logistic rows on w')

/** The sufficient statistics of a Gaussian regression plate: `X'X`, `X'y`, `y'y`. */
function statistics(c: OperatorContext): {
  xtx: readonly number[]
  xty: readonly number[]
  yty: number
  n: number
} {
  return {
    xtx: vectorParam(c.params, 'xtx'),
    xty: vectorParam(c.params, 'xty'),
    yty: numberParam(c.params, 'yty'),
    n: numberParam(c.params, 'n'),
  }
}

/**
 * A plate of Gaussian regression rows, `y_i ~ N(a_i . w, 1 / precision)`, as one factor on the weights, given by its
 * sufficient statistics: `xtx` (`X'X`, row-major), `xty` (`X'y`), `yty` (`y'y`) and `n` (the number of rows), and
 * `precision`. The message to the weights is exact and the same for every cavity, `[precision X'y, precision X'X]`; the
 * evidence and the bound are closed forms. Nothing in it depends on the number of rows but the four numbers it was made of.
 *
 * @concept factor-operators
 * @index structure A plate of Gaussian regression rows given by sufficient statistics, exact.
 * @example
 * ```ts
 * import { gaussianMv, linearRegressionPlate, messageArg } from 'omai-engine/message-passing'
 * const w = gaussianMv(2)
 * const args = [messageArg(w, w.flat)]
 * const params = { xtx: [2, 1, 1, 5], xty: [3, 4], yty: 7, n: 2, precision: 2 }
 * linearRegressionPlate.ep?.({ mode: 'ep', to: 0, args, params, power: 1 }).message // => [6, 8, 4, 2, 2, 10]
 * ```
 */
export const linearRegressionPlate: FactorOperator = {
  name: 'linear-regression-plate',
  description: 'a plate of rows y_i ~ N(a_i . w, 1 / precision), by sufficient statistics',
  ports: [{ name: 'w', accepts: ['gaussian-mv'] }],
  support: { ep: 'exact', vmp: 'exact' },
  check: (params, families) => {
    const d = families[0]?.dim ?? 0
    return vectorParam(params, 'xtx').length === d * d && vectorParam(params, 'xty').length === d
      ? undefined
      : 'linear-regression-plate: xtx must be d x d and xty of length d for d weights'
  },
  ep: (c): OperatorResult => {
    const s = statistics(c)
    const tau = numberParam(c.params, 'precision')
    return { message: [...s.xty.map((v) => tau * v), ...s.xtx.map((v) => tau * v)] }
  },
  vmp: (c): OperatorResult => {
    const s = statistics(c)
    const tau = numberParam(c.params, 'precision')
    return { message: [...s.xty.map((v) => tau * v), ...s.xtx.map((v) => tau * v)] }
  },
  logEvidence: (c) => {
    if (!allProper(c.args)) return Number.NaN
    const s = statistics(c)
    const tau = numberParam(c.params, 'precision')
    const d = s.xty.length
    const eta = etaOf(arg(c, 0))
    const post = eta.map((v, i) => v + tau * (i < d ? (s.xty[i] ?? 0) : (s.xtx[i - d] ?? 0)))
    return 0.5 * s.n * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * s.yty + gaussianMv(d).logPartition(post)
  },
  expectedLog: (c) => {
    const s = statistics(c)
    const tau = numberParam(c.params, 'precision')
    const d = s.xty.length
    const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 0)), d)
    let cross = 0
    let quad = 0
    for (let i = 0; i < d; i++) {
      cross += (s.xty[i] ?? 0) * (mean[i] ?? 0)
      for (let j = 0; j < d; j++)
        quad += (s.xtx[i * d + j] ?? 0) * ((covariance[i * d + j] ?? 0) + (mean[i] ?? 0) * (mean[j] ?? 0))
    }
    return 0.5 * s.n * Math.log(tau / (2 * Math.PI)) - 0.5 * tau * (s.yty - 2 * cross + quad)
  },
}

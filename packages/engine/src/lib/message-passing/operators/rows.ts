/**
 * Plated row operators: a scalar link (the probit or logistic outcome of a Gaussian linear predictor) lifted to a
 * Gaussian weight vector, so one factor stands for a whole row of a regression, `y | a . w ~ link`. The message to the
 * weights is rank one, `[nu a, lambda a a']`, with `lambda` and `nu` the site that the scalar link sends to the predictor
 * `a . w` (whose cavity is the Gaussian `N(a . mean, a' covariance a)`), so the factor needs the cavity's moments and no
 * latent predictor per row. A graph of a thousand rows is a thousand factors on one variable, which is the shape the
 * stochastic and averaged EP of `stochastic.ts` runs on, and what the compiler of a model with a plate will emit.
 */

import { OmaiError } from '../../base/index.ts'
import { gaussian, gaussianEta } from '../families.ts'
import { gaussianMv, gaussianMvMoments } from '../families-vector.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { allProper, etaOf, messageArg, vectorParam } from './args.ts'
import { logistic } from './links.ts'
import { probit } from './truncation.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

/** The predictor's cavity: the mean and variance of a . w under the weights' Gaussian message. */
function predictor(c: OperatorContext, a: readonly number[]): { mean: number; variance: number } {
  const d = a.length
  const { mean, covariance } = gaussianMvMoments(etaOf(arg(c, 1)), d)
  let m = 0
  let v = 0
  for (let i = 0; i < d; i++) {
    m += (a[i] ?? 0) * (mean[i] ?? 0)
    let row = 0
    for (let j = 0; j < d; j++) row += (covariance[i * d + j] ?? 0) * (a[j] ?? 0)
    v += (a[i] ?? 0) * row
  }
  return { mean: m, variance: v }
}

/**
 * A link factor `link(y, z)` with `z = a . w` as a factor of the weight vector w: the lift of a scalar link operator
 * (any with ports `[outcome, z]`, an observed outcome and a Gaussian predictor) to a Gaussian vector of weights. The
 * parameters are `a` (the row) and the link's own (`noise` of the probit). The message to w is the link's message to the
 * predictor, put back on w as a rank-one natural parameter; its power and its evidence are the link's.
 *
 * @param name - The operator's name.
 * @param link - The scalar link.
 * @param description - The operator's description.
 * @returns The operator.
 * @concept factor-operators
 * @index function Lift a scalar link operator (probit, logistic) to a row of a regression on a weight vector.
 * @example
 * ```ts
 * import { gaussianMv, gaussianMvEta, glmRow, messageArg, probit, valueArg, bernoulli } from 'omai-engine/message-passing'
 * const row = glmRow('my-row', probit, 'a probit row')
 * const w = gaussianMv(2)
 * const args = [valueArg(bernoulli, 1), messageArg(w, gaussianMvEta([0, 0], [1, 0, 0, 1]))]
 * row.ep?.({ mode: 'ep', to: 1, args, params: { a: [1, 0] }, power: 1 }).message.map((v) => v.toFixed(4)) // => ['0.8276', '0.0000', '0.4669', '0.0000', '0.0000', '0.0000']
 * ```
 */
export function glmRow(name: string, link: FactorOperator, description: string): FactorOperator {
  if (link.ep === undefined) throw new OmaiError('value', `glmRow: ${link.name} has no EP rule to lift`)
  const lift = (c: OperatorContext): { z: Arg; a: readonly number[] } => {
    const a = vectorParam(c.params, 'a')
    const z = predictor(c, a)
    return { z: messageArg(gaussian, gaussianEta(z.mean, z.variance)), a }
  }
  return {
    name,
    description,
    ports: [
      { name: 'y', accepts: ['bernoulli'], constant: true, observed: true },
      { name: 'w', accepts: ['gaussian-mv'] },
    ],
    support: { ep: link.support.ep ?? 'numeric' },
    readsTarget: (to) => to === 1,
    ...(link.power === true ? { power: true } : {}),
    check: (params, families) =>
      vectorParam(params, 'a').length === families[1]?.dim ? undefined : `${name}: a must have the length of w`,
    ep: (c): OperatorResult => {
      if (c.to !== 1) throw new OmaiError('value', `${name}: nothing is sent to the observed outcome`)
      const { z, a } = lift(c)
      const site = (link.ep as NonNullable<FactorOperator['ep']>)({ ...c, to: 1, args: [arg(c, 0), z] }).message
      const nu = site[0] ?? 0
      const lambda = site[1] ?? 0
      const d = a.length
      const message: Eta = [
        ...a.map((x) => nu * x),
        ...Array.from({ length: d * d }, (_, k) => lambda * (a[Math.floor(k / d)] ?? 0) * (a[k % d] ?? 0)),
      ]
      return { message }
    },
    logEvidence: (c) => {
      if (!allProper(c.args) || link.logEvidence === undefined) return Number.NaN
      const { z, a } = lift(c)
      const w = gaussianMv(a.length)
      // The link's value counts the mass of the predictor's message (a normalised Gaussian here); the weights' own mass replaces it.
      const normalised = gaussian.logPartition(etaOf(z))
      return link.logEvidence({ ...c, args: [arg(c, 0), z] }) - normalised + w.logPartition(etaOf(arg(c, 1)))
    },
  }
}

/**
 * One row of probit regression: `y | a . w ~ Bernoulli(Phi(a . w / noise))` on a Gaussian weight vector, with the
 * exact moment match of the probit link (and a quadrature one under a power).
 *
 * @concept factor-operators
 * @index structure One probit regression row on a Gaussian weight vector, exact EP.
 */
export const probitRow: FactorOperator = glmRow('probit-row', probit, 'y | a . w ~ Bernoulli(Phi(a . w))')

/**
 * One row of logistic regression: `y | a . w ~ Bernoulli(sigmoid(a . w))` on a Gaussian weight vector, by the
 * quadrature moment match of the logistic link.
 *
 * @concept factor-operators
 * @index structure One logistic regression row on a Gaussian weight vector, quadrature EP.
 */
export const logisticRow: FactorOperator = glmRow('logistic-row', logistic, 'y | a . w ~ Bernoulli(sigmoid(a . w))')

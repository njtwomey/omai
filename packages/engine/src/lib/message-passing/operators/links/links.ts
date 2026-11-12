/**
 * The logistic link: a Bernoulli outcome whose log odds is a Gaussian variable. EP is numerical (the moments of a
 * Gaussian times a logistic function by Gauss-Hermite quadrature); variational message passing uses Jaakkola and
 * Jordan's quadratic lower bound on the log logistic, with its variational parameter set from the current belief.
 */

import { logAddExp, sigmoidOf } from '../../../base/index.ts'
import { arg } from '../../core/_messages.ts'
import { allProper, etaOf, gaussianOf, gaussianSiteMessage, massOf, scalarOf } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { bernoulli } from '../../core/families.ts'
import type { Arg, FactorOperator, OperatorResult } from '../../core/types.ts'
import { gaussianExpectation, tiltedGaussian, tiltedLogGrid } from '../../numerics/_quadrature.ts'
import { logSigmoidOf } from '../../numerics/_scalar.ts'

function trueProbability(a: Arg): number {
  if (a.value !== undefined) return scalarOf(a) > 0 ? 1 : 0
  return bernoulli.moments(etaOf(a))[0] ?? 0.5
}

/** Jaakkola and Jordan's lambda(xi) = (sigmoid(xi) - 1/2) / (2 xi), with its limit 1/8 at zero. */
const jaakkolaLambda = (xi: number): number => (xi < 1e-6 ? 0.125 : (sigmoidOf(xi) - 0.5) / (2 * xi))

/** E[g] of the logistic outcome's weight over a Gaussian: pi sigmoid(x) + (1 - pi) sigmoid(-x). */
const weight =
  (pi: number) =>
  (x: number): number =>
    pi * sigmoidOf(x) + (1 - pi) * sigmoidOf(-x)

const POINTS = 64

/** log sigmoid, stable in both tails. */

/** log of the weight the logistic factor to the power p gives x when the outcome is true with probability pi. */
const logLogisticWeight = (pi: number, x: number, p: number): number =>
  logAddExp(Math.log(pi) + p * logSigmoidOf(x), Math.log(1 - pi) + p * logSigmoidOf(-x))

/**
 * b | x ~ Bernoulli(sigmoid(x)): the logistic link of logistic regression. EP matches the moments of the Gaussian times
 * the logistic function by quadrature (a numeric rule); VMP uses the Jaakkola bound (`J = 2 lambda(xi)`, `h = (2 P(b) - 1) / 2`
 * with `xi^2 = E[x^2]`), whose message to b is exactly `E[x]`.
 *
 * @concept factor-operators
 * @index structure The logistic link b | x ~ Bernoulli(sigmoid(x)): quadrature EP, Jaakkola bound VMP.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, logistic, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 1), messageArg(gaussian, gaussianEta(0, 1))]
 * logistic.ep?.({ mode: 'ep', to: 1, args, params: {}, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.498', '0.206']
 * ```
 */
export const logistic: FactorOperator = defineOperator({
  name: 'logistic',
  description: 'b | x ~ Bernoulli(sigmoid(x))',
  ports: [
    { name: 'b', accepts: ['bernoulli'], constant: true },
    { name: 'x', accepts: ['gaussian'] },
  ],
  params: {},
  rules: {
    ep: messageRule((c): OperatorResult => {
      const x = gaussianOf(arg(c, 1))
      if (c.power !== 1) {
        const p = c.power
        if (c.to === 0) {
          const yes = tiltedLogGrid(x.mean, x.variance, (v) => p * logSigmoidOf(v)).logMass
          const no = tiltedLogGrid(x.mean, x.variance, (v) => p * logSigmoidOf(-v)).logMass
          return { message: [(yes - no) / p] }
        }
        const pi = trueProbability(arg(c, 0))
        const t = tiltedLogGrid(x.mean, x.variance, (v) => logLogisticWeight(pi, v, p))
        const site = gaussianSiteMessage(etaOf(arg(c, 1)), t.mean, t.variance)
        return { message: [(site[0] ?? 0) / p, (site[1] ?? 0) / p] }
      }
      if (c.to === 0) {
        const yes = gaussianExpectation(x.mean, x.variance, sigmoidOf, POINTS)
        const no = gaussianExpectation(x.mean, x.variance, (v) => sigmoidOf(-v), POINTS)
        return { message: [Math.log(yes) - Math.log(no)] }
      }
      const t = tiltedGaussian(x.mean, x.variance, weight(trueProbability(arg(c, 0))), POINTS)
      return { message: gaussianSiteMessage(etaOf(arg(c, 1)), t.mean, t.variance) }
    }, 'numeric'),
    vmp: messageRule((c): OperatorResult => {
      const x = gaussianOf(arg(c, 1))
      if (c.to === 0) return { message: [x.mean] }
      const xi = Math.sqrt(x.variance + x.mean * x.mean)
      return { message: [(2 * trueProbability(arg(c, 0)) - 1) / 2, 2 * jaakkolaLambda(xi)] }
    }, 'bound'),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const x = gaussianOf(arg(c, 1))
      const b = arg(c, 0)
      if (c.power !== 1) {
        const p = c.power
        const pi = trueProbability(b)
        return (
          massOf(arg(c, 1)) +
          (b.value === undefined ? massOf(b) : 0) +
          tiltedLogGrid(x.mean, x.variance, (v) => logLogisticWeight(pi, v, p)).logMass
        )
      }
      if (b.value !== undefined) {
        const g = scalarOf(b) > 0 ? sigmoidOf : (v: number) => sigmoidOf(-v)
        return massOf(arg(c, 1)) + Math.log(gaussianExpectation(x.mean, x.variance, g, POINTS))
      }
      const pi = trueProbability(b)
      return massOf(arg(c, 1)) + massOf(b) + Math.log(gaussianExpectation(x.mean, x.variance, weight(pi), POINTS))
    },
    vmp: (c) => {
      const x = gaussianOf(arg(c, 1))
      const xi = Math.sqrt(x.variance + x.mean * x.mean)
      const pi = trueProbability(arg(c, 0))
      return Math.log(sigmoidOf(xi)) - xi / 2 + ((2 * pi - 1) * x.mean) / 2
    },
  },
  traits: { readsTarget: (to) => to === 1, power: true },
})

/**
 * An observation with a random log scale and a random mean: `x ~ N(m, exp(2 s))` with the value `x` observed, the mean `m`
 * a Gaussian variable (or known) and the log scale `s` a Gaussian variable. The unknown spread of a hierarchy's noise
 * (the `sigma` of a regression, with a half-normal or half-Cauchy prior carried on its log) in one factor per observation.
 * The mean integrates out in closed form (`N(x; m, v + exp(2 s))` against the cavity `N(m, v)`), so every message is a
 * one-dimensional quadrature over the log scale, and the message to the mean is the mixture of the conditional Gaussians
 * over it; the work does not grow with the number of random arguments.
 */

import { arg } from '../../core/_messages.ts'
import { allProper, gaussianOf, logNormal, massOf, scalarOf } from '../../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../../core/define.ts'
import type { FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { tiltedByQuadrature } from '../../numerics/_tilted.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

/** The value, the cavity of the mean (variance 0 for a known mean) and the cavity of the log scale. */
function read(c: OperatorContext): {
  x: number
  m: { mean: number; variance: number }
  s: { mean: number; variance: number }
} {
  return { x: scalarOf(arg(c, 0)), m: gaussianOf(arg(c, 1)), s: gaussianOf(arg(c, 2)) }
}

/** The tilted distribution of the log scale: its cavity times the marginal likelihood `N(x; mean, v + exp(2 s))`. */
function logLikelihood(x: number, m: { mean: number; variance: number }, power: number): (s: number) => number {
  return (s) => {
    const total = m.variance + Math.exp(2 * s)
    return Number.isFinite(total) && total > 0 ? power * logNormal(x, m.mean, total) : Number.NEGATIVE_INFINITY
  }
}

/** log N(x; m, exp(2 s)) at the coordinates (x, m, s): the density is log-concave in the mean and in the log scale. */
const logScaleDensity = (v: readonly number[]): number =>
  -(v[2] as number) -
  0.5 * Math.log(2 * Math.PI) -
  0.5 * ((v[0] as number) - (v[1] as number)) ** 2 * Math.exp(-2 * (v[2] as number))

/**
 * x ~ N(m, exp(2 s)) for an observed x, a Gaussian mean m (or a known one) and a Gaussian log scale s. EP integrates the
 * mean out in closed form and the log scale by a quadrature that zooms onto where the tilted mass is: the message to the
 * log scale is its moment match, and the message to the mean is the moment match of the mixture of conditional Gaussians
 * over the tilted log scale; the moments are checked against a two-dimensional brute-force grid. Variational message passing is
 * non-conjugate (the expected log is `-E s - E[(x - m)^2] E[exp(-2 s)] / 2`, closed form, but `exp(-2 s)` is not a statistic of a
 * Gaussian): the message to the log scale is the natural gradient of that expectation, to the mean the exact conjugate one.
 *
 * @concept factor-operators
 * @index structure An observation with a Gaussian mean and a Gaussian log scale, x ~ N(m, exp(2 s)): EP by one quadrature.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianScale, messageArg, valueArg } from 'omai-engine/message-passing'
 * const args = [valueArg(gaussian, 2), messageArg(gaussian, gaussianEta(0, 1)), messageArg(gaussian, gaussianEta(0, 1))]
 * gaussianScale.ep?.({ mode: 'ep', to: 2, args, params: {}, power: 1 }).message.map((v) => v.toFixed(3)) // => ['0.151', '0.341']
 * ```
 */
export const gaussianScale: FactorOperator = defineOperator({
  name: 'gaussian-scale',
  description: 'x ~ N(m, exp(2 s)) with a Gaussian mean m and log scale s and an observed x',
  ports: [
    { name: 'x', accepts: ['gaussian'], constant: true, observed: true },
    { name: 'mean', accepts: ['gaussian'], constant: true },
    { name: 'logScale', accepts: ['gaussian'] },
  ],
  params: {},
  rules: {
    ep: siteRule((c) => {
      const { x, m, s } = read(c)
      const logf = logLikelihood(x, m, c.power)
      if (c.to === 2) return tiltedByQuadrature(s, logf)
      // The mean given the log scale is Gaussian: N(m, v) times N(x; m, exp(2 s)).
      const post = (s2: number): { mean: number; variance: number } => {
        const noise = Math.exp(-2 * s2)
        const variance = 1 / (1 / m.variance + noise)
        return { mean: variance * (m.mean / m.variance + x * noise), variance }
      }
      // The mixture of the conditionals over the tilted log scale: the mean of the variances plus the variance of the means.
      const t = tiltedByQuadrature(s, logf, undefined, undefined, [(s2) => post(s2).mean, (s2) => post(s2).variance])
      const [conditionalMean, conditionalVariance] = t.extra
      return {
        mean: conditionalMean?.mean ?? 0,
        variance: Math.max((conditionalVariance?.mean ?? 0) + (conditionalMean?.variance ?? 0), 1e-300),
      }
    }),
    vmp: messageRule(
      (c): OperatorResult => ({ message: numericMessage(quadraturePorts(c.args), c.to, logScaleDensity) }),
      'numeric',
    ),
  },
  evidence: {
    ep: (c) => {
      if (!allProper(c.args)) return Number.NaN
      const { x, m, s } = read(c)
      return massOf(arg(c, 1)) + massOf(arg(c, 2)) + tiltedByQuadrature(s, logLikelihood(x, m, c.power)).logMass
    },
    vmp: (c) => numericExpectedLog(quadraturePorts(c.args), logScaleDensity),
  },
  traits: { readsTarget: (to) => to !== 0, power: true },
})

/**
 * A bivariate Gaussian whose covariance is given by two standard deviations and a correlation, each a random variable of its own:
 * `x ~ N(mean, Sigma)` with `Sigma = [[1 / t1, rho / sqrt(t1 t2)], [rho / sqrt(t1 t2), 1 / t2]]`, the precisions `t1`, `t2` gamma variables and the
 * correlation `rho = 2 u - 1` for a beta variable `u` (a uniform correlation is `Beta(1, 1)`, the LKJ prior of a 2 by 2 matrix with
 * concentration `eta` is `Beta(eta, eta)`). The vector and the mean are Gaussian (or observed), integrated out in closed form given
 * the three; expectation propagation matches the gammas and the beta to the moments of a panelled three-dimensional grid over their
 * coordinates (`_grid.ts`: the log of each precision, the logit of `u`), and the Gaussians to the mixture of their conditionals over
 * it. A known precision or correlation is a single node of its axis, so a Gaussian with known standard deviations and an unknown
 * correlation is a one-dimensional integral.
 *
 * The beta is matched to the tilted mean and variance of `u` by default (`match: 'moments'`), not to `E log u` and `E log (1 - u)` (the KL
 * projection, `match: 'log-moments'`): a strong correlation from few pairs piles against its bound with a long tail toward the interior,
 * which the log moments weigh by the ratio to the bound and not by the distance, so the beta that matches them came out 0.45 to 0.95 of the
 * exact spread (at rho 0.99 from five pairs about half). With a single factor on the correlation (a plated likelihood and its beta prior)
 * the cavity is the prior itself, the tilted distribution is the exact posterior, and the belief has its mean and spread exactly.
 */

import { arg } from '../../core/_messages.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { gaussianMvEta } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { integrateAxes } from '../../numerics/_grid.ts'
import { axisOfPort, parameterMessage, precisionAt } from './_sdcorr.ts'
import { evaluatorOf, gaussianSide, type Rows, rowsOf } from './_sides.ts'

/** The integral over the three coordinates for the current arguments, and what the messages read from it. */
function tilt(c: OperatorContext, rows: Rows) {
  const d = 2
  const x = gaussianSide(arg(c, 0), d)
  const mean = gaussianSide(arg(c, 1), d)
  const eval1 = evaluatorOf(x, mean, d, rows)
  const axes = [
    axisOfPort(arg(c, 2), 'precision'),
    axisOfPort(arg(c, 3), 'precision'),
    axisOfPort(arg(c, 4), 'correlation'),
  ] as const
  const lambda = new Float64Array(4)
  const grid = integrateAxes(axes, (t) => evalAt(eval1, lambda, t))
  return { x, mean, eval1, axes, grid }
}

/** The log of the Gaussian part at the coordinates (the matrix is rebuilt into `lambda`). */
function evalAt(eval1: ReturnType<typeof evaluatorOf>, lambda: Float64Array, t: Float64Array): number {
  const logDet = precisionAt(t, lambda)
  return eval1(lambda, 0, logDet, -1)
}

/**
 * `x ~ N(mean, Sigma)` with `Sigma` built from two precisions (gamma variables) and a correlation `2 u - 1` (a beta variable `u`): the bivariate
 * normal with unknown standard deviations and correlation. The five ports are the vector, its mean, the two precisions and the beta; each of
 * the last three may be known. Like `mvn-wishart` it may stand for `count` i.i.d. rows (the parameters `count` and `scatter`, `x` their sample
 * mean). Expectation propagation integrates the three coordinates on a zooming grid and sends gammas, a beta, and Gaussians matched to the
 * moments (the parameter `match` chooses the beta's: its mean and variance, or its log moments); there is no variational message (the expected
 * log density has no closed form in the beta).
 *
 * @concept factor-operators
 * @index structure A bivariate Gaussian with sds and correlation as separate variables (gamma precisions, a beta correlation).
 * @example
 * ```ts
 * import { beta, betaEta, gamma, gammaEta, gaussianMv, messageArg, mvnSdCorr, valueArg } from 'omai-engine/message-passing'
 * const g2 = gaussianMv(2)
 * const args = [valueArg(g2, [1, 1]), valueArg(g2, [0, 0]), valueArg(gamma, 1), valueArg(gamma, 1), messageArg(beta, betaEta(1, 1))]
 * mvnSdCorr.ep?.({ mode: 'ep', to: 4, args, params: {}, power: 1 }).message.map((v) => v.toFixed(2)) // => ['0.98', '-0.38']
 * ```
 */
export const mvnSdCorr: FactorOperator = defineOperator({
  name: 'mvn-sd-corr',
  description: 'x ~ N(mean, Sigma) for a bivariate Gaussian with two gamma precisions and a beta correlation',
  ports: [
    { name: 'x', accepts: ['gaussian-mv'], constant: true },
    { name: 'mean', accepts: ['gaussian-mv'], constant: true },
    { name: 'precision 1', accepts: ['gamma'], constant: true },
    { name: 'precision 2', accepts: ['gamma'], constant: true },
    { name: 'correlation', accepts: ['beta'], constant: true },
  ],
  params: {
    count: param.number({ min: 1, optional: true }),
    scatter: param.vector({ optional: true }),
    match: param.text({ choices: ['moments', 'log-moments'], optional: true }),
  },
  check: (_params, families) =>
    families[0]?.dim === 2 && families[1]?.dim === 2
      ? undefined
      : 'mvn-sd-corr: the vector and its mean have two entries',
  rules: {
    // A message rule, not a site rule: the message to a Gaussian vector is taken against its cavity only when that is not flat
    // (`side.flat`, read with a tolerance), which is not the plain quotient the engine's `projectSite` makes.
    ep: messageRule((c): OperatorResult => {
      const rows = rowsOf(c.params, 2)
      const { x, mean, eval1, grid } = tilt(c, rows)
      const [n0, n1] = [grid.nodes[0]?.length ?? 1, grid.nodes[1]?.length ?? 1]
      if (c.to >= 2) {
        const axis = c.to - 2
        return {
          message: parameterMessage(
            c,
            c.to,
            c.to < 4 ? 'precision' : 'correlation',
            grid.nodes[axis] as Float64Array,
            grid.marginals[axis] as Float64Array,
          ),
        }
      }
      // a Gaussian port: the mixture of the conditionals over the grid
      const target = c.to as 0 | 1
      const side = target === 0 ? x : mean
      const m = new Float64Array(2)
      const s = new Float64Array(4)
      const nodeMean = new Float64Array(2)
      const nodeCov = new Float64Array(4)
      const lambda = new Float64Array(4)
      const t = new Float64Array(3)
      const n2 = grid.nodes[2]?.length ?? 1
      for (let k = 0; k < n2; k++)
        for (let j = 0; j < n1; j++)
          for (let i = 0; i < n0; i++) {
            const w = grid.weights[i + n0 * (j + n1 * k)] as number
            if (!(w > 1e-13)) continue
            t[0] = (grid.nodes[0] as Float64Array)[i] as number
            t[1] = (grid.nodes[1] as Float64Array)[j] as number
            t[2] = (grid.nodes[2] as Float64Array)[k] as number
            const logDet = precisionAt(t, lambda)
            eval1(lambda, 0, logDet, target, nodeMean, nodeCov)
            for (let a = 0; a < 2; a++) {
              m[a] = (m[a] as number) + w * (nodeMean[a] as number)
              for (let b = 0; b < 2; b++)
                s[a * 2 + b] =
                  (s[a * 2 + b] as number) +
                  w * ((nodeCov[a * 2 + b] as number) + (nodeMean[a] as number) * (nodeMean[b] as number))
            }
          }
      const cov = Array.from(s, (v, e) => v - (m[Math.floor(e / 2)] as number) * (m[e % 2] as number))
      const proj = gaussianMvEta(Array.from(m), cov)
      return { message: proj.map((v, i) => v - (side.flat ? 0 : ((arg(c, target).eta as Eta)[i] ?? 0))) }
    }, 'numeric'),
  },
  evidence: {
    ep: (c) => {
      if (c.args.some((a) => a.eta !== undefined && !a.family.isProper(a.eta))) return Number.NaN
      const { grid } = tilt(c, rowsOf(c.params, 2))
      let logZ = grid.logMass
      for (const i of [2, 3, 4]) {
        const a = arg(c, i)
        if (a.eta !== undefined) logZ += a.family.logPartition(a.eta)
      }
      return logZ
    },
  },
  traits: { readsTarget: (to) => to >= 2 },
  noRule: { vmp: 'the expected log density of a correlation that is a beta variable has no closed form' },
})

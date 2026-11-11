/**
 * A bivariate Gaussian whose covariance is given by two standard deviations and a correlation, each a random variable of its own:
 * `x ~ N(mean, Sigma)` with `Sigma = [[1 / t1, rho / sqrt(t1 t2)], [rho / sqrt(t1 t2), 1 / t2]]`, the precisions `t1`, `t2` gamma variables and the
 * correlation `rho = 2 u - 1` for a beta variable `u` (a uniform correlation is `Beta(1, 1)`, the LKJ prior of a 2 by 2 matrix with
 * concentration `eta` is `Beta(eta, eta)`). The vector and the mean are Gaussian (or observed), integrated out in closed form given
 * the three; expectation propagation matches the gammas and the beta to the moments of a panelled three-dimensional grid over their
 * coordinates (`_gridn.ts`: the log of each precision, the logit of `u`), and the Gaussians to the mixture of their conditionals over
 * it. A known precision or correlation is a single node of its axis, so a Gaussian with known standard deviations and an unknown
 * correlation is a one-dimensional integral.
 */

import { OmaiError } from '../../../base/index.ts'
import { arg, evidenceOrNaN } from '../../core/_messages.ts'
import { ImproperMessageError, siteMessage } from '../../core/args.ts'
import { beta, gamma } from '../../core/families.ts'
import { gaussianMvEta } from '../../core/families-vector.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { type GridAxis, gammaAxis, pointAxis } from '../../numerics/_grid.ts'
import { betaAxis, integrateAxes } from '../../numerics/_gridn.ts'
import { softplusOf } from '../../numerics/_scalar.ts'
import { evaluatorOf, gaussianSide, type Rows, rowsOf } from './_sides.ts'

/** The axis of one of the three parameter ports: a known value is one node, a message its cavity's density. */
function axisOfPort(a: Arg, kind: 'precision' | 'correlation'): GridAxis {
  if (a.value !== undefined) {
    const v = typeof a.value === 'number' ? a.value : Number.NaN
    if (kind === 'precision') {
      if (!(v > 0)) throw new OmaiError('value', 'mvn-sd-corr: a known precision is positive')
      return pointAxis(Math.log(v))
    }
    if (!(v > -1 && v < 1)) throw new OmaiError('value', 'mvn-sd-corr: a known correlation is in (-1, 1)')
    const u = (v + 1) / 2
    return pointAxis(Math.log(u / (1 - u)))
  }
  const eta = a.eta as Eta
  if (kind === 'precision') {
    const shape = (eta[0] ?? 0) + 1
    const rate = -(eta[1] ?? 0)
    if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
      throw new ImproperMessageError('a gamma message has no density')
    return gammaAxis(shape, rate)
  }
  const [p, q] = [(eta[0] ?? 0) + 1, (eta[1] ?? 0) + 1]
  if (!(p > 0 && q > 0 && Number.isFinite(p) && Number.isFinite(q)))
    throw new ImproperMessageError('a beta message has no density')
  return betaAxis(p, q)
}

/** The precision matrix at the coordinates `(log t1, log t2, logit u)`, written into `out`, and its log determinant. */
function precisionAt(t: Float64Array, out: Float64Array): number {
  const [l1, l2, s] = [t[0] as number, t[1] as number, t[2] as number]
  // 1 - rho^2 = 4 u (1 - u)
  const logOneMinusRho2 = Math.LN2 * 2 - softplusOf(-s) - softplusOf(s)
  const oneMinusRho2 = Math.exp(logOneMinusRho2)
  const rho = 2 / (1 + Math.exp(-s)) - 1
  const [t1, t2] = [Math.exp(l1), Math.exp(l2)]
  out[0] = t1 / oneMinusRho2
  out[3] = t2 / oneMinusRho2
  out[1] = (-rho * Math.sqrt(t1 * t2)) / oneMinusRho2
  out[2] = out[1] as number
  return l1 + l2 - logOneMinusRho2
}

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
 * moments; there is no variational message (the expected log density has no closed form in the beta).
 *
 * @concept factor-operators
 * @index structure A bivariate Gaussian with sds and correlation as separate variables (gamma precisions, a beta correlation).
 * @example
 * ```ts
 * import { beta, betaEta, gamma, gammaEta, gaussianMv, messageArg, mvnSdCorr, valueArg } from 'omai-engine/message-passing'
 * const g2 = gaussianMv(2)
 * const args = [valueArg(g2, [1, 1]), valueArg(g2, [0, 0]), valueArg(gamma, 1), valueArg(gamma, 1), messageArg(beta, betaEta(1, 1))]
 * mvnSdCorr.ep?.({ mode: 'ep', to: 4, args, params: {}, power: 1 }).message.map((v) => v.toFixed(2)) // => ['0.87', '-0.44']
 * ```
 */
export const mvnSdCorr: FactorOperator = {
  name: 'mvn-sd-corr',
  description: 'x ~ N(mean, Sigma) for a bivariate Gaussian with two gamma precisions and a beta correlation',
  ports: [
    { name: 'x', accepts: ['gaussian-mv'], constant: true },
    { name: 'mean', accepts: ['gaussian-mv'], constant: true },
    { name: 'precision 1', accepts: ['gamma'], constant: true },
    { name: 'precision 2', accepts: ['gamma'], constant: true },
    { name: 'correlation', accepts: ['beta'], constant: true },
  ],
  support: { ep: 'numeric' },
  readsTarget: (to) => to >= 2,
  noRule: { vmp: 'the expected log density of a correlation that is a beta variable has no closed form' },
  check: (_params, families) =>
    families[0]?.dim === 2 && families[1]?.dim === 2
      ? undefined
      : 'mvn-sd-corr: the vector and its mean have two entries',
  ep: (c): OperatorResult => {
    const rows = rowsOf(c.params, 2)
    const { x, mean, eval1, grid } = tilt(c, rows)
    const [n0, n1] = [grid.nodes[0]?.length ?? 1, grid.nodes[1]?.length ?? 1]
    if (c.to >= 2) {
      const axis = c.to - 2
      const nodes = grid.nodes[axis] as Float64Array
      const p = grid.marginals[axis] as Float64Array
      if (c.to < 4) {
        let logMean = 0
        let valueMean = 0
        for (let i = 0; i < nodes.length; i++) {
          logMean += (p[i] as number) * (nodes[i] as number)
          valueMean += (p[i] as number) * Math.exp(nodes[i] as number)
        }
        return { message: siteMessage(gamma, arg(c, c.to).eta as Eta, [logMean, valueMean]) }
      }
      let logU = 0
      let log1mU = 0
      for (let i = 0; i < nodes.length; i++) {
        logU -= (p[i] as number) * softplusOf(-(nodes[i] as number))
        log1mU -= (p[i] as number) * softplusOf(nodes[i] as number)
      }
      return { message: siteMessage(beta, arg(c, 4).eta as Eta, [logU, log1mU]) }
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
  },
  logEvidence: (c) => {
    if (c.args.some((a) => a.eta !== undefined && !a.family.isProper(a.eta))) return Number.NaN
    return evidenceOrNaN(() => {
      const { grid } = tilt(c, rowsOf(c.params, 2))
      let logZ = grid.logMass
      for (const i of [2, 3, 4]) {
        const a = arg(c, i)
        if (a.eta !== undefined) logZ += a.family.logPartition(a.eta)
      }
      return logZ
    })
  },
}

// The three parameter ports of the bivariate normal with its standard deviations and correlation as variables (`mvn-sd-corr`, and its group form
// `mvn-sd-corr-group`): the axes of their grid, the precision matrix at a node, and the messages the tilted marginals give the gammas and the beta.

import { arg } from '../../core/_messages.ts'
import { siteMessage } from '../../core/args.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { beta, betaEta, gamma } from '../../core/families.ts'
import type { Arg, Eta, OperatorContext } from '../../core/types.ts'
import { betaAxis, type GridAxis, gammaAxis, pointAxis } from '../../numerics/_grid.ts'
import { centredMoments } from '../../numerics/_moments.ts'
import { softplusOf } from '../../numerics/_scalar.ts'

/** The axis of one of the three parameter ports: a known value is one node, a message its cavity's density. */
export function axisOfPort(a: Arg, kind: 'precision' | 'correlation'): GridAxis {
  if (a.value !== undefined) {
    const v = typeof a.value === 'number' ? a.value : Number.NaN
    if (kind === 'precision') {
      if (!(v > 0)) throw new ParameterError('mvn-sd-corr: a known precision is positive')
      return pointAxis(Math.log(v))
    }
    if (!(v > -1 && v < 1)) throw new ParameterError('mvn-sd-corr: a known correlation is in (-1, 1)')
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

/**
 * The shape and scale `(a, b)` of the beta whose mean and variance are those of `u = 1 / (1 + e^-s)` under weights on a grid of the
 * logit `s` (`k = m (1 - m) / v - 1`, `a = m k`, `b = (1 - m) k`), or `undefined` when the tilted spread leaves no beta (a point mass).
 */
function betaByMoments(nodes: Float64Array, weights: Float64Array): readonly [number, number] | undefined {
  const u = Float64Array.from(nodes, (s) => Math.exp(-softplusOf(-s)))
  const { mean, variance } = centredMoments(weights, 1, u)
  const k = (mean * (1 - mean)) / variance - 1
  return k > 0 && Number.isFinite(k) && mean > 0 && mean < 1 ? [mean * k, (1 - mean) * k] : undefined
}

/** The precision matrix at the coordinates `(log t1, log t2, logit u)`, written into `out`, and its log determinant. */
export function precisionAt(t: Float64Array, out: Float64Array): number {
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

/**
 * The message to one of the three parameter ports from the tilted marginal of its coordinate: a gamma matched to `(E log t, E t)`, and for the
 * correlation a beta matched to the tilted mean and variance of `u` (`match: 'moments'`, the default) or to `E log u` and `E log (1 - u)`
 * (`'log-moments'`, the KL projection).
 *
 * @param c - The context (the parameter `match` is read from it).
 * @param port - The port the message is for.
 * @param kind - Which of the three it is.
 * @param nodes - The grid of its coordinate (the log of a precision, the logit of `u`).
 * @param p - The normalised marginal weights of the nodes.
 * @returns The message.
 */
export function parameterMessage(
  c: OperatorContext,
  port: number,
  kind: 'precision' | 'correlation',
  nodes: Float64Array,
  p: Float64Array,
): Eta {
  const cavity = arg(c, port).eta as Eta
  if (kind === 'precision') {
    let logMean = 0
    let valueMean = 0
    for (let i = 0; i < nodes.length; i++) {
      logMean += (p[i] as number) * (nodes[i] as number)
      valueMean += (p[i] as number) * Math.exp(nodes[i] as number)
    }
    return siteMessage(gamma, cavity, [logMean, valueMean])
  }
  if (c.params['match'] !== 'log-moments') {
    // the beta with the tilted mean and variance of `u`, so the belief has the spread of the correlation itself
    const fit = betaByMoments(nodes, p)
    if (fit !== undefined) return betaEta(fit[0], fit[1]).map((v, i) => v - (cavity[i] ?? 0))
  }
  let logU = 0
  let log1mU = 0
  for (let i = 0; i < nodes.length; i++) {
    logU -= (p[i] as number) * softplusOf(-(nodes[i] as number))
    log1mU -= (p[i] as number) * softplusOf(nodes[i] as number)
  }
  return siteMessage(beta, cavity, [logU, log1mU])
}

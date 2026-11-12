/**
 * The joint site of a group of bivariate Gaussian vectors that share a population mean and a covariance made of two standard deviations and a
 * correlation: `x_j ~ N(mean, Sigma)` for `j = 1..J`, `Sigma` from two gamma precisions and a beta correlation (`mvn-sd-corr-group`), as ONE
 * factor with the vectors, the mean and the three parameters as its ports, where the per-vector factor (`mvn-sd-corr`) has five ports each. Given the
 * three parameters everything else is Gaussian, so the factor integrates the vectors and the mean out together in closed form (the evaluator of
 * `mvn-wishart-group`, `_group.ts`) and expectation propagation matches the parameters to the moments of a panelled three-dimensional grid over
 * them (`_grid.ts`, as `mvn-sd-corr` does) and each vector and the mean to the mixture of their conditionals over its nodes. One tilt serves every
 * port of a sweep. Splitting the group into a factor per vector leaves a loop through the shared mean and parameters (a mean too narrow with few groups).
 */

import { OmaiError } from '../../../base/index.ts'
import { outer, reshape, sub } from '../../../tensor/index.ts'
import { arg, evidenceOrNaN, memoised } from '../../core/_messages.ts'
import { numberParam } from '../../core/args.ts'
import { defineOperator, messageRule } from '../../core/define.ts'
import { gaussianMvEta } from '../../core/families-vector.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, OperatorContext, OperatorResult, Port } from '../../core/types.ts'
import { float64Of } from '../../numerics/_arrays.ts'
import { type GridNResult, integrateAxes } from '../../numerics/_grid.ts'
import { nodeEvaluator, type Sides, tiltKey } from './_group.ts'
import { axisOfPort, parameterMessage, precisionAt } from './_sdcorr.ts'
import { type GaussianSide, gaussianSide } from './_sides.ts'

const D = 2

/** The number of vectors of the factor. */
const groupsOf = (params: OperatorContext['params']): number =>
  Math.max(1, Math.floor(numberParam(params, 'groups', 1)))

/** The ports: the group vectors, the shared mean, the two precisions and the correlation's beta. */
const portsOf = (params: OperatorContext['params']): readonly Port[] => [
  ...Array.from(
    { length: groupsOf(params) },
    (_, j): Port => ({ name: `x${j}`, accepts: ['gaussian-mv'], constant: true }),
  ),
  { name: 'mean', accepts: ['gaussian-mv'] },
  { name: 'precision 1', accepts: ['gamma'], constant: true },
  { name: 'precision 2', accepts: ['gamma'], constant: true },
  { name: 'correlation', accepts: ['beta'], constant: true },
]

/** The result of one tilt, shared by every port of the factor and its evidence. */
interface Tilt {
  readonly logZ: number
  readonly grid: GridNResult
  /** The tilted mean and covariance of each port's Gaussian (the vectors, then the mean), by `D` and `D * D` blocks. */
  readonly mean: readonly Float64Array[]
  readonly cov: readonly Float64Array[]
}

/** The tilt: the grid over the three parameters, and the mixture of the conditionals of every Gaussian over its nodes. */
function tiltOf(c: OperatorContext): Tilt {
  const count = groupsOf(c.params)
  const mean = gaussianSide(arg(c, count), D)
  if (mean.point)
    throw new OmaiError(
      'value',
      'mvn-sd-corr-group: the mean is a Gaussian variable (a known mean is one mvn-sd-corr factor per vector)',
    )
  // banned-ok: non-numeric: one side record per group
  const groups = Array.from({ length: count }, (_, j) => gaussianSide(arg(c, j), D))
  const sides: Sides = { d: D, groups, mean }
  const evaluate = nodeEvaluator(sides)
  const axes = [
    axisOfPort(arg(c, count + 1), 'precision'),
    axisOfPort(arg(c, count + 2), 'precision'),
    axisOfPort(arg(c, count + 3), 'correlation'),
  ] as const
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const lambda = new Float64Array(D * D)
  const grid = integrateAxes(axes, (t) => evaluate(lambda, 0, precisionAt(t, lambda), undefined))
  const [n0, n1, n2] = [grid.nodes[0]?.length ?? 1, grid.nodes[1]?.length ?? 1, grid.nodes[2]?.length ?? 1]
  const dd = D * D
  // banned-ok: kernel: working buffer of a sweep kernel (the message-passing buffers are not converted, by decision)
  const means = Array.from({ length: count + 1 }, () => new Float64Array(D))
  const second = Array.from({ length: count + 1 }, () => new Float64Array(dd))
  const conditionals = new Float64Array((count + 1) * (D + dd))
  const t = new Float64Array(3)
  for (let k = 0; k < n2; k++)
    for (let j = 0; j < n1; j++)
      for (let i = 0; i < n0; i++) {
        const w = grid.weights[i + n0 * (j + n1 * k)] as number
        if (!(w > 1e-13)) continue
        t[0] = (grid.nodes[0] as Float64Array)[i] as number
        t[1] = (grid.nodes[1] as Float64Array)[j] as number
        t[2] = (grid.nodes[2] as Float64Array)[k] as number
        const logDet = precisionAt(t, lambda)
        evaluate(lambda, 0, logDet, conditionals)
        for (let port = 0; port <= count; port++) {
          if (port < count && (groups[port] as GaussianSide).point) continue
          const at = port * (D + dd)
          const mk = means[port] as Float64Array
          const sk = second[port] as Float64Array
          for (let a = 0; a < D; a++) {
            const ma = conditionals[at + a] as number
            mk[a] = (mk[a] as number) + w * ma
            for (let b = 0; b < D; b++)
              sk[a * D + b] =
                (sk[a * D + b] as number) +
                w * ((conditionals[at + D + a * D + b] as number) + ma * (conditionals[at + b] as number))
          }
        }
      }
  const cov = second.map((sk, port) => {
    const mk = means[port] as Float64Array
    return float64Of(sub(reshape(sk, [D, D]), outer(mk, mk)))
  })
  let logZ = grid.logMass
  for (let q = 1; q <= 3; q++) {
    const a = arg(c, count + q)
    if (a.eta !== undefined) logZ += a.family.logPartition(a.eta)
  }
  return { logZ, grid, mean: means, cov }
}

/** The tilt of the factor, kept in the engine's cache for the stretch the messages of one sweep share. */
const tiltFor = (c: OperatorContext): Tilt => memoised(c, tiltKey('mvn-sd-corr-group', c), () => tiltOf(c))

/**
 * `x_j ~ N(mean, Sigma)` for `J` bivariate Gaussian vectors that share a Gaussian population mean and a covariance made of two gamma precisions and a
 * beta correlation: the correlated random effects of a hierarchical model whose standard deviations and correlation are variables of their own,
 * as one factor. Given the three parameters the vectors and the mean are jointly Gaussian, so the factor integrates them out together and expectation
 * propagation matches the gammas and the beta (to the tilted mean and variance of `u`, as `mvn-sd-corr` does) on a zooming three-dimensional grid
 * and each vector and the mean to the mixture of their conditionals over it; one tilt serves every port of a sweep (about a tenth of a second at
 * eight groups). A known mean is `mvn-sd-corr` per vector. EP only: the expected log density has no closed form in the beta.
 *
 * @concept factor-operators
 * @index structure Bivariate vectors sharing a mean, sds and correlation, as one joint site.
 * @example
 * ```ts
 * import { beta, betaEta, gamma, gammaEta, gaussianMv, gaussianMvEta, messageArg, mvnSdCorrGroup } from 'omai-engine/message-passing'
 * const g2 = gaussianMv(2)
 * const x = (m: number[]) => messageArg(g2, gaussianMvEta(m, [0.2, 0, 0, 0.2]))
 * const args = [x([1, 0]), x([-1, 0.5]), x([0.5, -1]), messageArg(g2, gaussianMvEta([0, 0], [9, 0, 0, 9])), messageArg(gamma, gammaEta(3, 2)), messageArg(gamma, gammaEta(3, 2)), messageArg(beta, betaEta(2, 2))]
 * mvnSdCorrGroup.ep?.({ mode: 'ep', to: 6, args, params: { groups: 3 }, power: 1 }).message.length // => 2
 * ```
 */
export const mvnSdCorrGroup: FactorOperator = defineOperator({
  name: 'mvn-sd-corr-group',
  description:
    'x_j ~ N(mean, Sigma) for J bivariate Gaussian vectors sharing a Gaussian mean and a covariance of two gamma precisions and a beta correlation: one joint site',
  ports: portsOf,
  params: {
    groups: param.integer({ min: 1 }),
    match: param.text({ choices: ['moments', 'log-moments'], optional: true }),
  },
  check: (params, families) => {
    if (families.length !== groupsOf(params) + 4)
      return 'mvn-sd-corr-group: the vectors, the mean, the two precisions and the correlation are one port each'
    return families.slice(0, groupsOf(params) + 1).every((f) => f.dim === D)
      ? undefined
      : 'mvn-sd-corr-group: the vectors and their mean have two entries'
  },
  rules: {
    // A message rule (like `mvn-sd-corr`): the message to a Gaussian is taken against its cavity only when that is not flat.
    ep: messageRule((c): OperatorResult => {
      const count = groupsOf(c.params)
      const t = tiltFor(c)
      if (c.to >= count + 1) {
        const axis = c.to - count - 1
        return {
          message: parameterMessage(
            c,
            c.to,
            axis < 2 ? 'precision' : 'correlation',
            t.grid.nodes[axis] as Float64Array,
            t.grid.marginals[axis] as Float64Array,
          ),
        }
      }
      const side = gaussianSide(arg(c, c.to), D)
      const proj = gaussianMvEta(Array.from(t.mean[c.to] as Float64Array), Array.from(t.cov[c.to] as Float64Array))
      const cavity = arg(c, c.to).eta as Eta
      return { message: proj.map((v, i) => v - (side.flat ? 0 : (cavity[i] ?? 0))) }
    }, 'numeric'),
  },
  evidence: {
    ep: (c) => {
      if (c.args.some((a) => a.eta !== undefined && !a.family.isProper(a.eta))) return Number.NaN
      return evidenceOrNaN(() => tiltFor(c).logZ)
    },
  },
  noRule: { vmp: 'the expected log density of a correlation that is a beta variable has no closed form' },
})

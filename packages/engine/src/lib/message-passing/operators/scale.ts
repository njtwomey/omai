/**
 * The factor of a Gaussian vector whose coefficients share one random precision multiplier: `beta_j ~ N(0, 1 / (s_j lambda
 * + o_j))` with `lambda` a gamma variable and `s`, `o` known (a penalised spline in the eigenbasis of its penalty, where `s_j`
 * is the eigenvalue of the penalty and `o_j` the fixed precision of the directions it does not touch; a ridge prior with a
 * learned strength). Given `lambda` the factor is a diagonal Gaussian, so against a Gaussian cavity of the vector the
 * coefficients integrate out in closed form and what is left is a function of the one scalar `lambda`, whose moments are
 * computed on a grid in `log lambda`; the vector's message is the moment-matched mixture of the Gaussian conditionals over
 * that grid, dense, so the posterior keeps the correlations between the coefficients that the data induce.
 */

import { OmaiError } from '../../base/index.ts'
import { spd } from '../_dense.ts'
import { digammaOf, trigammaOf } from '../_scalar.ts'
import { gamma, gammaShapeRate } from '../families.ts'
import type { Arg, Eta, FactorOperator, OperatorContext, OperatorResult } from '../types.ts'
import { etaOf, ImproperMessageError, siteMessage, vectorParam } from './args.ts'

const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

const COARSE_POINTS = 81
const FINE_POINTS = 121

/** The scales and offsets of the precisions `s_j lambda + o_j`, checked. */
function precisions(c: OperatorContext): { readonly scale: readonly number[]; readonly offset: readonly number[] } {
  const scale = vectorParam(c.params, 'scale')
  const offset = c.params['offset'] === undefined ? scale.map(() => 0) : vectorParam(c.params, 'offset')
  if (scale.length !== offset.length)
    throw new OmaiError('shape', 'gaussian-mv-precision-scale: scale and offset differ in length')
  for (let j = 0; j < scale.length; j++) {
    const s = scale[j] ?? 0
    const o = offset[j] ?? 0
    if (!(s >= 0 && o >= 0 && s + o > 0))
      throw new OmaiError(
        'value',
        'gaussian-mv-precision-scale: each precision needs a non-negative scale and offset, not both zero',
      )
  }
  return { scale, offset }
}

interface Node {
  readonly t: number
  readonly logWeight: number
  readonly mean?: Float64Array
  readonly cov?: Float64Array
}

/** The log of the cavity of the vector integrated against the factor at `log lambda = t`, plus the gamma cavity in t, and (if asked) the Gaussian conditional. */
function nodeAt(
  t: number,
  h: readonly number[],
  jmat: readonly number[],
  shape: number,
  rate: number,
  scale: readonly number[],
  offset: readonly number[],
  conditional: boolean,
): Node | undefined {
  const d = h.length
  const lam = Math.exp(t)
  const m = Float64Array.from(jmat)
  let logPrec = 0
  for (let j = 0; j < d; j++) {
    const p = (scale[j] ?? 0) * lam + (offset[j] ?? 0)
    m[j * d + j] = (m[j * d + j] ?? 0) + p
    logPrec += Math.log(p)
  }
  const factor = spd(m, d)
  if (factor === undefined) return undefined
  const mu = factor.solve(h)
  let quad = 0
  for (let j = 0; j < d; j++) quad += (h[j] ?? 0) * (mu[j] ?? 0)
  const logWeight = shape * t - rate * lam + 0.5 * logPrec - 0.5 * factor.logDet + 0.5 * quad
  return conditional ? { t, logWeight, mean: mu, cov: factor.inverse() } : { t, logWeight }
}

/** The weighted moments of the grid: the log mass, E[log lambda], E[lambda] and (when the nodes carry them) the mean and covariance of the vector. */
function summarise(nodes: readonly Node[], step: number, d: number) {
  let top = Number.NEGATIVE_INFINITY
  for (const n of nodes) if (n.logWeight > top) top = n.logWeight
  let z = 0
  let eLog = 0
  let eLam = 0
  let eLog2 = 0
  const mean = new Float64Array(d)
  const second = new Float64Array(d * d)
  for (const n of nodes) {
    const w = Math.exp(n.logWeight - top)
    z += w
    eLog += w * n.t
    eLog2 += w * n.t * n.t
    eLam += w * Math.exp(n.t)
    if (n.mean !== undefined && n.cov !== undefined) {
      for (let i = 0; i < d; i++) {
        mean[i] = (mean[i] ?? 0) + w * (n.mean[i] ?? 0)
        for (let k = 0; k < d; k++)
          second[i * d + k] =
            (second[i * d + k] ?? 0) + w * ((n.cov[i * d + k] ?? 0) + (n.mean[i] ?? 0) * (n.mean[k] ?? 0))
      }
    }
  }
  const mt = eLog / z
  return {
    logZ: top + Math.log(z * step),
    logLambda: mt,
    sdLogLambda: Math.sqrt(Math.max(eLog2 / z - mt * mt, 0)),
    lambda: eLam / z,
    mean: Float64Array.from(mean, (v) => v / z),
    second: Float64Array.from(second, (v) => v / z),
  }
}

/**
 * The tilted distribution of the vector and the precision multiplier: a coarse grid in `log lambda` over the gamma cavity's
 * range finds where the mass is, a finer grid around it gives the moments. The cavity of the vector may be improper (no
 * information yet): `J + diag(s lambda + o)` only has to be positive definite where the weight is taken.
 */
function scaleTilt(c: OperatorContext) {
  const { scale, offset } = precisions(c)
  const d = scale.length
  const vec = etaOf(arg(c, 0))
  const h = vec.slice(0, d)
  const jmat = vec.slice(d)
  const { shape, rate } = gammaShapeRate(etaOf(arg(c, 1)))
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  const centre = digammaOf(shape) - Math.log(rate)
  const half = 15 * Math.sqrt(trigammaOf(shape))
  const lo = Math.max(centre - half, -60)
  const hi = Math.min(centre + half, 60)
  const coarse: Node[] = []
  for (let i = 0; i < COARSE_POINTS; i++) {
    const n = nodeAt(lo + ((hi - lo) * i) / (COARSE_POINTS - 1), h, jmat, shape, rate, scale, offset, false)
    if (n !== undefined) coarse.push(n)
  }
  if (coarse.length === 0)
    throw new ImproperMessageError('gaussian-mv-precision-scale: no precision makes the vector proper')
  const first = summarise(coarse, (hi - lo) / (COARSE_POINTS - 1), d)
  const flo = Math.max(lo, first.logLambda - 8 * first.sdLogLambda - 0.5)
  const fhi = Math.min(hi, first.logLambda + 8 * first.sdLogLambda + 0.5)
  const step = (fhi - flo) / (FINE_POINTS - 1)
  const fine: Node[] = []
  for (let i = 0; i < FINE_POINTS; i++) {
    const n = nodeAt(flo + step * i, h, jmat, shape, rate, scale, offset, true)
    if (n !== undefined) fine.push(n)
  }
  if (fine.length === 0)
    throw new ImproperMessageError('gaussian-mv-precision-scale: no precision makes the vector proper')
  return { ...summarise(fine, step, d), d, h, jmat }
}

/**
 * `beta_j ~ N(0, 1 / (s_j lambda + o_j))` for a Gaussian vector `beta` and one gamma-distributed multiplier `lambda`, with
 * the scales `s` and offsets `o` known: the smoothing prior of a penalised regression spline in the eigenbasis of its
 * penalty (`s` the eigenvalues, `o` the fixed precision of the unpenalised directions), or a ridge prior of unknown strength.
 * Expectation propagation integrates the vector out against its Gaussian cavity (a function of `lambda` only, on a grid in
 * `log lambda`), sends the vector the dense Gaussian that has the mean and covariance of the mixture of conditionals over
 * that grid, and the multiplier the gamma with the tilted `E log lambda` and `E lambda`. The cavity of the vector may be
 * flat: the first message is then the prior's own scale mixture. Variational message passing is the usual conjugate form
 * (each coefficient with scale 0 and offset > 0 is a constant; the others need a zero offset).
 *
 * @concept factor-operators
 * @index structure A Gaussian vector with precisions s_j lambda + o_j of one gamma multiplier (a smoothing parameter): dense EP.
 * @example
 * ```ts
 * import { gamma, gammaEta, gaussianMv, gaussianMvPrecisionScale, messageArg } from 'omai-engine/message-passing'
 * const v = gaussianMv(2)
 * const args = [messageArg(v, v.flat), messageArg(gamma, gammaEta(2, 1))]
 * const m = gaussianMvPrecisionScale.ep?.({ mode: 'ep', to: 0, args, params: { scale: [1, 1] }, power: 1 }).message
 * m?.map((x) => x.toFixed(3)) // => ['0.000', '0.000', '1.000', '0.000', '0.000', '1.000']
 * ```
 */
export const gaussianMvPrecisionScale: FactorOperator = {
  name: 'gaussian-mv-precision-scale',
  description: 'beta_j ~ N(0, 1 / (s_j lambda + o_j)) for a Gaussian vector and one gamma multiplier lambda',
  ports: [
    { name: 'beta', accepts: ['gaussian-mv'] },
    { name: 'lambda', accepts: ['gamma'] },
  ],
  support: { ep: 'numeric', vmp: 'exact' },
  readsTarget: (to) => to === 1,
  check: (params, families) =>
    vectorParam(params, 'scale').length === families[0]?.dim
      ? undefined
      : 'gaussian-mv-precision-scale: scale must have the length of the vector',
  ep: (c): OperatorResult => {
    const t = scaleTilt(c)
    if (c.to === 1) return { message: siteMessage(gamma, etaOf(arg(c, 1)), [t.logLambda, t.lambda]) }
    const d = t.d
    const cov = Float64Array.from(t.second)
    for (let i = 0; i < d; i++)
      for (let k = 0; k < d; k++) cov[i * d + k] = (cov[i * d + k] ?? 0) - (t.mean[i] ?? 0) * (t.mean[k] ?? 0)
    const factor = spd(cov, d)
    if (factor === undefined)
      throw new ImproperMessageError('gaussian-mv-precision-scale: the tilted covariance is not positive definite')
    const jNew = factor.inverse()
    const hNew = Array.from({ length: d }, (_, i) => {
      let s = 0
      for (let k = 0; k < d; k++) s += (jNew[i * d + k] ?? 0) * (t.mean[k] ?? 0)
      return s
    })
    const cavity: Eta = etaOf(arg(c, 0))
    return { message: [...hNew, ...Array.from(jNew)].map((v, i) => v - (cavity[i] ?? 0)) }
  },
  vmp: (c): OperatorResult => {
    const { scale, offset } = precisions(c)
    const d = scale.length
    const lambda = arg(c, 1)
    const meanLambda = lambda.family.moments(etaOf(lambda))[1] ?? 0
    if (c.to === 0) {
      const j = new Array<number>(d * d).fill(0)
      for (let i = 0; i < d; i++) j[i * d + i] = (scale[i] ?? 0) * meanLambda + (offset[i] ?? 0)
      return { message: [...new Array<number>(d).fill(0), ...j] }
    }
    for (let i = 0; i < d; i++)
      if ((scale[i] ?? 0) > 0 && (offset[i] ?? 0) !== 0)
        throw new OmaiError(
          'not-implemented',
          'gaussian-mv-precision-scale: a coefficient with both a scale and an offset has no variational rule',
        )
    const vec = arg(c, 0)
    const mo = vec.family.moments(etaOf(vec))
    let active = 0
    let quad = 0
    for (let i = 0; i < d; i++) {
      if ((scale[i] ?? 0) === 0) continue
      active++
      quad += (scale[i] ?? 0) * (mo[d + i * d + i] ?? 0)
    }
    return { message: [0.5 * active, -0.5 * quad] }
  },
  logEvidence: (c) => {
    try {
      return scaleTilt(c).logZ
    } catch (err) {
      // A message with no density yet (the flat start of a run) has no evidence.
      if (err instanceof ImproperMessageError) return Number.NaN
      throw err
    }
  },
  expectedLog: (c) => {
    const { scale, offset } = precisions(c)
    const d = scale.length
    const lambda = arg(c, 1)
    const mo = lambda.family.moments(etaOf(lambda))
    const vec = arg(c, 0)
    const m = vec.family.moments(etaOf(vec))
    let total = -0.5 * d * Math.log(2 * Math.PI)
    for (let j = 0; j < d; j++) {
      const s = scale[j] ?? 0
      const o = offset[j] ?? 0
      if (s > 0 && o !== 0)
        throw new OmaiError(
          'not-implemented',
          'gaussian-mv-precision-scale: a coefficient with both a scale and an offset has no variational rule',
        )
      total +=
        s > 0
          ? 0.5 * (Math.log(s) + (mo[0] ?? 0)) - 0.5 * s * (mo[1] ?? 0) * (m[d + j * d + j] ?? 0)
          : 0.5 * Math.log(o) - 0.5 * o * (m[d + j * d + j] ?? 0)
    }
    return total
  },
}

/**
 * The factor of a Gaussian vector whose coefficients share one random precision multiplier: `beta_j ~ N(0, 1 / (s_j lambda
 * + o_j))` with `lambda` a gamma variable and `s`, `o` known (a penalised spline in the eigenbasis of its penalty, where `s_j`
 * is the eigenvalue of the penalty and `o_j` the fixed precision of the directions it does not touch; a ridge prior with a
 * learned strength). Given `lambda` the factor is a diagonal Gaussian, so against a Gaussian cavity of the vector the
 * coefficients integrate out in closed form and what is left is a function of the one scalar `lambda`, whose moments are
 * computed on a grid in `log lambda`; the vector's message is the moment-matched mixture of the Gaussian conditionals over
 * that grid, dense, so the posterior keeps the correlations between the coefficients that the data induce.
 */

import { OmaiError } from '../../../base/index.ts'
import { arg } from '../../core/_messages.ts'
import { etaOf, vectorParam } from '../../core/args.ts'
import { defineOperator, messageRule, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gammaShapeRate } from '../../core/families.ts'
import { param } from '../../core/params.ts'
import type { Eta, FactorOperator, OperatorContext, OperatorResult } from '../../core/types.ts'
import { spd } from '../../numerics/_dense.ts'
import { centredMoments, shiftedSum } from '../../numerics/_moments.ts'
import { gammaLogWindow, windowProfiles } from '../../numerics/_windows.ts'

const COARSE_POINTS = 109
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
      throw new ParameterError(
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

/**
 * What a dense scale factor needs of its one mixing variable, written in `t` (the log of a gamma variable, or a Gaussian variable
 * itself): the precision of each coefficient at `t` (undefined where it is not finite), the log of the mixing variable's
 * cavity as a density in `t`, and the range to search.
 */
export interface VectorScaleModel {
  readonly precisions: (t: number) => readonly number[] | undefined
  readonly logCavity: (t: number) => number
  readonly lo: number
  readonly hi: number
  /** The first grid is over the whole range; `points` is the size of each grid (default 81 coarse, 121 fine). */
  readonly coarsePoints?: number
  readonly finePoints?: number
}

/** The log of the cavity of the vector integrated against the factor at `t`, plus the mixing variable's cavity, and (if asked) the Gaussian conditional. */
function nodeAt(
  t: number,
  h: readonly number[],
  jmat: readonly number[],
  model: VectorScaleModel,
  conditional: boolean,
): Node | undefined {
  const d = h.length
  const p = model.precisions(t)
  if (p === undefined) return undefined
  const m = Float64Array.from(jmat)
  let logPrec = 0
  for (let j = 0; j < d; j++) {
    const pj = p[j] ?? 0
    m[j * d + j] = (m[j * d + j] ?? 0) + pj
    logPrec += Math.log(pj)
  }
  const factor = spd(m, d)
  if (factor === undefined) return undefined
  const mu = factor.solve(h)
  let quad = 0
  for (let j = 0; j < d; j++) quad += (h[j] ?? 0) * (mu[j] ?? 0)
  const logWeight = model.logCavity(t) + 0.5 * logPrec - 0.5 * factor.logDet + 0.5 * quad
  return conditional ? { t, logWeight, mean: mu, cov: factor.inverse() } : { t, logWeight }
}

/** What a grid over the mixing variable gives: the log mass, `E t`, the sd of `t`, `E exp(t)`, and the mean and second moment of the vector. */
export interface VectorScaleTilt {
  readonly logZ: number
  /** `E t` (for a gamma multiplier, the expected log). */
  readonly logLambda: number
  readonly sdLogLambda: number
  /** `E exp(t)` (for a gamma multiplier, its mean). */
  readonly lambda: number
  /** The mean of the vector. */
  readonly mean: Float64Array
  /** The covariance of the vector under the mixture of conditionals: the mean of their covariances plus the covariance of their means. */
  readonly covariance: Float64Array
}

/** The weighted moments of the grid: the log mass, E[t], sd of t, E[exp t] and (when the nodes carry them) the mean and covariance of the vector. */
function summarise(nodes: readonly Node[], step: number, d: number): VectorScaleTilt {
  const { top, weights, total: z } = shiftedSum(nodes.map((n) => n.logWeight))
  const ts = nodes.map((n) => n.t)
  const { mean: mt, variance: varLog } = centredMoments(weights, z, ts)
  let eLam = 0
  for (const [k, n] of nodes.entries()) eLam += (weights[k] as number) * Math.exp(n.t)
  const mean = new Float64Array(d)
  const covariance = new Float64Array(d * d)
  const carriers = nodes.flatMap((n, k) =>
    n.mean !== undefined && n.cov !== undefined ? [{ n, w: weights[k] as number }] : [],
  )
  if (carriers.length > 0) {
    for (const { n, w } of carriers)
      for (let i = 0; i < d; i++) mean[i] = (mean[i] ?? 0) + w * ((n.mean as Float64Array)[i] ?? 0)
    for (let i = 0; i < d; i++) mean[i] = (mean[i] ?? 0) / z
    // The mean of the conditional covariances plus the covariance of the conditional means about their mean, both centred.
    for (const { n, w } of carriers) {
      const m = n.mean as Float64Array
      const cov = n.cov as Float64Array
      for (let i = 0; i < d; i++)
        for (let k = 0; k < d; k++)
          covariance[i * d + k] =
            (covariance[i * d + k] ?? 0) +
            w * ((cov[i * d + k] ?? 0) + ((m[i] ?? 0) - (mean[i] ?? 0)) * ((m[k] ?? 0) - (mean[k] ?? 0)))
    }
    for (let e = 0; e < d * d; e++) covariance[e] = (covariance[e] ?? 0) / z
  }
  return {
    logZ: top + Math.log(z * step),
    logLambda: mt,
    sdLogLambda: Math.sqrt(varLog),
    lambda: eLam / z,
    mean,
    covariance,
  }
}

/**
 * The tilted distribution of a Gaussian vector (cavity `exp(h . beta - beta' J beta / 2)`) and one mixing variable `t` under a
 * prior `beta_j | t ~ N(0, 1 / p_j(t))`, on a grid in `t`: a coarse grid over the model's range finds where the mass is (and
 * zooms in again while the mass is narrower than a few grid steps), a finer grid around it gives the moments. The cavity of
 * the vector may be improper (no information yet): `J + diag(p)` only has to be positive definite where the weight is taken.
 * Shared by the gamma multiplier ({@link gaussianMvPrecisionScale}) and the Gaussian or gamma input of a chain of functions
 * (`gaussian-mv-function-scale`); returns the log mass, the moments of `t`, `E exp(t)` and the mean and second moment of the vector.
 *
 * @param h - The cavity's linear natural parameter.
 * @param jmat - The cavity's precision, row-major.
 * @param model - The mixing variable.
 * @returns The tilted moments.
 */
export function vectorScaleTilt(
  h: readonly number[],
  jmat: readonly number[],
  model: VectorScaleModel,
): VectorScaleTilt & { readonly d: number } {
  const d = h.length
  const coarsePoints = model.coarsePoints ?? COARSE_POINTS
  const finePoints = model.finePoints ?? FINE_POINTS
  const grid = (lo: number, hi: number, points: number, conditional: boolean) => {
    const step = (hi - lo) / (points - 1)
    const nodes: Node[] = []
    for (let i = 0; i < points; i++) {
      const n = nodeAt(lo + step * i, h, jmat, model, conditional)
      if (n !== undefined) nodes.push(n)
    }
    return { nodes, step }
  }
  let lo = model.lo
  let hi = model.hi
  let coarse = grid(lo, hi, coarsePoints, false)
  if (coarse.nodes.length === 0)
    throw new ImproperMessageError('gaussian-mv-precision-scale: no precision makes the vector proper')
  let first = summarise(coarse.nodes, coarse.step, d)
  // A mass narrower than a few steps of the grid is not resolved: look again, over the window that holds it.
  for (let round = 0; round < 3 && first.sdLogLambda < 2 * coarse.step; round++) {
    const half = Math.max(8 * first.sdLogLambda, 4 * coarse.step)
    lo = Math.max(lo, first.logLambda - half)
    hi = Math.min(hi, first.logLambda + half)
    coarse = grid(lo, hi, coarsePoints, false)
    if (coarse.nodes.length === 0) break
    first = summarise(coarse.nodes, coarse.step, d)
  }
  const flo = Math.max(lo, first.logLambda - 8 * first.sdLogLambda - 0.5)
  const fhi = Math.min(hi, first.logLambda + 8 * first.sdLogLambda + 0.5)
  const fine = grid(flo, fhi, finePoints, true)
  if (fine.nodes.length === 0)
    throw new ImproperMessageError('gaussian-mv-precision-scale: no precision makes the vector proper')
  return { ...summarise(fine.nodes, fine.step, d), d }
}

/** The tilted distribution of the vector and the precision multiplier of a gamma cavity (`t` is `log lambda`). */
function scaleTilt(c: OperatorContext): VectorScaleTilt & { readonly d: number } {
  const { scale, offset } = precisions(c)
  const d = scale.length
  const vec = etaOf(arg(c, 0))
  const h = vec.slice(0, d)
  const jmat = vec.slice(d)
  const { shape, rate } = gammaShapeRate(etaOf(arg(c, 1)))
  if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
    throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
  const { lo, hi } = gammaLogWindow(shape, rate, windowProfiles.logTilt)
  return vectorScaleTilt(h, jmat, {
    precisions: (t) => {
      const lam = Math.exp(t)
      return scale.map((s, j) => s * lam + (offset[j] ?? 0))
    },
    logCavity: (t) => shape * t - rate * Math.exp(t),
    lo,
    hi,
  })
}

/** The natural parameters of the dense Gaussian with the mean and covariance of a tilted mixture (`vectorScaleTilt`). */
export function tiltedVectorOf(t: VectorScaleTilt & { readonly d: number }, what: string): Eta {
  const d = t.d
  const cov = Float64Array.from(t.covariance)
  const factor = spd(cov, d)
  if (factor === undefined) throw new ImproperMessageError(`${what}: the tilted covariance is not positive definite`)
  const jNew = factor.inverse()
  const hNew = Array.from({ length: d }, (_, i) => {
    let s = 0
    for (let k = 0; k < d; k++) s += (jNew[i * d + k] ?? 0) * (t.mean[k] ?? 0)
    return s
  })
  return [...hNew, ...Array.from(jNew)]
}

/**
 * The message to the vector from a tilted mixture (`vectorScaleTilt`): the dense Gaussian with the mixture's mean and covariance
 * divided by the cavity.
 */
export function vectorMessageOf(t: VectorScaleTilt & { readonly d: number }, cavity: Eta, what: string): Eta {
  return tiltedVectorOf(t, what).map((v, i) => v - (cavity[i] ?? 0))
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
 * @index structure A Gaussian vector whose precisions share one gamma multiplier (a smoothing parameter).
 * @example
 * ```ts
 * import { gamma, gammaEta, gaussianMv, gaussianMvPrecisionScale, messageArg } from 'omai-engine/message-passing'
 * const v = gaussianMv(2)
 * const args = [messageArg(v, v.flat), messageArg(gamma, gammaEta(2, 1))]
 * const m = gaussianMvPrecisionScale.ep?.({ mode: 'ep', to: 0, args, params: { scale: [1, 1] }, power: 1 }).message
 * m?.map((x) => x.toFixed(2)) // => ['0.00', '0.00', '1.00', '0.00', '0.00', '1.00']
 * ```
 */
export const gaussianMvPrecisionScale: FactorOperator = defineOperator({
  name: 'gaussian-mv-precision-scale',
  description: 'beta_j ~ N(0, 1 / (s_j lambda + o_j)) for a Gaussian vector and one gamma multiplier lambda',
  ports: [
    { name: 'beta', accepts: ['gaussian-mv'] },
    { name: 'lambda', accepts: ['gamma'] },
  ],
  params: { scale: param.vector(), offset: param.vector({ optional: true }) },
  check: (params, families) =>
    vectorParam(params, 'scale').length === families[0]?.dim
      ? undefined
      : 'gaussian-mv-precision-scale: scale must have the length of the vector',
  rules: {
    ep: siteRule((c) => {
      const t = scaleTilt(c)
      if (c.to === 1) return { moments: [t.logLambda, t.lambda] }
      return { natural: tiltedVectorOf(t, 'gaussian-mv-precision-scale') }
    }),
    vmp: messageRule((c): OperatorResult => {
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
    }),
  },
  evidence: {
    ep: (c) => scaleTilt(c).logZ,
    vmp: (c) => {
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
  },
  traits: { readsTarget: (to) => to === 1 },
})

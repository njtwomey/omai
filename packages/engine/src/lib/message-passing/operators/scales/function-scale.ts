/**
 * The factor of a Gaussian vector whose coefficients are standard normal draws scaled by functions of one latent variable:
 * `beta_j = z_j u_j(x)` with `z_j` independent standard normal, `x` a Gaussian (the log smoothing parameter) or a gamma variable
 * and `u_j` a chain of unary steps (`1 / sqrt(exp(x) d_j + n_j)`: the non-centred penalised spline). It is the dense counterpart of
 * the per-coefficient `apply-product` factor: given `x` the vector is Gaussian with the diagonal covariance `u_j(x)^2`, so against a
 * Gaussian cavity of the whole vector the coefficients integrate out in closed form and what is left is a function of the one
 * scalar `x`, whose moments are computed on a grid (the grid and the mixture of conditionals are those of
 * `gaussian-mv-precision-scale`, which is the same factor for a gamma multiplier). The vector's message is the dense Gaussian with
 * the moments of the mixture of conditionals, so the posterior keeps the correlations between the coefficients that the data
 * induce, which the marginal per coefficient of the scalar factor cannot.
 */

import { arg, errorText } from '../../core/_messages.ts'
import { etaOf } from '../../core/args.ts'
import { defineOperator, siteRule } from '../../core/define.ts'
import { ImproperMessageError, ParameterError } from '../../core/errors.ts'
import { gammaShapeRate } from '../../core/families.ts'
import { param } from '../../core/params.ts'
import type { FactorOperator, OperatorContext } from '../../core/types.ts'
import { resolvedVariance } from '../../numerics/_moments.ts'
import { gammaLogWindow, windowProfiles } from '../../numerics/_windows.ts'
import { chainOf, type Step, unaryFunctions } from '../likelihoods/functions.ts'
import { tiltedVectorOf, type VectorScaleTilt, vectorScaleTilt } from './scale.ts'

/** The functions `u_j` of the parameter `steps`: one chain of steps for each coefficient. */
function chains(params: Readonly<Record<string, unknown>>): ((x: number) => number)[] {
  const steps = params['steps']
  if (!Array.isArray(steps) || steps.length === 0 || !steps.every((s) => Array.isArray(s)))
    throw new ParameterError('gaussian-mv-function-scale: steps is one chain of unary steps for each coefficient')
  const table = unaryFunctions()
  return (steps as Step[][]).map((chain) => chainOf(table, chain).f)
}

/** The tilted vector and mixing variable: the precision of coefficient `j` at `x` is `1 / u_j(x)^2`. */
function functionScaleTilt(c: OperatorContext): VectorScaleTilt & { readonly d: number } {
  const us = chains(c.params)
  const d = us.length
  const vec = etaOf(arg(c, 0))
  const h = vec.slice(0, d)
  const jmat = vec.slice(d)
  const input = arg(c, 1)
  const eta = etaOf(input)
  const precisionsAt = (x: number): readonly number[] | undefined => {
    const p = new Array<number>(d)
    for (let j = 0; j < d; j++) {
      const u = (us[j] as (x: number) => number)(x)
      const pj = 1 / (u * u)
      if (!(u !== 0 && Number.isFinite(u) && pj > 0 && Number.isFinite(pj))) return undefined
      p[j] = pj
    }
    return p
  }
  if (input.family.kind === 'gamma') {
    const { shape, rate } = gammaShapeRate(eta)
    if (!(shape > 0 && rate > 0 && Number.isFinite(shape) && Number.isFinite(rate)))
      throw new ImproperMessageError('a gamma message has no density (its shape and rate are not positive)')
    const { lo, hi } = gammaLogWindow(shape, rate, windowProfiles.logTilt)
    return vectorScaleTilt(h, jmat, {
      precisions: (t) => precisionsAt(Math.exp(t)),
      logCavity: (t) => shape * t - rate * Math.exp(t),
      lo,
      hi,
    })
  }
  const hx = eta[0] ?? 0
  const jx = eta[1] ?? 0
  if (!(jx > 0 && Number.isFinite(jx) && Number.isFinite(hx)))
    throw new ImproperMessageError('a Gaussian message has no density (its precision is not positive)')
  const mean = hx / jx
  const sd = 1 / Math.sqrt(jx)
  return vectorScaleTilt(h, jmat, {
    precisions: precisionsAt,
    logCavity: (t) => hx * t - 0.5 * jx * t * t,
    lo: mean - 12 * sd,
    hi: mean + 12 * sd,
    coarsePoints: 161,
  })
}

/**
 * `beta_j = z_j u_j(x)` for a Gaussian vector `beta`, independent standard normals `z_j` (integrated out) and one Gaussian or
 * gamma variable `x`, the `u_j` chains of unary steps (`steps`, one chain for each coefficient, as for `apply`): the
 * non-centred penalised spline `beta_j = z_j / sqrt(exp(x) d_j + n_j)` with a log-normal smoothing parameter. Expectation
 * propagation integrates the vector out against its Gaussian cavity (a function of `x` only, on a grid), sends the vector the
 * dense Gaussian with the moments of the mixture of conditionals over that grid, and `x` the Gaussian (or the gamma) with the
 * tilted moments. Expectation propagation only.
 *
 * @concept factor-operators
 * @index structure A Gaussian vector of coefficients z_j u_j(x) with one Gaussian or gamma variable x: the dense non-centred scale.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, gaussianMv, gaussianMvFunctionScale, messageArg } from 'omai-engine/message-passing'
 * const v = gaussianMv(2)
 * // beta_j = z_j exp(-x / 2) with x ~ N(0, 0.0001): told nothing about beta, its prior precision is about one.
 * const args = [messageArg(v, v.flat), messageArg(gaussian, gaussianEta(0, 0.0001))]
 * const steps = [[['mul', -0.5], ['exp']], [['mul', -0.5], ['exp']]]
 * const m = gaussianMvFunctionScale.ep?.({ mode: 'ep', to: 0, args, params: { steps }, power: 1 }).message
 * m?.map((x) => x.toFixed(2)) // => ['0.00', '0.00', '1.00', '0.00', '0.00', '1.00']
 * ```
 */
export const gaussianMvFunctionScale: FactorOperator = defineOperator({
  name: 'gaussian-mv-function-scale',
  description: 'beta_j = z_j u_j(x) for a Gaussian vector, standard normal z and one Gaussian or gamma variable x',
  ports: [
    { name: 'beta', accepts: ['gaussian-mv'] },
    { name: 'x', accepts: ['gaussian', 'gamma'] },
  ],
  params: { steps: param.any({ example: [['square']] }) },
  check: (params, families) => {
    try {
      return chains(params).length === families[0]?.dim
        ? undefined
        : 'gaussian-mv-function-scale: steps must have one chain for each coefficient of the vector'
    } catch (e) {
      return errorText(e)
    }
  },
  rules: {
    ep: siteRule((c) => {
      const t = functionScaleTilt(c)
      if (c.to === 0) return { natural: tiltedVectorOf(t, 'gaussian-mv-function-scale') }
      const input = arg(c, 1)
      if (input.family.kind === 'gamma') return { moments: [t.logLambda, t.lambda] }
      return { mean: t.logLambda, variance: resolvedVariance(t.sdLogLambda ** 2, 'gaussian-mv-function-scale') }
    }),
  },
  evidence: { improperIsNaN: true, ep: (c) => functionScaleTilt(c).logZ },
  traits: { readsTarget: (to) => to === 1 },
})

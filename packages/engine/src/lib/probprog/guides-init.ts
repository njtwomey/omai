/**
 * Laplace-initialised guides: the starting parameters of an automatic guide set from a Laplace fit of the model (the
 * MAP point of its unconstrained log density and the inverse Hessian there), so stochastic gradient steps begin at a
 * Gaussian that is already right near the mode and only have to bend it towards the skew and the tails. Each guide family
 * is given the Gaussian in its own parameters (locations and scales, a Cholesky factor, a low-rank factor, a structured
 * factorisation).
 */

import { OmaiError } from '../base/index.ts'
import { cholesky, eigh } from '../numerics/linalg/index.ts'
import { lbfgs } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  arange,
  diagonal,
  eye,
  fromStorage,
  full,
  maximum,
  mul,
  neg,
  primalOf,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  take,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { isolated } from './_stack.ts'
import { findInitialParams, initToMedian, potentialEnergy, type UnconstrainedModel } from './density.ts'
import { type LaplaceFit, laplaceFit } from './engines/laplace.ts'
import { cachedGuideSites } from './guides.ts'
import { type AutoStructuredOptions, structuredInitialParams } from './guides-structured.ts'
import type { Model } from './primitives.ts'

/** The guide families {@link laplaceInitialParams} can initialise. */
export type LaplaceInitGuide =
  | 'autoNormal'
  | 'autoDelta'
  | 'autoMultivariateNormal'
  | 'autoLowRankMultivariateNormal'
  | 'autoStructured'
  | 'autoIafNormal'

/** Options of {@link laplaceInitialParams}. */
export interface LaplaceInitOptions {
  /** The guide family to initialise (default `autoNormal`). */
  readonly guide?: LaplaceInitGuide
  /** The rank of a low-rank guide (default ceil sqrt D, as the guide). */
  readonly rank?: number
  /** Options of a structured guide (the structure the factors are fitted to). */
  readonly structured?: AutoStructuredOptions
  /** The key of the starting point's search (default `key(0)`). */
  readonly key?: Key
  /** The scale given to every coordinate when the curvature at the mode is not usable (default 0.1). */
  readonly fallbackScale?: number
  /** Iteration cap of the mode search (default 1000). */
  readonly maxIterations?: number
}

/** The result of {@link laplaceInitialParams}. */
export interface LaplaceInit {
  /** The guide's parameters on the constrained scale, by name, for `initParams` of `svi`. */
  readonly params: Readonly<Record<string, Tensor>>
  /** The mode on the unconstrained scale. */
  readonly mode: Tensor
  /** The covariance there, or undefined when the curvature was not positive definite (the scale is then the fallback). */
  readonly covariance: Tensor | undefined
  /** Why the curvature was not used, when it was not. */
  readonly reason: string | undefined
  readonly fit: LaplaceFit | undefined
}

const asArray = (t: Tensor): Float64Array => Float64Array.from(toFlat(t))

/**
 * The parameters that start an automatic guide at the Laplace approximation: find the mode of the model's unconstrained
 * log density by L-BFGS, take the inverse Hessian there, and write that Gaussian in the guide's parameters. When the
 * curvature is not positive definite (a scale running to its boundary, as in a funnel) the mode is still used, with the
 * fallback scale, and `reason` says so. The result's `params` go to `svi` as `initParams`.
 *
 * @param model - The model.
 * @param data - Its data.
 * @param options - The guide family and its options.
 * @returns The parameters, the mode, the covariance and, if the curvature failed, the reason.
 * @throws OmaiError with code `value` when no starting point has a finite log density.
 * @concept laplace-approximation
 * @index function Initial parameters of an automatic guide from a Laplace fit of the model.
 * @example
 * ```ts
 * import { laplaceInitialParams, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 2) }
 * const init = laplaceInitialParams(model, undefined)
 * [Math.round(item(init.params['m_auto_loc'] as never) * 1e6) / 1e6, Math.round(item(init.params['m_auto_scale'] as never) * 1e6) / 1e6] // => [1, 0.707107]
 * ```
 */
export function laplaceInitialParams<D = void>(
  model: Model<D>,
  data: D,
  options: LaplaceInitOptions = {},
): LaplaceInit {
  const kind = options.guide ?? 'autoNormal'
  const key = options.key ?? makeKey(0)
  const fallback = options.fallbackScale ?? 0.1
  const um: UnconstrainedModel = isolated(() => potentialEnergy(model, data, { key: foldIn(key, 0) }))
  const start = findInitialParams(um, model, data, { key: foldIn(key, 1), strategy: initToMedian() })
  if (!start.valid) throw new OmaiError('value', 'laplaceInitialParams: no starting point with a finite log density')
  let fit: LaplaceFit | undefined
  let reason: string | undefined
  let mode: Tensor
  try {
    fit = laplaceFit(um, start.unconstrained, { maxIterations: options.maxIterations ?? 1000 })
    mode = fit.mode
  } catch (e) {
    // The curvature at the mode is unusable; the mode itself is still the better start.
    reason = e instanceof Error ? e.message : String(e)
    const r = lbfgs(
      (u: Value) => neg(um.fn(u)),
      fromStorage(Float64Array.from(start.unconstrained), [um.dim], 'float64'),
      {
        maxIterations: options.maxIterations ?? 1000,
        gradTol: 1e-8,
      },
    )
    mode = primalOf(r.x as Value)
  }
  const D = um.dim
  const covariance = fit?.covariance
  const modeFlat = asArray(mode)
  // The variance of each coordinate: the covariance's diagonal, or the fallback's square where there is no covariance.
  const variances: Value = covariance === undefined ? full([D], fallback ** 2) : diagonal(covariance)
  const sites = isolated(() => cachedGuideSites(model)(data))
  const params: Record<string, Tensor> = {}
  const slice = (offset: number, size: number, shape: readonly number[], values: Value): Tensor =>
    reshape(sliceAxis(values, 0, offset, offset + size), [...shape]) as Tensor
  if (kind === 'autoNormal' || kind === 'autoDelta') {
    for (const s of sites) {
      params[`${s.name}_auto_loc`] =
        kind === 'autoNormal'
          ? slice(s.offset, s.size, s.unconstrainedShape, mode)
          : primalOf(s.bijector.forward(slice(s.offset, s.size, s.unconstrainedShape, mode)))
      if (kind === 'autoNormal')
        params[`${s.name}_auto_scale`] = slice(s.offset, s.size, s.unconstrainedShape, sqrt(maximum(variances, 1e-12)))
    }
  } else if (kind === 'autoMultivariateNormal') {
    params['auto_loc'] = mode
    params['auto_scale_tril'] =
      covariance === undefined ? (mul(eye(D), fallback) as Tensor) : primalOf(cholesky(covariance))
  } else if (kind === 'autoLowRankMultivariateNormal') {
    const r = Math.min(options.rank ?? Math.ceil(Math.sqrt(D)), D)
    params['auto_loc'] = mode
    if (covariance === undefined) {
      params['auto_cov_factor'] = zeros([D, r])
      params['auto_scale'] = full([D], fallback)
    } else {
      const { eigenvalues, eigenvectors } = eigh(covariance)
      // The r largest eigenpairs (the last columns, largest first) make the factor; the diagonal takes what they leave of each variance.
      const largest = arange(D - 1, D - 1 - r, -1)
      const sd = sqrt(maximum(take(primalOf(eigenvalues), largest, 0), 0))
      const factor = mul(take(primalOf(eigenvectors), largest, 1), sd)
      const scale = sqrt(maximum(sub(variances, sum(square(factor), 1)), 1e-8))
      params['auto_cov_factor'] = factor as Tensor
      params['auto_scale'] = scale as Tensor
    }
  } else if (kind === 'autoStructured') {
    const cov = covariance ?? (mul(eye(D), fallback ** 2) as Tensor)
    Object.assign(
      params,
      structuredInitialParams(model, data, { mean: modeFlat, covariance: cov }, options.structured ?? {}),
    )
  } else {
    // The flows start at the identity: the mean and the scales are the Gaussian's, the correlations are left to the flows.
    params['auto_loc'] = mode
    params['auto_scale'] = sqrt(maximum(variances, 1e-12)) as Tensor
  }
  return { params, mode, covariance, reason, fit }
}

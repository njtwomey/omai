/**
 * The distributional (GAMLSS) families as extended families: every parameter of the distribution is a linear predictor
 * with its own link, the log-likelihood is the family's `logProb` over tensor operations, and the derivatives in the
 * predictors come by automatic differentiation. `gamlssExtended` builds one by name with starting values from the data
 * (the moments of the response) for the location, scale and shape.
 */

import { MdmlError } from '../../base/index.ts'
import { fromStorage } from '../../tensor/index.ts'
import {
  type DistributionalFamily,
  gamlssGamma,
  gamlssLogNormal,
  gamlssNegativeBinomial,
  gamlssNormal,
  gamlssStudentT,
} from './distributional.ts'
import { type ExtendedFamily, tensorExtendedFamily } from './extended.ts'

/** The distributions `gamlssExtended` knows. */
export type GamlssName = 'normal' | 'gamma' | 'negativeBinomial' | 'studentT' | 'logNormal'

const moments = (y: Float64Array, w: Float64Array): { mean: number; variance: number } => {
  let s = 0
  let t = 0
  for (let i = 0; i < y.length; i++) {
    s += (w[i] ?? 1) * (y[i] ?? 0)
    t += w[i] ?? 1
  }
  const mean = s / Math.max(t, 1e-300)
  let v = 0
  for (let i = 0; i < y.length; i++) v += (w[i] ?? 1) * ((y[i] ?? 0) - mean) ** 2
  return { mean, variance: Math.max(v / Math.max(t, 1e-300), 1e-12) }
}

/**
 * A distributional family as an extended family: the normal `(mu, log sigma)`, the gamma `(log mu, log sigma)` with
 * `sigma` the coefficient of variation, the negative binomial `(log mu, log sigma)` with variance `mu + sigma mu^2`, the
 * Student-t `(mu, log sigma, log nu)` and the log-normal `(mu, log sigma)` of `log Y`.
 *
 * @param name - The distribution.
 * @returns The family, one predictor per parameter.
 * @throws MdmlError `value` for an unknown name.
 * @concept distributional-regression
 * @example
 * ```ts
 * import { families } from 'mdml-engine/dists'
 * families.gamlssExtended('studentT').predictors // => ['mu', 'sigma', 'nu']
 * ```
 */
export function gamlssExtended(name: GamlssName): ExtendedFamily {
  const build = (
    family: DistributionalFamily,
    start: (y: Float64Array, w: Float64Array) => readonly number[],
  ): ExtendedFamily =>
    tensorExtendedFamily({
      name: `gamlss(${name})`,
      predictors: family.parameters.map((p) => p.name),
      links: family.parameters.map((p) => p.link),
      start,
      logProb: (y, eta) => family.fromPredictors(eta).logProb(fromStorage(Float64Array.from(y), [y.length], 'float64')),
    })
  switch (name) {
    case 'normal':
      return build(gamlssNormal(), (y, w) => {
        const m = moments(y, w)
        return [m.mean, 0.5 * Math.log(m.variance)]
      })
    case 'gamma':
      return build(gamlssGamma(), (y, w) => {
        const m = moments(y, w)
        return [Math.log(Math.max(m.mean, 1e-6)), 0.5 * Math.log(m.variance / Math.max(m.mean * m.mean, 1e-12))]
      })
    case 'negativeBinomial':
      return build(gamlssNegativeBinomial(), (y, w) => {
        const m = moments(y, w)
        const sigma = Math.max((m.variance - m.mean) / Math.max(m.mean * m.mean, 1e-12), 0.05)
        return [Math.log(Math.max(m.mean, 0.1)), Math.log(sigma)]
      })
    case 'studentT':
      return build(gamlssStudentT(), (y, w) => {
        const m = moments(y, w)
        return [m.mean, 0.5 * Math.log(m.variance) - 0.2, Math.log(8)]
      })
    case 'logNormal':
      return build(gamlssLogNormal(), (y, w) => {
        const logs = Float64Array.from(y, (v) => Math.log(Math.max(v, 1e-12)))
        const m = moments(logs, w)
        return [m.mean, 0.5 * Math.log(m.variance)]
      })
    default:
      throw new MdmlError('value', `gamlssExtended: unknown family "${String(name)}"`)
  }
}

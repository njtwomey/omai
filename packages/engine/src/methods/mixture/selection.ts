import { type CrossValidationCurve, crossValidationCurve } from '../../lib/exposition/index.ts'
import { type FitOptions, fitRunOptions } from '../../lib/learn/index.ts'
import { foldIn, type Key } from '../../lib/random/index.ts'
import type { TensorLike } from '../../lib/tensor/index.ts'
import { type CovarianceType, gaussianMixture, mixtureCriteria } from './gaussian.ts'

/** The information criteria of Gaussian mixtures over a range of component counts. */
export interface ComponentSelection {
  readonly components: readonly number[]
  readonly bic: number[]
  readonly aic: number[]
  /** The count with the lowest BIC. */
  readonly best: number
  /** The BIC curve as a record (scores are -BIC so higher is better; no standard errors). */
  readonly curve: CrossValidationCurve
}

/**
 * Choose the number of mixture components by BIC and AIC (sklearn's model-selection example): fit a Gaussian mixture
 * for each count and record both criteria. TODO: ICL, held-out likelihood and stability.
 *
 * @param x - Points `[n, d]`.
 * @param components - The counts to try.
 * @param options - The covariance type and the fit options.
 * @returns The criteria, the best count and a curve record.
 */
export function selectComponents(
  x: TensorLike,
  components: readonly number[],
  options: FitOptions & { readonly covarianceType?: CovarianceType } = {},
): ComponentSelection {
  const { covarianceType, ...fit } = options
  const key = fitRunOptions(fit, 'select-components').key as Key
  const crit = components.map((k, i) =>
    mixtureCriteria(
      gaussianMixture({ nComponents: k, ...(covarianceType === undefined ? {} : { covarianceType }) }).fit(
        { x },
        { key: foldIn(key, i) },
      ),
      x,
    ),
  )
  const bic = crit.map((c) => c.bic)
  const best = components[bic.indexOf(Math.min(...bic))] ?? components[0] ?? 1
  return {
    components,
    bic,
    aic: crit.map((c) => c.aic),
    best,
    curve: crossValidationCurve({
      parameter: 'components',
      values: [...components],
      meanScores: bic.map((v) => -v),
      standardErrors: bic.map(() => 0),
      higherIsBetter: true,
      best,
    }),
  }
}

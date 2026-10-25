import { WhiteboxError } from '../../lib/base/index.ts'
import { permutationImportance } from '../../lib/explain/index.ts'
import { type Explained, importances } from '../../lib/exposition/index.ts'
import { type Fitted, featureNamesOf, type LearnData, matrixView, type Scoring } from '../../lib/learn/index.ts'
import type { Key } from '../../lib/random/index.ts'

/** Options of {@link featureImportance}. */
export interface FeatureImportanceOptions {
  /** `permutation` (default: any model, on the data given) or `impurity` (a tree model's mean impurity decrease). */
  readonly method?: 'permutation' | 'impurity'
  /** The scorer of permutation importance (default the estimator's score: accuracy or R²). */
  readonly scoring?: Scoring
  /** Shuffles per feature (default 5). */
  readonly nRepeats?: number
  /** Which data the importances are measured on, for the record (`training`, `held out`). */
  readonly on?: string
  readonly key?: Key
}

/** Feature importances: one mean per feature and its spread across repeats. */
export interface FeatureImportanceResult {
  readonly mean: readonly number[]
  readonly spread: readonly number[]
  /** The score on the unshuffled data (permutation importance). */
  readonly baseline?: number
}

function metricName(fitted: Fitted, scoring: Scoring): string {
  if (typeof scoring === 'string') return scoring
  if (scoring === undefined) return fitted.kind === 'classifier' ? 'accuracy' : 'R²'
  return 'score'
}

/**
 * How much a fitted model relies on each feature: the drop in its score when the feature's values are shuffled
 * across rows (permutation importance, repeated with the spread recorded) or, for trees and their ensembles, the
 * impurity decrease of the splits on it.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data to measure on (features and targets; held-out data shows what generalises).
 * @param options - method, scoring, nRepeats, on and key.
 * @returns The importances with an `importances` record.
 */
export function featureImportance(
  fitted: Fitted,
  data: LearnData,
  options: FeatureImportanceOptions = {},
): Explained<FeatureImportanceResult> {
  const x = matrixView(data.x, 'featureImportance', { allowNaN: true })
  const features = featureNamesOf(data, x.d)
  const on = options.on === undefined ? {} : { on: options.on }
  if (options.method === 'impurity') {
    const given = (fitted as { readonly featureImportances?: readonly number[] }).featureImportances
    if (given === undefined)
      throw new WhiteboxError('value', `featureImportance: ${fitted.estimator} has no impurity importances`)
    const mean = [...given]
    const spread = mean.map(() => 0)
    return {
      mean,
      spread,
      exposition: [
        importances({ method: 'impurity', features, mean, metric: 'impurity decrease', direction: 'higher', ...on }),
      ],
    }
  }
  const r = permutationImportance(fitted, data, {
    ...(options.scoring === undefined ? {} : { scoring: options.scoring }),
    ...(options.nRepeats === undefined ? {} : { nRepeats: options.nRepeats }),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  const metric = metricName(fitted, options.scoring)
  return {
    mean: r.mean,
    spread: r.std,
    baseline: r.baseline,
    exposition: [
      importances({
        method: 'permutation',
        features,
        mean: r.mean,
        spread: r.std,
        repeats: r.importances,
        metric,
        // Scorers are larger-is-better (errors are negated), so a drop is always a loss.
        direction: 'higher',
        baseline: r.baseline,
        ...on,
      }),
    ],
  }
}

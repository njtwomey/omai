/**
 * Exposition records for feature attributions: additive per-row attributions such as SHAP values or LIME weights (a
 * base value plus one contribution per feature), the running estimates of a sampled attribution as samples accrue, and
 * global importances (permutation importance, impurity decrease, mean |SHAP|) with their spread.
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/**
 * How a sampled attribution estimate settled as samples accrued (sampling SHAP's running means): the samples taken at
 * each record, and the running estimate for every feature then, with its standard error.
 */
export interface AttributionConvergence {
  /** Which explained row (an index into the record's rows) the estimates are of. */
  readonly row: number
  /** Samples (permutations, coalitions) taken at each record. */
  readonly samples: readonly number[]
  /** estimates[step][feature]. */
  readonly estimates: readonly (readonly number[])[]
  /** standardErrors[step][feature]. */
  readonly standardErrors?: readonly (readonly number[])[]
}

/**
 * How an attribution record's features are laid out, for the views of other modalities: an image (`shape` [h, w],
 * features row-major), the tokens of a text (features in reading order; `text` and `spans` place them in it), or a
 * series (features in time order). Without a layout the features are a table's columns.
 */
export type AttributionLayout =
  | { readonly kind: 'image'; readonly shape: readonly [number, number] }
  | {
      readonly kind: 'tokens'
      readonly text?: string
      /** Each token's [start, end) in `text`. */
      readonly spans?: readonly (readonly [number, number])[]
    }
  | { readonly kind: 'series'; readonly time?: readonly number[] }

/**
 * Additive feature attributions of a model's predictions (SHAP values, LIME weights times the feature's offset):
 * for each explained row, the base value the attributions start from, one value per feature, the feature values,
 * and the model's prediction. For SHAP the base value plus the row's values is the prediction (efficiency); a local
 * surrogate (LIME) ends at its own prediction, which may differ.
 */
export interface FeatureAttributions {
  readonly kind: 'featureAttributions'
  /** The explainer (`tree`, `kernel`, `exact`, `sampling`, `lime`). */
  readonly method: string
  readonly features: readonly string[]
  /** What is explained (`prediction`, `P(malignant)`, `log odds`). */
  readonly output: string
  /** The expected output the attributions start from (E f(x) over the background). */
  readonly baseValue: number
  /** A base value per row, when they differ (each row's local surrogate has its own intercept). */
  readonly baseValues?: readonly number[]
  /** values[row][feature]. */
  readonly values: readonly (readonly number[])[]
  /** The explained rows' feature values, data[row][feature]. */
  readonly data: readonly (readonly number[])[]
  /** The model's output at each row. */
  readonly predictions: readonly number[]
  /** Which rows of the dataset were explained (default 0, 1, ...). */
  readonly indices?: readonly number[]
  /** Standard errors of sampled estimates, standardErrors[row][feature]. */
  readonly standardErrors?: readonly (readonly number[])[]
  /** SHAP interaction values, interactions[row][i][j] (each row of a matrix sums to the main attribution). */
  readonly interactions?: readonly (readonly (readonly number[])[])[]
  /** A sampled estimate's convergence. */
  readonly convergence?: AttributionConvergence
  /**
   * For each feature, the feature that most interacts with it (from the interaction values, or shap's approximate
   * interactions), to colour its dependence scatter by.
   */
  readonly partners?: readonly number[]
  /** How well a local surrogate fits its neighbourhood (LIME's weighted R^2), per row. */
  readonly localFit?: readonly number[]
  /** How the features are laid out (an image, a text's tokens, a series); default a table's columns. */
  readonly layout?: AttributionLayout
  /** The interpretable units an attribution was computed over (superpixel or window ids per feature), for outlines. */
  readonly segments?: readonly number[]
}

/**
 * A feature-attribution record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a row's values or data do not have one entry per feature, or the predictions,
 *   rows, errors, interactions or convergence do not match the rows; `value` when the convergence names a row that was
 *   not explained.
 * @example
 * ```ts
 * import { featureAttributions } from 'mdml-engine/exposition'
 * const r = featureAttributions({ method: 'exact', features: ['a', 'b'], output: 'prediction', baseValue: 1, values: [[2, -0.5]], data: [[3, 4]], predictions: [2.5] })
 * r.values[0].length // => 2
 * ```
 */
export function featureAttributions(record: Omit<FeatureAttributions, 'kind'>): FeatureAttributions {
  const k = 'featureAttributions'
  const d = record.features.length
  const n = record.values.length
  rowsOf(k, 'values', record.values, d)
  sameLength(k, 'data', record.data.length, n)
  rowsOf(k, 'data', record.data, d)
  sameLength(k, 'predictions', record.predictions.length, n)
  if (record.indices !== undefined) sameLength(k, 'indices', record.indices.length, n)
  if (record.localFit !== undefined) sameLength(k, 'localFit', record.localFit.length, n)
  if (record.baseValues !== undefined) sameLength(k, 'baseValues', record.baseValues.length, n)
  if (record.partners !== undefined) {
    sameLength(k, 'partners', record.partners.length, d)
    for (const p of record.partners)
      if (!(Number.isInteger(p) && p >= -1 && p < d))
        throw new MdmlError('value', `${k}: partner ${p} is not a feature (or -1 for none)`)
  }
  if (record.standardErrors !== undefined) {
    sameLength(k, 'standardErrors', record.standardErrors.length, n)
    rowsOf(k, 'standardErrors', record.standardErrors, d)
  }
  if (record.interactions !== undefined) {
    sameLength(k, 'interactions', record.interactions.length, n)
    record.interactions.forEach((m, i) => {
      sameLength(k, `interactions[${i}]`, m.length, d)
      rowsOf(k, `interactions[${i}]`, m, d)
    })
  }
  const layout = record.layout
  if (layout?.kind === 'image' && layout.shape[0] * layout.shape[1] !== d)
    throw new MdmlError('shape', `${k}: an image of ${layout.shape.join('x')} for ${d} features`)
  if (layout?.kind === 'tokens' && layout.spans !== undefined) sameLength(k, 'layout.spans', layout.spans.length, d)
  if (layout?.kind === 'series' && layout.time !== undefined) sameLength(k, 'layout.time', layout.time.length, d)
  if (record.segments !== undefined) sameLength(k, 'segments', record.segments.length, d)
  const c = record.convergence
  if (c !== undefined) {
    if (!(Number.isInteger(c.row) && c.row >= 0 && c.row < n))
      throw new MdmlError('value', `${k}: convergence.row ${c.row} is not an explained row`)
    sameLength(k, 'convergence.estimates', c.estimates.length, c.samples.length)
    rowsOf(k, 'convergence.estimates', c.estimates, d)
    if (c.standardErrors !== undefined) {
      sameLength(k, 'convergence.standardErrors', c.standardErrors.length, c.samples.length)
      rowsOf(k, 'convergence.standardErrors', c.standardErrors, d)
    }
  }
  return { kind: k, ...record }
}

/**
 * Global feature importances (permutation importance, impurity decrease, mean |SHAP|): per feature the mean over
 * repeats and its spread, the metric it is measured in and whether a larger value of that metric is better.
 */
export interface Importances {
  readonly kind: 'importances'
  /** How they were measured (`permutation`, `impurity`, `mean |SHAP|`). */
  readonly method: string
  readonly features: readonly string[]
  readonly mean: readonly number[]
  /** The spread across repeats (a standard deviation). */
  readonly spread?: readonly number[]
  /** Every repeat, repeats[feature][repeat]. */
  readonly repeats?: readonly (readonly number[])[]
  /** What an importance is a change in (`accuracy`, `R²`, `|SHAP|`). */
  readonly metric: string
  /** Whether larger values of the metric are better (`higher`) or worse (`lower`). */
  readonly direction: 'higher' | 'lower'
  /** The metric on the unpermuted data, for permutation importance. */
  readonly baseline?: number
  /** Which data the importances were measured on (`training`, `held out`). */
  readonly on?: string
}

/**
 * An importances record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the means, spreads or repeats do not have one entry per feature, or the repeats
 *   differ in count.
 * @example
 * ```ts
 * import { importances } from 'mdml-engine/exposition'
 * const r = importances({ method: 'permutation', features: ['a', 'b'], mean: [0.3, 0.01], spread: [0.05, 0.01], metric: 'accuracy', direction: 'higher' })
 * r.kind // => 'importances'
 * ```
 */
export function importances(record: Omit<Importances, 'kind'>): Importances {
  const k = 'importances'
  const d = record.features.length
  sameLength(k, 'mean', record.mean.length, d)
  if (record.spread !== undefined) sameLength(k, 'spread', record.spread.length, d)
  if (record.repeats !== undefined) {
    sameLength(k, 'repeats', record.repeats.length, d)
    rowsOf(k, 'repeats', record.repeats, record.repeats[0]?.length ?? 0)
  }
  return { kind: k, ...record }
}

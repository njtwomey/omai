/**
 * The record of the evaluation of explanations: several explainers scored by metrics in Quantus's categories
 * (faithfulness, robustness, localisation, complexity, randomisation, axioms), each with its direction, and how far
 * the explainers agree with one another.
 */

import { rowsOf, sameLength, seal } from './_check.ts'

/** One metric of an explanation-quality record, a value per explainer. */
export interface ExplanationMetric {
  readonly name: string
  readonly category: 'faithfulness' | 'robustness' | 'localisation' | 'complexity' | 'randomisation' | 'axiomatic'
  /** One value per explainer (a mean over rows). */
  readonly values: readonly number[]
  /** The spread over rows, per explainer. */
  readonly spread?: readonly number[]
  /** Whether larger values are better. */
  readonly better: 'higher' | 'lower'
}

/** Explainers compared on metrics of explanation quality, and their agreement. */
export interface ExplanationQuality {
  readonly kind: 'explanationQuality'
  readonly explainers: readonly string[]
  readonly metrics: readonly ExplanationMetric[]
  /** Mean pairwise similarity between the explainers' explanations. */
  readonly agreement?: { readonly metric: string; readonly matrix: readonly (readonly number[])[] }
  /** How many rows the metrics average over. */
  readonly nRows?: number
}

/**
 * An explanation-quality record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a metric or the agreement matrix does not have one entry per explainer.
 * @example
 * ```ts
 * import { explanationQuality } from 'mdml-engine/exposition'
 * const r = explanationQuality({ explainers: ['SHAP', 'LIME'], metrics: [{ name: 'deletion AUC', category: 'faithfulness', values: [0.2, 0.3], better: 'lower' }] })
 * r.metrics[0].values.length // => 2
 * ```
 */
export function explanationQuality(record: Omit<ExplanationQuality, 'kind'>): ExplanationQuality {
  const k = 'explanationQuality'
  const e = record.explainers.length
  for (const m of record.metrics) {
    sameLength(k, `${m.name}.values`, m.values.length, e)
    if (m.spread !== undefined) sameLength(k, `${m.name}.spread`, m.spread.length, e)
  }
  if (record.agreement !== undefined) {
    sameLength(k, 'agreement.matrix', record.agreement.matrix.length, e)
    rowsOf(k, 'agreement.matrix', record.agreement.matrix, e)
  }
  return seal({ kind: k, ...record })
}

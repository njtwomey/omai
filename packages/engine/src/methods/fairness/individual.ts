/**
 * Individual-fairness audits of a fitted classifier: consistency of its decisions among each row's nearest neighbours,
 * and Lipschitz fairness of its scores under the standardised feature distance (without the protected attribute), as
 * a `coefficientTable` and a `curves` record of the pairs' output gap against their distance.
 */

import { coefficientTable, curveSet, type Explained } from '../../lib/exposition/index.ts'
import { consistency, lipschitzFairness } from '../../lib/fairness/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { type MitigationOptions, mitigationData } from './_mitigation.ts'
import { type FairnessData, positiveScores } from './groups.ts'

/** Options of {@link individualFairnessAudit}. */
export interface IndividualFairnessOptions extends MitigationOptions {
  /** Neighbours for consistency (default 5). */
  readonly k?: number
  /** The Lipschitz constant on the standardised distance (default 1). */
  readonly constant?: number
  /** Pairs checked (default 5000, keyed). */
  readonly pairs?: number
  /** Score the model without the protected column (default true). */
  readonly dropSensitive?: boolean
}

/** The result of {@link individualFairnessAudit}. */
export interface IndividualFairnessAudit {
  readonly consistency: number
  readonly violationRate: number
  readonly worstRatio: number
}

/**
 * An individual-fairness audit of a fitted binary classifier: the consistency of its decisions among nearest neighbours
 * (AIF360's), and the share of row pairs whose score gap exceeds the Lipschitz constant times their distance in
 * standardised features. Exposition: a `coefficientTable` of the measures and a `curves` record of the sampled pairs'
 * score gap against their distance, with the Lipschitz line.
 *
 * @param model - A fitted binary classifier.
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, neighbours, constant, pairs and key.
 * @returns The measures, with the records.
 */
export function individualFairnessAudit(
  model: Fitted,
  data: FairnessData,
  options: IndividualFairnessOptions = {},
): Explained<IndividualFairnessAudit> {
  const d = mitigationData(data, options, 'individualFairnessAudit')
  const { scores, probability } = positiveScores(model, options.dropSensitive === false ? data.x : d.unaware.x)
  const dims = d.rows[0]?.length ?? 0
  const mean = Array.from({ length: dims }, (_, j) => d.rows.reduce((s, r) => s + (r[j] ?? 0), 0) / d.rows.length)
  const sd = Array.from(
    { length: dims },
    (_, j) => Math.sqrt(d.rows.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / d.rows.length) || 1,
  )
  const z = d.rows.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] ?? 1)))
  const threshold = probability ? 0.5 : 0
  const c = consistency(
    z,
    scores.map((s) => (s >= threshold ? 1 : 0)),
    { k: options.k ?? 5 },
  )
  const L = options.constant ?? 1
  const lip = lipschitzFairness(z, scores, { constant: L, pairs: options.pairs ?? 5000, key: d.key })
  const shown = lip.distances
    .map((_, i) => i)
    .filter((i) => i % Math.max(1, Math.floor(lip.distances.length / 400)) === 0)
  const order = shown.sort((a, b) => (lip.distances[a] ?? 0) - (lip.distances[b] ?? 0))
  const dist = order.map((i) => lip.distances[i] ?? 0)
  return {
    consistency: c,
    violationRate: lip.violationRate,
    worstRatio: lip.worstRatio,
    exposition: [
      coefficientTable({
        names: ['consistency', 'Lipschitz violation rate', 'smallest Lipschitz constant'],
        estimates: [c, lip.violationRate, lip.worstRatio],
      }),
      curveSet({
        x: dist,
        xLabel: 'distance between the rows (standardised)',
        yLabel: 'gap in score',
        series: [
          { name: 'pairs', y: order.map((i) => lip.gaps[i] ?? 0) },
          { name: `Lipschitz bound (L = ${L})`, y: dist.map((v) => Math.min(1, L * v)) },
        ],
      }),
    ],
  }
}

import { sameLength } from './_check.ts'

/** Anomaly scores with the threshold that flags outliers, and per-point evidence. */
export interface AnomalyScores {
  readonly kind: 'anomalyScores'
  readonly method: string
  /** Higher means more anomalous. */
  readonly scores: readonly number[]
  readonly threshold: number
  /** Per-point evidence: isolation path lengths, local reachability densities, Mahalanobis distances, ... */
  readonly evidence?: { readonly name: string; readonly values: readonly number[] }
}

/**
 * Anomaly scores, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the evidence does not match the scores.
 * @example
 * ```ts
 * import { anomalyScores } from 'whitebox-engine/exposition'
 * anomalyScores({ method: 'isolationForest', scores: [0.4, 0.9], threshold: 0.6 }).threshold // => 0.6
 * ```
 */
export function anomalyScores(record: Omit<AnomalyScores, 'kind'>): AnomalyScores {
  if (record.evidence !== undefined)
    sameLength('anomalyScores', 'evidence.values', record.evidence.values.length, record.scores.length)
  return { kind: 'anomalyScores', ...record }
}

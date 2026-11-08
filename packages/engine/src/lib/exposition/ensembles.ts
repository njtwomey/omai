/**
 * The exposition record of an ensemble: its members and how the combined prediction builds up stage by stage (bagging,
 * boosting, stacking), with a decision surface per stage where the inputs allow one.
 */

import { sameLength, seal } from './_check.ts'
import type { DecisionSurface } from './surfaces.ts'

/** The members of an ensemble and how its prediction builds up stage by stage. */
export interface EnsembleStages {
  readonly kind: 'ensembleStages'
  /** The method (bagging, random forest, AdaBoost, gradient boosting, ...). */
  readonly method: string
  /** Per member: its weight (AdaBoost's alpha, a learning rate, 1 for averaging). */
  readonly weights: readonly number[]
  /** Training score after each stage. */
  readonly trainScores: readonly number[]
  /** Validation or out-of-bag score after each stage, when measured. */
  readonly validationScores?: readonly number[]
  /** Per stage: the sample weights (AdaBoost) or residuals (gradient boosting) the next member was fitted to. */
  readonly targets?: readonly (readonly number[])[]
  /** Indices of the bootstrap rows each member saw (bagging, forests). */
  readonly bootstraps?: readonly (readonly number[])[]
  /** What the stage scores are ('training accuracy', 'log loss'). */
  readonly scoreName?: string
  /** Whether higher scores are better (accuracy) rather than lower (a loss). */
  readonly higherIsBetter?: boolean
  /** The ensemble's decision surface over two features, with its steps stage by stage (iterations are stage counts). */
  readonly surface?: DecisionSurface
}

/**
 * An ensemble's stages, with their shapes checked.
 *
 * @param stages - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when the per-stage columns differ in length.
 * @example
 * ```ts
 * import { ensembleStages } from 'omai-engine/exposition'
 * ensembleStages({ method: 'adaboost', weights: [0.8, 0.5], trainScores: [0.7, 0.85] }).method // => 'adaboost'
 * ```
 */
export function ensembleStages(stages: Omit<EnsembleStages, 'kind'>): EnsembleStages {
  const n = stages.weights.length
  sameLength('ensembleStages', 'trainScores', stages.trainScores.length, n)
  for (const field of ['validationScores', 'targets', 'bootstraps'] as const) {
    const column = stages[field]
    if (column !== undefined) sameLength('ensembleStages', field, column.length, n)
  }
  return seal({ kind: 'ensembleStages', ...stages })
}

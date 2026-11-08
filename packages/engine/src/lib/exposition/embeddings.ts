/**
 * Exposition records for representations: linear decompositions (PCA, factor analysis, ICA, NMF, PLS) with their
 * components, variance explained and scores, and the steps of a nonlinear embedding (t-SNE, UMAP, MDS, Isomap,
 * self-organising maps) from its start to the end.
 */

import { rowsOf, sameLength, seal } from './_check.ts'
import type { RowColours } from './rows.ts'

/** A linear decomposition (PCA, factor analysis, ICA, NMF, PLS): components, variance explained, scores. */
export interface ComponentsRecord {
  readonly kind: 'components'
  readonly method: string
  readonly features: readonly string[]
  /** components[k][f]: loading of feature f on component k. */
  readonly components: readonly (readonly number[])[]
  /**
   * What the components weight: the input features (default), the training rows (kernel methods' dual coefficients,
   * whose 'features' are rows and have no loadings worth drawing), or a vocabulary (topic models: each component a
   * distribution over words, best read as its top words).
   */
  readonly space?: 'features' | 'samples' | 'vocabulary'
  /** Variance explained by each component, and its ratio to the total. */
  readonly explainedVariance?: readonly number[]
  readonly explainedVarianceRatio?: readonly number[]
  /** scores[i][k]: row i in component coordinates. */
  readonly scores?: readonly (readonly number[])[]
  /** Reconstruction error with the first r components, r = 1, 2, ... */
  readonly reconstructionError?: readonly number[]
  /** What colours the rows: their class (an index per row, -1 unlabelled, with the class names) or a numeric target. */
  readonly colourBy?: RowColours
}

/**
 * A components record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when loadings, variances or scores do not match.
 * @example
 * ```ts
 * import { componentsRecord } from 'omai-engine/exposition'
 * componentsRecord({ method: 'pca', features: ['x', 'y'], components: [[0.6, 0.8]], explainedVarianceRatio: [0.9] }).components // => [[0.6, 0.8]]
 * ```
 */
export function componentsRecord(record: Omit<ComponentsRecord, 'kind'>): ComponentsRecord {
  const k = record.components.length
  rowsOf('components', 'components', record.components, record.features.length)
  for (const field of ['explainedVariance', 'explainedVarianceRatio'] as const) {
    const column = record[field]
    if (column !== undefined) sameLength('components', field, column.length, k)
  }
  if (record.scores !== undefined) rowsOf('components', 'scores', record.scores, k)
  return seal({ kind: 'components', ...record })
}

/** How a nonlinear embedding (t-SNE, UMAP, MDS, Isomap, SOM) moved from its start to the end. */
export interface EmbeddingSteps {
  readonly kind: 'embeddingSteps'
  readonly method: string
  /** positions[t][i]: row i's embedding after recorded step t (a subset of iterations). */
  readonly positions: readonly (readonly (readonly number[])[])[]
  /** The iteration number of each recorded step. */
  readonly iterations: readonly number[]
  /** The objective (KL divergence, cross-entropy, stress) at each recorded step. */
  readonly objective?: readonly number[]
  /** The neighbour graph the method used, as edges with weights. */
  readonly graph?: readonly (readonly [number, number, number])[]
  /** Per-row bandwidths (t-SNE's sigmas from the perplexity, UMAP's local scales). */
  readonly bandwidths?: readonly number[]
  /** What colours the rows: their class (an index per row, -1 unlabelled, with the class names) or a numeric target. */
  readonly colourBy?: RowColours
}

/**
 * Embedding steps, with their shapes checked.
 *
 * @param steps - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when per-step columns differ in length.
 * @example
 * ```ts
 * import { embeddingSteps } from 'omai-engine/exposition'
 * embeddingSteps({ method: 'tsne', positions: [[[0, 0], [1, 1]]], iterations: [0] }).iterations // => [0]
 * ```
 */
export function embeddingSteps(steps: Omit<EmbeddingSteps, 'kind'>): EmbeddingSteps {
  const t = steps.positions.length
  sameLength('embeddingSteps', 'iterations', steps.iterations.length, t)
  if (steps.objective !== undefined) sameLength('embeddingSteps', 'objective', steps.objective.length, t)
  return seal({ kind: 'embeddingSteps', ...steps })
}

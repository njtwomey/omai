/**
 * The closed-form route to word vectors (Levy, Goldberg and Dagan 2015): the windowed co-occurrence matrix, its shifted
 * positive pointwise mutual information with smoothed context counts, and a truncated SVD with the singular values
 * split between words and contexts. Skip-gram with negative sampling implicitly factorises the same matrix
 * (shifted by log k), so this is the comparison for the trained models. The three steps are `text/representations.ts`'s;
 * this composes them into one embedding.
 */

import { MdmlError } from '../base/index.ts'
import { cooccurrence, pmi, svdWordVectors, type WordMatrix } from '../text/index.ts'

/** Options of {@link ppmiEmbedding}. */
export interface PpmiOptions {
  /** Context words on each side (default 5). */
  readonly window?: number
  /** The dimension (default 50, at most the vocabulary size). */
  readonly dimension?: number
  /** The context-distribution exponent of PMI (default 0.75). */
  readonly alpha?: number
  /** The shift log(shift) subtracted from PMI before the positive part; the number of negatives k of skip-gram (default 1). */
  readonly shift?: number
  /** The singular values enter as S^power (default 0.5; 0 drops them, 1 is the plain SVD). */
  readonly power?: number
  /** Weight co-occurrences by 1 / distance (default false). */
  readonly harmonic?: boolean
  /** Words seen fewer times are dropped (default 1). */
  readonly minCount?: number
}

/**
 * Word vectors in closed form: PPMI of the window co-occurrence counts and the rank-d SVD.
 *
 * @param sentences - Token lists.
 * @param options - Window, dimension, context smoothing, shift and eigenvalue weighting.
 * @returns The word vectors, words in code point order.
 * @throws MdmlError with code `value` for a dimension above the vocabulary size.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { ppmiEmbedding } from 'mdml-engine/embedding'
 * ppmiEmbedding([['a', 'b', 'c', 'a', 'b', 'c']], { dimension: 2, window: 1 }).words // => ['a', 'b', 'c']
 * ```
 */
export function ppmiEmbedding(sentences: readonly (readonly string[])[], options: PpmiOptions = {}): WordMatrix {
  const counts = cooccurrence(sentences, {
    window: options.window ?? 5,
    weighting: options.harmonic === true ? 'harmonic' : 'uniform',
    minCount: options.minCount ?? 1,
  })
  const d = options.dimension ?? 50
  if (d > counts.words.length) throw new MdmlError('value', 'ppmiEmbedding: the dimension exceeds the vocabulary')
  const m = pmi(counts, { positive: true, shift: options.shift ?? 1, alpha: options.alpha ?? 0.75 })
  return svdWordVectors(m, d, { power: options.power ?? 0.5 })
}

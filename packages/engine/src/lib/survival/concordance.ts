/**
 * Harrell's concordance index for survival predictions: among the pairs of subjects whose order of failure is known
 * (the one who failed first was seen to fail), the share the model orders correctly, counting tied predictions as
 * one half. 0.5 is a coin flip and 1 a perfect ranking; it is the AUC of a survival model.
 */

import { MdmlError } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { survivalSample } from './data.ts'

/** The result of {@link concordanceIndex}. */
export interface Concordance {
  /** `(concordant + tied / 2) / comparable`. */
  readonly concordance: number
  /** Comparable pairs ordered correctly by the score. */
  readonly concordant: number
  /** Comparable pairs ordered the wrong way. */
  readonly discordant: number
  /** Comparable pairs with equal scores. */
  readonly tied: number
  /** Pairs whose order of failure is known. */
  readonly comparable: number
}

/** Options of {@link concordanceIndex}. */
export interface ConcordanceOptions {
  /**
   * What the score means: `'risk'` (default; a higher score predicts an earlier event, like a Cox linear predictor) or
   * `'time'` (a higher score predicts a later event, like an AFT median).
   */
  readonly score?: 'risk' | 'time'
}

/**
 * Harrell's C: over the pairs `(i, j)` with `time_i < time_j` and an event at `i` (or equal times with an event at `i`
 * only, as R's `survival::concordance` counts them), a pair is concordant when the score ranks `i` as the riskier.
 * Pairs of equal times with two events are not comparable.
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param score - The prediction for every subject.
 * @param options - Whether the score is a risk (default) or a time.
 * @returns The index and the pair counts.
 * @throws MdmlError with code `value` when no pair is comparable.
 * @example
 * ```ts
 * import { concordanceIndex } from 'mdml-engine/survival'
 * concordanceIndex([1, 2, 3, 4], [1, 1, 0, 1], [4, 3, 2, 1]).concordance // => 1
 * concordanceIndex([1, 2, 3, 4], [1, 1, 0, 1], [1, 2, 3, 4]).concordance // => 0
 * ```
 */
export function concordanceIndex(
  time: TensorLike,
  event: TensorLike | undefined,
  score: TensorLike,
  options: ConcordanceOptions = {},
): Concordance {
  const sample = survivalSample(time, event, 'concordanceIndex')
  const s = toFlat(asTensor(score))
  if (s.length !== sample.n) throw new MdmlError('shape', `concordanceIndex: ${s.length} scores for ${sample.n} times`)
  const sign = options.score === 'time' ? -1 : 1
  let concordant = 0
  let discordant = 0
  let tied = 0
  for (let i = 0; i < sample.n; i++) {
    if (sample.event[i] !== 1) continue
    const ti = sample.time[i] ?? 0
    const si = sign * (s[i] ?? 0)
    for (let j = 0; j < sample.n; j++) {
      if (i === j) continue
      const tj = sample.time[j] ?? 0
      const comparable = ti < tj || (ti === tj && sample.event[j] === 0)
      if (!comparable) continue
      const sj = sign * (s[j] ?? 0)
      if (si > sj) concordant++
      else if (si < sj) discordant++
      else tied++
    }
  }
  const comparable = concordant + discordant + tied
  if (comparable === 0) throw new MdmlError('value', 'concordanceIndex: no comparable pairs')
  return { concordance: (concordant + tied / 2) / comparable, concordant, discordant, tied, comparable }
}

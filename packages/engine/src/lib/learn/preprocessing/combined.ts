/**
 * Combined samplers, imbalanced-learn style: over-sample the minority with SMOTE, then clean the result, with edited
 * nearest neighbours (`smoteEnn`) or by removing Tomek links (`smoteTomek`). The cleaning is applied to every class, so
 * it also removes synthetic samples that landed in the wrong region. Each is a chain of two resamplers, and the
 * indices of the result point back to the original rows (-1 for a synthetic one).
 */

import { OmaiError } from '../../base/index.ts'
import type { LearnData } from '../estimator.ts'
import { type Resampled, type Resampler, type SamplingStrategy, smote, tomekLinks } from './resampling.ts'
import { editedNearestNeighbours } from './undersampling.ts'

/**
 * Run resamplers one after the other, each on the output of the one before, with indices composed so that they refer
 * to the rows of the original data.
 *
 * @param name - The name of the chain.
 * @param steps - The resamplers, in order (at least one).
 * @returns The resampler.
 * @throws OmaiError with code `value` for an empty chain.
 * @example
 * ```ts
 * import { chainResamplers, randomUnderSampler, tomekLinks } from 'omai-engine/learn/preprocessing'
 * const r = chainResamplers('clean', [tomekLinks(), randomUnderSampler()]).fitResample({
 *   x: [[0], [0.4], [0.6], [3], [3.2], [3.4]],
 *   y: [0, 0, 1, 0, 0, 0],
 * })
 * r.y.shape // => [2]
 * ```
 */
export function chainResamplers(name: string, steps: readonly Resampler[]): Resampler {
  if (steps.length === 0) throw new OmaiError('value', 'chainResamplers: needs at least one resampler')
  return {
    name,
    kind: 'resampler',
    fitResample: (data, fit) => {
      let current: LearnData = data
      let indices: Int32Array | undefined
      let last: Resampled | undefined
      for (const step of steps) {
        last = step.fitResample(current, fit)
        const before = indices
        indices = before === undefined ? last.indices : last.indices.map((i) => (i < 0 ? -1 : (before[i] ?? -1)))
        current = { ...current, x: last.x, y: last.y }
      }
      return { x: (last as Resampled).x, y: (last as Resampled).y, indices: indices as Int32Array }
    },
  }
}

/** Options of {@link smoteEnn} and {@link smoteTomek}. */
export interface SmoteCombinedOptions {
  /** The sampling strategy of the SMOTE step (default `'auto'`). */
  readonly samplingStrategy?: SamplingStrategy
  /** The over-sampler (default {@link smote} with the strategy above). */
  readonly smote?: Resampler
}

/** Options of {@link smoteEnn}. */
export interface SmoteEnnOptions extends SmoteCombinedOptions {
  /** The cleaner (default {@link editedNearestNeighbours} over all the classes). */
  readonly enn?: Resampler
}

/**
 * SMOTE followed by edited nearest neighbours (Batista et al. 2004; imblearn SMOTEENN): after balancing with SMOTE,
 * every sample whose three nearest neighbours do not all share its class is removed, which cleans both classes.
 *
 * @param options - The strategy and the two steps (default SMOTE and ENN over all classes).
 * @returns The resampler.
 * @concept class-imbalance
 * @index estimator SMOTE then edited nearest neighbours.
 * @example
 * ```ts
 * import { smote, smoteEnn } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.1], [0.2], [0.3], [0.4], [0.5], [3], [3.1], [3.2], [3.3]]
 * const r = smoteEnn({ smote: smote({ kNeighbors: 2 }) }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1] }, { key: key(0) })
 * toFlat(r.y).length <= 12 // => true
 * ```
 */
export function smoteEnn(options: SmoteEnnOptions = {}): Resampler {
  return chainResamplers('smote-enn', [
    options.smote ?? smote({ samplingStrategy: options.samplingStrategy ?? 'auto' }),
    options.enn ?? editedNearestNeighbours({ samplingStrategy: 'all' }),
  ])
}

/** Options of {@link smoteTomek}. */
export interface SmoteTomekOptions extends SmoteCombinedOptions {
  /** The cleaner (default {@link tomekLinks} over all the classes). */
  readonly tomek?: Resampler
}

/**
 * SMOTE followed by Tomek link removal (Batista et al. 2003; imblearn SMOTETomek): after balancing with SMOTE, both
 * members of every pair of mutual nearest neighbours of different classes are removed.
 *
 * @param options - The strategy and the two steps (default SMOTE and Tomek links over all classes).
 * @returns The resampler.
 * @concept class-imbalance
 * @index estimator SMOTE then Tomek link removal.
 * @example
 * ```ts
 * import { smote, smoteTomek } from 'omai-engine/learn/preprocessing'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = [[0], [0.1], [0.2], [0.3], [0.4], [0.5], [3], [3.1], [3.2], [3.3]]
 * const r = smoteTomek({ smote: smote({ kNeighbors: 2 }) }).fitResample({ x, y: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1] }, { key: key(0) })
 * toFlat(r.y).length <= 12 // => true
 * ```
 */
export function smoteTomek(options: SmoteTomekOptions = {}): Resampler {
  return chainResamplers('smote-tomek', [
    options.smote ?? smote({ samplingStrategy: options.samplingStrategy ?? 'auto' }),
    options.tomek ?? tomekLinks({ samplingStrategy: 'all' }),
  ])
}

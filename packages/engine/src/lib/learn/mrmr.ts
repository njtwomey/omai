/**
 * Minimum-redundancy maximum-relevance feature selection (Peng, Long and Ding 2005): choose features one at a time that
 * carry the most information about the target while sharing the least with the features already chosen. Each step adds the
 * feature maximising relevance minus mean redundancy (the MID criterion) or relevance over mean redundancy (MIQ), with
 * mutual information estimated on equal-frequency bins or by nearest-neighbour estimators. A stepped algorithm, so the
 * order of selection and the scores that decided it are on the tape.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { ksgMutualInformation, mixedMutualInformation, plugInMutualInformation } from '../information/index.ts'
import { ranksOf } from '../stats/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { columnOf, matrixView, targetVector } from './validation.ts'

/** How relevance and redundancy are combined: their difference (MID) or their quotient (MIQ). */
export type MrmrCriterion = 'mid' | 'miq'

/** Options of {@link mrmrAlgorithm}. */
export interface MrmrOptions {
  /** `'mid'` (default) or `'miq'`. */
  readonly criterion?: MrmrCriterion
  /** `'binned'` (default): plug-in mutual information on equal-frequency bins; `'ksg'`: Kraskov and Ross nearest-neighbour estimates on the raw values. */
  readonly estimator?: 'binned' | 'ksg'
  /** Bins per variable for the binned estimator (default 10); variables with fewer distinct values keep them. */
  readonly bins?: number
  /** Whether the target is a class label (default: detected, integer-valued with at most 20 distinct values). */
  readonly discreteTarget?: boolean
}

/** State of {@link mrmrAlgorithm}. */
export interface MrmrState {
  /** The features chosen so far, in order. */
  readonly selected: readonly number[]
  /** The criterion value of each chosen feature at the step it was chosen (its relevance at the first). */
  readonly scores: readonly number[]
  /** The relevance I(f; y) of every feature. */
  readonly relevance: readonly number[]
  /** For each feature, the sum of I(f; s) over the chosen features s. */
  readonly redundancy: readonly number[]
}

/** Equal-frequency codes of a column (the values themselves when there are few distinct ones). */
function binCodes(v: ArrayLike<number>, bins: number): number[] {
  const distinct = new Set(Array.from(v))
  if (distinct.size <= bins) return Array.from(v)
  const r = ranksOf(Array.from(v), 'ordinal').data
  return Array.from(r, (rank) => Math.min(bins - 1, Math.floor(((rank - 1) * bins) / v.length)))
}

/**
 * mRMR as a stepped algorithm, one chosen feature per step. The first step takes the feature of greatest relevance
 * I(f; y); each later step scores every unchosen feature by relevance - mean_s I(f; s) (`mid`) or relevance / mean_s I(f; s)
 * (`miq`) over the chosen features s and takes the best, logging `mrmr/score` and `mrmr/chosen`.
 *
 * @param x - The features `[n, d]`.
 * @param y - The target `[n]` (class labels or a continuous response).
 * @param k - How many features to choose.
 * @param options - Criterion, estimator, bins.
 * @returns The algorithm over {@link MrmrState}.
 * @concept feature-selection
 * @example
 * ```ts
 * import { mrmrAlgorithm } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0, 0], [1, 1], [0, 0], [1, 1], [0, 1], [1, 0]]
 * runAlgorithm(mrmrAlgorithm(x, [0, 1, 0, 1, 0, 1], 1), { maxSteps: 1 }).state.selected // => [0]
 * ```
 */
export function mrmrAlgorithm(
  x: TensorLike,
  y: TensorLike,
  k: number,
  options: MrmrOptions = {},
): Algorithm<MrmrState> {
  const m = matrixView(x, 'mrmr')
  const yv = targetVector(y, 'mrmr', m.n)
  const { n, d } = m
  if (!(Number.isInteger(k) && k >= 1 && k <= d)) throw new OmaiError('value', `mrmr: k must be between 1 and ${d}`)
  const criterion = options.criterion ?? 'mid'
  const estimator = options.estimator ?? 'binned'
  const bins = options.bins ?? 10
  const distinctY = new Set(Array.from(yv))
  const discreteY = options.discreteTarget ?? (distinctY.size <= 20 && Array.from(yv).every((v) => Number.isInteger(v)))
  const raw = Array.from({ length: d }, (_, j) => columnOf(m, j))
  const coded = estimator === 'binned' ? raw.map((c) => binCodes(c, bins)) : []
  const yCodes = estimator === 'binned' ? (discreteY ? Array.from(yv) : binCodes(yv, bins)) : []
  const relevanceOf = (j: number): number =>
    estimator === 'binned'
      ? plugInMutualInformation(coded[j] ?? [], yCodes)
      : discreteY
        ? mixedMutualInformation(raw[j] as Float64Array, Array.from(yv))
        : ksgMutualInformation(raw[j] as Float64Array, yv)
  const redundancyOf = (a: number, b: number): number =>
    estimator === 'binned'
      ? plugInMutualInformation(coded[a] ?? [], coded[b] ?? [])
      : ksgMutualInformation(raw[a] as Float64Array, raw[b] as Float64Array)
  return {
    name: 'mrmr',
    config: { n, d, k, criterion, estimator, bins, discreteTarget: discreteY } as Json,
    schema: channels({
      metrics: { score: { description: 'the criterion value of the chosen feature' } },
      tensors: { chosen: { role: 'state', description: 'indices of the features chosen so far' } },
    }),
    init: () => ({
      selected: [],
      scores: [],
      relevance: Array.from({ length: d }, (_, j) => relevanceOf(j)),
      redundancy: new Array<number>(d).fill(0),
    }),
    step: (s) => {
      let redundancy = s.redundancy
      const last = s.selected[s.selected.length - 1]
      if (last !== undefined)
        redundancy = s.redundancy.map((r, j) => (s.selected.includes(j) ? r : r + redundancyOf(j, last)))
      let best = -1
      let bestScore = Number.NEGATIVE_INFINITY
      for (let j = 0; j < d; j++) {
        if (s.selected.includes(j)) continue
        const rel = s.relevance[j] ?? 0
        const red = s.selected.length === 0 ? 0 : (redundancy[j] ?? 0) / s.selected.length
        const score = s.selected.length === 0 ? rel : criterion === 'mid' ? rel - red : rel / (red + 1e-10)
        if (score > bestScore) {
          bestScore = score
          best = j
        }
      }
      return { selected: [...s.selected, best], scores: [...s.scores, bestScore], relevance: s.relevance, redundancy }
    },
    done: (s) => (s.selected.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { score: s.scores[s.scores.length - 1] ?? 0 },
      tensors: { chosen: { role: 'state', value: s.selected } },
    }),
    serialise: (s) => ({
      selected: [...s.selected],
      scores: [...s.scores],
      relevance: [...s.relevance],
      redundancy: [...s.redundancy],
    }),
    deserialise: (j) => j as unknown as MrmrState,
  }
}

/**
 * Choose k features by mRMR ({@link mrmrAlgorithm} run to the end).
 *
 * @param x - The features `[n, d]`.
 * @param y - The target.
 * @param k - How many to choose.
 * @param options - Criterion, estimator, bins.
 * @returns The chosen features in order, with the criterion at each choice and every feature's relevance.
 * @concept feature-selection
 * @example
 * ```ts
 * import { mrmr } from 'omai-engine/learn'
 * mrmr([[0, 0], [1, 1], [0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1, 0, 1], 2).selected // => [0, 1]
 * ```
 */
export function mrmr(
  x: TensorLike,
  y: TensorLike,
  k: number,
  options: MrmrOptions = {},
): { readonly selected: readonly number[]; readonly scores: readonly number[]; readonly relevance: readonly number[] } {
  const r = runAlgorithm(mrmrAlgorithm(x, y, k, options), { maxSteps: k })
  return { selected: r.state.selected, scores: r.state.scores, relevance: r.state.relevance }
}

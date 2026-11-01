/**
 * Stream-based active learning: rows arrive one at a time and each is labelled only when the model is unsure of it
 * (Cohn, Atlas and Ladner 1994). `streamActiveLearningAlgorithm` takes one arriving row per step, with the uncertainty
 * measure as a slot (`streamUncertainties`); `streamActiveLearning` is that run to the end. The pool-based case is
 * `activeLearningAlgorithm` in `active-learning.ts`.
 */

import { foldIn } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import { appendRows } from './active-learning.ts'
import { streamUncertainties } from './active-learning-options.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireVerb } from './estimator.ts'
import { type Scoring, scorersOf } from './scoring.ts'
import { fromView, type MatrixView, matrixView, takeRows, targetVector } from './validation.ts'

const stackRows = (a: MatrixView, b: MatrixView): MatrixView =>
  matrixView(appendRows(fromView(a), fromView(b)), 'streamActiveLearning')

/** Options of {@link streamActiveLearning}. */
export interface StreamOptions extends FitOptions {
  readonly labeled: LearnData
  /** Rows arriving in order `[T, d]`. */
  readonly stream: TensorLike
  readonly oracle: (index: number) => number
  /** Query a row when its uncertainty exceeds this (default 0.3). */
  readonly threshold?: number
  /** The most labels bought (default unlimited). */
  readonly budget?: number
  /** The uncertainty measure, a value of `streamUncertainties.space` (default least confidence). */
  readonly uncertainty?: SpaceValue
  /** Score the model on this data after every row (logs `stream/score`). */
  readonly evaluation?: LearnData
  readonly scoring?: Scoring
}

/** The paths {@link streamActiveLearningAlgorithm} logs. */
export type StreamPath = 'stream/row' | 'stream/uncertainty' | 'stream/queried' | 'stream/labeled' | 'stream/score'

/** What a stepped stream learner logs, by path: the row, its uncertainty, whether it was bought, labels so far. */
export const streamContract: Contract<unknown, StreamPath> = defineContract({
  'stream/row': scalar({ description: 'the index of the arriving row' }),
  'stream/uncertainty': scalar({ description: 'the model’s uncertainty about the row, in [0, 1]', direction: 'lower' }),
  'stream/queried': scalar({ description: '1 when the row was labelled, else 0' }),
  'stream/labeled': scalar({ description: 'labels bought so far' }),
  'stream/score': scalar({
    description: 'the model’s score on the evaluation data after this row',
    direction: 'higher',
  }),
})

/**
 * The state of {@link streamActiveLearningAlgorithm}: the arriving row, the labelled data, and the stream indices
 * queried so far. The model is not in the state; it is refitted from the labelled data when needed.
 */
export interface StreamState {
  readonly t: number
  readonly x: MatrixView
  readonly y: readonly number[]
  readonly queried: readonly number[]
  /** The evaluation score after each row (when an evaluation set was given). */
  readonly scores: readonly number[]
}

function streamRun(
  estimator: Estimator,
  options: StreamOptions,
): {
  readonly algorithm: Algorithm<StreamState, StreamPath>
  readonly modelOf: (s: StreamState, parent?: FitOptions['parent']) => Fitted
  readonly rows: number
} {
  const stream = matrixView(options.stream, 'streamActiveLearning')
  const x0 = matrixView(options.labeled.x, 'streamActiveLearning', { nFeatures: stream.d })
  const y0 = targetVector(options.labeled.y ?? [], 'streamActiveLearning', x0.n)
  const threshold = options.threshold ?? 0.3
  const budget = options.budget ?? Number.POSITIVE_INFINITY
  const measure = streamUncertainties.make(options.uncertainty)
  const scorer = options.evaluation === undefined ? undefined : Object.values(scorersOf(options.scoring))[0]
  const key = keyFor(options, 'streamActiveLearning')
  // The model for a labelled set is a function of how many labels were bought, so a resumed run refits the same one.
  let cache: { readonly count: number; readonly fitted: Fitted } | undefined
  const modelOf = (s: StreamState, parent: FitOptions['parent'] = options.parent): Fitted => {
    if (cache !== undefined && cache.count === s.queried.length) return cache.fitted
    const fitted = estimator.fit(
      { x: fromView(s.x), y: [...s.y] },
      innerFitOptions(options, foldIn(key, s.queried.length), parent),
    )
    cache = { count: s.queried.length, fitted }
    return fitted
  }
  const algorithm: Algorithm<StreamState, StreamPath> = {
    name: 'stream-active-learning',
    config: { estimator: estimator.name, threshold, rows: stream.n },
    schema: channels({
      metrics: {
        uncertainty: { description: 'the uncertainty of the arriving row' },
        labeled: { description: 'labels bought' },
      },
    }),
    init: () => ({ t: 0, x: x0, y: Array.from(y0), queried: [], scores: [] }),
    step: (s, ctx) => {
      const row = takeRows(stream, [s.t])
      const u = measure(requireVerb(modelOf(s, ctx.run), 'predictProba', 'streamActiveLearning')(fromView(row)))[0] ?? 0
      const ask = !(u <= threshold) && s.queried.length < budget
      const next: StreamState = ask
        ? {
            t: s.t + 1,
            x: stackRows(s.x, row),
            y: [...s.y, options.oracle(s.t)],
            queried: [...s.queried, s.t],
            scores: s.scores,
          }
        : { ...s, t: s.t + 1 }
      ctx.log('stream/row', s.t)
      ctx.log('stream/uncertainty', u)
      ctx.log('stream/queried', ask ? 1 : 0)
      ctx.log('stream/labeled', next.queried.length)
      if (scorer === undefined || options.evaluation === undefined) return next
      const score = scorer.score(modelOf(next, ctx.run), options.evaluation)
      ctx.log('stream/score', score)
      return { ...next, scores: [...next.scores, score] }
    },
    done: (s) => (s.t >= stream.n ? 'completed' : undefined),
    observe: (s) => ({ metrics: { labeled: s.queried.length } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamState,
  }
  return { algorithm, modelOf, rows: stream.n }
}

/**
 * Stream-based selective sampling (Cohn, Atlas and Ladner 1994) as a stepped algorithm: one arriving row per step,
 * labelled when the current model's uncertainty on it exceeds the threshold and the budget allows. The state (the
 * labelled data and the indices bought) serialises; the model is refitted from the labelled data as needed, under a
 * key that depends only on how many labels were bought.
 *
 * @param estimator - The classifier.
 * @param options - labeled, stream, oracle, threshold, budget, uncertainty measure, evaluation data and key.
 * @returns The algorithm, one step per row.
 * @concept active-learning
 * @example
 * ```ts
 * import { streamActiveLearningAlgorithm, defineEstimator } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { tensor } from 'mdml-engine/tensor'
 * const logistic = defineEstimator(
 *   { name: 'l', kind: 'classifier', fit: () => ({}), methods: () => ({ classes: [0, 1], predictProba: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => { const p = 1 / (1 + Math.exp(-x.data[i])); return [1 - p, p] })) }) },
 *   {},
 * )
 * const alg = streamActiveLearningAlgorithm(logistic, { labeled: { x: [[-5], [5]], y: [0, 1] }, stream: [[4], [0.2], [-6], [-0.1]], oracle: () => 0 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.queried // => [1, 3]
 * ```
 */
export function streamActiveLearningAlgorithm(
  estimator: Estimator,
  options: StreamOptions,
): Algorithm<StreamState, StreamPath> {
  return streamRun(estimator, options).algorithm
}

/**
 * Stream-based selective sampling (Cohn, Atlas and Ladner 1994): rows arrive one at a time and each is labelled (and
 * the model refitted) only when the current model's uncertainty on it exceeds a threshold, until the budget is spent.
 * This is {@link streamActiveLearningAlgorithm} run to the end, a traced algorithm with one step per arriving row.
 *
 * @param estimator - The classifier.
 * @param options - labeled, stream, oracle, threshold, budget, key and tracing.
 * @returns The final model and the stream indices queried.
 * @example
 * ```ts
 * import { streamActiveLearning, defineEstimator } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
 * const logistic = defineEstimator(
 *   { name: 'l', kind: 'classifier', fit: () => ({}), methods: () => ({ classes: [0, 1], predictProba: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => { const p = 1 / (1 + Math.exp(-x.data[i])); return [1 - p, p] })) }) },
 *   {},
 * )
 * const r = streamActiveLearning(logistic, { labeled: { x: [[-5], [5]], y: [0, 1] }, stream: [[4], [0.2], [-6], [-0.1]], oracle: () => 0 })
 * r.queried // => [1, 3]
 * ```
 */
export function streamActiveLearning(
  estimator: Estimator,
  options: StreamOptions,
): { readonly fitted: Fitted; readonly queried: readonly number[] } {
  const { algorithm, modelOf, rows } = streamRun(estimator, options)
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'streamActiveLearning'), maxSteps: rows })
  return { fitted: modelOf(state), queried: state.queried }
}

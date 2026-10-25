/**
 * Active learning, where the model chooses which unlabelled examples to have labelled next so it learns from fewer
 * labels, modAL style. Uncertainty measures and query strategies (least confidence, margin, entropy, query by
 * committee, expected error reduction, ranked batches), pool-based `activeLearning` as a traced loop of fit, query and
 * label, and stream-based selective sampling.
 */

import { WhiteboxError } from '../base/index.ts'
import { bootstrapIndices, foldIn, type Key, permutation } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from './_shared.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireVerb } from './estimator.ts'
import { type Scorer, type Scoring, scorersOf } from './scoring.ts'
import { encodeLabels, fromView, type MatrixView, matrixView, takeRows, targetVector } from './validation.ts'

/** Class probabilities as rows. */
function probaRows(
  fitted: Fitted,
  x: TensorLike,
): { readonly p: Float64Array; readonly n: number; readonly k: number } {
  const out = requireVerb(fitted, 'predictProba', 'activeLearning')(x)
  return { p: Float64Array.from(toFlat(out)), n: out.shape[0] ?? 0, k: out.shape[1] ?? 1 }
}

/**
 * Least-confidence uncertainty, 1 - max_c p(c | x), of each row of probabilities (modAL's classifier_uncertainty).
 *
 * @param proba - Probabilities `[n, k]`.
 * @returns One score per row (higher is more uncertain).
 * @concept active-learning
 * @example
 * ```ts
 * import { leastConfidence } from 'whitebox-engine/learn'
 * leastConfidence([[0.9, 0.1], [0.5, 0.5]]) // => [0.09999999999999998, 0.5]
 * ```
 */
export function leastConfidence(proba: TensorLike): number[] {
  const m = matrixView(proba, 'leastConfidence')
  return Array.from(
    { length: m.n },
    (_, i) => 1 - m.data.subarray(i * m.d, (i + 1) * m.d).reduce((a, b) => Math.max(a, b), 0),
  )
}

/**
 * The margin between the two most probable classes of each row (modAL's classifier_margin; lower is more uncertain).
 *
 * @param proba - Probabilities `[n, k]`.
 * @returns One margin per row.
 * @example
 * ```ts
 * import { classMargin } from 'whitebox-engine/learn'
 * classMargin([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]).map((v) => Math.round(v * 100) / 100) // => [0.5, 0.05]
 * ```
 */
export function classMargin(proba: TensorLike): number[] {
  const m = matrixView(proba, 'classMargin')
  return Array.from({ length: m.n }, (_, i) => {
    const row = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)).sort((a, b) => b - a)
    return (row[0] ?? 0) - (row[1] ?? 0)
  })
}

/**
 * The Shannon entropy (nats) of each row of probabilities (modAL's classifier_entropy).
 *
 * @param proba - Probabilities `[n, k]`.
 * @returns One entropy per row.
 * @example
 * ```ts
 * import { predictiveEntropy } from 'whitebox-engine/learn'
 * predictiveEntropy([[1, 0], [0.5, 0.5]]) // => [0, Math.LN2]
 * ```
 */
export function predictiveEntropy(proba: TensorLike): number[] {
  const m = matrixView(proba, 'predictiveEntropy')
  return Array.from({ length: m.n }, (_, i) => {
    let s = 0
    for (let c = 0; c < m.d; c++) {
      const p = m.data[i * m.d + c] ?? 0
      if (p > 0) s -= p * Math.log(p)
    }
    return s
  })
}

/** What a query strategy sees: the current model (and committee), the labelled data and the pool. */
export interface QueryContext {
  readonly fitted: Fitted
  /** Committee members (for query by committee), fitted on resamples of the labelled data. */
  readonly committee: readonly Fitted[]
  readonly labeled: LearnData
  readonly pool: MatrixView
  readonly estimator: Estimator
  readonly key: Key
  readonly fitOptions: FitOptions
}

/**
 * The query-strategy contract: choose `n` rows of the unlabelled pool (indices into it) to send to the oracle.
 */
export interface QueryStrategy {
  readonly name: string
  /** Whether the strategy needs a committee. */
  readonly committee?: boolean
  select(ctx: QueryContext, n: number): number[]
}

/** The n best indices of scores (highest first; ties by index). */
function topIndices(scores: readonly number[], n: number): number[] {
  return scores
    .map((s, i) => ({ s, i }))
    .sort((a, b) => b.s - a.s || a.i - b.i)
    .slice(0, n)
    .map((e) => e.i)
}

function scoreStrategy(
  name: string,
  score: (ctx: QueryContext) => readonly number[],
  committee = false,
): QueryStrategy {
  return { name, committee, select: (ctx, n) => topIndices(score(ctx), n) }
}

/**
 * Uncertainty sampling by least confidence (Lewis and Gale 1994; modAL uncertainty_sampling).
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { uncertaintySampling } from 'whitebox-engine/learn'
 * uncertaintySampling().name // => 'uncertainty'
 * ```
 */
export function uncertaintySampling(): QueryStrategy {
  return scoreStrategy('uncertainty', (ctx) =>
    leastConfidence(requireVerb(ctx.fitted, 'predictProba', 'uncertaintySampling')(fromView(ctx.pool))),
  )
}

/**
 * Margin sampling (Scheffer et al. 2001; modAL margin_sampling): the rows whose two most likely classes are closest.
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { marginSampling } from 'whitebox-engine/learn'
 * marginSampling().name // => 'margin'
 * ```
 */
export function marginSampling(): QueryStrategy {
  return scoreStrategy('margin', (ctx) =>
    classMargin(requireVerb(ctx.fitted, 'predictProba', 'marginSampling')(fromView(ctx.pool))).map((v) => -v),
  )
}

/**
 * Entropy sampling (Settles 2009; modAL entropy_sampling): the rows with the most uncertain predictive distribution.
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { entropySampling } from 'whitebox-engine/learn'
 * entropySampling().name // => 'entropy'
 * ```
 */
export function entropySampling(): QueryStrategy {
  return scoreStrategy('entropy', (ctx) =>
    predictiveEntropy(requireVerb(ctx.fitted, 'predictProba', 'entropySampling')(fromView(ctx.pool))),
  )
}

/**
 * Random sampling, the passive-learning baseline.
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { randomSampling } from 'whitebox-engine/learn'
 * randomSampling().name // => 'random'
 * ```
 */
export function randomSampling(): QueryStrategy {
  return { name: 'random', select: (ctx, n) => toFlat(permutation(ctx.key, ctx.pool.n)).slice(0, n) }
}

function committeeVotes(ctx: QueryContext): { readonly classes: readonly number[]; readonly votes: number[][] } {
  const classes = encodeLabels(ctx.labeled.y ?? [], 'committee').classes
  const votes = ctx.committee.map((f) => toFlat(requireVerb(f, 'predict', 'committee')(fromView(ctx.pool))))
  return { classes, votes }
}

/**
 * Vote entropy (Dagan and Engelson 1995; modAL vote_entropy_sampling): query by committee, the rows whose members'
 * predicted labels disagree most.
 *
 * @returns The strategy (it uses the committee).
 * @example
 * ```ts
 * import { voteEntropySampling } from 'whitebox-engine/learn'
 * voteEntropySampling().committee // => true
 * ```
 */
export function voteEntropySampling(): QueryStrategy {
  return scoreStrategy(
    'vote-entropy',
    (ctx) => {
      const { classes, votes } = committeeVotes(ctx)
      return Array.from({ length: ctx.pool.n }, (_, i) => {
        const p = classes.map((c) => votes.filter((v) => v[i] === c).length / votes.length)
        return p.reduce((s, q) => (q > 0 ? s - q * Math.log(q) : s), 0)
      })
    },
    true,
  )
}

function committeeProba(ctx: QueryContext): {
  readonly mean: Float64Array
  readonly members: Float64Array[]
  readonly k: number
} {
  const classes = encodeLabels(ctx.labeled.y ?? [], 'committee').classes
  const k = classes.length
  const members = ctx.committee.map((f) => {
    const r = probaRows(f, fromView(ctx.pool))
    const fc = f.classes ?? classes
    const out = new Float64Array(ctx.pool.n * k)
    for (let i = 0; i < r.n; i++)
      fc.forEach((c, j) => {
        out[i * k + classes.indexOf(c)] = r.p[i * r.k + j] ?? 0
      })
    return out
  })
  const mean = new Float64Array(ctx.pool.n * k)
  for (const m of members)
    m.forEach((v, i) => {
      mean[i] = (mean[i] ?? 0) + v / members.length
    })
  return { mean, members, k }
}

/**
 * Consensus entropy (modAL consensus_entropy_sampling): the entropy of the committee's averaged probabilities.
 *
 * @returns The strategy (it uses the committee).
 * @example
 * ```ts
 * import { consensusEntropySampling } from 'whitebox-engine/learn'
 * consensusEntropySampling().name // => 'consensus-entropy'
 * ```
 */
export function consensusEntropySampling(): QueryStrategy {
  return scoreStrategy(
    'consensus-entropy',
    (ctx) => {
      const { mean, k } = committeeProba(ctx)
      return predictiveEntropy(fromStorage(mean, [ctx.pool.n, k], 'float64'))
    },
    true,
  )
}

/**
 * Maximum disagreement (McCallum and Nigam 1998; modAL max_disagreement_sampling): the largest Kullback-Leibler
 * divergence of a committee member's probabilities from the consensus.
 *
 * @returns The strategy (it uses the committee).
 * @example
 * ```ts
 * import { maxDisagreementSampling } from 'whitebox-engine/learn'
 * maxDisagreementSampling().name // => 'max-disagreement'
 * ```
 */
export function maxDisagreementSampling(): QueryStrategy {
  return scoreStrategy(
    'max-disagreement',
    (ctx) => {
      const { mean, members, k } = committeeProba(ctx)
      return Array.from({ length: ctx.pool.n }, (_, i) =>
        members.reduce((best, m) => {
          let kl = 0
          for (let c = 0; c < k; c++) {
            const p = m[i * k + c] ?? 0
            const q = mean[i * k + c] ?? 0
            if (p > 0) kl += p * Math.log(p / q)
          }
          return Math.max(best, kl)
        }, 0),
      )
    },
    true,
  )
}

/**
 * Expected error reduction (Roy and McCallum 2001; modAL expected_error_reduction): for each candidate and each
 * possible label, refit with it and sum the model's uncertainty (least confidence, or entropy for `'log'`) over the
 * rest of the pool, weighted by the current probability of the label; query the candidates with the lowest expected
 * error. Costs one refit per candidate and label.
 *
 * @param options - loss (`'binary'`, default, or `'log'`).
 * @returns The strategy.
 * @example
 * ```ts
 * import { expectedErrorReduction } from 'whitebox-engine/learn'
 * expectedErrorReduction({ loss: 'log' }).name // => 'expected-error-reduction'
 * ```
 */
export function expectedErrorReduction(options: { readonly loss?: 'binary' | 'log' } = {}): QueryStrategy {
  return {
    name: 'expected-error-reduction',
    select: (ctx, n) => {
      const labeledX = matrixView(ctx.labeled.x, 'expectedErrorReduction')
      const labeledY = targetVector(ctx.labeled.y ?? [], 'expectedErrorReduction', labeledX.n)
      const classes = Array.from(new Set(labeledY)).sort((a, b) => a - b)
      const current = probaRows(ctx.fitted, fromView(ctx.pool))
      const fc = ctx.fitted.classes ?? classes
      const errors = Array.from({ length: ctx.pool.n }, (_, i) => {
        const rest = takeRows(
          ctx.pool,
          Array.from({ length: ctx.pool.n }, (_, r) => r).filter((r) => r !== i),
        )
        const row = takeRows(ctx.pool, [i])
        return classes.reduce((total, c) => {
          const refit = ctx.estimator.fit(
            { x: fromView(stackRows(labeledX, row)), y: [...labeledY, c] },
            innerFitOptions(ctx.fitOptions, foldIn(ctx.key, i), ctx.fitOptions.parent),
          )
          const p = requireVerb(refit, 'predictProba', 'expectedErrorReduction')(fromView(rest))
          const loss = options.loss === 'log' ? predictiveEntropy(p) : leastConfidence(p)
          const pc = current.p[i * current.k + fc.indexOf(c)] ?? 0
          return total + pc * loss.reduce((a, b) => a + b, 0)
        }, 0)
      })
      return topIndices(
        errors.map((e) => -e),
        n,
      )
    },
  }
}

function stackRows(a: MatrixView, b: MatrixView): MatrixView {
  const out = new Float64Array((a.n + b.n) * a.d)
  out.set(a.data)
  out.set(b.data, a.n * a.d)
  return { n: a.n + b.n, d: a.d, data: out }
}

/**
 * Ranked batch-mode sampling (Cardoso et al. 2017; modAL uncertainty_batch_sampling): build a batch greedily, each
 * pick maximising alpha (1 - similarity to the labelled and already picked rows) + (1 - alpha) uncertainty, with
 * alpha the unlabelled share and similarity 1 / (1 + the Euclidean distance to the nearest of them).
 *
 * @returns The strategy.
 * @example
 * ```ts
 * import { rankedBatchSampling } from 'whitebox-engine/learn'
 * rankedBatchSampling().name // => 'ranked-batch'
 * ```
 */
export function rankedBatchSampling(): QueryStrategy {
  return {
    name: 'ranked-batch',
    select: (ctx, n) => {
      const uncertainty = leastConfidence(
        requireVerb(ctx.fitted, 'predictProba', 'rankedBatchSampling')(fromView(ctx.pool)),
      )
      let labeled = matrixView(ctx.labeled.x, 'rankedBatchSampling')
      const mask = new Array<boolean>(ctx.pool.n).fill(true)
      const chosen: number[] = []
      for (let t = 0; t < Math.min(n, ctx.pool.n); t++) {
        const candidates = mask.flatMap((m, i) => (m ? [i] : []))
        const alpha = candidates.length / (candidates.length + labeled.n)
        let best = -1
        let bestScore = Number.NEGATIVE_INFINITY
        for (const i of candidates) {
          let dmin = Number.POSITIVE_INFINITY
          for (let r = 0; r < labeled.n; r++) {
            let s = 0
            for (let j = 0; j < ctx.pool.d; j++)
              s += ((ctx.pool.data[i * ctx.pool.d + j] ?? 0) - (labeled.data[r * labeled.d + j] ?? 0)) ** 2
            dmin = Math.min(dmin, Math.sqrt(s))
          }
          const score = alpha * (1 - 1 / (1 + dmin)) + (1 - alpha) * (uncertainty[i] ?? 0)
          if (score > bestScore) {
            bestScore = score
            best = i
          }
        }
        if (best < 0) break
        chosen.push(best)
        mask[best] = false
        labeled = stackRows(labeled, takeRows(ctx.pool, [best]))
      }
      return chosen
    },
  }
}

/** Options of {@link activeLearning}. */
export interface ActiveLearningOptions extends FitOptions {
  /** The initial labelled data. */
  readonly labeled: LearnData
  /** The unlabelled pool `[m, d]`. */
  readonly pool: TensorLike
  /** Labels for pool rows, by their indices (the human or the simulator). */
  readonly oracle: (indices: readonly number[]) => readonly number[]
  readonly strategy: QueryStrategy
  /** Rows queried per round (default 1). */
  readonly batchSize?: number
  /** The most labels bought (default the pool size). */
  readonly budget?: number
  /** Committee members for committee strategies (default 5), each fitted on a bootstrap of the labelled data. */
  readonly committeeSize?: number
  /** Score each round's model on this data (for learning curves). */
  readonly evaluation?: LearnData
  readonly scoring?: Scoring
}

/** The result of {@link activeLearning}. */
export interface ActiveLearningResult {
  /** The model fitted on all labelled data at the end. */
  readonly fitted: Fitted
  /** Pool indices in the order they were queried. */
  readonly queried: readonly number[]
  /** The labelled data at the end. */
  readonly labeled: LearnData
  /** The evaluation score after each round (when an evaluation set was given). */
  readonly scores: readonly number[]
}

/**
 * Pool-based active learning as a traced algorithm (Settles 2009; modAL's ActiveLearner and Committee): each round
 * fits the model (and a bagged committee if the strategy needs one) on the labelled data, asks the strategy for a
 * batch from the pool, buys the labels from the oracle, and moves them to the labelled set; it stops when the budget
 * is spent or the pool is empty. Each fit nests as a child run.
 *
 * @param estimator - The classifier.
 * @param options - labeled, pool, oracle, strategy, batchSize, budget, committeeSize, evaluation, scoring, key and tracing.
 * @returns The final model, the queried indices, the labelled data and the per-round scores.
 * @example
 * ```ts
 * import { activeLearning, uncertaintySampling } from 'whitebox-engine/learn'
 * import { defineEstimator } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
 * const threshold = defineEstimator(
 *   { name: 't', kind: 'classifier', fit: () => ({}), methods: () => ({ classes: [0, 1], predictProba: (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => { const p = 1 / (1 + Math.exp(-x.data[i])); return [1 - p, p] })) }) },
 *   {},
 * )
 * const pool = [[-3], [0.1], [3], [-0.2]]
 * const r = activeLearning(threshold, { labeled: { x: [[-5], [5]], y: [0, 1] }, pool, oracle: (idx) => idx.map((i) => (pool[i][0] > 0 ? 1 : 0)), strategy: uncertaintySampling(), budget: 2 })
 * r.queried // => [1, 3]
 * ```
 */
export function activeLearning(estimator: Estimator, options: ActiveLearningOptions): ActiveLearningResult {
  const pool = matrixView(options.pool, 'activeLearning')
  const labeled0 = matrixView(options.labeled.x, 'activeLearning', { nFeatures: pool.d })
  const y0 = targetVector(options.labeled.y ?? [], 'activeLearning', labeled0.n)
  const batch = options.batchSize ?? 1
  if (!(Number.isInteger(batch) && batch >= 1))
    throw new WhiteboxError('value', 'activeLearning: batchSize must be >= 1')
  const budget = Math.min(options.budget ?? pool.n, pool.n)
  const key = keyFor(options, 'activeLearning')
  const scorer: Scorer | undefined =
    options.evaluation === undefined ? undefined : Object.values(scorersOf(options.scoring))[0]
  type State = {
    readonly x: MatrixView
    readonly y: readonly number[]
    readonly remaining: readonly number[]
    readonly queried: readonly number[]
    readonly fitted?: Fitted
    readonly scores: readonly number[]
  }
  const fitAll = (s: State, k: Key, parent: FitOptions['parent']): Fitted =>
    estimator.fit({ x: fromView(s.x), y: [...s.y] }, innerFitOptions(options, k, parent))
  const algorithm: Algorithm<State> = {
    name: 'active-learning',
    config: { estimator: estimator.name, strategy: options.strategy.name, batchSize: batch, budget },
    schema: channels({
      metrics: {
        labeled: { description: 'labelled rows' },
        score: { description: 'the evaluation score', direction: 'higher' },
      },
      messages: { query: { description: 'pool rows sent to the oracle', payload: { indices: 'number[]' } } },
    }),
    init: () => ({
      x: labeled0,
      y: Array.from(y0),
      remaining: Array.from({ length: pool.n }, (_, i) => i),
      queried: [],
      scores: [],
    }),
    step: (s, ctx) => {
      const fitted = fitAll(s, foldIn(ctx.key, 0), ctx.run)
      const sub = takeRows(pool, s.remaining)
      const committee =
        options.strategy.committee === true
          ? Array.from({ length: options.committeeSize ?? 5 }, (_, c) => {
              const idx = toFlat(bootstrapIndices(foldIn(ctx.key, 10 + c), s.x.n))
              return estimator.fit(
                { x: fromView(takeRows(s.x, idx)), y: idx.map((i) => s.y[i] ?? 0) },
                innerFitOptions(options, foldIn(ctx.key, 100 + c), ctx.run),
              )
            })
          : []
      const n = Math.min(batch, budget - s.queried.length, s.remaining.length)
      const picks = options.strategy.select(
        {
          fitted,
          committee,
          labeled: { x: fromView(s.x), y: [...s.y] },
          pool: sub,
          estimator,
          key: foldIn(ctx.key, 1),
          fitOptions: options,
        },
        n,
      )
      const indices = picks.map((p) => s.remaining[p] ?? -1)
      if (indices.some((i) => i < 0) || new Set(indices).size !== indices.length) {
        throw new WhiteboxError('value', `activeLearning: ${options.strategy.name} returned invalid pool indices`)
      }
      const labels = options.oracle(indices)
      if (labels.length !== indices.length)
        throw new WhiteboxError('shape', 'activeLearning: the oracle must label every queried row')
      const x = stackRows(s.x, takeRows(pool, indices))
      const chosen = new Set(indices)
      const next: State = {
        x,
        y: [...s.y, ...labels],
        remaining: s.remaining.filter((i) => !chosen.has(i)),
        queried: [...s.queried, ...indices],
        scores: s.scores,
      }
      if (scorer !== undefined && options.evaluation !== undefined) {
        return {
          ...next,
          scores: [...s.scores, scorer.score(fitAll(next, foldIn(ctx.key, 2), ctx.run), options.evaluation)],
        }
      }
      return next
    },
    done: (s) => (s.queried.length >= budget || s.remaining.length === 0 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { labeled: s.x.n, ...(s.scores.length === 0 ? {} : { score: s.scores[s.scores.length - 1] ?? 0 }) },
      messages: s.queried.length === 0 ? [] : [{ kind: 'query', payload: { indices: [...s.queried] } }],
    }),
  }
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'activeLearning'),
    maxSteps: Math.ceil(budget / batch) + 1,
  })
  const labeled = { x: fromView(state.x), y: [...state.y] }
  return {
    fitted: estimator.fit(labeled, innerFitOptions(options, foldIn(key, 9), options.parent)),
    queried: state.queried,
    labeled,
    scores: state.scores,
  }
}

/** Options of {@link streamActiveLearning}. */
export interface StreamOptions extends FitOptions {
  readonly labeled: LearnData
  /** Rows arriving in order `[T, d]`. */
  readonly stream: TensorLike
  readonly oracle: (index: number) => number
  /** Query a row when its least-confidence uncertainty exceeds this (default 0.3). */
  readonly threshold?: number
  /** The most labels bought (default unlimited). */
  readonly budget?: number
}

/**
 * Stream-based selective sampling (Cohn, Atlas and Ladner 1994): rows arrive one at a time and each is labelled (and
 * the model refitted) only when the current model's uncertainty on it exceeds a threshold, until the budget is spent.
 * A traced algorithm with one step per arriving row.
 *
 * @param estimator - The classifier.
 * @param options - labeled, stream, oracle, threshold, budget, key and tracing.
 * @returns The final model and the stream indices queried.
 * @example
 * ```ts
 * import { streamActiveLearning, defineEstimator } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
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
  const stream = matrixView(options.stream, 'streamActiveLearning')
  const x0 = matrixView(options.labeled.x, 'streamActiveLearning', { nFeatures: stream.d })
  const y0 = targetVector(options.labeled.y ?? [], 'streamActiveLearning', x0.n)
  const threshold = options.threshold ?? 0.3
  const budget = options.budget ?? Number.POSITIVE_INFINITY
  type State = {
    readonly t: number
    readonly x: MatrixView
    readonly y: readonly number[]
    readonly fitted: Fitted
    readonly queried: readonly number[]
  }
  const key = keyFor(options, 'streamActiveLearning')
  const algorithm: Algorithm<State> = {
    name: 'stream-active-learning',
    config: { estimator: estimator.name, threshold, rows: stream.n },
    schema: channels({
      metrics: {
        uncertainty: { description: 'the uncertainty of the arriving row' },
        labeled: { description: 'labels bought' },
      },
    }),
    init: () => ({
      t: 0,
      x: x0,
      y: Array.from(y0),
      fitted: estimator.fit(
        { x: fromView(x0), y: Array.from(y0) },
        innerFitOptions(options, foldIn(key, 0), options.parent),
      ),
      queried: [],
    }),
    step: (s, ctx) => {
      const row = takeRows(stream, [s.t])
      const u = leastConfidence(requireVerb(s.fitted, 'predictProba', 'streamActiveLearning')(fromView(row)))[0] ?? 0
      if (u <= threshold || s.queried.length >= budget) return { ...s, t: s.t + 1 }
      const x = stackRows(s.x, row)
      const y = [...s.y, options.oracle(s.t)]
      return {
        t: s.t + 1,
        x,
        y,
        fitted: estimator.fit({ x: fromView(x), y }, innerFitOptions(options, foldIn(ctx.key, 0), ctx.run)),
        queried: [...s.queried, s.t],
      }
    },
    done: (s) => (s.t >= stream.n ? 'completed' : undefined),
    observe: (s) => ({ metrics: { labeled: s.queried.length } }),
  }
  const { state } = runAlgorithm(algorithm, { ...runOptionsFor(options, 'streamActiveLearning'), maxSteps: stream.n })
  return { fitted: state.fitted, queried: state.queried }
}

/**
 * Stack two feature matrices by rows (for strategies written outside this module).
 *
 * @param a - Rows `[n, d]`.
 * @param b - More rows `[m, d]`.
 * @returns `[n + m, d]`.
 * @example
 * ```ts
 * import { appendRows } from 'whitebox-engine/learn'
 * appendRows([[1, 2]], [[3, 4]]).shape // => [2, 2]
 * ```
 */
export function appendRows(a: TensorLike, b: TensorLike): Tensor {
  const m = matrixView(a, 'appendRows')
  return fromView(stackRows(m, matrixView(b, 'appendRows', { nFeatures: m.d })))
}

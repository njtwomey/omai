/**
 * Active learning, where the model chooses which unlabelled examples to have labelled next so it learns from fewer
 * labels, modAL style. Uncertainty measures and query strategies (least confidence, margin, entropy, query by
 * committee, expected error reduction, expected model change, BALD and BatchBALD, density-weighted uncertainty, greedy
 * k-center core-sets, BADGE, ranked batches), pool-based `activeLearningAlgorithm` and `activeLearning` as a traced
 * loop of fit, query and label (the stream-based case is in `stream-active-learning.ts`).
 */

import { MdmlError } from '../base/index.ts'
import { ddot, dnrm2 } from '../numerics/linalg/index.ts'
import {
  bootstrapIndices,
  categorical,
  foldIn,
  type Key,
  key as makeKey,
  permutation,
  uniform,
} from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { innerFitOptions, keyFor, rbfGram, runOptionsFor } from './_shared.ts'
import { kCenterGreedy } from './coresets.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireVerb } from './estimator.ts'
import { type Scorer, type Scoring, scorersOf } from './scoring.ts'
import {
  encodeLabels,
  fromView,
  integerOption,
  type MatrixView,
  matrixView,
  takeRows,
  targetVector,
} from './validation.ts'

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
 * import { leastConfidence } from 'mdml-engine/learn'
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
 * import { classMargin } from 'mdml-engine/learn'
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
 * import { predictiveEntropy } from 'mdml-engine/learn'
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
 * import { uncertaintySampling } from 'mdml-engine/learn'
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
 * import { marginSampling } from 'mdml-engine/learn'
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
 * import { entropySampling } from 'mdml-engine/learn'
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
 * import { randomSampling } from 'mdml-engine/learn'
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
 * import { voteEntropySampling } from 'mdml-engine/learn'
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
 * import { consensusEntropySampling } from 'mdml-engine/learn'
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
 * import { maxDisagreementSampling } from 'mdml-engine/learn'
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
 * import { expectedErrorReduction } from 'mdml-engine/learn'
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
 * import { rankedBatchSampling } from 'mdml-engine/learn'
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

/**
 * Expected gradient length (EGL) scores for expected model change (Settles, Craven and Ray 2008): for each pool row i
 * with predicted probabilities p_i and feature vector x_i (augmented with an intercept 1), the expected Euclidean norm
 * sum_c p_{i,c} ||(p_i - e_c) (x) [x_i, 1]||_2 of the cross-entropy gradient induced by labelling x_i.
 *
 * @param proba - Class probabilities `[n, k]`.
 * @param x - Feature matrix `[n, d]`.
 * @returns One expected gradient norm per row.
 * @concept active-learning
 * @example
 * ```ts
 * import { expectedModelChangeScores } from 'mdml-engine/learn'
 * const s = expectedModelChangeScores([[0.9, 0.1], [0.5, 0.5]], [[1], [1]])
 * s[1] > s[0] // => true
 * ```
 */
export function expectedModelChangeScores(proba: TensorLike, x: TensorLike): number[] {
  const p = matrixView(proba, 'expectedModelChangeScores')
  const f = matrixView(x, 'expectedModelChangeScores')
  if (p.n !== f.n) throw new MdmlError('shape', 'expectedModelChangeScores: proba and x must have the same rows')
  return Array.from({ length: p.n }, (_, i) => {
    const rowX = f.data.subarray(i * f.d, (i + 1) * f.d)
    const xNorm = Math.hypot(dnrm2(rowX), 1)
    let pNormSq = 0
    for (let c = 0; c < p.d; c++) pNormSq += (p.data[i * p.d + c] ?? 0) ** 2
    let egl = 0
    for (let c = 0; c < p.d; c++) {
      const pc = p.data[i * p.d + c] ?? 0
      const diffNorm = Math.sqrt(Math.max(0, pNormSq - 2 * pc + 1))
      egl += pc * diffNorm * xNorm
    }
    return egl
  })
}

/** Options of {@link expectedModelChange}. */
export interface ExpectedModelChangeOptions {
  /** Custom per-candidate, per-class gradient norms `[n][k]` (defaults to cross-entropy gradient norms on `[x, 1]`). */
  readonly gradNorms?: (ctx: QueryContext) => readonly (readonly number[])[]
}

/**
 * Expected model change (Settles, Craven and Ray 2008): select the pool rows with the largest expected gradient norm
 * sum_c p(y = c | x) ||grad_theta l(f_theta(x), c)||_2 under the current model's predictive distribution.
 *
 * @param options - Optional custom gradient-norm evaluator.
 * @returns The query strategy.
 * @concept active-learning
 * @example
 * ```ts
 * import { expectedModelChange } from 'mdml-engine/learn'
 * expectedModelChange().name // => 'expected-model-change'
 * ```
 */
export function expectedModelChange(options: ExpectedModelChangeOptions = {}): QueryStrategy {
  return scoreStrategy('expected-model-change', (ctx) => {
    const proba = requireVerb(ctx.fitted, 'predictProba', 'expectedModelChange')(fromView(ctx.pool))
    if (options.gradNorms === undefined) return expectedModelChangeScores(proba, fromView(ctx.pool))
    const p = matrixView(proba, 'expectedModelChange')
    const norms = options.gradNorms(ctx)
    return Array.from({ length: p.n }, (_, i) => {
      let s = 0
      for (let c = 0; c < p.d; c++) s += (p.data[i * p.d + c] ?? 0) * (norms[i]?.[c] ?? 0)
      return s
    })
  })
}

/**
 * Bayesian Active Learning by Disagreement (BALD; Houlsby et al. 2011): the mutual information
 * I(y; theta | x) = H(mean_m p_m(y | x)) - mean_m H(p_m(y | x)) between predictions and posterior/committee members.
 * High when members are individually confident and disagree with each other (epistemic uncertainty), near zero when
 * every member is equally uncertain (aleatoric noise).
 *
 * @param memberProbas - Sequence of member probability matrices, each `[n, k]`.
 * @returns One BALD mutual-information score (in nats) per row.
 * @concept active-learning
 * @example
 * ```ts
 * import { baldAcquisition } from 'mdml-engine/learn'
 * const s = baldAcquisition([[[0.99, 0.01], [0.5, 0.5]], [[0.01, 0.99], [0.5, 0.5]]])
 * s[0] > 0.6 && s[1] === 0 // => true
 * ```
 */
export function baldAcquisition(memberProbas: readonly TensorLike[]): number[] {
  if (memberProbas.length === 0) throw new MdmlError('value', 'baldAcquisition: need at least one member')
  const views = memberProbas.map((m) => matrixView(m, 'baldAcquisition'))
  const n = views[0]?.n ?? 0
  const k = views[0]?.d ?? 0
  const M = views.length
  for (const v of views) {
    if (v.n !== n || v.d !== k) throw new MdmlError('shape', 'baldAcquisition: all member probas must be [n, k]')
  }
  const mean = new Float64Array(n * k)
  const expectedEnt = new Float64Array(n)
  for (const v of views) {
    const ent = predictiveEntropy(fromView(v))
    for (let i = 0; i < n; i++) {
      expectedEnt[i] = (expectedEnt[i] ?? 0) + (ent[i] ?? 0) / M
      for (let c = 0; c < k; c++) mean[i * k + c] = (mean[i * k + c] ?? 0) + (v.data[i * k + c] ?? 0) / M
    }
  }
  const totalEnt = predictiveEntropy(fromStorage(mean, [n, k], 'float64'))
  return Array.from({ length: n }, (_, i) => Math.max(0, (totalEnt[i] ?? 0) - (expectedEnt[i] ?? 0)))
}

/**
 * BALD query strategy over the committee's member probability predictions (Houlsby et al. 2011).
 *
 * @returns The strategy (uses the committee).
 * @concept active-learning
 * @example
 * ```ts
 * import { baldSampling } from 'mdml-engine/learn'
 * baldSampling().committee // => true
 * ```
 */
export function baldSampling(): QueryStrategy {
  return scoreStrategy(
    'bald',
    (ctx) => {
      const { members, k } = committeeProba(ctx)
      return baldAcquisition(members.map((m) => fromStorage(m, [ctx.pool.n, k], 'float64')))
    },
    true,
  )
}

/** Options of {@link batchBald}. */
export interface BatchBaldOptions {
  /** Monte Carlo joint label configurations when k^b exceeds this cap (default 512). */
  readonly numSamples?: number
  readonly key?: Key
}

/**
 * BatchBALD (Kirsch, van Amersfoort and Gal 2019): greedily build a batch of size `n` maximising the joint mutual
 * information I(y_{1..b}; theta) = H(y_{1..b}) - E_theta[H(y_{1..b} | theta)] across committee/ensemble members,
 * avoiding redundant near-duplicate queries that single-point BALD picks together.
 *
 * @param memberProbas - Member probability matrices, each `[poolSize, k]`.
 * @param n - Batch size to select.
 * @param options - Monte Carlo configuration cap and PRNG key.
 * @returns Chosen pool `indices` and `scores`, the joint mutual information of the batch so far after each pick.
 * @concept active-learning
 * @example
 * ```ts
 * import { batchBald } from 'mdml-engine/learn'
 * const m1 = [[0.95, 0.05], [0.95, 0.05], [0.5, 0.5]]
 * const m2 = [[0.05, 0.95], [0.05, 0.95], [0.5, 0.5]]
 * batchBald([m1, m2], 1).indices // => [0]
 * ```
 */
export function batchBald(
  memberProbas: readonly TensorLike[],
  n: number,
  options: BatchBaldOptions = {},
): { readonly indices: number[]; readonly scores: number[] } {
  if (memberProbas.length === 0) throw new MdmlError('value', 'batchBald: need at least one member')
  const views = memberProbas.map((m) => matrixView(m, 'batchBald'))
  const poolSize = views[0]?.n ?? 0
  const k = views[0]?.d ?? 0
  const M = views.length
  for (const v of views) {
    if (v.n !== poolSize || v.d !== k) throw new MdmlError('shape', 'batchBald: all member probas must be [n, k]')
  }
  const batchSize = Math.min(integerOption(n, 'n', 'batchBald'), poolSize)
  const individualEnt = new Float64Array(poolSize)
  for (const v of views) {
    const ent = predictiveEntropy(fromView(v))
    for (let i = 0; i < poolSize; i++) individualEnt[i] = (individualEnt[i] ?? 0) + (ent[i] ?? 0) / M
  }
  const maxExact = options.numSamples ?? 512
  const k0 = options.key ?? makeKey(0)
  // Maintain joint probabilities P_m(y_{1..t}) for each member m across configurations.
  let jointPerMember: Float64Array[] = Array.from({ length: M }, () => Float64Array.of(1))
  let configWeights = Float64Array.of(1)
  const chosen: number[] = []
  const scores: number[] = []
  const used = new Set<number>()
  let condEntSum = 0

  for (let step = 0; step < batchSize; step++) {
    let bestIdx = -1
    let bestScore = Number.NEGATIVE_INFINITY
    const numConfigs = configWeights.length
    for (let i = 0; i < poolSize; i++) {
      if (used.has(i)) continue
      let jointEnt = 0
      for (let cfg = 0; cfg < numConfigs; cfg++) {
        const w = configWeights[cfg] ?? 1
        for (let c = 0; c < k; c++) {
          let meanP = 0
          for (let m = 0; m < M; m++) {
            meanP += ((jointPerMember[m]?.[cfg] ?? 0) * (views[m]?.data[i * k + c] ?? 0)) / M
          }
          if (meanP > 0) jointEnt -= (meanP / w) * Math.log(meanP)
        }
      }
      const mi = jointEnt - (condEntSum + (individualEnt[i] ?? 0))
      if (mi > bestScore) {
        bestScore = mi
        bestIdx = i
      }
    }
    if (bestIdx < 0) break
    chosen.push(bestIdx)
    scores.push(Math.max(0, bestScore))
    used.add(bestIdx)
    condEntSum += individualEnt[bestIdx] ?? 0

    const nextConfigs = numConfigs * k
    const expanded: Float64Array[] = Array.from({ length: M }, () => new Float64Array(nextConfigs))
    const expandedW = new Float64Array(nextConfigs)
    for (let cfg = 0; cfg < numConfigs; cfg++) {
      for (let c = 0; c < k; c++) {
        const idx = cfg * k + c
        expandedW[idx] = configWeights[cfg] ?? 1
        for (let m = 0; m < M; m++) {
          const arr = expanded[m] as Float64Array
          arr[idx] = (jointPerMember[m]?.[cfg] ?? 0) * (views[m]?.data[bestIdx * k + c] ?? 0)
        }
      }
    }
    if (nextConfigs <= maxExact) {
      jointPerMember = expanded
      configWeights = expandedW
    } else {
      // Too many label configurations to enumerate: draw `maxExact` of them from the joint predictive. A draw s stands for
      // probability 1 / S, so the entropy sums weight it by 1 / (S P(s)), which is the divisor `w` below.
      const joint = Float64Array.from({ length: nextConfigs }, (_, idx) => {
        let sum = 0
        for (let m = 0; m < M; m++) sum += expanded[m]?.[idx] ?? 0
        return sum / M
      })
      const total = joint.reduce((a, b) => a + b, 0)
      const draws = toFlat(
        categorical(
          foldIn(k0, step),
          Array.from(joint, (p) => p / total),
          [maxExact],
        ),
      )
      jointPerMember = Array.from({ length: M }, (_, m) => Float64Array.from(draws, (d) => expanded[m]?.[d] ?? 0))
      configWeights = Float64Array.from(draws, (d) => Math.max(maxExact * (joint[d] ?? 0), 1e-300))
    }
  }
  return { indices: chosen, scores }
}

/** Options of {@link densityWeightedUncertainty} and {@link densityWeightedSampling}. */
export interface DensityWeightedOptions {
  /** Density exponent beta >= 0 (default 1; 0 recovers pure uncertainty). */
  readonly beta?: number
  /** RBF similarity bandwidth gamma > 0 (default 1 / d). */
  readonly gamma?: number
  /** Base uncertainty measure for {@link densityWeightedSampling} (default `'uncertainty'`). */
  readonly base?: 'uncertainty' | 'margin' | 'entropy'
}

/**
 * Information-density weighted uncertainty (Settles and Craven 2008): weight each candidate's uncertainty u(x_i) by its
 * average RBF similarity s(x_i)^beta to the pool so isolated outliers are down-weighted in favour of dense regions.
 *
 * @param uncertainty - Base uncertainty scores `[n]`.
 * @param x - Pool feature matrix `[n, d]`.
 * @param options - Exponent `beta` (default 1) and RBF bandwidth `gamma` (default 1 / d).
 * @returns Density-weighted scores `[n]`.
 * @concept active-learning
 * @example
 * ```ts
 * import { densityWeightedUncertainty } from 'mdml-engine/learn'
 * // Outlier at x = 100 has slightly higher raw uncertainty but much lower pool density than the cluster near 0.
 * const scores = densityWeightedUncertainty([0.45, 0.45, 0.45, 0.5], [[0], [0.1], [-0.1], [100]], { beta: 1, gamma: 1 })
 * scores[0] > scores[3] // => true
 * ```
 */
export function densityWeightedUncertainty(
  uncertainty: ArrayLike<number>,
  x: TensorLike,
  options: DensityWeightedOptions = {},
): number[] {
  const m = matrixView(x, 'densityWeightedUncertainty')
  const u = targetVector(Array.from(uncertainty), 'densityWeightedUncertainty', m.n)
  const beta = options.beta ?? 1
  if (!(beta >= 0)) throw new MdmlError('value', 'densityWeightedUncertainty: beta must be >= 0')
  const gamma = options.gamma ?? 1 / Math.max(1, m.d)
  if (!(gamma > 0)) throw new MdmlError('value', 'densityWeightedUncertainty: gamma must be > 0')
  const gram = rbfGram(m, m, gamma)
  return Array.from({ length: m.n }, (_, i) => {
    let sim = 0
    for (let j = 0; j < m.n; j++) sim += gram[i * m.n + j] ?? 0
    return (u[i] ?? 0) * (sim / m.n) ** beta
  })
}

/**
 * Density-weighted uncertainty sampling strategy (Settles and Craven 2008).
 *
 * @param options - `beta`, `gamma` and `base` uncertainty measure (`'uncertainty'`, `'margin'` or `'entropy'`).
 * @returns The query strategy.
 * @concept active-learning
 * @example
 * ```ts
 * import { densityWeightedSampling } from 'mdml-engine/learn'
 * densityWeightedSampling({ beta: 1.5 }).name // => 'density-weighted'
 * ```
 */
export function densityWeightedSampling(options: DensityWeightedOptions = {}): QueryStrategy {
  return scoreStrategy('density-weighted', (ctx) => {
    const proba = requireVerb(ctx.fitted, 'predictProba', 'densityWeightedSampling')(fromView(ctx.pool))
    const base =
      options.base === 'entropy'
        ? predictiveEntropy(proba)
        : options.base === 'margin'
          ? classMargin(proba).map((v) => 1 - v)
          : leastConfidence(proba)
    return densityWeightedUncertainty(base, fromView(ctx.pool), options)
  })
}

/** Options of {@link coreSetGreedy}. */
export interface CoreSetOptions {
  /** Already labelled feature rows `[m, d]` that act as initial centres. */
  readonly labeled?: TensorLike
  /** Initial pool index when `labeled` is omitted (default 0). */
  readonly seedIndex?: number
}

/**
 * Greedy k-center core-set selection (Sener and Savarese 2018; Gonzalez 1985): iteratively pick the pool point
 * farthest from the set of already labelled and selected centres, minimising the covering radius of the pool.
 *
 * @param pool - Unlabelled pool features `[n, d]`.
 * @param n - Number of points to select.
 * @param options - Optional already labelled features `labeled` or initial `seedIndex`.
 * @returns Selected pool `indices`, final minimum `distances` from each pool point, and the covering `radius`.
 * @concept active-learning
 * @example
 * ```ts
 * import { coreSetGreedy } from 'mdml-engine/learn'
 * coreSetGreedy([[0], [2], [10]], 2, { labeled: [[0]] }).indices // => [2, 1]
 * ```
 */
export function coreSetGreedy(
  pool: TensorLike,
  n: number,
  options: CoreSetOptions = {},
): { readonly indices: number[]; readonly distances: number[]; readonly radius: number } {
  const r = kCenterGreedy(pool, n, {
    seedIndex: options.seedIndex ?? 0,
    ...(options.labeled === undefined ? {} : { centres: options.labeled }),
  })
  return { indices: [...r.indices], distances: [...r.distances], radius: r.radius }
}

/**
 * Greedy k-center core-set query strategy (Sener and Savarese 2018).
 *
 * @returns The query strategy.
 * @concept active-learning
 * @example
 * ```ts
 * import { coreSetSampling } from 'mdml-engine/learn'
 * coreSetSampling().name // => 'core-set'
 * ```
 */
export function coreSetSampling(): QueryStrategy {
  return {
    name: 'core-set',
    select: (ctx, n) => coreSetGreedy(fromView(ctx.pool), n, { labeled: ctx.labeled.x }).indices,
  }
}

/** Options of {@link badgeSelection}. */
export interface BadgeOptions {
  /** The key of the k-means++ draws (default `key(0)`). */
  readonly key?: Key
}

/**
 * BADGE selection (Ash et al. 2020, Batch Active learning by Diverse Gradient Embeddings): form hallucinated
 * last-layer loss gradient embeddings g_i = (p_i - e_{yhat_i}) (x) [x_i, 1] using each candidate's most likely predicted
 * label yhat_i = argmax_c p_{i,c}, whose norm grows with the uncertainty of the prediction, then run k-means++ seeding on
 * {g_i}: the first centre is the largest embedding, each next one is drawn with probability proportional to its squared
 * distance to the nearest centre already chosen (a duplicate of a chosen row has distance zero and is never drawn).
 *
 * @param proba - Predicted class probabilities `[n, k]`.
 * @param x - Feature or penultimate-layer representations `[n, d]`.
 * @param n - Batch size to select.
 * @param options - The PRNG `key`.
 * @returns Chosen pool indices `[n]`.
 * @concept active-learning
 * @example
 * ```ts
 * import { badgeSelection } from 'mdml-engine/learn'
 * // Rows 0 and 1 are the same uncertain point: a batch of three never takes both before the other two rows.
 * const p = [[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.55, 0.45]]
 * badgeSelection(p, [[-4], [-4], [4], [0]], 3).includes(1) // => false
 * ```
 */
export function badgeSelection(proba: TensorLike, x: TensorLike, n: number, options: BadgeOptions = {}): number[] {
  const p = matrixView(proba, 'badgeSelection')
  const f = matrixView(x, 'badgeSelection')
  if (p.n !== f.n) throw new MdmlError('shape', 'badgeSelection: proba and x must have the same number of rows')
  const count = Math.min(integerOption(n, 'n', 'badgeSelection'), p.n)
  const dAug = f.d + 1
  const dim = p.d * dAug
  const grads = new Float64Array(p.n * dim)
  const normsSq = new Float64Array(p.n)

  for (let i = 0; i < p.n; i++) {
    let yhat = 0
    for (let c = 1; c < p.d; c++) {
      if ((p.data[i * p.d + c] ?? 0) > (p.data[i * p.d + yhat] ?? 0)) yhat = c
    }
    for (let c = 0; c < p.d; c++) {
      const delta = (p.data[i * p.d + c] ?? 0) - (c === yhat ? 1 : 0)
      const base = i * dim + c * dAug
      for (let j = 0; j < f.d; j++) grads[base + j] = delta * (f.data[i * f.d + j] ?? 0)
      grads[base + f.d] = delta
    }
    const gi = grads.subarray(i * dim, (i + 1) * dim)
    normsSq[i] = ddot(gi, gi)
  }

  const draws = toFlat(uniform(options.key ?? makeKey(0), [count]))
  let first = 0
  for (let i = 1; i < p.n; i++) if ((normsSq[i] ?? 0) > (normsSq[first] ?? 0)) first = i
  const chosen = [first]
  const used = new Set<number>([first])
  const distSq = new Float64Array(p.n).fill(Number.POSITIVE_INFINITY)
  const updateDistances = (centre: number): void => {
    const gc = grads.subarray(centre * dim, (centre + 1) * dim)
    for (let i = 0; i < p.n; i++) {
      let sq = 0
      const gi = grads.subarray(i * dim, (i + 1) * dim)
      for (let j = 0; j < dim; j++) sq += ((gi[j] ?? 0) - (gc[j] ?? 0)) ** 2
      if (sq < (distSq[i] ?? Number.POSITIVE_INFINITY)) distSq[i] = sq
    }
  }
  updateDistances(first)
  for (let t = 1; t < count; t++) {
    let total = 0
    for (let i = 0; i < p.n; i++) if (!used.has(i)) total += distSq[i] ?? 0
    let pick = -1
    if (total > 0) {
      let target = (draws[t] ?? 0) * total
      for (let i = 0; i < p.n; i++) {
        if (used.has(i)) continue
        target -= distSq[i] ?? 0
        if (target <= 0) {
          pick = i
          break
        }
      }
    }
    // Every remaining embedding coincides with a centre: take the first row left.
    if (pick < 0) pick = Array.from({ length: p.n }, (_, i) => i).filter((i) => !used.has(i))[0] ?? -1
    if (pick < 0) break
    chosen.push(pick)
    used.add(pick)
    updateDistances(pick)
  }
  return chosen
}

/**
 * BADGE query strategy (Ash et al. 2020): diverse uncertain batches via k-means++ on hallucinated gradient embeddings.
 *
 * @returns The query strategy.
 * @concept active-learning
 * @example
 * ```ts
 * import { badgeSampling } from 'mdml-engine/learn'
 * badgeSampling().name // => 'badge'
 * ```
 */
export function badgeSampling(): QueryStrategy {
  return {
    name: 'badge',
    select: (ctx, n) => {
      const proba = requireVerb(ctx.fitted, 'predictProba', 'badgeSampling')(fromView(ctx.pool))
      return badgeSelection(proba, fromView(ctx.pool), n, { key: ctx.key })
    },
  }
}

/** Options of {@link activeLearning} and {@link activeLearningAlgorithm}. */
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

/** Serialisable state of {@link activeLearningAlgorithm}. */
export interface ActiveLearningState {
  readonly x: MatrixView
  readonly y: readonly number[]
  readonly remaining: readonly number[]
  readonly queried: readonly number[]
  readonly scores: readonly number[]
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
 * Stepped pool-based active learning {@link Algorithm}: each step fits the estimator (and a bootstrap committee when
 * the strategy asks for one), selects a batch of unlabelled pool rows, queries the oracle, appends them to the
 * labelled set, and optionally evaluates on held-out data.
 *
 * @param estimator - The classifier.
 * @param options - labeled, pool, oracle, strategy, batchSize, budget, committeeSize, evaluation and scoring.
 * @returns The stepped {@link Algorithm} over {@link ActiveLearningState}.
 * @concept active-learning
 * @example
 * ```ts
 * import { activeLearningAlgorithm, defineEstimator, uncertaintySampling } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const est = defineEstimator({ name: 't', kind: 'classifier', fit: () => ({}), methods: () => ({ classes: [0, 1], predictProba: (x) => tensor(Array.from({ length: x.shape[0] }, () => [0.5, 0.5])) }) }, {})
 * const alg = activeLearningAlgorithm(est, { labeled: { x: [[0], [1]], y: [0, 1] }, pool: [[0.2], [0.8]], oracle: () => [0], strategy: uncertaintySampling(), budget: 1 })
 * runAlgorithm(alg, { maxSteps: 2 }).state.queried.length // => 1
 * ```
 */
export function activeLearningAlgorithm(
  estimator: Estimator,
  options: ActiveLearningOptions,
): Algorithm<ActiveLearningState> {
  const pool = matrixView(options.pool, 'activeLearning')
  const labeled0 = matrixView(options.labeled.x, 'activeLearning', { nFeatures: pool.d })
  const y0 = targetVector(options.labeled.y ?? [], 'activeLearning', labeled0.n)
  const batch = options.batchSize ?? 1
  if (!(Number.isInteger(batch) && batch >= 1)) throw new MdmlError('value', 'activeLearning: batchSize must be >= 1')
  const budget = Math.min(options.budget ?? pool.n, pool.n)
  const scorer: Scorer | undefined =
    options.evaluation === undefined ? undefined : Object.values(scorersOf(options.scoring))[0]
  const fitAll = (s: ActiveLearningState, k: Key, parent: FitOptions['parent']): Fitted =>
    estimator.fit({ x: fromView(s.x), y: [...s.y] }, innerFitOptions(options, k, parent))
  return {
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
        throw new MdmlError('value', `activeLearning: ${options.strategy.name} returned invalid pool indices`)
      }
      const labels = options.oracle(indices)
      if (labels.length !== indices.length)
        throw new MdmlError('shape', 'activeLearning: the oracle must label every queried row')
      const x = stackRows(s.x, takeRows(pool, indices))
      const chosen = new Set(indices)
      const next: ActiveLearningState = {
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
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ActiveLearningState,
  }
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
 * @index function Pool-based active learning with query strategies.
 * @example
 * ```ts
 * import { activeLearning, uncertaintySampling } from 'mdml-engine/learn'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { tensor } from 'mdml-engine/tensor'
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
  const algorithm = activeLearningAlgorithm(estimator, options)
  const pool = matrixView(options.pool, 'activeLearning')
  const batch = options.batchSize ?? 1
  const budget = Math.min(options.budget ?? pool.n, pool.n)
  const key = keyFor(options, 'activeLearning')
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

/**
 * Stack two feature matrices by rows (for strategies written outside this module).
 *
 * @param a - Rows `[n, d]`.
 * @param b - More rows `[m, d]`.
 * @returns `[n + m, d]`.
 * @example
 * ```ts
 * import { appendRows } from 'mdml-engine/learn'
 * appendRows([[1, 2]], [[3, 4]]).shape // => [2, 2]
 * ```
 */
export function appendRows(a: TensorLike, b: TensorLike): Tensor {
  const m = matrixView(a, 'appendRows')
  return fromView(stackRows(m, matrixView(b, 'appendRows', { nFeatures: m.d })))
}

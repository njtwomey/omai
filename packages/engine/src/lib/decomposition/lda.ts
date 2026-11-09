/**
 * Latent Dirichlet allocation by variational Bayes (scikit-learn's `LatentDirichletAllocation`): a topic model of a
 * document-word count matrix in which every document mixes topics and every topic is a distribution over words. The
 * fit is Hoffman, Blei and Bach's (2010) variational inference, in batch (every document each step) or online
 * (one mini-batch each step, the topic-word parameters moving by a decaying weight), with the per-document E-step of
 * scikit-learn. One step is one EM update, and the tape records the variational bound and the perplexity with the
 * topic-word matrix, so the topics can be watched forming.
 */

import { type Json, OmaiError, toRows } from '../base/index.ts'
import { componentsRecord } from '../exposition/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { digamma, lgamma } from '../numerics/special/index.ts'
import { foldIn, gammaVariates, type Key } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** How the topic-word parameters are updated: every document each step (batch) or one mini-batch each step (online). */
export type LdaLearningMethod = 'batch' | 'online'

type Rows = readonly (readonly number[])[]

/** A document as the words it uses (their indices) and how many times each occurs. */
interface Doc {
  readonly ids: readonly number[]
  readonly counts: readonly number[]
}

const docsOf = (x: Rows): Doc[] =>
  x.map((row) => {
    const ids: number[] = []
    const counts: number[] = []
    for (const [j, v] of row.entries())
      if (v !== 0) {
        ids.push(j)
        counts.push(v)
      }
    return { ids, counts }
  })

// The digamma and log-gamma of a list of numbers, by the special functions of numerics.
const psi = (v: ArrayLike<number>): number[] => toFlat(digamma(Float64Array.from(v)))
const logGamma = (v: ArrayLike<number>): number[] => toFlat(lgamma(Float64Array.from(v)))

const sum = (v: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] ?? 0
  return s
}

/** E[log p] of a Dirichlet row: psi(row) - psi(sum of the row) (scikit-learn's `_dirichlet_expectation_2d`). */
function dirichletExpectation(rows: Rows): number[][] {
  const flat = psi(rows.flat())
  const totals = psi(rows.map(sum))
  const d = rows[0]?.length ?? 0
  return rows.map((_, i) => Array.from({ length: d }, (_, j) => (flat[i * d + j] ?? 0) - (totals[i] ?? 0)))
}

/** exp(E[log theta]) of each row of Dirichlet parameters, with one digamma call for all of them. */
function expElogRows(rows: Rows): number[][] {
  const k = rows[0]?.length ?? 0
  const p = psi(rows.flatMap((g) => [...g, sum(g)]))
  return rows.map((_, i) =>
    Array.from({ length: k }, (_, q) => Math.exp((p[i * (k + 1) + q] ?? 0) - (p[i * (k + 1) + k] ?? 0))),
  )
}

const expOf = (rows: Rows): number[][] => rows.map((r) => r.map(Math.exp))

/** What the E-step needs: the priors and the stopping rule of the per-document iteration. */
interface EStepOptions {
  readonly alpha: number
  readonly maxDocUpdateIter: number
  readonly meanChangeTol: number
}

/**
 * The E-step (scikit-learn's `_update_doc_distribution`): for each document, alternate the topic proportions gamma
 * and the word-topic responsibilities until gamma stops moving; and, for a training step, the expected sufficient
 * statistics of the topic-word update. The documents iterate side by side (each stops at its own convergence), so the
 * digamma of one round is one call over every document still moving.
 */
function eStep(
  docs: readonly Doc[],
  expBeta: Rows,
  options: EStepOptions,
  start: Rows | undefined,
  withStats: boolean,
): { gamma: number[][]; stats: number[][] | undefined } {
  const k = expBeta.length
  const v = expBeta[0]?.length ?? 0
  const eps = Number.EPSILON
  const gamma: number[][] = docs.map((_, d) =>
    start === undefined ? new Array<number>(k).fill(1) : [...(start[d] ?? [])],
  )
  const expTheta = expElogRows(gamma)
  // exp(E[log beta]) of the words of each document, [k][m].
  const etw = docs.map((doc) => expBeta.map((row) => doc.ids.map((id) => row[id] ?? 0)))
  const normPhi = (d: number): number[] => {
    const m = docs[d]?.ids.length ?? 0
    const theta = expTheta[d] as number[]
    return Array.from({ length: m }, (_, j) => {
      let s = 0
      for (let q = 0; q < k; q++) s += (theta[q] ?? 0) * (etw[d]?.[q]?.[j] ?? 0)
      return s + eps
    })
  }
  let active = docs.map((_, d) => d)
  for (let it = 0; it < options.maxDocUpdateIter && active.length > 0; it++) {
    const updated = active.map((d) => {
      const doc = docs[d] as Doc
      const norm = normPhi(d)
      const theta = expTheta[d] as number[]
      return Array.from({ length: k }, (_, q) => {
        let s = 0
        for (let j = 0; j < doc.ids.length; j++) s += ((doc.counts[j] ?? 0) / (norm[j] ?? 1)) * (etw[d]?.[q]?.[j] ?? 0)
        return (theta[q] ?? 0) * s + options.alpha
      })
    })
    const fresh = expElogRows(updated)
    const moving: number[] = []
    for (const [i, d] of active.entries()) {
      const last = gamma[d] as number[]
      const g = updated[i] as number[]
      let change = 0
      for (let q = 0; q < k; q++) change += Math.abs((last[q] ?? 0) - (g[q] ?? 0))
      gamma[d] = g
      expTheta[d] = fresh[i] as number[]
      if (!(change / k < options.meanChangeTol)) moving.push(d)
    }
    active = moving
  }
  let stats: number[][] | undefined
  if (withStats) {
    const out = Array.from({ length: k }, () => new Array<number>(v).fill(0))
    for (const [d, doc] of docs.entries()) {
      const norm = normPhi(d)
      const theta = expTheta[d] as number[]
      for (let q = 0; q < k; q++) {
        const row = out[q] as number[]
        for (let j = 0; j < doc.ids.length; j++) {
          const id = doc.ids[j] as number
          row[id] = (row[id] ?? 0) + (theta[q] ?? 0) * ((doc.counts[j] ?? 0) / (norm[j] ?? 1))
        }
      }
    }
    for (const [q, row] of out.entries()) for (let w = 0; w < v; w++) row[w] = (row[w] ?? 0) * (expBeta[q]?.[w] ?? 0)
    stats = out
  }
  return { gamma, stats }
}

// E[log p(x | prior) - log q(x | distr)] of a set of Dirichlet rows (scikit-learn's `_loglikelihood`).
function dirichletBound(prior: number, distr: Rows, expected: Rows, size: number): number {
  let score = 0
  const flat = distr.flat()
  const lg = logGamma(flat)
  const lgPrior = logGamma([prior, prior * size])
  const flatExpected = expected.flat()
  for (let i = 0; i < flat.length; i++)
    score += (prior - (flat[i] ?? 0)) * (flatExpected[i] ?? 0) + (lg[i] ?? 0) - (lgPrior[0] ?? 0)
  const totals = logGamma(distr.map(sum))
  for (const t of totals) score += (lgPrior[1] ?? 0) - t
  return score
}

/**
 * The variational lower bound on the log-likelihood of the documents given their topic proportions `gamma` and the
 * topic-word parameters `components` (scikit-learn's `_approx_bound`).
 *
 * @param x - Word counts `[n, V]`.
 * @param gamma - Unnormalised document-topic parameters `[n, K]`.
 * @param components - Topic-word parameters `[K, V]` (the variational lambda).
 * @param priors - The document-topic prior alpha and the topic-word prior eta.
 * @param subSampling - Scale the document terms up from `totalSamples` documents (for a sample of a larger corpus).
 * @param totalSamples - The corpus size assumed by sub-sampling.
 * @returns The bound.
 * @example
 * ```ts
 * import { ldaBound } from 'omai-engine/decomposition'
 * const b = ldaBound([[2, 0], [0, 2]], [[3, 1], [1, 3]], [[3, 1], [1, 3]], { alpha: 0.5, eta: 0.5 })
 * Number.isFinite(b) // => true
 * ```
 */
export function ldaBound(
  x: Rows,
  gamma: Rows,
  components: Rows,
  priors: { readonly alpha: number; readonly eta: number },
  subSampling = false,
  totalSamples = 1e6,
): number {
  const k = components.length
  const v = components[0]?.length ?? 0
  const elogTheta = dirichletExpectation(gamma)
  const elogBeta = dirichletExpectation(components)
  let score = 0
  for (const [d, doc] of docsOf(x).entries()) {
    const theta = elogTheta[d] as number[]
    for (const [j, id] of doc.ids.entries()) {
      let top = Number.NEGATIVE_INFINITY
      const terms = theta.map((t, q) => {
        const term = t + (elogBeta[q]?.[id] ?? 0)
        top = Math.max(top, term)
        return term
      })
      let s = 0
      for (const term of terms) s += Math.exp(term - top)
      score += (doc.counts[j] ?? 0) * (top + Math.log(s))
    }
  }
  score += dirichletBound(priors.alpha, gamma, elogTheta, k)
  if (subSampling) score *= totalSamples / x.length
  score += dirichletBound(priors.eta, components, elogBeta, v)
  return score
}

/** The state of LDA: the variational parameters, the bound, and what the update schedule and stopping rule need. */
export interface LdaState {
  /** The topic-word parameters (the variational lambda) `[K, V]`. */
  readonly components: number[][]
  /** The unnormalised document-topic parameters (gamma) of the corpus at the last evaluation `[n, K]`. */
  readonly docTopic: number[][]
  /** The variational bound on the log-likelihood of the corpus at the last evaluation. */
  readonly bound: number
  /** The perplexity of the corpus at the last evaluation, `exp(-bound / words)`. */
  readonly perplexity: number
  /** The mini-batch updates done, plus one (scikit-learn's `n_batch_iter_`): the clock of the online weight. */
  readonly nBatchIter: number
  /** The passes over the corpus completed. */
  readonly epoch: number
  /** The perplexity at the last evaluation of the stopping rule, `null` before the first. */
  readonly lastPerplexity: number | null
  /** Whether the perplexity moved less than the tolerance between two evaluations. */
  readonly stalled: boolean
}

/** Options of {@link ldaAlgorithm}. */
export interface LdaOptions {
  /** The document-topic Dirichlet prior (default `1 / K`). */
  readonly docTopicPrior?: number
  /** The topic-word Dirichlet prior (default `1 / K`). */
  readonly topicWordPrior?: number
  readonly learningMethod?: LdaLearningMethod
  /** The exponent of the online weight `(offset + t) ^ -decay`, in (0.5, 1] for convergence (default 0.7). */
  readonly learningDecay?: number
  /** The online weight's offset, down-weighting the first mini-batches (default 10). */
  readonly learningOffset?: number
  /** The documents per mini-batch of the online method (default 128). */
  readonly batchSize?: number
  /** The most iterations of one document's E-step (default 100). */
  readonly maxDocUpdateIter?: number
  /** The mean absolute change of a document's topic parameters that stops its E-step (default 1e-3). */
  readonly meanChangeTol?: number
  /** Evaluate the perplexity every this many passes and stop when it moves less than `perpTol` (default -1: never). */
  readonly evaluateEvery?: number
  readonly perpTol?: number
  /**
   * Where each document's E-step starts in training: `random` draws gamma(100, 0.01) from the step's key as
   * scikit-learn does, `constant` starts every document at ones (deterministic; evaluation always does).
   */
  readonly docInit?: 'random' | 'constant'
  /** Evaluate the bound after every step (default) or only after each pass over the corpus (cheaper online). */
  readonly evaluate?: 'step' | 'epoch'
}

/**
 * Latent Dirichlet allocation as a traced {@link Algorithm}, one variational EM update per step: the E-step fits every
 * document's topic proportions (all documents in batch mode, the next mini-batch in online mode), the M-step moves the
 * topic-word parameters (to `eta + statistics` in batch mode, by the weight `(offset + t) ^ -decay` towards them in
 * online mode, scaled up to the corpus). It observes the bound, the perplexity, the topic-word parameters and the
 * document-topic proportions.
 *
 * @param x - Word counts `[n, V]`.
 * @param components0 - Initial topic-word parameters `[K, V]` (scikit-learn draws them from gamma(100, 0.01)).
 * @param options - Priors, update method, schedule and stopping rule.
 * @returns The algorithm.
 * @throws OmaiError `value` for negative counts, `shape` for an empty or ragged corpus.
 * @example
 * ```ts
 * import { ldaAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[4, 3, 0, 0], [3, 4, 0, 1], [0, 0, 4, 3], [0, 1, 3, 4]]
 * const start = [[1.1, 0.9, 1, 1], [1, 1, 0.9, 1.1]]
 * const out = runAlgorithm(ldaAlgorithm(x, start, { docInit: 'constant' }), { maxSteps: 20 })
 * out.state.perplexity < 4 // => true
 * ```
 */
export function ldaAlgorithm(x: Rows, components0: Rows, options: LdaOptions = {}): Algorithm<LdaState> {
  const n = x.length
  const v = x[0]?.length ?? 0
  const k = components0.length
  if (n === 0 || v === 0) throw new OmaiError('shape', 'lda: the corpus needs at least one document and one word')
  if (x.some((r) => r.length !== v || r.some((c) => !Number.isFinite(c))))
    throw new OmaiError('shape', 'lda: every document needs the same number of finite counts')
  if (x.some((r) => r.some((c) => c < 0))) throw new OmaiError('value', 'lda: the counts must be non-negative')
  if (components0.length === 0 || components0.some((r) => r.length !== v))
    throw new OmaiError('shape', `lda: the initial components must be [K, ${v}]`)
  const alpha = options.docTopicPrior ?? 1 / k
  const eta = options.topicWordPrior ?? 1 / k
  const method = options.learningMethod ?? 'batch'
  const decay = options.learningDecay ?? 0.7
  const offset = options.learningOffset ?? 10
  const batchSize = options.batchSize ?? 128
  const evaluateEvery = options.evaluateEvery ?? -1
  const perpTol = options.perpTol ?? 0.1
  const docInit = options.docInit ?? 'random'
  const evaluate = options.evaluate ?? 'step'
  const e: EStepOptions = {
    alpha,
    maxDocUpdateIter: options.maxDocUpdateIter ?? 100,
    meanChangeTol: options.meanChangeTol ?? 1e-3,
  }
  if (!(alpha > 0) || !(eta > 0)) throw new OmaiError('value', 'lda: the priors must be positive')
  if (method === 'online' && !(batchSize >= 1)) throw new OmaiError('value', 'lda: batchSize must be at least 1')
  const docs = docsOf(x)
  const words = sum(x.map(sum))
  const batches = method === 'online' ? Math.ceil(n / batchSize) : 1
  // The corpus evaluated at a given lambda: constant start, no statistics.
  const evaluated = (components: number[][]): { docTopic: number[][]; bound: number; perplexity: number } => {
    const { gamma } = eStep(docs, expOf(dirichletExpectation(components)), e, undefined, false)
    const bound = ldaBound(x, gamma, components, { alpha, eta })
    return { docTopic: gamma, bound, perplexity: Math.exp(-bound / words) }
  }
  return {
    name: `lda-${method}`,
    config: {
      alpha,
      eta,
      learningMethod: method,
      learningDecay: decay,
      learningOffset: offset,
      batchSize,
      evaluateEvery,
    },
    schema: channels({
      metrics: {
        bound: { description: 'the variational lower bound on the log-likelihood of the corpus', direction: 'higher' },
        perplexity: { description: 'exp(-bound / words) of the corpus', direction: 'lower' },
      },
      tensors: {
        components: { role: 'state', description: 'the topic-word parameters (lambda)', shape: ['k', 'v'] },
        docTopic: { role: 'state', description: 'the topic proportions of each document', shape: ['n', 'k'] },
      },
    }),
    init: () => {
      const components = components0.map((r) => [...r])
      return { components, ...evaluated(components), nBatchIter: 1, epoch: 0, lastPerplexity: null, stalled: false }
    },
    step: (s, ctx) => {
      const b = (s.nBatchIter - 1) % batches
      const lo = method === 'online' ? b * batchSize : 0
      const hi = method === 'online' ? Math.min(n, lo + batchSize) : n
      const batch = docs.slice(lo, hi)
      let start: number[][] | undefined
      if (docInit === 'random') {
        const g = gammaVariates(ctx.key, new Array<number>(batch.length * k).fill(100))
        start = toRows(g, batch.length, k).map((r) => r.map((c) => c * 0.01))
      }
      const { stats } = eStep(batch, expOf(dirichletExpectation(s.components)), e, start, true)
      let components: number[][]
      if (method === 'batch') components = (stats as number[][]).map((r) => r.map((c) => eta + c))
      else {
        const weight = (offset + s.nBatchIter) ** -decay
        const ratio = n / batch.length
        components = s.components.map((row, q) =>
          row.map((c, w) => c * (1 - weight) + weight * (eta + ratio * ((stats as number[][])[q]?.[w] ?? 0))),
        )
      }
      const endOfPass = b === batches - 1
      const epoch = s.epoch + (endOfPass ? 1 : 0)
      const check = endOfPass && evaluateEvery > 0 && epoch % evaluateEvery === 0
      const now = evaluate === 'step' || endOfPass || check ? evaluated(components) : undefined
      const view = now ?? { docTopic: s.docTopic, bound: s.bound, perplexity: s.perplexity }
      return {
        components,
        ...view,
        nBatchIter: s.nBatchIter + 1,
        epoch,
        lastPerplexity: check ? view.perplexity : s.lastPerplexity,
        stalled: check && s.lastPerplexity !== null && Math.abs(s.lastPerplexity - view.perplexity) < perpTol,
      }
    },
    done: (s) => (s.stalled ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { bound: s.bound, perplexity: s.perplexity },
      tensors: {
        components: { role: 'state', value: s.components.flat(), shape: [k, v] },
        docTopic: { role: 'state', value: normalised(s.docTopic).flat(), shape: [n, k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LdaState,
  }
}

const normalised = (rows: Rows): number[][] =>
  rows.map((r) => {
    const t = sum(r)
    return r.map((c) => c / t)
  })

/** Hyperparameters of {@link latentDirichletAllocation}. */
export interface LdaHyper {
  readonly nComponents: number
  /** The document-topic prior, `null` for `1 / nComponents`. */
  readonly docTopicPrior: number | null
  /** The topic-word prior, `null` for `1 / nComponents`. */
  readonly topicWordPrior: number | null
  readonly learningMethod: LdaLearningMethod
  readonly learningDecay: number
  readonly learningOffset: number
  /** Passes over the corpus. */
  readonly maxIter: number
  readonly batchSize: number
  readonly evaluateEvery: number
  readonly perpTol: number
  readonly meanChangeTol: number
  readonly maxDocUpdateIter: number
  /** The corpus size a sub-sampled bound or perplexity is scaled to. */
  readonly totalSamples: number
  /** Initial topic-word parameters `[K, V]`, `null` to draw them from gamma(100, 0.01) with the key. */
  readonly componentsInit: number[][] | null
  readonly docInit: 'random' | 'constant'
  readonly evaluate: 'step' | 'epoch'
  /** The words of the vocabulary, for the records (default `word0`, `word1`, ...). */
  readonly vocabulary: string[] | null
}

/** The fitted parameters of {@link latentDirichletAllocation}. */
export type LdaParams = {
  /** The topic-word parameters (scikit-learn's `components_`, the variational lambda) `[K, V]`. */
  readonly components: number[][]
  readonly docTopicPrior: number
  readonly topicWordPrior: number
  /** The normalised document-topic proportions of the training corpus `[n, K]`. */
  readonly docTopic: number[][]
  /** The variational bound of the training corpus. */
  readonly bound: number
  /** The perplexity of the training corpus (scikit-learn's `bound_`). */
  readonly perplexity: number
  /** Mini-batch (online) or full (batch) updates done. */
  readonly nBatchIterations: number
  /** Passes over the corpus. */
  readonly nIter: number
  readonly meanChangeTol: number
  readonly maxDocUpdateIter: number
  readonly totalSamples: number
}

/** The verbs of a fitted LDA. */
export interface LdaVerbs {
  /** Topic proportions of documents (the E-step at the fitted topics); `normalize: false` gives the gamma. */
  readonly transform: (x: TensorLike, options?: { readonly normalize?: boolean }) => Tensor
  /** The approximate perplexity `exp(-bound / words)` of documents; `subSampling` scales them up to `totalSamples`. */
  readonly perplexity: (x: TensorLike, subSampling?: boolean) => number
  /** The approximate log-likelihood bound of documents (higher is better). */
  readonly score: (data: { readonly x: TensorLike }) => number
  readonly nFeaturesIn: number
  readonly featureNamesOut: () => string[]
}

const asRows = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return toRows(m.data, m.n, m.d)
}
const toTensor = (rows: number[][]): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')

/** The paths LDA adds to the iterative base's. */
export type LdaPath = 'lda/topics'

const topicRecord = (components: Rows, docTopic: Rows | undefined, vocabulary: readonly string[] | null) =>
  componentsRecord({
    method: 'lda',
    features: components[0]?.map((_, w) => vocabulary?.[w] ?? `word${w}`) ?? [],
    components: normalised(components),
    space: 'vocabulary',
    ...(docTopic === undefined ? {} : { scores: normalised(docTopic) }),
  })

/**
 * Latent Dirichlet allocation (sklearn.decomposition.LatentDirichletAllocation) on the iterative base: variational
 * Bayes, batch or online (Hoffman et al. 2010), one EM update per step, with the same E-step, M-step, online weight,
 * bound and perplexity as scikit-learn. `fit` takes the document-word counts as `x`; `transform` gives topic
 * proportions, `perplexity` and `score` the held-out bound. Scikit-learn starts from random topic-word parameters;
 * pass `componentsInit` (and `docInit: 'constant'`) for a deterministic run. The `lda/topics` path of a session is a
 * components record of the topics at any step.
 *
 * @index estimator Latent Dirichlet allocation topic model by batch or online variational Bayes.
 * @example
 * ```ts
 * import { latentDirichletAllocation } from 'omai-engine/decomposition'
 * const x = [[4, 3, 0, 0], [3, 4, 0, 1], [0, 0, 4, 3], [0, 1, 3, 4]]
 * const f = latentDirichletAllocation.with({ nComponents: 2, maxIter: 30 }).fit({ x })
 * f.params.components.length // => 2
 * ```
 */
export const latentDirichletAllocation: IterativeEstimator<LdaHyper, LdaState, LdaParams, LdaVerbs, LdaPath> =
  iterative<LdaHyper, LdaState, LdaParams, LdaVerbs, LdaPath>({
    name: 'decomposition/lda',
    kind: 'transformer',
    defaults: {
      nComponents: 10,
      docTopicPrior: null,
      topicWordPrior: null,
      learningMethod: 'batch',
      learningDecay: 0.7,
      learningOffset: 10,
      maxIter: 10,
      batchSize: 128,
      evaluateEvery: -1,
      perpTol: 0.1,
      meanChangeTol: 1e-3,
      maxDocUpdateIter: 100,
      totalSamples: 1e6,
      componentsInit: null,
      docInit: 'random',
      evaluate: 'step',
      vocabulary: null,
    },
    hyperSpace: (h) =>
      dict({
        nComponents: int(1, 100, { default: h.nComponents, label: 'topics' }),
        learningMethod: choice(['batch', 'online'] as const, { default: h.learningMethod, label: 'learning method' }),
        learningDecay: real(0.5, 1, { default: h.learningDecay, label: 'learning decay' }),
        learningOffset: real(1, 100, { default: h.learningOffset, label: 'learning offset' }),
        maxIter: int(1, 500, { default: h.maxIter, label: 'passes' }),
        batchSize: int(1, 10_000, { default: h.batchSize, label: 'mini-batch size' }),
      }),
    algorithm: (data, h, key: Key) => {
      const x = asRows(data.x, 'latentDirichletAllocation')
      const d = x[0]?.length ?? 0
      let start: number[][]
      if (h.componentsInit !== null) start = h.componentsInit.map((r) => [...r])
      else {
        const g = gammaVariates(foldIn(key, 0), new Array<number>(h.nComponents * d).fill(100))
        start = toRows(g, h.nComponents, d).map((r) => r.map((c) => c * 0.01))
      }
      return ldaAlgorithm(x, start, {
        ...(h.docTopicPrior === null ? {} : { docTopicPrior: h.docTopicPrior }),
        ...(h.topicWordPrior === null ? {} : { topicWordPrior: h.topicWordPrior }),
        learningMethod: h.learningMethod,
        learningDecay: h.learningDecay,
        learningOffset: h.learningOffset,
        batchSize: h.batchSize,
        maxDocUpdateIter: h.maxDocUpdateIter,
        meanChangeTol: h.meanChangeTol,
        evaluateEvery: h.evaluateEvery,
        perpTol: h.perpTol,
        docInit: h.docInit,
        evaluate: h.evaluate,
      })
    },
    maxSteps: (h, data) =>
      h.maxIter * (h.learningMethod === 'online' ? Math.ceil(matrixView(data.x, 'lda').n / h.batchSize) : 1),
    contract: (_data, h): Contract<LdaState, LdaPath> =>
      defineContract({
        'lda/topics': derived<LdaState>({
          description: 'the topics (normalised topic-word distributions) and document proportions at this step',
          cost: 'dear',
          compute: (s) => topicRecord(s.components, s.docTopic, h.vocabulary) as unknown as Json,
        }),
      }),
    params: (s, _data, h) => {
      const k = s.components.length
      return {
        components: s.components,
        docTopicPrior: h.docTopicPrior ?? 1 / k,
        topicWordPrior: h.topicWordPrior ?? 1 / k,
        docTopic: normalised(s.docTopic),
        bound: s.bound,
        perplexity: s.perplexity,
        nBatchIterations: s.nBatchIter - 1,
        nIter: s.epoch,
        meanChangeTol: h.meanChangeTol,
        maxDocUpdateIter: h.maxDocUpdateIter,
        totalSamples: h.totalSamples,
      }
    },
    records: (s, _data, h) => [topicRecord(s.components, s.docTopic, h.vocabulary)],
    verbs: (p) => {
      const v = p.components[0]?.length ?? 0
      const e: EStepOptions = {
        alpha: p.docTopicPrior,
        maxDocUpdateIter: p.maxDocUpdateIter,
        meanChangeTol: p.meanChangeTol,
      }
      const expBeta = expOf(dirichletExpectation(p.components))
      const gammaOf = (rows: number[][]): number[][] => eStep(docsOf(rows), expBeta, e, undefined, false).gamma
      const boundOf = (rows: number[][], subSampling: boolean): number =>
        ldaBound(
          rows,
          gammaOf(rows),
          p.components,
          { alpha: p.docTopicPrior, eta: p.topicWordPrior },
          subSampling,
          p.totalSamples,
        )
      return {
        transform: (x, options) => {
          const gamma = gammaOf(asRows(x, 'lda transform', v))
          return toTensor(options?.normalize === false ? gamma : normalised(gamma))
        },
        perplexity: (x, subSampling = false) => {
          const rows = asRows(x, 'lda perplexity', v)
          const words = sum(rows.map(sum)) * (subSampling ? p.totalSamples / rows.length : 1)
          return Math.exp(-boundOf(rows, subSampling) / words)
        },
        score: (data) => boundOf(asRows(data.x, 'lda score', v), false),
        nFeaturesIn: v,
        featureNamesOut: () => p.components.map((_, q) => `topic${q}`),
      }
    },
  })

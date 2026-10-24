// TODO: LDA as a probprog program with the variational engine (sklearn's online and batch VB), labelled LDA, held-out
// perplexity by importance sampling; pLSA, HDP, correlated and dynamic topic models (topic-extensions.ts).
import { WhiteboxError } from '../../lib/base/index.ts'
import { coefficientTable, componentsRecord, type Exposition, mixtureSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  featureNamesOf,
  fitRunOptions,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { type Key, uniform } from '../../lib/random/index.ts'
import { dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'

/** Hyperparameters of {@link latentDirichletAllocation}. */
export interface LdaHyper {
  /** Topics (default 5). */
  readonly nTopics: number
  /** The document-topic prior (default 1 / nTopics, as scikit-learn). */
  readonly alpha?: number
  /** The topic-word prior (default 1 / nTopics, as scikit-learn). */
  readonly eta?: number
  /** Gibbs sweeps (default 200). */
  readonly nIter: number
  /** Words per topic in the top-words table (default 8). */
  readonly topWords: number
  /** Most snapshots kept in the exposition (default 30). */
  readonly maxFrames: number
}

/** Parameters of a fitted topic model. */
export type LdaParams = {
  /** topicWord[k][w]: P(word w | topic k). */
  readonly topicWord: number[][]
  /** docTopic[d][k]: the topic proportions of training document d. */
  readonly docTopic: number[][]
  readonly alpha: number
  readonly eta: number
  /** The mean log likelihood per token of the training corpus. */
  readonly logLikelihood: number
  readonly perplexity: number
  readonly nIter: number
}

/** Verbs of a fitted topic model. */
export interface LdaVerbs {
  /** Topic proportions of documents (rows of word counts), with the topics held fixed. */
  readonly transform: (x: TensorLike) => Tensor
  /** The mean log likelihood per token of documents. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** A fitted topic model. */
export type FittedLda = ExplainedFitted<LdaParams, LdaVerbs>

/** The state of collapsed Gibbs sampling for LDA: topic assignments and the count tables. */
export interface LdaGibbsState {
  readonly z: Int32Array
  readonly docTopic: Int32Array
  readonly topicWord: Int32Array
  readonly topicTotal: Int32Array
  readonly logLikelihood: number
}

/** The tokens of a document-term count matrix: each word repeated by its count. */
function tokensOf(x: TensorLike, who: string): { doc: Int32Array; word: Int32Array; nDocs: number; vocab: number } {
  const m = matrixView(x, who)
  const doc: number[] = []
  const word: number[] = []
  for (let d = 0; d < m.n; d++)
    for (let w = 0; w < m.d; w++) {
      const c = m.data[d * m.d + w] ?? 0
      if (!(Number.isInteger(c) && c >= 0)) throw new WhiteboxError('value', `${who}: counts must be integers >= 0`)
      for (let r = 0; r < c; r++) {
        doc.push(d)
        word.push(w)
      }
    }
  if (doc.length === 0) throw new WhiteboxError('value', `${who}: the corpus has no tokens`)
  return { doc: Int32Array.from(doc), word: Int32Array.from(word), nDocs: m.n, vocab: m.d }
}

/** Topic-word and document-topic estimates from the counts (posterior means given the assignments). */
function estimates(
  s: Pick<LdaGibbsState, 'docTopic' | 'topicWord' | 'topicTotal'>,
  K: number,
  V: number,
  D: number,
  alpha: number,
  eta: number,
): { phi: number[][]; theta: number[][] } {
  const phi = Array.from({ length: K }, (_, k) =>
    Array.from({ length: V }, (_, w) => ((s.topicWord[k * V + w] ?? 0) + eta) / ((s.topicTotal[k] ?? 0) + V * eta)),
  )
  const theta = Array.from({ length: D }, (_, d) => {
    let n = 0
    for (let k = 0; k < K; k++) n += s.docTopic[d * K + k] ?? 0
    return Array.from({ length: K }, (_, k) => ((s.docTopic[d * K + k] ?? 0) + alpha) / (n + K * alpha))
  })
  return { phi, theta }
}

/** The mean log likelihood per token, sum over tokens of log sum_k theta_dk phi_kw. */
function perToken(doc: Int32Array, word: Int32Array, phi: readonly number[][], theta: readonly number[][]): number {
  let s = 0
  for (let i = 0; i < doc.length; i++) {
    const th = theta[doc[i] ?? 0] ?? []
    let p = 0
    for (let k = 0; k < phi.length; k++) p += (th[k] ?? 0) * (phi[k]?.[word[i] ?? 0] ?? 0)
    s += Math.log(p)
  }
  return s / doc.length
}

/**
 * Collapsed Gibbs sampling for latent Dirichlet allocation (Griffiths and Steyvers 2004) as a traced algorithm, one
 * sweep over every token per step: each token's topic is resampled from p(z = k | rest) proportional to
 * (n_dk + alpha)(n_kw + eta) / (n_k + V eta), with its own count removed.
 *
 * @param doc - The document of each token.
 * @param word - The word of each token.
 * @param sizes - Topics K, vocabulary V and documents D.
 * @param priors - alpha and eta.
 * @returns The algorithm.
 */
export function ldaGibbsAlgorithm(
  doc: Int32Array,
  word: Int32Array,
  sizes: { readonly K: number; readonly V: number; readonly D: number },
  priors: { readonly alpha: number; readonly eta: number },
): Algorithm<LdaGibbsState> {
  const { K, V, D } = sizes
  const { alpha, eta } = priors
  const N = doc.length
  const ll = (s: Pick<LdaGibbsState, 'docTopic' | 'topicWord' | 'topicTotal'>): number => {
    const { phi, theta } = estimates(s, K, V, D, alpha, eta)
    return perToken(doc, word, phi, theta)
  }
  return {
    name: 'lda-collapsed-gibbs',
    config: { topics: K, vocabulary: V, documents: D, tokens: N, alpha, eta },
    schema: channels({ metrics: { logLikelihood: { description: 'log likelihood per token', direction: 'higher' } } }),
    init: (key: Key) => {
      const u = uniform(key, [N]).data
      const z = Int32Array.from({ length: N }, (_, i) => Math.min(K - 1, Math.floor((u[i] ?? 0) * K)))
      const docTopic = new Int32Array(D * K)
      const topicWord = new Int32Array(K * V)
      const topicTotal = new Int32Array(K)
      for (let i = 0; i < N; i++) {
        const k = z[i] ?? 0
        docTopic[(doc[i] ?? 0) * K + k] = (docTopic[(doc[i] ?? 0) * K + k] ?? 0) + 1
        topicWord[k * V + (word[i] ?? 0)] = (topicWord[k * V + (word[i] ?? 0)] ?? 0) + 1
        topicTotal[k] = (topicTotal[k] ?? 0) + 1
      }
      const s = { z, docTopic, topicWord, topicTotal }
      return { ...s, logLikelihood: ll(s) }
    },
    step: (s, ctx) => {
      const z = Int32Array.from(s.z)
      const docTopic = Int32Array.from(s.docTopic)
      const topicWord = Int32Array.from(s.topicWord)
      const topicTotal = Int32Array.from(s.topicTotal)
      const u = uniform(ctx.key, [N]).data
      const p = new Float64Array(K)
      for (let i = 0; i < N; i++) {
        const d = doc[i] ?? 0
        const w = word[i] ?? 0
        const old = z[i] ?? 0
        docTopic[d * K + old] = (docTopic[d * K + old] ?? 0) - 1
        topicWord[old * V + w] = (topicWord[old * V + w] ?? 0) - 1
        topicTotal[old] = (topicTotal[old] ?? 0) - 1
        let total = 0
        for (let k = 0; k < K; k++) {
          total +=
            (((docTopic[d * K + k] ?? 0) + alpha) * ((topicWord[k * V + w] ?? 0) + eta)) /
            ((topicTotal[k] ?? 0) + V * eta)
          p[k] = total
        }
        const r = (u[i] ?? 0) * total
        let k = 0
        while (k < K - 1 && (p[k] ?? 0) <= r) k++
        z[i] = k
        docTopic[d * K + k] = (docTopic[d * K + k] ?? 0) + 1
        topicWord[k * V + w] = (topicWord[k * V + w] ?? 0) + 1
        topicTotal[k] = (topicTotal[k] ?? 0) + 1
      }
      const next = { z, docTopic, topicWord, topicTotal }
      return { ...next, logLikelihood: ll(next) }
    },
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood } }),
  }
}

/** Topic proportions of count rows with the topics fixed: EM on the proportions alone (fold-in). */
function foldInDocuments(x: TensorLike, phi: readonly number[][], alpha: number): number[][] {
  const m = matrixView(x, 'latentDirichletAllocation', { nFeatures: phi[0]?.length ?? 0 })
  const K = phi.length
  return Array.from({ length: m.n }, (_, d) => {
    let theta = new Array<number>(K).fill(1 / K)
    for (let it = 0; it < 50; it++) {
      const acc = new Array<number>(K).fill(alpha)
      for (let w = 0; w < m.d; w++) {
        const c = m.data[d * m.d + w] ?? 0
        if (c === 0) continue
        let z = 0
        for (let k = 0; k < K; k++) z += (theta[k] ?? 0) * (phi[k]?.[w] ?? 0)
        for (let k = 0; k < K; k++) acc[k] = (acc[k] ?? 0) + (c * (theta[k] ?? 0) * (phi[k]?.[w] ?? 0)) / z
      }
      const s = acc.reduce((a, v) => a + v, 0)
      theta = acc.map((v) => v / s)
    }
    return theta
  })
}

function ldaVerbs(p: LdaParams): LdaVerbs {
  const V = p.topicWord[0]?.length ?? 0
  return {
    transform: (x) => {
      const t = foldInDocuments(x, p.topicWord, p.alpha)
      return fromStorage(Float64Array.from(t.flat()), [t.length, p.topicWord.length], 'float64')
    },
    score: (data) => {
      const tok = tokensOf(data.x, 'latentDirichletAllocation')
      return perToken(tok.doc, tok.word, p.topicWord, foldInDocuments(data.x, p.topicWord, p.alpha))
    },
    nFeaturesIn: V,
  }
}

/** Evenly thinned indices 0..last, at most max, keeping both ends. */
function thinned(last: number, max: number): number[] {
  if (last + 1 <= max) return Array.from({ length: last + 1 }, (_, t) => t)
  const out = new Set<number>()
  for (let j = 0; j < max; j++) out.add(Math.round((j * last) / (max - 1)))
  return [...out].sort((a, b) => a - b)
}

function fitLda(data: LearnData, h: LdaHyper, options: FitOptions): { params: LdaParams; exposition: Exposition[] } {
  const who = 'latentDirichletAllocation'
  const K = h.nTopics
  if (!(Number.isInteger(K) && K >= 1)) throw new WhiteboxError('value', `${who}: nTopics must be a positive integer`)
  const alpha = h.alpha ?? 1 / K
  const eta = h.eta ?? 1 / K
  const tok = tokensOf(data.x, who)
  const { nDocs: D, vocab: V } = tok
  const snapshots: { shares: number[]; phi: number[][]; ll: number }[] = []
  const base = ldaGibbsAlgorithm(tok.doc, tok.word, { K, V, D }, { alpha, eta })
  const record = (s: LdaGibbsState): void => {
    const { phi } = estimates(s, K, V, D, alpha, eta)
    snapshots.push({ shares: Array.from(s.topicTotal, (c) => c / tok.doc.length), phi, ll: s.logLikelihood })
  }
  const keepAll = h.nIter + 1 <= h.maxFrames
  const wanted = new Set(thinned(h.nIter, h.maxFrames))
  const algorithm: Algorithm<LdaGibbsState> = {
    ...base,
    init: (key) => {
      snapshots.length = 0
      const s = base.init(key)
      record(s)
      return s
    },
    step: (s, ctx) => {
      const next = base.step(s, ctx)
      if (keepAll || wanted.has(ctx.t + 1)) record(next)
      return next
    },
  }
  const out = runAlgorithm(algorithm, { ...fitRunOptions(options, who), maxSteps: h.nIter })
  const s = out.state
  const { phi, theta } = estimates(s, K, V, D, alpha, eta)
  const params: LdaParams = {
    topicWord: phi,
    docTopic: theta,
    alpha,
    eta,
    logLikelihood: s.logLikelihood,
    perplexity: Math.exp(-s.logLikelihood),
    nIter: out.steps,
  }
  if (snapshots.at(-1)?.ll !== s.logLikelihood) record(s)
  const vocabulary = featureNamesOf(data, V)
  const names: string[] = []
  const weights: number[] = []
  for (const [k, row] of phi.entries()) {
    const top = row
      .map((p, w) => ({ p, w }))
      .sort((a, b) => b.p - a.p || a.w - b.w)
      .slice(0, h.topWords)
    for (const t of top) {
      names.push(`topic ${k}: ${vocabulary[t.w] ?? t.w}`)
      weights.push(t.p)
    }
  }
  return {
    params,
    exposition: [
      componentsRecord({ method: 'lda', features: vocabulary, components: phi, scores: theta, space: 'vocabulary' }),
      mixtureSteps({
        family: 'lda',
        weights: snapshots.map((r) => r.shares),
        means: snapshots.map((r) => r.phi),
        objective: snapshots.map((r) => r.ll),
      }),
      coefficientTable({ names, estimates: weights }),
    ],
  }
}

/**
 * Latent Dirichlet allocation (Blei, Ng and Jordan 2003; sklearn.decomposition.LatentDirichletAllocation's model,
 * gensim's) fitted by traced collapsed Gibbs sampling on a document-term count matrix (rows documents, columns
 * words; feature names as the vocabulary). `transform` gives new documents' topic proportions with the topics held
 * fixed. Exposition: a `components` record (each topic's word distribution, each document's proportions), a
 * `mixtureSteps` record per sweep (topic shares, topic-word distributions and the log likelihood per token), and the
 * top words of every topic as a coefficient table.
 *
 * @param hyper - Topics, priors, sweeps, top words and snapshot cap.
 * @returns The estimator.
 */
export function latentDirichletAllocation(hyper: Partial<LdaHyper> = {}): ExplainedEstimator<LdaHyper, FittedLda> {
  return defineExplainedEstimator<LdaHyper, LdaParams, LdaVerbs>(
    {
      name: 'latent-dirichlet-allocation',
      kind: 'transformer',
      hyperSpace: (h: LdaHyper): Space =>
        dict({
          nTopics: int(1, 30, { default: h.nTopics, label: 'topics' }),
          nIter: int(1, 1000, { default: h.nIter, label: 'Gibbs sweeps' }),
          alpha: real(0.01, 10, { log: true, default: h.alpha ?? 1 / h.nTopics, label: 'document-topic prior' }),
          eta: real(0.001, 10, { log: true, default: h.eta ?? 1 / h.nTopics, label: 'topic-word prior' }),
        }),
      fit: fitLda,
      methods: ldaVerbs,
    },
    {
      nTopics: hyper.nTopics ?? 5,
      ...(hyper.alpha === undefined ? {} : { alpha: hyper.alpha }),
      ...(hyper.eta === undefined ? {} : { eta: hyper.eta }),
      nIter: hyper.nIter ?? 200,
      topWords: hyper.topWords ?? 8,
      maxFrames: hyper.maxFrames ?? 30,
    },
  )
}

/** {@link latentDirichletAllocation} as a class. */
export class LatentDirichletAllocation extends ExplainedEstimatorClass<LdaHyper, FittedLda> {
  /** @param hyper - As {@link latentDirichletAllocation}. */
  constructor(hyper: Partial<LdaHyper> = {}) {
    super(latentDirichletAllocation(hyper))
  }
}

/**
 * The document-term count matrix of a tokenised corpus (token ids per document) as learn data, with the vocabulary
 * as feature names.
 *
 * @param corpus - Token ids per document and the vocabulary.
 * @returns `{ x, featureNames }`.
 * @throws WhiteboxError `value` when the corpus has no tokens.
 */
export function documentTermCounts(corpus: {
  readonly tokens?: readonly (readonly number[])[]
  readonly vocabulary?: readonly string[]
}): LearnData & { readonly featureNames: string[] } {
  const tokens = corpus.tokens ?? []
  let V = corpus.vocabulary?.length ?? 0
  for (const doc of tokens) for (const w of doc) V = Math.max(V, w + 1)
  if (tokens.length === 0 || V === 0) throw new WhiteboxError('value', 'documentTermCounts: the corpus has no tokens')
  const counts = new Float64Array(tokens.length * V)
  for (const [d, doc] of tokens.entries()) for (const w of doc) counts[d * V + w] = (counts[d * V + w] ?? 0) + 1
  const names = Array.from({ length: V }, (_, w) => corpus.vocabulary?.[w] ?? `w${w}`)
  return { x: fromStorage(counts, [tokens.length, V], 'float64'), featureNames: names }
}
